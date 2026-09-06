import Foundation
import Synchronization
import Testing

@testable import Swiftuna

private struct HillClimbSampler: CustomSampler {
    let step: Double
    // Fixed pseudo-random stream: the convergence test stays reproducible
    // without touching product code (LCG, values in [-1, 1)).
    static let draws: [Double] = (0..<400).map { i in
        var x = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
        x ^= x >> 29
        return Double(x >> 32) / Double(UInt64(1) << 32) * 2.0 - 1.0
    }

    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        let draw = Self.draws[(trialNumber * 2) % Self.draws.count]
        let jitter = Self.draws[(trialNumber * 2 + 1) % Self.draws.count]
        guard let bx = history.best?.params["x"]?.asDouble else {
            return ["x": .double(draw * 10.0)]
        }
        return ["x": .double(min(10.0, max(-10.0, bx + jitter * step)))]
    }
}

private struct Boom: Error {}

private struct ThrowingSampler: CustomSampler {
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        throw Boom()
    }
}

private struct TaggedSampler: CustomSampler {
    let tag: String
    let range: ClosedRange<Double>
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        [
            "x": .double(Double.random(in: range)),
            "who": .string(tag),
        ]
    }
}

private struct FixedSampler: CustomSampler {
    let params: [String: ParameterValue]
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        params
    }
}

private struct TwoDSampler: CustomSampler {
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        [
            "x": .double(Double.random(in: -5.0...5.0)),
            "y": .double(Double.random(in: -5.0...5.0)),
        ]
    }
}

@Suite("Custom Sampler Tests", .serialized)
struct CustomSamplerTests {
    @Test("Hill-climbing protocol sampler converges through optimize(using:)")
    func testCustomSamplerConverges() throws {
        let study = try Swiftuna.createStudy(name: "custom_hill_\(UUID().uuidString)")
        try study.optimize(nTrials: 50, using: HillClimbSampler(step: 1.0)) { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            return (x - 2.0) * (x - 2.0)
        }
        let trials = try study.trials
        #expect(trials.count == 50)
        let best = try #require(
            trials.filter { $0.state == .complete }.min {
                ($0.values.first ?? .infinity) < ($1.values.first ?? .infinity)
            })
        #expect((best.values.first ?? .infinity) < 1.0)
    }

    @Test("Closure sampler converges and sees incremental history")
    func testClosureSampler() throws {
        let study = try Swiftuna.createStudy(name: "custom_closure_\(UUID().uuidString)")
        let seenNewCounts = Mutex<[Int]>([])
        try study.optimize(
            nTrials: 20,
            using: { (history: StudyHistory, _: Int) throws -> [String: ParameterValue] in
                seenNewCounts.withLock { $0.append(history.new.count) }
                let bx = history.best?.params["x"]?.asDouble ?? Double.random(in: -10.0...10.0)
                return ["x": .double(min(10.0, max(-10.0, bx + Double.random(in: -1.0...1.0))))]
            }
        ) { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            return (x - 2.0) * (x - 2.0)
        }
        #expect(try study.trials.count == 20)
        // First call sees nothing new; every later call sees exactly one.
        let counts = seenNewCounts.withLock { $0 }
        #expect(counts.first == 0)
        #expect(counts.dropFirst().allSatisfy { $0 == 1 })
    }

    @Test("Sampler throw aborts loudly with zero trials consumed")
    func testSamplerThrowAborts() throws {
        let study = try Swiftuna.createStudy(name: "custom_throw_\(UUID().uuidString)")
        do {
            try study.optimize(nTrials: 3, using: ThrowingSampler()) { _ in 0.0 }
            Issue.record("expected Boom")
        } catch is Boom {
            // Original error preserved, not wrapped.
        }
        #expect(try study.trials.count == 0)
    }

    @Test("Pruned trials record with params in custom history")
    func testCustomPrunePath() throws {
        let study = try Swiftuna.createStudy(name: "custom_prune_\(UUID().uuidString)")
        // Annotated: a Never-ending closure fits single and vector alike.
        try study.optimize(nTrials: 3, using: FixedSampler(params: ["x": .double(1.0)])) {
            (trial: inout Trial) throws(SwiftunaError) -> Double in
            _ = try trial.suggest("x", in: -10.0...10.0)
            try trial.prune()
        }
        let trials = try study.trials
        #expect(trials.count == 3)
        #expect(trials.allSatisfy { $0.state == .pruned })
        #expect(trials.allSatisfy { $0.params["x"]?.asDouble == 1.0 })
    }

    @Test("Two drivers sharing a study never receive each other's configs")
    func testParallelDriversAtomic() async throws {
        let study = try Swiftuna.createStudy(name: "custom_parallel_\(UUID().uuidString)")
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try study.optimize(nTrials: 25, using: TaggedSampler(tag: "A", range: 0.0...1.0)) { trial in
                    let x = try trial.suggest("x", in: -10.0...10.0)
                    _ = try trial.suggest("who", choices: ["A", "B"])
                    return x * x
                }
            }
            group.addTask {
                try study.optimize(nTrials: 25, using: TaggedSampler(tag: "B", range: 9.0...10.0)) { trial in
                    let x = try trial.suggest("x", in: -10.0...10.0)
                    _ = try trial.suggest("who", choices: ["A", "B"])
                    return x * x
                }
            }
            try await group.waitForAll()
        }
        let trials = try study.trials
        #expect(trials.count == 50)
        for t in trials {
            let who = t.params["who"]?.asString
            let x = t.params["x"]?.asDouble
            switch who {
            case "A": #expect(x.map { (0.0...1.0).contains($0) } ?? false)
            case "B": #expect(x.map { (9.0...10.0).contains($0) } ?? false)
            default: Issue.record("untagged trial #\(t.number)")
            }
        }
    }

    @Test("Multi-objective custom driver records vectors")    func testCustomMultiObjective() throws {
        let study = try Swiftuna.createStudy(
            name: "custom_mo_\(UUID().uuidString)",
            directions: [.minimize, .minimize])
        try study.optimize(nTrials: 10, using: TwoDSampler()) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return [x * x, y * y]
        }
        let trials = try study.trials
        #expect(trials.count == 10)
        #expect(trials.allSatisfy { $0.values.count == 2 })
    }

    @Test("Reentrant checkout fails loudly instead of deadlocking", .timeLimit(.minutes(1)))
    func testReentrantAskFails() throws {
        // One shared pair: the closure reads whichever study the current
        // phase installed. Both phases must observe reentrantAsk.
        let box = Mutex<Study?>(nil)
        let inner = Mutex<SwiftunaError?>(nil)
        let sampler = CallbackSampler(onFloat: { _, _, _, _, _, _ in
            do {
                if let study = box.withLock({ $0 }) {
                    _ = try study.askEnqueued(["z": .double(1.0)])
                }
            } catch let error as SwiftunaError {
                inner.withLock { $0 = error }
            } catch {}
            return 1.0
        })
        func expectReentrantAsk(on study: Study) throws {
            // The upcall below runs on this thread: the nested checkout must
            // throw, not hang (the time limit guards that).
            var trial = try study.ask()
            #expect(try trial.suggest("x", in: -10.0...10.0) == 1.0)
            try study.tell(consuming: trial, value: 1.0)
            if case .reentrantAsk = inner.withLock({ $0 }) {} else {
                Issue.record("expected reentrantAsk, got \(String(describing: inner.withLock { $0 }))")
            }
        }

        // Phase 1: original study.
        let study = try Swiftuna.createStudy(
            name: "reentrant_\(UUID().uuidString)", sampler: sampler)
        box.withLock { $0 = study }
        try expectReentrantAsk(on: study)

        // Phase 2: the Rust-side sampler (and its live callback context)
        // travels with copy(to:), so the guard must travel too.
        inner.withLock { $0 = nil }
        let study2 = try Swiftuna.createStudy(
            name: "reentrant_src_\(UUID().uuidString)", sampler: sampler)
        let copy = try study2.copy(to: .inMemory, as: "reentrant_copy_\(UUID().uuidString)")
        box.withLock { $0 = copy }
        try expectReentrantAsk(on: copy)
    }

    @Test("Callback sampler table: kinds × valid/nil × trial identity")
    func testCallbackSamplerTable() throws {
        // Phase 1: Valid values are suggested and trialNumber is forwarded identically.
        let seenFloats = Mutex<[Int]>([])
        let seenInts = Mutex<[Int]>([])
        let seenCats = Mutex<[Int]>([])
        let validSampler = CallbackSampler(
            onFloat: { _, low, high, _, _, trialNumber in
                seenFloats.withLock { $0.append(trialNumber) }
                return (low + high) / 2
            },
            onInt: { _, low, high, _, _, trialNumber in
                seenInts.withLock { $0.append(trialNumber) }
                return (low + high) / 2
            },
            onCategorical: { _, choices, trialNumber in
                seenCats.withLock { $0.append(trialNumber) }
                return choices.count - 1
            }
        )
        let study = try Swiftuna.createStudy(name: "cb_table_\(UUID().uuidString)", sampler: validSampler)
        try study.optimize(nTrials: 3) { trial in
            let f = try trial.suggest("f", in: 0.0...10.0)
            let i = try trial.suggest("i", in: 10...30)
            let c = try trial.suggest("c", choices: ["low", "high"])
            #expect(f == 5.0)
            #expect(i == 20)
            #expect(c == "high")
            return f + Double(i)
        }
        #expect(seenFloats.withLock { $0 } == [0, 1, 2])
        #expect(seenInts.withLock { $0 } == [0, 1, 2])
        #expect(seenCats.withLock { $0 } == [0, 1, 2])

        // Phase 2: Refusing (nil) fails immediately across all three kinds without hanging.
        let refusingSampler = CallbackSampler(
            onFloat: { _, _, _, _, _, _ in nil },
            onInt: { _, _, _, _, _, _ in nil },
            onCategorical: { _, _, _ in nil }
        )
        let failStudy = try Swiftuna.createStudy(name: "cb_fail_\(UUID().uuidString)", sampler: refusingSampler)
        var t1 = try failStudy.ask()
        #expect(throws: SwiftunaError.self) { try t1.suggest("f", in: 0.0...10.0) }
        var t2 = try failStudy.ask()
        #expect(throws: SwiftunaError.self) { try t2.suggest("i", in: 1...10) }
        var t3 = try failStudy.ask()
        #expect(throws: SwiftunaError.self) { try t3.suggest("c", choices: ["a", "b"]) }

        // Phase 3: Partial fallback - omitted closures fall back to uniform sampling.
        let partialSampler = CallbackSampler(onFloat: { _, low, high, _, _, _ in (low + high) / 2 })
        let fallbackStudy = try Swiftuna.createStudy(name: "cb_fb_\(UUID().uuidString)", sampler: partialSampler)
        try fallbackStudy.optimize(nTrials: 3) { trial in
            let f = try trial.suggest("f", in: 0.0...10.0)
            #expect(f == 5.0)
            let i = try trial.suggest("i", in: 1...10)
            #expect((1...10).contains(i))
            return f
        }
    }

    @Test("Callback categorical closure rejects NUL-containing labels loudly")
    func testCallbackNulLabelFails() throws {
        let sampler = CallbackSampler(onCategorical: { _, choices, _ in 0 })
        let study = try Swiftuna.createStudy(name: "cb_nul_\(UUID().uuidString)", sampler: sampler)
        var trial = try study.ask()
        // Label containing an embedded NUL cannot cross the C ABI and must fail
        // with samplerError instead of silently truncating.
        #expect(throws: SwiftunaError.self) {
            try trial.suggest("opt", choices: ["clean", "embed\0nul"])
        }
    }

    @Test("Partial fixing still records exact history, including Rust-sampled rest")
    func testPartialFixHistoryExact() throws {
        struct FixXOnly: CustomSampler {
            func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
                ["x": .double(2.0)]
            }
        }
        let study = try Swiftuna.createStudy(name: "custom_partial_\(UUID().uuidString)")
        try study.optimize(nTrials: 3, using: FixXOnly()) { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            let y = try trial.suggest("y", in: 0.0...1.0) // omitted: Rust-sampled
            return x + y
        }
        let trials = try study.trials
        #expect(trials.count == 3)
        for t in trials {
            // Fixed param kept; the leftover the sampler never saw is recorded too.
            #expect(t.params["x"]?.asDouble == 2.0)
            #expect(t.params["y"]?.asDouble != nil)
        }
    }

    @Test("Running-best fold matches the scanning init, ties included")
    func testFoldMatchesScanOnTies() throws {
        func history(values: [Double], state: TrialState = .complete) -> [PersistedTrial] {
            values.enumerated().map { i, v in
                PersistedTrial(number: i, state: state, value: v, values: [v], params: [:])
            }
        }
        // Minimize with a tie for best; maximize with a tie; multi; empty; non-complete.
        let cases: [([Double], [Direction])] = [
            ([3.0, 1.0, 1.0, 2.0], [.minimize]),
            ([1.0, 3.0, 3.0, 2.0], [.maximize]),
            ([1.0, 2.0], [.minimize, .minimize]),
            ([], [.minimize]),
        ]
        for (values, directions) in cases {
            let all = history(values: values)
            let scanned = StudyHistory(all: all, newSince: 0, directions: directions).best
            let folded = all.reduce(nil as PersistedTrial?) {
                StudyHistory.fold($1, into: $0, directions: directions)
            }
            #expect(scanned?.number == folded?.number)
            #expect(scanned?.values.first == folded?.values.first)
        }
        // Pruned/failed trials can never take the lead in either path.
        let mixed = [
            PersistedTrial(number: 0, state: .fail, value: -100.0, values: [-100.0], params: [:]),
            PersistedTrial(number: 1, state: .pruned, value: -200.0, values: [-200.0], params: [:]),
            PersistedTrial(number: 2, state: .complete, value: 5.0, values: [5.0], params: [:]),
        ]
        let scanned = StudyHistory(all: mixed, newSince: 0, directions: [.minimize]).best
        let folded = mixed.reduce(nil as PersistedTrial?) {
            StudyHistory.fold($1, into: $0, directions: [.minimize])
        }
        #expect(scanned?.number == 2)
        #expect(folded?.number == 2)
    }

    @Test("Custom driver stops cleanly when the grid is exhausted")
    func testGridExhaustionStopsCustomDriver() throws {
        struct FixNothing: CustomSampler {
            func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
                [:]
            }
        }
        let grid = GridSampler(searchSpace: ["x": [1.0, 2.0]], seed: 42)
        let study = try Swiftuna.createStudy(name: "custom_grid_\(UUID().uuidString)", sampler: grid)
        // Must return, not throw: the driver breaks on searchSpaceExhausted.
        try study.optimize(nTrials: 5, using: FixNothing()) { trial in
            let x = try trial.suggest("x", in: 0.0...10.0)
            return x
        }
        let trials = try study.trials
        // Both grid points were evaluated. (The engine records a fail side
        // effect on the draining checkout, so count past 2 is engine detail
        // this test deliberately does not pin.)
        let completedXs = Set(trials.filter { $0.state == .complete }.compactMap { $0.params["x"]?.asDouble })
        #expect(completedXs == [1.0, 2.0])
    }
}
