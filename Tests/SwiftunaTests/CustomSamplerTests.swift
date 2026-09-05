import Foundation
import Synchronization
import Testing

@testable import Swiftuna

private struct HillClimbSampler: CustomSampler {
    let step: Double
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        guard let bx = history.best?.params["x"]?.asDouble else {
            return ["x": .double(Double.random(in: -10.0...10.0))]
        }
        return ["x": .double(min(10.0, max(-10.0, bx + Double.random(in: -step...step))))]
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
        let study = try Swiftuna.createStudy(name: "reentrant_\(UUID().uuidString)", sampler: sampler)
        box.withLock { $0 = study }
        // The upcall below runs on this thread under a held ask slot: the
        // nested checkout must throw, not hang (the time limit guards that).
        var trial = try study.ask()
        #expect(try trial.suggest("x", in: -10.0...10.0) == 1.0)
        try study.tell(consuming: trial, value: 1.0)
        if case .reentrantAsk = inner.withLock({ $0 }) {} else {
            Issue.record("expected reentrantAsk, got \(String(describing: inner.withLock { $0 }))")
        }
    }
}
