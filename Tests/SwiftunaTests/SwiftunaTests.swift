import Foundation
import PropertyBased
import Synchronization
import Testing

@testable import Swiftuna

@Suite("Typed Throws and Static Sampler Tests")
struct TypedErrorAndSamplerTests {

    @Test("Catching specific typed error cases without type casting")
    func testTypedErrorCatching() {
        do {
            let study = try Swiftuna.createStudy(sampler: TPESampler(seed: 1))
            var trial = try study.ask()
            let emptyList: [String] = []
            _ = try trial.suggest("choices", choices: emptyList)
            Issue.record("Should have thrown emptyChoices error")
        } catch {
            // Because of throws(SwiftunaError), 'error' is statically known to be SwiftunaError!
            #expect(error == SwiftunaError.emptyChoices("Choices cannot be empty for parameter 'choices'"))
        }
    }

    @Test("Invalid range throws typed invalidRange error")
    func testInvalidRangeTypedError() {
        do {
            let study = try Swiftuna.createStudy(sampler: TPESampler(seed: 2))
            var trial = try study.ask()
            let infRange = Double.infinity...Double.infinity
            _ = try trial.suggest("inf", in: infRange)
            Issue.record("Should have thrown invalidRange error")
        } catch {
            switch error {
            case .invalidRange:
                #expect(true)
            default:
                Issue.record("Expected invalidRange but got: \(error)")
            }
        }
    }

    @Test("Static generic sampler specialization with RandomSampler")
    func testRandomSamplerSpecialization() throws(SwiftunaError) {
        let sampler = RandomSampler(seed: 12345)
        let study = try Swiftuna.createStudy(name: "random_test", sampler: sampler)

        var trial = try study.ask()
        let val = try trial.suggest("p", in: 0.0...1.0)
        try study.tell(consuming: trial, value: val)

        let best = try study.bestValue
        #expect(best == val)
    }

    @Test("Default sampler generic parameter uses TPESampler")
    func testDefaultSamplerSpecialization() throws(SwiftunaError) {
        let study = try Swiftuna.createStudy(name: "default_test")
        var trial = try study.ask()
        let val = try trial.suggest("p", in: -5.0...5.0)
        try study.tell(consuming: trial, value: val * val)

        let best = try study.bestValue
        #expect(best >= 0.0)
    }
}

// Test Attribute Keys
enum ModelAccuracy: AttributeKeyProtocol {
    typealias Value = Double
    static let name = "model_accuracy"
}

enum WeightFingerprint: AttributeKeyProtocol {
    typealias Value = String
    static let name = "weight_fingerprint"
}

extension AttributeKey where Value == Double {
    static let accuracy = AttributeKey<Double>("model_accuracy")
}
extension AttributeKey where Value == String {
    static let fingerprint = AttributeKey<String>("weight_fingerprint")
}

@Suite("Optuna Calibration & Type-Safe Attributes Tests")
struct OptunaCalibrationTests {

    @Test("Trial pruning handles SMT constraint violation without aborting study")
    func testPruningInOptimizationLoop() throws {
        let study = try Swiftuna.createStudy(name: "prune_test", sampler: TPESampler(seed: 42))

        try study.optimize(nTrials: 10) { (trial: inout Trial) throws(SwiftunaError) -> Double in
            let w = try trial.suggest("weight", in: 0.0...1.0)
            if w > 0.5 {
                throw SwiftunaError.trialPruned(reason: "SMT statement ceiling exceeded: \(w) > 0.5")
            }
            return w * w
        }

        let allTrials = try study.trials
        #expect(allTrials.count == 10)

        let prunedTrials = allTrials.filter { $0.state == .pruned }
        let completedTrials = allTrials.filter { $0.state == .complete }

        #expect(!prunedTrials.isEmpty)
        #expect(!completedTrials.isEmpty)
        #expect(prunedTrials.count + completedTrials.count == 10)

        for pt in prunedTrials {
            #expect(pt.value == nil)
        }
        for ct in completedTrials {
            #expect(ct.value != nil)
            #expect(ct.value! <= 0.25)
        }
    }

    @Test("Type-safe and string user attributes across active Trial, PersistedTrial, and Study")
    func testTypeSafeTrialAndStudyUserAttributes() throws {
        let study = try Swiftuna.createStudy(name: "attrs_test", sampler: TPESampler(seed: 101))

        // Study-level user attributes
        try study.setUserAttr(WeightFingerprint.self, value: "global_corpus_hash_xyz")
        try study.setUserAttr("dataset_version", value: "v2.1")
        #expect(try study.userAttr(WeightFingerprint.self) == "global_corpus_hash_xyz")
        #expect(try study.userAttr("dataset_version") == "v2.1")

        // Trial-level user attributes
        var trial = try study.ask()
        trial[ModelAccuracy.self] = 0.985
        trial[WeightFingerprint.self] = "sha256:test_weights_123"
        trial["ad_hoc_note"] = "baseline calibration"

        let p = try trial.suggest("p", in: 1.0...2.0)
        try study.tell(consuming: trial, value: p)

        let bestTrial = try study.bestTrial
        let persisted = try #require(bestTrial)
        let acc: Double? = persisted[ModelAccuracy.self]
        let fp: String? = persisted[WeightFingerprint.self]
        let note: String? = persisted["ad_hoc_note"]

        #expect(acc == 0.985)
        #expect(fp == "sha256:test_weights_123")
        #expect(note == "baseline calibration")
    }

    @Test("Parameter stability interval calculation on good trials")
    func testParameterStabilityIntervals() throws {
        let study = try Swiftuna.createStudy(name: "stability_test", sampler: TPESampler(seed: 42))

        try study.optimize(nTrials: 20) { (trial: inout Trial) throws(SwiftunaError) -> Double in
            let x = try trial.suggest("param_w", in: -2.0...2.0)
            return x * x
        }

        let intervals = try study.parameterIntervals(tolerance: 0.5)
        let interval = try #require(intervals["param_w"])

        #expect(interval.lowerBound <= interval.upperBound)
        #expect(interval.lowerBound >= -2.0)
        #expect(interval.upperBound <= 2.0)
    }
}

@Suite("Trial Enqueue, Importance & Functional Toolkit Tests")
struct EnqueueAndImportanceTests {

    @Test("Typed enqueue fixes parameters without JSON round trip")
    func testTypedEnqueue() throws {
        let study = try Swiftuna.createStudy(name: "typed_enqueue_test", sampler: TPESampler(seed: 42))

        try study.enqueue(
            [
                "weight": .double(1.5),
                "min_comp": .int(8),
                "mode": .string("fast"),
                "flag": .bool(true),
            ], userAttrs: ["tag": "typed"])

        // Ask for trial 0
        var trial0 = try study.ask()
        #expect(trial0.number == 0)

        let w = try trial0.suggest("weight", in: 0.0...10.0)
        let c = try trial0.suggest("min_comp", in: 1...20)
        let m = try trial0.suggest("mode", choices: ["fast", "precise"])
        let f = try trial0.suggest("flag", choices: [true, false])

        #expect(w == 1.5)
        #expect(c == 8)
        #expect(m == "fast")
        #expect(f == true)

        try study.tell(consuming: trial0, value: w * Double(c))

        // Subsequent trials are sampled normally
        var trial1 = try study.ask()
        #expect(trial1.number == 1)
        _ = try trial1.suggest("weight", in: 0.0...10.0)
        try study.tell(consuming: trial1, value: 5.0)

        let trials = try study.trials
        #expect(trials.count == 2)
        #expect(trials[0].params["weight"] == 1.5)
        #expect(trials[0].params["min_comp"] == 8.0)
        #expect(trials[0].params["mode"] == "fast")
        #expect(trials[0].params["flag"]?.asBool == true)
    }

    @Test("Natural enqueue syntax resolves without verbosity or ambiguity")
    func testEnqueueNaturalSyntax() throws {
        let study = try Swiftuna.createStudy(name: "natural_enqueue_test", sampler: TPESampler(seed: 42))

        // Heterogeneous literals: must pick an overload without ambiguity.
        try study.enqueue(["lr": 0.01, "hidden_dim": 32])
        // Variables (not literals): generic overload auto-upgrades to typed.
        // UInt64 has no exact conversion: exercises the JSON fallback.
        let w = 1.5
        let n = 8
        let m = "fast"
        let big = UInt64(42)
        try study.enqueue(["weight": w, "min_comp": n, "mode": m, "big": big])

        var t0 = try study.ask()
        #expect(try t0.suggest("lr", in: 0.0...1.0) == 0.01)
        #expect(try t0.suggest("hidden_dim", in: 1...64) == 32)
        try study.tell(consuming: t0, value: 1.0)

        var t1 = try study.ask()
        #expect(try t1.suggest("weight", in: 0.0...10.0) == 1.5)
        #expect(try t1.suggest("min_comp", in: 1...20) == 8)
        #expect(try t1.suggest("mode", choices: ["fast", "precise"]) == "fast")
        #expect(try t1.suggest("big", in: 1...100) == 42)
        try study.tell(consuming: t1, value: 1.0)
    }

    @Test("TPE sampler configuration matrix: multivariate × startup (incl. 0) × seeds")
    func testTPESamplerConfig() throws {
        for multivariate in [nil, false, true] as [Bool?] {
            for startup in [0, 2, 5] {
                let study = try Swiftuna.createStudy(
                    name: "tpe_matrix_\(String(describing: multivariate))_\(startup)",
                    sampler: TPESampler(seed: 42, multivariate: multivariate, nStartupTrials: startup))
                try study.optimize(nTrials: 6) { trial in
                    let x = try trial.suggest("x", in: -5.0...5.0)
                    return x * x
                }
                #expect(try study.trials.count == 6)
            }
        }
        // Seeded determinism through the full-config constructor across independent studies.
        func run(seed: UInt64) throws -> [Double] {
            let study = try Swiftuna.createStudy(
                name: "tpe_det_\(seed)_\(UUID().uuidString)",
                sampler: TPESampler(seed: seed, multivariate: true, nStartupTrials: 2))
            try study.optimize(nTrials: 6) { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return x * x
            }
            return try study.trials.map { $0.values.first ?? .nan }
        }
        #expect(try run(seed: 7) == run(seed: 7))
        #expect(try run(seed: 7) != run(seed: 8))
    }

    @Test("paramImportances returns expected Result.failure on empty study")
    func testParamImportancesEmptyStudy() throws {
        let study = try Swiftuna.createStudy(name: "empty_importance_test")

        let result = study.paramImportances()
        switch result {
        case .success:
            Issue.record("Expected failure when evaluating importances on empty study")
        case .failure(let err):
            #expect(
                err == .noCompletedTrial || err == .invalidArgument("Failed to evaluate parameter importances")
                    || "\(err)".contains("No completed trial"))
        }
    }

    @Test("Functional pipeline chaining on sequence of trials")
    func testFunctionalPipelineChaining() throws {
        let study = try Swiftuna.createStudy(name: "functional_test", sampler: TPESampler(seed: 42))

        try study.optimize(nTrials: 15) { (trial: inout Trial) throws(SwiftunaError) -> Double in
            let w = try trial.suggest("weight", in: 0.0...10.0)
            if w > 8.0 {
                throw SwiftunaError.trialPruned(reason: "weight too large")
            }
            return (w - 3.0) * (w - 3.0)
        }

        let allTrials = try study.trials

        // Functional pipeline dot-chaining
        let top3 = allTrials.completed().sortedByValue(ascending: true).top(3)
        #expect(top3.count == 3)
        #expect(top3[0].value! <= top3[1].value!)
        #expect(top3[1].value! <= top3[2].value!)

        let prunedCount = allTrials.pruned().count
        #expect(prunedCount > 0)

        let weights = top3.values(for: "weight")
        #expect(weights.count == 3)

        let intervals = top3.parameterIntervals()
        let weightInterval = try #require(intervals["weight"])
        #expect(weightInterval.lowerBound <= weightInterval.upperBound)
    }

    @Test("Filtered trials support incremental refresh with since")
    func testTrialsSinceIncrementalRefresh() throws {
        let study = try Swiftuna.createStudy(name: "since_test", sampler: RandomSampler(seed: 7))
        try study.optimize(nTrials: 5) { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            return x * x
        }
        let all = try study.trials(where: Set(TrialState.allCases))
        #expect(all.count == 5)
        // Default since: 0 returns everything, matching the unfiltered count.
        let tail = try study.trials(where: Set(TrialState.allCases), since: 3)
        #expect(tail.count == 2)
        #expect(tail.map(\.number) == [3, 4])
        // Past the end: empty, still success.
        #expect(try study.trials(where: Set(TrialState.allCases), since: 99).isEmpty)
        // State filter composes with the tail window.
        #expect(try study.trials(where: [.fail], since: 0).isEmpty)
    }

    @Test("Unnormalized and subset parameter importances evaluation")
    func testUnnormalizedAndSubsetParamImportances() throws {
        let study = try Swiftuna.createStudy(name: "unnorm_test", sampler: TPESampler(seed: 42))

        try study.optimize(nTrials: 25) { (trial: inout Trial) throws(SwiftunaError) -> Double in
            let a = try trial.suggest("param_a", in: -5.0...5.0)
            let b = try trial.suggest("param_b", in: -5.0...5.0)
            return a * a * 10.0 + b * b
        }

        // Test 1: Unnormalized importances
        let unnormResult = study.paramImportances(normalize: false)
        let unnorm = try unnormResult.get()
        #expect(unnorm["param_a"] != nil)
        #expect(unnorm["param_b"] != nil)
        #expect(unnorm["param_a"]! >= 0.0)
        #expect(unnorm["param_b"]! >= 0.0)

        // Test 2: Targeted subset parameter evaluation
        let subsetResult = study.paramImportances(normalize: true, params: ["param_a"])
        let subset = try subsetResult.get()
        #expect(subset.count == 1)
        #expect(subset["param_a"] != nil)
        #expect(subset["param_b"] == nil)
    }

    private enum ActivationTestEnum: String, CaseIterable, Sendable {
        case relu, gelu, silu
    }

    @Test("Float ML interop, flexible constraint types, and type-inferred enum extraction")
    func testFloatAndErgonomicUtilities() throws {
        let study = try Swiftuna.createStudy(name: "float_utility_test")

        try study.optimize(nTrials: 5) { trial in
            // 1. Suggest as Float (Float32)
            let lr: Float = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
            #expect(lr >= 1e-4 && lr <= 1e-1)

            let hiddenDim = try trial.suggest("hidden_dim", choices: [32, 64])
            _ = try trial.suggest("activation", choices: ActivationTestEnum.allCases)

            // 2. Set constraints using Int and Float variables directly
            let totalParams: Int = hiddenDim * 100
            try trial.setConstraint("param_bound", value: totalParams - 5000)

            let latency: Float = 12.5
            try trial.setConstraint("latency_bound", value: latency - 20.0)

            // 3. Report Float intermediate value
            let intermediateLoss: Float = 0.42
            try trial.report(intermediateLoss, step: 1)

            return Double(lr) + Double(hiddenDim)
        }

        let best = try #require(try study.bestTrial)

        // 4. Read back Float scalar
        let bestLR: Float? = best.float("lr")
        #expect(bestLR != nil)
        #expect(bestLR! >= 1e-4 && bestLR! <= 1e-1)

        // 5. Type-inferred enum parameter extraction
        let actEnum: ActivationTestEnum? = best.param("activation")
        #expect(actEnum != nil)
        #expect(ActivationTestEnum.allCases.contains(actEnum!))

        // 6. Verify constraints were recorded properly
        #expect(best.constraints["param_bound"] != nil)
        #expect(best.constraints["latency_bound"] != nil)
        #expect(best.constraints["latency_bound"]! < 0.0)
    }

    @Test("Declarative sequence dot-chaining: best(), failed(), and type-safe filtering")
    func testDeclarativeSequenceOperations() throws {
        let archKey = AttributeKey<String>("arch")
        enum OptEnum: String, CaseIterable, Sendable {
            case adam, sgd, adamw
        }

        let t1 = PersistedTrial(
            number: 0,
            state: .complete,
            value: 0.5,
            params: ["optimizer": .string("adam")],
            userAttrs: ["arch": "Transformer"]
        )
        let t2 = PersistedTrial(
            number: 1,
            state: .complete,
            value: 0.2,
            params: ["optimizer": .string("sgd")],
            userAttrs: ["arch": "Transformer"]
        )
        let t3 = PersistedTrial(
            number: 2,
            state: .complete,
            value: 0.8,
            params: ["optimizer": .string("adam")],
            userAttrs: ["arch": "CNN"]
        )
        let t4 = PersistedTrial(
            number: 3,
            state: .pruned,
            value: 2.0,
            params: [:]
        )
        let t5 = PersistedTrial(
            number: 4,
            state: .fail,
            value: nil,
            params: [:]
        )

        let trials = [t1, t2, t3, t4, t5]

        // State filters
        #expect(trials.completed().count == 3)
        #expect(trials.pruned().count == 1)
        #expect(trials.failed().count == 1)
        #expect(trials.failed().first?.number == 4)

        // Best on filtered sequence
        let bestTransformer = trials.completed().filter(where: archKey, equals: "Transformer").best()
        #expect(bestTransformer?.number == 1)
        #expect(bestTransformer?.value == 0.2)

        let bestAdam = trials.completed().filter(param: "optimizer", equals: OptEnum.adam).best()
        #expect(bestAdam?.number == 0)
        #expect(bestAdam?.value == 0.5)

        // Best maximize
        let maxTrial = trials.completed().best(direction: .maximize)
        #expect(maxTrial?.number == 2)
        #expect(maxTrial?.value == 0.8)
    }

    @Test("sortedByValue edge cases: sort stability, identical values, and nil filtering")
    func testSortedByValueEdgeCases() throws {
        // 1. Empty sequence
        let emptyTrials: [PersistedTrial] = []
        #expect(emptyTrials.sortedByValue(ascending: true).isEmpty)
        #expect(emptyTrials.sortedByValue(ascending: false).isEmpty)

        // 2. Single trial with nil value vs single trial with value
        let nilTrial = PersistedTrial(number: 0, state: .fail, value: nil, params: [:])
        let validTrial = PersistedTrial(number: 1, state: .complete, value: 42.0, params: [:])
        #expect([nilTrial].sortedByValue().isEmpty)
        #expect([validTrial].sortedByValue().map(\.number) == [1])

        // 3. Sort stability with duplicate / identical values and nil values intermixed
        let t0 = PersistedTrial(number: 0, state: .complete, value: 10.0, params: [:])
        let t1 = PersistedTrial(number: 1, state: .fail, value: nil, params: [:])
        let t2 = PersistedTrial(number: 2, state: .complete, value: 5.0, params: [:])
        let t3 = PersistedTrial(number: 3, state: .complete, value: 10.0, params: [:])
        let t4 = PersistedTrial(number: 4, state: .pruned, value: nil, params: [:])
        let t5 = PersistedTrial(number: 5, state: .complete, value: 5.0, params: [:])
        let t6 = PersistedTrial(number: 6, state: .complete, value: 10.0, params: [:])

        let trials = [t0, t1, t2, t3, t4, t5, t6]

        // Ascending sort:
        // Expected values order: 5.0, 5.0, 10.0, 10.0, 10.0
        // Expected trial numbers preserving stability for equal values:
        // 5.0 group: t2 (number 2), t5 (number 5)
        // 10.0 group: t0 (number 0), t3 (number 3), t6 (number 6)
        let sortedAsc = trials.sortedByValue(ascending: true)
        #expect(sortedAsc.map(\.number) == [2, 5, 0, 3, 6])
        #expect(sortedAsc.map(\.value) == [5.0, 5.0, 10.0, 10.0, 10.0])

        // Descending sort:
        // Expected values order: 10.0, 10.0, 10.0, 5.0, 5.0
        // Expected trial numbers preserving stability for equal values:
        // 10.0 group: t0 (number 0), t3 (number 3), t6 (number 6)
        // 5.0 group: t2 (number 2), t5 (number 5)
        let sortedDesc = trials.sortedByValue(ascending: false)
        #expect(sortedDesc.map(\.number) == [0, 3, 6, 2, 5])
        #expect(sortedDesc.map(\.value) == [10.0, 10.0, 10.0, 5.0, 5.0])

        // 4. All identical values
        let identicalValTrials = [
            PersistedTrial(number: 10, state: .complete, value: 7.0, params: [:]),
            PersistedTrial(number: 11, state: .complete, value: 7.0, params: [:]),
            PersistedTrial(number: 12, state: .complete, value: 7.0, params: [:]),
        ]
        #expect(identicalValTrials.sortedByValue(ascending: true).map(\.number) == [10, 11, 12])
        #expect(identicalValTrials.sortedByValue(ascending: false).map(\.number) == [10, 11, 12])
    }
}
