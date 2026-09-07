import Foundation
import Testing
@testable import Swiftuna

@Suite("BruteForceSampler Tests")
struct BruteForceSamplerTests {
    @Test("Flat grid parameter space evaluates all combinations without replacement")
    func testFlatGridExhaustion() throws {
        let sampler = BruteForceSampler(seed: 42)
        let study = try Swiftuna.createStudy(
            name: "brute_force_flat_test",
            direction: .minimize,
            sampler: sampler
        )

        // 2 choices for x * 3 choices for opt = 6 total combinations
        var evaluatedCombinations = Set<String>()
        try study.optimize(nTrials: 20) { trial in
            let x = try trial.suggest("x", in: 1...2)
            let opt = try trial.suggest("opt", choices: ["a", "b", "c"])
            let key = "\(x)_\(opt)"
            #expect(!evaluatedCombinations.contains(key), "Duplicate combination evaluated: \(key)")
            evaluatedCombinations.insert(key)
            return Double(x)
        }

        let trials = try study.trials
        #expect(trials.count == 6)
        #expect(evaluatedCombinations.count == 6)
        #expect(evaluatedCombinations == ["1_a", "1_b", "1_c", "2_a", "2_b", "2_c"])
    }

    @Test("Dynamic conditional search space dynamically branches and explores all leaves")
    func testDynamicConditionalSearchSpace() throws {
        let sampler = BruteForceSampler()
        let study = try Swiftuna.createStudy(
            name: "brute_force_conditional_test",
            direction: .minimize,
            sampler: sampler
        )

        // 1 leaf for "linear" + 3 leaves for "mlp" (layers: 1, 2, 3) = 4 total paths
        var visitedPaths = Set<String>()
        try study.optimize(nTrials: 20) { trial in
            let model = try trial.suggest("model", choices: ["linear", "mlp"])
            if model == "linear" {
                visitedPaths.insert("linear")
                return 1.0
            } else {
                let layers = try trial.suggest("layers", in: 1...3)
                visitedPaths.insert("mlp_\(layers)")
                return Double(layers)
            }
        }

        let trials = try study.trials
        #expect(trials.count == 4)
        #expect(visitedPaths == ["linear", "mlp_1", "mlp_2", "mlp_3"])
    }

    @Test("Deterministic exploration with seed produces identical parameter order")
    func testSeedDeterminism() throws {
        let sampler1 = BruteForceSampler(seed: 999)
        let study1 = try Swiftuna.createStudy(name: "brute_det1", sampler: sampler1)
        var sequence1: [String] = []
        try study1.optimize(nTrials: 10) { trial in
            let x = try trial.suggest("x", in: 1...3)
            let y = try trial.suggest("y", in: 10...12)
            sequence1.append("\(x)_\(y)")
            return Double(x + y)
        }

        let sampler2 = BruteForceSampler(seed: 999)
        let study2 = try Swiftuna.createStudy(name: "brute_det2", sampler: sampler2)
        var sequence2: [String] = []
        try study2.optimize(nTrials: 10) { trial in
            let x = try trial.suggest("x", in: 1...3)
            let y = try trial.suggest("y", in: 10...12)
            sequence2.append("\(x)_\(y)")
            return Double(x + y)
        }

        #expect(sequence1.count == 9)
        #expect(sequence1 == sequence2)
    }

    @Test("Float suggestions with step evaluate all discrete points accurately")
    func testFloatStepExploration() throws {
        let sampler = BruteForceSampler()
        let study = try Swiftuna.createStudy(name: "brute_float_test", sampler: sampler)

        var points: [Double] = []
        try study.optimize(nTrials: 10) { trial in
            let val = try trial.suggest("alpha", in: 0.1...0.5, step: 0.1)
            points.append(val)
            return val
        }

        #expect(points.count == 5)
        let sortedPoints = points.sorted()
        #expect(abs(sortedPoints[0] - 0.1) < 1e-6)
        #expect(abs(sortedPoints[1] - 0.2) < 1e-6)
        #expect(abs(sortedPoints[2] - 0.3) < 1e-6)
        #expect(abs(sortedPoints[3] - 0.4) < 1e-6)
        #expect(abs(sortedPoints[4] - 0.5) < 1e-6)
    }
}
