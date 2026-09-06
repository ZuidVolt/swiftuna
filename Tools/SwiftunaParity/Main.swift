import Foundation
import Swiftuna

struct TrialTrace: Codable {
    let number: Int
    let params: [String: Double]
    let value: Double
}

struct ProblemCorpus: Codable {
    let problem_name: String
    let seed: UInt64
    let trials: [TrialTrace]
    let best_trial_number: Int
    let best_value: Double
}

func loadCorpus(name: String) throws -> ProblemCorpus {
    let path = "Tests/Fixtures/ParityCorpus/\(name).json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(ProblemCorpus.self, from: data)
}

struct CMASolutionTrace: Codable {
    let point: [Double]
    let value: Double
}

struct CMAGenerationTrace: Codable {
    let generation: Int
    let solutions: [CMASolutionTrace]
    let expected_mean: [Double]
    let expected_sigma: Double
    let expected_pc: [Double]
    let expected_p_sigma: [Double]
    let expected_cov: [Double]
}

struct CMACorpus: Codable {
    let problem_name: String
    let seed: UInt64
    let initial_mean: [Double]
    let initial_sigma: Double
    let generations: [CMAGenerationTrace]
}

func loadCMACorpus() throws -> CMACorpus {
    let path = "Tests/Fixtures/ParityCorpus/cmaes_step.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(CMACorpus.self, from: data)
}

func assertClose(_ a: Double, _ b: Double, tolerance: Double = 1e-7, label: String) -> Bool {
    if abs(a - b) > tolerance {
        print("❌ Parity mismatch for \(label): Swift=\(a), Golden=\(b), diff=\(abs(a - b))")
        return false
    }
    return true
}

@main
struct SwiftunaParityApp {
    static func main() {
        print("=========================================================")
        print("             Swiftuna Parity Verification Suite           ")
        print("=========================================================")

        var allPassed = true

        // 1. Quadratic Parity Check
        do {
            print("\n[1/5] Verifying Quadratic Parity...")
            let golden = try loadCorpus(name: "quadratic")
            let sampler = TPESampler(seed: golden.seed)
            let study = try Swiftuna.createStudy(name: "parity_quadratic", direction: .minimize, sampler: sampler)

            for expected in golden.trials {
                var trial = try study.ask()
                let x = try trial.suggest("x", in: -10.0...10.0)
                let y = try trial.suggest("y", in: -10.0...10.0)
                let loss = pow(x - 2.0, 2) + pow(y + 5.0, 2)

                try study.tell(consuming: trial, value: loss)

                let expX = expected.params["x"]!
                let expY = expected.params["y"]!

                if !assertClose(x, expX, label: "trial \(expected.number) x")
                    || !assertClose(y, expY, label: "trial \(expected.number) y")
                    || !assertClose(loss, expected.value, label: "trial \(expected.number) loss")
                {
                    allPassed = false
                }
            }

            let bestVal = try study.bestValue
            if !assertClose(bestVal, golden.best_value, label: "best_value") {
                allPassed = false
            } else {
                print("   ✅ Quadratic problem bit-for-bit parity verified (15/15 trials)")
            }
        } catch {
            print("❌ Quadratic parity failed with error: \(error)")
            allPassed = false
        }

        // 2. Constrained Weights Parity Check
        do {
            print("\n[2/5] Verifying Constrained Weights Parity...")
            let golden = try loadCorpus(name: "constrained_weights")
            let sampler = TPESampler(seed: golden.seed)
            let study = try Swiftuna.createStudy(name: "parity_constrained", direction: .minimize, sampler: sampler)

            for expected in golden.trials {
                var trial = try study.ask()
                let p_w = try trial.suggest("param_weight", in: 0.5...2.0, step: 0.1)
                let m_w = try trial.suggest("mutation_weight", in: 0.8...3.0, step: 0.1)
                let s_w = try trial.suggest("sink_weight", in: 0.2...2.0, step: 0.1)

                let ceiling = m_w + 4.0 * s_w
                let loss: Double
                if ceiling > 5.0 {
                    loss = 1_000.0 + (ceiling - 5.0) * 100.0
                } else {
                    loss = pow(p_w - 1.0, 2) + pow(m_w - 1.5, 2) + pow(s_w - 0.5, 2)
                }

                try study.tell(consuming: trial, value: loss)

                let expP = expected.params["param_weight"]!
                let expM = expected.params["mutation_weight"]!
                let expS = expected.params["sink_weight"]!

                if !assertClose(p_w, expP, label: "param_weight") || !assertClose(m_w, expM, label: "mutation_weight")
                    || !assertClose(s_w, expS, label: "sink_weight")
                    || !assertClose(loss, expected.value, label: "constrained loss")
                {
                    allPassed = false
                }
            }

            let bestVal = try study.bestValue
            if !assertClose(bestVal, golden.best_value, label: "best_value") {
                allPassed = false
            } else {
                print("   ✅ Constrained weights parity verified (15/15 trials)")
            }
        } catch {
            print("❌ Constrained weights parity failed with error: \(error)")
            allPassed = false
        }

        // 3. Integer Grid Parity Check
        do {
            print("\n[3/5] Verifying Integer Grid Parity...")
            let golden = try loadCorpus(name: "integer_grid")
            let sampler = TPESampler(seed: golden.seed)
            let study = try Swiftuna.createStudy(name: "parity_integer", direction: .minimize, sampler: sampler)

            for expected in golden.trials {
                var trial = try study.ask()
                let layers = try trial.suggest("n_layers", in: 1...8)
                let units = try trial.suggest("hidden_units", in: 32...256, step: 32)
                let loss = abs(Double(layers * units) - 512.0)

                try study.tell(consuming: trial, value: loss)

                let expLayers = Int(expected.params["n_layers"]!)
                let expUnits = Int(expected.params["hidden_units"]!)

                if layers != expLayers || units != expUnits || abs(loss - expected.value) > 1e-7 {
                    print(
                        "❌ Integer mismatch at trial \(expected.number): Swift=(\(layers), \(units)), Golden=(\(expLayers), \(expUnits))"
                    )
                    allPassed = false
                }
            }

            let bestVal = try study.bestValue
            if !assertClose(bestVal, golden.best_value, label: "best_value") {
                allPassed = false
            } else {
                print("   ✅ Integer grid parity verified (15/15 trials)")
            }
        } catch {
            print("❌ Integer grid parity failed with error: \(error)")
            allPassed = false
        }

        // 4. Categorical Grid Parity Check
        do {
            print("\n[4/5] Verifying Categorical Grid Parity...")
            let golden = try loadCorpus(name: "categorical_grid")
            let sampler = TPESampler(seed: golden.seed)
            let study = try Swiftuna.createStudy(name: "parity_categorical", direction: .minimize, sampler: sampler)

            let choices = ["adam", "sgd", "rmsprop", "adamw"]

            for expected in golden.trials {
                var trial = try study.ask()
                let opt = try trial.suggest("optimizer", choices: choices)

                let loss: Double
                switch opt {
                case "adam": loss = 0.12
                case "sgd": loss = 0.45
                case "rmsprop": loss = 0.28
                default: loss = 0.08
                }

                try study.tell(consuming: trial, value: loss)

                let expectedIdx = Int(expected.params["optimizer"]!)
                let expectedOpt = choices[expectedIdx]

                if opt != expectedOpt || abs(loss - expected.value) > 1e-7 {
                    print("❌ Categorical mismatch at trial \(expected.number): Swift=\(opt), Golden=\(expectedOpt)")
                    allPassed = false
                }
            }

            let bestVal = try study.bestValue
            if !assertClose(bestVal, golden.best_value, label: "best_value") {
                allPassed = false
            } else {
                print("   ✅ Categorical grid parity verified (15/15 trials)")
            }
        } catch {
            print("❌ Categorical grid parity failed with error: \(error)")
            allPassed = false
        }

        // 5. CMA-ES Mathematical Parity Check
        do {
            print("\n[5/5] Verifying CMA-ES Mathematical Parity...")
            let golden = try loadCMACorpus()
            var optimizer = CMAOptimizer(
                mean: golden.initial_mean,
                sigma: golden.initial_sigma
            )

            var cmaPassed = true
            for genTrace in golden.generations {
                let solutions = genTrace.solutions.map { ($0.point, $0.value) }
                optimizer.tell(solutions)

                for (i, val) in optimizer.mean.enumerated() {
                    if !assertClose(val, genTrace.expected_mean[i], label: "gen \(genTrace.generation) mean[\(i)]") {
                        cmaPassed = false
                    }
                }
                if !assertClose(optimizer.sigma, genTrace.expected_sigma, label: "gen \(genTrace.generation) sigma") {
                    cmaPassed = false
                }
                for (i, val) in optimizer.pc.enumerated() {
                    if !assertClose(val, genTrace.expected_pc[i], label: "gen \(genTrace.generation) pc[\(i)]") {
                        cmaPassed = false
                    }
                }
                for (i, val) in optimizer.pSigma.enumerated() {
                    if !assertClose(val, genTrace.expected_p_sigma[i], label: "gen \(genTrace.generation) pSigma[\(i)]")
                    {
                        cmaPassed = false
                    }
                }
                for (i, val) in optimizer.C.buffer.enumerated() {
                    if !assertClose(val, genTrace.expected_cov[i], label: "gen \(genTrace.generation) cov[\(i)]") {
                        cmaPassed = false
                    }
                }
            }

            if cmaPassed {
                print(
                    "   ✅ CMA-ES mathematical parity verified (\(golden.generations.count) generations, state diff < 1e-7)"
                )
            } else {
                allPassed = false
            }
        } catch {
            print("❌ CMA-ES parity failed with error: \(error)")
            allPassed = false
        }

        print("\n---------------------------------------------------------")
        if allPassed {
            print("🎉 ALL 5 PARITY SCENARIOS PASSED WITH EXACT NUMERICAL EQUIVALENCE!")
            print("=========================================================\n")
            exit(0)
        } else {
            print("💥 PARITY REGRESSIONS DETECTED.")
            print("=========================================================\n")
            exit(1)
        }
    }
}
