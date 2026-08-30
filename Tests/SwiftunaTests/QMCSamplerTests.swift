import Foundation
import Testing
@testable import Swiftuna

@Suite("Quasi-Monte Carlo (QMC) Sobol Sampler Tests")
struct QMCSamplerTests {

    @Test("QMCSampler with identical seed generates deterministic parameter sequences")
    func testQMCSamplerDeterminism() throws {
        func runStudy() throws -> [Double] {
            let study = try Swiftuna.createStudy(
                name: "test_qmc_det_\(UUID().uuidString)",
                sampler: QMCSampler(seed: 999)
            )

            var suggestions: [Double] = []
            try study.optimize(nTrials: 10) { (trial: inout Trial) in
                let x = try trial.suggest("x", in: -10.0...10.0)
                suggestions.append(x)
                return x * x
            }
            return suggestions
        }

        let run1 = try runStudy()
        let run2 = try runStudy()

        #expect(run1.count == 10)
        #expect(run1 == run2, "QMCSampler must produce identical low-discrepancy sequences given the same seed")
    }

    @Test("QMCSampler Sobol sequence provides low-discrepancy quadrant balance")
    func testQMCSamplerQuadrantBalance() throws {
        let study = try Swiftuna.createStudy(
            name: "test_qmc_balance_\(UUID().uuidString)",
            sampler: QMCSampler(seed: 42)
        )

        var points: [(x: Double, y: Double)] = []
        try study.optimize(nTrials: 64) { (trial: inout Trial) in
            let x = try trial.suggest("x", in: 0.0...1.0)
            let y = try trial.suggest("y", in: 0.0...1.0)
            points.append((x, y))
            return 0.0
        }

        #expect(points.count == 64)

        // Count samples in each of the 4 quadrants:
        // Q1: [0, 0.5) x [0, 0.5)
        // Q2: [0.5, 1.0] x [0, 0.5)
        // Q3: [0, 0.5) x [0.5, 1.0]
        // Q4: [0.5, 1.0] x [0.5, 1.0]
        var q1 = 0, q2 = 0, q3 = 0, q4 = 0
        for p in points {
            if p.x < 0.5 && p.y < 0.5 { q1 += 1 }
            else if p.x >= 0.5 && p.y < 0.5 { q2 += 1 }
            else if p.x < 0.5 && p.y >= 0.5 { q3 += 1 }
            else { q4 += 1 }
        }

        // A power-of-2 Sobol sequence (2^6 = 64) distributes points with exact or near-exact quadrant uniformity (16 each).
        #expect(q1 >= 14 && q1 <= 18, "Quadrant 1 should contain ~16 points, got \(q1)")
        #expect(q2 >= 14 && q2 <= 18, "Quadrant 2 should contain ~16 points, got \(q2)")
        #expect(q3 >= 14 && q3 <= 18, "Quadrant 3 should contain ~16 points, got \(q3)")
        #expect(q4 >= 14 && q4 <= 18, "Quadrant 4 should contain ~16 points, got \(q4)")
    }

    @Test("QMCSampler minimizes 2D sphere objective function")
    func testQMCOptimizationConvergence() throws {
        let study = try Swiftuna.createStudy(
            name: "test_qmc_sphere_\(UUID().uuidString)",
            direction: .minimize,
            sampler: QMCSampler(seed: 123)
        )

        try study.optimize(nTrials: 64) { (trial: inout Trial) in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return x * x + y * y
        }

        let best = try #require(try study.bestTrial)
        let bestVal = try #require(best.value)
        #expect(bestVal < 1.0)
    }

    @Test("QMCSampler supports mixed distributions: Float, Int, and Categorical")
    func testQMCMixedDistributions() throws {
        let study = try Swiftuna.createStudy(
            name: "test_qmc_mixed_\(UUID().uuidString)",
            sampler: QMCSampler(seed: 77)
        )

        let architectures = ["transformer", "mamba", "rwkv"]

        try study.optimize(nTrials: 20) { (trial: inout Trial) in
            let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
            let layers = try trial.suggest("layers", in: 1...12)
            let arch = try trial.suggest("arch", choices: architectures)

            #expect(lr >= 1e-4 && lr <= 1e-1)
            #expect(layers >= 1 && layers <= 12)
            #expect(architectures.contains(arch))

            return lr * Double(layers)
        }

        let trials = try study.trials
        #expect(trials.count == 20)
    }
}
