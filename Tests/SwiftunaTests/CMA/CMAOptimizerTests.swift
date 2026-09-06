//
//  CMAOptimizerTests.swift
//  SwiftunaTests
//

import Testing
import Foundation
@testable import Swiftuna

@Suite("CMA-ES Optimizer Convergence & State Tests")
struct CMAOptimizerTests {

    @Test("CMAOptimizer converges to global minimum on 2D Sphere problem")
    func testSphereConvergence() {
        // Sphere problem: f(x0, x1) = (x0 - 3.0)^2 + (x1 + 2.0)^2
        // Global minimum at (3.0, -2.0) with value 0.0
        let target = [3.0, -2.0]
        func evaluate(_ x: [Double]) -> Double {
            let dx = x[0] - target[0]
            let dy = x[1] - target[1]
            return dx * dx + dy * dy
        }

        var optimizer = CMAOptimizer(
            mean: [0.0, 0.0],
            sigma: 1.3
        )
        var rng = FastPRNG(seed: 42)

        var bestValue = Double.infinity
        var bestSolution = [0.0, 0.0]

        // 35 generations * population_size (6) = ~210 evaluations
        for _ in 0..<35 {
            var solutions = [(point: [Double], value: Double)]()
            for _ in 0..<optimizer.populationSize {
                let x = optimizer.ask(rng: &rng)
                let val = evaluate(x)
                solutions.append((point: x, value: val))
                if val < bestValue {
                    bestValue = val
                    bestSolution = x
                }
            }
            optimizer.tell(solutions)
        }

        // Must converge to within 1e-3 of global minimum
        #expect(bestValue < 1e-3, "Failed to converge on Sphere: bestValue=\(bestValue)")
        #expect(abs(bestSolution[0] - target[0]) < 0.05)
        #expect(abs(bestSolution[1] - target[1]) < 0.05)
    }

    @Test("CMAOptimizer checkpoint serialization preserves state identically")
    func testCheckpointRoundTrip() throws {
        var optimizer = CMAOptimizer(
            mean: [1.5, -0.5],
            sigma: 0.8
        )
        var rng = FastPRNG(seed: 123)

        // Run 2 generations to populate evolution paths and covariance
        for _ in 0..<2 {
            var solutions = [(point: [Double], value: Double)]()
            for _ in 0..<optimizer.populationSize {
                let x = optimizer.ask(rng: &rng)
                let val = x[0] * x[0] + x[1] * x[1]
                solutions.append((point: x, value: val))
            }
            optimizer.tell(solutions)
        }

        let ckpt = optimizer.exportCheckpoint()
        let data = try JSONEncoder().encode(ckpt)
        let decoded = try JSONDecoder().decode(CMACheckpoint.self, from: data)

        var restored = CMAOptimizer(mean: [0.0, 0.0], sigma: 1.0)
        restored.restoreCheckpoint(decoded)

        #expect(restored.generation == optimizer.generation)
        #expect(restored.sigma == optimizer.sigma)
        #expect(restored.mean == optimizer.mean)
        #expect(restored.pSigma == optimizer.pSigma)
        #expect(restored.pc == optimizer.pc)
        #expect(restored.C.buffer == optimizer.C.buffer)
    }

    @Test("CMAOptimizer matches Python cmaes tell() step mathematically")
    func testPythonParitySingleTell() {
        var optimizer = CMAOptimizer(mean: [0.5, 0.5], sigma: 0.2)
        let solutions: [(point: [Double], value: Double)] = [
            (point: [0.55, 0.45], value: 0.1),
            (point: [0.48, 0.52], value: 0.2),
            (point: [0.60, 0.40], value: 0.3),
            (point: [0.40, 0.60], value: 0.4),
            (point: [0.65, 0.35], value: 0.5),
            (point: [0.35, 0.65], value: 0.6),
        ]
        optimizer.tell(solutions)

        // Golden values from CyberAgent cmaes 0.12.0
        let expectedMean = [0.5339994405453752, 0.4660005594546248]
        let expectedSigma = 0.15757743095596663
        let expectedPc = [0.2244130378186392, -0.22441303781863897]
        let expectedPSigma = [0.20160685027382935, -0.20160685027382916]
        let expectedC = [
            0.7987243784763906, 0.11631486368195525,
            0.11631486368195525, 0.7987243784763904
        ]

        for i in 0..<2 {
            #expect(abs(optimizer.mean[i] - expectedMean[i]) < 1e-12)
            #expect(abs(optimizer.pc[i] - expectedPc[i]) < 1e-12)
            #expect(abs(optimizer.pSigma[i] - expectedPSigma[i]) < 1e-12)
        }
        #expect(abs(optimizer.sigma - expectedSigma) < 1e-12)
        for i in 0..<4 {
            #expect(abs(optimizer.C.buffer[i] - expectedC[i]) < 1e-12)
        }
    }
}
