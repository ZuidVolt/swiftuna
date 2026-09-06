//
//  CMASamplerTests.swift
//  SwiftunaTests
//

import Testing
import Foundation
@testable import Swiftuna

@Suite("CMA-ES Sampler End-to-End Tests")
struct CMASamplerTests {

    @Test("CMASampler optimizes continuous 2D quadratic study to minimum")
    func testCMASamplerConvergence() throws {
        let sampler = CMASampler(
            dimensions: [
                .continuous(name: "x", lower: -5.0, upper: 5.0),
                .continuous(name: "y", lower: -5.0, upper: 5.0)
            ],
            seed: 42
        )

        let study = try Swiftuna.createStudy(
            name: "cma_convergence_test",
            direction: .minimize,
            storage: .inMemory
        )

        // Target at (1.5, -2.5) with global minimum 0.0
        try study.optimize(nTrials: 40, using: sampler) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            let dx = x - 1.5
            let dy = y + 2.5
            return dx * dx + dy * dy
        }

        guard let best = try study.bestTrial else {
            Issue.record("Study has no best trial")
            return
        }

        #expect(best.value != nil)
        let loss = best.value!
        #expect(loss < 0.05, "CMA-ES failed to optimize quadratic: best loss=\(loss)")

        let bestX = best.params["x"]?.asDouble ?? 0.0
        let bestY = best.params["y"]?.asDouble ?? 0.0
        #expect(abs(bestX - 1.5) < 0.25)
        #expect(abs(bestY + 2.5) < 0.25)
    }

    @Test("CMASampler respects discrete and continuous boundaries")
    func testCMASamplerMixedBoundaries() throws {
        let sampler = CMASampler(
            dimensions: [
                .continuous(name: "lr", lower: 1e-4, upper: 1e-1, log: true),
                .discrete(name: "layers", lower: 1, upper: 8, step: 1)
            ],
            seed: 99
        )

        let study = try Swiftuna.createStudy(
            name: "cma_boundary_test",
            direction: .minimize,
            storage: .inMemory
        )

        try study.optimize(nTrials: 20, using: sampler) { trial in
            let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
            let layers = try trial.suggest("layers", in: 1...8)

            #expect(lr >= 1e-4 && lr <= 1e-1, "lr out of bounds: \(lr)")
            #expect(layers >= 1 && layers <= 8, "layers out of bounds: \(layers)")

            return (lr - 0.01) * (lr - 0.01) + Double(layers)
        }

        #expect(try study.trials.count == 20)
    }

    @Test("Unified createStudy(sampler: CMASampler) drives optimize() without 'using:' argument")
    func testUnifiedCreateStudyAPI() throws {
        let sampler = CMASampler(
            dimensions: [
                .continuous(name: "x", lower: -5.0, upper: 5.0),
                .continuous(name: "y", lower: -5.0, upper: 5.0)
            ],
            seed: 42
        )

        let study = try Swiftuna.createStudy(
            name: "cma_unified_api_test",
            direction: .minimize,
            sampler: sampler
        )

        #expect(study.customSampler != nil)

        // Directly call optimize(nTrials:) without `using:` - 1:1 parity with Optuna!
        try study.optimize(nTrials: 40) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            let dx = x - 1.5
            let dy = y + 2.5
            return dx * dx + dy * dy
        }

        #expect(try study.trials.count == 40)
        guard let best = try study.bestTrial, let loss = best.value else {
            Issue.record("Study has no completed best trial")
            return
        }
        #expect(loss < 0.05, "Unified API optimization failed: best loss=\(loss)")
    }

    @Test("Stateful Continuous Optimization preserves sampler state across successive optimize() calls")
    func testContinuousMultiStageOptimization() throws {
        let sampler = CMASampler(
            dimensions: [
                .continuous(name: "x", lower: -5.0, upper: 5.0),
                .continuous(name: "y", lower: -5.0, upper: 5.0)
            ],
            seed: 42
        )

        let study = try Swiftuna.createStudy(
            name: "cma_multistage_test",
            direction: .minimize,
            sampler: sampler
        )

        // Stage 1: 20 trials
        try study.optimize(nTrials: 20) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return (x - 1.5) * (x - 1.5) + (y + 2.5) * (y + 2.5)
        }
        #expect(try study.trials.count == 20)

        // Stage 2: 20 more trials (total 40) resumes from same sampler state
        try study.optimize(nTrials: 20) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return (x - 1.5) * (x - 1.5) + (y + 2.5) * (y + 2.5)
        }
        #expect(try study.trials.count == 40)

        guard let best = try study.bestTrial, let loss = best.value else {
            Issue.record("Study has no completed best trial")
            return
        }
        #expect(loss < 0.05, "Multi-stage optimization failed to converge: best loss=\(loss)")
    }
}

