import Foundation
import Testing

@testable import Swiftuna

private struct CMAPruningTrialTrace: Codable, Sendable {
    let number: Int
    let state: String
    let params: [String: Double]
    let value: Double?
    let intermediate_values: [String: Double]
}

private struct CMAPruningCorpus: Codable, Sendable {
    let problem_name: String
    let seed: UInt64
    let trials: [CMAPruningTrialTrace]
    let best_trial_number: Int
    let best_value: Double
}

private struct ProblemTrialTrace: Codable, Sendable {
    let number: Int
    let state: String
    let params: [String: Double]
    let value: Double
}

private struct ProblemCorpus: Codable, Sendable {
    let problem_name: String
    let seed: UInt64
    let trials: [ProblemTrialTrace]
    let best_trial_number: Int
    let best_value: Double
}

@Suite("Composite Golden Parity Tests", .serialized)
struct CompositeGoldenParityTests {

    private func loadCorpus(name: String) throws -> ProblemCorpus {
        let fileURL = URL(fileURLWithPath: #filePath)
        let repoRoot =
            fileURL
            .deletingLastPathComponent()  // Tests/SwiftunaTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // Repo root
        let fixtureURL =
            repoRoot
            .appendingPathComponent("Tests/Fixtures/ParityCorpus")
            .appendingPathComponent("\(name).json")

        let data: Data
        if FileManager.default.fileExists(atPath: fixtureURL.path) {
            data = try Data(contentsOf: fixtureURL)
        } else {
            data = try Data(contentsOf: URL(fileURLWithPath: "Tests/Fixtures/ParityCorpus/\(name).json"))
        }
        return try JSONDecoder().decode(ProblemCorpus.self, from: data)
    }

    private func loadCMAPruningCorpus() throws -> CMAPruningCorpus {
        let fileURL = URL(fileURLWithPath: #filePath)
        let repoRoot =
            fileURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixtureURL =
            repoRoot
            .appendingPathComponent("Tests/Fixtures/ParityCorpus/cmaes_pruning.json")

        let data: Data
        if FileManager.default.fileExists(atPath: fixtureURL.path) {
            data = try Data(contentsOf: fixtureURL)
        } else {
            data = try Data(contentsOf: URL(fileURLWithPath: "Tests/Fixtures/ParityCorpus/cmaes_pruning.json"))
        }
        return try JSONDecoder().decode(CMAPruningCorpus.self, from: data)
    }

    @Test("Golden parity: CMA-ES sampler combined with MedianPruner reproduces Optuna reference trajectory")
    func testGoldenParityCMAMedianPruner() throws {
        let golden = try loadCMAPruningCorpus()
        let sampler = CMASampler(
            dimensions: [
                .continuous(name: "x", lower: -5.0, upper: 5.0),
                .continuous(name: "y", lower: -5.0, upper: 5.0),
            ],
            seed: golden.seed,
            useNumpyPRNG: true
        )
        let pruner = MedianPruner(nStartupTrials: 2, nWarmupSteps: 1, intervalSteps: 1)
        let study = try Swiftuna.createStudy(
            name: "smoke_parity_cma_median_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)

            for step in 0..<5 {
                let stepVal = pow(x - 1.0, 2) + pow(y + 2.0, 2) + Double(4 - step) * 1.0
                try trial.report(stepVal, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }

            return pow(x - 1.0, 2) + pow(y + 2.0, 2)
        }

        let studyTrials = try study.trials
        #expect(studyTrials.count == golden.trials.count)

        for (actual, expected) in zip(studyTrials, golden.trials) {
            #expect(actual.number == expected.number)
            let expectedState: TrialState = expected.state == "COMPLETE" ? .complete : .pruned
            #expect(
                actual.state == expectedState,
                "Trial \(actual.number) state mismatch: got \(actual.state), expected \(expectedState)"
            )

            let expX = expected.params["x"]!
            let expY = expected.params["y"]!
            let actX = actual.params["x"]?.asDouble ?? 0.0
            let actY = actual.params["y"]?.asDouble ?? 0.0

            #expect(abs(actX - expX) < 1e-7, "Trial \(actual.number) param 'x' drifted: got \(actX), expected \(expX)")
            #expect(abs(actY - expY) < 1e-7, "Trial \(actual.number) param 'y' drifted: got \(actY), expected \(expY)")

            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(
                    abs(actVal - expVal) < 1e-7,
                    "Trial \(actual.number) objective value drifted: got \(actVal), expected \(expVal)"
                )
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7, "Best study value drifted from Optuna reference")
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number, "Best trial number drifted from Optuna reference")
    }

    @Test("Golden parity: BruteForce sampler explores exact reference trajectory matching Optuna")
    func testGoldenParityBruteForceFlat() throws {
        let golden = try loadCorpus(name: "bruteforce")
        let sampler = BruteForceSampler(seed: golden.seed, useNumpyPRNG: true)
        let study = try Swiftuna.createStudy(
            name: "bruteforce_flat",
            direction: .minimize,
            sampler: sampler
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            let x = try trial.suggest("x", in: 0...2)
            let y = try trial.suggest("y", in: 0.0...1.0, step: 0.5)
            return pow(Double(x) - 1.0, 2) + pow(y - 0.5, 2)
        }

        let studyTrials = try study.trials
        #expect(studyTrials.count == golden.trials.count)

        for (actual, expected) in zip(studyTrials, golden.trials) {
            #expect(actual.number == expected.number)
            #expect(actual.state == .complete)

            let expX = expected.params["x"]!
            let expY = expected.params["y"]!
            let actX = actual.params["x"]?.asDouble ?? 0.0
            let actY = actual.params["y"]?.asDouble ?? 0.0

            #expect(abs(actX - expX) < 1e-7, "Trial \(actual.number) param 'x' drifted: got \(actX), expected \(expX)")
            #expect(abs(actY - expY) < 1e-7, "Trial \(actual.number) param 'y' drifted: got \(actY), expected \(expY)")

            let actVal = actual.value ?? 0.0
            #expect(
                abs(actVal - expected.value) < 1e-7,
                "Trial \(actual.number) value drifted: got \(actVal), expected \(expected.value)"
            )
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: BruteForce sampler combined with HyperbandPruner reproduces Optuna reference trajectory")
    func testGoldenParityBruteForceWithHyperbandPruner() throws {
        let golden = try loadCMAPruningCorpusGeneric(name: "bruteforce_hyperband")

        let sampler = BruteForceSampler(seed: golden.seed, useNumpyPRNG: true)
        let pruner = HyperbandPruner(minResource: 1, maxResource: 8, reductionFactor: 2)
        let study = try Swiftuna.createStudy(
            name: "bruteforce_hyperband",
            direction: .minimize,
            sampler: sampler,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            let x = try trial.suggest("x", in: 0...2)
            let y = try trial.suggest("y", in: 0.0...1.0, step: 0.5)

            for step in 0..<8 {
                let stepVal = pow(Double(x) - 1.0, 2) + pow(y - 0.5, 2) + Double(7 - step) * 0.5
                try trial.report(stepVal, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }

            return pow(Double(x) - 1.0, 2) + pow(y - 0.5, 2)
        }

        let studyTrials = try study.trials
        #expect(studyTrials.count == golden.trials.count)

        for (actual, expected) in zip(studyTrials, golden.trials) {
            #expect(actual.number == expected.number)
            let expectedState: TrialState = expected.state == "COMPLETE" ? .complete : .pruned
            #expect(
                actual.state == expectedState,
                "Trial \(actual.number) state mismatch: got \(actual.state), expected \(expectedState)"
            )

            let expX = expected.params["x"]!
            let expY = expected.params["y"]!
            let actX = actual.params["x"]?.asDouble ?? 0.0
            let actY = actual.params["y"]?.asDouble ?? 0.0

            #expect(abs(actX - expX) < 1e-7, "Trial \(actual.number) param 'x' drifted: got \(actX), expected \(expX)")
            #expect(abs(actY - expY) < 1e-7, "Trial \(actual.number) param 'y' drifted: got \(actY), expected \(expY)")

            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(
                    abs(actVal - expVal) < 1e-7,
                    "Trial \(actual.number) objective value drifted: got \(actVal), expected \(expVal)"
                )
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7, "Best study value drifted from Optuna reference")
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number, "Best trial number drifted from Optuna reference")
    }

    private func loadCMAPruningCorpusGeneric(name: String) throws -> CMAPruningCorpus {
        let fileURL = URL(fileURLWithPath: #filePath)
        let repoRoot = fileURL.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixtureURL = repoRoot.appendingPathComponent("Tests/Fixtures/ParityCorpus/\(name).json")
        let data: Data
        if FileManager.default.fileExists(atPath: fixtureURL.path) {
            data = try Data(contentsOf: fixtureURL)
        } else {
            data = try Data(contentsOf: URL(fileURLWithPath: "Tests/Fixtures/ParityCorpus/\(name).json"))
        }
        return try JSONDecoder().decode(CMAPruningCorpus.self, from: data)
    }
}
