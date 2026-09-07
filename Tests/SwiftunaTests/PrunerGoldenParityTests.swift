import Foundation
import Testing

@testable import Swiftuna

private struct PruningTrialTrace: Codable, Sendable {
    let number: Int
    let state: String
    let params: [String: Double]
    let value: Double?
    let intermediate_values: [String: Double]
}

private struct PruningCorpus: Codable, Sendable {
    let problem_name: String
    let seed: UInt64
    let trials: [PruningTrialTrace]
    let best_trial_number: Int
    let best_value: Double
}

@Suite("Pruner Golden Parity Tests", .serialized)
struct PrunerGoldenParityTests {

    private func loadCorpus(name: String) throws -> PruningCorpus {
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
        return try JSONDecoder().decode(PruningCorpus.self, from: data)
    }

    @Test("Golden parity: ThresholdPruner matches Optuna reference trajectory")
    func testGoldenParityThresholdPruner() throws {
        let golden = try loadCorpus(name: "threshold_pruning")
        let pruner = ThresholdPruner(lower: 0.0, upper: 10.0, nWarmupSteps: 2, intervalSteps: 2)
        let study = try Swiftuna.createStudy(
            name: "threshold_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let n = trial.number
            var finalVal = 0.0
            for step in 0..<6 {
                let val: Double
                if n == 0 {
                    val = 5.0 - Double(step) * 0.5
                } else if n == 1 {
                    val = 15.0 - Double(step) * 2.0
                } else if n == 2 {
                    val = step == 4 ? 20.0 : 4.0
                } else if n == 3 {
                    val = step == 4 ? -2.0 : 2.0
                } else if n == 4 {
                    val = step == 2 ? -10.0 : 5.0
                } else {
                    val = 3.0 + Double(step) * 0.2
                }
                try trial.report(val, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
                finalVal = val
            }
            return finalVal
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) intermediate values count mismatch: got \(actual.intermediateValues.count), expected \(expected.intermediate_values.count)"
            )
            for (k, v) in expected.intermediate_values {
                let step = Int(k)!
                let actVal = actual.intermediateValues[step] ?? 0.0
                #expect(abs(actVal - v) < 1e-7)
            }
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: PercentilePruner matches Optuna reference trajectory")
    func testGoldenParityPercentilePruner() throws {
        let golden = try loadCorpus(name: "percentile_pruning")
        let trajectories: [[Double]] = [
            [10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            [12.0, 10.0, 8.0, 6.0, 4.0, 3.0],
            [8.0, 6.0, 4.0, 2.0, 1.0, 0.5],
            [5.0, 15.0, 4.0, 3.0, 2.0, 1.0],
            [20.0, 25.0, 25.0, 25.0, 25.0, 25.0],
            [7.0, 7.0, 12.0, 12.0, 2.0, 1.0],
            [9.0, 9.0, 2.0, 2.0, 1.0, 0.5],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        ]

        let pruner = PercentilePruner(
            percentile: 25.0,
            nStartupTrials: 2,
            nWarmupSteps: 1,
            intervalSteps: 2,
            nMinTrials: 2
        )
        let study = try Swiftuna.createStudy(
            name: "percentile_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let curve = trajectories[trial.number]
            for step in 0..<curve.count {
                let val = curve[step]
                try trial.report(val, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }
            return curve.last ?? 0.0
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) steps count mismatch"
            )
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: SuccessiveHalvingPruner explicit matches Optuna reference trajectory")
    func testGoldenParitySuccessiveHalvingExplicit() throws {
        let golden = try loadCorpus(name: "asha_pruning")
        let trajectories: [[Double]] = (0..<7).map { trialIdx in
            let base: Double
            switch trialIdx {
            case 0: base = 10.0
            case 1: base = 15.0
            case 2: base = 8.0
            case 3: base = 12.0
            case 4: base = 6.0
            case 5: base = 11.0
            default: base = 5.0
            }
            return (0..<10).map { base - Double($0) * 0.5 }
        }

        let pruner = SuccessiveHalvingPruner(
            minResource: 1,
            reductionFactor: 3,
            minEarlyStoppingRate: 0,
            bootstrapCount: 0
        )
        let study = try Swiftuna.createStudy(
            name: "asha_explicit_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let curve = trajectories[trial.number]
            for step in 0..<curve.count {
                let val = curve[step]
                try trial.report(val, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }
            return curve.last ?? 0.0
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) intermediate steps mismatch"
            )
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: SuccessiveHalvingPruner auto matches Optuna reference trajectory")
    func testGoldenParitySuccessiveHalvingAuto() throws {
        let golden = try loadCorpus(name: "asha_auto_pruning")
        let pruner = SuccessiveHalvingPruner(minResource: .auto, reductionFactor: 2)
        let study = try Swiftuna.createStudy(
            name: "asha_auto_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let n = trial.number
            if n == 0 {
                for step in 0..<200 {
                    try trial.report(50.0 - Double(step) * 0.1, step: step)
                }
                return 30.1
            } else if n == 1 {
                for step in 0..<10 {
                    try trial.report(10.0 - Double(step) * 0.5, step: step)
                    if try trial.shouldPrune {
                        throw SwiftunaError.trialPruned(reason: nil)
                    }
                }
                return 5.5
            } else if n == 2 {
                for step in 0..<10 {
                    try trial.report(99.0, step: step)
                    if try trial.shouldPrune {
                        throw SwiftunaError.trialPruned(reason: nil)
                    }
                }
                return 99.0
            } else if n == 3 {
                for step in 0..<10 {
                    try trial.report(5.0 - Double(step) * 0.5, step: step)
                    if try trial.shouldPrune {
                        throw SwiftunaError.trialPruned(reason: nil)
                    }
                }
                return 0.5
            } else {
                for step in 0..<10 {
                    let val = step < 2 ? 4.0 : 50.0
                    try trial.report(val, step: step)
                    if try trial.shouldPrune {
                        throw SwiftunaError.trialPruned(reason: nil)
                    }
                }
                return 50.0
            }
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) intermediate steps count mismatch"
            )
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: PatientPruner standalone matches Optuna reference trajectory")
    func testGoldenParityPatientStandalone() throws {
        let golden = try loadCorpus(name: "patient_standalone")
        let trajectories: [[Double]] = [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            [10.0, 9.0, 10.0, 10.0, 10.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1],
            [8.0, 7.0, 8.0, 8.0, 8.0],
        ]

        let pruner = PatientPruner(wrappedPruner: nil, patience: 2, minDelta: 0.5)
        let study = try Swiftuna.createStudy(
            name: "patient_sa_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let curve = trajectories[trial.number]
            for step in 0..<curve.count {
                let val = curve[step]
                try trial.report(val, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }
            return curve.last ?? 0.0
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) steps count mismatch"
            )
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }

    @Test("Golden parity: PatientPruner wrapped matches Optuna reference trajectory")
    func testGoldenParityPatientWrapped() throws {
        let golden = try loadCorpus(name: "patient_wrapped")
        let trajectories: [[Double]] = [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [10.0, 5.0, 10.0, 10.0, 10.0],
        ]

        let base = MedianPruner(nStartupTrials: 2)
        let pruner = PatientPruner(wrappedPruner: base, patience: 2, minDelta: 0.5)
        let study = try Swiftuna.createStudy(
            name: "patient_wrapped_parity_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        try study.optimize(nTrials: golden.trials.count) { trial in
            _ = try trial.suggest("p", in: 0.0...10.0)
            let curve = trajectories[trial.number]
            for step in 0..<curve.count {
                let val = curve[step]
                try trial.report(val, step: step)
                if try trial.shouldPrune {
                    throw SwiftunaError.trialPruned(reason: nil)
                }
            }
            return curve.last ?? 0.0
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
            #expect(
                actual.intermediateValues.count == expected.intermediate_values.count,
                "Trial \(actual.number) steps count mismatch"
            )
            if expected.state == "COMPLETE" {
                let expVal = expected.value!
                let actVal = actual.value ?? 0.0
                #expect(abs(actVal - expVal) < 1e-7)
            }
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
        let bestTrial = try study.bestTrial
        #expect(bestTrial?.number == golden.best_trial_number)
    }
}
