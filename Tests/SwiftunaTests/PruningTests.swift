import Foundation
import Testing
@testable import Swiftuna

@Suite("Pruning & Intermediate Evaluation Tests")
struct PruningTests {

    private func makeTempDBURL() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        return tempDir.appendingPathComponent("swiftuna_pruning_\(UUID().uuidString).db")
    }

    @Test("MedianPruner correctly prunes unpromising trial trajectories and persists intermediate values")
    func testMedianPrunerAndPersistence() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let storage = StorageBackend.sqlite(url: dbURL)
        let pruner = MedianPruner(nStartupTrials: 3, nWarmupSteps: 3, intervalSteps: 1)
        let study = try Swiftuna.createStudy(
            name: "pruning_study",
            direction: .minimize,
            storage: storage,
            pruner: pruner
        )

        // Baseline trials 0, 1, 2 (good performance across 5 steps)
        for _ in 0..<3 {
            var trial = try study.ask()
            for step in 0..<5 {
                try trial.report(Double(step) * 0.1, step: step)
            }
            try study.tell(consuming: trial, value: 0.5)
        }

        #expect((try study.trials).count == 3)

        // Trial 3 has very bad loss at step 3 (> median of 0.3)
        var trial3 = try study.ask()
        try trial3.report(0.1, step: 0)
        try trial3.report(0.2, step: 1)
        try trial3.report(0.3, step: 2)

        // Warmup steps are 2, so at step 2 it should not prune yet
        #expect(try trial3.shouldPrune == false)

        // At step 3, report 999.0 (much worse than median 0.3)
        try trial3.report(999.0, step: 3)
        #expect(try trial3.shouldPrune == true)

        // Pruning via tell
        try study.tell(consuming: trial3, state: .pruned)

        // Reload from SQLite to verify storage persistence
        let reloaded = try Swiftuna.loadStudy(name: "pruning_study", storage: storage)
        let trials = try reloaded.trials
        #expect(trials.count == 4)

        let prunedTrial = trials[3]
        #expect(prunedTrial.state == .pruned)
        #expect(prunedTrial.intermediateValues.count == 4)
        #expect(prunedTrial.intermediateValues[3] == 999.0)
    }

    @Test("trial.report with pruneIfWorse throws trialPruned inside optimize loop")
    func testInlineReportOrPrune() throws {
        let pruner = MedianPruner(nStartupTrials: 2, nWarmupSteps: 1, intervalSteps: 1)
        let study = try Swiftuna.createStudy(
            name: "inline_prune_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Startup trials
        for _ in 0..<2 {
            try study.optimize(nTrials: 1) { trial in
                for step in 0..<4 {
                    try trial.report(Double(step) * 1.0, step: step)
                }
                return 4.0
            }
        }

        // Trial that gets pruned inside the closure
        try study.optimize(nTrials: 1) { trial in
            for step in 0..<4 {
                let loss = (step >= 2) ? 1000.0 : Double(step)
                // This will throw SwiftunaError.trialPruned at step 2
                try trial.report(loss, step: step, pruneIfWorse: true)
            }
            return 1000.0
        }

        let allTrials = try study.trials
        #expect(allTrials.count == 3)
        #expect(allTrials[2].state == .pruned)
        #expect(allTrials[2].intermediateValues.count == 3) // steps 0, 1, 2
    }

    @Test("ThresholdPruner prunes values exceeding upper or lower bounds")
    func testThresholdPruner() throws {
        let pruner = ThresholdPruner(lower: 0.0, upper: 100.0)
        let study = try Swiftuna.createStudy(
            name: "threshold_study_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        var trial = try study.ask()
        try trial.report(50.0, step: 0)
        #expect(try trial.shouldPrune == false)

        try trial.report(150.0, step: 1)
        #expect(try trial.shouldPrune == true)

        try study.tell(consuming: trial, state: .pruned)
        #expect((try study.trials)[0].state == .pruned)
    }

    @Test("PercentilePruner prunes trials in the lower percentile")
    func testPercentilePruner() throws {
        let pruner = PercentilePruner(percentile: 25.0, nStartupTrials: 4, nWarmupSteps: 0)
        let study = try Swiftuna.createStudy(
            name: "percentile_study_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Baseline: values 10, 20, 30, 40 at step 1
        for val in [10.0, 20.0, 30.0, 40.0] {
            var trial = try study.ask()
            try trial.report(val, step: 1)
            try study.tell(consuming: trial, value: val)
        }

        var trial = try study.ask()
        try trial.report(25.0, step: 1)
        // 25th percentile of [10, 20, 30, 40] is index 0 -> 10.0, so 25.0 > 10.0 -> prune
        #expect(try trial.shouldPrune == true)
        try study.tell(consuming: trial, state: .pruned)
    }

    @Test("High-throughput hot-loop cache retains 100 intermediate steps per trial with zero errors")
    func testHotLoopCachePerformance() throws {
        let study = try Swiftuna.createStudy(name: "hot_loop_\(UUID().uuidString)")
        var trial = try study.ask()

        // 100 intermediate steps
        for step in 0..<100 {
            try trial.report(Double(step) * 0.05, step: step)
        }

        try study.tell(consuming: trial, value: 5.0)
        let pTrial = (try study.trials)[0]
        #expect(pTrial.intermediateValues.count == 100)
        #expect(pTrial.intermediateValues[99] == 4.95)
    }
}
