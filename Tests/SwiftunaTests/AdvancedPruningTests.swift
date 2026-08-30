import Foundation
import Testing
import Swiftuna

@Suite("Advanced Bandits & Patient Pruner Tests", .serialized)
struct AdvancedPruningTests {

    @Test("SuccessiveHalvingPruner only prunes on geometric rungs and promotes top 1/eta fraction")
    func testSuccessiveHalvingRungsAndPromotion() throws {
        // minResource = 1, eta = 2 -> rungs: 1, 2, 4, 8...
        let pruner = SuccessiveHalvingPruner(
            minResource: 1,
            reductionFactor: 2,
            minEarlyStoppingRate: 0,
            bootstrapCount: 2
        )

        #expect(!pruner.isRung(step: 0))
        #expect(pruner.isRung(step: 1))
        #expect(pruner.isRung(step: 2))
        #expect(!pruner.isRung(step: 3))
        #expect(pruner.isRung(step: 4))

        let study = try Swiftuna.createStudy(
            name: "asha_rungs_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Seed 2 baseline trials at rung 1
        var t0 = try study.ask()
        try t0.report(10.0, step: 1)
        try study.tell(consuming: t0, value: 10.0)

        var t1 = try study.ask()
        try t1.report(20.0, step: 1)
        try study.tell(consuming: t1, value: 20.0)

        // Trial 2: At non-rung step 3, should NOT prune even with terrible value
        var t2 = try study.ask()
        try t2.report(999.0, step: 3)
        let pruneNonRung = try pruner.shouldPrune(study: study, trialNumber: t2.number, step: 3, currentValue: 999.0)
        #expect(!pruneNonRung)

        // At rung step 1: values are [10.0, 20.0]. Top 1/2 = top 1 (threshold 10.0).
        // If t2 reports 15.0 (worse than 10.0), it must be pruned.
        try t2.report(15.0, step: 1)
        let pruneRung1 = try pruner.shouldPrune(study: study, trialNumber: t2.number, step: 1, currentValue: 15.0)
        #expect(pruneRung1)

        // If t2 reports 5.0 (better than 10.0), it should NOT be pruned.
        let pruneGood = try pruner.shouldPrune(study: study, trialNumber: t2.number, step: 1, currentValue: 5.0)
        #expect(!pruneGood)
        try study.tell(consuming: t2, value: 5.0)
    }

    @Test("HyperbandPruner partitions trials into deterministic round-robin brackets")
    func testHyperbandBracketPartitioning() throws {
        // minResource = 1, maxResource = 8, eta = 2 -> 4 brackets (0, 1, 2, 3)
        let hyperband = HyperbandPruner(
            minResource: 1,
            maxResource: 8,
            reductionFactor: 2
        )

        #expect(hyperband.nBrackets == 4)
        #expect(hyperband.bracket(for: 0) == 0)
        #expect(hyperband.bracket(for: 1) == 1)
        #expect(hyperband.bracket(for: 2) == 2)
        #expect(hyperband.bracket(for: 3) == 3)
        #expect(hyperband.bracket(for: 4) == 0)
        #expect(hyperband.bracket(for: 7) == 3)

        let study = try Swiftuna.createStudy(
            name: "hyperband_run_\(UUID().uuidString)",
            pruner: hyperband
        )

        // Run 16 trials with reporting
        try study.optimize(nTrials: 16) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            for step in 1...4 {
                try trial.report(x * x + Double(step), step: step, pruneIfWorse: true)
            }
            return x * x
        }

        let trials = try study.trials
        #expect(trials.count == 16)
        let completed = trials.filter { $0.state == TrialState.complete }
        let pruned = trials.filter { $0.state == TrialState.pruned }
        #expect(completed.count > 0)
        #expect(pruned.count >= 0)
    }

    @Test("PatientPruner suppresses pruning during patience window and delegates or triggers standalone")
    func testPatientPrunerDualMode() throws {
        // 1. Wrapped mode: Wraps ThresholdPruner(upper: 5.0) with patience = 2
        let wrappedPruner = PatientPruner(
            wrappedPruner: ThresholdPruner(upper: 5.0),
            patience: 2,
            minDelta: 0.0
        )

        let studyWrapped = try Swiftuna.createStudy(
            name: "patient_wrapped_\(UUID().uuidString)",
            direction: .minimize,
            pruner: wrappedPruner
        )

        var t0 = try studyWrapped.ask()
        // Step 0: 2.0 (below 5.0)
        try t0.report(2.0, step: 0)
        #expect(try !wrappedPruner.shouldPrune(study: studyWrapped, trialNumber: t0.number, step: 0, currentValue: 2.0))

        // Step 1: 10.0 (exceeds threshold! But patience=2 allows it)
        try t0.report(10.0, step: 1)
        #expect(try !wrappedPruner.shouldPrune(study: studyWrapped, trialNumber: t0.number, step: 1, currentValue: 10.0))

        // Step 2: 10.0 (second violation, still within patience limit of 2 steps)
        try t0.report(10.0, step: 2)
        #expect(try !wrappedPruner.shouldPrune(study: studyWrapped, trialNumber: t0.number, step: 2, currentValue: 10.0))

        // Step 3: 10.0 (third violation, patience exhausted! Pruning triggers)
        try t0.report(10.0, step: 3)
        #expect(try wrappedPruner.shouldPrune(study: studyWrapped, trialNumber: t0.number, step: 3, currentValue: 10.0))
        try studyWrapped.tell(consuming: t0, values: [], state: .pruned)

        // 2. Standalone mode: No wrapped pruner, stops when not improving for patience = 2
        let standalonePruner = PatientPruner(
            wrappedPruner: nil,
            patience: 2,
            minDelta: 0.5
        )

        let studyStandalone = try Swiftuna.createStudy(
            name: "patient_standalone_\(UUID().uuidString)",
            direction: .minimize,
            pruner: standalonePruner
        )

        var t1 = try studyStandalone.ask()
        // Step 0: 5.0
        try t1.report(5.0, step: 0)
        #expect(try !standalonePruner.shouldPrune(study: studyStandalone, trialNumber: t1.number, step: 0, currentValue: 5.0))

        // Step 1: 4.0 (improved by 1.0 > 0.5 minDelta!)
        try t1.report(4.0, step: 1)
        #expect(try !standalonePruner.shouldPrune(study: studyStandalone, trialNumber: t1.number, step: 1, currentValue: 4.0))

        // Step 2: 4.0 (no improvement, 1 step)
        try t1.report(4.0, step: 2)
        #expect(try !standalonePruner.shouldPrune(study: studyStandalone, trialNumber: t1.number, step: 2, currentValue: 4.0))

        // Step 3: 4.0 (no improvement, 2 steps: patience boundary)
        try t1.report(4.0, step: 3)
        #expect(try !standalonePruner.shouldPrune(study: studyStandalone, trialNumber: t1.number, step: 3, currentValue: 4.0))

        // Step 4: 4.0 (no improvement, 3 steps: patience exhausted!)
        try t1.report(4.0, step: 4)
        #expect(try standalonePruner.shouldPrune(study: studyStandalone, trialNumber: t1.number, step: 4, currentValue: 4.0))
        try studyStandalone.tell(consuming: t1, values: [], state: .pruned)
    }
}
