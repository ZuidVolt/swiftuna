import Foundation
import Testing

@testable import Swiftuna

@Suite("Pruner Parity & High-Performance Tests")
struct PrunerParityTests {

    @Test("ThresholdPruner honors warmup steps, interval windows, and threshold bounds")
    func testThresholdPruner() throws {
        // Upper = 10.0, warmup = 3 steps (steps 0, 1, 2 never prune), interval = 2
        let pruner = ThresholdPruner(lower: 0.0, upper: 10.0, nWarmupSteps: 3, intervalSteps: 2)

        let study = try Swiftuna.createStudy(
            name: "thresh_test_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        var trial = try study.ask()

        // Step 0: Value 100.0 (exceeds upper, but step < nWarmupSteps 3)
        try trial.report(100.0, step: 0)
        #expect(try !trial.shouldPrune)

        // Step 1: Value 100.0 (warmup)
        try trial.report(100.0, step: 1)
        #expect(try !trial.shouldPrune)

        // Step 2: Value 100.0 (warmup)
        try trial.report(100.0, step: 2)
        #expect(try !trial.shouldPrune)

        // Step 3: First step in interval bucket [3, 5) -> evaluated! 100.0 > 10.0 -> PRUNES!
        try trial.report(100.0, step: 3)
        #expect(try trial.shouldPrune)

        // Reset trial with valid values
        var trial2 = try study.ask()
        try trial2.report(5.0, step: 3)
        #expect(try !trial2.shouldPrune)

        // Step 4: Second step in interval bucket [3, 5) -> skipped by interval window check
        try trial2.report(100.0, step: 4)
        #expect(try !trial2.shouldPrune)

        // Step 5: First step in interval bucket [5, 7) -> evaluated!
        try trial2.report(100.0, step: 5)
        #expect(try trial2.shouldPrune)

        // NaN pruning
        var trial3 = try study.ask()
        try trial3.report(Double.nan, step: 3)
        #expect(try trial3.shouldPrune)
    }

    @Test("PercentilePruner tracks best-so-far intermediate values and respects nMinTrials")
    func testPercentilePruner() throws {
        // Prune if best-so-far is worse than 50th percentile (median) of completed trials
        let pruner = PercentilePruner(
            percentile: 50.0,
            nStartupTrials: 1,
            nWarmupSteps: 0,
            intervalSteps: 1,
            nMinTrials: 2
        )

        let study = try Swiftuna.createStudy(
            name: "percentile_test_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Trial 0: Completes with step 1 value 10.0
        var t0 = try study.ask()
        try t0.report(10.0, step: 1)
        try study.tell(consuming: t0, value: 10.0, state: .complete)

        // Trial 1: Step 1 value is 20.0, but nMinTrials=2 is not yet satisfied (only 1 trial at step 1)
        var t1 = try study.ask()
        try t1.report(20.0, step: 1)
        #expect(try !t1.shouldPrune, "Should not prune because nMinTrials = 2 is not yet reached")
        try study.tell(consuming: t1, value: 20.0, state: .complete)

        // Now we have 2 completed trials at step 1 (10.0 and 20.0). Median threshold = 10.0 (or 15.0).
        // Trial 2: At step 0 reports 5.0 (great score!). At step 1 reports 50.0 (bad current score).
        // Because best-so-far is min(5.0, 50.0) = 5.0, which is better than 10.0, it should NOT prune!
        var t2 = try study.ask()
        try t2.report(5.0, step: 0)
        try t2.report(50.0, step: 1)
        #expect(try !t2.shouldPrune, "Best-so-far score (5.0) is superior to threshold, should not prune")

        // Trial 3: At step 0 reports 30.0. At step 1 reports 35.0.
        // Best-so-far is 30.0, which is worse than the threshold at step 1 -> Prunes!
        var t3 = try study.ask()
        try t3.report(30.0, step: 0)
        try t3.report(35.0, step: 1)
        #expect(try t3.shouldPrune, "Best-so-far score (30.0) is worse than threshold, should prune")
    }

    @Test("SuccessiveHalvingPruner infers minResource automatically when set to .auto")
    func testSuccessiveHalvingAutoResource() throws {
        let pruner = SuccessiveHalvingPruner(minResource: .auto, reductionFactor: 2)

        let study = try Swiftuna.createStudy(
            name: "sha_auto_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Trial 0 runs before any completed trial exists -> cannot prune with .auto
        var t0 = try study.ask()
        try t0.report(100.0, step: 1)
        #expect(try !t0.shouldPrune, "Cannot prune before first completed trial resolves .auto minResource")

        // Complete t0 with max step = 200. Inferred minResource = max(200 / 100, 1) = 2.
        // Rungs with reductionFactor = 2: 2, 4, 8, 16...
        try t0.report(10.0, step: 200)
        try study.tell(consuming: t0, value: 10.0, state: .complete)

        // Trial 1: Step 1 is NOT a rung (step 1 < effectiveMin 2)
        var t1 = try study.ask()
        try t1.report(999.0, step: 1)
        #expect(try !t1.shouldPrune, "Step 1 is below initial rung 2")

        // Step 2 IS a rung! Completed t0 has no value at step 2, so valuesAtRung has only t1 (count 1).
        // 1 trial <= 1 promoted -> survives
        try t1.report(50.0, step: 2)
        #expect(try !t1.shouldPrune)
        try study.tell(consuming: t1, value: 50.0, state: .complete)

        // Now completed trials has t1 with step 2 value = 50.0.
        // Trial 2: At step 2 reports 100.0 (worse than t1's 50.0).
        // valuesAtRung = [50.0]. 1 / 2 promotes 1 trial (50.0). 100.0 > 50.0 -> PRUNES!
        var t2 = try study.ask()
        try t2.report(100.0, step: 2)
        #expect(try t2.shouldPrune, "Trial with worse score at rung 2 should be pruned")
    }

    @Test("HyperbandPruner infers maxResource automatically when set to .auto")
    func testHyperbandAutoResource() throws {
        let pruner = HyperbandPruner(minResource: 1, maxResource: .auto, reductionFactor: 2)

        let study = try Swiftuna.createStudy(
            name: "hb_auto_\(UUID().uuidString)",
            direction: .minimize,
            pruner: pruner
        )

        // Trial 0: Before completion, .auto cannot prune
        var t0 = try study.ask()
        try t0.report(100.0, step: 1)
        #expect(try !t0.shouldPrune)

        // Complete t0 with max step = 7. Inferred maxResource = 8.
        try t0.report(5.0, step: 7)
        try study.tell(consuming: t0, value: 5.0, state: .complete)

        // Subsequent trials now have a resolved bracket ladder
        var t1 = try study.ask()
        try t1.report(2.0, step: 1)
        #expect(try !t1.shouldPrune)
    }

    @Test("PatientPruner correctly detects stagnation in maximization direction")
    func testPatientPrunerMaximize() throws {
        let pruner = PatientPruner(wrappedPruner: nil, patience: 2, minDelta: 0.5)

        let study = try Swiftuna.createStudy(
            name: "patient_max_\(UUID().uuidString)",
            direction: .maximize,
            pruner: pruner
        )

        var t = try study.ask()
        try t.report(10.0, step: 0)
        #expect(try !t.shouldPrune)

        try t.report(10.0, step: 1)
        #expect(try !t.shouldPrune)

        // Worsens from 10.0 to 9.0 (step 2 <= patience + 1)
        try t.report(9.0, step: 2)
        #expect(try !t.shouldPrune)

        // Step 3: scoresBefore = [10.0], scoresAfter = [10.0, 9.0, 9.0], maxAfter = 10.0 >= 10.0 - 0.5 -> not pruned
        try t.report(9.0, step: 3)
        #expect(try !t.shouldPrune)

        // Step 4: scoresBefore = [10.0, 10.0] (max 10.0), scoresAfter = [9.0, 9.0, 9.0] (max 9.0)
        // 10.0 - 0.5 = 9.5 > 9.0 -> PRUNES!
        try t.report(9.0, step: 4)
        #expect(try t.shouldPrune, "Maximization trial that drops below maxBefore - minDelta should be pruned")
    }
}
