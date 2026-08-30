import Foundation
import Testing

@testable import Swiftuna

private enum MaxLatency: ConstraintKey {
    static let name = "max_latency"
}

private enum MemoryBudget: ConstraintKey {
    static let name = "memory_budget"
}

@Suite("Constrained Optimization & Feasibility Invariant Tests")
struct ConstrainedOptimizationTests {

    @Test("Active trial sets single, batch, and type-safe constraints with accurate read-back")
    func testConstraintSettingAndReadback() throws {
        let study = try Swiftuna.createStudy(
            name: "test_constraints_basic_\(UUID().uuidString)"
        )

        var trial = try study.ask()
        let x = try trial.suggest("x", in: 0.0...10.0)

        // Type-safe protocol constraint
        try trial.setConstraint(MaxLatency.self, value: -2.5)  // Feasible

        // String-keyed constraint
        try trial.setConstraint("power_budget", value: 1.2)  // Infeasible

        #expect(trial.constraints["max_latency"] == -2.5)
        #expect(trial.constraints["power_budget"] == 1.2)

        try study.tell(consuming: trial, value: x * x)

        let persisted = try #require(try study.trials.first)
        #expect(persisted.constraints["max_latency"] == -2.5)
        #expect(persisted.constraints["power_budget"] == 1.2)
        #expect(persisted[MaxLatency.self] == -2.5)
        #expect(!persisted.isFeasible)  // Because power_budget is > 0.0
    }

    @Test("Duplicate constraint key throws attrOverwriteNotAllowed error")
    func testDuplicateConstraintThrows() throws {
        let study = try Swiftuna.createStudy(
            name: "test_constraints_dup_\(UUID().uuidString)"
        )

        var trial = try study.ask()
        try trial.setConstraint("c1", value: 0.0)

        #expect(throws: SwiftunaError.self) {
            try trial.setConstraint("c1", value: 1.0)
        }

        // Clean up trial
        try study.tell(consuming: trial, value: 0.0)
    }

    @Test("NaN constraint value throws invalidArgument error")
    func testNaNConstraintThrows() throws {
        let study = try Swiftuna.createStudy(
            name: "test_constraints_nan_\(UUID().uuidString)"
        )

        var trial = try study.ask()

        #expect(throws: SwiftunaError.self) {
            try trial.setConstraint("bad_nan", value: Double.nan)
        }

        try study.tell(consuming: trial, value: 1.0)
    }

    @Test("Feasibility partitioning and functional pipeline helpers")
    func testFeasibilityPipeline() throws {
        let study = try Swiftuna.createStudy(
            name: "test_feasibility_pipeline_\(UUID().uuidString)"
        )

        // Trial 0: Feasible (objective 50.0)
        var t0 = try study.ask()
        _ = try t0.suggest("x", in: 0.0...10.0)
        try t0.setConstraints(["c1": -1.0, "c2": -0.5])
        try study.tell(consuming: t0, value: 50.0)

        // Trial 1: Infeasible (lower objective 10.0, but violates c2)
        var t1 = try study.ask()
        _ = try t1.suggest("x", in: 0.0...10.0)
        try t1.setConstraints(["c1": -2.0, "c2": 0.5])
        try study.tell(consuming: t1, value: 10.0)

        // Trial 2: Feasible (objective 30.0)
        var t2 = try study.ask()
        _ = try t2.suggest("x", in: 0.0...10.0)
        try t2.setConstraints(["c1": 0.0, "c2": -0.1])
        try study.tell(consuming: t2, value: 30.0)

        let trials = try study.trials
        let feasible = trials.feasible()
        let infeasible = trials.infeasible()

        #expect(feasible.count == 2)
        #expect(infeasible.count == 1)
        #expect(infeasible[0].number == 1)

        // Raw bestTrial gives unconstrained minimum (Trial 1 with 10.0)
        let unconstrainedBest = try #require(try study.bestTrial)
        #expect(unconstrainedBest.number == 1)
        #expect(unconstrainedBest.value == 10.0)

        // bestFeasibleTrial gives best feasible trial (Trial 2 with 30.0)
        let feasibleBest = try #require(try study.bestFeasibleTrial)
        #expect(feasibleBest.number == 2)
        #expect(feasibleBest.value == 30.0)

        // Pipeline helper matches
        let pipelineBest = try #require(trials.bestFeasible(direction: .minimize))
        #expect(pipelineBest.number == 2)
        #expect(pipelineBest.value == 30.0)
    }

    @Test("TPESampler steers optimization into feasible subspace")
    func testTPESamplerSteeringWithConstraints() throws {
        // Objective: minimize f(x) = (x - 2.0)^2 for x in [0.0, 10.0].
        // Constraint: x >= 5.0  <=>  5.0 - x <= 0.0
        // Unconstrained global minimum is at x = 2.0 (objective = 0.0, infeasible).
        // Feasible minimum is at x = 5.0 (objective = (5 - 2)^2 = 9.0).
        let study = try Swiftuna.createStudy(
            name: "test_tpe_steering_\(UUID().uuidString)",
            direction: .minimize,
            sampler: TPESampler(seed: 42)
        )

        try study.optimize(nTrials: 40) { (trial: inout Trial) in
            let x = try trial.suggest("x", in: 0.0...10.0)
            // Constraint: 5.0 - x <= 0.0 (feasible when x >= 5.0)
            try trial.setConstraint("x_ge_5", value: 5.0 - x)
            return (x - 2.0) * (x - 2.0)
        }

        let allTrials = try study.trials
        let feasibleTrials = allTrials.feasible()

        // Feasible trials must exist
        #expect(!feasibleTrials.isEmpty)

        // Best feasible trial should have x >= 5.0 and objective near 9.0
        let bestFeasible = try #require(try study.bestFeasibleTrial)
        let bestX = try #require(bestFeasible.params["x"])
        #expect(bestX >= 4.99)
        #expect(bestFeasible.value! >= 8.9)

        // In the latter part of the study, TPE learns that x < 5 is infeasible
        let lateTrials = Array(allTrials.suffix(15))
        let lateFeasibleCount = lateTrials.filter(\.isFeasible).count
        #expect(lateFeasibleCount >= 9, "TPE should predominantly sample the feasible region once trained")
    }

    @Test("Multi-objective NSGA-II with constraints prioritizes feasible solutions on Pareto frontier")
    func testNSGAIIWithConstraints() throws {
        let study = try Swiftuna.createStudy(
            name: "test_nsgaii_constraints_\(UUID().uuidString)",
            directions: [.minimize, .minimize],
            sampler: NSGAIISampler(populationSize: 30, seed: 123)
        )

        try study.optimize(nTrials: 40) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            let x = try trial.suggest("x", in: 0.0...10.0)
            let y = try trial.suggest("y", in: 0.0...10.0)

            // Constraint: x + y >= 4.0 <=> 4.0 - (x + y) <= 0.0
            try trial.setConstraint("sum_ge_4", value: 4.0 - (x + y))

            let obj1 = (x - 1.0) * (x - 1.0)
            let obj2 = (y - 1.0) * (y - 1.0)
            return [obj1, obj2]
        }

        let paretoTrials = try study.bestTrials
        #expect(!paretoTrials.isEmpty)

        // Any trial on the Pareto frontier when feasible solutions exist should be feasible
        let feasiblePareto = paretoTrials.feasible()
        #expect(!feasiblePareto.isEmpty)
    }
}
