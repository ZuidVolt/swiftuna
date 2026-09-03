import Distributed
import Foundation
import Testing

@testable import Swiftuna
@testable import SwiftunaDistributed

@Suite("Swiftuna Distributed Actor System Tests")
struct SwiftunaDistributedTests {

    @Test("Basic Ask-and-Tell lifecycle over LocalTestingDistributedActorSystem")
    func testBasicDistributedAskAndTell() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_test_1")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            let opt = try trial.suggest("optimizer", choices: ["adam", "sgd"])
            return [
                "x": .double(x),
                "optimizer": .string(opt),
            ]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        // 1. Worker asks for trial
        let trialSpec = try await coordinator.ask()
        #expect(trialSpec.trialNumber == 0)
        #expect(trialSpec.double("x") != nil)
        #expect(trialSpec.double("x")! >= -10.0 && trialSpec.double("x")! <= 10.0)
        #expect(trialSpec.string("optimizer") == "adam" || trialSpec.string("optimizer") == "sgd")

        let inFlight = try await coordinator.inFlightCount()
        #expect(inFlight == 1)

        // 2. Worker computes loss and tells result
        let xVal = trialSpec.double("x")!
        let loss = (xVal - 3.0) * (xVal - 3.0)
        let result = DistributedTrialResult(
            trialNumber: trialSpec.trialNumber,
            value: loss,
            constraints: ["limit": -1.5],
            userAttrs: ["worker_id": "mac-studio-01"]
        )

        try await coordinator.tell(result)

        // 3. Verify state
        let completed = try await coordinator.completedTrialsCount()
        #expect(completed == 1)
        let inFlightAfter = try await coordinator.inFlightCount()
        #expect(inFlightAfter == 0)

        let best = try await coordinator.bestTrial()
        #expect(best != nil)
        #expect(best?.number == 0)
        #expect(best?.value == loss)
        #expect(best?.constraints["limit"] == -1.5)
        #expect(best?.userAttrs["worker_id"] == "mac-studio-01")
    }

    @Test("Distributed intermediate step reporting and early stopping pruning")
    func testDistributedEarlyStoppingPruning() async throws {
        let system = LocalTestingDistributedActorSystem()
        let pruner = MedianPruner(nStartupTrials: 2, nWarmupSteps: 1)
        let study = try createStudy(name: "dist_pruning_test", pruner: pruner)

        let space = SearchSpace { trial in
            let lr = try trial.suggest("lr", in: 1e-4...1e-1)
            return ["lr": .double(lr)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        // Prime the study with 2 successful baseline trials
        for _ in 0..<2 {
            let spec = try await coordinator.ask()
            _ = try await coordinator.report(trialNumber: spec.trialNumber, step: 1, value: 1.0)
            _ = try await coordinator.report(trialNumber: spec.trialNumber, step: 2, value: 0.8)
            try await coordinator.tell(DistributedTrialResult(trialNumber: spec.trialNumber, value: 0.8))
        }

        let completedEarly = try await coordinator.completedTrialsCount()
        #expect(completedEarly == 2)

        // Third trial reports a terribly diverging validation loss
        let badTrial = try await coordinator.ask()
        let shouldStopStep1 = try await coordinator.report(trialNumber: badTrial.trialNumber, step: 1, value: 1.0)
        #expect(shouldStopStep1 == false)  // warmup step

        let shouldStopStep2 = try await coordinator.report(trialNumber: badTrial.trialNumber, step: 2, value: 50.0)
        #expect(shouldStopStep2 == true)  // Pruned by MedianPruner!

        // Worker respects pruning and tells .pruned state
        try await coordinator.tell(DistributedTrialResult.pruned(trialNumber: badTrial.trialNumber))

        let inFlightAfterPruning = try await coordinator.inFlightCount()
        #expect(inFlightAfterPruning == 0)
        let allTrials = try await coordinator.trials()
        #expect(allTrials.count == 3)
        #expect(allTrials.last?.state == .pruned)
    }

    @Test("Concurrent multi-worker cluster simulation with Swift TaskGroup")
    func testConcurrentMultiWorkerCluster() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_cluster_sim")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let totalTrials = 16
        let workerCount = 4

        // Simulate 4 distributed workers pulling trials concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            for workerId in 0..<workerCount {
                group.addTask {
                    var trialsEvaluated = 0
                    while trialsEvaluated < (totalTrials / workerCount) {
                        let spec = try await coordinator.ask()
                        let x = spec.double("x")!

                        // Simulate remote compute
                        try await Task.sleep(for: .milliseconds(5))

                        let loss = (x - 2.5) * (x - 2.5)
                        try await coordinator.tell(
                            DistributedTrialResult(
                                trialNumber: spec.trialNumber,
                                value: loss,
                                userAttrs: ["worker": "worker-\(workerId)"]
                            ))
                        trialsEvaluated += 1
                    }
                }
            }
            while (try await group.next()) != nil {}
        }

        let completed = try await coordinator.completedTrialsCount()
        #expect(completed == totalTrials)
        let inFlightEnd = try await coordinator.inFlightCount()
        #expect(inFlightEnd == 0)

        let best = try await coordinator.bestTrial()
        #expect(best != nil)
        #expect(best!.value! < 25.0)
    }

    @Test("Failed tell retains in-flight trial for retry")
    func testFailedTellIsRetryable() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_retry_test", directions: [.minimize, .minimize])

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let spec = try await coordinator.ask()

        // Wrong values count for a 2-direction study: must throw and retain the trial.
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
            Issue.record("Expected tell with mismatched values count to throw")
        } catch {
            // Expected.
        }
        #expect(try await coordinator.inFlightCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 0)

        // Retry with corrected values succeeds.
        let x = spec.double("x")!
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, values: [x * x, x * x]))
        #expect(try await coordinator.inFlightCount() == 0)
        #expect(try await coordinator.completedTrialsCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
    }

    @Test("Completed vs finished counts split pruned and failed trials")
    func testCompletedFinishedSplit() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_counts_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let complete = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: complete.trialNumber, value: 1.0))

        let pruned = try await coordinator.ask()
        try await coordinator.tell(.pruned(trialNumber: pruned.trialNumber))

        let failed = try await coordinator.ask()
        try await coordinator.tell(.failed(trialNumber: failed.trialNumber))

        #expect(try await coordinator.completedTrialsCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 3)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("Repeated report for the same step keeps the latest value")
    func testReportLatestWins() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_report_overwrite_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let spec = try await coordinator.ask()
        _ = try await coordinator.report(trialNumber: spec.trialNumber, step: 1, value: 1.0)
        _ = try await coordinator.report(trialNumber: spec.trialNumber, step: 1, value: 2.0)
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, value: 0.5))

        let persisted = try await coordinator.trials().first { $0.number == spec.trialNumber }
        #expect(persisted?.intermediateValues[1] == 2.0)
    }

    @Test("Exhausted grid space surfaces typed searchSpaceExhausted error")
    func testExhaustedGridThrowsTypedError() async throws {
        let system = LocalTestingDistributedActorSystem()
        let sampler = GridSampler(searchSpace: ["a": [0.0, 1.0]])
        let study = try createStudy(name: "dist_exhausted_test", sampler: sampler)

        let space = SearchSpace { trial in
            let a = try trial.suggest("a", in: 0.0...1.0)
            return ["a": .double(a)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        for _ in 0..<2 {
            let spec = try await coordinator.ask()
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
        }

        do {
            _ = try await coordinator.ask()
            Issue.record("Expected searchSpaceExhausted error")
        } catch SwiftunaDistributedError.searchSpaceExhausted {
            // Expected: typed, matchable, no string parsing.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("maxInFlight cap rejects excess asks until a slot frees")
    func testMaxInFlightCap() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_cap_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(
            study: study, searchSpace: space, actorSystem: system, maxInFlight: 1)

        let first = try await coordinator.ask()

        do {
            _ = try await coordinator.ask()
            Issue.record("Expected tooManyInFlight error")
        } catch SwiftunaDistributedError.tooManyInFlight(let count) {
            #expect(count == 1)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        try await coordinator.tell(
            DistributedTrialResult(trialNumber: first.trialNumber, value: 1.0))

        // Slot freed: asking works again.
        let second = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: second.trialNumber, value: 0.5))
        #expect(try await coordinator.completedTrialsCount() == 2)
    }

    @Test("Read APIs mirror local study state")
    func testReadAPIs() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(
            name: "dist_reads_test", directions: [.minimize, .minimize])

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        for _ in 0..<5 {
            let spec = try await coordinator.ask()
            let x = spec.double("x")!
            try await coordinator.tell(
                DistributedTrialResult(
                    trialNumber: spec.trialNumber, values: [x * x, (x - 1.0) * (x - 1.0)]))
        }
        let pruned = try await coordinator.ask()
        try await coordinator.tell(.pruned(trialNumber: pruned.trialNumber))

        #expect(try await coordinator.trials().count == 6)
        #expect(try await coordinator.trials(where: [.complete]).count == 5)
        #expect(try await coordinator.trials(where: [.pruned]).count == 1)

        let pareto = try await coordinator.bestTrials()
        let localPareto = try study.bestTrials
        #expect(pareto.count == localPareto.count)
        #expect(Set(pareto.map(\.number)) == Set(localPareto.map(\.number)))

        try await coordinator.setUserAttr("region", value: "eu")
        #expect(try await coordinator.userAttr("region") == "eu")
        #expect(try await coordinator.userAttr("missing") == nil)
    }

    @Test("paramImportances scores sampled parameters")
    func testParamImportances() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_importance_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        for _ in 0..<10 {
            let spec = try await coordinator.ask()
            let x = spec.double("x")!
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: spec.trialNumber, value: x * x))
        }

        let importances = try await coordinator.paramImportances(normalize: true, params: nil)
        #expect(importances["x"] != nil)
    }

    @Test("Expired lease is reaped as failed on next ask")
    func testLeaseExpiryReapsAsFailed() async throws {
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_lease",
            leasePolicy: LeasePolicy(timeoutSeconds: 0.05)
        ) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let spec = try await coordinator.ask()
        try await Task.sleep(for: .milliseconds(150))

        // Next ask triggers the reap: slot freed, trial recorded as failed.
        _ = try await coordinator.ask()
        #expect(try await coordinator.inFlightCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
        #expect(try await coordinator.completedTrialsCount() == 0)
        #expect(try await coordinator.trials(where: [.fail]).count == 1)

        // Stale tell for the expired trial reports the lease, not a missing trial.
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
            Issue.record("Expected leaseExpired error")
        } catch SwiftunaDistributedError.leaseExpired(let num) {
            #expect(num == spec.trialNumber)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Report heartbeats keep a lease alive")
    func testHeartbeatExtendsLease() async throws {
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_heartbeat",
            leasePolicy: LeasePolicy(timeoutSeconds: 0.5)
        ) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let spec = try await coordinator.ask()
        for step in 1...3 {
            try await Task.sleep(for: .milliseconds(100))
            _ = try await coordinator.report(trialNumber: spec.trialNumber, step: step, value: 1.0)
        }
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
        #expect(try await coordinator.completedTrialsCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
    }

    @Test("Duplicate tell reports trialAlreadyFinished")
    func testDuplicateTell() async throws {        let (coordinator, _) = try makeTestCoordinator(named: "dist_dupe") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let spec = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))

        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: spec.trialNumber, value: 2.0))
            Issue.record("Expected trialAlreadyFinished error")
        } catch SwiftunaDistributedError.trialAlreadyFinished(let num) {
            #expect(num == spec.trialNumber)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        do {
            _ = try await coordinator.report(trialNumber: spec.trialNumber, step: 1, value: 1.0)
            Issue.record("Expected trialAlreadyFinished error")
        } catch SwiftunaDistributedError.trialAlreadyFinished {
            // Expected.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        // Unknown numbers still report trialNotFound.
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: 999_999, value: 1.0))
            Issue.record("Expected trialNotFound error")
        } catch SwiftunaDistributedError.trialNotFound {
            // Expected.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("NaN constraint throws invalidConstraint and trial stays retryable")
    func testNaNConstraintRejected() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_nan_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let spec = try await coordinator.ask()

        do {
            try await coordinator.tell(
                DistributedTrialResult(
                    trialNumber: spec.trialNumber,
                    value: 1.0,
                    constraints: ["bad": .nan]))
            Issue.record("Expected invalidConstraint error")
        } catch SwiftunaDistributedError.invalidConstraint {
            // Expected: mirrors Trial.setConstraint NaN rejection.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(try await coordinator.inFlightCount() == 1)

        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
        #expect(try await coordinator.completedTrialsCount() == 1)
    }

    @Test("Typed key-pair payloads round-trip through storage")
    func testTypedKeyPairPayloads() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_typed_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        let spec = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(
                trialNumber: spec.trialNumber,
                value: 0.25,
                constraintPairs: [(ConstraintKey("dist_limit"), -1.5)],
                userAttrPairs: [(AttributeKey.distRegion.name, "eu")]))

        let persisted = try await coordinator.trials().first { $0.number == spec.trialNumber }
        #expect(persisted?.constraints["dist_limit"] == -1.5)
        #expect(persisted?[AttributeKey.distRegion] == "eu")
    }

    @Test("Context checkout loop matches manual ask-tell state")
    func testContextLoop() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_context_test")

        let space = SearchSpace { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)

        for _ in 0..<3 {
            let ctx = try await DistributedTrialContext.checkout(from: coordinator)
            let x = ctx.spec.double("x")!
            _ = try await ctx.report(step: 1, value: x * x)
            try await ctx.tell(
                value: x * x,
                constraintPairs: [(ConstraintKey("ctx_limit"), -1.0)],
                userAttrPairs: [(AttributeKey.distRegion.name, "local")])
        }

        #expect(try await coordinator.completedTrialsCount() == 3)
        #expect(try await coordinator.finishedTrialsCount() == 3)
        #expect(try await coordinator.inFlightCount() == 0)
        let best = try await coordinator.bestTrial()
        #expect(best?.constraints["ctx_limit"] == -1.0)
        #expect(best?[AttributeKey.distRegion] == "local")
    }

    @Test("DSL space matches closure space on a fixed grid")
    func testDSLEquivalence() async throws {
        let grid: [String: GridSampler.ValueList] = ["a": [0.0, 1.0], "b": [10.0, 20.0]]
        let dsl = SearchSpaceParams([
            .float(name: "a", lower: 0.0, upper: 1.0),
            .float(name: "b", lower: 10.0, upper: 20.0),
        ])

        let (dslCoordinator, _) = try makeTestCoordinator(
            named: "dist_dsl", sampler: GridSampler(searchSpace: grid, seed: 7),
            sample: { [dsl] trial in try dsl.sample(trial: &trial) }
        )
        let (closureCoordinator, _) = try makeTestCoordinator(
            named: "dist_closure", sampler: GridSampler(searchSpace: grid, seed: 7)
        ) { trial in
            let a = try trial.suggest("a", in: 0.0...1.0)
            let b = try trial.suggest("b", in: 10.0...20.0)
            return ["a": .double(a), "b": .double(b)]
        }

        for _ in 0..<4 {
            let dslSpec = try await dslCoordinator.ask()
            let closureSpec = try await closureCoordinator.ask()
            #expect(dslSpec.params == closureSpec.params)
            try await dslCoordinator.tell(
                DistributedTrialResult(trialNumber: dslSpec.trialNumber, value: 1.0))
            try await closureCoordinator.tell(
                DistributedTrialResult(trialNumber: closureSpec.trialNumber, value: 1.0))
        }
    }

    @Test("Custom hatch overrides declarative params and fills gaps")
    func testCustomHatch() async throws {
        let dsl = SearchSpaceParams([.float(name: "x", lower: 0.0, upper: 1.0)])
        let space = SearchSpace(params: dsl) { trial in
            let x = try trial.suggest("x", in: 0.0...1.0)
            // Conditional dim the DSL cannot express alone.
            let layers = x > 0.5 ? 4 : 2
            return ["x": .double(42.0), "layers": .int(layers)]
        }

        let (coordinator, _) = try makeTestCoordinator(named: "dist_hatch", sample: { trial in
            try space.sample(trial: &trial)
        })

        let spec = try await coordinator.ask()
        #expect(spec.double("x") == 42.0) // procedural entry wins on collision
        #expect(spec.int("layers") == 2 || spec.int("layers") == 4)
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: spec.trialNumber, value: 1.0))
    }

    @Test("SearchSpaceParams round-trips through Codable and rejects duplicates")
    func testDSLCodableAndDuplicates() async throws {
        let dsl = SearchSpaceParams([
            .float(name: "lr", lower: 1e-4, upper: 1e-1, log: true),
            .int(name: "layers", lower: 1, upper: 8),
            .categorical(name: "opt", choices: ["adam", "sgd"]),
        ])
        let data = try JSONEncoder().encode(dsl)
        let decoded = try JSONDecoder().decode(SearchSpaceParams.self, from: data)
        #expect(decoded == dsl)

        let dupe = SearchSpaceParams([
            .float(name: "x", lower: 0.0, upper: 1.0),
            .int(name: "x", lower: 0, upper: 2),
        ])
        let (coordinator, _) = try makeTestCoordinator(named: "dist_dupe") { [dupe] trial in
            try dupe.sample(trial: &trial)
        }
        do {
            _ = try await coordinator.ask()
            Issue.record("Expected invalidArgument for duplicate names")
        } catch SwiftunaDistributedError.studyError {
            // Expected: wrapped invalidArgument from the DSL validation.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

extension AttributeKey where Value == String {
    static let distRegion = AttributeKey<String>("dist_region")
}
