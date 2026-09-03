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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            let opt = try trial.suggest("optimizer", choices: ["adam", "sgd"])
            return [
                "x": .double(x),
                "optimizer": .string(opt),
            ]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

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
        let completed = try await coordinator.finishedTrialsCount(where: [.complete])
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

        let space = AskFunction { trial in
            let lr = try trial.suggest("lr", in: 1e-4...1e-1)
            return ["lr": .double(lr)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        // Prime the study with 2 successful baseline trials
        for _ in 0..<2 {
            let trial = try await coordinator.ask()
            _ = try await coordinator.report(trialNumber: trial.trialNumber, step: 1, value: 1.0)
            _ = try await coordinator.report(trialNumber: trial.trialNumber, step: 2, value: 0.8)
            try await coordinator.tell(DistributedTrialResult(trialNumber: trial.trialNumber, value: 0.8))
        }

        let completedEarly = try await coordinator.finishedTrialsCount(where: [.complete])
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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let totalTrials = 16
        let workerCount = 4

        // Simulate 4 distributed workers pulling trials concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            for workerId in 0..<workerCount {
                group.addTask {
                    var trialsEvaluated = 0
                    while trialsEvaluated < (totalTrials / workerCount) {
                        let trial = try await coordinator.ask()
                        let x = trial.double("x")!

                        // Simulate remote compute
                        try await Task.sleep(for: .milliseconds(5))

                        let loss = (x - 2.5) * (x - 2.5)
                        try await coordinator.tell(
                            DistributedTrialResult(
                                trialNumber: trial.trialNumber,
                                value: loss,
                                userAttrs: ["worker": "worker-\(workerId)"]
                            ))
                        trialsEvaluated += 1
                    }
                }
            }
            while (try await group.next()) != nil {}
        }

        let completed = try await coordinator.finishedTrialsCount(where: [.complete])
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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let trial = try await coordinator.ask()

        // Wrong values count for a 2-direction study: must throw and retain the trial.
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
            Issue.record("Expected tell with mismatched values count to throw")
        } catch {
            // Expected.
        }
        #expect(try await coordinator.inFlightCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 0)

        // Retry with corrected values succeeds.
        let x = trial.double("x")!
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, values: [x * x, x * x]))
        #expect(try await coordinator.inFlightCount() == 0)
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
    }

    @Test("Completed vs finished counts split pruned and failed trials")
    func testCompletedFinishedSplit() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_counts_test")

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let complete = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: complete.trialNumber, value: 1.0))

        let pruned = try await coordinator.ask()
        try await coordinator.tell(.pruned(trialNumber: pruned.trialNumber))

        let failed = try await coordinator.ask()
        try await coordinator.tell(.failed(trialNumber: failed.trialNumber))

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 1)
        #expect(try await coordinator.finishedTrialsCount() == 3)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("Repeated report for the same step keeps the latest value")
    func testReportLatestWins() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_report_overwrite_test")

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let trial = try await coordinator.ask()
        _ = try await coordinator.report(trialNumber: trial.trialNumber, step: 1, value: 1.0)
        _ = try await coordinator.report(trialNumber: trial.trialNumber, step: 1, value: 2.0)
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 0.5))

        let persisted = try await coordinator.trials().first { $0.number == trial.trialNumber }
        #expect(persisted?.intermediateValues[1] == 2.0)
    }

    @Test("Exhausted grid space surfaces typed searchSpaceExhausted error")
    func testExhaustedGridThrowsTypedError() async throws {
        let system = LocalTestingDistributedActorSystem()
        let sampler = GridSampler(searchSpace: ["a": [0.0, 1.0]])
        let study = try createStudy(name: "dist_exhausted_test", sampler: sampler)

        let space = AskFunction { trial in
            let a = try trial.suggest("a", in: 0.0...1.0)
            return ["a": .double(a)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        for _ in 0..<2 {
            let trial = try await coordinator.ask()
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(
            study: study, askFunction: space, actorSystem: system, maxInFlight: 1)

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
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 2)
    }

    @Test("Read APIs mirror local study state")
    func testReadAPIs() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(
            name: "dist_reads_test", directions: [.minimize, .minimize])

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        for _ in 0..<5 {
            let trial = try await coordinator.ask()
            let x = trial.double("x")!
            try await coordinator.tell(
                DistributedTrialResult(
                    trialNumber: trial.trialNumber, values: [x * x, (x - 1.0) * (x - 1.0)]))
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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        for _ in 0..<10 {
            let trial = try await coordinator.ask()
            let x = trial.double("x")!
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: x * x))
        }

        let importances = try await coordinator.paramImportances(normalize: true, params: nil)
        #expect(importances["x"] != nil)
    }

    @Test("Expired lease is reaped as failed on next ask")
    func testLeaseExpiryReapsAsFailed() async throws {
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_lease",
            leasePolicy: LeasePolicy(timeout: .seconds(0.05))
        ) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let trial = try await coordinator.ask()
        try await Task.sleep(for: .milliseconds(150))

        // Next ask triggers the reap: slot freed, trial recorded as failed.
        _ = try await coordinator.ask()
        #expect(try await coordinator.inFlightCount() == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 0)
        #expect(try await coordinator.trials(where: [.fail]).count == 1)

        // Stale tell for the expired trial reports the lease, not a missing trial.
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
            Issue.record("Expected leaseExpired error")
        } catch SwiftunaDistributedError.leaseExpired(let num) {
            #expect(num == trial.trialNumber)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Report heartbeats keep a lease alive")
    func testHeartbeatExtendsLease() async throws {
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_heartbeat",
            leasePolicy: LeasePolicy(timeout: .seconds(0.5))
        ) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let trial = try await coordinator.ask()
        for step in 1...3 {
            try await Task.sleep(for: .milliseconds(100))
            _ = try await coordinator.report(trialNumber: trial.trialNumber, step: step, value: 1.0)
        }
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 1)
        #expect(try await coordinator.finishedTrialsCount() == 1)
    }

    @Test("Duplicate tell reports trialAlreadyFinished")
    func testDuplicateTell() async throws {
        let (coordinator, _) = try makeTestCoordinator(named: "dist_dupe") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let trial = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))

        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: 2.0))
            Issue.record("Expected trialAlreadyFinished error")
        } catch SwiftunaDistributedError.trialAlreadyFinished(let num) {
            #expect(num == trial.trialNumber)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        do {
            _ = try await coordinator.report(trialNumber: trial.trialNumber, step: 1, value: 1.0)
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

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let trial = try await coordinator.ask()

        do {
            try await coordinator.tell(
                DistributedTrialResult(
                    trialNumber: trial.trialNumber,
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
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 1)
    }

    @Test("Typed key-pair payloads round-trip through storage")
    func testTypedKeyPairPayloads() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_typed_test")

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        let trial = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(
                trialNumber: trial.trialNumber,
                value: 0.25,
                constraintPairs: [(ConstraintKey("dist_limit"), -1.5)],
                userAttrPairs: [(AttributeKey.distRegion.name, "eu")]))

        let persisted = try await coordinator.trials().first { $0.number == trial.trialNumber }
        #expect(persisted?.constraints["dist_limit"] == -1.5)
        #expect(persisted?[AttributeKey.distRegion] == "eu")
    }

    @Test("Context checkout loop matches manual ask-tell state")
    func testContextLoop() async throws {
        let system = LocalTestingDistributedActorSystem()
        let study = try createStudy(name: "dist_context_test")

        let space = AskFunction { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

        for _ in 0..<3 {
            let ctx = try await DistributedTrialContext.checkout(from: coordinator)
            let x = ctx.trial.double("x")!
            _ = try await ctx.report(step: 1, value: x * x)
            try await ctx.tell(
                value: x * x,
                constraintPairs: [(ConstraintKey("ctx_limit"), -1.0)],
                userAttrPairs: [(AttributeKey.distRegion.name, "local")])
        }

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 3)
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
        let space = AskFunction(params: dsl) { trial in
            let x = try trial.suggest("x", in: 0.0...1.0)
            // Conditional dim the DSL cannot express alone.
            let layers = x > 0.5 ? 4 : 2
            return ["x": .double(42.0), "layers": .int(layers)]
        }

        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_hatch",
            sample: { trial in
                try space.sample(trial: &trial)
            })

        let trial = try await coordinator.ask()
        #expect(trial.double("x") == 42.0)  // procedural entry wins on collision
        #expect(trial.int("layers") == 2 || trial.int("layers") == 4)
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
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

    @Test("optimize() runs trials across workers in one call")
    func testDrive() async throws {
        let (coordinator, _) = try makeTestCoordinator(named: "dist_drive") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        try await optimize(coordinator: coordinator, nTrials: 20, workers: 4) { ctx in
            let x = ctx.trial.double("x")!
            return x * x
        }

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 20)
        #expect(try await coordinator.inFlightCount() == 0)
        let best = try await coordinator.bestTrial()
        #expect(best?.value != nil)
    }

    @Test("optimize() records failed trial and rethrows objective errors")
    func testDriveThrowAborts() async throws {
        struct Boom: Error {}
        let (coordinator, _) = try makeTestCoordinator(named: "dist_drive_throw") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        do {
            try await optimize(coordinator: coordinator, nTrials: 10, workers: 1) { ctx in
                if ctx.trialNumber == 2 { throw Boom() }
                return 1.0
            }
            Issue.record("Expected Boom to propagate")
        } catch is Boom {
            // Expected.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 2)
        #expect(try await coordinator.trials(where: [.fail]).count == 1)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("optimize() skips completion tell when objective settles the trial")
    func testDriveSelfSettled() async throws {        let (coordinator, _) = try makeTestCoordinator(named: "dist_drive_self") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        try await optimize(coordinator: coordinator, nTrials: 4, workers: 1) { ctx in
            let x = ctx.trial.double("x")!
            if x > 0.0 {
                try await ctx.prune()
                return 0.0  // ignored: trial already settled
            }
            return x * x
        }

        #expect(try await coordinator.finishedTrialsCount() == 4)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("ask(waitingUpTo:) returns when a slot frees mid-wait")
    func testBlockingAskWakesOnTell() async throws {
        let (coordinator, _) = try makeTestCoordinator(named: "dist_wait", maxInFlight: 1) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        let first = try await coordinator.ask()
        _ = Task {
            try await Task.sleep(for: .milliseconds(50))
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: first.trialNumber, value: 1.0))
        }

        let start = ContinuousClock.now
        let second = try await coordinator.ask(waitingUpTo: .seconds(5))
        let waited = ContinuousClock.now - start
        #expect(waited >= .milliseconds(30)) // proof it waited, not spuriously passed
        #expect(second.trialNumber != first.trialNumber)
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: second.trialNumber, value: 0.5))
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 2)
    }

    @Test("ask(waitingUpTo:) throws tooManyInFlight after the timeout")
    func testBlockingAskTimesOut() async throws {
        let (coordinator, _) = try makeTestCoordinator(named: "dist_wait_timeout", maxInFlight: 1) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        _ = try await coordinator.ask() // slot stays full: nobody tells
        do {
            _ = try await coordinator.ask(waitingUpTo: .milliseconds(100))
            Issue.record("Expected tooManyInFlight error")
        } catch SwiftunaDistributedError.tooManyInFlight(let count) {
            #expect(count == 1)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("ask(waitingUpTo:) surfaces exhaustion at once, not after the timeout")
    func testBlockingAskExhaustionFast() async throws {
        let grid: [String: GridSampler.ValueList] = ["a": [0.0]]
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_wait_exhausted",
            sampler: GridSampler(searchSpace: grid)
        ) { trial in
            let a = try trial.suggest("a", in: 0.0...1.0)
            return ["a": .double(a)]
        }

        let trial = try await coordinator.ask()
        try await coordinator.tell(
            DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))

        let start = ContinuousClock.now
        do {
            _ = try await coordinator.ask(waitingUpTo: .seconds(30))
            Issue.record("Expected searchSpaceExhausted error")
        } catch SwiftunaDistributedError.searchSpaceExhausted {
            #expect(ContinuousClock.now - start < .seconds(5))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("ParamReadable reads identically on both carriers")
    func testParamReadableParity() {
        enum Optimizer: String {
            case adam, sgd
        }
        let params: [String: ParameterValue] = [
            "lr": .double(0.01),
            "layers": .int(4),
            "opt": .string("adam"),
            "fp16": .bool(true),
        ]
        let persisted = PersistedTrial(
            number: 0, state: .complete, value: 1.0,
            values: [1.0], params: params)
        let trial = DistributedTrial(trialNumber: 0, params: params)

        let carriers: [any ParamReadable] = [persisted, trial]
        for carrier in carriers {
            #expect(carrier.double("lr") == 0.01)
            #expect(carrier.float("lr") == 0.01)
            #expect(carrier.int("layers") == 4)
            #expect(carrier.string("opt") == "adam")
            #expect(carrier.bool("fp16") == true)
            #expect(carrier.param("opt", as: Optimizer.self) == .adam)
            #expect(carrier.double("missing") == nil)
        }
    }

    @Test("Local and distributed trials reject the same NaN constraint")
    func testValidationParity() async throws {
        let study = try createStudy(name: "dist_parity_local")
        var trial = try study.ask()
        do {
            try trial.setConstraint("bad", value: .nan)
            Issue.record("Expected invalidArgument error")
        } catch SwiftunaError.invalidArgument {
            // Expected.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        try study.tell(consuming: trial, value: 1.0)

        let (coordinator, _) = try makeTestCoordinator(named: "dist_parity_remote") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }
        let remote = try await coordinator.ask()
        do {
            try await coordinator.tell(
                DistributedTrialResult(
                    trialNumber: remote.trialNumber, value: 1.0,
                    constraints: ["bad": .nan]))
            Issue.record("Expected invalidConstraint error")
        } catch SwiftunaDistributedError.invalidConstraint {
            // Expected: same input, distributed spelling of the same rule.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("DSL samples every distribution type through the shared path")
    func testDSLAllDistributions() async throws {
        let dsl = SearchSpaceParams([
            .float(name: "lr", lower: 1e-4, upper: 1e-1, log: true),
            .float(name: "dropout", lower: 0.0, upper: 0.5, step: 0.05),
            .int(name: "layers", lower: 1, upper: 8, step: 2),
            .categorical(name: "opt", choices: ["adam", "sgd"]),
        ])
        let (coordinator, _) = try makeTestCoordinator(named: "dist_dsl_all") { [dsl] trial in
            try dsl.sample(trial: &trial)
        }

        for _ in 0..<5 {
            let trial = try await coordinator.ask()
            let lr = trial.double("lr")!
            #expect(lr >= 1e-4 && lr <= 1e-1)
            #expect(trial.int("layers")! >= 1 && trial.int("layers")! <= 8)
            #expect(trial.string("opt") == "adam" || trial.string("opt") == "sgd")
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: trial.trialNumber, value: 1.0))
        }
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 5)
    }
}

extension AttributeKey where Value == String {
    static let distRegion = AttributeKey<String>("dist_region")
}

@Suite("Worker Loop Tests")
struct WorkerLoopTests {
    @Test("runWorker with fixed count matches drive state")
    func testRunWorkerFixedCount() async throws {
        let (coordinator, _) = try makeTestCoordinator(named: "dist_worker_fixed") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        try await runWorker(coordinator: coordinator, nTrials: 5) { ctx in
            let x = ctx.trial.double("x")!
            return x * x
        }

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 5)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("runWorker without count stops at exhaustion")
    func testRunWorkerUntilExhausted() async throws {
        let grid: [String: GridSampler.ValueList] = ["a": [0.0, 1.0]]
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_worker_open",
            sampler: GridSampler(searchSpace: grid)
        ) { trial in
            let a = try trial.suggest("a", in: 0.0...1.0)
            return ["a": .double(a)]
        }

        try await runWorker(coordinator: coordinator) { _ in 1.0 }

        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 2)
        #expect(try await coordinator.inFlightCount() == 0)
    }

    @Test("isRetryable is true only for capacity errors")
    func testIsRetryable() {
        #expect(SwiftunaDistributedError.tooManyInFlight(3).isRetryable)
        #expect(!SwiftunaDistributedError.trialNotFound(0).isRetryable)
        #expect(!SwiftunaDistributedError.trialAlreadyFinished(0).isRetryable)
        #expect(!SwiftunaDistributedError.leaseExpired(0).isRetryable)
        #expect(!SwiftunaDistributedError.searchSpaceExhausted.isRetryable)
        #expect(!SwiftunaDistributedError.invalidConstraint("x").isRetryable)
        #expect(!SwiftunaDistributedError.studyError("x").isRetryable)
    }

    @Test("Multi-objective optimize and runWorker record vectors")
    func testMultiObjectiveDriver() async throws {
        let (coordinator, _) = try makeTestCoordinator(
            named: "dist_multi_driver",
            directions: [.minimize, .maximize]
        ) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }

        try await optimize(coordinator: coordinator, nTrials: 6, workers: 2) { ctx -> [Double] in
            let x = ctx.trial.double("x")!
            return [x * x, -x * x]
        }
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 6)

        try await runWorker(coordinator: coordinator, nTrials: 2) { ctx -> [Double] in
            let x = ctx.trial.double("x")!
            return [x * x, -x * x]
        }
        #expect(try await coordinator.finishedTrialsCount(where: [.complete]) == 8)
        #expect(try await coordinator.bestTrials().count > 0)
    }
}
