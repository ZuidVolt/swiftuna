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
        let allTrials = try study.trials
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
            while let _ = try await group.next() {}
        }

        let completed = try await coordinator.completedTrialsCount()
        #expect(completed == totalTrials)
        let inFlightEnd = try await coordinator.inFlightCount()
        #expect(inFlightEnd == 0)

        let best = try await coordinator.bestTrial()
        #expect(best != nil)
        #expect(best!.value! < 25.0)
    }
}
