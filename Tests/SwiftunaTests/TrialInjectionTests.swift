import Foundation
import Testing
@testable import Swiftuna

@Suite("Trial Injection & Seeding Tests")
struct TrialInjectionTests {

    private func makeTempDBURL() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        return tempDir.appendingPathComponent("swiftuna_inject_\(UUID().uuidString).db")
    }

    @Test("Injecting historical completed trials primes the study and updates bestTrial")
    func testAddTrialSingle() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let storage = StorageBackend.sqlite(url: dbURL)
        let study = try Swiftuna.createStudy(
            name: "seeded_study",
            direction: .minimize,
            storage: storage
        )

        // Inject 2 baseline historical trials
        let trial1 = Swiftuna.createTrial(
            state: .complete,
            value: 10.5,
            params: ["x": 2.0, "y": 3.0],
            userAttrs: ["source": "historical_sweep_v1"],
            intermediateValues: [0: 20.0, 1: 10.5]
        )

        let trial2 = Swiftuna.createTrial(
            state: .complete,
            value: 2.1,
            params: ["x": 0.5, "y": 1.0],
            userAttrs: ["source": "expert_prior"],
            intermediateValues: [0: 5.0, 1: 2.1]
        )

        try study.addTrial(trial1)
        try study.addTrial(trial2)

        let trials = try study.trials
        #expect(trials.count == 2)
        #expect(try study.bestValue == 2.1)
        #expect(try study.bestParams["x"] == 0.5)

        // Continue optimizing with TPE on top of the seeded study
        try study.optimize(nTrials: 5) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return x * x + y * y
        }

        let totalTrials = try study.trials
        #expect(totalTrials.count == 7)
    }

    @Test("Batch trial injection via addTrials")
    func testAddTrialsBatch() throws {
        let study = try Swiftuna.createStudy(name: "batch_seeded_\(UUID().uuidString)")

        var seeded: [PersistedTrial] = []
        for i in 1...5 {
            seeded.append(Swiftuna.createTrial(
                state: .complete,
                value: Double(i) * 10.0,
                params: ["alpha": Double(i)],
                userAttrs: ["batch_index": "\(i)"]
            ))
        }

        try study.addTrials(seeded)

        let allTrials = try study.trials
        #expect(allTrials.count == 5)
        #expect(try study.bestValue == 10.0)
    }

    @Test("Adding trial with mismatched objective count throws invalidArgument error")
    func testAddTrialMismatchedObjectivesThrows() throws {
        let study = try Swiftuna.createStudy(
            name: "multi_obj_\(UUID().uuidString)",
            directions: [.minimize, .maximize]
        )

        // Single objective trial cannot be added to a 2-objective study
        let invalidTrial = Swiftuna.createTrial(
            state: .complete,
            value: 42.0, // 1 value
            params: ["p": 1.0]
        )

        #expect(throws: SwiftunaError.self) {
            try study.addTrial(invalidTrial)
        }
    }
}
