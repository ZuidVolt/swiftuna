import Foundation
import Testing
@testable import Swiftuna

@Suite("Trial Injection & Seeding Tests")
struct TrialInjectionTests {

    private func makeTempDBURL() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        return tempDir.appendingPathComponent("swiftuna_inject_\(UUID().uuidString).db")
    }

    @Test("Single and batch trial injection, continuous optimization, and objective mismatch validation")
    func testTrialInjectionBatchAndValidation() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let storage = StorageBackend.sqlite(url: dbURL)
        let study = try Swiftuna.createStudy(
            name: "seeded_study",
            direction: .minimize,
            storage: storage
        )

        // 1. Inject single baseline historical trials with user attributes and intermediate values
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

        #expect((try study.trials).count == 2)
        #expect(try study.bestValue == 2.1)
        #expect(try study.bestParams["x"] == 0.5)

        // 2. Inject batch trials via addTrials
        var batch: [PersistedTrial] = []
        for i in 1...3 {
            batch.append(Swiftuna.createTrial(
                state: .complete,
                value: Double(i) * 15.0,
                params: ["x": Double(i), "y": Double(i)],
                userAttrs: ["batch_index": "\(i)"]
            ))
        }
        try study.addTrials(batch)
        #expect((try study.trials).count == 5)

        // 3. Continue optimizing downstream on top of seeded study
        try study.optimize(nTrials: 3) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            return x * x + y * y
        }
        #expect((try study.trials).count == 8)

        // 4. Mismatched objective count injection must throw error
        let multiStudy = try Swiftuna.createStudy(
            name: "multi_obj_\(UUID().uuidString)",
            directions: [.minimize, .maximize]
        )
        let invalidTrial = Swiftuna.createTrial(
            state: .complete,
            value: 42.0, // 1 value for 2 directions
            params: ["p": 1.0]
        )
        #expect(throws: SwiftunaError.self) {
            try multiStudy.addTrial(invalidTrial)
        }
    }
}
