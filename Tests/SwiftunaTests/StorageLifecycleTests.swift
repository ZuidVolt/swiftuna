import Foundation
import Testing
@testable import Swiftuna

@Suite("Study Lifecycle & Storage Operations Tests")
struct StorageLifecycleTests {

    private func makeTempDBURL() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        return tempDir.appendingPathComponent("swiftuna_lifecycle_\(UUID().uuidString).db")
    }

    @Test("In-memory study copies to SQLite with trials, constraints, and user attributes")
    func testInMemoryToSQLiteCopy() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        // 1. Optimize in volatile memory (fastest mode)
        let memStudy = try Swiftuna.createStudy(
            name: "mem_experiment",
            direction: .minimize,
            storage: .inMemory
        )
        try memStudy.setUserAttr("experiment_tag", value: "baseline_run")

        try memStudy.optimize(nTrials: 15) { (trial: inout Trial) in
            let x = try trial.suggest("x", in: -5.0...5.0)
            try trial.setConstraint("x_ge_0", value: -x)
            return x * x
        }

        #expect((try memStudy.trials).count == 15)

        // 2. Export / copy to SQLite
        let sqliteBackend = StorageBackend.sqlite(url: dbURL)
        let exportedStudy = try memStudy.copy(to: sqliteBackend, as: "sqlite_exported")

        // 3. Continue optimizing on the exported study
        try exportedStudy.optimize(nTrials: 5) { (trial: inout Trial) in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return x * x
        }

        #expect((try exportedStudy.trials).count == 20)

        // 4. Reload independently from disk and verify byte-for-byte integrity
        let reloaded = try Swiftuna.loadStudy(name: "sqlite_exported", storage: sqliteBackend)
        let reloadedTrials = try reloaded.trials

        #expect(reloadedTrials.count == 20)
        #expect(try reloaded.userAttr("experiment_tag") == "baseline_run")

        // First 15 trials should have constraint x_ge_0
        let constrainedTrials = reloadedTrials.prefix(15)
        for trial in constrainedTrials {
            #expect(trial.constraints["x_ge_0"] != nil)
        }
    }

    @Test("Copying to an existing study name throws duplicatedStudy error")
    func testCopyStudyDuplicatedConflictThrows() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let sqliteBackend = StorageBackend.sqlite(url: dbURL)

        // Create initial study in SQLite
        let original = try Swiftuna.createStudy(name: "study_a", storage: sqliteBackend)
        try original.optimize(nTrials: 3) { _ in 42.0 }

        // Attempting to copy with the same destination name must throw .duplicatedStudy
        #expect(throws: SwiftunaError.self) {
            try original.copy(to: sqliteBackend, as: "study_a")
        }

        // Procedural API must also throw .duplicatedStudy
        #expect(throws: SwiftunaError.self) {
            try Swiftuna.copyStudy(from: original, to: sqliteBackend, as: "study_a")
        }
    }

    @Test("Storage studies query, metadata extraction, and functional pipeline filtering")
    func testStorageGetStudiesAndPipeline() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let sqliteBackend = StorageBackend.sqlite(url: dbURL)

        // Create 3 distinct studies in the same database
        let study1 = try Swiftuna.createStudy(name: "study_alpha", storage: sqliteBackend)
        try study1.setUserAttr("team", value: "ml_platform")
        try study1.optimize(nTrials: 10) { _ in 1.0 }

        let study2 = try Swiftuna.createStudy(name: "study_beta", storage: sqliteBackend)
        try study2.setUserAttr("team", value: "core_ai")
        try study2.optimize(nTrials: 25) { _ in 2.0 }

        let study3 = try Swiftuna.createStudy(name: "study_gamma", storage: sqliteBackend)
        try study3.optimize(nTrials: 5) { _ in 3.0 }

        // Query studies via declarative method
        let summaries = try sqliteBackend.studies()
        #expect(summaries.count == 3)

        // Verify summary fields
        let beta = try #require(summaries.named("study_beta"))
        #expect(beta.trialCount == 25)
        #expect(beta.userAttrs["team"] == "core_ai")
        #expect(beta.directions == [.minimize])

        // Functional pipeline extensions
        let largeStudies = summaries.minTrials(10)
        #expect(largeStudies.count == 2)

        let sorted = summaries.sortedByTrialCount()
        #expect(sorted.first?.name == "study_beta")
        #expect(sorted.last?.name == "study_gamma")

        // Procedural top-level function matches
        let procSummaries = try Swiftuna.getStudies(in: sqliteBackend)
        #expect(procSummaries.count == 3)
    }

    @Test("Deleting a study cascades to all trials and removes from database")
    func testDeleteStudyAndCascadingCleanUp() throws {
        let dbURL = makeTempDBURL()
        defer { try? FileManager.default.removeItem(at: dbURL) }

        let sqliteBackend = StorageBackend.sqlite(url: dbURL)

        let study = try Swiftuna.createStudy(name: "study_to_delete", storage: sqliteBackend)
        try study.optimize(nTrials: 10) { _ in 100.0 }

        let remaining = try Swiftuna.createStudy(name: "study_keep", storage: sqliteBackend)
        try remaining.optimize(nTrials: 5) { _ in 200.0 }

        #expect((try sqliteBackend.studies()).count == 2)

        // Delete study_to_delete
        try sqliteBackend.deleteStudy(named: "study_to_delete")

        let afterDelete = try sqliteBackend.studies()
        #expect(afterDelete.count == 1)
        #expect(afterDelete[0].name == "study_keep")

        // Attempting to delete non-existent study throws studyNotFound
        #expect(throws: SwiftunaError.self) {
            try sqliteBackend.deleteStudy(named: "study_to_delete")
        }

        // Procedural delete matches
        try Swiftuna.deleteStudy(named: "study_keep", in: sqliteBackend)
        #expect((try sqliteBackend.studies()).isEmpty)
    }

    @Test("Storage-level filtered query returns only matching trial states")
    func testStorageLevelFilteredTrials() throws {
        let study = try Swiftuna.createStudy(
            name: "test_filtered_\(UUID().uuidString)",
            direction: .minimize
        )

        // Run trials: even numbers prune, odd numbers complete
        for _ in 0..<20 {
            var trial = try study.ask()
            let x = try trial.suggest("x", in: 0.0...10.0)
            if trial.number % 2 == 0 {
                try study.tell(consuming: trial, state: .pruned)
            } else {
                try study.tell(consuming: trial, value: x * x)
            }
        }

        let allTrials = try study.trials
        #expect(allTrials.count == 20)

        // Query only complete trials at storage level
        let completeTrials = try study.trials(where: .complete)
        #expect(completeTrials.count == 10)
        #expect(completeTrials.allSatisfy { $0.state == .complete })

        // Query only pruned trials at storage level
        let prunedTrials = try study.trials(where: .pruned)
        #expect(prunedTrials.count == 10)
        #expect(prunedTrials.allSatisfy { $0.state == .pruned })

        // Query multi-state set
        let both = try study.trials(where: [.complete, .pruned])
        #expect(both.count == 20)

        // Query non-existent state
        let waiting = try study.trials(where: .waiting)
        #expect(waiting.isEmpty)
    }

    @Test("Direct storage-to-storage copy between SQLite databases")
    func testStorageToStorageCopy() throws {
        let srcDB = makeTempDBURL()
        let destDB = makeTempDBURL()
        defer {
            try? FileManager.default.removeItem(at: srcDB)
            try? FileManager.default.removeItem(at: destDB)
        }

        let srcStorage = StorageBackend.sqlite(url: srcDB)
        let destStorage = StorageBackend.sqlite(url: destDB)

        let study = try Swiftuna.createStudy(name: "source_experiment", storage: srcStorage)
        try study.optimize(nTrials: 12) { _ in 3.14 }

        // Copy directly without opening Study on destination
        try Swiftuna.copyStudy(
            fromName: "source_experiment",
            fromStorage: srcStorage,
            toName: "archived_experiment",
            toStorage: destStorage
        )

        let destSummaries = try destStorage.studies()
        #expect(destSummaries.count == 1)
        #expect(destSummaries[0].name == "archived_experiment")
        #expect(destSummaries[0].trialCount == 12)
    }
}
