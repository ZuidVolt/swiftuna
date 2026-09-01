import Foundation
import Testing
@testable import Swiftuna

@Suite("Storage, Concurrency & Multi-Objective Tests")
struct StorageConcurrencyMultiObjectiveTests {

    // MARK: - Persistent Storage Tests

    @Test("SQLite persistent storage saves trials and reloads identically across process sessions")
    func testSQLiteStoragePersistenceAndReload() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbFile = tempDir.appendingPathComponent("swiftuna_test_\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: dbFile) }

        let originalBestVal: Double
        do {
            let study = try Swiftuna.createStudy(
                name: "calibration_sqlite",
                storage: .sqlite(url: dbFile),
                sampler: TPESampler(seed: 42)
            )

            try study.optimize(nTrials: 12) { (trial: inout Trial) throws(SwiftunaError) -> Double in
                let x = try trial.suggest("x", in: -10.0...10.0)
                return (x - 3.5) * (x - 3.5)
            }

            let trials = try study.trials
            #expect(trials.count == 12)
            originalBestVal = try study.bestValue
        }

        // Simulate new process session: reload study from disk
        let reloadedStudy = try Swiftuna.loadStudy(
            name: "calibration_sqlite",
            storage: .sqlite(url: dbFile),
            sampler: TPESampler(seed: 42)
        )

        let reloadedTrials = try reloadedStudy.trials
        #expect(reloadedTrials.count == 12)

        let reloadedBestVal = try reloadedStudy.bestValue
        #expect(reloadedBestVal == originalBestVal)

        let bestTrial = try #require(try reloadedStudy.bestTrial)
        #expect(bestTrial.params["x"] != nil)
    }

    @Test("createStudy with loadIfExists: true resumes existing SQLite study without error")
    func testSQLiteLoadIfExists() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbFile = tempDir.appendingPathComponent("swiftuna_resume_\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: dbFile) }

        // Initial creation
        let study1 = try Swiftuna.createStudy(
            name: "resume_test",
            storage: .sqlite(url: dbFile)
        )
        try study1.optimize(nTrials: 5) { trial in 1.0 }
        #expect(try study1.trials.count == 5)

        // Resuming with loadIfExists: true
        let study2 = try Swiftuna.createStudy(
            name: "resume_test",
            storage: .sqlite(url: dbFile),
            loadIfExists: true
        )
        #expect(try study2.trials.count == 5)

        // Add 5 more trials
        try study2.optimize(nTrials: 5) { trial in 2.0 }
        #expect(try study2.trials.count == 10)
    }

    @Test("Loading a non-existent study name throws studyNotFound error")
    func testLoadNonExistentStudyThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbFile = tempDir.appendingPathComponent("swiftuna_missing_\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: dbFile) }

        // Create empty db with a dummy study
        let _ = try Swiftuna.createStudy(name: "existing", storage: .sqlite(url: dbFile))

        // Attempting to load non-existent study
        do {
            _ = try Swiftuna.loadStudy(name: "ghost_study", storage: .sqlite(url: dbFile))
            Issue.record("Expected loadStudy to fail with studyNotFound")
        } catch {
            switch error {
            case .studyNotFound:
                #expect(true)
            default:
                Issue.record("Expected studyNotFound but got: \(error)")
            }
        }
    }

    @Test("Journal lockless storage persists append-only logs and reloads correctly")
    func testJournalStoragePersistenceAndReload() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let logFile = tempDir.appendingPathComponent("swiftuna_journal_\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: logFile) }

        let originalBestVal: Double
        do {
            let study = try Swiftuna.createStudy(
                name: "journal_study",
                storage: .journal(url: logFile),
                sampler: TPESampler(seed: 99)
            )

            try study.optimize(nTrials: 8) { (trial: inout Trial) throws(SwiftunaError) -> Double in
                let p = try trial.suggest("p", in: 0.0...10.0)
                return p * p
            }
            originalBestVal = try study.bestValue
            #expect(try study.trials.count == 8)
        }

        // Reload from journal log
        let reloaded = try Swiftuna.loadStudy(
            name: "journal_study",
            storage: .journal(url: logFile),
            sampler: TPESampler(seed: 99)
        )
        #expect(try reloaded.trials.count == 8)
        #expect(try reloaded.bestValue == originalBestVal)
    }

    // MARK: - Concurrent Optimization Tests

    @Test("Structured concurrent optimization with Swift 6.4 TaskGroups")
    func testConcurrentTaskGroupOptimization() async throws {
        let study = try Swiftuna.createStudy(
            name: "concurrency_test",
            sampler: TPESampler(seed: 123)
        )

        // Concurrently run 24 trials across 4 worker tasks
        try await study.optimize(nTrials: 24, concurrency: 4) { (trial: inout Trial) async throws(SwiftunaError) -> Double in
            let w = try trial.suggest("worker_weight", in: -5.0...5.0)
            // Simulate light asynchronous compute workload
            try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
            return w * w
        }

        let allTrials = try study.trials
        #expect(allTrials.count == 24)

        let completed = allTrials.completed()
        #expect(completed.count == 24)

        // Verify trial numbers are contiguous 0..<24
        let numbers = Set(allTrials.map(\.number))
        #expect(numbers.count == 24)
        for i in 0..<24 {
            #expect(numbers.contains(i))
        }

        #expect(try study.bestValue >= 0.0)
    }

    @Test("Concurrent optimization properly isolates trial pruning")
    func testConcurrentOptimizationWithPruning() async throws {
        let study = try Swiftuna.createStudy(
            name: "concurrent_pruning_test",
            sampler: TPESampler(seed: 42)
        )

        try await study.optimize(nTrials: 20, concurrency: 4) { (trial: inout Trial) async throws(SwiftunaError) -> Double in
            let v = try trial.suggest("v", in: 0.0...10.0)
            if v > 6.0 {
                throw SwiftunaError.trialPruned(reason: "SMT violation in worker")
            }
            return v * v
        }

        let allTrials = try study.trials
        #expect(allTrials.count == 20)

        let prunedCount = allTrials.pruned().count
        let completedCount = allTrials.completed().count
        #expect(prunedCount + completedCount == 20)
        #expect(prunedCount > 0)
    }

    // MARK: - Multi-Objective NSGA-II & Pareto Front Tests

    @Test("Multi-objective NSGA-II study yields non-dominated Pareto frontier")
    func testMultiObjectiveParetoFrontier() throws {
        let study = try Swiftuna.createStudy(
            name: "pareto_study",
            directions: [.minimize, .maximize],
            sampler: NSGAIISampler(populationSize: 20, seed: 123)
        )

        #expect(study.directions.count == 2)
        #expect(study.directions[0] == .minimize)
        #expect(study.directions[1] == .maximize)

        // Calling bestTrial on multi-objective study must throw unsupportedMultiObjective
        do {
            _ = try study.bestTrial
            Issue.record("Expected unsupportedMultiObjective error for bestTrial on multi-objective study")
        } catch {
            #expect(error == .unsupportedMultiObjective)
        }

        // Optimize conflicting objectives: f1 = x^2 (minimize), f2 = (x - 2)^2 (maximize)
        try study.optimize(nTrials: 25) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let f1 = x * x
            let f2 = (x - 2.0) * (x - 2.0)
            return [f1, f2]
        }

        let trials = try study.trials
        #expect(trials.count == 25)

        for trial in trials {
            #expect(trial.values.count == 2)
        }

        // Query Pareto frontier
        let paretoTrials = try study.bestTrials
        #expect(!paretoTrials.isEmpty)
        for pt in paretoTrials {
            #expect(pt.values.count == 2)
            #expect(pt.params["x"] != nil)
        }
    }

    // MARK: - Concurrent Error Handling & Thread Isolation Tests

    @Test("Concurrent tasks taking errors isolate error slots with zero cross-talk")
    func testConcurrentErrorTakeAndIsolation() async throws {
        let taskCount = 16

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<taskCount {
                group.addTask {
                    let study = try! Swiftuna.createStudy(direction: .minimize)
                    var trial = try! study.ask()

                    // Trigger invalid log scale range in Rust: low <= 0 with log scale
                    do {
                        _ = try trial.suggest("x_\(i)", in: 0.0...Double(i + 1), log: true)
                        Issue.record("Expected invalidRange error")
                    } catch let error as SwiftunaError {
                        switch error {
                        case .invalidRange(let msg):
                            #expect(msg.contains("log scale requires low"))
                        default:
                            Issue.record("Expected invalidRange error, got \(error)")
                        }
                    } catch {
                        Issue.record("Expected SwiftunaError, got \(error)")
                    }

                    // Verify that subsequent valid operation on this task succeeds cleanly
                    do {
                        let validVal = try trial.suggest("valid_param", in: 0.0...10.0)
                        #expect(validVal >= 0.0 && validVal <= 10.0)
                        try study.tell(consuming: trial, value: validVal)
                        let best = try study.bestValue
                        #expect(best == validVal)
                    } catch {
                        Issue.record("Subsequent operation failed: \(error)")
                    }
                }
            }
        }
    }
}
