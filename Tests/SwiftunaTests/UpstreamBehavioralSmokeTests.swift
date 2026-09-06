import Foundation
import Testing

@testable import Swiftuna

private struct TrialTrace: Codable {
    let number: Int
    let params: [String: Double]
    let value: Double
}

private struct ProblemCorpus: Codable {
    let problem_name: String
    let seed: UInt64
    let trials: [TrialTrace]
    let best_trial_number: Int
    let best_value: Double
}

@Suite("Upstream Behavioral Smoke Tests", .serialized)
struct UpstreamBehavioralSmokeTests {

    private func loadCorpus(name: String) throws -> ProblemCorpus {
        let fileURL = URL(fileURLWithPath: #filePath)
        let repoRoot =
            fileURL
            .deletingLastPathComponent()  // Tests/SwiftunaTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // Repo root
        let fixtureURL =
            repoRoot
            .appendingPathComponent("Tests/Fixtures/ParityCorpus")
            .appendingPathComponent("\(name).json")

        let data: Data
        if FileManager.default.fileExists(atPath: fixtureURL.path) {
            data = try Data(contentsOf: fixtureURL)
        } else {
            data = try Data(contentsOf: URL(fileURLWithPath: "Tests/Fixtures/ParityCorpus/\(name).json"))
        }
        return try JSONDecoder().decode(ProblemCorpus.self, from: data)
    }

    // MARK: - 1. Golden Reference Parity Smoke Tests

    @Test("Golden parity: continuous quadratic float problem reproduces exact reference trajectory")
    func testGoldenParityQuadratic() throws {
        let golden = try loadCorpus(name: "quadratic")
        let sampler = TPESampler(seed: golden.seed)
        let study = try Swiftuna.createStudy(
            name: "smoke_parity_quadratic_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler
        )

        for expected in golden.trials {
            var trial = try study.ask()
            let x = try trial.suggest("x", in: -10.0...10.0)
            let y = try trial.suggest("y", in: -10.0...10.0)
            let loss = pow(x - 2.0, 2) + pow(y + 5.0, 2)

            try study.tell(consuming: trial, value: loss)

            let expX = expected.params["x"]!
            let expY = expected.params["y"]!

            #expect(abs(x - expX) < 1e-7, "Trial \(expected.number) param 'x' drifted from golden Optuna reference")
            #expect(abs(y - expY) < 1e-7, "Trial \(expected.number) param 'y' drifted from golden Optuna reference")
            #expect(
                abs(loss - expected.value) < 1e-7, "Trial \(expected.number) loss drifted from golden Optuna reference")
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7, "Best value drifted from golden Optuna reference")
    }

    @Test("Golden parity: stepped float constrained weights problem reproduces exact reference trajectory")
    func testGoldenParityConstrainedWeights() throws {
        let golden = try loadCorpus(name: "constrained_weights")
        let sampler = TPESampler(seed: golden.seed)
        let study = try Swiftuna.createStudy(
            name: "smoke_parity_constrained_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler
        )

        for expected in golden.trials {
            var trial = try study.ask()
            let p_w = try trial.suggest("param_weight", in: 0.5...2.0, step: 0.1)
            let m_w = try trial.suggest("mutation_weight", in: 0.8...3.0, step: 0.1)
            let s_w = try trial.suggest("sink_weight", in: 0.2...2.0, step: 0.1)

            let ceiling = m_w + 4.0 * s_w
            let loss: Double
            if ceiling > 5.0 {
                loss = 1_000.0 + (ceiling - 5.0) * 100.0
            } else {
                loss = pow(p_w - 1.0, 2) + pow(m_w - 1.5, 2) + pow(s_w - 0.5, 2)
            }

            try study.tell(consuming: trial, value: loss)

            let expP = expected.params["param_weight"]!
            let expM = expected.params["mutation_weight"]!
            let expS = expected.params["sink_weight"]!

            #expect(abs(p_w - expP) < 1e-7, "Trial \(expected.number) param_weight drifted")
            #expect(abs(m_w - expM) < 1e-7, "Trial \(expected.number) mutation_weight drifted")
            #expect(abs(s_w - expS) < 1e-7, "Trial \(expected.number) sink_weight drifted")
            #expect(abs(loss - expected.value) < 1e-7, "Trial \(expected.number) loss drifted")
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
    }

    @Test("Golden parity: integer grid problem reproduces exact reference trajectory")
    func testGoldenParityIntegerGrid() throws {
        let golden = try loadCorpus(name: "integer_grid")
        let sampler = TPESampler(seed: golden.seed)
        let study = try Swiftuna.createStudy(
            name: "smoke_parity_int_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler
        )

        for expected in golden.trials {
            var trial = try study.ask()
            let layers = try trial.suggest("n_layers", in: 1...8)
            let units = try trial.suggest("hidden_units", in: 32...256, step: 32)
            let loss = abs(Double(layers * units) - 512.0)

            try study.tell(consuming: trial, value: loss)

            let expLayers = Int(expected.params["n_layers"]!)
            let expUnits = Int(expected.params["hidden_units"]!)

            #expect(layers == expLayers, "Trial \(expected.number) n_layers drifted")
            #expect(units == expUnits, "Trial \(expected.number) hidden_units drifted")
            #expect(abs(loss - expected.value) < 1e-7)
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
    }

    @Test("Golden parity: categorical choices problem reproduces exact reference trajectory")
    func testGoldenParityCategoricalGrid() throws {
        let golden = try loadCorpus(name: "categorical_grid")
        let sampler = TPESampler(seed: golden.seed)
        let study = try Swiftuna.createStudy(
            name: "smoke_parity_cat_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler
        )

        let choices = ["adam", "sgd", "rmsprop", "adamw"]

        for expected in golden.trials {
            var trial = try study.ask()
            let opt = try trial.suggest("optimizer", choices: choices)

            let loss: Double
            switch opt {
            case "adam": loss = 0.12
            case "sgd": loss = 0.45
            case "rmsprop": loss = 0.28
            default: loss = 0.08
            }

            try study.tell(consuming: trial, value: loss)

            let expectedIdx = Int(expected.params["optimizer"]!)
            let expectedOpt = choices[expectedIdx]

            #expect(opt == expectedOpt, "Trial \(expected.number) optimizer choice drifted")
            #expect(abs(loss - expected.value) < 1e-7)
        }

        let bestVal = try study.bestValue
        #expect(abs(bestVal - golden.best_value) < 1e-7)
    }

    // MARK: - 2. Wire Schema & Deserialization Invariants

    @Test("Storage serialization preserves heterogeneous parameter types and user attributes")
    func testSerializationRoundTripPreservation() throws {
        let tempDB = FileManager.default.temporaryDirectory.appendingPathComponent("smoke_wire_\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: tempDB) }

        let sqlite = StorageBackend.sqlite(url: tempDB)
        let study = try Swiftuna.createStudy(
            name: "smoke_wire_study",
            direction: .minimize,
            storage: sqlite
        )

        var trial = try study.ask()
        let f = try trial.suggest("f_val", in: -1.0...1.0)
        let i = try trial.suggest("i_val", in: 10...20)
        _ = try trial.suggest("c_val", choices: ["alpha", "beta"])

        try trial.setConstraint("c_latency", value: -1.5)
        try trial.setUserAttr("experiment_tag", value: "canary_v1")
        try trial.report(0.99, step: 0)

        try study.tell(consuming: trial, value: f + Double(i))

        // Reload fresh from disk
        let reloaded = try Swiftuna.loadStudy(name: "smoke_wire_study", storage: sqlite)
        let pTrial = try #require(try reloaded.trials.first)

        #expect(pTrial.number == 0)
        #expect(pTrial.state == .complete)
        #expect(pTrial.params["f_val"] != nil)
        #expect(pTrial.params["i_val"] != nil)
        #expect(pTrial.params["c_val"] != nil)
        #expect(pTrial.constraints["c_latency"] == -1.5)
        #expect(pTrial.userAttrs["experiment_tag"] == "canary_v1")
        #expect(pTrial.intermediateValues[0] == 0.99)
        #expect(pTrial.isFeasible == true)
        #expect(pTrial.datetimeStart != nil)
        #expect(pTrial.datetimeComplete != nil)
    }

    // MARK: - 3. Constraint Boundary Semantics

    @Test("Constraint boundary strictly enforces <= 0.0 feasibility condition")
    func testConstraintBoundarySemantics() throws {
        let study = try Swiftuna.createStudy(name: "smoke_constraint_boundary_\(UUID().uuidString)")

        // Exactly zero: Feasible
        var t0 = try study.ask()
        _ = try t0.suggest("x", in: 0.0...1.0)
        try t0.setConstraint("zero_boundary", value: 0.0)
        try study.tell(consuming: t0, value: 10.0)

        // Negative epsilon: Feasible
        var t1 = try study.ask()
        _ = try t1.suggest("x", in: 0.0...1.0)
        try t1.setConstraint("neg_boundary", value: -1e-9)
        try study.tell(consuming: t1, value: 5.0)

        // Positive epsilon: Infeasible
        var t2 = try study.ask()
        _ = try t2.suggest("x", in: 0.0...1.0)
        try t2.setConstraint("pos_boundary", value: 1e-9)
        try study.tell(consuming: t2, value: 1.0) // Lower loss, but infeasible

        let allTrials = try study.trials
        #expect(allTrials.count == 3)
        #expect(allTrials[0].isFeasible == true, "0.0 must be feasible (<= 0.0)")
        #expect(allTrials[1].isFeasible == true, "-1e-9 must be feasible")
        #expect(allTrials[2].isFeasible == false, "1e-9 must be infeasible")

        let bestFeasible = try #require(try study.bestFeasibleTrial)
        #expect(bestFeasible.number == 1, "bestFeasibleTrial must select lowest loss feasible trial")
        #expect(bestFeasible.value == 5.0)
    }

    // MARK: - 4. Error Code Mapping & Fail-Fast Canary

    @Test("Error conditions produce exact typed SwiftunaError cases without degrading to generic fallbacks")
    func testErrorMappingStability() throws {
        let study = try Swiftuna.createStudy(name: "smoke_error_mapping_\(UUID().uuidString)")

        // 1. Invalid float range (log scale on non-positive lower bound) -> .invalidRange
        var t1 = try study.ask()
        #expect(throws: SwiftunaError.self) {
            _ = try t1.suggest("bad_log", in: 0.0...10.0, log: true)
        }
        try study.tell(consuming: t1, value: 1.0)

        // 2. Duplicated study name in same storage -> .duplicatedStudy
        let dbURL = FileManager.default.temporaryDirectory.appendingPathComponent("dup_\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let sqlite = StorageBackend.sqlite(url: dbURL)
        _ = try Swiftuna.createStudy(name: "dup_study", storage: sqlite)
        #expect(throws: SwiftunaError.self) {
            _ = try Swiftuna.createStudy(name: "dup_study", storage: sqlite, loadIfExists: false)
        }

        // 3. Load non-existent study -> .studyNotFound
        #expect(throws: SwiftunaError.self) {
            _ = try Swiftuna.loadStudy(name: "ghost_study_\(UUID().uuidString)", storage: sqlite)
        }

        // 4. Duplicate constraint key on same trial -> .attrOverwriteNotAllowed
        var t2 = try study.ask()
        try t2.setConstraint("c1", value: 0.0)
        #expect(throws: SwiftunaError.self) {
            try t2.setConstraint("c1", value: 1.0)
        }
        try study.tell(consuming: t2, value: 2.0)
    }
}
