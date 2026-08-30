import Foundation
import Testing
import Swiftuna

@Suite("GridSampler & Cartesian Exhaustion Tests", .serialized)
struct GridSamplerTests {

    @Test("GridSampler precomputes Cartesian product and optimize finishes early on exhaustion")
    func testGridSamplerCartesianExhaustion() throws {
        // 2 x 3 = 6 combinations
        let sampler = GridSampler(searchSpace: [
            "x": [1.0, 2.0],
            "y": [10.0, 20.0, 30.0]
        ])

        let study = try Swiftuna.createStudy(
            name: "grid_exhaust_\(UUID().uuidString)",
            sampler: sampler
        )

        // Request 15 trials, but only 6 exist in the grid
        try study.optimize(nTrials: 15) { trial in
            let x = try trial.suggest("x", in: 1.0...2.0)
            let y = try trial.suggest("y", in: 10.0...30.0)
            return x + y
        }

        let trials = try study.trials
        let completed = trials.completed()
        #expect(completed.count == 6)

        // Check that all 6 combinations are unique
        var observedPairs: Set<String> = []
        for t in completed {
            let xVal = t.params["x"]!
            let yVal = t.params["y"]!
            observedPairs.insert("\(xVal),\(yVal)")
        }
        #expect(observedPairs.count == 6)
        #expect(observedPairs.contains("1.0,10.0"))
        #expect(observedPairs.contains("1.0,20.0"))
        #expect(observedPairs.contains("1.0,30.0"))
        #expect(observedPairs.contains("2.0,10.0"))
        #expect(observedPairs.contains("2.0,20.0"))
        #expect(observedPairs.contains("2.0,30.0"))
    }

    @Test("Explicit study.ask() throws searchSpaceExhausted when grid combinations are depleted")
    func testGridSamplerExplicitAskThrowsWhenExhausted() throws {
        // 2 x 2 = 4 combinations
        let sampler = GridSampler(searchSpace: [
            "a": [0.0, 1.0],
            "b": [10.0, 20.0]
        ])

        let study = try Swiftuna.createStudy(
            name: "grid_ask_throw_\(UUID().uuidString)",
            sampler: sampler
        )

        for _ in 0..<4 {
            var trial = try study.ask()
            _ = try trial.suggest("a", in: 0.0...1.0)
            _ = try trial.suggest("b", in: 10.0...20.0)
            try study.tell(consuming: trial, value: 1.0)
        }

        // 5th ask must throw searchSpaceExhausted
        #expect(throws: SwiftunaError.self) {
            _ = try study.ask()
        }

        do {
            _ = try study.ask()
            Issue.record("Expected searchSpaceExhausted error")
        } catch SwiftunaError.searchSpaceExhausted {
            // Expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("GridSampler with identical seed produces identical shuffled evaluation order")
    func testGridSamplerSeedDeterminism() throws {
        let space: [String: GridSampler.ValueList] = [
            "dim1": [1.0, 2.0, 3.0],
            "dim2": [10.0, 20.0, 30.0]
        ]

        let study1 = try Swiftuna.createStudy(
            name: "grid_seed_1_\(UUID().uuidString)",
            sampler: GridSampler(searchSpace: space, seed: 999)
        )
        let study2 = try Swiftuna.createStudy(
            name: "grid_seed_2_\(UUID().uuidString)",
            sampler: GridSampler(searchSpace: space, seed: 999)
        )

        try study1.optimize(nTrials: 9) { trial in
            let d1 = try trial.suggest("dim1", in: 1.0...3.0)
            let d2 = try trial.suggest("dim2", in: 10.0...30.0)
            return d1 + d2
        }

        try study2.optimize(nTrials: 9) { trial in
            let d1 = try trial.suggest("dim1", in: 1.0...3.0)
            let d2 = try trial.suggest("dim2", in: 10.0...30.0)
            return d1 + d2
        }

        let trials1 = try study1.trials
        let trials2 = try study2.trials

        #expect(trials1.count == 9)
        #expect(trials2.count == 9)

        for i in 0..<9 {
            #expect(trials1[i].params["dim1"] == trials2[i].params["dim1"])
            #expect(trials1[i].params["dim2"] == trials2[i].params["dim2"])
        }
    }

    @Test("GridSampler ValueList expressible by array literals with mixed float, int, and categorical")
    func testGridSamplerArrayLiteralErgonomics() throws {
        let sampler = GridSampler(searchSpace: [
            "float_param": [0.1, 0.5],
            "int_param": .init([1, 2]),
            "cat_param": .init(categorical: ["optA", "optB"])
        ])

        let study = try Swiftuna.createStudy(
            name: "grid_types_\(UUID().uuidString)",
            sampler: sampler
        )

        try study.optimize(nTrials: 8) { trial in
            let f = try trial.suggest("float_param", in: 0.1...0.5)
            let i = try trial.suggest("int_param", in: 1...2)
            let c = try trial.suggest("cat_param", choices: ["optA", "optB"])
            return f + Double(i) + (c == "optA" ? 0.0 : 1.0)
        }

        let trials = try study.trials
        #expect(trials.count == 8) // 2 * 2 * 2 = 8
    }
}
