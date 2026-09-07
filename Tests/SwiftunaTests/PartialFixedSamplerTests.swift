import Foundation
import Testing
@testable import Swiftuna

@Suite("PartialFixedSampler Tests")
struct PartialFixedSamplerTests {
    @Test("PartialFixed with Rust TPESampler fixes specified parameter while free parameter varies")
    func testPartialFixedWithRustTPE() throws {
        let base = TPESampler(seed: 42)
        let partial = PartialFixedSampler(
            fixedParams: ["fixed_x": .double(1.5)],
            baseSampler: base
        )

        let study = try Swiftuna.createStudy(
            name: "partial_fixed_tpe_test",
            direction: .minimize,
            sampler: partial
        )

        try study.optimize(nTrials: 20) { trial in
            let x = try trial.suggest("fixed_x", in: -10.0...10.0)
            let y = try trial.suggest("free_y", in: -10.0...10.0)
            #expect(x == 1.5)
            return (x - 1.5) * (x - 1.5) + y * y
        }

        let trials = try study.trials
        #expect(trials.count == 20)
        for t in trials {
            #expect(t.params["fixed_x"]?.asDouble == 1.5)
            #expect(t.params["free_y"] != nil)
        }

        // Verify free_y varied across trials
        let yValues = Set(trials.compactMap { $0.params["free_y"]?.asDouble })
        #expect(yValues.count > 5)
    }

    @Test("PartialFixed with Swift CMASampler preserves fixed dimension while optimizing active dimensions")
    func testPartialFixedWithCMASampler() throws {
        let cma = CMASampler(
            dimensions: [
                .continuous(name: "x", lower: -5.0, upper: 5.0),
                .continuous(name: "y", lower: -5.0, upper: 5.0),
            ],
            seed: 42
        )
        let partial = PartialFixedSampler(
            fixedParams: ["z": .double(0.0)],
            baseSampler: cma
        )

        let study = try Swiftuna.createStudy(
            name: "partial_fixed_cma_test",
            direction: .minimize,
            sampler: partial
        )

        try study.optimize(nTrials: 30) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            let y = try trial.suggest("y", in: -5.0...5.0)
            let z = try trial.suggest("z", in: -5.0...5.0)
            #expect(z == 0.0)
            let dx = x - 1.0
            let dy = y - 2.0
            return dx * dx + dy * dy + z * z
        }

        let trials = try study.trials
        #expect(trials.count == 30)
        for t in trials {
            #expect(t.params["z"]?.asDouble == 0.0)
        }

        guard let best = try study.bestTrial, let val = best.value else {
            Issue.record("Missing best trial or value")
            return
        }
        #expect(val < 1.0)
    }

    @Test("PartialFixed preserves multiple parameter types (float, int, string, bool)")
    func testPartialFixedMultipleTypes() throws {
        let partial = PartialFixedSampler(
            fixedParams: [
                "lr": .double(0.001),
                "batch_size": .int(64),
                "optimizer": .string("adam"),
                "use_gpu": .bool(true),
            ],
            baseSampler: RandomSampler(seed: 123)
        )

        let study = try Swiftuna.createStudy(
            name: "partial_fixed_types_test",
            direction: .minimize,
            sampler: partial
        )

        try study.optimize(nTrials: 10) { trial in
            let lr = try trial.suggest("lr", in: 1e-5...1e-1)
            let batch = try trial.suggest("batch_size", in: 16...128)
            let opt = try trial.suggest("optimizer", choices: ["adam", "sgd", "rmsprop"])
            let useGpu = try trial.suggest("use_gpu", choices: [false, true])
            let freeVal = try trial.suggest("dropout", in: 0.1...0.5)

            #expect(lr == 0.001)
            #expect(batch == 64)
            #expect(opt == "adam")
            #expect(useGpu == true)
            return freeVal
        }

        let trials = try study.trials
        #expect(trials.count == 10)
        for t in trials {
            #expect(t.params["lr"]?.asDouble == 0.001)
            #expect(t.params["batch_size"]?.asInt == 64)
            #expect(t.params["optimizer"]?.asString == "adam")
            #expect(t.params["use_gpu"]?.asBool == true)
        }
    }

    @Test("PartialFixed with multi-objective study")
    func testPartialFixedMultiObjective() throws {
        let partial = PartialFixedSampler(
            fixedParams: ["mode": .string("fast")],
            baseSampler: NSGAIISampler(populationSize: 20, seed: 42)
        )

        let study = try Swiftuna.createStudy(
            name: "partial_fixed_mo_test",
            directions: [.minimize, .maximize],
            sampler: partial
        )

        try study.optimize(nTrials: 15) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            let mode = try trial.suggest("mode", choices: ["fast", "accurate"])
            let x = try trial.suggest("x", in: 0.0...10.0)
            #expect(mode == "fast")
            return [x * x, 10.0 - x]
        }

        let trials = try study.trials
        #expect(trials.count == 15)
        for t in trials {
            #expect(t.params["mode"]?.asString == "fast")
        }
    }
}
