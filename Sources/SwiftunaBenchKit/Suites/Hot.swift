import Dispatch
import Distributed
import Foundation
import Swiftuna
import SwiftunaDistributed

/// Push-gate suite: B5 round-trip, B12 disabled path, B16 enqueue tax,
/// ask-only probe. Ported verbatim from the scratch HotSubset blocks —
/// same loops, same counts — so historical numbers stay comparable.
/// Expected runtime ~30s at 5 reps.
public struct HotSuite: BenchSuite {
    public init() {}

    public var name: String { "hot" }
    public var description: String { "Push gates: coordinator round-trip, disabled path, enqueue tax, bare ask" }
    public var requiresNewAPI: Bool { false }
    public var metricNames: [String] {
        ["b5_total_ms", "b12_flag_ns", "b12_disabled_us",
         "b16_first500_us", "b16_enqueue_us", "b16_second500_us", "b16_append_ns",
         "ask_ns"]
    }

    public func run(_ ctx: SuiteContext) async throws -> SuiteResult {
        // (metric, unit, probe). Sync probes convert to async implicitly.
        let probes: [(String, String, () async throws -> Double)] = [
            ("b5_total_ms", "ms", { try await hotB5TotalMs() }),
            ("b12_flag_ns", "ns", { try hotB12FlagNs() }),
            ("b12_disabled_us", "us", { try await hotB12DisabledUs() }),
            ("b16_first500_us", "us", { try hotB16First500Us() }),
            ("b16_enqueue_us", "us", { try hotB16EnqueueUs() }),
            ("b16_second500_us", "us", { try hotB16Second500Us() }),
            ("b16_append_ns", "ns", { hotB16AppendNs() }),
            ("ask_ns", "ns", { try hotAskNs() }),
        ]
        var metrics: [MetricResult] = []
        for (name, unit, probe) in probes {
            let samples = try await measureRepsAsync(reps: ctx.reps) { try await probe() }
            metrics.append(summarizeMetric(name: name, unit: unit, samples: samples))
        }
        return SuiteResult(suite: name, branch: "", commit: "", metrics: metrics, environment: [:])
    }
}

private func hotB5TotalMs() async throws -> Double {
    let study = try createStudy(direction: .minimize)
    let space = AskFunction { trial in
        let x = try trial.suggest("x", in: -10.0...10.0)
        return ["x": .double(x)]
    }
    let system = LocalTestingDistributedActorSystem()
    let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)
    let start = DispatchTime.now()
    for _ in 0..<1000 {
        let trial = try await coordinator.ask()
        let x = trial.double("x")!
        let loss = (x - 2.0) * (x - 2.0)
        try await coordinator.tell(DistributedTrialResult(trialNumber: trial.trialNumber, value: loss))
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    let completed = try await coordinator.finishedTrialsCount(where: [.complete])
    precondition(completed == 1000)
    return Double(nanos) / 1_000_000.0
}

private func hotB12FlagNs() throws -> Double {
    precondition(!SwiftunaTelemetry.shared.isEnabled)
    var sinks = 0
    let start = DispatchTime.now()
    for _ in 0..<10_000_000 {
        if SwiftunaTelemetry.shared.isEnabled { sinks += 1 }
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(sinks == 0)
    return Double(nanos) / 10_000_000.0
}

private func hotB12DisabledUs() async throws -> Double {
    let study = try createStudy(name: "bench_tele_\(UUID().uuidString)", direction: .minimize)
    let space = AskFunction { trial in
        let x = try trial.suggest("x", in: -5.0...5.0)
        return ["x": .double(x)]
    }
    let system = LocalTestingDistributedActorSystem()
    let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)
    let start = DispatchTime.now()
    for _ in 0..<1000 {
        let ctx = try await DistributedTrialContext.checkout(from: coordinator)
        let x = ctx.trial.double("x")!
        _ = try await ctx.report(step: 1, value: x * x)
        try await ctx.tell(value: x * x)
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / 1000.0
}

private func hotEnqueueBatch(count: Int) throws -> Double {
    let study = try createStudy(name: "b16_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    let start = DispatchTime.now()
    for i in 0..<count {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(count)
}

private func hotB16First500Us() throws -> Double { try hotEnqueueBatch(count: 500) }
private func hotB16EnqueueUs() throws -> Double { try hotEnqueueBatch(count: 1000) }

private func hotB16Second500Us() throws -> Double {
    let study = try createStudy(name: "b16b_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    for i in 0..<500 {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    let start = DispatchTime.now()
    for i in 0..<500 {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / 500.0
}

private func hotB16AppendNs() -> Double {
    var arr: [PersistedTrial] = []
    arr.reserveCapacity(100_000)
    let start = DispatchTime.now()
    for i in 0..<100_000 {
        arr.append(PersistedTrial(number: i, state: .complete, value: Double(i), params: ["x": .double(Double(i))]))
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(arr.count == 100_000)
    return Double(nanos) / 100_000.0
}

private func hotAskNs() throws -> Double {
    // 10k asks per rep: the loop is milliseconds either way, and the wider
    // sample keeps per-rep means out of scheduling noise.
    let depth = 10_000
    let study = try createStudy(name: "askprobe_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    for i in 0..<depth {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<depth {
        // Untold trials leak their Rust boxes until process exit; accepted
        // for a probe, freed with the study.
        let t = try study.ask()
        sink += t.number
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(sink == depth * (depth - 1) / 2)
    return Double(nanos) / Double(depth)
}
