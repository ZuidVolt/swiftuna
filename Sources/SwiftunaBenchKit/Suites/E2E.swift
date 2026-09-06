import Dispatch
import Foundation
import Swiftuna

/// End-to-end suites kept from the original bench tool: full TPE and random
/// loops plus trials deserialization. Slower than the hot gates by design —
/// these measure whole-system throughput, not single-path regressions.
public struct E2ESuite: BenchSuite {
    public init() {}

    public var name: String { "e2e" }
    public var description: String { "End-to-end TPE/random loops and trials deserialization throughput" }
    public var requiresNewAPI: Bool { false }
    public var metricNames: [String] { ["e2e_tpe_us", "e2e_random_us", "trials_us"] }

    public func run(_ ctx: SuiteContext) async throws -> SuiteResult {
        let probes: [(String, String, () async throws -> Double)] = [
            ("e2e_tpe_us", "us", { try e2eLoop(nTrials: 200, sampler: TPESampler(seed: 42)) }),
            ("e2e_random_us", "us", { try e2eLoop(nTrials: 200, sampler: RandomSampler(seed: 42)) }),
            ("trials_us", "us", { try trialsDeserUs() }),
        ]
        var metrics: [MetricResult] = []
        for (name, unit, probe) in probes {
            let samples = try await measureRepsAsync(reps: ctx.reps) { try await probe() }
            metrics.append(summarizeMetric(name: name, unit: unit, samples: samples))
        }
        return SuiteResult(suite: name, branch: "", commit: "", metrics: metrics, environment: [:])
    }
}

private func e2eLoop<S: Sampler>(nTrials: Int, sampler: S) throws -> Double {
    let study = try Swiftuna.createStudy(name: "bench_e2e_\(UUID().uuidString)", direction: .minimize, sampler: sampler)
    let start = DispatchTime.now()
    for _ in 0..<nTrials {
        var t = try study.ask()
        let x = try t.suggest("x", in: -10.0...10.0)
        let y = try t.suggest("y", in: -10.0...10.0)
        try study.tell(consuming: t, value: (x - 2) * (x - 2) + (y + 5) * (y + 5))
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(nTrials)
}

private func trialsDeserUs() throws -> Double {
    let study = try Swiftuna.createStudy(name: "bench_query_\(UUID().uuidString)")
    for i in 0..<100 {
        var t = try study.ask()
        _ = try t.suggest("param_a", in: -10.0...10.0)
        _ = try t.suggest("param_b", in: 1...100)
        try t.setUserAttr("tag", value: "epoch_\(i)")
        try t.setConstraint("c1", value: Double(i - 50))
        try t.report(Double(i) * 0.1, step: 0)
        try t.report(Double(i) * 0.05, step: 1)
        try study.tell(consuming: t, value: Double(i) * 1.5)
    }
    _ = try study.trials
    let iters = 1_000
    let start = DispatchTime.now()
    var total = 0
    for _ in 0..<iters { total += try study.trials.count }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(total == 100 * iters)
    return Double(nanos) / 1_000.0 / Double(total)
}
