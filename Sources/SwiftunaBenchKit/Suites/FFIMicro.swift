import Dispatch
import Foundation
import Swiftuna
internal import LibRustuna

/// FFI micro-op suite: each layer measured natively (Rust, in-process) and
/// from Swift, so the Swift-minus-native delta isolates exactly one cost —
/// FFI transition, JSON serialization, or Swift glue. Requires the expanded
/// bench FFI, so the compare tool skips this suite on older branches.
public struct FFIMicroSuite: BenchSuite {
    public init() {}

    public var name: String { "ffi" }
    public var description: String { "Swift-vs-native per micro-op: ask/tell, suggest, enqueue, fetch" }
    public var requiresNewAPI: Bool { true }
    public var metricNames: [String] {
        ["ffi_asktell_sw_us", "ffi_asktell_rs_us",
         "ffi_suggest_sw_us_1", "ffi_suggest_rs_us_1",
         "ffi_suggest_sw_us_4", "ffi_suggest_rs_us_4",
         "ffi_suggest_sw_us_8", "ffi_suggest_rs_us_8",
         "ffi_enqueue_sw_us", "ffi_enqueue_rs_us",
         "ffi_fetch_sw_us", "ffi_fetch_rs_us"]
    }

    public func run(_ ctx: SuiteContext) async throws -> SuiteResult {
        var probes: [(String, String, () async throws -> Double)] = [
            ("ffi_asktell_sw_us", "us", { try ffiAskTellSwift(trials: 2000) }),
            ("ffi_asktell_rs_us", "us", { try ffiAskTellNative(trials: 2000) }),
        ]
        for k in [1, 4, 8] {
            probes.append(("ffi_suggest_sw_us_\(k)", "us", { try ffiSuggestSwift(trials: 200, params: k) }))
            probes.append(("ffi_suggest_rs_us_\(k)", "us", { try ffiSuggestNative(trials: 200, params: k) }))
        }
        probes += [
            ("ffi_enqueue_sw_us", "us", { try ffiEnqueueSwift(calls: 1000) }),
            ("ffi_enqueue_rs_us", "us", { try ffiEnqueueNative(calls: 1000) }),
            ("ffi_fetch_sw_us", "us", { try ffiFetchSwift(trials: 1000, repeats: 20) }),
            ("ffi_fetch_rs_us", "us", { try ffiFetchNative(trials: 1000, repeats: 20) }),
        ]
        var metrics: [MetricResult] = []
        for (name, unit, probe) in probes {
            let samples = try await measureRepsAsync(reps: ctx.reps) { try await probe() }
            metrics.append(summarizeMetric(name: name, unit: unit, samples: samples))
        }
        return SuiteResult(suite: name, branch: "", commit: "", metrics: metrics, environment: [:])
    }
}

private func checkBench(_ code: Int32, _ name: String) throws {
    if code != 0 {
        throw BenchError.suiteFailed("\(name) failed with code \(code)")
    }
}

private func ffiAskTellSwift(trials n: Int) throws -> Double {
    let study = try createStudy(name: "ffi_at_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 42))
    let start = DispatchTime.now()
    for i in 0..<n {
        let t = try study.ask()
        try study.tell(consuming: t, value: Double(i) * 0.5)
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(n)
}

private func ffiAskTellNative(trials n: Int) throws -> Double {
    var ns: UInt64 = 0
    try checkBench(rustuna_bench_ask_tell(n, 42, &ns), "rustuna_bench_ask_tell")
    return Double(ns) / 1_000.0
}

private func ffiSuggestSwift(trials n: Int, params k: Int) throws -> Double {
    let study = try createStudy(name: "ffi_sg_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 42))
    let start = DispatchTime.now()
    for _ in 0..<n {
        var t = try study.ask()
        var v = 0.0
        for p in 0..<k {
            let x = try t.suggest("p\(p)", in: -10.0...10.0)
            v += (x - 2.0) * (x - 2.0)
        }
        try study.tell(consuming: t, value: v)
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(n * k)
}

private func ffiSuggestNative(trials n: Int, params k: Int) throws -> Double {
    var ns: UInt64 = 0
    try checkBench(rustuna_bench_suggest(n, k, 42, &ns), "rustuna_bench_suggest")
    return Double(ns) / 1_000.0
}

private func ffiEnqueueSwift(calls n: Int) throws -> Double {
    let study = try createStudy(name: "ffi_eq_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    let start = DispatchTime.now()
    for i in 0..<n {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(n)
}

private func ffiEnqueueNative(calls n: Int) throws -> Double {
    var ns: UInt64 = 0
    try checkBench(rustuna_bench_enqueue(n, &ns), "rustuna_bench_enqueue")
    return Double(ns) / 1_000.0
}

private func ffiFetchSwift(trials n: Int, repeats r: Int) throws -> Double {
    let study = try createStudy(name: "ffi_fe_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    for i in 0..<n {
        let t = try study.ask()
        try study.tell(consuming: t, value: Double(i))
    }
    let start = DispatchTime.now()
    var total = 0
    for _ in 0..<r {
        total += try study.trials.count
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(total == n * r)
    return Double(nanos) / 1_000.0 / Double(total)
}

private func ffiFetchNative(trials n: Int, repeats r: Int) throws -> Double {
    var ns: UInt64 = 0
    try checkBench(rustuna_bench_fetch(n, r, &ns), "rustuna_bench_fetch")
    return Double(ns) / 1_000.0
}
