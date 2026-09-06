#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Dispatch
import Foundation
import Swiftuna

/// Scaling suite: per-unit cost vs N for every custom-sampler code path.
/// Uses old APIs only, so it builds and runs on any branch. Slow by nature
/// (A1 at n=5000, A3 history builds) — expect several minutes at 5 reps.
/// Flat per-unit cost across N is the pass criterion; see the gate command.
public struct ScaleSuite: BenchSuite {
    public init() {}

    public var name: String { "scale" }
    public var description: String { "Scaling shapes: driver best-scan vs cache, fetch, ask depth, merge, TLS" }
    public var requiresNewAPI: Bool { false }
    public var metricNames: [String] {
        ["a1_scan_us_200", "a1_scan_us_1000", "a1_scan_us_5000",
         "a1_cached_us_200", "a1_cached_us_1000", "a1_cached_us_5000",
         "a2_scan_ns_100", "a2_scan_ns_1000", "a2_scan_ns_10000",
         "a2_copy_ns_100", "a2_copy_ns_1000", "a2_copy_ns_10000",
         "a3_full_us_500", "a3_full_us_2000", "a3_full_us_5000",
         "a4_ask_ns_2000", "a4_ask_ns_10000",
         "a5_merge_ns_1", "a5_merge_ns_8", "a5_merge_ns_32",
         "h1_dict_ns", "h1_pthread_ns", "h1_bool_ns"]
    }

    public func run(_ ctx: SuiteContext) async throws -> SuiteResult {
        // Explicit assembly, deliberately not table-driven like Hot/E2E: the
        // A3 probes build history incrementally across H points, so per-metric
        // probes are not independent here.
        var metrics: [MetricResult] = []
        for n in [200, 1000, 5000] {
            let scan = try measureReps(reps: ctx.reps) { try scaleA1(n: n, cached: false) }
            metrics.append(summarizeMetric(name: "a1_scan_us_\(n)", unit: "us", samples: scan))
            let cached = try measureReps(reps: ctx.reps) { try scaleA1(n: n, cached: true) }
            metrics.append(summarizeMetric(name: "a1_cached_us_\(n)", unit: "us", samples: cached))
        }
        for h in [100, 1000, 10000] {
            let scan = try measureReps(reps: ctx.reps) { scaleA2Scan(historySize: h) }
            metrics.append(summarizeMetric(name: "a2_scan_ns_\(h)", unit: "ns", samples: scan))
            let copy = try measureReps(reps: ctx.reps) { scaleA2Copy(historySize: h) }
            metrics.append(summarizeMetric(name: "a2_copy_ns_\(h)", unit: "ns", samples: copy))
        }
        let fetches = try scaleA3()
        for (h, us) in fetches {
            metrics.append(summarizeMetric(name: "a3_full_us_\(h)", unit: "us", samples: [us]))
        }
        for depth in [2000, 10000] {
            let ask = try measureReps(reps: ctx.reps) { try scaleA4(depth: depth) }
            metrics.append(summarizeMetric(name: "a4_ask_ns_\(depth)", unit: "ns", samples: ask))
        }
        for k in [1, 8, 32] {
            let merge = try measureReps(reps: ctx.reps) { scaleA5(params: k) }
            metrics.append(summarizeMetric(name: "a5_merge_ns_\(k)", unit: "ns", samples: merge))
        }
        let dict = try measureReps(reps: ctx.reps) { scaleH1Dict() }
        metrics.append(summarizeMetric(name: "h1_dict_ns", unit: "ns", samples: dict))
        let pthread = try measureReps(reps: ctx.reps) { scaleH1Pthread() }
        metrics.append(summarizeMetric(name: "h1_pthread_ns", unit: "ns", samples: pthread))
        let boolean = try measureReps(reps: ctx.reps) { scaleH1Bool() }
        metrics.append(summarizeMetric(name: "h1_bool_ns", unit: "ns", samples: boolean))
        return SuiteResult(suite: name, branch: "", commit: "", metrics: metrics, environment: [:])
    }
}

private func scaleA1(n: Int, cached: Bool) throws -> Double {
    let study = try createStudy(name: "a1_\(n)_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    var local: [PersistedTrial] = []
    local.reserveCapacity(n)
    var cachedBest = Double.infinity
    let start = DispatchTime.now()
    for _ in 0..<n {
        let bx: Double?
        if cached {
            bx = (cachedBest == .infinity) ? nil : cachedBest
        } else {
            let m = local.lazy.filter { $0.state == .complete }.map { $0.value ?? .infinity }.min()
            bx = (m == .infinity) ? nil : m
        }
        let fx = bx.map { $0 + 0.1 } ?? Double.random(in: -10.0...10.0)
        try study.enqueue(["x": fx])
        var t = try study.ask()
        let x = try t.suggest("x", in: -10.0...10.0)
        let v = x * x
        try study.tell(consuming: t, value: v)
        local.append(PersistedTrial(number: local.count, state: .complete, value: v, params: ["x": .double(x)]))
        if v < cachedBest { cachedBest = v }
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(n)
}

private func scaleA2History(historySize h: Int) -> [PersistedTrial] {
    var arr: [PersistedTrial] = []
    arr.reserveCapacity(h)
    for i in 0..<h {
        arr.append(PersistedTrial(number: i, state: .complete, value: Double(i % 97), params: ["x": .double(Double(i))]))
    }
    return arr
}

private func scaleA2Scan(historySize h: Int) -> Double {
    let arr = scaleA2History(historySize: h)
    var sink = 0.0
    let start = DispatchTime.now()
    for _ in 0..<200 {
        let m = arr.lazy.filter { $0.state == .complete }.map { $0.value ?? .infinity }.min()!
        sink += m
    }
    let end = DispatchTime.now()
    precondition(sink > 0)
    return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 200.0
}

private func scaleA2Copy(historySize h: Int) -> Double {
    let arr = scaleA2History(historySize: h)
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<200 {
        let tail = Array(arr.suffix(max(0, arr.count - (arr.count - 1))))
        sink += tail.count
    }
    let end = DispatchTime.now()
    precondition(sink > 0)
    return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 200.0
}

private func scaleA3() throws -> [(Int, Double)] {
    let study = try createStudy(name: "a3_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    var out: [(Int, Double)] = []
    var built = 0
    for target in [500, 2000, 5000] {
        while built < target {
            let t = try study.ask()
            try study.tell(consuming: t, value: Double(built))
            built += 1
        }
        let start = DispatchTime.now()
        let all = try study.trials
        let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
        out.append((all.count, Double(nanos) / 1_000.0 / Double(all.count)))
    }
    return out
}

private func scaleA4(depth: Int) throws -> Double {
    let study = try createStudy(name: "a4_\(depth)_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    for i in 0..<depth {
        try study.enqueue(["x": Double(i % 21) - 10.0])
    }
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<depth {
        let t = try study.ask()
        sink += t.number
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(sink > 0)
    return Double(nanos) / Double(depth)
}

private func scaleA5(params k: Int) -> Double {
    var fixed: [String: ParameterValue] = [:]
    var sugg: [String: ParameterValue] = [:]
    for p in 0..<k {
        fixed["f\(p)"] = .double(Double(p))
        sugg["s\(p)"] = .double(Double(p))
    }
    let start = DispatchTime.now()
    var count = 0
    for _ in 0..<100_000 {
        let m = fixed.merging(sugg) { _, new in new }
        count += m.count
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    precondition(count > 0)
    return Double(nanos) / 100_000.0
}

private func scaleH1Dict() -> Double {
    let key = "org.swiftuna.benchkit.probe"
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<1_000_000 {
        if Thread.current.threadDictionary[key] != nil { sink += 1 }
    }
    let end = DispatchTime.now()
    precondition(sink >= 0)
    return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
}

private func scaleH1Pthread() -> Double {
    var pkey = pthread_key_t()
    pthread_key_create(&pkey, nil)
    pthread_setspecific(pkey, UnsafeMutableRawPointer(bitPattern: 0x1))
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<1_000_000 {
        if pthread_getspecific(pkey) != nil { sink += 1 }
    }
    let end = DispatchTime.now()
    pthread_key_delete(pkey)
    precondition(sink >= 0)
    return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
}

private func scaleH1Bool() -> Double {
    let flag = false
    var sink = 0
    let start = DispatchTime.now()
    for _ in 0..<1_000_000 {
        if flag { sink += 1 }
    }
    let end = DispatchTime.now()
    precondition(sink >= 0)
    return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
}
