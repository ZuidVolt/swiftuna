import Dispatch
import Foundation
import Swiftuna

/// New-API scaling suite: the real custom-sampler driver and tail fetch.
/// Requires CustomSampler/trials(since:) — the compare tool skips this suite
/// on branches that cannot build it (see list-suites negotiation).
public struct ScaleNewAPISuite: BenchSuite {
    public init() {}

    public var name: String { "scale-newapi" }
    public var description: String { "Real custom driver per-trial cost and tail-fetch scaling (new APIs only)" }
    public var requiresNewAPI: Bool { true }
    public var metricNames: [String] {
        ["a1real_us_200", "a1real_us_1000", "a1real_us_5000",
         "a3since_us_500", "a3since_us_2000", "a3since_us_5000"]
    }

    public func run(_ ctx: SuiteContext) async throws -> SuiteResult {
        var metrics: [MetricResult] = []
        for n in [200, 1000, 5000] {
            let per = try measureReps(reps: ctx.reps) { try scaleRealDriver(trials: n) }
            metrics.append(summarizeMetric(name: "a1real_us_\(n)", unit: "us", samples: per))
        }
        let tails = try scaleSinceFetch()
        for (h, us) in tails {
            metrics.append(summarizeMetric(name: "a3since_us_\(h)", unit: "us", samples: [us]))
        }
        return SuiteResult(suite: name, branch: "", commit: "", metrics: metrics, environment: [:])
    }
}

private struct TrivialClimb: CustomSampler {
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        guard let bx = history.best?.params["x"]?.asDouble else {
            return ["x": .double(1.0)]
        }
        return ["x": .double(bx * 0.99)]
    }
}

private func scaleRealDriver(trials n: Int) throws -> Double {
    let study = try createStudy(name: "a1r_\(n)_\(UUID().uuidString)", direction: .minimize,
                                sampler: RandomSampler(seed: 7))
    let start = DispatchTime.now()
    try study.optimize(nTrials: n, using: TrivialClimb()) { trial in
        let x = try trial.suggest("x", in: -10.0...10.0)
        return x * x
    }
    let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(nanos) / 1_000.0 / Double(n)
}

private func scaleSinceFetch() throws -> [(Int, Double)] {
    let study = try createStudy(name: "a3s_\(UUID().uuidString)", direction: .minimize,
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
        let tail = try study.trials(where: Set(TrialState.allCases), since: built - 100)
        let nanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
        out.append((built, Double(nanos) / 1_000.0 / Double(tail.count)))
    }
    return out
}
