import Foundation

/// Summary statistics over one metric's repetitions.
public struct BenchStats: Sendable, Codable {
    public var median: Double
    public var mean: Double
    public var stdev: Double
    public var min: Double
    public var max: Double
    /// Coefficient of variation in percent.
    public var cvPct: Double
    public var n: Int

    public init(median: Double, mean: Double, stdev: Double, min: Double, max: Double, cvPct: Double, n: Int) {
        self.median = median
        self.mean = mean
        self.stdev = stdev
        self.min = min
        self.max = max
        self.cvPct = cvPct
        self.n = n
    }
}

/// Median-first summary. Median resists the one-off scheduling spikes that
/// dominate short-bench noise; mean/stdev ride along for the CI math.
public func computeStats(_ samples: [Double]) -> BenchStats {
    precondition(!samples.isEmpty, "computeStats needs at least one sample")
    let sorted = samples.sorted()
    let n = sorted.count
    let mean = sorted.reduce(0, +) / Double(n)
    let median =
        n % 2 == 0
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        : sorted[n / 2]
    let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n)
    let stdev = variance.squareRoot()
    return BenchStats(
        median: median, mean: mean, stdev: stdev,
        min: sorted.first!, max: sorted.last!,
        cvPct: mean > 0 ? stdev / mean * 100 : 0, n: n)
}

/// IQR trim (Tukey fences, k=2) with a floor on survivors: if trimming would
/// eat more than half the samples, something is wrong with the rig, so keep
/// everything and let the CV show it. Returns stats plus trimmed count.
public func trimmedStats(_ samples: [Double], k: Double = 2.0) -> (BenchStats, Int) {
    let full = computeStats(samples)
    let sorted = samples.sorted()
    let q1 = sorted[sorted.count / 4]
    let q3 = sorted[sorted.count * 3 / 4]
    let iqr = q3 - q1
    let lo = q1 - k * iqr
    let hi = q3 + k * iqr
    let kept = sorted.filter { $0 >= lo && $0 <= hi }
    if kept.count >= max(5, sorted.count / 2) {
        return (computeStats(kept), sorted.count - kept.count)
    }
    return (full, 0)
}

/// Calibrated gate from measured variance: twice the baseline CV, with a
/// floor so tight rigs still demand a meaningful delta. Both as fractions.
public func calibratedGate(baselineCV: Double, floor: Double) -> Double {
    max(2 * baselineCV, floor)
}

/// Worst of a set of verdicts: regression dominates inconclusive dominates
/// pass. Single ordering shared by compare and gate.
public func worstStatus(_ statuses: [BenchExit]) -> BenchExit {
    if statuses.contains(.regression) { return .regression }
    if statuses.contains(.inconclusive) { return .inconclusive }
    return .pass
}

/// Median-first summary of raw repetitions as a result metric.
public func summarizeMetric(name: String, unit: String, samples: [Double]) -> MetricResult {
    let s = computeStats(samples)
    return MetricResult(name: name, unit: unit, median: s.median, mean: s.mean,
                        stdev: s.stdev, cv: s.cvPct / 100, n: s.n)
}

/// One metric's verdict against its baseline.
public struct MetricVerdict: Sendable {
    public var metric: String
    public var status: BenchExit
    /// Observed relative delta, fraction (+0.03 = 3% slower).
    public var delta: Double
    /// Gate applied, fraction.
    public var gate: Double
    public var reason: String

    public init(metric: String, status: BenchExit, delta: Double, gate: Double, reason: String) {
        self.metric = metric
        self.status = status
        self.delta = delta
        self.gate = gate
        self.reason = reason
    }
}

/// Judges one metric. Improvements always pass when `improvementOK`
/// (gates are anti-regression, not anti-change). Baselines at zero or
/// jitter above `jitterLimit` yield inconclusive, never a false pass.
public func judgeMetric(
    name: String,
    baselineMedian: Double,
    baselineCV: Double,
    currentMedian: Double,
    floor: Double,
    jitterLimit: Double = 0.15,
    improvementOK: Bool = true
) -> MetricVerdict {
    guard baselineMedian > 0 else {
        return MetricVerdict(
            metric: name, status: .inconclusive, delta: 0, gate: floor,
            reason: "baseline median is zero")
    }
    let gate = calibratedGate(baselineCV: baselineCV, floor: floor)
    let delta = (currentMedian - baselineMedian) / baselineMedian
    if baselineCV > jitterLimit {
        return MetricVerdict(
            metric: name, status: .inconclusive, delta: delta, gate: gate,
            reason: "baseline CV \(Int(baselineCV * 100))% exceeds jitter limit")
    }
    if improvementOK, delta <= 0 {
        return MetricVerdict(
            metric: name, status: .pass, delta: delta, gate: gate,
            reason: "improvement")
    }
    if delta <= gate {
        return MetricVerdict(
            metric: name, status: .pass, delta: delta, gate: gate,
            reason: "within gate")
    }
    return MetricVerdict(
        metric: name, status: .regression, delta: delta, gate: gate,
        reason: "exceeds gate")
}
