import Foundation

/// One measured metric: medians over reps, computed by the runner.
public struct MetricResult: Sendable, Codable {
    public var name: String
    public var unit: String
    public var median: Double
    public var mean: Double
    public var stdev: Double
    /// Coefficient of variation, fraction (0.01 = 1%).
    public var cv: Double
    public var n: Int

    public init(name: String, unit: String, median: Double, mean: Double, stdev: Double, cv: Double, n: Int) {
        self.name = name
        self.unit = unit
        self.median = median
        self.mean = mean
        self.stdev = stdev
        self.cv = cv
        self.n = n
    }
}

/// Everything one suite run produced, plus the environment it ran in.
public struct SuiteResult: Sendable, Codable {
    public var schemaVersion: Int = 1
    public var suite: String
    public var branch: String
    public var commit: String
    public var metrics: [MetricResult]
    /// Snapshot of machine state for auditability (loadavg, thermal).
    public var environment: [String: String]

    public init(suite: String, branch: String, commit: String, metrics: [MetricResult], environment: [String: String]) {
        self.suite = suite
        self.branch = branch
        self.commit = commit
        self.metrics = metrics
        self.environment = environment
    }
}

/// A benchmark suite: a named set of metrics measured together in-process.
public protocol BenchSuite: Sendable {
    /// Registry name, e.g. "hot".
    var name: String { get }
    /// One-line description for list-suites.
    var description: String { get }
    /// True when the suite needs APIs that may not exist on older branches.
    /// The compare tool uses this (via list-suites intersection) to skip
    /// suites a side cannot build.
    var requiresNewAPI: Bool { get }
    /// Metric names in stable order, for list-suites and result validation.
    var metricNames: [String] { get }
    /// Run all metrics, `ctx.reps` repetitions each. Returns one median set.
    /// Branch/commit/environment are stamped by `executeSuite`, not here.
    func run(_ ctx: SuiteContext) async throws -> SuiteResult
}

/// How to run a suite. Threaded through so CLI flags reach every suite.
public struct SuiteContext: Sendable {
    public var reps: Int
    public var requireQuiet: Bool

    public init(reps: Int = 5, requireQuiet: Bool = false) {
        self.reps = reps
        self.requireQuiet = requireQuiet
    }
}

/// The suite registry. Suites register here as they are ported.
public enum SuiteRegistry {
    public static var suites: [any BenchSuite] {
        [HotSuite(), ScaleSuite(), ScaleNewAPISuite(), E2ESuite(), FFIMicroSuite()]
    }

    public static func named(_ name: String) -> (any BenchSuite)? {
        suites.first { $0.name == name }
    }
}
