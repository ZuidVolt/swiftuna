import Foundation

/// Minimal ANSI. Plain words carry verdicts (PASS/FAIL/INCONCLUSIVE);
/// color is decoration only and disabled when piped or --json.
public enum ANSI {
    public static let reset = "\u{001B}[0m"
    public static let bold = "\u{001B}[1m"
    public static let dim = "\u{001B}[2m"
    public static let green = "\u{001B}[32m"
    public static let yellow = "\u{001B}[33m"
    public static let cyan = "\u{001B}[36m"
    public static let red = "\u{001B}[31m"
}

/// Renders one metric row of a results table.
public func metricRow(_ m: MetricResult, useColor: Bool) -> String {
    String(
        format: "%-24@ %12.3f %-8@ mean=%9.3f stdev=%8.3f CV=%5.1f%% n=%d",
        m.name as NSString, m.median, m.unit as NSString,
        m.mean, m.stdev, m.cv * 100, m.n)
}

/// Renders a full suite result as a human-readable table.
public func renderSuiteResult(_ r: SuiteResult, useColor: Bool) -> String {
    func c(_ s: String, _ code: String) -> String { useColor ? "\(code)\(s)\(ANSI.reset)" : s }
    var lines: [String] = []
    lines.append("\(c(r.suite, ANSI.bold))  branch=\(r.branch) commit=\(r.commit)")
    for m in r.metrics {
        lines.append("  " + metricRow(m, useColor: useColor))
    }
    if let load = r.environment["load1"], let thermal = r.environment["thermal"] {
        lines.append(c("  env: load1=\(load) thermal=\(thermal)", ANSI.dim))
    }
    return lines.joined(separator: "\n")
}

/// Renders metric verdicts. One line per metric, worst status wins upstream.
public func renderVerdicts(_ verdicts: [MetricVerdict], useColor: Bool) -> String {
    func c(_ s: String, _ code: String) -> String { useColor ? "\(code)\(s)\(ANSI.reset)" : s }
    return verdicts.map { v in
        let label: String
        switch v.status {
        case .pass: label = c("PASS", ANSI.green)
        case .regression: label = c("FAIL", ANSI.red)
        case .inconclusive: label = c("INCONCLUSIVE", ANSI.yellow)
        }
        let sign = v.delta >= 0 ? "+" : ""
        return
            "  [\(label)] \(v.metric): \(sign)\(String(format: "%.1f", v.delta * 100))% (gate \(String(format: "%.1f", v.gate * 100))%) — \(v.reason)"
    }.joined(separator: "\n")
}

/// Serializes a result file (pretty, sorted keys).
public func encodeResult(_ r: SuiteResult) throws -> String {
    let data = try JSONEncoder().encode(r)
    let obj = try JSONSerialization.jsonObject(with: data)
    let pretty = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    guard let s = String(data: pretty, encoding: .utf8) else {
        throw BenchError.suiteFailed("result JSON is not UTF-8")
    }
    return s
}

/// Reads a result file back. Schema mismatches throw.
public func decodeResult(from json: String) throws -> SuiteResult {
    guard let data = json.data(using: .utf8) else {
        throw BenchError.suiteFailed("result file is not UTF-8")
    }
    return try JSONDecoder().decode(SuiteResult.self, from: data)
}
