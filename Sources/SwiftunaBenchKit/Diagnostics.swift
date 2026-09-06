import Foundation

/// Stable codes for every failure mode the bench tool knows how to explain.
/// Codes make diagnostics greppable; the help text does the actual helping.
public enum BenchDiagCode: String, Sendable {
    case worktreeConflict = "BENCH001"
    case staleWorktree = "BENCH002"
    case buildFailed = "BENCH003"
    case suiteMissing = "BENCH004"
    case identicalRefs = "BENCH005"
    case dirtyTree = "BENCH006"
    case notACheckout = "BENCH007"
    case loadedMachine = "BENCH008"
    case thermalThrottled = "BENCH009"
    case highJitter = "BENCH010"
    case unknownSuite = "BENCH011"
    case suiteCrashed = "BENCH012"
}

/// One rustc-shaped diagnostic: a headline, optional evidence, and
/// `= note:` / `= help:` lines. Rendered, never thrown.
public struct Diagnostic: Sendable {
    public var level: String
    public var code: BenchDiagCode?
    public var message: String
    /// Evidence lines, rendered dim under the headline (commands, excerpts).
    public var labels: [String]
    public var notes: [String]
    public var helps: [String]

    public init(level: String = "error", code: BenchDiagCode? = nil, message: String,
                labels: [String] = [], notes: [String] = [], helps: [String] = []) {
        self.level = level
        self.code = code
        self.message = message
        self.labels = labels
        self.notes = notes
        self.helps = helps
    }

    public func render(useColor: Bool) -> String {
        func c(_ s: String, _ code: String) -> String { useColor ? "\(code)\(s)\(ANSI.reset)" : s }
        let headColor = level == "error" ? ANSI.red : level == "warning" ? ANSI.yellow : ANSI.cyan
        let tag = code.map { "[\(c($0.rawValue, ANSI.dim))] " } ?? ""
        var lines = ["\(c(level, headColor))\(c(": ", headColor))\(tag)\(message)"]
        for l in labels {
            lines.append("  \(c("|", ANSI.dim)) \(c(l, ANSI.dim))")
        }
        for n in notes {
            lines.append("  \(c("=", ANSI.dim)) note: \(n)")
        }
        for h in helps {
            lines.append("  \(c("=", ANSI.dim)) help: \(h)")
        }
        return lines.joined(separator: "\n")
    }
}

/// Turns any bench failure into an explained diagnostic. Every branch here
/// was earned the hard way — a failure mode hit during real sessions, with
/// the fix that actually resolved it.
public func classify(_ error: any Error) -> Diagnostic {
    guard let bench = error as? BenchError else {
        return Diagnostic(message: String(describing: error),
                          helps: ["re-run with --verbose if available; report with the full log"])
    }
    switch bench {
    case .commandFailed(let command, let status, let stderr):
        return classifyCommand(command: command, status: status, stderr: stderr)
    case .loadedEnvironment(let m):
        return Diagnostic(
            code: .loadedMachine, message: "refusing to measure: \(m)",
            notes: ["sub-microsecond numbers taken under load are fiction — medians drift, CVs explode"],
            helps: ["close CPU-heavy apps and re-run",
                    "pass --require-quiet to make every command enforce this up front"])
    case .suiteFailed(let m):
        // Structured cases below handle the known shapes; this is the fallback.
        if m.hasPrefix("suite '") && m.contains("missing on") {
            return Diagnostic(code: .suiteMissing, message: m,
                              notes: ["old branches only build the suites their APIs support; compare intersects listings"],
                              helps: ["run list-suites on each side to see the overlap",
                                      "new-API suites (scale-newapi, ffi) need a branch that ships BenchKit"])
        }
        if m.hasPrefix("no shared suites") {
            return Diagnostic(code: .suiteMissing, message: m,
                              helps: ["compare newer refs, or port the suite down to old APIs"])
        }
        if m.contains("must run from a swiftuna checkout") {
            return Diagnostic(code: .notACheckout, message: m,
                              helps: ["cd to the swiftuna checkout (compare copies BenchKit sources from it)"])
        }
        if m.contains("list-suites undecodable") {
            return Diagnostic(code: .suiteCrashed, message: m,
                              notes: ["the side built, but its listing is not valid JSON"],
                              helps: ["re-run that side's binary directly: <binary> list-suites --json"])
        }
        return Diagnostic(message: m)
    }
}

private func classifyCommand(command: String, status: Int32, stderr: String) -> Diagnostic {
    let labels = ["command: \(command)", "exit: \(status)"] + (stderr.isEmpty ? [] : ["stderr: \(stderr)"])
    if command.contains("git") && command.contains("worktree") {
        if stderr.contains("already exists") || stderr.contains("already used") {
            return Diagnostic(code: .worktreeConflict, message: "worktree path collision for an existing checkout",
                              labels: labels,
                              notes: ["usually a killed run: the directory is gone but git still has it registered, or vice versa"],
                              helps: ["git worktree prune, then re-run",
                                      "pass --fresh to force a clean rebuild of that side"])
        }
        if stderr.contains("not a working tree") || stderr.contains("not a worktree") {
            return Diagnostic(code: .staleWorktree, message: "scratch directory is not a registered worktree",
                              labels: labels,
                              notes: ["an aborted run left a plain directory where a worktree should be"],
                              helps: ["delete .bench-worktrees/<name> (plain rm is safe here — it was never registered) and re-run",
                                      "the tool clears this itself on current builds; update and retry"])
        }
        if stderr.contains("not a git repository") || stderr.contains("not a checkout") {
            return Diagnostic(code: .notACheckout, message: "not inside a git checkout",
                              labels: labels,
                              helps: ["cd to the swiftuna checkout and re-run"])
        }
    }
    if command.contains("swift build") || command.contains("swift package") {
        return Diagnostic(code: .buildFailed, message: "Swift build failed",
                          labels: labels,
                          helps: ["re-run with --fresh to drop incremental state (stale .build dirs bite after toolchain updates)",
                                  "on old-branch overlays: the branch may genuinely not compile current BenchKit — check the tail above for the first error",
                                  "verify the toolchain with swift --version (repo pins Swift 6.3, Xcode beta)"])
    }
    if stderr.contains("no such file") || stderr.contains("NotFound") || status == 1 && stderr.isEmpty {
        return Diagnostic(message: "command failed: \(command)",
                          labels: labels,
                          helps: ["check the binary exists and is executable; rebuild with --fresh"])
    }
    return Diagnostic(message: "command failed with exit \(status): \(command)", labels: labels)
}

/// Help lines for a finished verdict table: when metrics come back
/// inconclusive or the rig was loud, say what to do instead of just red.
public func hintsForVerdicts(_ verdicts: [MetricVerdict], load1: Double?, cores: Int?, lowPower: Bool = false) -> [String] {
    var hints: [String] = []
    if verdicts.contains(where: { $0.status == .inconclusive }) {
        hints.append("inconclusive metrics are not passes: re-run with --reps 8+ to shrink the confidence interval")
    }
    if lowPower {
        hints.append("Low Power Mode was on during this run — the OS throttled clocks, so treat every number as a lower bound and re-run plugged in with it off")
    }
    if let load = load1, let cores, cores > 0, load > Double(cores) {
        hints.append("machine was loaded (1m avg \(String(format: "%.1f", load)) vs \(cores) cores) — deltas this small are scheduling noise; --require-quiet refuses such runs up front")
    }
    if verdicts.contains(where: { $0.reason.contains("jitter") }) {
        hints.append("jitter above the limit usually means thermal throttling or background work, not a real change — check thermal state and close heavy apps")
    }
    return hints
}
