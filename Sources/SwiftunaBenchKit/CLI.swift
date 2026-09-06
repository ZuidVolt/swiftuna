/// Exit codes for the bench tool. Shared by run/compare/gate so CI can rely on them.
public enum BenchExit: Int32, Sendable {
    /// All gates passed.
    case pass = 0
    /// A calibrated gate failed: measured regression.
    case regression = 1
    /// No verdict possible: jitter too high, environment too loaded, or missing data.
    case inconclusive = 2
}

/// Top-level subcommand.
public enum BenchCommand: String, Sendable {
    case run
    case compare
    case gate
    case listSuites = "list-suites"
}

/// Parsed CLI configuration. Defaults match the historical bench rig.
public struct BenchConfig: Sendable {
    public var command: BenchCommand = .run
    public var suite: String = "hot"
    /// True when --suite was passed explicitly (compare defaults to all shared).
    public var suiteSet: Bool = false
    public var reps: Int = 5
    public var json: Bool = false
    public var requireQuiet: Bool = false
    /// Compare only: run even when both refs resolve identically, and use
    /// committed worktree states even when the tree is dirty.
    public var force: Bool = false
    /// Compare only: rebuild worktrees/binaries from scratch instead of reusing.
    public var fresh: Bool = false
    public var refA: String = "main"
    public var refB: String = "HEAD"
    public var baselineFile: String = ""
    public var currentFile: String = ""
    public var gateFloor: Double = 0.05
    public var showHelp: Bool = false
    public var parseError: String? = nil

    public init() {}
}

/// Parses `CommandLine.arguments` (minus argv[0]) into a `BenchConfig`.
public func parseBenchArgs(_ args: [String]) -> BenchConfig {
    var cfg = BenchConfig()
    var positional: [String] = []
    var i = args.startIndex
    // Flags taking a value accept both --flag=value and --flag value.
    func takeValue(prefix: String, arg: String, next: inout Array<String>.Index) -> String? {
        if arg.hasPrefix(prefix + "=") {
            return String(arg.dropFirst(prefix.count + 1))
        }
        if arg == prefix {
            let j = args.index(after: next)
            if j < args.endIndex {
                next = j
                return args[j]
            }
        }
        return nil
    }
    while i < args.endIndex {
        let arg = args[i]
        switch arg {
        case "--help", "-h":
            cfg.showHelp = true
        case "--json":
            cfg.json = true
        case "--require-quiet":
            cfg.requireQuiet = true
        case "--force":
            cfg.force = true
        case "--fresh":
            cfg.fresh = true
        default:
            if let v = takeValue(prefix: "--suite", arg: arg, next: &i) {
                cfg.suite = v
                cfg.suiteSet = true
            } else if let v = takeValue(prefix: "--reps", arg: arg, next: &i), let n = Int(v) {
                cfg.reps = n
            } else if let v = takeValue(prefix: "--floor", arg: arg, next: &i), let f = Double(v) {
                cfg.gateFloor = f
            } else if let v = takeValue(prefix: "--baseline", arg: arg, next: &i) {
                cfg.baselineFile = v
            } else if let v = takeValue(prefix: "--current", arg: arg, next: &i) {
                cfg.currentFile = v
            } else if arg.hasPrefix("-") {
                cfg.parseError = "Unknown flag: \(arg)"
            } else {
                positional.append(arg)
            }
        }
        i = args.index(after: i)
    }
    if let first = positional.first, let cmd = BenchCommand(rawValue: first) {
        cfg.command = cmd
        positional.removeFirst()
    }
    // compare consumes two positional refs.
    if cfg.command == .compare {
        if positional.count >= 2 {
            cfg.refA = positional[0]
            cfg.refB = positional[1]
        } else {
            cfg.parseError = "compare needs two refs: compare <A> <B> [--suite=...]"
        }
    }
    return cfg
}

/// One-line usage.
public func benchHelp() -> String {
    """
    swiftunabench <command> [options]

    COMMANDS
      run --suite=<name> [--reps=N] [--json] [--require-quiet]
        Run one suite from the registry. See list-suites.
      list-suites [--json]
        Print registered suites and their metric names.
      compare <A> <B> --suite=<name> [--reps=N] [--json] [--force]
        Build and run the suite on two git refs via worktrees, then judge.
        Refusing to compare a ref to itself (exit 0) unless --force.
        With uncommitted changes, the dirty side builds from the working
        tree so the changes are actually measured.
      gate --baseline=<f> --current=<f> [--floor=P] [--json]
        Judge two result files. Exit 0 pass, 1 regression, 2 inconclusive.

    OPTIONS
      --reps=N           repetitions per metric (default 5)
      --floor=P          minimum gate fraction, e.g. 0.05 (default 0.05)
      --require-quiet    refuse to run when the machine is loaded or hot
      --json             machine-readable output on stdout
      -h, --help         show this help
    """
}
