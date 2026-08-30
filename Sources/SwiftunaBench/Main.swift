import Darwin
import Foundation
import LibRustuna
import Swiftuna

// MARK: - ANSI

enum ANSI {
    static let reset = "\u{001B}[0m"
    static let bold = "\u{001B}[1m"
    static let dim = "\u{001B}[2m"
    static let green = "\u{001B}[32m"
    static let yellow = "\u{001B}[33m"
    static let cyan = "\u{001B}[36m"
    static let red = "\u{001B}[31m"
    static func gray(_ s: String) -> String { "\u{001B}[90m\(s)\(reset)" }
}

struct BenchConfig {
    var nTrials = 2_000
    var iters = 20
    var warmupIters = 3
    var verbose = false
    var json = false
    var showHelp = false
}

func parseArgs() -> BenchConfig {
    var c = BenchConfig()
    for arg in CommandLine.arguments.dropFirst() {
        switch arg {
        case "--help", "-h": c.showHelp = true
        case "--verbose", "-v": c.verbose = true
        case "--json": c.json = true
        case _ where arg.hasPrefix("--n="): c.nTrials = Int(arg.dropFirst(4)) ?? c.nTrials
        case _ where arg.hasPrefix("--iters="): c.iters = Int(arg.dropFirst(8)) ?? c.iters
        case _ where arg.hasPrefix("--warmup="): c.warmupIters = Int(arg.dropFirst(9)) ?? c.warmupIters
        default:
            if arg.hasPrefix("-") { fputs("Unknown flag: \(arg)\n", stderr) }
        }
    }
    return c
}

func printHelp() {
    print("""
    \(ANSI.bold)SwiftunaBench\(ANSI.reset) — Swift+C ABI vs Rustuna overhead

    \(ANSI.bold)USAGE\(ANSI.reset)
      swiftunabench [--n=N] [--iters=I] [--warmup=W] [--verbose] [--json]

    \(ANSI.bold)FLAGS\(ANSI.reset)
      --n=N        trials per iteration (default 2000)
      --iters=I    paired iterations (default 20)
      --warmup=W   warmup pairs (default 3)
      --verbose    print per-pair deltas
      --json       emit JSON summary to stdout
      -h, --help   show this help

    \(ANSI.bold)EXAMPLES\(ANSI.reset)
      just bench
      swiftunabench --n=5000 --iters=30 --verbose
    """)
}

// MARK: - Core pinning (macOS best-effort)

func pinCurrentThread() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    var policy = thread_affinity_policy_data_t(affinity_tag: 1)
    withUnsafeMutablePointer(to: &policy) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: 1) { intPtr in
            _ = thread_policy_set(
                pthread_mach_thread_np(pthread_self()),
                thread_policy_flavor_t(THREAD_AFFINITY_POLICY),
                intPtr, 1)
        }
    }
}

// MARK: - Stats

struct Stats {
    let median, mean, stdev, min, max, cvPct: Double
    let samples: [Double]
}

func computeStats(_ samples: [Double]) -> Stats {
    precondition(!samples.isEmpty)
    let sorted = samples.sorted()
    let n = Double(sorted.count)
    let mean = sorted.reduce(0, +) / n
    let median = sorted.count % 2 == 0
        ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        : sorted[sorted.count / 2]
    let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n
    let stdev = variance.squareRoot()
    return Stats(median: median, mean: mean, stdev: stdev, min: sorted.first!, max: sorted.last!, cvPct: mean > 0 ? stdev / mean * 100 : 0, samples: sorted)
}

func trimmedStats(_ samples: [Double], k: Double = 2.0) -> (Stats, Int) {
    let s = computeStats(samples)
    let sorted = s.samples
    let q1 = sorted[sorted.count / 4]
    let q3 = sorted[sorted.count * 3 / 4]
    let iqr = q3 - q1
    let lo = q1 - k * iqr, hi = q3 + k * iqr
    let trimmed = sorted.filter { $0 >= lo && $0 <= hi }
    if trimmed.count >= max(5, sorted.count / 2) { return (computeStats(trimmed), sorted.count - trimmed.count) }
    return (s, 0)
}

// MARK: - Measurement

func measureSwiftE2E(nTrials: Int) throws -> Double {
    let s = ContinuousClock.now
    try swiftE2E(nTrials: nTrials)
    let e = ContinuousClock.now - s
    return (Double(e.components.seconds) + Double(e.components.attoseconds) * 1e-18) * 1_000_000 / Double(nTrials)
}
func measureSwiftRandom(nTrials: Int) throws -> Double {
    let s = ContinuousClock.now
    try swiftRandomNoTPE(nTrials: nTrials)
    let e = ContinuousClock.now - s
    return (Double(e.components.seconds) + Double(e.components.attoseconds) * 1e-18) * 1_000_000 / Double(nTrials)
}
func measureRust(nTrials: Int, useRandom: Bool) -> Double? {
    var ns: UInt64 = 0
    guard rustuna_bench_e2e(nTrials, 42, useRandom, &ns) == 0 else { return nil }
    return Double(ns) / 1_000.0
}

func benchPaired(nTrials: Int, iters: Int, verbose: Bool, swiftBody: () throws -> Double, rustBody: () -> Double?) throws -> (swift: Stats, rust: Stats, paired: Stats) {
    var sw: [Double] = [], ru: [Double] = [], pd: [Double] = []
    sw.reserveCapacity(iters); ru.reserveCapacity(iters); pd.reserveCapacity(iters)
    for i in 0..<iters {
        let label = String(format: " %2d/%2d", i + 1, iters)
        if !verbose {
            // inline progress: \r
            let bar = String(repeating: "█", count: (i * 20) / iters) + String(repeating: "░", count: 20 - (i * 20) / iters)
            print("\r  \(ANSI.dim)▸\(ANSI.reset) \(bar)\(label)", terminator: "")
            fflush(stdout)
        }
        if i % 2 == 0 {
            let s = try swiftBody(); guard let r = rustBody() else { break }
            sw.append(s); ru.append(r); pd.append(s - r)
            if verbose { print(String(format: "  pair %2d: swift %6.1f  rust %6.1f  Δ %+5.1f", i+1, s, r, s-r)) }
        } else {
            guard let r = rustBody() else { break }
            let s = try swiftBody()
            sw.append(s); ru.append(r); pd.append(s - r)
            if verbose { print(String(format: "  pair %2d: rust  %6.1f  swift %6.1f  Δ %+5.1f", i+1, r, s, s-r)) }
        }
        sched_yield()
    }
    if !verbose { print("\r\(ANSI.gray(String(repeating: " ", count: 48)))\r", terminator: "") }
    return (computeStats(sw), computeStats(ru), computeStats(pd))
}

// MARK: - Suites

func swiftE2E(nTrials: Int) throws {
    let study = try Swiftuna.createStudy(name: "bench_e2e_\(UUID().uuidString)", direction: .minimize, sampler: TPESampler(seed: 42))
    for _ in 0..<nTrials {
        var t = try study.ask()
        let x = try t.suggest("x", in: -10.0...10.0)
        let y = try t.suggest("y", in: -10.0...10.0)
        try study.tell(consuming: t, value: (x-2)*(x-2)+(y+5)*(y+5))
    }
}
func swiftRandomNoTPE(nTrials: Int) throws {
    let study = try Swiftuna.createStudy(name: "bench_rand_\(UUID().uuidString)", direction: .minimize, sampler: RandomSampler(seed: 42))
    for _ in 0..<nTrials {
        var t = try study.ask()
        let x = try t.suggest("x", in: -10.0...10.0)
        let y = try t.suggest("y", in: -10.0...10.0)
        try study.tell(consuming: t, value: (x-2)*(x-2)+(y+5)*(y+5))
    }
}

// MARK: - Render helpers

func fmt(_ v: Double, digits: Int = 1) -> String { String(format: "%.\(digits)f", v) }

func verdict(pct: Double, cv: Double, stdev: Double) -> String {
    if abs(pct) < 5 && cv < 5 && stdev < 5 { return "\(ANSI.green)✓ zero overhead\(ANSI.reset) \(ANSI.dim)(<5%, CV<5%, paired σ<5µs)\(ANSI.reset)" }
    if stdev > 10 { return "\(ANSI.yellow)⚠ high jitter\(ANSI.reset) \(ANSI.dim)(close apps, plug in power)\(ANSI.reset)" }
    return "\(ANSI.yellow)◐ within noise\(ANSI.reset)"
}

@main
struct SwiftunaBenchApp {
    static func main() {
        let cfg = parseArgs()
        if cfg.showHelp { printHelp(); return }

        do {
            pinCurrentThread()

            let isTTY = isatty(STDOUT_FILENO) != 0
            let useColor = isTTY && !cfg.json
            func c(_ s: String, _ code: String) -> String { useColor ? "\(code)\(s)\(ANSI.reset)" : s }
            func dim(_ s: String) -> String { useColor ? "\(ANSI.dim)\(s)\(ANSI.reset)" : s }

            if !cfg.json {
                print(dim("┌─ Swiftuna Bench ──────────────────────────────────────"))
                print(dim("│ Swift+C ABI vs Rustuna · in-process · interleaved · core-pinned"))
                print(dim("│ n=\(cfg.nTrials)  iters=\(cfg.iters)  warmup=\(cfg.warmupIters)  release  ") + c("●", ANSI.green))
                print(dim("└──────────────────────────────────────────────────────"))
                fflush(stdout)
            }

            // Warmup (not measured)
            if !cfg.json { print("\n\(c("warmup", ANSI.cyan)) \(cfg.warmupIters)×\(cfg.nTrials/2) …", terminator: ""); fflush(stdout) }
            for _ in 0..<cfg.warmupIters {
                _ = try measureSwiftE2E(nTrials: cfg.nTrials/2)
                _ = measureRust(nTrials: cfg.nTrials/2, useRandom: false)
            }
            if !cfg.json { print(c(" done", ANSI.green)) }

            // E2E TPE
            if !cfg.json { print("\n\(c("● E2E TPE", ANSI.bold))  \(ANSI.dim)quadratic x∈[-10,10] y∈[-10,10]  TPE(seed=42)\(ANSI.reset)"); fflush(stdout) }
            let (swiftRaw, rustRaw, pairedRaw) = try benchPaired(nTrials: cfg.nTrials, iters: cfg.iters, verbose: cfg.verbose,
                swiftBody: { try measureSwiftE2E(nTrials: cfg.nTrials) },
                rustBody: { measureRust(nTrials: cfg.nTrials, useRandom: false) })
            let (swiftStats, sTrim) = trimmedStats(swiftRaw.samples)
            let (rustStats, rTrim) = trimmedStats(rustRaw.samples)
            let (pairedStats, pTrim) = trimmedStats(pairedRaw.samples)
            let pctPaired = rustStats.median > 0 ? pairedStats.median / rustStats.median * 100 : 0
            let pairedFactor = 1 + pctPaired / 100
            let ci = 1.96 * pairedStats.stdev / (Double(cfg.iters).squareRoot())

            if !cfg.json {
                if sTrim + rTrim + pTrim > 0 { print(dim("  (trimmed \(sTrim)+\(rTrim)+\(pTrim) outliers IQR k=2.0)")) }
                print("")
                print(dim("  ┌──────────────┬────────┬────────┬───────┬──────┬────────┬────────┐"))
                print(dim("  │              │ median │   mean │ stdev │   CV │    min │    max │"))
                print(dim("  ├──────────────┼────────┼────────┼───────┼──────┼────────┼────────┤"))
                func row(_ name: String, _ s: Stats) {
                    print(String(format: "  │ %-12@ │ %6.1f │ %6.1f │ %5.2f │ %4.1f%% │ %6.1f │ %6.1f │",
                        name as NSString, s.median, s.mean, s.stdev, s.cvPct, s.min, s.max))
                }
                row("Swift TPE", swiftStats)
                row("Rust TPE", rustStats)
                print(dim("  └──────────────┴────────┴────────┴───────┴──────┴────────┴────────┘"))
                let sign = pairedStats.median >= 0 ? "+" : ""
                let pctStr = String(format: "%+.2f%%", pctPaired)
                print("\n  \(c("overhead", ANSI.bold)) \(sign)\(fmt(pairedStats.median, digits: 2)) µs  (\(pctStr))  \(c(String(format: "%.3f×", pairedFactor), ANSI.cyan))")
                print("  paired median \(fmt(pairedStats.median, digits: 2))  mean \(fmt(pairedStats.mean, digits: 2)) ± \(fmt(ci, digits: 2)) (95% CI)  σ \(fmt(pairedStats.stdev, digits: 2))")
                print("  median diff \(fmt(swiftStats.median - rustStats.median, digits: 2)) µs  Swift \(fmt(swiftStats.median)) vs Rust \(fmt(rustStats.median))")
                print("  \(verdict(pct: pctPaired, cv: max(swiftStats.cvPct, rustStats.cvPct), stdev: pairedStats.stdev))")
            }

            // Random FFI
            if !cfg.json { print("\n\(c("● Random", ANSI.bold))  \(ANSI.dim)isolates FFI (no TPE)\(ANSI.reset)"); fflush(stdout) }
            let (swRandRaw, ruRandRaw, paRandRaw) = try benchPaired(nTrials: cfg.nTrials, iters: cfg.iters, verbose: cfg.verbose,
                swiftBody: { try measureSwiftRandom(nTrials: cfg.nTrials) },
                rustBody: { measureRust(nTrials: cfg.nTrials, useRandom: true) })
            let (swRand, _) = trimmedStats(swRandRaw.samples)
            let (ruRand, _) = trimmedStats(ruRandRaw.samples)
            let (paRand, _) = trimmedStats(paRandRaw.samples)
            let randPct = ruRand.median > 0 ? paRand.median / ruRand.median * 100 : 0
            let randFactor = ruRand.median > 0 ? swRand.median / ruRand.median : 1
            if !cfg.json {
                print(dim("  ┌──────────────┬────────┬────────┬───────┬──────┐"))
                print(dim("  │              │ median │   mean │ stdev │   CV │"))
                print(dim("  ├──────────────┼────────┼────────┼───────┼──────┤"))
                func r2(_ n: String, _ s: Stats) {
                    print(String(format: "  │ %-12@ │ %6.2f │ %6.2f │ %5.3f │ %4.1f%% │", n as NSString, s.median, s.mean, s.stdev, s.cvPct))
                }
                r2("Swift Rand", swRand)
                r2("Rust Rand", ruRand)
                print(dim("  └──────────────┴────────┴────────┴───────┴──────┘"))
                print("  FFI Δ \(fmt(paRand.median, digits: 3)) µs (\(String(format: "%+.1f%%", randPct)))  \(String(format: "%.3f×", randFactor))  σ \(fmt(paRand.stdev, digits: 3))")
            }

            // Deserialization
            if !cfg.json { print("\n\(c("● Trials", ANSI.bold))  \(ANSI.dim)deserialization  100 rich trials ×1000 fetches\(ANSI.reset)"); fflush(stdout) }
            let qStudy = try Swiftuna.createStudy(name: "bench_query_\(UUID().uuidString)")
            for i in 0..<100 {
                var t = try qStudy.ask()
                _ = try t.suggest("param_a", in: -10.0...10.0)
                _ = try t.suggest("param_b", in: 1...100)
                try t.setUserAttr("tag", value: "epoch_\(i)")
                try t.setConstraint("c1", value: Double(i-50))
                try t.report(Double(i)*0.1, step: 0); try t.report(Double(i)*0.05, step: 1)
                try qStudy.tell(consuming: t, value: Double(i)*1.5)
            }
            _ = try qStudy.trials
            let qIters = 1000
            let qStart = ContinuousClock.now
            var qTotal = 0
            for _ in 0..<qIters { qTotal += try qStudy.trials.count }
            let qElapsed = ContinuousClock.now - qStart
            let qMs = (Double(qElapsed.components.seconds) + Double(qElapsed.components.attoseconds)*1e-18)*1000
            let qUs = qMs*1000/Double(qTotal)
            if !cfg.json {
                print(String(format: "  %.1f ms total  %.2f µs/trial  %.0f trials/s  (%d trials)", qMs, qUs, Double(qTotal)/(qMs/1000), qTotal))
                print("\n\(c("done", ANSI.green)) \(ANSI.dim)— overhead is paired median; factor = swift/rust. Re-run with --verbose for per-pair deltas, --json for CI.\(ANSI.reset)")
            }

            if cfg.json {
                let out: [String: Any] = [
                    "nTrials": cfg.nTrials, "iters": cfg.iters,
                    "swift": ["median": swiftStats.median, "mean": swiftStats.mean, "stdev": swiftStats.stdev, "cv": swiftStats.cvPct],
                    "rust": ["median": rustStats.median, "mean": rustStats.mean, "stdev": rustStats.stdev, "cv": rustStats.cvPct],
                    "paired": ["median": pairedStats.median, "mean": pairedStats.mean, "stdev": pairedStats.stdev, "pct": pctPaired, "factor": pairedFactor, "ci95": ci],
                    "random": ["swiftMedian": swRand.median, "rustMedian": ruRand.median, "pairedMedian": paRand.median, "factor": randFactor],
                    "trials": ["usPerTrial": qUs, "throughput": Double(qTotal)/(qMs/1000)]
                ]
                let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
                print(String(data: data, encoding: .utf8)!)
            }

        } catch {
            fputs("Benchmark failed: \(error)\n", stderr)
            exit(1)
        }
    }
}
