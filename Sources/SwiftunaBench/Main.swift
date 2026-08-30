import Foundation
import LibRustuna
import Swiftuna

// MARK: - Stats

struct Stats {
    let median: Double
    let mean: Double
    let stdev: Double
    let min: Double
    let max: Double
    let cvPct: Double
    let samples: [Double] // µs per trial
}

func computeStats(_ samples: [Double]) -> Stats {
    let sorted = samples.sorted()
    let n = Double(sorted.count)
    let mean = sorted.reduce(0, +) / n
    let median: Double
    if sorted.count % 2 == 0 {
        median = (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    } else {
        median = sorted[sorted.count / 2]
    }
    let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n
    let stdev = variance.squareRoot()
    let cv = mean > 0 ? stdev / mean * 100 : 0
    return Stats(median: median, mean: mean, stdev: stdev, min: sorted.first ?? 0, max: sorted.last ?? 0, cvPct: cv, samples: sorted)
}

func benchWallClockUsPerTrial(nTrials: Int, iters: Int, body: () throws -> Void) rethrows -> Stats {
    var samples: [Double] = []
    samples.reserveCapacity(iters)
    for i in 0..<iters {
        print("  swift iter \(i+1)/\(iters)...", terminator: " "); fflush(stdout)
        let start = ContinuousClock.now
        try body()
        let elapsed = ContinuousClock.now - start
        let sec = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let us = sec * 1_000_000 / Double(nTrials)
        print(String(format: "%.1f µs", us)); fflush(stdout)
        samples.append(us)
    }
    return computeStats(samples)
}

func rustInProcessStats(nTrials: Int, iters: Int, seed: UInt64 = 42, useRandom: Bool) -> Stats? {
    var samples: [Double] = []
    samples.reserveCapacity(iters)
    for i in 0..<iters {
        print("  rust iter \(i+1)/\(iters)...", terminator: " "); fflush(stdout)
        var ns: UInt64 = 0
        let rc = rustuna_bench_e2e(nTrials, seed, useRandom, &ns)
        guard rc == 0 else { print("fail rc \(rc)"); return nil }
        let us = Double(ns) / 1_000.0
        print(String(format: "%.1f µs", us)); fflush(stdout)
        samples.append(us)
    }
    return computeStats(samples)
}

// MARK: - Suites

func swiftE2E(nTrials: Int) throws {
    let study = try Swiftuna.createStudy(name: "bench_e2e_\(UUID().uuidString)", direction: .minimize, sampler: TPESampler(seed: 42))
    for _ in 0..<nTrials {
        var trial = try study.ask()
        let x = try trial.suggest("x", in: -10.0...10.0)
        let y = try trial.suggest("y", in: -10.0...10.0)
        try study.tell(consuming: trial, value: (x - 2) * (x - 2) + (y + 5) * (y + 5))
    }
}

func swiftRandomNoTPE(nTrials: Int) throws {
    let study = try Swiftuna.createStudy(name: "bench_rand_\(UUID().uuidString)", direction: .minimize, sampler: RandomSampler(seed: 42))
    for _ in 0..<nTrials {
        var trial = try study.ask()
        let x = try trial.suggest("x", in: -10.0...10.0)
        let y = try trial.suggest("y", in: -10.0...10.0)
        try study.tell(consuming: trial, value: (x - 2) * (x - 2) + (y + 5) * (y + 5))
    }
}

@main
struct SwiftunaBenchApp {
    static func main() {
        do {
            print("=========================================================")
            print("  Swiftuna Overhead Benchmark — Swift+C ABI vs Rustuna  ")
            print("  In-process Rust baseline (no fork), release build      ")
            print("=========================================================\n")
            fflush(stdout)

            let nTrials = 1_000
            let iters = 10
            let warmupIters = 2

            print("Warmup (\(warmupIters)×500)..."); fflush(stdout)
            for _ in 0..<warmupIters { try swiftE2E(nTrials: 500) }
            for _ in 0..<warmupIters { _ = rustInProcessStats(nTrials: 500, iters: 1, useRandom: false) }
            print("Warmup done"); fflush(stdout)

            print("Suite: E2E TPE quadratic (n=\(nTrials), iters=\(iters), warmup=\(warmupIters))"); fflush(stdout)
            let swiftStats = try benchWallClockUsPerTrial(nTrials: nTrials, iters: iters) { try swiftE2E(nTrials: nTrials) }
            guard let rustStats = rustInProcessStats(nTrials: nTrials, iters: iters, useRandom: false) else {
                print("Rust in-process bench failed"); exit(1)
            }

            print("|                | median |   mean |  stdev |    CV |    min |    max |")
            print("| -------------- | -----: | -----: | -----: | -----: | -----: | -----: |")
            func row(_ name: String, _ s: Stats) {
                let line = String(format: "| %-14@ | %6.2f | %6.2f | %6.2f | %5.1f%% | %6.2f | %6.2f |", name as NSString, s.median, s.mean, s.stdev, s.cvPct, s.min, s.max)
                print(line); fflush(stdout)
            }
            row("Swiftuna TPE", swiftStats)
            row("Rustuna TPE", rustStats)
            let overhead = swiftStats.median - rustStats.median
            let pct = rustStats.median > 0 ? overhead / rustStats.median * 100 : 0
            // 95% CI approx ±1.96*stdev/sqrt(n)
            let swiftCI = 1.96 * swiftStats.stdev / (Double(iters).squareRoot())
            let rustCI = 1.96 * rustStats.stdev / (Double(iters).squareRoot())
            print(String(format: "\nOverhead (median): %+.2f µs/trial (%+.2f%%)  Swift %.2f±%.2f vs Rust %.2f±%.2f (95%% CI)", overhead, pct, swiftStats.median, swiftCI, rustStats.median, rustCI))
            if abs(pct) < 5 && swiftStats.cvPct < 15 && rustStats.cvPct < 15 {
                print("✓ Within noise (<5%, CV<15%) — effectively zero overhead")
            }

            // FFI isolation
            print("\nSuite: RandomSampler (isolates FFI, no TPE) n=\(nTrials), iters=\(iters)")
            let swiftRandStats = try benchWallClockUsPerTrial(nTrials: nTrials, iters: iters) { try swiftRandomNoTPE(nTrials: nTrials) }
            guard let rustRandStats = rustInProcessStats(nTrials: nTrials, iters: iters, useRandom: true) else {
                print("Rust random bench failed"); exit(1)
            }
            row("Swift Random", swiftRandStats)
            row("Rust Random", rustRandStats)
            let randOverhead = swiftRandStats.median - rustRandStats.median
            let randPct = rustRandStats.median > 0 ? randOverhead / rustRandStats.median * 100 : 0
            print(String(format: "FFI overhead (median): %+.3f µs/trial (%+.1f%%)", randOverhead, randPct))

            // Deserialization
            print("\nSuite: Study.trials deserialization (100 rich trials ×1000 fetches = 100k trials)"); fflush(stdout)
            let qStudy = try Swiftuna.createStudy(name: "bench_query_\(UUID().uuidString)")
            for i in 0..<100 {
                var t = try qStudy.ask()
                _ = try t.suggest("param_a", in: -10.0...10.0)
                _ = try t.suggest("param_b", in: 1...100)
                try t.setUserAttr("tag", value: "epoch_\(i)")
                try t.setConstraint("c1", value: Double(i - 50))
                try t.report(Double(i) * 0.1, step: 0)
                try t.report(Double(i) * 0.05, step: 1)
                try qStudy.tell(consuming: t, value: Double(i) * 1.5)
            }
            print("  query warmup..."); fflush(stdout)
            _ = try qStudy.trials
            let qIters = 1000
            print("  benching \(qIters) fetches..."); fflush(stdout)
            let qStart = ContinuousClock.now
            var qTotal = 0
            for _ in 0..<qIters { qTotal += try qStudy.trials.count }
            let qElapsed = ContinuousClock.now - qStart
            let qMs = (Double(qElapsed.components.seconds) + Double(qElapsed.components.attoseconds) * 1e-18) * 1000
            let qUs = qMs * 1000 / Double(qTotal)
            print(String(format: "Total: %.2f ms  Per-trial: %.2f µs  Throughput: %.0f trials/sec", qMs, qUs, Double(qTotal) / (qMs / 1000))); fflush(stdout)

            print("\n=========================================================")
        } catch {
            print("Benchmark failed: \(error)")
            exit(1)
        }
    }
}
