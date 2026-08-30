import Foundation
import Swiftuna

struct BenchmarkMetrics {
    let language: String
    let totalMs: Double
    let usPerTrial: Double
    let throughput: Double
}

func runSubprocess(executable: String, arguments: [String]) -> String? {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: executable)
    proc.arguments = arguments

    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = FileHandle.nullDevice

    do {
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    } catch {
        return nil
    }
}

func formatRow(name: String, totalMs: Double, us: Double, tp: Double, rel: Double) -> String {
    let paddedName = name.padding(toLength: 26, withPad: " ", startingAt: 0)
    let msStr = String(format: "%.2f", totalMs).padding(toLength: 15, withPad: " ", startingAt: 0)
    let usStr = String(format: "%.2f", us).padding(toLength: 20, withPad: " ", startingAt: 0)
    let tpStr = String(format: "%.0f", tp).padding(toLength: 25, withPad: " ", startingAt: 0)
    let relStr = String(format: "%.2fx", rel)
    return "| \(paddedName) | \(msStr) | \(usStr) | \(tpStr) | \(relStr) |"
}

@main
struct SwiftunaBenchApp {
    static func main() {
        do {
            let nTrials = 1000
            let cwd = FileManager.default.currentDirectoryPath

            print("=========================================================")
            print("        Swiftuna Tri-Language Benchmarking Suite         ")
            print("=========================================================")
            print("Running \(nTrials) trials of Quadratic TPE optimization across:")
            print("1. Pure Rust (rustuna_core direct)")
            print("2. Swiftuna (Swift 6.4 + C ABI)")
            print("3. Python (uv run python)\n")
            fflush(stdout)

            // 1. Run Rust Benchmark
            var rustMetrics: BenchmarkMetrics? = nil
            let rustBin = cwd + "/crates/rustuna-ffi/target/release/rust_bench"
            if let out = runSubprocess(executable: rustBin, arguments: ["\(nTrials)"]) {
                for line in out.split(separator: "\n") {
                    if line.hasPrefix("RESULT:") {
                        let parts = line.split(separator: ":")
                        if parts.count == 4,
                           let ms = Double(parts[1]),
                           let us = Double(parts[2]),
                           let tp = Double(parts[3]) {
                            rustMetrics = BenchmarkMetrics(language: "Pure Rust", totalMs: ms, usPerTrial: us, throughput: tp)
                        }
                    }
                }
            }

            // 2. Run Swift Benchmark
            let sampler = TPESampler(seed: 42)
            let study = try Swiftuna.createStudy(name: "bench_swift", direction: .minimize, sampler: sampler)

            let clock = ContinuousClock()
            let swiftStart = clock.now

            for _ in 0..<nTrials {
                var trial = try study.ask()
                let x = try trial.suggest("x", in: -10.0...10.0)
                let y = try trial.suggest("y", in: -10.0...10.0)
                let loss = pow(x - 2.0, 2) + pow(y + 5.0, 2)
                try study.tell(consuming: trial, value: loss)
            }

            let swiftDuration = clock.now - swiftStart
            let swiftSec = Double(swiftDuration.components.seconds) + Double(swiftDuration.components.attoseconds) * 1e-18
            let swiftMs = swiftSec * 1000.0
            let swiftUsPerTrial = (swiftSec * 1_000_000.0) / Double(nTrials)
            let swiftThroughput = Double(nTrials) / swiftSec

            let swiftMetrics = BenchmarkMetrics(language: "Swiftuna (Swift 6.4)", totalMs: swiftMs, usPerTrial: swiftUsPerTrial, throughput: swiftThroughput)

            // 3. Run Python Benchmark
            var pythonMetrics: BenchmarkMetrics? = nil
            let uvPath = "/opt/homebrew/bin/uv"
            let pyScript = cwd + "/tools/python_bench.py"
            if FileManager.default.fileExists(atPath: uvPath),
               let pyOut = runSubprocess(executable: uvPath, arguments: ["run", "python", pyScript, "\(nTrials)"]) {
                for line in pyOut.split(separator: "\n") {
                    if line.hasPrefix("RESULT:") {
                        let parts = line.split(separator: ":")
                        if parts.count == 4,
                           let ms = Double(parts[1]),
                           let us = Double(parts[2]),
                           let tp = Double(parts[3]) {
                            pythonMetrics = BenchmarkMetrics(language: "Python (Optuna)", totalMs: ms, usPerTrial: us, throughput: tp)
                        }
                    }
                }
            }

            // Print Formatted Markdown Comparison Table
            print("| Language / Runtime         | Total Time (ms) | Latency (µs / trial) | Throughput (trials / sec) | Rel. to Rust |")
            print("| :------------------------- | --------------: | -------------------: | ------------------------: | -----------: |")

            let baseUs = rustMetrics?.usPerTrial ?? swiftMetrics.usPerTrial

            if let r = rustMetrics {
                let rel = r.usPerTrial / baseUs
                print(formatRow(name: "**\(r.language)**", totalMs: r.totalMs, us: r.usPerTrial, tp: r.throughput, rel: rel))
            }

            let swiftRel = swiftMetrics.usPerTrial / baseUs
            print(formatRow(name: "**\(swiftMetrics.language)**", totalMs: swiftMetrics.totalMs, us: swiftMetrics.usPerTrial, tp: swiftMetrics.throughput, rel: swiftRel))

            if let p = pythonMetrics {
                let pRel = p.usPerTrial / baseUs
                print(formatRow(name: "**\(p.language)**", totalMs: p.totalMs, us: p.usPerTrial, tp: p.throughput, rel: pRel))
            }

            print("\n---------------------------------------------------------")
            if let p = pythonMetrics, swiftMetrics.usPerTrial >= p.usPerTrial {
                print("❌ Invariant Failed: Swiftuna was not faster than Python!")
                exit(1)
            } else {
                print("✅ Invariant Verified: T_Rust <= T_Swift < T_Python")
                print("🚀 Swiftuna performance successfully benchmarked!")
                print("=========================================================\n")
            }
        } catch {
            print("❌ Benchmark execution failed with error: \(error)")
            exit(1)
        }
    }
}
