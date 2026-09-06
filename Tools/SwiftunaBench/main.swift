import Foundation
import SwiftunaBenchKit
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@main
struct SwiftunaBenchApp {
    static func main() async {
        let cfg = parseBenchArgs(Array(CommandLine.arguments.dropFirst()))
        if cfg.showHelp || CommandLine.arguments.count == 1 {
            print(benchHelp())
            return
        }
        if let err = cfg.parseError {
            fputs("swiftunabench: \(err)\n", stderr)
            exit(BenchExit.inconclusive.rawValue)
        }
        switch cfg.command {
        case .listSuites:
            listSuites(json: cfg.json)
        case .run:
            guard let suite = SuiteRegistry.named(cfg.suite) else {
                fputs("swiftunabench: unknown suite '\(cfg.suite)'. See list-suites.\n", stderr)
                exit(BenchExit.inconclusive.rawValue)
            }
            await runSuite(suite: suite, cfg: cfg)
        case .compare:
            do {
                let code = try await compareRefsAsync(refA: cfg.refA, refB: cfg.refB, suite: cfg.suiteSet ? cfg.suite : nil,
                                                      reps: cfg.reps, floor: cfg.gateFloor, json: cfg.json, fresh: cfg.fresh,
                                                      force: cfg.force)
                exit(code.rawValue)
            } catch {
                fputs(classify(error).render(useColor: isatty(STDOUT_FILENO) != 0) + "\n", stderr)
                exit(BenchExit.inconclusive.rawValue)
            }
        case .gate:
            gateFiles(cfg: cfg)
        }
    }

    static func runSuite(suite: any BenchSuite, cfg: BenchConfig) async {
        do {
            let ctx = SuiteContext(reps: cfg.reps, requireQuiet: cfg.requireQuiet)
            let result = try await executeSuite(suite, ctx: ctx)
            if cfg.json {
                do {
                    print(try encodeResult(result))
                } catch {
                    fputs(classify(error).render(useColor: isatty(STDOUT_FILENO) != 0) + "\n", stderr)
                    exit(BenchExit.inconclusive.rawValue)
                }
            } else {
                print(renderSuiteResult(result, useColor: isatty(STDOUT_FILENO) != 0))
            }
        } catch {
            fputs(classify(error).render(useColor: isatty(STDOUT_FILENO) != 0) + "\n", stderr)
            exit(BenchExit.inconclusive.rawValue)
        }
    }
    static func listSuites(json: Bool) {
        let suites = SuiteRegistry.suites
        if json {
            let entries = suites.map {
                SuiteListing(name: $0.name, description: $0.description, requiresNewAPI: $0.requiresNewAPI, metrics: $0.metricNames)
            }
            if let data = try? JSONEncoder().encode(entries),
               let s = String(data: data, encoding: .utf8)
            {
                print(s)
            }
        } else if suites.isEmpty {
            print("No suites registered yet.")
        } else {
            for s in suites {
                print("\(s.name) — \(s.description)\(s.requiresNewAPI ? " [new-api]" : "")")
            }
        }
    }

    /// gate --baseline=<f> --current=<f>: judge two result files.
    /// Exit 0 pass, 1 regression, 2 inconclusive.
    static func gateFiles(cfg: BenchConfig) {
        guard !cfg.baselineFile.isEmpty, !cfg.currentFile.isEmpty else {
            fputs("swiftunabench: gate needs --baseline=<f> --current=<f>\n", stderr)
            exit(BenchExit.inconclusive.rawValue)
        }
        do {
            let baseText = try String(contentsOfFile: cfg.baselineFile, encoding: .utf8)
            let curText = try String(contentsOfFile: cfg.currentFile, encoding: .utf8)
            let base = try decodeResult(from: baseText)
            let cur = try decodeResult(from: curText)
            var verdicts: [MetricVerdict] = []
            if base.suite != cur.suite {
                fputs("swiftunabench: suite mismatch '\(base.suite)' vs '\(cur.suite)'\n", stderr)
                exit(BenchExit.inconclusive.rawValue)
            }
            let curByName = Dictionary(uniqueKeysWithValues: cur.metrics.map { ($0.name, $0) })
            for b in base.metrics {
                guard let c = curByName[b.name] else {
                    verdicts.append(MetricVerdict(metric: b.name, status: .inconclusive, delta: 0,
                                                  gate: cfg.gateFloor, reason: "missing from current file"))
                    continue
                }
                verdicts.append(judgeMetric(name: b.name, baselineMedian: b.median, baselineCV: b.cv,
                                            currentMedian: c.median, floor: cfg.gateFloor))
            }
            if cfg.json {
                let worst = worstStatus(verdicts.map { $0.status })
                print("{\"suite\": \"\(base.suite)\", \"status\": \(worst.rawValue)}")
                exit(worst.rawValue)
            }
            print(renderVerdicts(verdicts, useColor: isatty(STDOUT_FILENO) != 0))
            let load = base.environment["load1"].flatMap(Double.init)
            let cores = base.environment["cores"].flatMap(Int.init)
            let lowPower = base.environment["low_power"] == "true"
            for h in hintsForVerdicts(verdicts, load1: load, cores: cores, lowPower: lowPower) {
                print("  \(h)")
            }
            exit(worstStatus(verdicts.map { $0.status }).rawValue)
        } catch {
            fputs(classify(error).render(useColor: isatty(STDOUT_FILENO) != 0) + "\n", stderr)
            exit(BenchExit.inconclusive.rawValue)
        }
    }
}
