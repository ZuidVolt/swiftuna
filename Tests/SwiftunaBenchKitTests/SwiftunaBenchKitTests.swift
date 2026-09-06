import Testing

@testable import SwiftunaBenchKit

@Suite("Bench statistics")
struct BenchStatsTests {
    @Test("Summary statistics, IQR trimming, and calibrated gate math")
    func testBenchStatisticsMathAndTrimming() {
        let odd = computeStats([3.0, 1.0, 2.0])
        #expect(odd.median == 2.0)
        #expect(abs(odd.mean - 2.0) < 1e-12)
        #expect(odd.min == 1.0 && odd.max == 3.0 && odd.n == 3)

        let even = computeStats([4.0, 1.0, 3.0, 2.0])
        #expect(even.median == 2.5)

        let (trimmed, count) = trimmedStats([10.0, 10.1, 9.9, 10.2, 10.0, 50.0])
        #expect(count == 1)
        #expect(abs(trimmed.median - 10.05) < 0.1)

        let (_, refused) = trimmedStats([1.0, 100.0])
        #expect(refused == 0)

        #expect(calibratedGate(baselineCV: 0.01, floor: 0.05) == 0.05)
        #expect(calibratedGate(baselineCV: 0.04, floor: 0.05) == 0.08)
    }
}

@Suite("Gate verdicts")
struct GateVerdictTests {
    @Test("Gate verdicts: pass, regression, improvements, and inconclusive cases")
    func testGateVerdictEvaluation() {
        #expect(judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 103, floor: 0.05).status == .pass)
        #expect(judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 110, floor: 0.05).status == .regression)
        #expect(judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 40, floor: 0.05).status == .pass)
        #expect(judgeMetric(name: "m", baselineMedian: 0, baselineCV: 0, currentMedian: 5, floor: 0.05).status == .inconclusive)
        #expect(judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.30, currentMedian: 101, floor: 0.05).status == .inconclusive)
    }
}

@Suite("CLI parsing")
struct CLIParsingTests {
    @Test("CLI arguments: commands, flags, forms, and parse errors")
    func testCommandLineArgumentParsing() {
        let runEq = parseBenchArgs(["run", "--suite=hot", "--reps=3"])
        #expect(runEq.command == .run && runEq.suite == "hot" && runEq.reps == 3 && runEq.suiteSet)

        let runSp = parseBenchArgs(["run", "--suite", "scale", "--reps", "7"])
        #expect(runSp.suite == "scale" && runSp.reps == 7)

        let cmp = parseBenchArgs(["compare", "main", "feat", "--suite", "hot"])
        #expect(cmp.command == .compare && cmp.refA == "main" && cmp.refB == "feat")

        #expect(parseBenchArgs(["compare", "--suite", "hot"]).parseError != nil)
        #expect(parseBenchArgs(["run", "--frobnicate"]).parseError != nil)
    }
}

@Suite("Result codec")
struct ResultCodecTests {
    @Test("SuiteResult round-trips through JSON")
    func testSuiteResultJsonRoundTrip() throws {
        let r = SuiteResult(
            suite: "hot", branch: "b", commit: "c",
            metrics: [MetricResult(name: "m", unit: "us", median: 1.5, mean: 1.6,
                                   stdev: 0.1, cv: 0.06, n: 5)],
            environment: ["load1": "0.5"])
        let back = try decodeResult(from: encodeResult(r))
        #expect(back.suite == "hot" && back.metrics.count == 1)
        #expect(back.metrics[0].median == 1.5 && back.metrics[0].n == 5)
    }
}

@Suite("Diagnostics engine")
struct DiagnosticsTests {
    @Test("Diagnostics classification, rendering, hints, and task CPU inspection")
    func testDiagnosticsClassificationAndRendering() {
        let dConflict = classify(BenchError.commandFailed(
            command: "/usr/bin/git worktree add --detach wt abc123",
            status: 128, stderrTail: "fatal: 'wt' already exists"))
        #expect(dConflict.code == .worktreeConflict && dConflict.helps.contains(where: { $0.contains("prune") }))

        let dStale = classify(BenchError.commandFailed(
            command: "/usr/bin/git worktree remove --force wt",
            status: 128, stderrTail: "fatal: 'wt' is not a working tree"))
        #expect(dStale.code == .staleWorktree && !dStale.helps.isEmpty)

        let dBuild = classify(BenchError.commandFailed(
            command: "/usr/bin/swift build -c release", status: 1, stderrTail: "error: foo"))
        #expect(dBuild.code == .buildFailed && dBuild.helps.contains(where: { $0.contains("--fresh") }))

        let dLoad = classify(BenchError.loadedEnvironment("1m load 19.5 exceeds 10 cores"))
        #expect(dLoad.code == .loadedMachine && dLoad.helps.contains(where: { $0.contains("--require-quiet") }))

        struct Weird: Error {}
        let dUnknown = classify(Weird())
        #expect(dUnknown.code == nil && !dUnknown.helps.isEmpty)

        let rendered = Diagnostic(level: "error", code: .highJitter, message: "jitter!",
                                  labels: ["evidence"], notes: ["context"], helps: ["fix it"]).render(useColor: false)
        #expect(rendered.contains("error:") && rendered.contains("[BENCH010]") && rendered.contains("= help: fix it"))

        let v = MetricVerdict(metric: "m", status: .inconclusive, delta: 0, gate: 0.05, reason: "x")
        #expect(hintsForVerdicts([v], load1: 19.5, cores: 10).count == 2)
        #expect(hintsForVerdicts([], load1: 1.0, cores: 10).isEmpty)
        #expect(hintsForVerdicts([], load1: 1.0, cores: 10, lowPower: true).first?.contains("Low Power Mode") == true)

        let cpu = taskCPUSeconds()
        #expect(cpu == -1 || cpu >= 0)
    }
}
