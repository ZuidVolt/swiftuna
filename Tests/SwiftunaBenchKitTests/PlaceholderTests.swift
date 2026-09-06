import Testing

@testable import SwiftunaBenchKit

@Suite("Bench statistics")
struct BenchStatsTests {
    @Test("Median-first summary on odd samples")
    func computeStatsOdd() {
        let s = computeStats([3.0, 1.0, 2.0])
        #expect(s.median == 2.0)
        #expect(abs(s.mean - 2.0) < 1e-12)
        #expect(s.min == 1.0 && s.max == 3.0 && s.n == 3)
    }

    @Test("Even samples average the middle two")
    func computeStatsEven() {
        let s = computeStats([4.0, 1.0, 3.0, 2.0])
        #expect(s.median == 2.5)
    }

    @Test("IQR trim removes spikes but keeps the floor")
    func trimmedStatsOutlier() {
        let (s, trimmed) = trimmedStats([10.0, 10.1, 9.9, 10.2, 10.0, 50.0])
        #expect(trimmed == 1)
        #expect(abs(s.median - 10.05) < 0.1)
    }

    @Test("Trim refusal keeps everything when the rig is broken")
    func trimmedStatsRefusal() {
        let (_, trimmed) = trimmedStats([1.0, 100.0])
        #expect(trimmed == 0)
    }

    @Test("Gate is twice CV with a floor")
    func calibratedGateMath() {
        #expect(calibratedGate(baselineCV: 0.01, floor: 0.05) == 0.05)
        #expect(calibratedGate(baselineCV: 0.04, floor: 0.05) == 0.08)
    }
}

@Suite("Gate verdicts")
struct GateVerdictTests {
    @Test("Within gate passes")
    func withinGate() {
        let v = judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 103, floor: 0.05)
        #expect(v.status == .pass && v.gate == 0.05)
    }

    @Test("Regression fails")
    func regression() {
        let v = judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 110, floor: 0.05)
        #expect(v.status == .regression)
    }

    @Test("Improvements always pass")
    func improvement() {
        let v = judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.01, currentMedian: 40, floor: 0.05)
        #expect(v.status == .pass)
    }

    @Test("Zero baseline is inconclusive, never a false pass")
    func zeroBaseline() {
        let v = judgeMetric(name: "m", baselineMedian: 0, baselineCV: 0, currentMedian: 5, floor: 0.05)
        #expect(v.status == .inconclusive)
    }

    @Test("Jittery baseline is inconclusive")
    func jittery() {
        let v = judgeMetric(name: "m", baselineMedian: 100, baselineCV: 0.30, currentMedian: 101, floor: 0.05)
        #expect(v.status == .inconclusive)
    }
}

@Suite("CLI parsing")
struct CLIParsingTests {
    @Test("Defaults and = form")
    func defaults() {
        let c = parseBenchArgs(["run", "--suite=hot", "--reps=3"])
        #expect(c.command == .run && c.suite == "hot" && c.reps == 3 && c.suiteSet)
    }

    @Test("Space-separated values")
    func spaceForm() {
        let c = parseBenchArgs(["run", "--suite", "scale", "--reps", "7"])
        #expect(c.suite == "scale" && c.reps == 7)
    }

    @Test("Compare consumes two refs")
    func compareRefs() {
        let c = parseBenchArgs(["compare", "main", "feat", "--suite", "hot"])
        #expect(c.command == .compare && c.refA == "main" && c.refB == "feat")
    }

    @Test("Compare without refs is a parse error")
    func compareMissing() {
        let c = parseBenchArgs(["compare", "--suite", "hot"])
        #expect(c.parseError != nil)
    }

    @Test("Unknown flags are reported")
    func unknownFlag() {
        let c = parseBenchArgs(["run", "--frobnicate"])
        #expect(c.parseError != nil)
    }
}

@Suite("Result codec")
struct ResultCodecTests {
    @Test("SuiteResult round-trips through JSON")
    func roundTrip() throws {
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
    @Test("Worktree collision classifies with prune help")
    func worktreeConflict() {
        let d = classify(BenchError.commandFailed(
            command: "/usr/bin/git worktree add --detach wt abc123",
            status: 128, stderrTail: "fatal: 'wt' already exists"))
        #expect(d.code == .worktreeConflict)
        #expect(d.helps.contains(where: { $0.contains("prune") }))
    }

    @Test("Stale worktree classifies with delete help")
    func staleWorktree() {
        let d = classify(BenchError.commandFailed(
            command: "/usr/bin/git worktree remove --force wt",
            status: 128, stderrTail: "fatal: 'wt' is not a working tree"))
        #expect(d.code == .staleWorktree)
        #expect(!d.helps.isEmpty)
    }

    @Test("Build failure carries toolchain help")
    func buildFailed() {
        let d = classify(BenchError.commandFailed(
            command: "/usr/bin/swift build -c release", status: 1, stderrTail: "error: foo"))
        #expect(d.code == .buildFailed)
        #expect(d.helps.contains(where: { $0.contains("--fresh") }))
    }

    @Test("Loaded environment refuses with reason")
    func loadedEnv() {
        let d = classify(BenchError.loadedEnvironment("1m load 19.5 exceeds 10 cores"))
        #expect(d.code == .loadedMachine)
        #expect(d.helps.contains(where: { $0.contains("--require-quiet") }))
    }

    @Test("Unknown errors degrade to a plain diagnostic")
    func unknownPassthrough() {
        struct Weird: Error {}
        let d = classify(Weird())
        #expect(d.code == nil && !d.helps.isEmpty)
    }

    @Test("Render puts code, message, and help on separate lines")
    func renderShape() {
        let d = Diagnostic(level: "error", code: .highJitter, message: "jitter!",
                           labels: ["evidence"], notes: ["context"], helps: ["fix it"])
        let text = d.render(useColor: false)
        #expect(text.contains("error:"))
        #expect(text.contains("[BENCH010]"))
        #expect(text.contains("= help: fix it"))
        #expect(!text.contains("\u{001B}"))
    }

    @Test("Hints fire on inconclusive verdicts and loaded machines")
    func verdictHints() {
        let v = MetricVerdict(metric: "m", status: .inconclusive, delta: 0, gate: 0.05, reason: "x")
        let h = hintsForVerdicts([v], load1: 19.5, cores: 10)
        #expect(h.count == 2)
        #expect(hintsForVerdicts([], load1: 1.0, cores: 10).isEmpty)
    }

    @Test("Low Power Mode adds its own hint")
    func lowPowerHint() {
        let h = hintsForVerdicts([], load1: 1.0, cores: 10, lowPower: true)
        #expect(h.count == 1)
        #expect(h[0].contains("Low Power Mode"))
    }

    @Test("Task CPU time is non-negative where supported")
    func taskCPU() {
        let cpu = taskCPUSeconds()
        #expect(cpu == -1 || cpu >= 0) // -1 = unsupported platform
    }
}
