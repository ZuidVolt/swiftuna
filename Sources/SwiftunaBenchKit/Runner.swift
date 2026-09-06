import Foundation

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// Bench rig failures: environment too noisy, or a suite body threw.
public enum BenchError: Error, Sendable, CustomStringConvertible {
    case loadedEnvironment(String)
    case suiteFailed(String)
    /// A shelled-out command failed. Structured (not a string) so the
    /// diagnostics engine can classify it into actionable hints.
    case commandFailed(command: String, status: Int32, stderrTail: String)

    public var description: String {
        switch self {
        case .loadedEnvironment(let m): return "Environment too noisy: \(m)"
        case .suiteFailed(let m): return "Suite failed: \(m)"
        case .commandFailed(let c, let s, _): return "`\(c)` exited \(s)"
        }
    }
}

/// Machine state snapshot, stored with every result file for auditability.
public struct EnvironmentSnapshot: Sendable {
    public var load1: Double
    public var load5: Double
    public var thermal: String
    public var cores: Int
    public var lowPower: Bool

    public func asDictionary() -> [String: String] {
        [
            "load1": String(format: "%.2f", load1),
            "load5": String(format: "%.2f", load5),
            "thermal": thermal,
            "cores": "\(cores)",
            "low_power": lowPower ? "true" : "false",
        ]
    }
}

/// Current 1-minute load average and core count, for verdict hints.
/// (-1 load when unreadable.)
public func systemLoad() -> (load1: Double, cores: Int) {
    var avg = [Double](repeating: 0, count: 3)
    let got = avg.withUnsafeMutableBufferPointer { buf in
        getloadavg(buf.baseAddress, 3)
    }
    return (got == 3 ? avg[0] : -1, ProcessInfo.processInfo.activeProcessorCount)
}

/// Whole-process CPU seconds (user + system) via task_info. Unlike wall
/// time this excludes descheduling, so wall/cpu divergence in a result file
/// means the process was preempted, not that the code was slow. Follows the
/// thread across pool hops (suites await), which per-thread clocks cannot.
/// -1 where unsupported.
public func taskCPUSeconds() -> Double {
#if canImport(Darwin)
    var info = task_basic_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
        + Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
#else
    return -1
#endif
}

/// Low Power Mode throttles exactly the clocks we measure. macOS-only check;
/// everywhere else reports false.
public func isLowPowerMode() -> Bool {
#if canImport(Darwin)
    if #available(macOS 12.0, *) {
        return ProcessInfo.processInfo.isLowPowerModeEnabled
    }
#endif
    return false
}

/// Best-effort core pinning (macOS): QoS plus affinity tag. Reduces
/// migration jitter on short probes. No-op where unsupported.
public func pinCurrentThread() {
#if canImport(Darwin)
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
    #endif
}

/// Reads load and thermal state. When `requireQuiet` is set, throws instead
/// of measuring on a loaded or hot machine — sub-microsecond numbers taken
/// at load 19 are fiction, and the tool should say so up front.
public func checkEnvironment(requireQuiet: Bool) throws -> EnvironmentSnapshot {
    var avg = [Double](repeating: 0, count: 3)
    let got = avg.withUnsafeMutableBufferPointer { buf in
        getloadavg(buf.baseAddress, 3)
    }
    let load1 = got == 3 ? avg[0] : -1
    let load5 = got == 3 ? avg[1] : -1
    let cores = ProcessInfo.processInfo.activeProcessorCount
    let thermal: String
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: thermal = "nominal"
    case .fair: thermal = "fair"
    case .serious: thermal = "serious"
    case .critical: thermal = "critical"
    @unknown default: thermal = "unknown"
    }
    if requireQuiet {
        if load1 >= 0, load1 > Double(cores) {
            throw BenchError.loadedEnvironment("1m load \(String(format: "%.1f", load1)) exceeds \(cores) cores")
        }
        if thermal == "serious" || thermal == "critical" {
            throw BenchError.loadedEnvironment("thermal state is \(thermal)")
        }
        if isLowPowerMode() {
            throw BenchError.loadedEnvironment("Low Power Mode is enabled (clocks throttled)")
        }
    }
    return EnvironmentSnapshot(load1: load1, load5: load5, thermal: thermal, cores: cores,
                               lowPower: isLowPowerMode())
}

/// Runs `body` `pairs` times, discarded. Warms caches, branch predictors,
/// and the Rust allocator before anything is measured.
public func warmup(pairs: Int, body: () throws -> Void) rethrows {
    for _ in 0..<pairs {
        try body()
    }
}

/// Measures `body` `reps` times with an optional cooldown between reps so
/// one rep's heat doesn't bleed into the next.
public func measureReps(reps: Int, cooldownSeconds: Double = 0, body: () throws -> Double) throws -> [Double] {
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        samples.append(try body())
        if cooldownSeconds > 0 {
            Thread.sleep(forTimeInterval: cooldownSeconds)
        }
    }
    return samples
}

/// Async variant for suites that await (coordinator benches).
public func measureRepsAsync(reps: Int, cooldownSeconds: Double = 0, body: () async throws -> Double) async throws
    -> [Double]
{
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        samples.append(try await body())
        if cooldownSeconds > 0 {
            try await Task.sleep(nanoseconds: UInt64(cooldownSeconds * 1_000_000_000))
        }
    }
    return samples
}

/// Runs one suite end to end: pin, environment check, measure, then stamp
/// branch/commit/environment onto the result. Suites only produce metrics.
///
/// macOS only for now: thread pinning, QoS classes, and several timing
/// assumptions have only been validated there. Other platforms get a clear
/// refusal (exit 2) instead of silently dubious numbers — Linux CI still
/// compiles and unit-tests BenchKit, it just cannot execute suites.
public func executeSuite(_ suite: any BenchSuite, ctx: SuiteContext) async throws -> SuiteResult {
    #if !os(macOS)
    throw BenchError.suiteFailed("benchmark suite '\(suite.name)' runs on macOS only for now")
    #else
    pinCurrentThread()
    let env = try checkEnvironment(requireQuiet: ctx.requireQuiet)
    let wallStart = DispatchTime.now()
    let cpuStart = taskCPUSeconds()
    var result = try await suite.run(ctx)
    let wallS = Double(DispatchTime.now().uptimeNanoseconds - wallStart.uptimeNanoseconds) / 1_000_000_000.0
    let cpuS = taskCPUSeconds() - cpuStart
    let info = currentGitInfo()
    result.branch = info.branch
    result.commit = info.commit
    var envDict = env.asDictionary()
    envDict["wall_s"] = String(format: "%.2f", wallS)
    if cpuS >= 0 {
        envDict["cpu_s"] = String(format: "%.2f", cpuS)
    }
    result.environment = envDict
    return result
    #endif
}

/// Best-effort git identity for result files. "unknown" when not in a repo.
public func currentGitInfo() -> (branch: String, commit: String) {
    func run(_ args: String...) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = args
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return nil }
        p.waitUntilExit()
        guard p.terminationStatus == 0 else { return nil }
        return String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return (run("branch", "--show-current") ?? "unknown", run("rev-parse", "--short", "HEAD") ?? "unknown")
}
