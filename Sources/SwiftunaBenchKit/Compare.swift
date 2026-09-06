import Foundation

/// One entry of list-suites output. Public so the compare tool can decode
/// listings from binaries built out of other branches.
public struct SuiteListing: Sendable, Codable {
    public var name: String
    public var description: String
    public var requiresNewAPI: Bool
    public var metrics: [String]

    public init(name: String, description: String, requiresNewAPI: Bool, metrics: [String]) {
        self.name = name
        self.description = description
        self.requiresNewAPI = requiresNewAPI
        self.metrics = metrics
    }
}

/// How a side was built (for the report).
public struct CompareSide: Sendable {
    public var ref: String
    public var binary: String
    public var suites: [SuiteListing]
}

/// Shells out. Returns trimmed stdout; throws on nonzero exit.
@discardableResult
public func shell(_ executable: String, _ args: [String], cwd: String) throws -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: executable)
    p.arguments = args
    p.currentDirectoryURL = URL(fileURLWithPath: cwd)
    let out = Pipe()
    let err = Pipe()
    p.standardOutput = out
    p.standardError = err
    try p.run()
    p.waitUntilExit()
    let text = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    if p.terminationStatus != 0 {
        let tail = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        throw BenchError.commandFailed(command: ([executable] + args).joined(separator: " "),
                                       status: p.terminationStatus,
                                       stderrTail: String(tail.suffix(500)))
    }
    return text.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Repo root of the checkout this tool runs from.
public func repoRoot() throws -> String {
    try shell("/usr/bin/git", ["rev-parse", "--show-toplevel"], cwd: FileManager.default.currentDirectoryPath)
}

func sanitizeRef(_ ref: String) -> String {
    let ok = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
    return ref.unicodeScalars.map { ok.contains($0) ? String($0) : "_" }.joined()
}

/// Timestamped phase logging to stderr. Compare runs for many minutes;
/// silent stretches look hung, so every slow step announces itself.
public func benchLog(_ message: String) {
    let timestamp = Date.now.formatted(date: .omitted, time: .standard)
    fputs("[\(timestamp)] \(message)\n", stderr)
}

/// Prepares a worktree for `ref` and returns a bench binary path.
/// Branches that already ship BenchKit build in-tree; older ones get an
/// overlay of the current BenchKit sources, minus new-API suites up front.
/// Worktrees and binaries are reused across runs unless `fresh` is set.
func prepareSide(ref: String, root: String, fresh: Bool) throws -> CompareSide {
    let name = sanitizeRef(ref)
    let wt = "\(root)/.bench-worktrees/\(name)"
    let fm = FileManager.default
    // Prune first: a killed run can leave a registered-but-missing worktree
    // behind (plain rm never unregisters), which makes add fail.
    _ = try? shell("/usr/bin/git", ["worktree", "prune"], cwd: root)
    // Detached by commit: the ref may already be checked out (e.g. the
    // working tree itself), and worktrees cannot share a branch.
    let sha = try shell("/usr/bin/git", ["rev-parse", ref], cwd: root)
    // Reuse the worktree when it already sits at this commit (skip the
    // checkout, not the build: sources may have changed since, and an
    // incremental rebuild is cheap next to a from-scratch one). No
    // cleanliness check: the only untracked content in here is our own
    // .bench-overlay, which buildOverlay refreshes every run.
    let reuseWorktree = !fresh && fm.fileExists(atPath: wt)
        && (try? shell("/usr/bin/git", ["rev-parse", "HEAD"], cwd: wt)) == sha
    if reuseWorktree {
        benchLog("reusing \(ref) worktree (use --fresh to rebuild from scratch)")
    } else {
        if fm.fileExists(atPath: wt) {
            benchLog("removing stale worktree for \(ref)...")
            do {
                try shell("/usr/bin/git", ["worktree", "remove", "--force", wt], cwd: root)
            } catch {
                // Not a registered working tree (aborted run left a plain
                // directory): our scratch dir, safe to delete outright.
                try fm.removeItem(atPath: wt)
            }
        }
        benchLog("worktree add \(ref) (\(String(sha.prefix(8))))...")
        try shell("/usr/bin/git", ["worktree", "add", "--detach", wt, sha], cwd: root)
    }

    let inTreeKit = "\(wt)/Sources/SwiftunaBenchKit"
    let binary: String
    if fm.fileExists(atPath: inTreeKit) {
        benchLog("building in-tree SwiftunaBench for \(ref)...")
        try shell("/usr/bin/swift", ["build", "-c", "release", "--package-path", wt, "--product", "SwiftunaBench"], cwd: root)
        binary = "\(wt)/.build/release/SwiftunaBench"
    } else {
        benchLog("building overlay SwiftunaBench for \(ref)...")
        binary = try buildOverlay(worktree: wt, packageName: name, root: root)
    }
    benchLog("listing suites on \(ref)...")
    return try listSide(ref: ref, binary: binary, root: root)
}

private func listSide(ref: String, binary: String, root: String) throws -> CompareSide {
    let listingJSON = try shell(binary, ["list-suites", "--json"], cwd: root)
    guard let data = listingJSON.data(using: .utf8),
          let suites = try? JSONDecoder().decode([SuiteListing].self, from: data)
    else {
        throw BenchError.suiteFailed("list-suites undecodable on \(ref)")
    }
    return CompareSide(ref: ref, binary: binary, suites: suites)
}

/// New-API suite files, dropped from the overlay up front: old branches
/// lack the Swift APIs (ScaleNewAPI) and the FFI symbols (FFIMicro) they
/// need, so attempting them first would only burn a full failed build.
private let newAPISuiteFiles = ["ScaleNewAPI.swift", "FFIMicro.swift"]

/// Builds an overlay package inside an old worktree: current BenchKit
/// sources (old-API suites only) against the worktree's Swiftuna.
func buildOverlay(worktree wt: String, packageName: String, root: String) throws -> String {
    let fm = FileManager.default
    let overlay = "\(wt)/.bench-overlay"
    // Refresh sources but keep .build: incremental rebuilds stay fast
    // across runs while never going stale.
    try? fm.removeItem(atPath: "\(overlay)/Sources")
    try? fm.removeItem(atPath: "\(overlay)/Package.swift")
    let srcDir = "\(overlay)/Sources/OverlayBench"
    try fm.createDirectory(atPath: srcDir, withIntermediateDirectories: true)

    let kit = "\(root)/Sources/SwiftunaBenchKit"
    for file in (try fm.contentsOfDirectory(atPath: kit)) {
        if file.hasSuffix(".swift") {
            try fm.copyItem(atPath: "\(kit)/\(file)", toPath: "\(srcDir)/\(file)")
        }
    }
    for file in (try fm.contentsOfDirectory(atPath: "\(kit)/Suites")) {
        if newAPISuiteFiles.contains(file) { continue }
        try fm.copyItem(atPath: "\(kit)/Suites/\(file)", toPath: "\(srcDir)/\(file)")
    }
    // Executable entry: dispatch straight into the registry (no subcommands).
    try overlayMain().write(toFile: "\(srcDir)/OverlayMain.swift", atomically: true, encoding: .utf8)
    try overlayPackage(packageName: packageName, worktree: wt).write(
        toFile: "\(overlay)/Package.swift", atomically: true, encoding: .utf8)

    // Registry references the dropped suites; strip those lines.
    let reg = "\(srcDir)/SuiteRegistry.swift"
    if var text = try? String(contentsOfFile: reg, encoding: .utf8) {
        text = text.replacingOccurrences(of: "ScaleNewAPISuite(), ", with: "")
        text = text.replacingOccurrences(of: "FFIMicroSuite(), ", with: "")
        text = text.replacingOccurrences(of: ", ScaleNewAPISuite()", with: "")
        text = text.replacingOccurrences(of: ", FFIMicroSuite()", with: "")
        try text.write(toFile: reg, atomically: true, encoding: .utf8)
    }
    try shell("/usr/bin/swift", ["build", "-c", "release", "--package-path", overlay], cwd: root)
    return "\(overlay)/.build/release/OverlayBench"
}

private func overlayMain() -> String {
    """
    #if canImport(Darwin)
    import Darwin
    #elseif canImport(Glibc)
    import Glibc
    #endif
    import Foundation
    import Swiftuna
    import SwiftunaDistributed

    // NOTE: no `import SwiftunaBenchKit` — the overlay compiles the kit
    // sources into this same module (old branches have no such product).

    @main
    struct OverlayBench {
        static func main() async {
            let args = Array(CommandLine.arguments.dropFirst())
            if args == ["list-suites", "--json"] {
                let entries = SuiteRegistry.suites.map {
                    SuiteListing(name: $0.name, description: $0.description,
                                 requiresNewAPI: $0.requiresNewAPI, metrics: $0.metricNames)
                }
                if let data = try? JSONEncoder().encode(entries),
                   let s = String(data: data, encoding: .utf8)
                {
                    print(s)
                    return
                }
                fputs("encode failed\\n", stderr)
                exit(2)
            }
            guard args.count >= 3, args[0] == "run-once", args[1] == "--suite" else {
                fputs("usage: OverlayBench [list-suites --json | run-once --suite <name>]\\n", stderr)
                exit(2)
            }
            let name = args[2]
            guard let suite = SuiteRegistry.named(name) else {
                fputs("unknown suite \\(name)\\n", stderr)
                exit(2)
            }
            do {
                let result = try await executeSuite(suite, ctx: SuiteContext(reps: 1, requireQuiet: false))
                print(try encodeResult(result))
            } catch {
                fputs("suite failed: \\(error)\\n", stderr)
                exit(1)
            }
        }
    }
    """
}

private func overlayPackage(packageName: String, worktree: String) -> String {
    // Mirror of the root Package.swift settings: the overlay must compile
    // the worktree's Swift sources with IDENTICAL codegen, or engine
    // comparisons measure compiler flags instead of code. Keep in sync.
    let vendoredMac = "\(worktree)/Sources/LibRustuna/artifacts/macos-arm64"
    let vendoredLinuxAarch64 = "\(worktree)/Sources/LibRustuna/artifacts/linux-aarch64"
    return """
    // swift-tools-version: 6.3
    import PackageDescription

    let package = Package(
        name: "BenchOverlay",
        platforms: [.macOS(.v26)],
        dependencies: [.package(path: "..")],
        targets: [
            .executableTarget(
                name: "OverlayBench",
                dependencies: [
                    .product(name: "Swiftuna", package: "\(packageName)"),
                    .product(name: "SwiftunaDistributed", package: "\(packageName)"),
                ],
                path: "Sources",
                swiftSettings: [
                    .swiftLanguageMode(.v6),
                    .defaultIsolation(nil),
                    .enableUpcomingFeature("ExistentialAny"),
                    .enableUpcomingFeature("InternalImportsByDefault"),
                    .enableUpcomingFeature("MemberImportVisibility"),
                    .enableUpcomingFeature("InferIsolatedConformances"),
                    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
                    .enableUpcomingFeature("ImmutableWeakCaptures"),
                ],
                linkerSettings: [
                    .unsafeFlags(["-L\(vendoredMac)", "-lrustuna_ffi"], .when(platforms: [.macOS])),
                    .unsafeFlags(["-L\(vendoredLinuxAarch64)", "-lrustuna_ffi"], .when(platforms: [.linux])),
                ]
            ),
        ]
    )
    """
}

/// Runs one suite binary once (reps=1), returning the decoded result.
func runOnce(binary: String, suite: String, cwd: String, overlay: Bool) throws -> SuiteResult {
    let args = overlay ? ["run-once", "--suite", suite] : ["run", "--suite", suite, "--reps", "1", "--json"]
    let text = try shell(binary, args, cwd: cwd)
    // In-tree JSON mode prints only the result object; be lenient about
    // surrounding lines anyway by scanning for the leading brace.
    let json = text.firstIndex(of: "{").map { String(text[$0...]) } ?? text
    guard let jsonData = json.data(using: .utf8) else {
        throw BenchError.suiteFailed("non-UTF8 output from \(binary)")
    }
    return try JSONDecoder().decode(SuiteResult.self, from: jsonData)
}

/// Compares two refs on shared suites with cross-process interleaving.
/// First outer rep per side is dropped as warmup; order alternates per rep.
public func compareRefsAsync(
    refA: String,
    refB: String,
    suite requested: String?,
    reps: Int,
    floor: Double,
    json: Bool,
    fresh: Bool,
    force: Bool
) async throws -> BenchExit {
    let root = try repoRoot()
    _ = try checkEnvironment(requireQuiet: false)
    // Compare orchestrates from a swiftuna checkout: worktrees need its refs
    // and the overlay copies its BenchKit sources. Fail fast anywhere else.
    guard FileManager.default.fileExists(atPath: "\(root)/Sources/SwiftunaBenchKit") else {
        throw BenchError.suiteFailed("compare must run from a swiftuna checkout (no Sources/SwiftunaBenchKit under \(root))")
    }
    let shaA = try shell("/usr/bin/git", ["rev-parse", refA], cwd: root)
    let shaB = try shell("/usr/bin/git", ["rev-parse", refB], cwd: root)
    let headSha = try shell("/usr/bin/git", ["rev-parse", "HEAD"], cwd: root)
    let dirty = !(try shell("/usr/bin/git", ["status", "--porcelain"], cwd: root)).isEmpty

    // Same commit on both sides: without --force there is nothing to learn.
    // A dirty tree is the exception — it means something, so measure the
    // working tree against its own HEAD instead of refusing.
    if shaA == shaB, !force {
        if !dirty {
            let short = String(shaA.prefix(8))
            if json {
                print("{\"status\": 0, \"reason\": \"identical refs \(short)\"}")
            } else {
                print("cannot compare \(refA) to itself (both resolve to \(short)) — nothing to learn.")
                print("Pass --force to run it anyway.")
            }
            return .pass
        }
        benchLog("uncommitted changes detected: comparing working tree against HEAD (\(String(headSha.prefix(8))))")
        let base = try prepareSide(ref: refA, root: root, fresh: fresh)
        let work = try prepareWorkingTreeSide(label: "\(refB)+dirty", root: root)
        return try await judgeSides(sideA: base, sideB: work, overlayA: base.binary.hasSuffix("OverlayBench"),
                              overlayB: false, refA: refA, refB: "\(refB) (working tree)",
                              requested: requested, reps: reps, floor: floor, json: json, root: root)
    }

    var sideA = try prepareSide(ref: refA, root: root, fresh: fresh)
    var sideB = try prepareSide(ref: refB, root: root, fresh: fresh)
    var labelB = refB
    // A dirty tree only counts if one side IS the checkout: otherwise the
    // worktrees hold committed states and the dirt is on an unrelated ref.
    if dirty, !force {
        if shaB == headSha {
            benchLog("uncommitted changes detected: side B builds from the working tree")
            sideB = try prepareWorkingTreeSide(label: "\(refB)+dirty", root: root)
            labelB = "\(refB) (working tree)"
        } else if shaA == headSha {
            benchLog("uncommitted changes detected: side A builds from the working tree")
            sideA = try prepareWorkingTreeSide(label: "\(refA)+dirty", root: root)
        } else {
            benchLog("warning: uncommitted changes present but neither side is the checkout — measuring committed states only")
        }
    } else if dirty {
        benchLog("warning: --force with uncommitted changes measures committed states, not the working tree")
    }
    return try await judgeSides(sideA: sideA, sideB: sideB,
                          overlayA: sideA.binary.hasSuffix("OverlayBench"),
                          overlayB: sideB.binary.hasSuffix("OverlayBench"),
                          refA: refA, refB: labelB,
                          requested: requested, reps: reps, floor: floor, json: json, root: root)
}

/// A side built from the current checkout instead of a worktree.
///
/// The tool binary itself was built from these sources (`swift run`
/// rebuilds when stale), so its own image IS the working-tree binary —
/// no rebuild, no stash dance.
private func prepareWorkingTreeSide(label: String, root: String) throws -> CompareSide {
    guard let binary = Bundle.main.executablePath else {
        throw BenchError.suiteFailed("cannot locate own binary for working-tree side")
    }
    return try listSide(ref: label, binary: binary, root: root)
}

private func judgeSides(
    sideA: CompareSide,
    sideB: CompareSide,
    overlayA: Bool,
    overlayB: Bool,
    refA: String,
    refB: String,
    requested: String?,
    reps: Int,
    floor: Double,
    json: Bool,
    root: String
) async throws -> BenchExit {

    let namesA = Set(sideA.suites.map { $0.name })
    let shared: [String]
    if let requested {
        guard namesA.contains(requested), sideB.suites.map({ $0.name }).contains(requested) else {
            throw BenchError.suiteFailed("suite '\(requested)' missing on \(namesA.contains(requested) ? refB : refA)")
        }
        shared = [requested]
    } else {
        shared = sideA.suites.map { $0.name }.filter { sideB.suites.map { $0.name }.contains($0) }
    }
    if shared.isEmpty {
        throw BenchError.suiteFailed("no shared suites between \(refA) and \(refB)")
    }

    var worst = BenchExit.pass
    var report: [String] = []
    var allVerdicts: [MetricVerdict] = []
    // Color only on a real terminal; piped/CI output stays plain.
    let useColor = !json && isatty(STDOUT_FILENO) != 0
    for suite in shared {
        benchLog("suite \(suite): \(reps + 1) rounds (first is warmup)...")
        var samplesA: [String: [Double]] = [:]
        var samplesB: [String: [Double]] = [:]
        // reps+1 rounds, first dropped as warmup; A/B order alternates.
        // Sides are tagged at collection, so no index juggling below.
        for round in 0...reps {
            let started = ContinuousClock.now
            let firstIsA = round % 2 == 0
            let sides: [(Bool, String, Bool)] = firstIsA
                ? [(true, sideA.binary, overlayA), (false, sideB.binary, overlayB)]
                : [(false, sideB.binary, overlayB), (true, sideA.binary, overlayA)]
            for (isA, binary, overlay) in sides {
                let r = try runOnce(binary: binary, suite: suite, cwd: root, overlay: overlay)
                guard round > 0 else { continue }
                for m in r.metrics {
                    if isA {
                        samplesA[m.name, default: []].append(m.median)
                    } else {
                        samplesB[m.name, default: []].append(m.median)
                    }
                }
            }
            if round == 0 { continue }
            let elapsed = ContinuousClock.now - started
            benchLog("round \(round)/\(reps) done in \(Int(elapsed.components.seconds))s")
        }
        var verdicts: [MetricVerdict] = []
        for m in samplesA.keys.sorted() {
            guard let a = samplesA[m], let b = samplesB[m], !a.isEmpty, !b.isEmpty else { continue }
            let sa = computeStats(a)
            let sb = computeStats(b)
            verdicts.append(judgeMetric(name: m, baselineMedian: sa.median, baselineCV: sa.stdev / max(sa.mean, 1e-12),
                                        currentMedian: sb.median, floor: floor))
        }
        let block = "\(suite): \(refA) vs \(refB)\n" + renderVerdicts(verdicts, useColor: useColor)
        report.append(block)
        worst = worstStatus([worst] + verdicts.map { $0.status })
        allVerdicts += verdicts
    }
    let (load1, cores) = systemLoad()
    let hints = hintsForVerdicts(allVerdicts, load1: load1, cores: cores, lowPower: isLowPowerMode())
    if json {
        print("{\"suites\": \(shared.count), \"status\": \(worst.rawValue)}")
    } else {
        print("compare \(refA) vs \(refB)  reps=\(reps) floor=\(floor)")
        for b in report { print(b) }
        for h in hints {
            print("  \(h)")
        }
        print("overall: \(worst)")
    }
    return worst
}
