import Foundation

struct CommandResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
}

func runProcess(executable: String, arguments: [String], currentDirectory: String? = nil) -> CommandResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    if let currentDirectory {
        process.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
    }

    let outPipe = Pipe()
    let errPipe = Pipe()
    process.standardOutput = outPipe
    process.standardError = errPipe

    do {
        try process.run()
        process.waitUntilExit()
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let outStr = String(data: outData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let errStr = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return CommandResult(exitCode: process.terminationStatus, stdout: outStr, stderr: errStr)
    } catch {
        return CommandResult(exitCode: -1, stdout: "", stderr: error.localizedDescription)
    }
}

func findGit() -> String {
    let res = runProcess(executable: "/usr/bin/which", arguments: ["git"])
    return res.exitCode == 0 ? res.stdout : "/usr/bin/git"
}

func findProjectRoot() -> String {
    // 1. Check compile-time file path hierarchy
    var candidate = URL(fileURLWithPath: #filePath)
    while candidate.pathComponents.count > 1 {
        let pkgPath = candidate.appendingPathComponent("Package.swift").path
        let refPath = candidate.appendingPathComponent("ref/rustuna").path
        if FileManager.default.fileExists(atPath: pkgPath) && FileManager.default.fileExists(atPath: refPath) {
            return candidate.path
        }
        candidate.deleteLastPathComponent()
    }

    // 2. Fallback: Traverse upwards from current working directory
    var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    while dir.pathComponents.count > 1 {
        let pkgPath = dir.appendingPathComponent("Package.swift").path
        let refPath = dir.appendingPathComponent("ref/rustuna").path
        if FileManager.default.fileExists(atPath: pkgPath) && FileManager.default.fileExists(atPath: refPath) {
            return dir.path
        }
        dir.deleteLastPathComponent()
    }

    return FileManager.default.currentDirectoryPath
}

let projectRoot = findProjectRoot()
let workflowPath = URL(fileURLWithPath: projectRoot).appendingPathComponent(
    ".github/workflows/update-linux-binaries.yml"
).path
let rustunaDir = URL(fileURLWithPath: projectRoot).appendingPathComponent("ref/rustuna").path

func readCIPinnedCommit() -> String? {
    guard let content = try? String(contentsOfFile: workflowPath, encoding: .utf8) else {
        return nil
    }
    for line in content.split(separator: "\n") {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("RUSTUNA_COMMIT:") {
            let parts = trimmed.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                return parts[1].trimmingCharacters(in: .whitespaces)
            }
        }
    }
    return nil
}

func updateCIPinnedCommit(_ newCommit: String) throws {
    guard let content = try? String(contentsOfFile: workflowPath, encoding: .utf8) else {
        throw NSError(
            domain: "SwiftunaMigrator", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to read \(workflowPath)"]
        )
    }

    var lines = content.components(separatedBy: "\n")
    var updated = false

    for i in 0..<lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("RUSTUNA_COMMIT:") {
            let leadingSpaces = lines[i].prefix(while: { $0 == " " })
            lines[i] = "\(leadingSpaces)RUSTUNA_COMMIT: \(newCommit)"
            updated = true
            break
        }
    }

    if !updated {
        throw NSError(
            domain: "SwiftunaMigrator", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Could not find 'RUSTUNA_COMMIT:' in \(workflowPath)"])
    }

    let newContent = lines.joined(separator: "\n")
    try newContent.write(toFile: workflowPath, atomically: true, encoding: .utf8)
}

@main
struct SwiftunaMigratorApp {
    static func main() {
        let git = findGit()
        let args = CommandLine.arguments

        print("=========================================================")
        print("           Swiftuna Upstream Migration Tracker           ")
        print("=========================================================")

        let currentLog = runProcess(executable: git, arguments: ["-C", rustunaDir, "log", "-n", "1", "--oneline"])
        let currentFullHash = runProcess(executable: git, arguments: ["-C", rustunaDir, "rev-parse", "HEAD"]).stdout
        if currentLog.exitCode != 0 {
            print("❌ Error reading git log in \(rustunaDir): \(currentLog.stderr)")
            exit(1)
        }
        print("📌 Local ref/rustuna commit: \(currentLog.stdout)")

        let ciPinnedCommit = readCIPinnedCommit()
        if let ciPinnedCommit {
            print("☁️  CI workflow pinned commit: \(ciPinnedCommit)")
            if ciPinnedCommit != currentFullHash {
                print(
                    "⚠️  Warning: Local ref/rustuna (\(currentFullHash.prefix(8))) does not match CI workflow pin (\(ciPinnedCommit.prefix(8)))"
                )
            } else {
                print("✅ Local ref/rustuna and CI workflow are in sync.")
            }
        } else {
            print("⚠️  Warning: No 'RUSTUNA_COMMIT' found in \(workflowPath)")
        }

        // Check if user requested to accept/pin a specific commit or origin/main
        if args.contains("--accept") || args.contains("--apply") {
            print("\n🔄 Applying migration to latest upstream 'origin/main'...")
            _ = runProcess(executable: git, arguments: ["-C", rustunaDir, "fetch", "origin"])
            let checkoutRes = runProcess(executable: git, arguments: ["-C", rustunaDir, "checkout", "origin/main"])
            if checkoutRes.exitCode != 0 {
                // Fallback to fetch and checkout FETCH_HEAD if detached
                _ = runProcess(executable: git, arguments: ["-C", rustunaDir, "checkout", "FETCH_HEAD"])
            }

            let newHash = runProcess(executable: git, arguments: ["-C", rustunaDir, "rev-parse", "HEAD"]).stdout
            let newLog = runProcess(executable: git, arguments: ["-C", rustunaDir, "log", "-n", "1", "--oneline"])
                .stdout

            do {
                try updateCIPinnedCommit(newHash)
                print("✅ Updated ref/rustuna to \(newLog)")
                print("✅ Updated \(workflowPath) to pin RUSTUNA_COMMIT: \(newHash)")
                print("\n💡 Next steps:")
                print("   1. Rebuild local macOS binaries: just build-release")
                print("   2. Run parity & unit tests: just parity && just test")
                print("   3. Commit changes and push to trigger CI Linux binary build.")
            } catch {
                print("❌ Failed to update \(workflowPath): \(error.localizedDescription)")
                exit(1)
            }
            return
        }

        if let pinIdx = args.firstIndex(of: "--pin"), pinIdx + 1 < args.count {
            let targetPin = args[pinIdx + 1]
            print("\n📌 Pinning ref/rustuna and CI to commit: \(targetPin)...")
            _ = runProcess(executable: git, arguments: ["-C", rustunaDir, "fetch", "origin", targetPin])
            let checkoutRes = runProcess(executable: git, arguments: ["-C", rustunaDir, "checkout", targetPin])
            if checkoutRes.exitCode != 0 {
                print("❌ Failed to checkout \(targetPin) in \(rustunaDir): \(checkoutRes.stderr)")
                exit(1)
            }

            let newHash = runProcess(executable: git, arguments: ["-C", rustunaDir, "rev-parse", "HEAD"]).stdout
            do {
                try updateCIPinnedCommit(newHash)
                print("✅ Checked out \(newHash) in ref/rustuna")
                print("✅ Updated \(workflowPath) to pin RUSTUNA_COMMIT: \(newHash)")
            } catch {
                print("❌ Failed to update \(workflowPath): \(error.localizedDescription)")
                exit(1)
            }
            return
        }

        // Standard check / diff mode
        print("\n📡 Fetching upstream origin in \(rustunaDir)...")
        let fetchRes = runProcess(executable: git, arguments: ["-C", rustunaDir, "fetch", "origin"])
        if fetchRes.exitCode != 0 {
            print("⚠️  Warning: Unable to fetch origin (offline or no remote access): \(fetchRes.stderr)")
        }

        let diffStat = runProcess(
            executable: git, arguments: ["-C", rustunaDir, "di`ff", "HEAD..origin/main", "--stat"])
        let diffCommits = runProcess(
            executable: git, arguments: ["-C", rustunaDir, "log", "HEAD..origin/main", "--oneline"])

        if diffCommits.stdout.isEmpty {
            print("✅ Swiftuna is completely up-to-date with upstream 'origin/main'.")
            print("   No pending changes in ref/rustuna.")
        } else {
            let commitCount = diffCommits.stdout.split(separator: "\n").count
            print("⚠️  Upstream has \(commitCount) new commit(s) ahead of pinned commit:")
            print("---------------------------------------------------------")
            print(diffCommits.stdout)
            print("---------------------------------------------------------")
            print("\n📂 Modified files:")
            print(diffStat.stdout)

            let diffFull = runProcess(executable: git, arguments: ["-C", rustunaDir, "diff", "HEAD..origin/main"])
            var breaking = false
            var reasons: [String] = []

            if diffFull.stdout.contains("pub trait Sampler") {
                breaking = true
                reasons.append("Sampler trait modified in rustuna_core")
            }
            if diffFull.stdout.contains("pub trait Storage") {
                breaking = true
                reasons.append("Storage trait modified in rustuna_core")
            }
            if diffFull.stdout.contains("pub enum ErrorKind") {
                reasons.append("ErrorKind modified (new variants or alterations)")
            }
            if diffFull.stdout.contains("pub struct Trial") || diffFull.stdout.contains("impl Trial") {
                reasons.append("Trial struct or methods modified")
            }

            print("\n🔍 API Impact Analysis:")
            if breaking {
                print("🔴 CRITICAL: Upstream changes may contain breaking trait or ABI alterations:")
                for r in reasons { print("   - \(r)") }
                print("👉 Action: Review crates/rustuna-ffi and Sources/LibRustuna before updating.")
            } else if !reasons.isEmpty {
                print("🟡 NOTICE: Upstream additions or updates detected:")
                for r in reasons { print("   - \(r)") }
            } else {
                print("🟢 MINOR: Internal implementation changes, no public API disruption expected.")
            }

            print("\n💡 To accept this migration and update the CI pinned commit, run:")
            print("   swift run SwiftunaMigrator --accept")
        }
        print("=========================================================\n")
    }
}
