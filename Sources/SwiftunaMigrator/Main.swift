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

@main
struct SwiftunaMigratorApp {
    static func main() {
        let git = findGit()
        let rustunaDir = "ref/rustuna"

        print("=========================================================")
        print("           Swiftuna Upstream Migration Tracker           ")
        print("=========================================================")

        let currentLog = runProcess(executable: git, arguments: ["-C", rustunaDir, "log", "-n", "1", "--oneline"])
        if currentLog.exitCode != 0 {
            print("❌ Error reading git log in \(rustunaDir): \(currentLog.stderr)")
            exit(1)
        }
        print("📌 Currently pinned rustuna commit: \(currentLog.stdout)")

        print("\n📡 Fetching upstream origin in \(rustunaDir)...")
        let fetchRes = runProcess(executable: git, arguments: ["-C", rustunaDir, "fetch", "origin"])
        if fetchRes.exitCode != 0 {
            print("⚠️  Warning: Unable to fetch origin (offline or no remote access): \(fetchRes.stderr)")
        }

        let diffStat = runProcess(executable: git, arguments: ["-C", rustunaDir, "diff", "HEAD..origin/main", "--stat"])
        let diffCommits = runProcess(executable: git, arguments: ["-C", rustunaDir, "log", "HEAD..origin/main", "--oneline"])

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
        }
        print("=========================================================\n")
    }
}
