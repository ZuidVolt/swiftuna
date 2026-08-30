import Foundation
import Testing
import Swiftuna

@Suite("Optimization Timeout Tests", .serialized)
struct TimeoutTests {

    @Test("study.optimize with timeout stops execution gracefully within target duration")
    func testTimeoutStopsOptimizationGracefully() throws {
        let study = try Swiftuna.createStudy(name: "timeout_graceful_\(UUID().uuidString)")
        let clock = ContinuousClock()
        let start = clock.now

        try study.optimize(timeout: .milliseconds(120)) { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            Thread.sleep(forTimeInterval: 0.03) // 30ms per trial
            return x * x
        }

        let elapsed = clock.now - start
        let elapsedMs = Double(elapsed.components.attoseconds) / 1e15

        let trials = try study.trials
        #expect(trials.count >= 2)
        #expect(trials.count <= 6)
        #expect(elapsedMs >= 90.0)
        #expect(elapsedMs <= 300.0)

        // All completed trials are in .complete state
        for t in trials {
            #expect(t.state == .complete)
        }
    }

    @Test("study.optimize with both nTrials and timeout stops as soon as first limit is met")
    func testTimeoutAndNTrialsCoexist() throws {
        let study = try Swiftuna.createStudy(name: "timeout_coexist_\(UUID().uuidString)")

        // nTrials = 1000, but timeout = 50ms (can only run 1-3 trials)
        try study.optimize(nTrials: 1000, timeout: .milliseconds(60)) { trial in
            let x = try trial.suggest("x", in: 0.0...10.0)
            Thread.sleep(forTimeInterval: 0.02)
            return x
        }

        let trials = try study.trials
        #expect(trials.count < 10)
        #expect(!trials.isEmpty)
    }

    @Test("Concurrent study.optimize with TaskGroup respects timeout gracefully")
    func testConcurrentTimeoutGracefulCompletion() async throws {
        let study = try Swiftuna.createStudy(name: "timeout_concurrent_\(UUID().uuidString)")

        try await study.optimize(nTrials: 500, timeout: .milliseconds(100), concurrency: 4) { trial in
            let x = try trial.suggest("x", in: -1.0...1.0)
            try await Task.sleep(for: .milliseconds(25))
            return x * x
        }

        let trials = try study.trials
        #expect(trials.count > 0)
        #expect(trials.count < 50)
        for t in trials {
            #expect(t.state == .complete)
        }
    }
}
