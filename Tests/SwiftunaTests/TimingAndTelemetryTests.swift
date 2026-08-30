import Foundation
import Testing
@testable import Swiftuna

final class MockSpan: TelemetrySpan, @unchecked Sendable {
    let name: String
    var attributes: [String: String]
    var isEnded: Bool = false
    var status: SpanStatus?

    init(name: String, attributes: [String: String]) {
        self.name = name
        self.attributes = attributes
    }

    func setAttribute(_ key: String, value: String) {
        attributes[key] = value
    }

    func recordEvent(name: String, attributes: [String: String]) {}

    func end(status: SpanStatus) {
        self.isEnded = true
        self.status = status
    }
}

final class MockTracer: TelemetryTracer, @unchecked Sendable {
    var spans: [MockSpan] = []

    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        let span = MockSpan(name: name, attributes: attributes)
        spans.append(span)
        return span
    }
}

@Suite("Timing & Telemetry Analytics Tests", .serialized)
struct TimingAndTelemetryTests {

    @Test("Trial timestamps and Swift 6 Duration are captured across optimization")
    func testTrialDurationAndTimestamps() throws {
        let study = try Swiftuna.createStudy(name: "timing_test_\(UUID().uuidString)")

        try study.optimize(nTrials: 3) { trial in
            let x = try trial.suggest("x", in: 0.0...1.0)
            return x
        }

        let trials = try study.trials
        #expect(trials.count == 3)

        for trial in trials {
            #expect(trial.datetimeStart != nil)
            #expect(trial.datetimeComplete != nil)
            #expect(trial.duration != nil)
            if let duration = trial.duration {
                // Duration should be >= 0
                #expect(duration >= .zero)
            }
        }
    }

    @Test("Custom TelemetryTracer captures trial spans and metadata without external dependencies")
    func testTelemetryTracingIntegration() throws {
        let studyName = "telemetry_study_\(UUID().uuidString)"
        let mockTracer = MockTracer()
        SwiftunaTelemetry.shared.registerTracer(mockTracer)
        defer { SwiftunaTelemetry.shared.registerTracer(nil) }

        let study = try Swiftuna.createStudy(name: studyName)

        try study.optimize(nTrials: 2) { trial in
            return 3.14
        }

        let relevantSpans = mockTracer.spans.filter { $0.attributes["study.name"] == studyName }
        #expect(relevantSpans.count == 2)
        let firstSpan = relevantSpans[0]
        #expect(firstSpan.name == "swiftuna.trial")
        #expect(firstSpan.attributes["study.name"] == studyName)
        #expect(firstSpan.attributes["trial.number"] == "0")
        #expect(firstSpan.attributes["trial.status"] == "complete")
        #expect(firstSpan.attributes["trial.duration_ms"] != nil)
        #expect(firstSpan.isEnded == true)
        #expect(firstSpan.status == .ok)
    }
}
