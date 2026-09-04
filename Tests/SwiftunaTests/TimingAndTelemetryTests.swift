import Foundation
import Testing
@testable import Swiftuna

import Synchronization

final class MockSpan: TelemetrySpan, Sendable {
    let name: String
    private struct State: Sendable {
        var attributes: [String: String]
        var events: [(name: String, attributes: [String: String])] = []
        var children: [MockSpan] = []
        var isEnded: Bool = false
        var status: SpanStatus?
    }
    private let state: Mutex<State>

    var attributes: [String: String] {
        state.withLock { $0.attributes }
    }

    var events: [(name: String, attributes: [String: String])] {
        state.withLock { $0.events }
    }

    var children: [MockSpan] {
        state.withLock { $0.children }
    }

    var isEnded: Bool {
        state.withLock { $0.isEnded }
    }

    var status: SpanStatus? {
        state.withLock { $0.status }
    }

    init(name: String, attributes: [String: String]) {
        self.name = name
        self.state = Mutex(State(attributes: attributes))
    }

    func setAttribute(_ key: String, value: String) {
        state.withLock { $0.attributes[key] = value }
    }

    func recordEvent(name: String, attributes: [String: String]) {
        state.withLock { $0.events.append((name, attributes)) }
    }

    func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan {
        let child = MockSpan(name: name, attributes: attributes)
        state.withLock { $0.children.append(child) }
        return child
    }

    func end(status: SpanStatus) {
        state.withLock {
            $0.isEnded = true
            $0.status = status
        }
    }
}

final class MockTracer: TelemetryTracer, Sendable {
    private let _spans = Mutex<[MockSpan]>([])

    var spans: [MockSpan] {
        _spans.withLock { $0 }
    }

    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        let span = MockSpan(name: name, attributes: attributes)
        _spans.withLock { $0.append(span) }
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
            _ = try trial.suggest("x", in: 0.0...1.0)
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
        // Raw double form: parses back regardless of device locale.
        #expect(Double(firstSpan.attributes["trial.duration_ms"] ?? "") != nil)
        #expect(firstSpan.attributes["param.x"] != nil)
        #expect(firstSpan.isEnded == true)
        #expect(firstSpan.status == .ok)
    }

    @Test("Report heartbeats and pruner votes land on the trial span as events")
    func testReportAndPruneVoteEvents() throws {
        let studyName = "telemetry_events_\(UUID().uuidString)"
        let mockTracer = MockTracer()
        SwiftunaTelemetry.shared.registerTracer(mockTracer)
        defer { SwiftunaTelemetry.shared.registerTracer(nil) }

        let study = try Swiftuna.createStudy(name: studyName)

        try study.optimize(nTrials: 1) { trial in
            let x = try trial.suggest("x", in: 0.0...1.0)
            for step in 0..<3 {
                try trial.report(x * Double(step), step: step)
            }
            _ = try trial.shouldPrune
            return x
        }

        let relevantSpans = mockTracer.spans.filter { $0.attributes["study.name"] == studyName }
        let span = try #require(relevantSpans.first)
        let reports = span.events.filter { $0.name == "trial.report" }
        #expect(reports.count == 3)
        #expect(reports.first?.attributes["trial.step"] == "0")
        let votes = span.events.filter { $0.name == "trial.prune_vote" }
        #expect(votes.count == 1)
        let prunerSpans = span.children.filter { $0.name == "swiftuna.pruner" }
        #expect(prunerSpans.count == 1)
        #expect(prunerSpans.first?.isEnded == true)
        #expect(prunerSpans.first?.attributes["trial.prune_vote"] != nil)
    }

    @Test("Pruned trials carry sampled params on the span")
    func testPrunedTrialSpanCarriesParams() throws {
        let studyName = "telemetry_pruned_params_\(UUID().uuidString)"
        let mockTracer = MockTracer()
        SwiftunaTelemetry.shared.registerTracer(mockTracer)
        defer { SwiftunaTelemetry.shared.registerTracer(nil) }

        let study = try Swiftuna.createStudy(name: studyName)

        try study.optimize(nTrials: 1) { trial in
            _ = try trial.suggest("x", in: 0.0...1.0)
            _ = try trial.suggest("layers", in: 1...4)
            try trial.prune()
        }

        let relevantSpans = mockTracer.spans.filter { $0.attributes["study.name"] == studyName }
        let span = try #require(relevantSpans.first)
        #expect(span.attributes["trial.status"] == "pruned")
        #expect(span.attributes["param.x"] != nil)
        #expect(span.attributes["param.layers"] != nil)
        #expect(span.isEnded == true)
    }
}
