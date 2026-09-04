import Distributed
import Foundation
import Synchronization
import Testing

@testable import Swiftuna
@testable import SwiftunaDistributed

/// In-memory tracer capturing every span for assertions. Thread-safe: the
/// global registry is shared across parallel test suites, so assertions
/// filter by exact study name.
private final class RecordingSpan: TelemetrySpan, Sendable {
    let name: String
    let parent: String?
    private let state = Mutex<State>(State())

    private struct State {
        var attributes: [String: String] = [:]
        var typedAttributes: [String: TelemetryAttribute] = [:]
        var events: [(name: String, attributes: [String: String])] = []
        var endStatus: SpanStatus?
    }

    init(name: String, attributes: [String: String], parent: String?) {
        self.name = name
        self.parent = parent
        state.withLock { $0.attributes = attributes }
    }

    init(name: String, attributes: [String: TelemetryAttribute], parent: String?) {
        self.name = name
        self.parent = parent
        state.withLock {
            $0.typedAttributes = attributes
            $0.attributes = Dictionary(
                uniqueKeysWithValues: attributes.map { ($0.key, $0.value.stringValue) })
        }
    }

    var traceParent: String? { "00-trace-\(name)-01" }

    func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan {
        RecordingSpan(name: name, attributes: attributes, parent: traceParent)
    }

    func setAttribute(_ key: String, value: String) {
        state.withLock { $0.attributes[key] = value }
    }

    func setAttribute(_ key: String, value: TelemetryAttribute) {
        state.withLock {
            $0.typedAttributes[key] = value
            $0.attributes[key] = value.stringValue
        }
    }

    func recordEvent(name: String, attributes: [String: String]) {
        state.withLock { $0.events.append((name, attributes)) }
    }

    func end(status: SpanStatus) {
        state.withLock { $0.endStatus = status }
    }

    var attributes: [String: String] { state.withLock { $0.attributes } }
    var typedAttributes: [String: TelemetryAttribute] { state.withLock { $0.typedAttributes } }
    var events: [(name: String, attributes: [String: String])] { state.withLock { $0.events } }
    var endStatus: SpanStatus? { state.withLock { $0.endStatus } }
}

private final class RecordingTracer: TelemetryTracer, Sendable {
    private let spansState = Mutex<[RecordingSpan]>([])

    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        startSpan(name: name, attributes: attributes, parent: nil)
    }

    func startSpan(name: String, attributes: [String: TelemetryAttribute]) -> any TelemetrySpan {
        let span = RecordingSpan(name: name, attributes: attributes, parent: nil)
        spansState.withLock { $0.append(span) }
        return span
    }

    func startSpan(name: String, attributes: [String: String], parent: String?) -> any TelemetrySpan {
        let span = RecordingSpan(name: name, attributes: attributes, parent: parent)
        spansState.withLock { $0.append(span) }
        return span
    }

    func startSpan(
        name: String, attributes: [String: TelemetryAttribute], parent: String?
    ) -> any TelemetrySpan {
        let span = RecordingSpan(name: name, attributes: attributes, parent: parent)
        spansState.withLock { $0.append(span) }
        return span
    }

    func spans(named name: String, study: String) -> [RecordingSpan] {
        spansState.withLock {
            $0.filter { $0.name == name && $0.attributes["study.name"] == study }
        }
    }
}

@Suite(.serialized)
struct TelemetryTests {
    private func withTracer<T>(_ body: (RecordingTracer) async throws -> T) async throws -> T {
        let tracer = RecordingTracer()
        SwiftunaTelemetry.shared.registerTracer(tracer)
        defer { SwiftunaTelemetry.shared.registerTracer(nil) }
        return try await body(tracer)
    }

    @Test func testWorkerSpanCompleteLifecycle() async throws {
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(named: "dist_tele_complete") { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            let ctx = try await DistributedTrialContext.checkout(from: coordinator)
            let x = ctx.trial.double("x")!
            _ = try await ctx.report(step: 1, value: x * x)
            try await ctx.tell(value: x * x)

            let workers = tracer.spans(named: "swiftuna.trial", study: study.name)
            #expect(workers.count == 1)
            let span = try #require(workers.first)
            #expect(span.attributes["trial.number"] == String(ctx.trialNumber))
            #expect(span.typedAttributes["trial.number"] == .int(ctx.trialNumber))
            #expect(span.attributes["trial.distributed"] == "true")
            #expect(span.typedAttributes["trial.distributed"] == .bool(true))
            #expect(span.attributes["trial.status"] == "complete")
            #expect(span.attributes["trial.duration_ms"] != nil)
            // Duration travels as a raw double: locale-invariant and aggregatable.
            if case .double(let ms) = span.typedAttributes["trial.duration_ms"] {
                #expect(ms >= 0)
            } else {
                Issue.record("trial.duration_ms must be a typed double")
            }
            // Sampled hyperparameters ride the span: the experiment, not just the run.
            #expect(span.typedAttributes["param.x"] != nil)
            if case .double = span.typedAttributes["param.x"] {} else {
                Issue.record("param.x must be a typed double")
            }
            #expect(span.events.count == 1)
            #expect(span.events.first?.name == "trial.report")
            #expect(span.events.first?.attributes["trial.step"] == "1")
            #expect(span.endStatus == .ok)

            let samples = tracer.spans(named: "swiftuna.coordinator.sample", study: study.name)
            #expect(samples.count == 1)
            let records = tracer.spans(named: "swiftuna.coordinator.record", study: study.name)
            #expect(records.count == 1)
            #expect(records.first?.endStatus == .ok)
        }
    }

    @Test func testWorkerSpanPruned() async throws {
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(named: "dist_tele_prune") { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            let ctx = try await DistributedTrialContext.checkout(from: coordinator)
            try await ctx.prune()

            let workers = tracer.spans(named: "swiftuna.trial", study: study.name)
            let span = try #require(workers.first)
            #expect(span.attributes["trial.status"] == "pruned")
            #expect(span.endStatus == .ok)
            // Pruned trials map the search space too: params ride every path.
            #expect(span.typedAttributes["param.x"] != nil)
        }
    }

    @Test func testWorkerSpanFailedOnObjectiveThrow() async throws {
        struct Boom: Error {}
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(named: "dist_tele_fail") { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            do {
                try await runWorker(coordinator: coordinator, nTrials: 1) {
                    (_: borrowing DistributedTrialContext<LocalTestingDistributedActorSystem>) async throws -> Double in
                    throw Boom()
                }
                Issue.record("expected objective error")
            } catch is Boom {
                // Expected: recorded as failed, then rethrown.
            }
            let workers = tracer.spans(named: "swiftuna.trial", study: study.name)
            let span = try #require(workers.first)
            #expect(span.attributes["trial.status"] == "failed")
            if case .error = span.endStatus {} else { Issue.record("expected error end status") }
            #expect(span.typedAttributes["param.x"] != nil)
        }
    }

    @Test func testFailedTellEndsSpanAsFailed() async throws {
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(named: "dist_tele_tellfail") { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            let ctx = try await DistributedTrialContext.checkout(from: coordinator)
            // Tell for a trial number that was never checked out: the outcome
            // is unknown, so the span must read failed — never complete.
            do {
                try await ctx.tell(DistributedTrialResult(trialNumber: 9999, value: 1.0))
                Issue.record("expected trialNotFound")
            } catch SwiftunaDistributedError.trialNotFound {}
            let workers = tracer.spans(named: "swiftuna.trial", study: study.name)
            let span = try #require(workers.first)
            #expect(span.attributes["trial.status"] == "failed")
            if case .error = span.endStatus {} else { Issue.record("expected error end status") }
            // Params are known even though the outcome isn't.
            #expect(span.typedAttributes["param.x"] != nil)
        }
    }

    @Test func testTraceParentRoundTrip() async throws {
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(named: "dist_tele_trace") { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            let spec = try await coordinator.ask()
            #expect(spec.studyName == study.name)
            let sampleParent = try #require(
                tracer.spans(named: "swiftuna.coordinator.sample", study: study.name).first?.traceParent)
            #expect(spec.traceParent == sampleParent)

            let ctx = DistributedTrialContext(trial: spec, coordinator: coordinator)
            let workers = tracer.spans(named: "swiftuna.trial", study: study.name)
            #expect(workers.first?.parent == sampleParent)
            try await ctx.tell(value: 1.0)
        }
    }

    @Test func testLeaseExpiryEmitsReapSpan() async throws {
        try await withTracer { tracer in
            let (coordinator, study) = try makeTestCoordinator(
                named: "dist_tele_reap",
                leasePolicy: LeasePolicy(timeout: .milliseconds(1))
            ) { trial in
                let x = try trial.suggest("x", in: -5.0...5.0)
                return ["x": .double(x)]
            }
            let first = try await coordinator.ask()
            try await Task.sleep(for: .milliseconds(10))
            _ = try await coordinator.ask() // triggers piggyback reap
            #expect(first.trialNumber != -1)

            let reaps = tracer.spans(named: "swiftuna.coordinator.reap", study: study.name)
            let reap = try #require(reaps.first)
            #expect(reap.events.contains {
                $0.name == "trial.lease_expired"
                    && $0.attributes["trial.number"] == String(first.trialNumber)
            })
        }
    }

    @Test func testOldPayloadDecodesWithDefaults() throws {
        let legacy = #"{"trialNumber":7,"params":{"x":0.5}}"#.data(using: .utf8)!
        let spec = try JSONDecoder().decode(DistributedTrial.self, from: legacy)
        #expect(spec.trialNumber == 7)
        #expect(spec.studyName == "")
        #expect(spec.traceParent == nil)
    }

    @Test func testDisabledPathEmitsNothing() async throws {
        #expect(!SwiftunaTelemetry.shared.isEnabled)
        let (coordinator, _) = try makeTestCoordinator(named: "dist_tele_off") { trial in
            let x = try trial.suggest("x", in: -5.0...5.0)
            return ["x": .double(x)]
        }
        let ctx = try await DistributedTrialContext.checkout(from: coordinator)
        _ = try await ctx.report(step: 1, value: 1.0)
        try await ctx.tell(value: 1.0)
        // No tracer registered: success is the assertion. Nothing to observe.
    }
}
