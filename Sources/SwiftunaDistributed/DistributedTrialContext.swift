public import Distributed
import Foundation
import Synchronization
public import Swiftuna

/// A worker-side handle for one checked-out trial.
///
/// `DistributedTrialContext` removes the `trialNumber` bookkeeping from worker
/// loops: `report` and `tell` already know which trial they belong to.
///
/// ```swift
/// let ctx = try await DistributedTrialContext.checkout(from: coordinator)
/// let x = ctx.trial.double("x") ?? 0.0
/// for epoch in 1...3 {
///     if try await ctx.report(step: epoch, value: x * x) {
///         try await ctx.prune()
///         break
///     }
/// }
/// try await ctx.tell(value: x * x)
/// ```
///
/// Constructing (via ``checkout(from:)``) and every method below are plain
/// local calls — only the underlying `ask`/`report`/`tell` cross the actor
/// boundary — so the context itself never needs to satisfy the actor system's
/// `Codable` requirement.

/// Worker-side span state: opened at checkout, ended exactly once by the
/// first terminal call (`tell`, `prune`, or `fail`).
///
/// `nil` on the context when no tracer is registered, so the uninstrumented
/// path pays one relaxed atomic load at checkout and a nil check per call.
/// No dictionaries, no strings, no locks.
private final class TrialSpanState: Sendable {
    private let span: any TelemetrySpan
    private let params: [String: ParameterValue]
    private let start: ContinuousClock.Instant
    private let ended = Atomic(false)

    init(span: any TelemetrySpan, params: [String: ParameterValue]) {
        self.span = span
        self.params = params
        self.start = .now
    }

    /// Records a report heartbeat as a span event. No-op after the span ends.
    func recordReport(step: Int, value: Double) {
        guard !ended.load(ordering: .relaxed) else { return }
        span.recordEvent(
            name: "trial.report",
            attributes: ["trial.step": .int(step), "trial.value": .double(value)]
        )
    }

    /// Sets status attributes and ends the span. Only the first call acts.
    ///
    /// Sampled hyperparameters ride every terminal path — including a failed
    /// tell, where the params are still known even though the outcome isn't.
    func finish(
        statusAttribute: String,
        status: SpanStatus,
        recordDuration: Bool
    ) {
        let alreadyEnded = ended.compareExchange(
            expected: false, desired: true, ordering: .relaxed
        ).original
        guard !alreadyEnded else { return }
        // The experiment, not just the run.
        for (name, value) in params {
            span.setAttribute("param.\(name)", value: value.telemetryAttribute)
        }
        span.setAttribute("trial.status", value: statusAttribute)
        if recordDuration {
            let elapsed = ContinuousClock.now - start
            span.setAttribute(
                "trial.duration_ms",
                value: .double(Double(elapsed.components.attoseconds) / 1e15)
            )
        }
        span.end(status: status)
    }
}

public struct DistributedTrialContext<ActorSystem>: ~Copyable, Sendable
where ActorSystem: DistributedActorSystem<any Codable> {
    /// The checked-out trial.
    public let trial: DistributedTrial

    private let coordinator: StudyCoordinator<ActorSystem>

    /// Worker-side span. `nil` unless a tracer was registered at checkout.
    private let spanState: TrialSpanState?

    /// The checked-out trial number.
    public var trialNumber: Int { trial.trialNumber }

    public init(trial: DistributedTrial, coordinator: StudyCoordinator<ActorSystem>) {
        self.trial = trial
        self.coordinator = coordinator
        // `span` returns nil when unregistered, and its attributes are
        // autoclosure-deferred, so the disabled path builds nothing here.
        self.spanState = SwiftunaTelemetry.shared.span(
            name: "swiftuna.trial",
            attributes: Self.trialAttributes(for: trial),
            parent: trial.traceParent
        ).map { TrialSpanState(span: $0, params: trial.params) }
    }

    /// Worker span attributes. A function (not inline statements) so the
    /// dictionary builds only inside `span`'s autoclosure when enabled.
    private static func trialAttributes(for trial: DistributedTrial) -> [String: TelemetryAttribute] {
        var attributes: [String: TelemetryAttribute] = ["trial.number": .int(trial.trialNumber)]
        if !trial.studyName.isEmpty {
            attributes["study.name"] = .string(trial.studyName)
        }
        attributes["trial.distributed"] = .bool(true)
        return attributes
    }

    /// Checks out the next trial from `coordinator` wrapped in a context.
    public static func checkout(from coordinator: StudyCoordinator<ActorSystem>) async throws -> Self {
        Self(trial: try await coordinator.ask(), coordinator: coordinator)
    }

    /// Checks out the next trial, waiting up to `timeout` for a free slot.
    public static func checkout(
        from coordinator: StudyCoordinator<ActorSystem>,
        waitingUpTo timeout: Duration
    ) async throws -> Self {
        Self(trial: try await coordinator.ask(waitingUpTo: timeout), coordinator: coordinator)
    }

    /// Reports an intermediate value. Returns `true` if the pruner recommends stopping.
    @discardableResult
    public func report(step: Int, value: Double) async throws -> Bool {
        let vote = try await coordinator.report(trialNumber: trial.trialNumber, step: step, value: value)
        spanState?.recordReport(step: step, value: value)
        return vote
    }

    /// Records the final outcome with a prebuilt result.
    ///
    /// The worker span ends after the coordinator confirms the outcome, so it
    /// records what the trial actually became — never the intent. A throwing
    /// tell ends the span as failed: the trial may still be in flight, and a
    /// span claiming `complete` would be the worse lie. (On error-blind
    /// transports a swallowed failure still reads `complete`; that transport
    /// limitation is documented, not fixable here.)
    public func tell(_ result: DistributedTrialResult) async throws {
        do {
            try await coordinator.tell(result)
        } catch {
            spanState?.finish(
                statusAttribute: "failed",
                status: .error("tell failed: \(error)"),
                recordDuration: false
            )
            throw error
        }
        switch result.state {
        case .complete:
            spanState?.finish(
                statusAttribute: "complete",
                status: .ok,
                recordDuration: true
            )
        case .pruned:
            spanState?.finish(statusAttribute: "pruned", status: .ok, recordDuration: false)
        case .fail, .running, .waiting:
            spanState?.finish(statusAttribute: "failed", status: .error("trial failed"), recordDuration: false)
        }
    }

    /// Records a single-objective completed outcome with typed keys.
    public func tell(
        value: Double,
        constraintPairs: [(ConstraintKey, Double)] = [],
        userAttrPairs: [(String, any AttributeConvertible)] = []
    ) async throws {
        try await tell(
            DistributedTrialResult(
                trialNumber: trial.trialNumber,
                value: value,
                constraintPairs: constraintPairs,
                userAttrPairs: userAttrPairs
            ))
    }

    /// Records the trial as pruned.
    public func prune() async throws {
        try await tell(.pruned(trialNumber: trial.trialNumber))
    }

    /// Records the trial as failed.
    public func fail() async throws {
        try await tell(.failed(trialNumber: trial.trialNumber))
    }
}
