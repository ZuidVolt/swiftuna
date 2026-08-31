import Foundation
import Synchronization

#if canImport(os)
    import os
#endif

/// Status of an active telemetry span.
public enum SpanStatus: Sendable, Equatable {
    /// The span completed normally without errors.
    case ok

    /// The span completed with an error message.
    case error(String)
}

/// Represents an active observability trace span.
///
/// Designed for 100% zero-dependency compatibility with OpenTelemetry and `swift-distributed-tracing`.
public protocol TelemetrySpan: Sendable {
    /// Adds or updates a string metadata attribute on this span.
    ///
    /// - Parameters:
    ///   - key: The attribute key (e.g. `"study.name"`, `"trial.status"`).
    ///   - value: The string value for the attribute.
    func setAttribute(_ key: String, value: String)

    /// Records a point-in-time event with optional attributes within the span lifecycle.
    ///
    /// - Parameters:
    ///   - name: The event name.
    ///   - attributes: Key-value metadata describing the event.
    func recordEvent(name: String, attributes: [String: String])

    /// Ends the span, setting its final completion status.
    ///
    /// - Parameter status: The final status (``SpanStatus/ok`` or ``SpanStatus/error(_:)``).
    func end(status: SpanStatus)
}

/// Tracer interface for instrumenting Swiftuna optimization operations.
///
/// Implement this protocol to forward trial spans and metrics to OpenTelemetry collectors,
/// Datadog, Prometheus, or custom logging pipelines.
public protocol TelemetryTracer: Sendable {
    /// Starts a new telemetry span with the specified name and initial attributes.
    ///
    /// - Parameters:
    ///   - name: Name of the operation being traced (e.g. `"swiftuna.trial"`).
    ///   - attributes: Initial key-value metadata attached to the span.
    /// - Returns: An active ``TelemetrySpan`` handle.
    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan
}

/// Default no-op span with zero CPU overhead.
public struct NoOpTelemetrySpan: TelemetrySpan {
    @inline(always) public init() {}
    @inline(always) public func setAttribute(_ key: String, value: String) {}
    @inline(always) public func recordEvent(name: String, attributes: [String: String]) {}
    @inline(always) public func end(status: SpanStatus) {}
}

/// Default no-op tracer with zero CPU overhead.
public struct NoOpTelemetryTracer: TelemetryTracer {
    @inline(always) public init() {}
    @inline(always) public func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        NoOpTelemetrySpan()
    }
}

#if canImport(os)
    import os
#endif

/// Thread-safe registry for Swiftuna telemetry and observability.
///
/// Use `SwiftunaTelemetry.shared` to register a global tracer that receives span notifications
/// across all study evaluations and trial executions.
///
/// ### Example
/// ```swift
/// final class PrintTracer: TelemetryTracer {
///     func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
///         print("Started span \(name) with attributes \(attributes)")
///         return NoOpTelemetrySpan()
///     }
/// }
///
/// SwiftunaTelemetry.shared.registerTracer(PrintTracer())
/// ```
public final class SwiftunaTelemetry: Sendable {
    /// The global shared telemetry registry instance.
    public static let shared = SwiftunaTelemetry()

    #if canImport(os)
        internal static let logger = Logger(subsystem: "org.swiftuna", category: "optimization")
    #endif

    private let tracerState = Mutex<(any TelemetryTracer)?>(nil)

    /// The currently active tracer, or ``NoOpTelemetryTracer`` if none is registered.
    public var tracer: any TelemetryTracer {
        tracerState.withLock { $0 } ?? NoOpTelemetryTracer()
    }

    /// Registers a global telemetry tracer, or unregisters the active tracer by passing `nil`.
    ///
    /// - Parameter tracer: The tracer instance to register, or `nil` to restore no-op tracing.
    public func registerTracer(_ tracer: (any TelemetryTracer)?) {
        tracerState.withLock { $0 = tracer }
    }
}
