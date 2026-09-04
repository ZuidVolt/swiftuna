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

/// A typed span attribute value.
///
/// Numeric and boolean values travel to the backend with their types intact,
/// so collectors can aggregate and alert on them. String-only backends receive
/// ``stringValue``, which is locale-invariant (`String(format:)` follows the
/// device locale and must never touch telemetry).
public enum TelemetryAttribute: Sendable, Hashable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    /// Locale-invariant string form for string-only backends.
    public var stringValue: String {
        switch self {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        }
    }
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

    /// Adds or updates a typed metadata attribute on this span.
    ///
    /// Defaults to the string form; typed backends override to preserve types.
    ///
    /// - Parameters:
    ///   - key: The attribute key (e.g. `"trial.number"`, `"trial.duration_ms"`).
    ///   - value: The typed attribute value.
    func setAttribute(_ key: String, value: TelemetryAttribute)

    /// Records a point-in-time event with optional attributes within the span lifecycle.
    ///
    /// - Parameters:
    ///   - name: The event name.
    ///   - attributes: Key-value metadata describing the event.
    func recordEvent(name: String, attributes: [String: String])

    /// Records a point-in-time event with typed attributes.
    ///
    /// Defaults to the string forms; typed backends override to preserve types.
    ///
    /// - Parameters:
    ///   - name: The event name.
    ///   - attributes: Typed key-value metadata describing the event.
    func recordEvent(name: String, attributes: [String: TelemetryAttribute])

    /// Ends the span, setting its final completion status.
    ///
    /// - Parameter status: The final status (``SpanStatus/ok`` or ``SpanStatus/error(_:)``).
    func end(status: SpanStatus)

    /// W3C `traceparent` header value identifying this span's trace context.
    ///
    /// `nil` unless the backend supports distributed context propagation.
    /// The core treats this as an opaque string: it is forwarded through
    /// ``DistributedTrial`` untouched and never parsed. OpenTelemetry adapters
    /// map it to a remote `SpanContext`.
    var traceParent: String? { get }

    /// Starts a child span parented to this span.
    ///
    /// Backends without context support fall back to an uncorrelated root span;
    /// correlate through `study.name` + `trial.number` attributes instead.
    func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan

    /// Starts a child span with typed initial attributes.
    ///
    /// Defaults to the string forms.
    func traceChild(name: String, attributes: [String: TelemetryAttribute]) -> any TelemetrySpan
}

/// Source-compatible defaults: existing ``TelemetrySpan`` conformances keep
/// compiling and behave as correlation-by-attributes backends.
extension TelemetrySpan {
    @inline(always) public func setAttribute(_ key: String, value: TelemetryAttribute) {
        setAttribute(key, value: value.stringValue)
    }

    @inline(always) public func recordEvent(name: String, attributes: [String: TelemetryAttribute]) {
        recordEvent(name: name, attributes: attributes.stringified())
    }

    @inline(always) public var traceParent: String? { nil }

    @inline(always) public func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan {
        NoOpTelemetrySpan()
    }

    @inline(always) public func traceChild(
        name: String, attributes: [String: TelemetryAttribute]
    ) -> any TelemetrySpan {
        traceChild(name: name, attributes: attributes.stringified())
    }
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

    /// Starts a new telemetry span with typed initial attributes.
    ///
    /// Defaults to the string forms; typed backends override to preserve types.
    ///
    /// - Parameters:
    ///   - name: Name of the operation being traced.
    ///   - attributes: Typed initial key-value metadata attached to the span.
    /// - Returns: An active ``TelemetrySpan`` handle.
    func startSpan(name: String, attributes: [String: TelemetryAttribute]) -> any TelemetrySpan

    /// Starts a span parented to a W3C `traceparent` received from another process.
    ///
    /// `parent` is opaque: adapters resolve it to a remote `SpanContext`,
    /// zero-dependency backends ignore it. Defaults to the unparented start.
    ///
    /// - Parameters:
    ///   - name: Name of the operation being traced.
    ///   - attributes: Initial key-value metadata attached to the span.
    ///   - parent: `traceparent` string from the upstream span, if any.
    /// - Returns: An active ``TelemetrySpan`` handle.
    func startSpan(name: String, attributes: [String: String], parent: String?) -> any TelemetrySpan

    /// Starts a parented span with typed initial attributes.
    ///
    /// Defaults to the string forms.
    func startSpan(
        name: String, attributes: [String: TelemetryAttribute], parent: String?
    ) -> any TelemetrySpan
}

/// Source-compatible default: existing ``TelemetryTracer`` conformances keep
/// compiling and behave as correlation-by-attributes backends.
extension TelemetryTracer {
    @inline(always) public func startSpan(
        name: String,
        attributes: [String: TelemetryAttribute]
    ) -> any TelemetrySpan {
        startSpan(name: name, attributes: attributes.stringified())
    }

    @inline(always) public func startSpan(
        name: String,
        attributes: [String: String],
        parent: String?
    ) -> any TelemetrySpan {
        startSpan(name: name, attributes: attributes)
    }

    @inline(always) public func startSpan(
        name: String,
        attributes: [String: TelemetryAttribute],
        parent: String?
    ) -> any TelemetrySpan {
        startSpan(name: name, attributes: attributes.stringified(), parent: parent)
    }
}

extension Dictionary where Key == String, Value == TelemetryAttribute {
    /// String forms for string-only backends. Each value conversion is
    /// locale-invariant by ``TelemetryAttribute/stringValue``.
    @usableFromInline internal func stringified() -> [String: String] {
        [String: String](uniqueKeysWithValues: map { ($0.key, $0.value.stringValue) })
    }
}

/// Default no-op span with zero CPU overhead.
public struct NoOpTelemetrySpan: TelemetrySpan {
    @inline(always) public init() {}
    @inline(always) public func setAttribute(_ key: String, value: String) {}
    @inline(always) public func setAttribute(_ key: String, value: TelemetryAttribute) {}
    @inline(always) public func recordEvent(name: String, attributes: [String: String]) {}
    @inline(always) public func recordEvent(name: String, attributes: [String: TelemetryAttribute]) {}
    @inline(always) public func end(status: SpanStatus) {}
    @inline(always) public var traceParent: String? { nil }
    @inline(always) public func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan {
        NoOpTelemetrySpan()
    }
    @inline(always) public func traceChild(
        name: String, attributes: [String: TelemetryAttribute]
    ) -> any TelemetrySpan {
        NoOpTelemetrySpan()
    }
}

/// Default no-op tracer with zero CPU overhead.
public struct NoOpTelemetryTracer: TelemetryTracer {
    @inline(always) public init() {}
    @inline(always) public func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        NoOpTelemetrySpan()
    }
    @inline(always) public func startSpan(
        name: String,
        attributes: [String: String],
        parent: String?
    ) -> any TelemetrySpan {
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

    /// Lock-free flag mirroring whether a tracer is registered.
    ///
    /// Branch on this before building span attribute dictionaries: the no-op
    /// span calls inline away, but dictionary and string allocations at the
    /// call site do not. Checking `isEnabled` first keeps the uninstrumented
    /// hot path allocation-free.
    private let enabledFlag = Atomic(false)

    /// Whether a real tracer is registered.
    ///
    /// Cheap relaxed atomic load: safe to check on every trial, even in
    /// sub-millisecond loops.
    public var isEnabled: Bool { enabledFlag.load(ordering: .relaxed) }

    /// Starts a span with typed attributes, or returns `nil` when no tracer
    /// is registered.
    ///
    /// This is the only branching call sites need: the flag check, the tracer
    /// lock, and every attribute-dict allocation below it are all skipped on
    /// the disabled path. Attributes are `@autoclosure`, so the dictionary at
    /// the call site is built only when a span will actually start.
    ///
    /// String values pass through ``TelemetryAttribute/string(_:)``; there is
    /// deliberately no string-dictionary overload, so every call site reads
    /// the same way.
    public func span(
        name: String,
        attributes: @autoclosure () -> [String: TelemetryAttribute],
        parent: String? = nil
    ) -> (any TelemetrySpan)? {
        guard isEnabled else { return nil }
        return tracer.startSpan(name: name, attributes: attributes(), parent: parent)
    }

    /// The currently active tracer, or ``NoOpTelemetryTracer`` if none is registered.
    public var tracer: any TelemetryTracer {
        tracerState.withLock { $0 } ?? NoOpTelemetryTracer()
    }

    /// Registers a global telemetry tracer, or unregisters the active tracer by passing `nil`.
    ///
    /// - Parameter tracer: The tracer instance to register, or `nil` to restore no-op tracing.
    public func registerTracer(_ tracer: (any TelemetryTracer)?) {
        tracerState.withLock { $0 = tracer }
        enabledFlag.store(tracer != nil, ordering: .relaxed)
    }
}
