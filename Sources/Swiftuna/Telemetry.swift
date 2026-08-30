import Foundation

#if canImport(os)
    import os
#endif

/// Status of an active telemetry span.
public enum SpanStatus: Sendable, Equatable {
    case ok
    case error(String)
}

/// Represents an active observability trace span.
/// Designed for 100% zero-dependency compatibility with OpenTelemetry and `swift-distributed-tracing`.
public protocol TelemetrySpan: Sendable {
    func setAttribute(_ key: String, value: String)
    func recordEvent(name: String, attributes: [String: String])
    func end(status: SpanStatus)
}

/// Tracer interface for instrumenting Swiftuna optimization operations.
public protocol TelemetryTracer: Sendable {
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

/// Thread-safe registry for Swiftuna telemetry and observability.
public final class SwiftunaTelemetry: Sendable {
    public static let shared = SwiftunaTelemetry()

    #if canImport(os)
        internal static let logger = Logger(subsystem: "org.swiftuna", category: "optimization")
        private let lock = OSAllocatedUnfairLock<(any TelemetryTracer)?>(initialState: nil)
    #else
        private final class Box: @unchecked Sendable {
            var tracer: (any TelemetryTracer)?
            let lock = NSLock()
        }
        private let box = Box()
    #endif

    public var tracer: any TelemetryTracer {
        #if canImport(os)
            lock.withLock { $0 } ?? NoOpTelemetryTracer()
        #else
            box.lock.lock()
            defer { box.lock.unlock() }
            return box.tracer ?? NoOpTelemetryTracer()
        #endif
    }

    public func registerTracer(_ tracer: (any TelemetryTracer)?) {
        #if canImport(os)
            lock.withLock { $0 = tracer }
        #else
            box.lock.lock()
            box.tracer = tracer
            box.lock.unlock()
        #endif
    }
}
