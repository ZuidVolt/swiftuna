# Observability & Telemetry

Instrument Swiftuna optimization runs with zero-overhead tracing, OpenTelemetry, and structured logging.

## Overview

In production machine learning systems, monitoring trial throughput, execution duration, and failure rates is essential. Swiftuna includes a zero-dependency observability layer built around ``SwiftunaTelemetry``, ``TelemetryTracer``, and ``TelemetrySpan``.

- **Zero-Overhead Default**: When no tracer is registered, calls resolve to inline no-ops (``NoOpTelemetryTracer``) with zero CPU overhead.
- **OpenTelemetry Ready**: Easily bridges to `swift-distributed-tracing` and OpenTelemetry collectors.

---

## The Telemetry Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Optimization Loop                       │
│    trial.suggest(...) ──> objective() ──> trial.tell()     │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│             SwiftunaTelemetry.shared.tracer                │
│    startSpan("swiftuna.trial", attributes: [...])          │
└──────────────┬──────────────────────────────┬──────────────┘
               │                              │
               ▼                              ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │  OpenTelemetry Span  │      │  Apple os.Logger     │
    └──────────────────────┘      └──────────────────────┘
```

---

## Implementing a Custom Telemetry Tracer

Conform to ``TelemetryTracer`` and ``TelemetrySpan`` to forward metrics to your monitoring pipeline:

```swift
import Swiftuna
import Foundation

final class ConsoleLoggingSpan: TelemetrySpan, @unchecked Sendable {
    private let name: String
    private var attrs: [String: String]

    init(name: String, attributes: [String: String]) {
        self.name = name
        self.attrs = attributes
        print("[Span Start] \(name) -> \(attributes)")
    }

    func setAttribute(_ key: String, value: String) {
        attrs[key] = value
    }

    func recordEvent(name: String, attributes: [String: String]) {
        print("  [Event] \(name): \(attributes)")
    }

    func end(status: SpanStatus) {
        print("[Span End] \(name) Status: \(status) Final Attributes: \(attrs)")
    }
}

final class ConsoleLoggingTracer: TelemetryTracer {
    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        ConsoleLoggingSpan(name: name, attributes: attributes)
    }
}
```

---

## Registering the Global Tracer

Register your tracer before starting studies:

```swift
// Register tracer globally
SwiftunaTelemetry.shared.registerTracer(ConsoleLoggingTracer())

let study = try Swiftuna.createStudy(name: "monitored_hpo")
try study.optimize(nTrials: 10) { trial in
    let x = try trial.suggest("x", in: -5.0...5.0)
    return x * x
}

// Restore default no-op tracing when done
SwiftunaTelemetry.shared.registerTracer(nil)
```
