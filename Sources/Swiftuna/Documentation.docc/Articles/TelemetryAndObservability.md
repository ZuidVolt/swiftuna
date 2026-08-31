# Observability and telemetry

Instrument Swiftuna optimization runs with tracing, OpenTelemetry, and structured logging.

## Overview

In production optimization pipelines, monitoring trial throughput, execution duration, parameter distributions, and failure rates is necessary. Swiftuna provides a zero-dependency observability layer built around three core types:
- ``SwiftunaTelemetry``: Thread-safe global telemetry registry.
- ``TelemetryTracer``: Factory interface for starting trace spans.
- ``TelemetrySpan``: Active trace span interface for recording attributes and completion status.

---

## Zero-overhead design

When no custom tracer is registered, calls to `SwiftunaTelemetry.shared.tracer` resolve to ``NoOpTelemetryTracer`` with inline empty implementations. The compiler eliminates tracing calls entirely in release builds, incurring zero runtime CPU or memory overhead during sub-millisecond optimization loops.

---

## Implementing a custom telemetry tracer

To forward telemetry events to your logging pipeline or an OpenTelemetry collector, implement ``TelemetryTracer`` and ``TelemetrySpan``:

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
        switch status {
        case .ok:
            print("[Span Complete] \(name): \(attrs)")
        case .error(let message):
            print("[Span Failed] \(name): \(message) (\(attrs))")
        }
    }
}

final class ConsoleLoggingTracer: TelemetryTracer {
    func startSpan(name: String, attributes: [String: String]) -> any TelemetrySpan {
        ConsoleLoggingSpan(name: name, attributes: attributes)
    }
}
```

---

## Registering the global tracer

Register your tracer before launching optimization runs:

```swift
// 1. Register tracer globally
SwiftunaTelemetry.shared.registerTracer(ConsoleLoggingTracer())

// 2. Run optimization study with automatic instrumentation
let study = try Swiftuna.createStudy(name: "monitored_study")
try study.optimize(nTrials: 20) { trial in
    let x = try trial.suggest("x", in: -5.0...5.0)
    return x * x
}

// 3. Restore default no-op tracer when finished
SwiftunaTelemetry.shared.registerTracer(nil)
```

During each trial, Swiftuna automatically records:
- `study.name`: The identifier of the enclosing study.
- `trial.number`: The sequential index of the running trial.
- `trial.status`: Final completion status (`"complete"`, `"pruned"`, or `"fail"`).
- `trial.duration_ms`: Wall-clock execution duration of the trial evaluation closure.
