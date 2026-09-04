# Observability and telemetry

Instrument Swiftuna optimization runs with tracing, OpenTelemetry, and structured logging.

## Overview

In production optimization pipelines, monitoring trial throughput, execution duration, parameter distributions, and failure rates is necessary. Swiftuna provides a zero-dependency observability layer built around three core types:
- ``SwiftunaTelemetry``: Thread-safe global telemetry registry.
- ``TelemetryTracer``: Factory interface for starting trace spans.
- ``TelemetrySpan``: Active trace span interface for recording attributes and completion status.

---

## Zero-overhead design

When no custom tracer is registered, every instrumented call site branches on `SwiftunaTelemetry.shared.isEnabled` first: one relaxed atomic load (~4ns), zero allocations, zero locks. The no-op span calls would inline away, but the attribute dictionaries and formatted strings built at call sites would not — so the flag guards construction, not just invocation. Disabled-path overhead is benchmarked, not asserted: see Benchmark 12.

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

`Trial.report` heartbeats land on the same span as `trial.report` events (`trial.step`, `trial.value`), and every `shouldPrune` evaluation opens a `swiftuna.pruner` child span (pruner evaluation can fetch every trial over FFI, so its cost is worth attributing) plus a `trial.prune_vote` event with the verdict. Manual ask/tell trials carry no span — events flow only under `optimize`. The disabled path pays one nil check per call.

Completed trial spans also carry the sampled hyperparameters as `param.*` attributes (`param.x`, `param.layers`, …), so dashboards can slice outcomes by what was tried, not just when. Local runs record them from the `suggest` calls; distributed workers emit them from the trial payload. Same keys, same types, both paths — and on every terminal path, not just `complete`: pruned and failed trials map the search space too.

### Manual ask/tell is dark (by design, not by accident)

No ambient span exists outside `optimize`: the trial struct only carries one when `optimize` attaches it. So manual ask/tell trials emit nothing even with a tracer registered — silence indistinguishable from a broken adapter unless you know. Per-step `report` events and pruner votes are optimize-only for the same reason.

To instrument a manual loop, own the span lifecycle yourself and set explicitly what `optimize` would have derived:

```swift
let tracer = SwiftunaTelemetry.shared.tracer
var trial = try study.ask()
let span = tracer.startSpan(
    name: "swiftuna.trial",
    attributes: ["study.name": .string(study.name), "trial.number": .int(trial.number)]
)
let x = try trial.suggest("x", in: 0.0...1.0)
span.setAttribute("param.x", value: .double(x))
try study.tell(consuming: trial, value: x * x)
span.setAttribute("trial.status", value: "complete")
span.end(status: .ok)
```

## Typed attributes

Numeric and boolean values travel as ``TelemetryAttribute`` (`.int`, `.double`, `.bool`) with types intact for backend aggregation — `trial.number`, `trial.duration_ms` (a raw double, never locale-formatted), `trial.step`, `trial.value`, `trial.prune_vote`, `trial.distributed`, and every `param.*`. String-only backends receive locale-invariant string forms automatically, and every typed method has a stringifying default, so existing tracer conformances compile unchanged. Typed backends override the typed overloads; a scratch OTEL check harness verified the mapping preserves types end to end (September 2026: 4/4 worker spans parented to sample spans, 8 typed keys, traceparent round-trip).

Attribute presence is total: every key appears on every span, on every path. `trial.distributed` is explicitly `false` on local spans, never absent — absence must not mean anything, so backend queries can filter on the key without guessing.

---

## Distributed telemetry

`SwiftunaDistributed` emits the same `swiftuna.trial` spans from workers, so local and distributed runs share one dashboard. Each process registers its own tracer through the same global registry: the coordinator host instruments sampling and recording, every worker instruments evaluation.

| Span | Opened by | Key attributes |
| --- | --- | --- |
| `swiftuna.trial` | worker, at checkout | `study.name`, `trial.number`, `trial.distributed="true"`, then `trial.status`, `trial.duration_ms` |
| `swiftuna.coordinator.sample` | coordinator, per `ask` | `study.name`, `trial.number` |
| `swiftuna.coordinator.record` | coordinator, per `tell` | `study.name`, `trial.number` |
| `swiftuna.coordinator.reap` | coordinator, per reap batch | `study.name`, plus one `trial.lease_expired` event per trial |

Report heartbeats become `trial.report` events (`trial.step`, `trial.value`) on the worker span. Payloads predating the new fields still decode: `studyName` defaults to `""` (attribute omitted) and `traceParent` to `nil`.

### Cross-process trace linkage

The coordinator forwards its sample span's `traceParent` through the trial payload, and the worker parents its trial span to it. Backends resolve the opaque string to a remote span context; backends without context support ignore it and correlate by `study.name` + `trial.number` instead. Existing tracer conformances keep compiling: `traceParent`, `traceChild`, and `startSpan(parent:)` all have no-op defaults.

```swift
import Swiftuna

// Adapter sketch: map the core onto your OpenTelemetry tracer.
final class OtelBridgeSpan: TelemetrySpan, Sendable {
    var traceParent: String? { /* remote SpanContext -> traceparent */ }
    func traceChild(name: String, attributes: [String: String]) -> any TelemetrySpan {
        /* start child of this span's context */ NoOpTelemetrySpan()
    }
    func setAttribute(_ key: String, value: String) { /* set OTel attribute */ }
    func recordEvent(name: String, attributes: [String: String]) { /* add OTel event */ }
    func end(status: SpanStatus) { /* .ok -> .ok, .error -> Status.error */ }
}
```

### Zero-overhead contract

Branch on `SwiftunaTelemetry.shared.isEnabled` before building attribute dictionaries: no-op calls inline away, but call-site allocations do not. The flag is a relaxed atomic load (~4ns), and the disabled path allocates nothing — verified by Benchmark 12 in the microbench harness (enabled/disabled ratio 1.01x in release).
