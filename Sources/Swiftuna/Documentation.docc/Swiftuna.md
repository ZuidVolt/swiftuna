# ``Swiftuna``

Idomatic swift 6 bindings for subset of rustuna / optuna apis

## Overview

**Swiftuna** brings production-grade Bayesian hyperparameter optimization (HPO) directly to the Swift ecosystem. It embeds Rustuna (`rustuna_core`) with as little overhead as possible using a C ABI bridge, delivering pure native Swift ergonomics and 97-104 ms latency per 1,000 trials—matching or exceeding pure Rust performance.

### Quickstart

```swift
import Swiftuna

// 1. Create a study
let study = try Swiftuna.createStudy(
    name: "quickstart_study",
    direction: .minimize
)

// 2. Run optimization
try study.optimize(nTrials: 100) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    let y = try trial.suggest("y", in: 0...5)
    return (x - 2.0) * (x - 2.0) + Double(y)
}

// 3. Retrieve best trial
let best = try study.bestTrial
print("Optimal x: \(best?.params["x"] ?? 0.0), value: \(try study.bestValue)")
```

## Topics

### Optimization Engine
- ``Study``
- ``Trial``
- ``PersistedTrial``
- ``Direction``
- ``TrialState``

### Study Lifecycle & Factory Methods
- ``createStudy(name:directions:storage:sampler:pruner:loadIfExists:)``
- ``createStudy(name:direction:storage:sampler:pruner:loadIfExists:)``
- ``loadStudy(name:storage:sampler:pruner:)``
- ``copyStudy(from:to:as:)``
- ``deleteStudy(named:in:)``
- ``getStudies(in:)``
- ``createTrial(state:value:values:params:userAttrs:constraints:intermediateValues:datetimeStart:datetimeComplete:)``

### Sampling Algorithms
- ``Sampler``
- ``TPESampler``
- ``QMCSampler``
- ``GridSampler``
- ``NSGAIISampler``
- ``RandomSampler``

### Pruners & Early Stopping
- ``Pruner``
- ``MedianPruner``
- ``PercentilePruner``
- ``ThresholdPruner``
- ``SuccessiveHalvingPruner``
- ``HyperbandPruner``
- ``PatientPruner``
- ``NopPruner``

### Persistent Storage
- ``StorageBackend``
- ``StudySummary``

### Type-Safe Attributes & Constraints
- ``AttributeKey``
- ``AttributeConvertible``
- ``ConstraintKey``
- ``CodableAttribute``

### Observability & Telemetry
- ``SwiftunaTelemetry``
- ``TelemetryTracer``
- ``TelemetrySpan``
- ``SpanStatus``
- ``NoOpTelemetryTracer``

### Error Handling
- ``SwiftunaError``
