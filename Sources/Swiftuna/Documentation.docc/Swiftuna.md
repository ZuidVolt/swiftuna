# ``Swiftuna``

Swift 6 bindings for the Optuna (Rustuna) Hyperparameter Optimization (HPO) ecosystem.

## Overview

Swiftuna provides native Swift bindings for [Rustuna](https://github.com/optuna/rustuna) (`rustuna_core`), the Rust rewrite of the engine behind Optuna.

### Design philosophy

Swiftuna follows the design of official Optuna and Rustuna bindings, such as Python (`rustuna_pyo3`) and JavaScript (`rustuna_js`). It keeps the same core concepts for studies, trials, distributions, samplers, pruners, and storage backends.

Where Swift offers better safety or ergonomics, Swiftuna makes deliberate language-first choices:

- **Non-copyable trials (`~Copyable`).** Swift 6 move semantics prevent use-after-free, double evaluation, and concurrent consumption at compile time.
- **Static attribute and constraint keys.** Typed `AttributeKey` and `ConstraintKey` declarations replace stringly-typed dictionaries with compile-time checks, Swift enum support, and automatic `Codable` JSON serialization.
- **Standard library ranges.** Parameter suggestions use `ClosedRange<Double>` and `ClosedRange<Int>` instead of raw lower and upper arguments.
- **Direct C ABI link.** Static linking against `librustuna_ffi` avoids intermediate allocation overhead
- **Optuna storage compatibility.** SQLite databases created by Swiftuna match Optuna `RDBStorage` byte-for-byte, so `optuna-dashboard` works out of the box.

---

### Quickstart

```swift
import Swiftuna

// 1. Create a study
let study = try Swiftuna.createStudy(
    name: "quickstart_study",
    direction: .minimize
)

// 2. Run the optimization loop
try study.optimize(nTrials: 100) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    let y = try trial.suggest("y", in: 0...5)
    return (x - 2.0) * (x - 2.0) + Double(y)
}

// 3. Inspect results
if let best = try study.bestTrial {
    print("Optimal trial #\(best.number): value = \(best.value ?? 0.0)")
    print("Parameters: \(best.params)")
}
```

---

## Topics

### Guides and tutorials
- <doc:GettingStarted>
- <doc:AskAndTellGuide>
- <doc:SamplersAndPruners>
- <doc:CustomSamplers>
- <doc:ConstrainedOptimization>
- <doc:StorageAndDashboard>
- <doc:TypeSafeAttributes>
- <doc:TelemetryAndObservability>
- <doc:DistributedOptimization>
- <doc:GPUAndMLX>

### Optimization core
- ``Study``
- ``Trial``
- ``PersistedTrial``
- ``Direction``
- ``TrialState``

### Study management and factory APIs
- ``createStudy(name:directions:storage:sampler:pruner:loadIfExists:)``
- ``createStudy(name:direction:storage:sampler:pruner:loadIfExists:)``
- ``loadStudy(name:storage:sampler:pruner:)``
- ``copyStudy(from:to:as:)``
- ``deleteStudy(named:in:)``
- ``getStudies(in:)``
- ``createTrial(state:value:values:params:userAttrs:constraints:intermediateValues:datetimeStart:datetimeComplete:)``

### Sampling algorithms
- ``Sampler``
- ``TPESampler``
- ``QMCSampler``
- ``GridSampler``
- ``NSGAIISampler``
- ``RandomSampler``
- ``CallbackSampler``
- ``CustomSampler``
- ``StudyHistory``

### Pruning and early stopping
- ``Pruner``
- ``MedianPruner``
- ``PercentilePruner``
- ``ThresholdPruner``
- ``SuccessiveHalvingPruner``
- ``HyperbandPruner``
- ``PatientPruner``
- ``NopPruner``

### Persistence and storage
- ``StorageBackend``
- ``StudySummary``

### Type-safe metadata and constraints
- ``ParameterValue``
- ``AttributeKey``
- ``AttributeConvertible``
- ``ConstraintKey``
- ``CodableAttribute``

### Observability and error handling
- ``SwiftunaTelemetry``
- ``TelemetryTracer``
- ``TelemetrySpan``
- ``TelemetryAttribute``
- ``SpanStatus``
- ``NoOpTelemetryTracer``
- ``NoOpTelemetrySpan``
- ``SwiftunaError``
