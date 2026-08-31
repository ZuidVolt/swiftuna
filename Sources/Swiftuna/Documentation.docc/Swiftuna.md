# ``Swiftuna``

Idiomatic Swift 6 bindings for the Rustuna and Optuna Hyperparameter Optimization (HPO) ecosystem.

## Overview

**Swiftuna** provides native Swift bindings for [Rustuna](https://github.com/optuna/rustuna) (`rustuna_core`), the high-performance Rust engine powering the Optuna ecosystem.

### Design Philosophy & Mentality

Swiftuna follows the design of official Optuna and Rustuna bindings—such as Python (`rustuna_pyo3`) and JavaScript (`rustuna_js`)—maintaining direct conceptual parity across studies, trials, distributions, samplers, pruners, and storage backends. 

Where Swift language features provide superior safety and performance, Swiftuna makes **opinionated, Swift-native architectural decisions**:
- **Compile-Time Trial Safety**: Leverages Swift 6 non-copyable types (`~Copyable`) so that `Trial` ownership prevents double-evaluation, use-after-free, or duplicate consumption at compile time.
- **Type-Safe Metadata & Constraints**: Replaces stringly-typed key-value dictionaries with compile-time ``AttributeKey`` and ``ConstraintKey`` schemas, supporting Swift enums and automatic `Codable` JSON serialization via ``CodableAttribute``.
- **Native Range Suggestion Ergonomics**: Utilizes idiomatic Swift standard library primitives like `ClosedRange<Double>` and `ClosedRange<Int>` instead of low/high parameter lists.
- **Zero-Overhead C ABI Interop**: Directly links `librustuna_ffi` with zero intermediate allocations, achieving 97–104 ms execution time per 1,000 trials.
- **Full Optuna Ecosystem Compatibility**: SQLite databases written by Swiftuna are 100% byte-compatible with Python Optuna (`RDBStorage`) and `optuna-dashboard`.

---

### Quickstart

```swift
import Swiftuna

// 1. Create a study
let study = try Swiftuna.createStudy(
    name: "quickstart_study",
    direction: .minimize
)

// 2. Run optimization loop
try study.optimize(nTrials: 100) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    let y = try trial.suggest("y", in: 0...5)
    return (x - 2.0) * (x - 2.0) + Double(y)
}

// 3. Retrieve best trial
if let best = try study.bestTrial {
    print("Optimal trial #\(best.number): value = \(best.value ?? 0.0)")
    print("Parameters: \(best.params)")
}
```

---

## Topics

### Guides & Learning Paths
Learn how to design, execute, and inspect optimization experiments.
- <doc:GettingStarted>
- <doc:AskAndTellGuide>
- <doc:SamplersAndPruners>
- <doc:ConstrainedOptimization>
- <doc:StorageAndDashboard>
- <doc:TypeSafeAttributes>
- <doc:TelemetryAndObservability>

### Optimization Core
The primary coordination engine and lifecycle types for executing studies and trials.
- ``Study``
- ``Trial``
- ``PersistedTrial``
- ``Direction``
- ``TrialState``

### Study Management & Factory APIs
Create, load, copy, and manage studies across persistent backends.
- ``createStudy(name:directions:storage:sampler:pruner:loadIfExists:)``
- ``createStudy(name:direction:storage:sampler:pruner:loadIfExists:)``
- ``loadStudy(name:storage:sampler:pruner:)``
- ``copyStudy(from:to:as:)``
- ``deleteStudy(named:in:)``
- ``getStudies(in:)``
- ``createTrial(state:value:values:params:userAttrs:constraints:intermediateValues:datetimeStart:datetimeComplete:)``

### Sampling Algorithms
Surrogate models and search strategies for generating candidate parameter configurations.
- ``Sampler``
- ``TPESampler``
- ``QMCSampler``
- ``GridSampler``
- ``NSGAIISampler``
- ``RandomSampler``

### Pruning & Early Stopping
Bandit algorithms and statistical rules for terminating unpromising trials early.
- ``Pruner``
- ``MedianPruner``
- ``PercentilePruner``
- ``ThresholdPruner``
- ``SuccessiveHalvingPruner``
- ``HyperbandPruner``
- ``PatientPruner``
- ``NopPruner``

### Persistence & Storage
Storage backends for saving experiments and integrating with Optuna Dashboard.
- ``StorageBackend``
- ``StudySummary``

### Type-Safe Metadata & Constraints
Compile-time schemas for trial metadata, custom enums, and mathematical constraints.
- ``AttributeKey``
- ``AttributeConvertible``
- ``ConstraintKey``
- ``CodableAttribute``

### Observability & Error Handling
Distributed tracing spans, telemetry registries, and error diagnostics.
- ``SwiftunaTelemetry``
- ``TelemetryTracer``
- ``TelemetrySpan``
- ``SpanStatus``
- ``NoOpTelemetryTracer``
- ``NoOpTelemetrySpan``
- ``SwiftunaError``
