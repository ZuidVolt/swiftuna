# Swiftuna

[![Swift 6.0](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-blue.svg?style=flat)](#installation)
[![Optuna Compatible](https://img.shields.io/badge/Optuna%20Dashboard-Compatible-blueviolet.svg)](#optuna-dashboard-integration)

Swiftuna provides Swift 6 bindings for [Rustuna](https://github.com/optuna/rustuna) (`rustuna_core`), the Rust rewrite of the engine behind Optuna.

---

## Design philosophy

Swiftuna follows the design of official Optuna and Rustuna bindings, such as Python (`rustuna_pyo3`) and JavaScript (`rustuna_js`). It keeps the same core concepts for studies, trials, distributions, samplers, pruners, and storage backends.

Where Swift offers better safety or ergonomics, Swiftuna makes deliberate language-first choices:

- **Non-copyable trials (`~Copyable`).** Swift 6 move semantics prevent use-after-free, double evaluation, and concurrent consumption at compile time.
- **Static attribute and constraint keys.** Typed `AttributeKey` and `ConstraintKey` declarations replace stringly-typed dictionaries with compile-time checks, Swift enum support, and automatic `Codable` JSON serialization.
- **Standard library ranges.** Parameter suggestions use `ClosedRange<Double>` and `ClosedRange<Int>` instead of raw lower and upper arguments.
- **Direct C ABI link.** Static linking against `librustuna_ffi` avoids intermediate allocation overhead.
- **Optuna storage compatibility.** SQLite databases created by Swiftuna match Optuna `RDBStorage` byte-for-byte, so `optuna-dashboard` works out of the box.

---

## Installation

Add Swiftuna to your `Package.swift` dependencies:

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MyOptimizer",
    platforms: [.macOS(.v13), .iOS(.v16)],
    dependencies: [
        .package(url: "https://github.com/ZuidVolt/swiftuna.git", branch: "main")
    ],
    targets: [
        .executableTarget(
            name: "MyOptimizer",
            dependencies: [
                .product(name: "Swiftuna", package: "swiftuna")
            ]
        )
    ]
)
```

---

## Quickstart

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

## Core features and examples

### Ask-and-tell and structured concurrency

Decouple parameter generation (`ask()`) from evaluation (`tell(consuming:value:state:)`) for distributed runners, batch GPU jobs, or Swift 6 `TaskGroup` workers. `tell` has three overloads: single value, multi-objective `values:`, and state-only (for `.fail` / `.pruned`):

```swift
let storage = StorageBackend.sqlite(path: "concurrent_study.db")
let study = try Swiftuna.createStudy(name: "parallel_hpo", storage: storage, loadIfExists: true)

try await withThrowingTaskGroup(of: Void.self) { group in
    for workerID in 1...4 {
        group.addTask {
            for _ in 1...25 {
                var trial = try study.ask()
                let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
                let loss = await evaluateModelOnGPU(lr: lr, worker: workerID)

                // Consumes trial; compiler prevents any further use
                try study.tell(consuming: trial, value: loss)
            }
        }
    }
    try await group.waitForAll()
}
```

Or use the built-in concurrent `optimize` (controls `concurrency`, defaults to `activeProcessorCount`):

```swift
try await study.optimize(nTrials: 100, concurrency: 4) { trial in
    let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
    return await evaluateModelOnGPU(lr: lr)
}
```

Optimization budgets: at least one of `nTrials` or `timeout` is required (`timeout: .seconds(1800)`), otherwise `optimize` throws `invalidArgument`. With `GridSampler`, `ask()` throws `SwiftunaError.searchSpaceExhausted` once the Cartesian product is drained — catch it to exit manual loops early.

### Sampling algorithms

```swift
// Tree-structured Parzen Estimator (TPE, default)
let tpe = TPESampler(seed: 42)

// Quasi-Monte Carlo (QMC) Sobol sequence up to 1,024 dimensions
let qmc = QMCSampler(seed: 123)

// Multi-objective NSGA-II for Pareto frontier discovery
let nsga = NSGAIISampler(
    populationSize: 50,
    crossoverProb: 0.9,
    swappingProb: 0.5,
    seed: 42
)

// Uniform random search baseline
let random = RandomSampler(seed: 123)

// Grid search over discrete spaces (note the explicit ValueList type)
let searchSpace: [String: GridSampler.ValueList] = [
    "lr": [0.01, 0.001],
    "batch_size": [32, 64],
    "optimizer": .init(categorical: ["adam", "sgd"]),
]
let grid = GridSampler(searchSpace: searchSpace, seed: 42)
```

`seed` is `UInt64?` on every sampler (`nil` = non-deterministic). Pass a sampler to `createStudy(sampler:)`; single-objective studies default to `TPESampler()`, multi-objective (`directions:`) studies default to `NSGAIISampler()`.

Categorical suggestions preserve Swift types without casting:

```swift
enum Optimizer: String, CaseIterable, Sendable { case adamw, sgd, rmsprop }

try study.optimize(nTrials: 100) { trial in
    let lr = try trial.suggest("lr", in: 1e-5...1e-1, log: true)
    let batchSize = try trial.suggest("batch_size", choices: [32, 64, 128]) // Int
    let optimizer = try trial.suggest("optimizer", choices: Optimizer.allCases) // Optimizer
    return trainAndEvaluate(lr: lr, batchSize: batchSize, opt: optimizer)
}
```

> Note: `Trial` is `~Copyable` and `suggest` is `mutating`. Inside `optimize` the trial is passed as `inout`, so you call `try trial.suggest(...)` directly without `&`.

### Early stopping pruners

Stop unpromising training runs early to save compute. Available pruners: `NopPruner` (default, never prunes), `MedianPruner`, `PercentilePruner`, `ThresholdPruner`, `SuccessiveHalvingPruner` (ASHA), `HyperbandPruner`, `PatientPruner`:

```swift
let pruner = HyperbandPruner(minResource: 1, maxResource: 80, reductionFactor: 3)
let study = try Swiftuna.createStudy(pruner: pruner)

try study.optimize(nTrials: 50) { trial in
    var model = initModel()
    for epoch in 1...80 {
        let valLoss = model.trainEpoch()

        // Option 1: automatic — throws `SwiftunaError.trialPruned` when triggered
        try trial.report(valLoss, step: epoch, pruneIfWorse: true)
    }
    return model.finalEvaluation()
}

// Option 2: manual inspection for cleanup / checkpointing
try study.optimize(nTrials: 50) { trial in
    var model = initModel()
    for epoch in 1...80 {
        let valLoss = model.trainEpoch()
        try trial.report(valLoss, step: epoch)
        if try trial.shouldPrune {
            try trial.prune()
        }
    }
    return model.finalEvaluation()
}
```

### Constrained optimization

Enforce mathematical constraints ($c_i(x) \le 0.0$) without penalty parameters:

```swift
extension ConstraintKey {
    static let memoryBound = ConstraintKey("memory_mb_bound")
}

try study.optimize(nTrials: 100) { trial in
    let batchSize = try trial.suggest("batch_size", in: 16...512)
    let (loss, peakMemoryMB) = evaluateModel(batchSize: batchSize)

    // Satisfied when peakMemoryMB <= 4096 MB. String literals work too:
    // trial[constraint: "power_watts_bound"] = powerWatts - 150.0
    trial[constraint: .memoryBound] = peakMemoryMB - 4096.0

    return loss
}

// Get the best feasible result
let allTrials = try study.trials
if let bestFeasible = allTrials.bestFeasible(direction: .minimize) {
    print("Best feasible parameters: \(bestFeasible.params)")
    print("Memory slack: \(bestFeasible[constraint: .memoryBound] ?? 0.0)")
}
```

For custom key types, conform to `ConstraintKeyProtocol` (`static var name: String`) and use `trial[MyKey.self] = ...` / `trial.setConstraint(_:value:)`. Query with `isFeasible`, `feasible()`, `bestFeasibleTrial`, and multi-objective Pareto frontiers via `try study.bestTrials`.

### Multi-objective optimization

```swift
let study = try Swiftuna.createStudy(
    name: "accuracy_vs_latency",
    directions: [.maximize, .minimize],
    sampler: NSGAIISampler(populationSize: 40)
)

try study.optimize(nTrials: 100) { trial -> [Double] in
    let depth = try trial.suggest("depth", in: 2...12)
    return [computeAccuracy(depth: depth), measureLatency(depth: depth)]
}

let paretoFront = try study.bestTrials
```

### Type-safe user attributes

Declare static schema keys for trial metadata, custom enums, and `Codable` structs. `AttributeKey<Value>` is a generic struct; custom key types conform to `AttributeKeyProtocol`:

```swift
public enum ModelArchitecture: String, AttributeConvertible, Sendable {
    case resnet18, resnet50, convnext
}

extension AttributeKey where Value == ModelArchitecture {
    static let architecture = AttributeKey<ModelArchitecture>("architecture")
}

public struct HardwareInfo: Codable, Sendable {
    public let device: String
    public let vramGB: Double
}

extension AttributeKey where Value == CodableAttribute<HardwareInfo> {
    static let hardware = AttributeKey<CodableAttribute<HardwareInfo>>("hardware")
}

try study.optimize(nTrials: 50) { trial in
    // Type-safe writes on active trial (dot-syntax)
    trial[.architecture] = .resnet50
    trial[.hardware] = CodableAttribute(HardwareInfo(device: "Apple M3 Max", vramGB: 36.0))
    // ...
    return train()
}

// Type-safe reads on completed trial
if let best = try study.bestTrial {
    let arch: ModelArchitecture? = best[.architecture]
    let hw: HardwareInfo? = best[.hardware]?.value
    // Typed param accessors: best.double(_:), best.int(_:),
    // best.float(_:), best.string(_:), best.bool(_:), best.param(_:as:)
}
```

Untyped access (`trial["key"]`, `trial[userAttr: "key"]`) remains available for dynamic keys. The same subscripts work on `Study` (study-level attrs) and `PersistedTrial`.

### Optuna Dashboard integration

Swiftuna supports `.inMemory`, `.sqlite(path:)` (byte-compatible with Optuna `RDBStorage`), and `.journal(path:)` (lockless append-only log for clusters/NFS). `sqlite(url:)` / `journal(url:)` URL conveniences exist:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///experiments.db
```

Open `http://127.0.0.1:8080` in a browser to inspect interactive optimization history plots, Pareto frontiers, and parameter slice curves.

Manage studies programmatically:

```swift
let storage = StorageBackend.sqlite(path: "experiments.db")
let summaries = try storage.studies()          // [StudySummary]
let study = try Swiftuna.loadStudy(name: "resnet_cifar10", storage: storage)
let copy = try study.copy(to: .inMemory, as: "fast_sweep")
try Swiftuna.copyStudy(fromName: "a", fromStorage: storage, toStorage: .inMemory)
try Swiftuna.deleteStudy(named: "old_experiment", in: storage)
try study.enqueue(["learning_rate": 0.01, "batch_size": 32]) // warm-start
try study.addTrial(Swiftuna.createTrial(value: 0.15, params: ["learning_rate": 0.001]))
```

### Parameter importance analysis

PED-ANOVA (Partial Dependence Analysis of Variance) measures parameter sensitivity (requires ≥ 2 completed trials):

```swift
if case .success(let importances) = study.paramImportances(normalize: true, params: nil) {
    for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
        print("\(param): \(String(format: "%.1f%%", score * 100))")
    }
}
```

### Observability

`Study` results include `bestTrial / bestValue / bestParams / bestInternalParams / bestFeasibleTrial / bestTrials / trials(where:)`. Per-trial tracing (`study.name`, `trial.number`, `trial.status`, `trial.duration_ms`) is emitted via `SwiftunaTelemetry.shared` and is zero-overhead with the default `NoOpTelemetryTracer` — register a custom `TelemetryTracer` to forward to OpenTelemetry/logging.

---

## Documentation

Read the full documentation on [DocC](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/):

- [Getting started](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/gettingstarted)
- [Ask-and-tell guide](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/askandtellguide)
- [Samplers and pruners](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/samplersandpruners)
- [Constrained and multi-objective optimization](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/constrainedoptimization)
- [Storage backends and dashboard](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/storageanddashboard)
- [Type-safe attributes](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/typesafeattributes)
- [Observability and telemetry](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/telemetryandobservability)

---

## Development

Swiftuna uses [`just`](https://github.com/casey/just) to run tasks:

```bash
# Build
just build

# Run tests
just test

# Build static DocC documentation for GitHub Pages
just docs-build

# Preview DocC documentation locally with live reload
just docs-preview 8080
```

---

## License

Swiftuna is open source under the [GNU LESSER GENERAL PUBLIC LICENSE](LICENSE).

### Credits
Swiftuna builds on [Rustuna](https://github.com/optuna/rustuna) and the algorithmic work of [Optuna](https://github.com/optuna/optuna).
