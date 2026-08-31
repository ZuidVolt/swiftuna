# Swiftuna

[![Swift 6.0](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-blue.svg?style=flat)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DocC](https://img.shields.io/badge/Documentation-DocC-brightgreen.svg)](https://zuidvolt.github.io/swiftuna/documentation/swiftuna/)
[![Optuna Compatible](https://img.shields.io/badge/Optuna%20Dashboard-Compatible-blueviolet.svg)](#optuna-dashboard-integration)

Swiftuna provides Swift 6 bindings for [Rustuna](https://github.com/optuna/rustuna) (`rustuna_core`), the Rust rewrite of the engine behind Optuna. It links directly to `librustuna_ffi` through a C ABI bridge with zero heap copies, running 1,000 trials in 97 to 104 ms.

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
        .package(url: "https://github.com/ZuidVolt/swiftuna.git", from: "0.1.0")
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

Decouple parameter generation from evaluation for distributed runners, batch GPU jobs, or Swift 6 `TaskGroup` workers:

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

### Sampling algorithms

```swift
// Tree-structured Parzen Estimator (TPE)
let tpe = TPESampler(seed: 42)

// Quasi-Monte Carlo (QMC) Sobol sequence up to 1,024 dimensions
let qmc = QMCSampler(seed: 123)

// Multi-objective NSGA-II for Pareto frontier discovery
let nsga = NSGAIISampler(populationSize: 50, crossoverProb: 0.9)

// Grid search over discrete spaces
let grid = GridSampler(searchSpace: [
    "lr": [0.01, 0.001],
    "batch_size": [32, 64],
    "optimizer": .init(categorical: ["adam", "sgd"])
])
```

### Early stopping pruners

Stop unpromising training runs early to save compute:

```swift
let pruner = HyperbandPruner(minResource: 1, maxResource: 81, reductionFactor: 3)
let study = try Swiftuna.createStudy(pruner: pruner)

try study.optimize(nTrials: 50) { trial in
    var model = initModel()
    for epoch in 1...81 {
        let valLoss = model.trainEpoch()
        
        // Report intermediate score and prune if trajectory is poor
        try trial.report(valLoss, step: epoch, pruneIfWorse: true)
    }
    return model.finalEvaluation()
}
```

### Constrained optimization

Enforce mathematical constraints ($c_i(x) \le 0.0$) without penalty parameters:

```swift
public enum MemoryBound: ConstraintKey {
    public static let name = "memory_mb_bound"
}

try study.optimize(nTrials: 100) { trial in
    let batchSize = try trial.suggest("batch_size", in: 16...512)
    let (loss, peakMemoryMB) = evaluateModel(batchSize: batchSize)

    // Satisfied when peakMemoryMB <= 4096 MB
    trial[constraint: MemoryBound.self] = peakMemoryMB - 4096.0

    return loss
}

// Get the best feasible result
if let bestFeasible = try study.trials.bestFeasible(direction: .minimize) {
    print("Best feasible parameters: \(bestFeasible.params)")
}
```

### Type-safe user attributes

Declare static schema keys for trial metadata, custom enums, and `Codable` structs:

```swift
public enum ModelArchitecture: String, AttributeConvertible, Sendable {
    case resnet18, resnet50, convnext
}

public enum ArchitectureKey: AttributeKey {
    public typealias Value = ModelArchitecture
    public static let name = "architecture"
}

public struct HardwareInfo: Codable, Sendable {
    public let device: String
    public let vramGB: Double
}

public enum HardwareKey: AttributeKey {
    public typealias Value = CodableAttribute<HardwareInfo>
    public static let name = "hardware"
}

// Type-safe writes on active trial
trial[ArchitectureKey.self] = .resnet50
trial[HardwareKey.self] = CodableAttribute(HardwareInfo(device: "Apple M3 Max", vramGB: 36.0))

// Type-safe reads on completed trial
if let best = try study.bestTrial {
    let arch: ModelArchitecture? = best[ArchitectureKey.self]
    let hw: HardwareInfo? = best[HardwareKey.self]?.value
}
```

### Optuna Dashboard integration

Because Swiftuna writes standard Optuna SQLite tables, you can launch `optuna-dashboard` directly:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///experiments.db
```

Open `http://127.0.0.1:8080` in a browser to inspect interactive optimization history plots, Pareto frontiers, and parameter slice curves.

### Parameter importance analysis

PED-ANOVA (Partial Dependence Analysis of Variance) measures parameter sensitivity:

```swift
if case .success(let importances) = study.paramImportances() {
    for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
        print("\(param): \(String(format: "%.1f%%", score * 100))")
    }
}
```

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

Swiftuna is open source under the [MIT License](LICENSE).

### Credits
Swiftuna builds on [Rustuna](https://github.com/optuna/rustuna) and the algorithmic work of [Optuna](https://github.com/optuna/optuna).
