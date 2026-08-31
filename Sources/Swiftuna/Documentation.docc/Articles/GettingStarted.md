# Getting Started with Swiftuna

Learn how to create studies, define search spaces, optimize objectives, and analyze trial results in Swift 6.

## Overview

**Swiftuna** brings high-performance Bayesian hyperparameter optimization (HPO) directly to Swift. Powered by Rustuna's embedded C ABI engine, Swiftuna provides type-safe Swift 6 ergonomics with under 100 ms latency per 1,000 trials.

### Installation

Add Swiftuna as a dependency in your `Package.swift` manifest:

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

## Step 1: Create a Study

A study manages optimization history and orchestrates parameter suggestions. You can create an in-memory study or persist it to SQLite:

```swift
import Swiftuna

// Single-objective minimization (default)
let study = try Swiftuna.createStudy(
    name: "learning_rate_search",
    direction: .minimize,
    storage: .inMemory
)
```

---

## Step 2: Define Objective & Suggest Hyperparameters

Pass an objective closure to ``Study/optimize(nTrials:timeout:objective:)-3gyl5``. Inside the closure, interact with the non-copyable ``Trial`` object to suggest parameters:

```swift
try study.optimize(nTrials: 100) { trial in
    // Continuous floating-point range (log scale)
    let learningRate = try trial.suggest("lr", in: 1e-5...1e-1, log: true)

    // Discrete integer range with step
    let batchSize = try trial.suggest("batch_size", in: 16...128, step: 16)

    // Categorical string choices
    let optimizer = try trial.suggest("optimizer", choices: ["adamw", "sgd", "rmsprop"])

    // Evaluate the model objective (e.g. validation loss)
    let loss = evaluateModel(lr: learningRate, batchSize: batchSize, opt: optimizer)
    return loss
}
```

---

## Step 3: Inspect the Optimal Results

Once optimization completes, query the best trial, parameter values, and objective score:

```swift
if let best = try study.bestTrial {
    print("Optimization finished in \(try study.trials.count) trials")
    print("Best Trial #\(best.number) achieved value: \(best.value ?? 0.0)")
    print("Optimal Parameters:")
    for (param, val) in best.params {
        print("  - \(param): \(val)")
    }
}
```

---

## Step 4: Parameter Importance & Sensitivity Analysis

Swiftuna includes built-in PED-ANOVA (Partial Dependence Analysis of Variance) to determine which hyperparameters have the highest impact on objective variance:

```swift
if case .success(let importances) = study.paramImportances() {
    print("Hyperparameter Importance Ranking:")
    for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
        print("  \(param): \(String(format: "%.1f%%", score * 100))")
    }
}
```

---

## Next Steps

- Explore the manual Ask-and-Tell interface: <doc:AskAndTellGuide>
- Compare available sampling algorithms and early stopping pruners: <doc:SamplersAndPruners>
- Enforce constraints and multi-objective Pareto optimization: <doc:ConstrainedOptimization>
- Real-time visualization with SQLite and Optuna Dashboard: <doc:StorageAndDashboard>
- Type-safe user attributes and custom Codable metadata: <doc:TypeSafeAttributes>
