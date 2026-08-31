# Getting started with Swiftuna

Create studies, define search spaces, optimize objective functions, and analyze trial results.

### Installation

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

## 1. Create a study

A study manages trial history, orchestrates parameter suggestions, and tracks the best results. You can create an in-memory study for fast single-process jobs, or store trials in SQLite for persistence and visualization.

```swift
import Swiftuna

// Create an in-memory study for single-objective minimization
let study = try Swiftuna.createStudy(
    name: "learning_rate_search",
    direction: .minimize,
    storage: .inMemory
)
```

> Tip: For reproducible runs, pass a seeded sampler such as `TPESampler(seed: 42)` to `createStudy`.

---

## 2. Define search spaces and suggest hyperparameters

Pass an evaluation closure to ``Study/optimize(nTrials:timeout:objective:)-3gyl5``. Inside the closure, interact with the non-copyable ``Trial`` to request parameter candidates.

Swiftuna supports three parameter distribution types:

### Floating-point parameters
Use `suggest(_:in:step:log:)` with a `ClosedRange<Double>`:
- **Uniform scale:** When the optimal value is expected to vary linearly (e.g. momentum in `0.8...0.99`).
- **Log scale (`log: true`):** When exploring across multiple orders of magnitude (e.g. learning rate in `1e-5...1e-1`, weight decay in `1e-6...1e-2`).
- **Step discretization:** Round candidates to multiples of `step` (e.g. dropout rate in `0.0...0.5` with `step: 0.05`).

### Integer parameters
Use `suggest(_:in:step:log:)` with a `ClosedRange<Int>`:
- Great for batch sizes, layer counts, embedding dimensions, or training epoch limits.

### Categorical parameters
Use `suggest(_:choices:)` with a list of strings or enum cases:
- Useful for optimizer selection (`["adamw", "sgd", "rmsprop"]`), activation functions (`["relu", "gelu", "swish"]`), or normalization methods.

```swift
try study.optimize(nTrials: 100) { trial in
    let lr = try trial.suggest("lr", in: 1e-5...1e-1, log: true)
    let batchSize = try trial.suggest("batch_size", in: 16...128, step: 16)
    let optimizer = try trial.suggest("optimizer", choices: ["adamw", "sgd", "rmsprop"])

    let validationLoss = trainAndEvaluate(lr: lr, batchSize: batchSize, opt: optimizer)
    return validationLoss
}
```

---

## 3. Controlling optimization budgets

You can cap optimization by trial count, wall-clock time, or both:

```swift
// Stop after 200 trials or after 30 minutes, whichever happens first
try study.optimize(
    nTrials: 200,
    timeout: .seconds(1800)
) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return evaluateModel(x)
}
```

If neither `nTrials` nor `timeout` is provided, `optimize` throws an `invalidArgument` error to prevent unintended infinite loops.

---

## 4. Inspecting optimal results

Once the optimization loop completes, read the best trial, parameter values, and objective score:

```swift
if let best = try study.bestTrial {
    print("Optimization completed across \(try study.trials.count) trials")
    print("Best trial #\(best.number): objective value = \(best.value ?? 0.0)")
    print("Optimal parameters:")
    for (param, val) in best.params {
        print("  - \(param): \(val)")
    }
}
```

You can also inspect all recorded trials or filter them using functional sequence extensions:

```swift
let completed = try study.trials.completed()
let topFive = completed.sortedByValue().prefix(5)

for trial in topFive {
    print("Trial #\(trial.number): \(trial.value ?? 0.0)")
}
```

---

## 5. Hyperparameter importance with PED-ANOVA

Swiftuna includes built-in PED-ANOVA (Partial Dependence Analysis of Variance) to determine which hyperparameters contribute most to the variance in your objective score:

```swift
if case .success(let importances) = study.paramImportances() {
    print("Hyperparameter sensitivity ranking:")
    for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
        print("  \(param): \(String(format: "%.1f%%", score * 100))")
    }
}
```

This analysis helps you drop uninfluential parameters in subsequent optimization rounds and narrow down high-impact search intervals.

---

## Next steps

- Learn the manual ask-and-tell loop and concurrency patterns: <doc:AskAndTellGuide>
- Choose the best sampler and pruner for your problem: <doc:SamplersAndPruners>
- Enforce constraints and multi-objective Pareto trade-offs: <doc:ConstrainedOptimization>
- Persist studies in SQLite and launch Optuna Dashboard: <doc:StorageAndDashboard>
- Declare type-safe user attributes and Codable schemas: <doc:TypeSafeAttributes>
