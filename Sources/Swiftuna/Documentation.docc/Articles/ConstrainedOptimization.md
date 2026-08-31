# Constrained & Multi-Objective Optimization

Learn how to enforce mathematical constraints and discover Pareto-optimal trade-offs across multiple competing objectives.

## Overview

Real-world machine learning and systems engineering rarely have a single unconstrained goal. For example:
- **Constrained Problem**: Minimize classification loss subject to memory usage $\le 4 \text{ GB}$ and inference latency $\le 20 \text{ ms}$.
- **Multi-Objective Problem**: Maximize accuracy while simultaneously minimizing model size and inference power consumption.

Swiftuna natively supports both constrained optimization and multi-objective Pareto front analysis.

---

## Mathematical Constraint Formulation

In Swiftuna, constraints follow standard numerical optimization convention:

$$c_i(x) \le 0.0 \iff \text{Feasible (Satisfied)}$$
$$c_i(x) > 0.0 \iff \text{Infeasible (Violation magnitude)}$$

Unlike naive penalty function methods (which distort the objective landscape and require manual hyperparameter tuning), Swiftuna's samplers handle constraints natively without penalty hyperparameters:
- ``TPESampler``: Feasible trials are strictly prioritized when building Parzen density models; infeasible trials are ordered by total violation magnitude.
- ``NSGAIISampler``: Implements Deb's constrained-domination tournament selection.

### Declaring Strongly-Typed Constraint Keys

Define static ``ConstraintKey`` types to eliminate string typos:

```swift
import Swiftuna

public enum MemoryBound: ConstraintKey {
    public static let name = "memory_mb_bound"
}

public enum LatencyBound: ConstraintKey {
    public static let name = "latency_ms_bound"
}
```

### Recording Constraints in an Objective Closure

```swift
let study = try Swiftuna.createStudy(
    name: "constrained_resnet",
    direction: .minimize
)

try study.optimize(nTrials: 100) { trial in
    let batchSize = try trial.suggest("batch_size", in: 16...512)
    let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)

    let (valLoss, memMB, latencyMs) = evaluateModel(batchSize: batchSize, lr: lr)

    // Enforce memMB <= 4096 MB and latencyMs <= 25.0 ms
    trial[constraint: MemoryBound.self] = memMB - 4096.0
    trial[constraint: LatencyBound.self] = latencyMs - 25.0

    return valLoss
}
```

### Querying Feasible Results

Use the functional sequence extensions to inspect feasible vs infeasible trials:

```swift
let allTrials = try study.trials

// Filter only trials meeting all constraints
let feasibleTrials = allTrials.feasible().completed()
print("Feasible trials found: \(feasibleTrials.count) / \(allTrials.count)")

if let bestFeasible = allTrials.bestFeasible(direction: .minimize) {
    print("Optimal feasible trial #\(bestFeasible.number) with loss \(bestFeasible.value ?? 0.0)")
    print("Parameters: \(bestFeasible.params)")
}
```

---

## Multi-Objective Pareto Optimization

When optimizing multiple objectives, no single solution is universally best. Instead, the optimizer seeks the **Pareto frontier**—the set of non-dominated solutions where improving one objective degrades another.

### Creating a Multi-Objective Study

Specify an array of ``Direction`` values corresponding to each objective:

```swift
let study = try Swiftuna.createStudy(
    name: "accuracy_vs_latency",
    directions: [.maximize, .minimize], // Obj 0: Accuracy, Obj 1: Latency (ms)
    sampler: NSGAIISampler(populationSize: 40)
)

try study.optimize(nTrials: 100) { trial in
    let depth = try trial.suggest("depth", in: 2...12)
    let width = try trial.suggest("width", in: 32...256, step: 32)

    let accuracy = computeAccuracy(depth: depth, width: width)
    let latency = measureLatency(depth: depth, width: width)

    return [accuracy, latency]
}
```

### Extracting the Pareto Frontier

Retrieve non-dominated solutions via ``Study/bestTrials``:

```swift
let paretoFront = try study.bestTrials
print("Found \(paretoFront.count) Pareto-optimal trade-off configurations:")

for trial in paretoFront {
    let acc = trial.values[0]
    let lat = trial.values[1]
    print("Trial #\(trial.number): Accuracy = \(String(format: "%.2f%%", acc * 100)), Latency = \(String(format: "%.1f ms", lat))")
}
```
