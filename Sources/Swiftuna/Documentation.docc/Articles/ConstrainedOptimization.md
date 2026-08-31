# Constrained and multi-objective optimization

Enforce mathematical constraints and discover Pareto trade-offs across competing objectives.

## Overview

Real-world engineering problems rarely have a single unconstrained objective:
- **Constrained optimization:** Minimize loss while ensuring peak memory $\le 4 \text{ GB}$ and inference latency $\le 20 \text{ ms}$.
- **Multi-objective optimization:** Maximize classification accuracy while minimizing model parameters and energy consumption.

---

## Constraint formulation

Swiftuna follows the standard mathematical optimization convention:

$$c_i(x) \le 0.0 \iff \text{Feasible (Satisfied)}$$
$$c_i(x) > 0.0 \iff \text{Infeasible (Violation magnitude)}$$

### Why penalty functions fall short
Traditional penalty formulations ($L = \text{loss} + \lambda \sum \max(0, c_i(x))^2$) require careful tuning of the penalty weight $\lambda$. If $\lambda$ is too small, the optimizer violates constraints; if too large, it creates steep numerical ravines that trap Bayesian surrogates.

Swiftuna avoids penalty weights entirely:
- ``TPESampler`` splits trials into feasible and infeasible groups. Density estimation focuses on feasible solutions, ranking infeasible trials solely by their violation magnitude.
- ``NSGAIISampler`` uses Deb's constrained-domination rules during tournament selection. Feasible solutions always dominate infeasible solutions, regardless of objective value.

---

## Declaring typed constraint keys

Define static ``ConstraintKey`` types to avoid string typos and enforce clear schema definitions:

```swift
import Swiftuna

public enum MemoryBound: ConstraintKey {
    public static let name = "memory_mb_bound"
}

public enum LatencyBound: ConstraintKey {
    public static let name = "latency_ms_bound"
}
```

### Recording constraints during trial evaluation

Compute the difference between measured metrics and your threshold:

```swift
let study = try Swiftuna.createStudy(
    name: "constrained_tuning",
    direction: .minimize
)

try study.optimize(nTrials: 100) { trial in
    let batchSize = try trial.suggest("batch_size", in: 16...512)
    let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)

    let (valLoss, memMB, latencyMs) = evaluateModel(batchSize: batchSize, lr: lr)

    // Satisfied when memMB <= 4096 MB and latencyMs <= 25.0 ms
    trial[constraint: MemoryBound.self] = memMB - 4096.0
    trial[constraint: LatencyBound.self] = latencyMs - 25.0

    return valLoss
}
```

### Querying feasible results

Use analytical sequence extensions to filter and find the best feasible trial:

```swift
let allTrials = try study.trials

let feasibleCount = allTrials.feasible().completed().count
print("Feasible trials found: \(feasibleCount) / \(allTrials.count)")

if let bestFeasible = allTrials.bestFeasible(direction: .minimize) {
    print("Optimal feasible trial #\(bestFeasible.number) with loss \(bestFeasible.value ?? 0.0)")
    print("Parameters: \(bestFeasible.params)")
    print("Constraints: \(bestFeasible.constraints)")
}
```

---

## Multi-objective Pareto optimization

In multi-objective optimization, no single configuration minimizes all objectives at once. Instead, the optimizer finds the **Pareto frontier**, the set of non-dominated solutions where improving one objective necessarily worsens another.

### Creating a multi-objective study

Pass an array of ``Direction`` values matching the returned objective vector:

```swift
let study = try Swiftuna.createStudy(
    name: "accuracy_vs_latency",
    directions: [.maximize, .minimize], // Objective 0: Accuracy, Objective 1: Latency (ms)
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

### Extracting the Pareto frontier

Read non-dominated solutions via ``Study/bestTrials``:

```swift
let paretoFront = try study.bestTrials
print("Discovered \(paretoFront.count) Pareto-optimal trade-offs:")

for trial in paretoFront {
    let accuracy = trial.values[0]
    let latency = trial.values[1]
    print("Trial #\(trial.number): Accuracy = \(String(format: "%.2f%%", accuracy * 100)), Latency = \(String(format: "%.1f ms", latency))")
}
```
