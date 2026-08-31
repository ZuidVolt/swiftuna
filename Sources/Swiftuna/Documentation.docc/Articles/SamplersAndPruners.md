# Samplers & Early Stopping Pruners

A comprehensive architectural guide to parameter sampling algorithms and early stopping pruners in Swiftuna.

## Overview

Hyperparameter optimization consists of two complementary components:
1. **Samplers** (Propose candidate parameter configurations)
2. **Pruners** (Terminate unpromising trial trajectories early)

---

## Sampler Comparison Matrix

| Sampler | Strategy | Best Use Case | Multi-Objective | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| ``TPESampler`` | Tree-structured Parzen Estimator | General continuous & discrete HPO | ✅ (MOTPE) | ✅ Feasibility Partitioning |
| ``QMCSampler`` | Quasi-Monte Carlo (Sobol) | Uniform space-filling, exploratory sweeps | ❌ Single | ❌ Unconstrained |
| ``GridSampler`` | Cartesian product grid | Small discrete search spaces, ablation studies | ❌ Single | ❌ Unconstrained |
| ``NSGAIISampler`` | Genetic Evolutionary Algorithm | Multi-objective Pareto frontier discovery | ✅ Native | ✅ Constrained-Domination |
| ``RandomSampler`` | Uniform random search | Fast unguided baseline | ✅ Multi-Objective | ❌ Unconstrained |

### 1. Tree-structured Parzen Estimator (``TPESampler``)
TPE fits two Gaussian Mixture Models over observed parameter values:
- $\ell(x)$: Density of parameters producing top-performing objective values
- $g(x)$: Density of remaining parameters

Candidates are selected to maximize the expected improvement ratio $\frac{\ell(x)}{g(x)}$.

```swift
// Seeded for reproducible optimization
let sampler = TPESampler(seed: 42)
let study = try Swiftuna.createStudy(sampler: sampler)
```

### 2. Quasi-Monte Carlo Sobol Sequence (``QMCSampler``)
QMC uses Antonov-Saleev Gray codes and Joe-Kuo direction numbers up to 1,024 dimensions to systematically eliminate space clustering and empty voids.

```swift
let sampler = QMCSampler(seed: 123)
let study = try Swiftuna.createStudy(sampler: sampler)
```

### 3. Exhaustive Grid Search (``GridSampler``)
Precomputes Cartesian products across discrete options, shuffling evaluation order deterministically:

```swift
let grid: [String: GridSampler.ValueList] = [
    "lr": [0.01, 0.001],
    "batch_size": [32, 64],
    "optimizer": .init(categorical: ["adam", "sgd"])
]
let sampler = GridSampler(searchSpace: grid, seed: 99)
```

### 4. Non-dominated Sorting Genetic Algorithm II (``NSGAIISampler``)
NSGA-II maintains a candidate population across generations, ranking individuals by non-domination rank and crowding distance to produce a diverse Pareto frontier.

```swift
let sampler = NSGAIISampler(
    populationSize: 50,
    crossoverProb: 0.9,
    swappingProb: 0.5,
    seed: 42
)
```

---

## Pruner Comparison Matrix

| Pruner | Mechanism | Best Use Case |
| :--- | :--- | :--- |
| ``MedianPruner`` | Stops trials worse than 50th percentile at same step | Deep learning training epochs |
| ``PercentilePruner`` | Stops trials outside top $P\%$ threshold | Aggressive pruning |
| ``SuccessiveHalvingPruner`` | Multi-armed bandit geometric rungs ($1/\eta$ retention) | Iterative resource allocation (ASHA) |
| ``HyperbandPruner`` | Multi-bracket Successive Halving | Large-scale neural network architecture search |
| ``PatientPruner`` | Grace-period delay around any pruner | Noisy learning curves, warm-up phases |
| ``ThresholdPruner`` | Absolute numerical upper/lower bounds | Safety limits, divergence prevention |
| ``NopPruner`` | No-op baseline (never prunes) | Debugging, fixed-budget evaluations |

### Successive Halving (``SuccessiveHalvingPruner``)
Trials start with `minResource` evaluations. At each geometric rung $r_k = \text{minResource} \cdot \eta^k$, only the top $1/\eta$ fraction is promoted.

```swift
let pruner = SuccessiveHalvingPruner(
    minResource: 1,      // First evaluation at epoch 1
    reductionFactor: 3,  // Promote top 1/3 at each rung
    minEarlyStoppingRate: 0
)
```

### Hyperband (``HyperbandPruner``)
Hyperband balances exploration vs exploitation by running multiple brackets of Successive Halving with varied early stopping aggressiveness:

```swift
let pruner = HyperbandPruner(
    minResource: 1,
    maxResource: 81,
    reductionFactor: 3
)
let study = try Swiftuna.createStudy(pruner: pruner)
```

### Patient Pruner (``PatientPruner``)
To prevent early stopping on temporary spikes or noisy loss plateaus, wrap any base pruner with a patience buffer:

```swift
let basePruner = MedianPruner(nStartupTrials: 5)
let robustPruner = PatientPruner(wrappedPruner: basePruner, patience: 3)
```
