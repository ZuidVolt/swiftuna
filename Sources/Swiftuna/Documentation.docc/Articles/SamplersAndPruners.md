# Samplers and early stopping pruners

Guide to parameter sampling algorithms and early stopping pruners in Swiftuna.

## Overview

Hyperparameter optimization pairs two distinct mechanisms:
1. **Samplers** propose parameter configurations by building surrogate models from previous trial evaluations.
2. **Pruners** monitor intermediate step values (such as per-epoch loss) and terminate unpromising trials early to save compute time.

---

## Sampler comparison

| Sampler | Strategy | Best for | Multi-objective | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| ``TPESampler`` | Tree-structured Parzen Estimator | General continuous and discrete HPO | Yes (MOTPE) | Feasibility partitioning |
| ``QMCSampler`` | Quasi-Monte Carlo (Sobol) | Low-discrepancy space filling | Single only | No |
| ``GridSampler`` | Cartesian product grid | Small discrete spaces, ablation sweeps | Single only | No |
| ``NSGAIISampler`` | Genetic evolutionary algorithm | Multi-objective Pareto frontier discovery | Yes (Native) | Constrained-domination |
| ``RandomSampler`` | Uniform random search | Fast baseline, high-noise environments | Yes | No |
| ``CallbackSampler`` | Swift closures called per suggestion | Custom strategies, conditional spaces | Yes | No (falls through to tell) |

---

## Deep dive: Sampling algorithms

### Tree-structured Parzen Estimator (``TPESampler``)

TPE is the default sampler in Swiftuna. Instead of modeling the objective function $P(y \mid x)$ directly with Gaussian Processes, TPE uses Bayes' rule to model parameter distributions given the objective value:

$$P(x \mid y) = \begin{cases} \ell(x) & \text{if } y < y^* \\ g(x) & \text{if } y \ge y^* \end{cases}$$

Here, $y^*$ is a splitting quantile separating the top-performing trials from the rest. Candidates are sampled to maximize the Expected Improvement ratio:

$$\text{EI}(x) \propto \frac{\ell(x)}{g(x)}$$

TPE fits Parzen window density estimators (Gaussian Mixture Models) for continuous variables and categorical frequency tables for discrete variables.

```swift
// Default TPE sampler with optional seed for deterministic reproducibility
let sampler = TPESampler(seed: 42)
let study = try Swiftuna.createStudy(sampler: sampler)
```

### Quasi-Monte Carlo Sobol sequences (``QMCSampler``)

Quasi-Monte Carlo methods generate deterministic low-discrepancy sequences designed to cover multi-dimensional spaces more uniformly than pseudo-random sampling.

Swiftuna's `QMCSampler` uses Antonov-Saleev Gray codes and Joe-Kuo direction numbers up to 1,024 dimensions. QMC is effective for exploratory parameter sweeps and initial global coverage before switching to surrogate-guided search.

```swift
let sampler = QMCSampler(seed: 123)
let study = try Swiftuna.createStudy(sampler: sampler)
```

### Exhaustive grid search (``GridSampler``)

`GridSampler` precomputes the full Cartesian product across all parameter domains. When a seed is specified, evaluation order is shuffled deterministically:

```swift
let grid: [String: GridSampler.ValueList] = [
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64],
    "optimizer": .init(categorical: ["adamw", "sgd"])
]

let sampler = GridSampler(searchSpace: grid, seed: 42)
let study = try Swiftuna.createStudy(sampler: sampler)
```

### Genetic evolutionary search (``NSGAIISampler``)

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is designed for multi-objective optimization. It maintains a population of candidates across generations:
1. **Non-dominated Sorting:** Groups candidates into Pareto hierarchical ranks.
2. **Crowding Distance:** Favors solutions located in less crowded areas along the Pareto frontier to maintain exploration diversity.
3. **Constrained-Domination:** Automatically enforces constraints without penalty tuning.

```swift
let sampler = NSGAIISampler(
    populationSize: 50,
    crossoverProb: 0.9,
    swappingProb: 0.5,
    seed: 42
)
```

### Custom samplers in Swift

For strategies Rustuna doesn't ship, implement the suggestion in Swift: ``CallbackSampler`` for per-suggestion control (conditional spaces), ``CustomSampler`` for history-driven trial-start strategies, raw enqueue loops for one-off scripts. Full guide, Optuna/Rustuna comparison, and API boundaries: <doc:CustomSamplers>.

---

## Pruner comparison

| Pruner | Mechanism | Best for |
| :--- | :--- | :--- |
| ``MedianPruner`` | Stops trials performing below the 50th percentile at the same step | Neural network training loops |
| ``PercentilePruner`` | Stops trials outside a target top $P\%$ threshold | Aggressive resource filtering |
| ``SuccessiveHalvingPruner`` | Geometric resource rungs with $1/\eta$ retention (ASHA) | Resource allocation sweeps |
| ``HyperbandPruner`` | Multi-bracket Successive Halving | Neural architecture search |
| ``PatientPruner`` | Delays pruning decisions across a window of steps | Noisy learning curves |
| ``ThresholdPruner`` | Cuts off trials crossing hard numerical bounds | Divergence limits |
| ``NopPruner`` | Never stops trials | Baseline runs, fixed budgets |

---

## Deep dive: Early stopping pruners

### Median and Percentile pruners

`MedianPruner` stops an active trial if its intermediate value at step $t$ is worse than the median (50th percentile) of previous completed or pruned trials at the exact same step.

Parameters:
- `nStartupTrials`: Number of initial trials run completely to build a reliable baseline before pruning starts.
- `nWarmupSteps`: Number of initial steps within each trial evaluated without pruning.
- `intervalSteps`: Frequency of pruning checks (e.g. check every 2 epochs).

```swift
let pruner = MedianPruner(
    nStartupTrials: 5,
    nWarmupSteps: 10,
    intervalSteps: 2
)
```

### Successive Halving (ASHA) and Hyperband

Successive Halving allocates resources geometrically across rungs:

$$r_k = \text{minResource} \cdot \eta^k$$

At each rung, only the top $1/\eta$ fraction of trials is promoted to continue to the next rung.

`HyperbandPruner` manages several brackets of Successive Halving with varying aggressive early-stopping rates. Trials are assigned to brackets deterministically based on `trialNumber % nBrackets`, making it thread-safe for parallel workers.

```swift
let pruner = HyperbandPruner(
    minResource: 1,      // First rung evaluated at epoch 1
    maxResource: 81,     // Maximum training epochs
    reductionFactor: 3   // Retain top 1/3 at each rung
)
let study = try Swiftuna.createStudy(pruner: pruner)
```

### Patient pruner

Training curves can be noisy, with temporary loss spikes that trigger premature pruning. `PatientPruner` wraps any underlying pruner, requiring it to signal pruning for `patience` consecutive steps before actually stopping the trial:

```swift
let basePruner = MedianPruner(nStartupTrials: 5)
let robustPruner = PatientPruner(wrappedPruner: basePruner, patience: 3)
```

---

## Reporting intermediate values and pruning

Inside your training loop, report step metrics (such as epoch validation loss) to enable pruners to evaluate trajectories. Swiftuna supports two ergonomic styles:

### Option 1: Automatic early stopping (throwing)
Pass `pruneIfWorse: true` to ``Trial/report(_:step:pruneIfWorse:)-(Double,_,_)``. If the pruner recommends early termination, it throws ``SwiftunaError/trialPruned(reason:)``:

```swift
try study.optimize(nTrials: 50) { trial in
    var model = initializeModel()
    
    for epoch in 1...100 {
        let loss = model.trainEpoch()
        
        // Reports intermediate value; throws trialPruned if pruner triggers
        try trial.report(loss, step: epoch, pruneIfWorse: true)
    }
    
    return model.finalValidationLoss()
}
```

### Option 2: Explicit inspection and cleanup (non-throwing)
When training involves resources that must be flushed or cleaned up before stopping (or when using nested `do-catch` blocks), inspect ``Trial/shouldPrune`` manually:

```swift
try study.optimize(nTrials: 50) { trial in
    var model = initializeModel()
    
    for epoch in 1...100 {
        let loss = model.trainEpoch()
        try trial.report(loss, step: epoch)

        if try trial.shouldPrune {
            print("Early stopping at epoch \(epoch)")
            // Perform custom checkpoint saving, GPU buffer deallocation, etc.
            try trial.prune()
        }
    }
    
    return model.finalValidationLoss()
}
```
