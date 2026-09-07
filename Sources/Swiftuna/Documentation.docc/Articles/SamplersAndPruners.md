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
| ``CMASampler`` | Covariance Matrix Adaptation (Active CMA-ES) | Continuous numerical optimization, ill-conditioned surfaces, high dimensional spaces | Single only | Unit-box clipping / resampling |
| ``QMCSampler`` | Quasi-Monte Carlo (Sobol) | Low-discrepancy space filling | Single only | No |
| ``GridSampler`` | Cartesian product grid | Small discrete spaces, ablation sweeps | Single only | No |
| ``BruteForceSampler`` | Dynamic prefix decision tree | Exhaustive search over discrete, step-float, and conditional spaces | Yes | No |
| ``PartialFixedSampler`` | Parameter partitioning | Enforcing fixed subsets while delegating free parameters | Inherited | Inherited |
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

### Covariance matrix adaptation (``CMASampler``)

`CMASampler` implements Active CMA-ES (Hansen 2016) in pure Swift 6, matching the numerical output of Python Optuna's `CmaEsSampler` (`CyberAgentAILab/cmaes`).

It adapts a multivariate normal distribution $\mathcal{N}(m, \sigma^2 C)$ across generations. It tracks step-size path cumulation ($p_\sigma$), rank-1 evolution paths ($p_c$), and active rank-$\mu$ covariance updates using in-place cyclic Jacobi eigendecomposition on flat 1D memory buffers:

- Runs roughly 7x to 15x faster than Python Optuna on continuous benchmarks, with flat sub-millisecond per-trial latency across tens of thousands of trials.
- Setting `retainsParameterHistory: false` reduces peak memory by approximately 20% to 25% at scale compared to Python Optuna.
- Setting `useNumpyPRNG: true` runs a 32-bit Mersenne Twister with Box-Muller Gaussian transforms for bit-exact parity with Python Optuna.

```swift
let cma = CMASampler(
    dimensions: [
        .continuous(name: "x0", lower: -5.0, upper: 5.0),
        .continuous(name: "x1", lower: -5.0, upper: 5.0)
    ],
    seed: 42
)

// Pass directly to createStudy; drives CMA-ES automatically without 'using:'
let study = try Swiftuna.createStudy(sampler: cma)
try study.optimize(nTrials: 100) { trial in
    let x0 = try trial.suggest("x0", in: -5.0...5.0)
    let x1 = try trial.suggest("x1", in: -5.0...5.0)
    return x0 * x0 + x1 * x1
}
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

### Dynamic exhaustive search (``BruteForceSampler``)

While ``GridSampler`` requires upfront static Cartesian product declarations, ``BruteForceSampler`` dynamically discovers parameter spaces, ranges, and conditional branching as the objective executes. It builds an internal prefix decision tree where each node represents a parameter choice, tracking four lifecycle states:
- **Unexpanded:** A candidate choice identified by the distribution but not yet evaluated.
- **Running:** A branch currently undergoing evaluation by an active trial.
- **Leaf:** A completed terminal parameter combination.
- **Internal:** An intermediate decision node branching into subsequent parameter decisions.

Candidate selection blends exact uniform sampling with flat uniform sampling ($\alpha = 0.5$) over unexpanded subtree counts:

$$w_i = (1 - \alpha) \frac{u_i}{\sum_j u_j} + \alpha \frac{\mathbb{I}(u_i > 0)}{\sum_j \mathbb{I}(u_j > 0)}$$

This balancing prevents starvation of deeper or conditional branches. When all paths are exhausted, ``Study/optimize(nTrials:timeout:objective:)-3gyl5`` stops automatically, and manual ``Study/ask()`` throws ``SwiftunaError/searchSpaceExhausted(_:)``.

```swift
let sampler = BruteForceSampler(seed: 42)
let study = try Swiftuna.createStudy(sampler: sampler)

try study.optimize(nTrials: 100) { trial in
    let model = try trial.suggest("model", choices: ["linear", "mlp"])
    if model == "linear" {
        let reg = try trial.suggest("reg", in: 0.1...0.3, step: 0.1)
        return evaluateLinear(reg: reg)
    } else {
        let layers = try trial.suggest("layers", in: 1...3)
        return evaluateMLP(layers: layers)
    }
}
```

### Partial parameter fixing (``PartialFixedSampler``)

``PartialFixedSampler`` pins a designated dictionary of hyperparameters to fixed values while delegating all remaining free parameters to a base sampler. This enables ablation experiments, sensitivity tests, or targeted fine-tuning where certain architecture choices are held constant while training hyperparameters continue to be optimized.

The delegate sampler can be a Rustuna engine sampler (such as ``TPESampler`` or ``QMCSampler``) or a native Swift custom sampler (such as ``CMASampler``):

```swift
// Fix architecture choices; optimize learning rate and weight decay with TPE
let fixed: [String: ParameterValue] = [
    "layers": .int(4),
    "activation": .string("gelu")
]

let sampler = PartialFixedSampler(
    fixedParams: fixed,
    baseSampler: TPESampler(seed: 42)
)

let study = try Swiftuna.createStudy(sampler: sampler)
try study.optimize(nTrials: 50) { trial in
    let layers = try trial.suggest("layers", in: 1...8) // Always returns 4
    let act = try trial.suggest("activation", choices: ["relu", "gelu", "swish"]) // Always returns "gelu"
    let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true) // Explored by TPE
    return trainModel(layers: layers, activation: act, lr: lr)
}
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
