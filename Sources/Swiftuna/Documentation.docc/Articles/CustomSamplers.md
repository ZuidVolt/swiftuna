# Custom samplers in Swift

Two Swift-native ways to implement your own sampling strategy: per-suggestion callbacks running inside the engine, or history-driven trial-start strategies running above it.

## Overview

Every sampler answers one question: what should the next trial try? Strategies differ in when they answer it. Swiftuna offers three customization points, picked by timing and search space structure:

1. **``CallbackSampler``** answers per suggestion inside the engine. Each closure fires at its own `suggest` call, conditioned on earlier parameters in the same trial. Use this for conditional search spaces and custom discrete distributions.
2. **``CustomSampler``** answers per trial above the engine. One method reads trial history and returns the full configuration before evaluation starts. Pass a `CustomSampler` directly to `Swiftuna.createStudy(sampler:)` or to `study.optimize(nTrials:using:)`. Use this for history-driven algorithms like CMA-ES, hill climbing, multi-armed bandits, or evolutionary strategies.
3. **Raw enqueue driver loops** around `Study.enqueue(_:userAttrs:)` answer per trial without a protocol. Use this for ad-hoc scripts.

Omitted parameters fall back to the study's underlying Rust sampler in every pattern, so partial strategies work with no extra code.

---

## Per-suggestion control (``CallbackSampler``)

Assign any subset of three closures. Distribution kinds without a closure fall back to uniform random sampling inside Rustuna:

```swift
let sampler = CallbackSampler(
    onFloat: { name, low, high, step, log, trialNumber in
        // `step` is nil for continuous ranges.
        Double.random(in: low...high)
    },
    onCategorical: { name, choices, trialNumber in
        choices.count - 1 // exploit the last arm
    }
)
let study = try Swiftuna.createStudy(sampler: sampler)
```

The closure signatures:

```swift
public typealias FloatFn = @Sendable (String, Double, Double, Double?, Bool, Int) -> Double?
public typealias IntFn = @Sendable (String, Int64, Int64, Int64, Bool, Int) -> Int64?
public typealias CategoricalFn = @Sendable (String, [String], Int) -> Int?
```

Each closure receives the decoded distribution bounds, step, and log flag, along with the trailing trial number matching `Trial.number` at tell time. Returning `nil` or an out-of-range categorical index fails the suggestion as a sampler error. The trial records the failure instead of proceeding with an invalid value.

- **Concurrency.** Closures run synchronously on the optimizing thread and may execute concurrently across tasks. Capture only Sendable state or synchronize inside the closure with a mutex.
- **History reads.** The engine passes distributions, not past trials. Querying history via `study.trials` inside a callback costs $O(\text{history})$ per suggestion. When every parameter decision depends on past results, use ``CustomSampler`` instead.
- **Reentrancy.** Calling `ask` from inside a callback closure throws ``SwiftunaError/reentrantAsk(_:)`` to prevent leaking active trial handles.

Example of an epsilon-greedy multi-armed bandit with random floats:

```swift
final class EpsilonGreedy: Sendable {
    private let bestArm = Mutex<Int>(0)

    func sampler() -> CallbackSampler {
        CallbackSampler(
            onFloat: { name, low, high, step, log, _ in
                Double.random(in: low...high)
            },
            onCategorical: { [self] name, choices, _ in
                return Double.random(in: 0...1) < 0.1
                    ? Int.random(in: 0..<choices.count) : bestArm.withLock { $0 }
            }
        )
    }
}
```

---

## History-driven strategies (``CustomSampler``)

When the algorithm maintains internal state across trials or needs the full history of past evaluations, implement ``CustomSampler``. The protocol requires one method:

```swift
public protocol CustomSampler: Sendable {
    /// Controls whether trial parameter dictionaries are retained in in-memory StudyHistory.
    var retainsParameterHistory: Bool { get }

    /// Proposes hyperparameter configurations for the next trial given completed history.
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue]
}
```

Default protocol extensions supply `retainsParameterHistory = true`. Basic custom samplers only need to provide `sample(history:trialNumber:)`.

### Example: A custom hill-climbing sampler

```swift
struct HillClimb: CustomSampler {
    let step: Double

    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        guard let bx = history.best?.params["x"]?.asDouble else {
            return ["x": .double(Double.random(in: -10.0...10.0))]
        }
        return ["x": .double(min(10.0, max(-10.0, bx + Double.random(in: -step...step))))]
    }
}
```

### Two ways to execute a `CustomSampler`

#### Direct study binding
Pass the custom sampler directly to `Swiftuna.createStudy(sampler:)`. The study attaches the custom sampler to its instance and automatically drives parameter generation during `study.optimize(nTrials:)`:

```swift
let sampler = HillClimb(step: 1.0)
let study = try Swiftuna.createStudy(sampler: sampler)

// Drives HillClimb automatically without requiring 'using:'
try study.optimize(nTrials: 100) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return x * x
}
```

#### Ad-hoc execution with `using:`
Supply a custom sampler or closure at optimization time via `study.optimize(nTrials:using:)`:

```swift
let study = try Swiftuna.createStudy(direction: .minimize)
try study.optimize(nTrials: 50, using: HillClimb(step: 1.0)) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return x * x
}
```

### Understanding ``StudyHistory``

``StudyHistory`` provides an immutable snapshot of completed trials:
- `all`: All completed trials in chronological order.
- `new`: Trials completed since the previous `sample` call. On the first call, `new == all`.
- `best`: The optimal completed trial for the primary study direction, or `nil` if no completed trial exists or if the study is multi-objective.

Fold over `new` instead of re-scanning `all` on every trial. Re-scanning `all` causes suggestion latency to grow quadratically with the total trial count.

---

## Memory optimization: retainsParameterHistory

When running large hyperparameter studies (such as 10,000 to 25,000 trials in 50 dimensions), storing a 50-key dictionary per trial inside Swiftuna's in-memory history array allocates over 146 MB of heap in Swift.

Algorithms that track their own state (such as covariance matrices, populations, or direction vectors) do not read past parameter dictionaries from `StudyHistory`. They only need trial numbers, objective values, and completion states. Set `retainsParameterHistory` to `false`:

```swift
public var retainsParameterHistory: Bool { false }
```

When `retainsParameterHistory` is `false`:
- The custom sampler loop stores an empty parameter dictionary in the local `StudyHistory`, recording trial numbers, objective values, and completion states.
- All trial parameters and results are still written to Rustuna's persistent storage backend (in-memory, SQLite, or journal).
- Peak memory at scale drops by roughly 20% to 25% compared to retaining duplicate dictionaries in the Swift heap.

---

## Ghost TPE suppression

When creating a study with `createStudy(sampler: any CustomSampler)`, Swiftuna suppresses the default background TPE engine. 

In Optuna and Rustuna, studies without an explicit sampler default to `TPESampler`. On every `ask()` call, an active TPE sampler runs kernel density estimation across all past trials to model parameter distributions. If a custom sampler subsequently overwrites those suggested parameters via fixed queue injection, all TPE computations are discarded.

To eliminate this waste, `Swiftuna.createStudy(sampler: any CustomSampler)` configures Rustuna with a zero-cost `RandomSampler` fallback trampoline. Only parameters omitted by the custom sampler trigger fallback sampling, and the engine never fits unused surrogate models.

---

## Built-in CMA-ES: ``CMASampler``

Swiftuna includes an implementation of Active CMA-ES (Covariance Matrix Adaptation Evolution Strategy, Hansen 2016) in ``CMASampler``:

- **Mathematical formulation.** Adapts a multivariate normal distribution $\mathcal{N}(m, \sigma^2 C)$ using cumulative step-size adaptation ($p_\sigma$), rank-1 paths ($p_c$), and active rank-$\mu$ covariance updates that incorporate both successful and unsuccessful candidate steps.
- **Flat buffer linear algebra.** Matrix operations and cyclic Jacobi eigendecompositions execute directly on contiguous 1D arrays (`ContiguousArray<Double>`) without per-trial matrix heap allocations.
- **Optuna and NumPy parity.** Setting `useNumpyPRNG: true` runs a 32-bit Mersenne Twister MT19937 generator with Box-Muller normal transforms, matching Python Optuna (`cmaes 0.12.0`) bit for bit.
- **Throughput and latency.** Typically 7x to 15x faster than Python Optuna on continuous benchmarks, maintaining flat sub-millisecond per-trial proposal latency without progressive degradation across tens of thousands of trials.
- **Memory scaling.** Pre-configured dimensions set `retainsParameterHistory` to `false`, cutting peak heap memory by approximately 20% to 25% at scale.

### Usage example

```swift
import Swiftuna

// 1. Define search dimensions
let cma = CMASampler(
    dimensions: [
        .continuous(name: "x0", lower: -5.0, upper: 5.0),
        .continuous(name: "x1", lower: -5.0, upper: 5.0),
    ],
    seed: 42,
    useNumpyPRNG: true // Set true for bit-exact parity with Python cmaes
)

// 2. Create the study; CMA-ES drives trial proposals automatically
let study = try Swiftuna.createStudy(sampler: cma)

// 3. Run the optimization loop
try study.optimize(nTrials: 1000) { trial in
    let x0 = try trial.suggest("x0", in: -5.0...5.0)
    let x1 = try trial.suggest("x1", in: -5.0...5.0)
    return (1.0 - x0) * (1.0 - x0) + 100.0 * (x1 - x0 * x0) * (x1 - x0 * x0)
}
```

---

## Comparison with Optuna and Rustuna

Optuna's custom sampler interface centers on `BaseSampler` (`ref/optuna/optuna/samplers/_base.py`). It uses `sample_independent` for single parameters, `sample_relative` for joint decisions over an inferred space, and `study.trials` queries for history.

Rustuna ports that design to Rust (`ref/rustuna/rustuna_core/src/sampler.rs`), condensing `Study` and `FrozenTrial` into a lightweight `Context` struct and gating joint sampling behind `sample_joint`.

Swiftuna provides equivalent capabilities with Swift 6 safety and flat-latency performance:

| Feature | Optuna / Rustuna | Swiftuna |
| :--- | :--- | :--- |
| Independent parameter sampling | `sample_independent(study, trial, name, distribution)` | ``CallbackSampler`` typed closures |
| Joint parameter sampling | `sample_relative` / `sample_joint` (engine-level) | ``CustomSampler`` whole-configuration loop |
| Study binding | `optuna.create_study(sampler=...)` | `Swiftuna.createStudy(sampler:)` |
| Search space declaration | `infer_relative_search_space` | Inferred automatically or declared via `CMASampler(dimensions:)` |
| Fixed parameter delivery | `enqueue_trial` | Typed `enqueue` with atomic `askEnqueued` |
| History access | Pull via `study.trials` | Pushed ``StudyHistory`` snapshot with `retainsParameterHistory` control |
| CMA-ES implementation | External `cmaes` Python wheel | Built-in zero-copy ``CMASampler`` in pure Swift 6 |
| Trial identifier | Context struct or trial object | Trailing `trialNumber` argument |

---

## Boundaries and runtime rules

- **Serial driver execution.** `sample` executes sequentially within the study's optimization loop. For distributed workers across multiple processes, use `SwiftunaDistributed` with a coordinator.
- **State persistence.** Custom sampler state lives in the conforming instance during process execution. Only completed trial records survive process restarts through SQLite or journal storage.
- **Multi-objective studies.** `StudyHistory.best` returns `nil` for multi-objective studies. Multi-objective strategies should inspect `history.all` directly to calculate Pareto dominance.
- **Callbacks see distributions, not history.** A `CallbackSampler` closure that queries `study.trials` pays $O(\text{history})$ per suggestion. Use ``CustomSampler`` when decisions depend on historical evaluations.
- **Reentrant ask calls.** Calling `ask` from inside a callback closure throws `reentrantAsk`. Keep parameter proposals independent of manual trial checkout.

---

## See also

- <doc:SamplersAndPruners> for built-in samplers and early stopping pruners.
- <doc:AskAndTellGuide> for manual ask-and-tell loops.
- <doc:TelemetryAndObservability> for tracing spans and parameter attributes.
