# Custom samplers in Swift

Two Swift-native ways to implement your own sampling strategy. Per-suggestion callbacks run inside the engine. History-driven trial-start strategies run above it.

## Overview

Every sampler answers one question, what should the next trial try. Strategies differ in when they answer it. Swiftuna offers three customization points, picked by timing:

1. **``CallbackSampler``** answers per suggestion, inside the engine. Each closure fires at its own `suggest` call, conditioned on earlier params in the same trial. This is the choice for conditional search spaces.
2. **``CustomSampler``** answers per trial, above the engine. One method reads history and returns the whole configuration before evaluation starts. This is the choice for history-driven strategies like hill-climbing, bandits, and Bayesian ideas of your own.
3. **Raw enqueue driver loops** around `Study.enqueue(_:userAttrs:)` answer per trial with no protocol at all. This is the choice for one-off scripts.

Omitted params fall back to the study's Rust sampler in every pattern, so partial strategies work with no extra code.

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
        choices.count - 1 // always exploit the last arm (demo only)
    }
)
let study = try Swiftuna.createStudy(sampler: sampler)
```

The full closure types:

```swift
public typealias FloatFn = @Sendable (String, Double, Double, Double?, Bool, Int) -> Double?
public typealias IntFn = @Sendable (String, Int64, Int64, Int64, Bool, Int) -> Int64?
public typealias CategoricalFn = @Sendable (String, [String], Int) -> Int?
```

Each closure receives the decoded distribution directly, bounds, step, and log flag, plus the trial number trailing to match `Trial.number` at tell time. Returning `nil`, or an out-of-range categorical index, fails the suggestion as a sampler error. The trial records the failure instead of continuing with a bad value.

Threads first. Closures run synchronously on the optimizing thread and may run concurrently across threads. Capture only `Sendable` state or synchronize inside the closure. History second. The engine passes distributions, not past trials. Read history yourself from the captured study with `study.trials`, or with `trials(where:since:)` and the last-seen count for incremental refresh. Do this sparingly. A history fetch on every suggestion costs O(history) per suggest.

Checking out a trial from inside a closure throws ``SwiftunaError/reentrantAsk(_:)``. It does not deadlock or scramble queue pairing. A trial checked out mid-suggestion would leak unfinished, so the refusal is loud on purpose.

A realistic sketch. Epsilon-greedy over categorical arms with random floats:

```swift
final class EpsilonGreedy: Sendable {
    private let lock = NSLock()
    private var bestArm: Int = 0

    func sampler() -> CallbackSampler {
        CallbackSampler(
            onFloat: { name, low, high, step, log, _ in
                Double.random(in: low...high)
            },
            onCategorical: { [self] name, choices, _ in
                lock.lock(); defer { lock.unlock() }
                return Double.random(in: 0...1) < 0.1
                    ? Int.random(in: 0..<choices.count) : bestArm
            }
        )
    }
}
```

---

## History-driven strategies (``CustomSampler``)

When the strategy needs past trials, implement one method and let the driver own the loop:

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
try study.optimize(nTrials: 50, using: HillClimb(step: 1.0)) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return x * x
}
```

``StudyHistory`` has three fields. `all` holds every known trial, oldest first. `new` holds the tail unseen since the last call. `best` holds the best completed trial for the study direction, or `nil` when none exists or the study is multi-objective. On the first call there is no last call, so `new == all`. Cold-start samplers see pre-existing history in full. Fold `new` instead of re-scanning `all`. Re-scanning makes suggest cost grow quadratically with study size.

The driver behind `optimize(using:)` does the rest:

- Reads history once and accumulates it locally. No refetch, O(1) amortized per trial.
- Fixes the returned params ahead of `ask`, atomically, so drivers sharing one study never receive each other's configurations. Partial dicts are fine. Omitted params are Rust-sampled.
- Merges the fixed dict over everything the objective actually suggested, including Rust-sampled leftovers the sampler never saw. The recorded params match what the trial ran, even with partial fixing.
- Records failed and pruned objectives exactly like `optimize`, with the same telemetry spans. A throwing `sample` aborts the run with the original error. No trial exists yet at sample time, so there is nothing to record, and silent fallback would corrupt the experiment.

Four `optimize(using:)` overloads handle every combination of protocol or closure strategy with single- or multi-value objectives. The closure form (`CustomSuggestClosure`) adapts to the protocol, so the driver implements one loop. User objectives keep their own error types through the generic-`E` bridge, mirroring the plain `optimize` overloads.

`sample` runs serially inside the driver's loop. Keep mutable state in the conforming type. A final class with a lock works. Do not rely on call order from multiple drivers.

---

## How this maps to Optuna and Rustuna

Optuna's custom-sampler API is `BaseSampler` (`ref/optuna/optuna/samplers/_base.py`). It has `sample_independent` for one parameter at a time, `sample_relative` for joint decisions over an inferred space, `infer_relative_search_space` to declare that space, `enqueue_trial` for fixed params, and `before_trial` / `after_trial` lifecycle hooks. Samplers pull history themselves via `study.trials`.

Rustuna ports that API to Rust (`ref/rustuna/rustuna_core/src/sampler.rs`) with two changes worth noticing. The full `Study` / `FrozenTrial` arguments shrink to a lightweight `Context` with study id, directions, trial number, and trial id. And `sample_relative` becomes `sample_joint`, gated behind `support_joint_sampling()`. History still pulls through `Storage`. The Python bridge (`rustuna_pyo3/src/sampler/to_rust.rs`) sets the precedent for foreign samplers. A `Mutex`-guarded external object adapts to the same trait, and foreign errors surface as sampler errors.

Swiftuna keeps the concepts and changes the shape where Swift or performance requires it:

| Optuna / Rustuna | Swiftuna |
| :--- | :--- |
| `sample_independent(study, trial, name, distribution)` | ``CallbackSampler`` per-kind closures |
| `sample_relative` / `sample_joint` (joint, in-engine) | ``CustomSampler`` whole-config, above the engine |
| `infer_relative_search_space` | No equivalent. Omitted params fall back |
| `enqueue_trial` fixed params | Typed `enqueue` plus atomic `askEnqueued` |
| History pull via `study` / `Storage` | ``StudyHistory`` push snapshot |
| `before_trial` / `after_trial` hooks | No equivalent |
| Full study/trial args (Optuna) or `Context` (Rustuna) | Trailing `trialNumber` only |

Where the API matches, it matches on purpose. FIFO enqueue-then-ask pairing. Omitted-params fallback. `trialNumber` equal to `Trial.number`. Foreign errors becoming sampler errors. Loud failure instead of silent corruption. All three systems agree here, so strategies port by reasoning rather than relearning.

Where it diverges, the reason is concrete:

- **Three closures instead of one `sample_independent`.** Matching on a distribution enum at every suggest call is unidiomatic Swift. Kind-specific typed closures receive decoded bounds directly and compose independently. Assign floats, leave categoricals to the engine. The `Distribution` split happens once at the FFI trampoline, not in user code.
- **Joint sampling above the engine instead of inside it.** Exposing `sample_joint` through the callback table would ship a search space across FFI per trial, plus storage access from Swift. Fixing the whole config ahead of `ask` produces the same observable behavior, because fixed params win over sampling, and it reuses the typed-enqueue path with measured cost. The tradeoff is granularity. There are no per-suggest joint decisions mid-trial. Use ``CallbackSampler`` for those.
- **History pushed, not pulled.** Swift cannot hold the storage lock across calls. A `Sendable` snapshot crosses Swift 6 concurrency boundaries where a storage handle cannot. The driver accumulates locally at O(1) amortized cost instead of refetching. `new` makes the incremental shape explicit instead of leaving it as convention.
- **`trialNumber` instead of a context object.** Rustuna already shrank Optuna's study/trial pair to a `Context` struct. Swiftuna keeps only the field strategies actually key on. Both sampler APIs carry it trailing, so state keyed by trial and logging read the same on either side.
- **Typed errors instead of exceptions.** Throwing `sample` aborts with the original error preserved. `nil` returns become sampler errors. Reentrant checkout throws `reentrantAsk` instead of deadlocking. Each failure mode uses the Swift construct that carries it exactly.

---

## Boundaries

The API is deliberately smaller than Optuna's. The edges below are current limits:

- **No `before_trial` / `after_trial` hooks.** Neither Swift sampler API exposes lifecycle hooks. Per-trial setup and teardown live in the objective or the driver loop around it.
- **No relative-space declaration.** There is no `infer_relative_search_space` equivalent. Strategies cannot announce which params they decide jointly. Anything not returned is engine-sampled.
- **Serial `CustomSampler` driver.** `sample` runs inside one driver's loop. Concurrent custom drivers over one study coordinate only through the atomic checkout. Parallel evaluation with a custom strategy means one driver loop.
- **No sampler state persistence.** Strategy state lives in the conforming type, or the closure capture, for the run's lifetime. Restarting a process restarts the strategy's memory. Only trial history survives, via storage.
- **`best` is single-objective.** Multi-objective studies get `best == nil`. Pareto-aware custom strategies rank `all` themselves.
- **Callbacks see distributions, not history.** A `CallbackSampler` closure that fetches `study.trials` on every suggestion pays O(history) per suggest. Cache the study handle in the capture and refresh incrementally with `trials(where:since:)`. When every decision needs history, use ``CustomSampler`` instead.
- **Reentrant checkout is an error.** Calling `ask` from inside a sampler closure throws. Design the strategy so suggestion never needs a trial handle.

---

## See also

- <doc:SamplersAndPruners> for the built-in sampler comparison and TPE configuration.
- <doc:AskAndTellGuide> for the manual ask/tell loop the custom drivers build on.
- <doc:TelemetryAndObservability> for the `param.*` span attributes both custom paths emit.
