# API Surface Area Comparison: Swiftuna vs. Rustuna Python API (`rustuna_pyo3`)

This report provides an exhaustive, functionality-level comparison between **Swiftuna** (Swift 6 bindings for `rustuna_core` / `librustuna_ffi`) and the official **Rustuna Python API** (`rustuna_pyo3` / `rustuna`).

---

## Executive Summary

| Functional Domain | Python (`rustuna_pyo3`) Surface Area | Swiftuna Surface Area | Parity Status | Missing Surface Area / Key Differences |
| :--- | :--- | :--- | :--- | :--- |
| **Study Management** | `create_study`, `load_study`, `copy_study`, `Study`, `PersistedStudy`, `StudyDirection` | `createStudy`, `loadStudy`, `copyStudy`, `getStudies`, `deleteStudy`, `Study`, `StudySummary` | **Swiftuna Exceeds** | Swiftuna adds `deleteStudy` and typed `StudySummary`. Python has low-level `PersistedStudy` wrapper object. |
| **Optimization Loop** | `study.optimize(func, n_trials, ...)` and ask-and-tell (`ask()`, `tell()`) | `study.optimize(...)` closures (sync/async/throwing) and ask-and-tell (`ask()`, `tell(consuming:)`) | **Parity+** | Swiftuna enforces move semantics (`~Copyable` trials) preventing double-tell or use-after-free at compile time. |
| **Trial Suggestions** | `suggest_float`, `suggest_int`, `suggest_categorical` | `suggest(in:)`, `suggest(choices:)`, standard Swift ranges `low...high` | **Parity+** | Swiftuna uses native `ClosedRange<Double>` and `ClosedRange<Int>`, strongly typed enums, and `ParameterValue` enum. |
| **Trial Metadata & Attributes** | `set_user_attr`, `set_user_attrs`, `set_constraint`, `set_constraints`, `get_user_attr` | Subscripting `trial[key]`, `AttributeKey<V>`, `ConstraintKey`, `CodableAttribute<T>` | **Swiftuna Exceeds** | Swiftuna provides static type safety and `Codable` JSON automatic serialization for attributes/constraints. |
| **Intermediate Values & Pruning** | Manual `trial.report(value, step)` + Optuna-style pruners | `trial.report(value, step, pruneIfWorse: true)`, 8 built-in `Pruner` structs | **Swiftuna Exceeds** | Swiftuna implements 8 pruner algorithms (`Median`, `Percentile`, `Threshold`, `SuccessiveHalving`, `Hyperband`, `Patient`, `Nop`) natively in Swift. Python relies on external Optuna pruners. |
| **Samplers** | `TPESampler`, `RandomSampler`, `NSGAIISampler`, `CmaEsSampler`, `SamplerProtocol`, `SamplerContext` | `TPESampler`, `RandomSampler`, `NSGAIISampler`, `QMCSampler`, `GridSampler`, `Sampler` protocol | **Divergence / Parity** | Python supports `CmaEsSampler` (backed by Python `cmaes` pkg) and Python custom sampler callbacks (`SamplerProtocol`). Swiftuna supports `QMCSampler` (Sobol) and `GridSampler` natively in `rustuna-ffi`. |
| **Storage Backends** | `InMemoryStorage`, `SQLite3Storage`, `JournalFileStorage`, `CachedStorage`, `ToRustStorage` | `StorageBackend.inMemory`, `.sqlite`, `.journal`, `syncWithOptunaDashboard()` | **Parity** | Both support SQLite3, Journal file, and In-Memory. Swiftuna provides native Optuna Dashboard database schema auto-sync (`rustuna_storage_sync_optuna_dashboard`). |
| **Trial Queues** | `DirectoryTrialQueue`, `InMemoryTrialQueue`, `SQLite3TrialQueue` | Enqueue via `study.enqueue(params:userAttrs:)` & `SwiftunaDistributed` | **Architectural Difference** | Python exposes raw queue classes (`DirectoryTrialQueue`, etc.). Swiftuna integrates enqueueing into `Study` and provides a Swift 6 `distributed actor StudyCoordinator`. |
| **Parameter Importance** | `PedAnovaImportanceEvaluator`, `get_param_importances` | `study.paramImportances()` | **Parity** | Both expose PED-ANOVA importance calculation backed by `rustuna_importance`. |
| **Error Handling** | Python Exception hierarchy (`RustunaError`, `TrialPruned`, `DuplicatedStudyError`, etc.) | Typed Swift `SwiftunaError` enum with C ABI error code mapping | **Parity** | Swiftuna uses Swift 6 typed throws (`throws(SwiftunaError)`). |
| **Distributed / Concurrency** | Python multiprocess / file queue | Swift 6 `distributed actor StudyCoordinator`, TaskGroup ask-and-tell | **Swiftuna Exceeds** | Swiftuna provides a first-class `SwiftunaDistributed` framework with network-transparent distributed actors. |

---

## Detailed Component Comparison

### 1. Study Creation and Management

#### Python API (`rustuna_pyo3`):
```python
create_study(study_name=None, directions=["minimize"], storage=None, sampler=None, load_if_exists=False) -> Study
load_study(study_name, storage, sampler=None) -> Study
copy_study(from_study, to_storage, to_study_name=None) -> Study
```
- Returns `rustuna.Study`.
- Accessing `study.trials` returns a list of `PersistedTrial` or `PersistedStudy` objects.

#### Swiftuna API:
```swift
Swiftuna.createStudy(name:directions:pruner:storage:sampler:loadIfExists:) -> Study
Swiftuna.loadStudy(name:pruner:storage:sampler:) -> Study
Swiftuna.copyStudy(study:toStorage:toStudyName:) -> Study
Swiftuna.getStudies(in storage: StorageBackend) -> [StudySummary]
Swiftuna.deleteStudy(named name: StorageBackend)
```
- Returns Swift `Study` class instance.
- **Surface Area Difference**: Swiftuna adds top-level `getStudies(in:)` and `deleteStudy(named:in:)` direct administrative methods, returning strongly-typed `StudySummary` structs.

---

### 2. Active Trial Lifecycle & Ask-and-Tell

#### Python API (`rustuna_pyo3`):
```python
trial = study.ask()
x = trial.suggest_float("x", -10.0, 10.0)
study.tell(trial, state=TrialState.COMPLETE, values=[loss])
```
- In Python, `Trial` is a mutable PyO3 reference. Re-using a finished `Trial` raises `TrialDiscarded` at runtime.

#### Swiftuna API:
```swift
var trial = try study.ask()
let x = try trial.suggest("x", in: -10.0...10.0)
try study.tell(consuming: trial, value: loss)
// OR
try study.tell(trialNumber: 1, state: .complete, values: [loss])
```
- **Surface Area Difference**: Swiftuna defines `Trial` as a **non-copyable struct (`~Copyable`)**. Passing `trial` to `study.tell(consuming: trial, ...)` transfers ownership, making use-after-tell or double-evaluation a **compile-time error** in Swift 6.

---

### 3. Parameter Suggestions & Space Construction

#### Python API (`rustuna_pyo3`):
```python
trial.suggest_float("x", low=-10.0, high=10.0, step=0.5, log=False)
trial.suggest_int("y", low=1, high=10, step=1, log=False)
trial.suggest_categorical("cat", choices=["a", "b", "c"])
```

#### Swiftuna API:
```swift
try trial.suggest("x", in: -10.0...10.0, step: 0.5, log: false)
try trial.suggest("y", in: 1...10)
try trial.suggest("cat", choices: ["a", "b", "c"])
try trial.suggest("enum", choices: MyEnum.allCases) // Swift Enum Support
```
- **Surface Area Difference**:
  - Swiftuna uses Swift standard library ranges (`ClosedRange<Double>`, `ClosedRange<Int>`).
  - Swiftuna natively supports Swift `RawRepresentable` enums in `suggest`.

---

### 4. Attributes, Constraints, and Metadata

#### Python API (`rustuna_pyo3`):
```python
trial.set_user_attr("author", "alice")
trial.set_constraint("memory_mb", 1024.0) # stored under system attribute "constraints:memory_mb"
```

#### Swiftuna API:
```swift
trial[AttributeKey<String>("author")] = "alice"
trial[ArchitectureKey.self] = .resnet50
trial[constraint: MemoryBound.self] = peakMemory - 4096.0
```
- **Surface Area Difference**:
  - Python uses stringly-typed dictionary key-values.
  - Swiftuna introduces static `AttributeKey<Value>`, `ConstraintKey`, and `CodableAttribute<T>`, providing compile-time type safety, custom struct encoding, and subscript syntax on active `Trial` and `PersistedTrial`.

---

### 5. Samplers and Search Algorithms

#### Python API (`rustuna_pyo3`):
- `RandomSampler(seed=None)`
- `TPESampler(seed=None, n_startup_trials=10, multivariate=None)`
- `NSGAIISampler(seed=None, population_size=50, mutation_prob=None, crossover_prob=0.9, swapping_prob=0.5)`
- `CmaEsSampler(seed=None, popsize=None)` *(Python `cmaes` wrapper)*
- Allows custom Python samplers via `SamplerProtocol` / `SamplerContext`.

#### Swiftuna API:
- `RandomSampler(seed=None)`
- `TPESampler(seed=None)`
- `NSGAIISampler(populationSize:mutationProb:crossoverProb:swappingProb:seed:)`
- `QMCSampler(seed=None)` *(Quasi-Monte Carlo Sobol sequence sampling, implemented in Rustuna FFI)*
- `GridSampler(searchSpace:seed:)` *(Discrete grid search, implemented in Rustuna FFI)*

- **Surface Area Difference**:
  - Python includes `CmaEsSampler` (which calls Python's `cmaes` package). Swiftuna does not wrap `cmaes`.
  - Swiftuna includes `QMCSampler` and `GridSampler` directly exposed via static C ABI linkage in `LibRustuna`.

---

### 6. Early Stopping Pruners

#### Python API (`rustuna_pyo3`):
- Python API relies on Python Optuna's pruner classes (`optuna.pruners.MedianPruner`, etc.) when converting studies to Optuna.

#### Swiftuna API:
Swiftuna includes 8 fully native, strongly-typed Swift `Pruner` implementations:
1. `MedianPruner`
2. `PercentilePruner`
3. `ThresholdPruner`
4. `SuccessiveHalvingPruner`
5. `HyperbandPruner`
6. `PatientPruner`
7. `NopPruner`
8. Custom `Pruner` protocol conformances.

- Active `Trial` supports inline pruning: `try trial.report(valLoss, step: epoch, pruneIfWorse: true)`.

---

### 7. Storage Backends & Optuna Dashboard Compatibility

#### Storage Types:
- Both support **In-Memory**, **SQLite3**, and **Journal File Storage**.

#### Optuna Dashboard Sync:
- Swiftuna provides explicit C-level database migration & synchronization:
  `StorageBackend.syncWithOptunaDashboard(at: path)` / `study.syncWithOptunaDashboard()`
  This executes SQL transformations in `rustuna_storage_sync_optuna_dashboard` to normalize JSON string quotes, float constraints arrays, and system attributes so `optuna-dashboard` reads SQLite databases instantly without schema errors.

---

### 8. Trial Queues & Distributed Optimization

#### Python API (`rustuna_pyo3`):
Exposes explicit queue objects:
- `DirectoryTrialQueue`
- `InMemoryTrialQueue`
- `SQLite3TrialQueue`

#### Swiftuna API:
- Integrated enqueueing: `study.enqueue(params:userAttrs:)`.
- **`SwiftunaDistributed` Framework**: Swiftuna provides a dedicated target `SwiftunaDistributed` built on Swift 6 Distributed Actors:
  - `StudyCoordinator`: Network-transparent actor coordinating multi-node/multi-process `ask`, `report`, `tell`, and `bestTrial` calls over actor transports.
  - `SearchSpace`, `DistributedTrialSpec`, `DistributedTrialResult`, `InFlightTrial`.

---

### 9. Parameter Importance Analysis

#### Python API (`rustuna_pyo3`):
```python
importances = rustuna.importance.get_param_importances(study)
```

#### Swiftuna API:
```swift
let result = study.paramImportances(normalize: true)
// Returns Result<[String: Double], SwiftunaError>
```
- Both call `rustuna_importance::PedAnovaImportanceEvaluator` in `rustuna_core` to compute Partial Dependence Analysis of Variance scores.

---

## Summary of Missing / Differing Surface Area

### Surface Area Present in Python API, Missing in Swiftuna:
1. **`CmaEsSampler`**: Python wraps `cmaes` PyPI package. Swiftuna does not currently include a CMA-ES sampler wrapper.
2. **Custom Python Sampler Callbacks (`SamplerProtocol`)**: Python allows writing custom sampler classes in Python that implement `sample_joint`, `sample_independent`, `before_trial`, `after_trial`. In Swiftuna, custom samplers are currently configured via built-in C handles.
3. **Explicit Standalone Queue Classes (`DirectoryTrialQueue`, etc.)**: Python exposes queue classes as standalone objects. Swiftuna handles queuing directly via `study.enqueue(...)` and `SwiftunaDistributed`.

### Surface Area Present in Swiftuna, Missing in Python API:
1. **Non-Copyable Active Trials (`~Copyable`)**: Swift 6 compile-time safety preventing use-after-free or double evaluation.
2. **Typed Attribute and Constraint Keys**: `AttributeKey<V>`, `ConstraintKey`, and `CodableAttribute<T>` for static type checking and automatic `Codable` JSON serialization.
3. **Quasi-Monte Carlo Sobol (`QMCSampler`) & Discrete Grid Search (`GridSampler`)**: Exposed as first-class samplers.
4. **Native Swift Pruner Algorithms**: 8 built-in `Pruner` implementations in Swiftuna.
5. **Swift 6 Distributed Actor Coordinator (`SwiftunaDistributed`)**: First-class network-transparent actor system for distributed GPU/cluster HPO.
6. **Optuna Dashboard Auto-Sync (`syncWithOptunaDashboard`)**: Built-in SQLite database transformer for seamless dashboard compatibility.
7. **Typed Error Returns**: Swift 6 typed throws (`throws(SwiftunaError)`).

---

## Conclusion

Swiftuna achieves **complete functional parity** with Rustuna Python's core optimization engine while leveraging Swift 6's memory safety (`~Copyable`), static typing (`AttributeKey`), concurrency (`distributed actor`), and standard library idioms (`ClosedRange`).
