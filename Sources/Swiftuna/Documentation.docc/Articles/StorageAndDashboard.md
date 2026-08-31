# Storage backends and Optuna Dashboard

Persist optimization trials across process restarts, scale concurrent workers, and visualize study progress in real time with Optuna Dashboard.

## Overview

Swiftuna provides three storage backends designed for different execution environments:

| Storage Backend | Persistence | Concurrency Model | Best for |
| :--- | :--- | :--- | :--- |
| ``StorageBackend/inMemory`` | RAM only | Thread-safe in-process | Fast unit tests, script evaluations |
| ``StorageBackend/sqlite(path:)`` | Disk (SQLite3) | SQLite file-locking / WAL | Multi-worker local runs, Optuna Dashboard |
| ``StorageBackend/journal(path:)`` | Disk (Append-only log) | Lockless append | Multi-process clusters, shared NFS filesystems |

---

## SQLite storage and resuming studies

Persisting studies to SQLite lets long optimization jobs accumulate trials across restarts or multiple worker processes:

```swift
import Swiftuna

let storage = StorageBackend.sqlite(path: "experiments.db")

// Resume existing study if found; otherwise create a new one
let study = try Swiftuna.createStudy(
    name: "resnet_cifar10",
    direction: .minimize,
    storage: storage,
    loadIfExists: true
)

try study.optimize(nTrials: 50) { trial in
    let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
    return train(lr: lr)
}
```

---

## Real-time visualization with Optuna Dashboard

Swiftuna's SQLite schema matches Python Optuna's `RDBStorage` byte-for-byte. You can run `optuna-dashboard` directly against any Swiftuna SQLite database:

```bash
# 1. Install optuna-dashboard via pip
pip install optuna-dashboard

# 2. Start the dashboard server pointing to your Swiftuna database
optuna-dashboard sqlite:///experiments.db
```

Open `http://127.0.0.1:8080` in your web browser to explore:
- **Optimization History:** Interactive curves tracking best objective values over time.
- **Pareto Frontier:** 2D and 3D scatter plots for multi-objective studies.
- **Parameter Relationships:** Slice and contour plots highlighting interaction between parameters.
- **Trial Timelines:** Execution duration and intermediate step reporting curves.

---

## Lockless journal storage for high-concurrency clusters

When running hundreds of worker tasks on shared network filesystems (such as NFS or Lustre), SQLite file-locking can cause lock contention and latency spikes.

The ``StorageBackend/journal(path:)`` engine uses an append-only log format:
- Workers append new trial events directly to the log file without table locks.
- State reconciliation happens in memory on demand.

```swift
let storage = StorageBackend.journal(path: "/shared/nfs/cluster_run.log")
let study = try Swiftuna.createStudy(
    name: "distributed_cluster",
    storage: storage
)
```

---

## Storage lifecycle operations

Manage studies programmatically across persistent databases:

```swift
let storage = StorageBackend.sqlite(path: "experiments.db")

// 1. List all studies and inspect metadata
let summaries = try storage.studies()
for summary in summaries.minTrials(10).sortedByTrialCount() {
    print("Study \(summary.name): \(summary.trialCount) trials recorded")
}

// 2. Load an existing study
if let study = try Swiftuna.loadStudy(name: "resnet_cifar10", storage: storage) {
    print("Loaded study '\(study.name)' with best value \(try study.bestValue)")
    
    // 3. Copy from SQLite to in-memory for rapid analytical sweeps
    let inMemoryCopy = try study.copy(to: .inMemory, as: "fast_sweep")
}

// 4. Delete an old study and cascade remove all its trials
try storage.deleteStudy(named: "old_experiment")
```
