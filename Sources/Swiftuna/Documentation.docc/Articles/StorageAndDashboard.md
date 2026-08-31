# Storage Backends & Optuna Dashboard

Persist optimization trials across process restarts, scale concurrent workers, and visualize study progress in real time with Optuna Dashboard.

## Overview

Swiftuna provides three storage backends:
1. ``StorageBackend/inMemory``: Fast in-memory storage for single-process jobs.
2. ``StorageBackend/sqlite(path:)``: SQLite database 100% byte-compatible with Python Optuna and `optuna-dashboard`.
3. ``StorageBackend/journal(path:)``: Append-only lockless journal log for multi-process / HPC environments.

---

## SQLite Storage & Resuming Studies

Persisting studies to SQLite allows long-running experiments to survive process restarts and accumulate trials incrementally.

```swift
import Swiftuna

let storage = StorageBackend.sqlite(path: "experiments.db")

// Create or resume study if already present
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

## Real-Time Visualization with Optuna Dashboard

Because Swiftuna's SQLite schema is 100% compatible with Optuna's `RDBStorage`, you can run `optuna-dashboard` directly against Swiftuna databases:

```bash
# 1. Install optuna-dashboard via pip or uv
pip install optuna-dashboard

# 2. Launch dashboard pointing to your Swiftuna SQLite database
optuna-dashboard sqlite:///experiments.db
```

Open `http://127.0.0.1:8080` in your web browser to explore:
- Interactive optimization history plots
- Hyperparameter slice and contour plots
- Empirical Pareto frontier scatter plots
- Real-time trial timeline metrics

---

## Lockless Journal Storage for High Concurrency

When running hundreds of concurrent workers on shared network filesystems (NFS/Lustre), SQLite table locking can create latency bottlenecks. The ``StorageBackend/journal(path:)`` engine uses an append-only log format:

```swift
let storage = StorageBackend.journal(path: "/shared/nfs/cluster_run.log")
let study = try Swiftuna.createStudy(
    name: "distributed_hpc",
    storage: storage
)
```

---

## Storage Lifecycle Operations

Query, filter, copy, and delete studies in any storage backend:

```swift
let storage = StorageBackend.sqlite(path: "experiments.db")

// List all studies
let summaries = try storage.studies()
for s in summaries.minTrials(10).sortedByTrialCount() {
    print("Study \(s.name): \(s.trialCount) trials")
}

// Copy a study from SQLite to In-Memory for rapid analysis
if let source = try Swiftuna.loadStudy(name: "resnet_cifar10", storage: storage) {
    let memStudy = try source.copy(to: .inMemory, as: "analysis_copy")
    print("Copied \(try memStudy.trials.count) trials to RAM")
}

// Delete a study and cascade remove its trials
try storage.deleteStudy(named: "old_experiment")
```
