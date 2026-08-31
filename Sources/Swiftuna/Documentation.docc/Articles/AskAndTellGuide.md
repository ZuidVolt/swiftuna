# The Ask-and-Tell Interface

Master manual execution loops, distributed workers, batch suggestions, and Swift 6 non-copyable trial semantics.

## Overview

While ``Study/optimize(nTrials:timeout:objective:)-3gyl5`` provides a convenient automated loop for in-process evaluations, real-world systems often require:
- Asynchronous or distributed evaluation across cluster nodes
- Batch evaluations on GPUs
- External process execution (e.g. CLI tools, Docker containers, remote APIs)
- Custom error recovery and trial retry policies

The **Ask-and-Tell** interface decouples parameter generation (``Study/ask()``) from result recording (``Study/tell(consuming:value:state:)``).

---

## Non-Copyable `~Copyable` Semantics

In Swiftuna, ``Trial`` is a non-copyable type (`~Copyable`). This brings critical safety guarantees at compile time:
- **No Use-After-Consume**: Once a trial is passed to `tell(consuming:)`, it cannot be accessed again.
- **No Accidental Duplication**: Multiple workers cannot evaluate the exact same trial instance.

```swift
var trial = try study.ask()
let lr = try trial.suggest("lr", in: 1e-4...1e-1)
let loss = trainModel(lr: lr)

// trial is consumed here - compiler prevents any further access
try study.tell(consuming: trial, value: loss)
```

---

## Standard Sequential Ask-and-Tell Loop

```swift
let study = try Swiftuna.createStudy(name: "manual_loop", direction: .minimize)

for _ in 1...50 {
    var trial = try study.ask()
    do {
        let x = try trial.suggest("x", in: -10.0...10.0)
        let y = try trial.suggest("y", in: -10.0...10.0)
        let loss = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0)
        
        try study.tell(consuming: trial, value: loss)
    } catch {
        // Record failure without crashing the optimization loop
        try study.tell(consuming: trial, state: .fail)
    }
}
```

---

## Concurrent & Distributed Execution with TaskGroups

Because ``Study`` is `Sendable`, you can coordinate concurrent evaluations across Swift structured concurrency `TaskGroup`s:

```swift
let storage = StorageBackend.sqlite(path: "concurrent_study.db")
let study = try Swiftuna.createStudy(
    name: "parallel_workers",
    storage: storage,
    loadIfExists: true
)

try await withThrowingTaskGroup(of: Void.self) { group in
    for workerID in 1...4 {
        group.addTask {
            for _ in 1...25 {
                var trial = try study.ask()
                let x = try trial.suggest("x", in: -5.0...5.0)
                
                // Simulate asynchronous GPU evaluation
                let loss = await evaluateOnGPU(x, worker: workerID)
                
                try study.tell(consuming: trial, value: loss)
            }
        }
    }
    try await group.waitForAll()
}
```

---

## Multi-Objective Ask-and-Tell

For studies configured with multiple optimization directions, pass an array of `Double` values to `tell`:

```swift
let study = try Swiftuna.createStudy(
    name: "pareto_search",
    directions: [.maximize, .minimize] // Maximize accuracy, minimize latency
)

var trial = try study.ask()
let depth = try trial.suggest("depth", in: 2...10)

let accuracy = evaluateAccuracy(depth: depth)
let latencyMs = measureLatency(depth: depth)

try study.tell(consuming: trial, values: [accuracy, latencyMs])
```

---

## Handling Discrete Exhaustion

When using a discrete sampler such as ``GridSampler``, ``Study/ask()`` will throw ``SwiftunaError/searchSpaceExhausted(_:)`` when all combinations have been evaluated:

```swift
do {
    var trial = try study.ask()
    // ... evaluate ...
    try study.tell(consuming: trial, value: score)
} catch SwiftunaError.searchSpaceExhausted {
    print("All grid points evaluated; search is complete!")
}
```
