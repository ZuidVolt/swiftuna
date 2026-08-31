# The ask-and-tell interface

Run manual execution loops, distributed worker pools, batch evaluations, and manage Swift 6 non-copyable trial ownership.

## Overview

The standard ``Study/optimize(nTrials:timeout:objective:)-3gyl5`` method runs a synchronous loop in a single process. However, real-world machine learning systems frequently demand more control:
- Distributing trials across worker nodes or GPU instances
- Grouping parameter candidates into batches for parallel tensor execution
- Interfacing with external subprocesses, Docker containers, or remote microservices
- Implementing custom trial retry policies or error recovery

The **ask-and-tell** interface splits parameter generation (``Study/ask()``) from result reporting (``Study/tell(consuming:value:state:)``).

---

## Non-copyable trial semantics

In Swiftuna, ``Trial`` is a non-copyable type (`~Copyable`). This brings compile-time safety to optimization workflows:
- **No use-after-consume.** Once a trial is passed to `study.tell(consuming:)`, Swift's ownership model marks the variable consumed. Accessing it again causes a compile-time error.
- **No duplicate execution.** Because non-copyable values cannot be cloned implicitly, multiple workers cannot accidentally run the same trial instance.

```swift
var trial = try study.ask()
let lr = try trial.suggest("lr", in: 1e-4...1e-1)
let loss = trainModel(lr: lr)

// trial is consumed here; subsequent lines cannot touch it
try study.tell(consuming: trial, value: loss)
```

---

## Sequential ask-and-tell loop

A basic manual loop queries a trial, runs the workload inside a `do-catch` block, and records either a `.complete` or `.fail` state:

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
        // If evaluation throws, record the trial as failed without crashing the study
        try study.tell(consuming: trial, state: .fail)
    }
}
```

---

## Batch suggestions for GPU parallelism

When evaluating models on GPUs, running one configuration at a time can leave hardware underutilized. You can ask for a batch of trials upfront, evaluate them as a parallel tensor batch, and report the scores back:

```swift
let study = try Swiftuna.createStudy(name: "batch_eval", direction: .minimize)

let batchSize = 8
var activeBatch: [Trial] = []

// 1. Collect a batch of candidate configurations
for _ in 0..<batchSize {
    activeBatch.append(try study.ask())
}

// 2. Suggest parameters for each trial in the batch
var configs: [(lr: Double, wd: Double)] = []
for i in 0..<activeBatch.count {
    let lr = try activeBatch[i].suggest("lr", in: 1e-4...1e-1, log: true)
    let wd = try activeBatch[i].suggest("wd", in: 1e-6...1e-2, log: true)
    configs.append((lr: lr, wd: wd))
}

// 3. Run parallel GPU batch evaluation
let losses = await evaluateBatchOnGPU(configs)

// 4. Report all results back
for (trial, loss) in zip(activeBatch, losses) {
    try study.tell(consuming: trial, value: loss)
}
```

---

## Parallel workers with Swift structured concurrency

Because ``Study`` is `Sendable`, multiple asynchronous tasks can safely share a single study instance. Below is a complete example coordinating parallel workers with a Swift `TaskGroup`:

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
                
                let loss = await evaluateOnGPU(x, worker: workerID)
                
                try study.tell(consuming: trial, value: loss)
            }
        }
    }
    try await group.waitForAll()
}
```

---

## Handling discrete search space exhaustion

When using discrete search algorithms like ``GridSampler``, calling ``Study/ask()`` will throw ``SwiftunaError/searchSpaceExhausted(_:)`` when all parameter combinations have been tested. Always catch this error when running open-ended manual loops:

```swift
while true {
    let trial: Trial
    do {
        trial = try study.ask()
    } catch SwiftunaError.searchSpaceExhausted {
        print("All grid points evaluated. Exiting loop.")
        break
    }

    var active = trial
    let x = try active.suggest("x", in: 1...5)
    let score = evaluate(x)
    try study.tell(consuming: active, value: score)
}
```
