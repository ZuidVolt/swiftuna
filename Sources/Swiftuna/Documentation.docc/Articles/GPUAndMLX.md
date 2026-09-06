# GPU optimization with MLX

Train on Apple Silicon without burning CPU cycles on synchronization. This is how MLX lazy evaluation, eval batching, and Swift 6 concurrency fit Swiftuna trials.

## How MLX actually executes

MLX records array ops into a graph and runs nothing until you eval or read a scalar with item. Apple Silicon has one memory pool for CPU and GPU, so there are no host to device copies to schedule. That removes a whole class of CUDA style tuning and replaces it with a simpler discipline. Eval once per boundary, sync to the CPU once per epoch, and never let the graph grow unbounded.

MLXArray is not Sendable on purpose. Arrays hold graph references, and lazy ops are not thread safe. Each task must own its model and arrays. Only plain Doubles cross task boundaries on the way to report and tell.

## The per-epoch shape that wastes the least

Suggest with the Float overload so learning rates feed the optimizer with no cast, build one model per trial, and keep a single barrier plus a single sync per epoch.

```swift
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

let lossAndGrad = valueAndGrad(model: model) { (m: TinyMLP, xb: MLXArray, yb: MLXArray) in
    ((m(xb) - yb).square().mean())
}

for _ in 1...maxEpochs {
    let (loss, grads) = lossAndGrad(model, x, y)
    optimizer.update(model: model, gradients: grads)
    eval(model, optimizer)
    eval(loss)
    let valLoss = Double(loss.item(Float.self))
    try trial.report(valLoss, step: epoch, pruneIfWorse: true)
}
```

One eval call for model plus optimizer fuses the update barrier. Splitting eval per array or per batch chops GPU submissions into pieces and stalls the Metal queue. Item forces a full graph sync, so calling it per batch serializes every micro-batch through the CPU. That is the most common way ports waste time. Accumulate on GPU, sync once, then report. You still eval every epoch no matter what, because a graph that never flushes eats memory and hides failures until the worst moment. Same for eval(x, y) after dataset creation and eval(model) after init.

## Report validation loss and let the pruner vote

This mirrors the Optuna PyTorch pattern. Log scale learning rates, report on the validation split, never the test set, and let Hyperband or Median decide. The throwing form suits plain loops. The explicit form suits loops with checkpoints to flush.

```swift
try trial.report(valLoss, step: epoch)
if try trial.shouldPrune {
    saveCheckpoint()
    try trial.prune()
}
```

Size pruner rungs to GPU cost. The demo runs HyperbandPruner(minResource: 3, maxResource: 9, reductionFactor: 3) on a 9 epoch budget, so bad architectures die after 3 epochs instead of burning all 9.

## Keep trials shaped for the GPU

Suggest Float rates and stepped Int widths so tensor shapes stay aligned. Generate the dataset once outside optimize, per-trial work should be model init plus matmuls. Cap the search with a constraint like totalParams minus your budget so TPE stops spending GPU epochs on giants that could never ship.

Warm the surrogate with a known good config. TPE builds its density models from finished trials, so the first trials are close to random. Handing it one solid point up front, a learning rate and width you already trust, gives every later suggestion something to improve on. Each GPU epoch is expensive, and a blind opening round is the easiest place to waste them.

```swift
try study.enqueue(["lr": .double(0.01), "hidden_dim": .int(32)])
try study.optimize(nTrials: 50) { trial in /* ... */ }
```

Enqueued trials run first, in order, before sampling takes over. Anything you omit from the dict still gets sampled, so enqueue a partial config and let the sampler fill the rest.

Once the study has finished trials, check which knobs actually mattered. PED-ANOVA scores each parameter by how much of the validation loss variance it explains, and it needs at least two completed trials to say anything.

```swift
if case .success(let importances) = study.paramImportances() {
    for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
        print("\(param): \(String(format: "%.1f%%", score * 100))")
    }
}
```

Read the ranking as a pruning tool for the search space itself. A parameter near zero never moved the needle, so fix it to its best value or drop it and spend those GPU epochs on the dims at the top. Then narrow the ranges around the winners and run the next study smaller and hotter.

## Swift 6 concurrency, TaskGroup, and NIO

Trials are independent, so the choice is how to overlap them. MLX serializes eval on a global lock, and in-memory ask and tell take microseconds, which means overlapping only pays off once a single trial runs seconds or minutes. Short trials run fastest in a plain serial loop. Reach for concurrency when each trial trains long enough that overlap hides real waiting.

The standard choice is a TaskGroup. Study is Sendable, so workers share it directly. Each task builds and owns its MLX model, and nothing but Doubles moves between tasks.

```swift
// Each worker owns its model. trainModel builds, trains, and evals locally.
func trainModel(lr: Float, hiddenDim: Int) -> Double { /* ... */ }

try await withThrowingTaskGroup(of: Void.self) { group in
    for _ in 0..<4 {
        group.addTask {
            for _ in 0..<(nTrials / 4) {
                var trial = try study.ask()
                let lr: Float = try trial.suggest("lr", in: 1e-3...1e-1, log: true)
                let hiddenDim = try trial.suggest("hidden_dim", in: 16...64, step: 16)
                let loss = trainModel(lr: lr, hiddenDim: hiddenDim)
                try study.tell(consuming: trial, value: loss)
            }
        }
    }
    try await group.waitForAll()
}
```

When the whole objective fits this shape, skip the hand-rolled group and use the built-in concurrent optimize with a concurrency limit. The same ownership rule applies. One task, one model, Doubles across the boundary.

NIO enters when the process already lives on event loops, like a server that also serves predictions. EventLoops must never block, and MLX eval plus item both block. Run the trial body on an NIOThreadPool and hand a future back to the loop.

```swift
let future: EventLoopFuture<Double> = pool.runIfActive(eventLoop: loop) {
    var trial = try study.ask()
    let lr: Float = try trial.suggest("lr", in: 1e-3...1e-1, log: true)
    let loss = trainModel(lr: lr, hiddenDim: 32)
    try study.tell(consuming: trial, value: loss)
    return loss
}
let losses = try await EventLoopFuture.whenAllSucceed(futures, on: loop).get()
```

A thread pool is not a faster engine for MLX math. Pool threads are plain pthreads, MLX still runs the compute and still serializes eval internally. The pool only decides where the blocking work waits, keeping event loops free for I/O. With no event loops in the process, a pool adds shutdown bookkeeping over a TaskGroup and nothing else.

One GPU means one model training at a time per process in practice. Run trials sequentially, or give each worker process its own model. For remote GPU workers, pair this with the distributed article. The MLX loop stays local to the worker while report and tell stream Doubles to the coordinator. Watch Memory.activeMemory during long studies. When it climbs, shrink the batch size or accumulate gradients before shrinking the model.

## Next steps

- Pick samplers and pruners for GPU budgets: <doc:SamplersAndPruners>
- Persist GPU studies for the dashboard: <doc:StorageAndDashboard>
- Scale GPU workers over WebSockets: <doc:DistributedOptimization>
