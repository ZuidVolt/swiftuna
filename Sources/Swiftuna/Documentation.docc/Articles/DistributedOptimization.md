# Distributed optimization with SwiftunaDistributed

Run hyperparameter trials on remote workers with Swift distributed actors, and use WebSocketActors for the transport.

## Why sampling stays on the coordinator

Trial is noncopyable, so it cannot cross a process boundary. That constraint shapes the whole design. The coordinator owns the Study and runs every suggest call locally. Workers get a plain Codable description of what to try and send back a Codable result. Only Doubles and strings cross the wire, never trial handles.

StudyCoordinator is a distributed actor generic over any DistributedActorSystem with a Codable requirement. It holds the Study, a SearchSpace closure, and a table of in-flight trials. Workers call ask to get a DistributedTrialSpec, stream per-epoch metrics through report, which returns the pruner's early stopping vote, and finish with tell and a DistributedTrialResult. The spec has typed readers like double, float, int, string, and param(as:) for enums. Results carry values, state, constraints, and userAttrs, plus pruned and failed factories for the unhappy paths. Monitoring comes from inFlightCount, completedTrialsCount, and bestTrial.

## Install it

SwiftunaDistributed ships as its own library product, separate from the Swiftuna target that holds Study and Trial. Add both products to the target that needs them.

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/ZuidVolt/swiftuna.git", branch: "main"),
    .package(url: "https://github.com/samalone/websocket-actor-system.git", from: "1.0.0"),
],
targets: [
    .executableTarget(
        name: "HPOCoordinator",
        dependencies: [
            .product(name: "Swiftuna", package: "swiftuna"),
            .product(name: "SwiftunaDistributed", package: "swiftuna"),
            .product(name: "WebSocketActors", package: "websocket-actor-system"),
        ]
    ),
]
```

Workers only need SwiftunaDistributed for the spec and result types, plus WebSocketActors for the connection. They never link Rustuna directly, since all FFI stays coordinator side.

## Define the search space once, on the coordinator

Every suggest call runs inside the coordinator process through a SearchSpace closure.

```swift
import Swiftuna
import SwiftunaDistributed

let study = try Swiftuna.createStudy(name: "dist_hpo", direction: .minimize)
let space = SearchSpace { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return ["x": .double(x)]
}
```

Use sqlite or journal storage with loadIfExists when the coordinator must survive restarts. The in-memory default in this snippet is fine for tests and that is what the harness runs.

For static spaces, declare params as data instead of a closure. The DSL lowers to the same sampling calls, stays Codable for logging, and duplicate names fail fast instead of trapping deep in a trial.

```swift
let space = SearchSpace(params: SearchSpaceParams([
    .float(name: "x", lower: -10.0, upper: 10.0, log: true),
    .int(name: "layers", lower: 1, upper: 8),
    .categorical(name: "opt", choices: ["adam", "sgd"]),
]))
```

Anything the DSL cannot express goes in the procedural hatch, which runs after the declarative params on the same trial. Conditional dims and derived values live there, and hatch entries win on name collision.

```swift
let space = SearchSpace(params: SearchSpaceParams([
    .float(name: "x", lower: -10.0, upper: 10.0),
])) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return ["layers": .int(x > 0.0 ? 4 : 2)]
}
```

## The worker loop

Once the coordinator exists, treat the study as its property. Touching the study directly while trials are in flight races on the same storage handle, and Swift 6 cannot express that exclusivity in the type system for a shared reference type, so it is a documented contract instead. Read through the coordinator. Direct reads are safe again once inFlightCount returns zero.


```swift
let system = LocalTestingDistributedActorSystem()
let coordinator = StudyCoordinator(study: study, searchSpace: space, actorSystem: system)
let spec = try await coordinator.ask()
let x = spec.double("x") ?? 0.0
for epoch in 1...3 {
    let shouldStop = try await coordinator.report(
        trialNumber: spec.trialNumber, step: epoch, value: (x - 3.0) * (x - 3.0))
    if shouldStop {
        try await coordinator.tell(.pruned(trialNumber: spec.trialNumber))
        break
    }
}
try await coordinator.tell(
    DistributedTrialResult(
        trialNumber: spec.trialNumber,
        value: (x - 3.0) * (x - 3.0),
        constraints: ["limit": -1.0],
        userAttrs: ["worker_id": "mac-studio-01"]))
```

Report evaluates the pruner remotely and hands back a vote. The worker still decides. That split matters on expensive GPU workers, because the worker can checkpoint before quitting instead of getting killed mid epoch. Concurrent workers just share one coordinator proxy inside a TaskGroup. The test target has a 4 worker, 16 trial simulation worth copying. Monitor with inFlightCount, completedTrialsCount for .complete trials, finishedTrialsCount for every terminal state, bestTrial, bestTrials for multi-objective frontiers, trials filtered by state, paramImportances, and study-level user attrs shared through setUserAttr and userAttr.

A failed tell keeps the trial in flight, so fix the payload and retry the same tell instead of asking for a new trial. Reporting one step twice keeps the latest value.

Two errors deserve explicit catches. When a grid space runs dry, ask throws searchSpaceExhausted instead of an opaque string, so break the worker loop on it. When the coordinator is full, ask throws tooManyInFlight with the current count. Set the limit with maxInFlight on init; it defaults to unbounded.

```swift
let coordinator = StudyCoordinator(
    study: study, searchSpace: space, actorSystem: system, maxInFlight: 64)

while true {
    do {
        let spec = try await coordinator.ask()
        // evaluate, report, tell as above
    } catch SwiftunaDistributedError.searchSpaceExhausted {
        break
    } catch SwiftunaDistributedError.tooManyInFlight {
        try await Task.sleep(for: .milliseconds(100))
    }
}
```

## The context shortcut and typed payloads

Threading trial numbers through every call gets old. Check out a context instead and the trial number travels with it. Pair it with typed key initializers so workers stop hand-rolling string keys.

```swift
extension AttributeKey where Value == String {
    static let region = AttributeKey<String>("region")
}

let ctx = try await DistributedTrialContext.checkout(from: coordinator)
let x = ctx.spec.double("x") ?? 0.0
for epoch in 1...3 {
    if try await ctx.report(step: epoch, value: (x - 3.0) * (x - 3.0)) {
        try await ctx.prune()
        break
    }
}
try await ctx.tell(
    value: (x - 3.0) * (x - 3.0),
    constraintPairs: [(ConstraintKey("limit"), -1.0)],
    userAttrPairs: [(AttributeKey.region.name, "eu")])
```

Constraint values are validated before anything reaches storage. NaN throws invalidConstraint and leaves the trial in flight for a corrected retry, mirroring the local Trial rules.

A repeated tell for a finished trial throws trialAlreadyFinished, so workers can tell a retry from a genuinely missing trial. Unknown numbers still throw trialNotFound.

## Leases for unreliable workers

A worker that dies after ask used to leak its trial forever. Hand the coordinator a lease policy and dead trials get reaped as failed the next time any worker calls ask or report. There are no background timers. The lease starts at ask and every report acts as a heartbeat, so long GPU epochs stay alive as long as they keep reporting.

```swift
let coordinator = StudyCoordinator(
    study: study, searchSpace: space, actorSystem: system,
    leasePolicy: LeasePolicy(timeoutSeconds: 300))
```

Leave the policy nil and behavior is exactly what it was before leases existed. When a lease lapses, the trial is recorded as failed with whatever intermediates it reported, its slot frees, and a stale tell for it throws leaseExpired with the trial number. Retired numbers are remembered approximately past a few thousand entries, so very old duplicates may degrade to trialNotFound. That bound only affects error specificity, never trial data.

## WebSocket transport, the parts that matter

You need three things from WebSocketActors. Stable identities in shared code, a server bootstrap on the coordinator host, and a client resolve on each worker. These compile against version 1.1.0 in the harness without opening a real connection.

```swift
import Distributed
import WebSocketActors

extension NodeIdentity {
    static let hpoServer = NodeIdentity(id: "hpo-server")
}
extension ActorIdentity {
    static let coordinatorID = ActorIdentity(id: "coordinator", node: .hpoServer)
}
```

Host the coordinator.

```swift
let address = ServerAddress(scheme: .insecure, host: "0.0.0.0", port: 8888)
let system = WebSocketActorSystem(id: .hpoServer)
try await system.runServer(at: address)
_ = system.makeLocalActor(id: .coordinatorID) {
    StudyCoordinator(study: study, searchSpace: space, actorSystem: system)
}
```

Connect each worker and reuse the loop from the previous section unchanged.

```swift
let client = WebSocketActorSystem()
try await client.connectClient(to: address)
let coordinator = try StudyCoordinator.resolve(id: .coordinatorID, using: client)
let spec = try await coordinator.ask()
// evaluate, report, tell exactly as above
```

WebSocketActors buys multi-client connections, automatic reconnection, server push, and SwiftLog, all derived from Apple's TicTacFish sample. What it does not fix is the gap in SwiftunaDistributed itself. There is no lease timeout, heartbeat, or redelivery. A worker that dies after ask leaves its in-flight trial stashed until the coordinator restarts. Run a sweeper on inFlightCount plus completedTrialsCount and requeue anything stuck past your own deadline until leases land upstream.

## Next steps

- Persist with journal for multi-writer clusters, sqlite for dashboard visibility: <doc:StorageAndDashboard>
- Tune pruning votes for remote workers: <doc:SamplersAndPruners>
- Keep GPU workers busy between report calls: <doc:GPUAndMLX>
