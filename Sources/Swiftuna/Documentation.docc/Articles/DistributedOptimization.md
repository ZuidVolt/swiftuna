# Distributed optimization with SwiftunaDistributed

Build a hyperparameter optimization system that outgrows one process: a coordinator that owns sampling, workers that evaluate trials, and a transport that connects them. This tutorial starts locally with no network, then deploys the same coordinator and workers over WebSockets.

One thing to know up front. Everything here except the transport section works with any Swift `DistributedActorSystem`. `StudyCoordinator` is generic over the system, and the driver and worker functions only ever call `ask`, `report`, and `tell` across the boundary. WebSocketActors is the example transport, not a requirement. Swap in your own system and nothing else changes.

## Why sampling stays on the coordinator

`Trial` is noncopyable, so it cannot cross a process boundary. That constraint shapes the whole design. The coordinator owns the `Study` and runs every suggest call locally. Workers receive a plain `Codable` description of what to try and send back a `Codable` result. Only values and strings cross the wire, never trial handles.

## Install it

`SwiftunaDistributed` ships as its own library product, separate from the `Swiftuna` target that holds `Study` and `Trial`. Add both products to any target that samples or coordinates. Workers that only evaluate need `SwiftunaDistributed` for the spec and result types. They never link Rustuna directly, since all FFI stays coordinator side. The WebSocket dependency below is only needed for the deployment section.

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

## Part 1: run twenty trials in one call (the primary path)

Define the search space on the coordinator as data. Every suggest call runs inside the coordinator process.

```swift
import Swiftuna
import SwiftunaDistributed

let study = try Swiftuna.createStudy(name: "dist_hpo", direction: .minimize)
let space = AskFunction(params: SearchSpaceParams([
    .float(name: "x", lower: -10.0, upper: 10.0),
]))

let system = LocalTestingDistributedActorSystem()
let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)

try await optimize(coordinator: coordinator, nTrials: 20, workers: 4) { ctx in
    let x = ctx.trial.double("x") ?? 0.0
    return x * x
}

if let best = try await coordinator.bestTrial {
    print("Best value: \(best.value ?? 0.0)")
}
```

That is the whole local program. The driver checks trials out across worker tasks, evaluates the objective, and records each returned value, matching `Study.optimize` semantics including fail recording on throws. For multi-objective studies the driver has a vector overload returning one value per direction.

The DSL lowers to the same sampling calls as hand-written closures, stays `Codable` for logging, and duplicate names fail fast instead of trapping deep in a trial. Declare the full static form when the space allows it.

Anything the DSL cannot express goes in the procedural hatch, which runs after the declarative params on the same trial. Conditional dims and derived values live there, and hatch entries win on name collision. The bare closure is the advanced escape hatch for spaces no static list can express at all.

```swift
let space = AskFunction(params: SearchSpaceParams([
    .float(name: "x", lower: -10.0, upper: 10.0),
])) { trial in
    let x = try trial.suggest("x", in: -10.0...10.0)
    return ["layers": .int(x > 0.0 ? 4 : 2)]
}
```

Use sqlite or journal storage with `loadIfExists` when the coordinator must survive restarts. The in-memory default above is fine for learning.

Once the coordinator exists, treat the study as its property. Touching the study directly while trials are in flight races on the same storage handle, and Swift 6 cannot express that exclusivity in the type system for a shared reference type, so it is a documented contract instead. Read through the coordinator: `trials`, `bestTrial`, `bestTrials`, `paramImportances`, and study attrs all have coordinator methods. Direct reads are safe again once `inFlightCount` returns zero.

## Part 2 (advanced): the manual loop, for when the driver is not enough

The driver covers the common shape. When workers need control, such as per-epoch pruning or custom retry, drop to ask, report, and tell directly. Threading trial numbers by hand gets old, so check out a context and let the number travel with it.

```swift
extension AttributeKey where Value == String {
    static let region = AttributeKey<String>("region")
}

let ctx = try await DistributedTrialContext.checkout(from: coordinator)
let x = ctx.trial.double("x") ?? 0.0
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

Report evaluates the pruner remotely and hands back a vote. The worker still decides, which matters on expensive GPU workers that should checkpoint before quitting instead of dying mid epoch. Results carry values, state, constraints, and user attrs, with `pruned` and `failed` factories for the unhappy paths. The trial offers typed readers like `double`, `float`, `int`, `string`, and `param(as:)` for enums. Prefer the typed key-pair initializers above; the raw `[String: ...]` initializers remain for fully dynamic workers whose keys are only known at runtime.

Worker loops that hit a dead end should consult `isRetryable` on the error: only capacity errors reward the same call again, everything else needs a different call.

A few rules worth memorizing. A failed tell keeps the trial in flight, so fix the payload and retry the same tell instead of asking for a new trial. Reporting one step twice keeps the latest value. Constraint values are validated before anything reaches storage: NaN throws `invalidConstraint` and leaves the trial in flight for a corrected retry, mirroring the local `Trial` rules. A repeated tell for a finished trial throws `trialAlreadyFinished`, so workers can tell a retry from a genuinely missing trial. Unknown numbers still throw `trialNotFound`.

## Part 3: run the fleet without babysitting it

Three knobs keep a fleet healthy. First, backpressure. Set `maxInFlight` on init and the coordinator rejects excess asks with `tooManyInFlight` instead of handing out unbounded work. To wait for a slot instead of polling, use the blocking ask with a timeout.

```swift
let coordinator = StudyCoordinator(
    study: study, askFunction: space, actorSystem: system, maxInFlight: 64)

while true {
    do {
        let spec = try await coordinator.ask(waitingUpTo: .seconds(30))
        // evaluate, report, tell as above
    } catch SwiftunaDistributedError.searchSpaceExhausted {
        break
    } catch SwiftunaDistributedError.tooManyInFlight {
        continue // timed out waiting; try again or shut down
    }
}
```

Second, grid exhaustion. When a discrete space runs dry, ask throws `searchSpaceExhausted` instead of an opaque string, so break the worker loop on it. The blocking ask surfaces it at once rather than after the full timeout.

Third, leases for unreliable workers. A worker that dies after ask used to leak its trial forever. Hand the coordinator a lease policy and dead trials get reaped as failed the next time any worker calls ask or report. There are no background timers. The lease starts at ask and every report acts as a heartbeat, so long GPU epochs stay alive as long as they keep reporting.

```swift
let coordinator = StudyCoordinator(
    study: study, askFunction: space, actorSystem: system,
    leasePolicy: LeasePolicy(timeout: .seconds(300)))
```

Leave the policy nil and behavior is exactly what it was before leases existed. When a lease lapses, the trial is recorded as failed with whatever intermediates it reported, its slot frees, and a stale tell for it throws `leaseExpired` with the trial number. Retired numbers are remembered approximately past a few thousand entries, so very old duplicates may degrade to `trialNotFound`. That bound only affects error specificity, never trial data.

Monitor with `inFlightCount`, `finishedTrialsCount` for every terminal state or filtered by state (pass `[.complete]` for completions only), and the read methods from Part 1.

## Part 4: deploy over WebSockets

Nothing above this point touched the network, and that is the point. The coordinator, the driver, the worker loop, leases, and caps are all transport-agnostic. Deploying means hosting the coordinator on a socket and pointing workers at it. The recipe below uses WebSocketActors 1.x and only its public surface, verified with a live round trip in the demo harness. Any `DistributedActorSystem` with `Codable` messages slots into the same shape.

Host the coordinator. Constructing it auto-registers the actor with the system, so read back its assigned id and share that id with workers through config or service discovery.

```swift
import Distributed
import WebSocketActors

// Coordinator host:
let address = ServerAddress(scheme: .insecure, host: "0.0.0.0", port: 8888)
let system = WebSocketActorSystem(id: "hpo-server")
let server = try await system.runServer(at: address) // hold this reference
let coordinator = StudyCoordinator(study: study, askFunction: space, actorSystem: system)
print("Coordinator id: \(coordinator.id)")
```

One caveat found while verifying: the upstream docs show stable ids via `makeLocalActor(id:)`, but that function is internal in 1.1.0, so external code cannot use it. Sharing the assigned `coordinator.id` is the path that actually compiles today.

Each worker connects, resolves the coordinator by id, and runs the same worker program it would run locally. This is the entire remote worker.

```swift
// Worker host:
let client = WebSocketActorSystem()
let connection = try await client.connectClient(to: address) // hold this reference
let coordinatorID = ActorIdentity(id: "<id printed by the host>", node: "hpo-server")
let coordinator = try StudyCoordinator.resolve(id: coordinatorID, using: client)

try await runWorker(coordinator: coordinator) { ctx in
    let x = ctx.trial.double("x") ?? 0.0
    return x * x
}
```

`runWorker` without a trial count loops until the search space is exhausted, which is the normal mode for long-lived remote workers. It sleeps briefly through full-coordinator spells and records failures like the driver does. WebSocketActors adds multi-client connections, automatic reconnection, and server push on top; the coordinator never knows the difference.

## Next steps

- Persist with journal for multi-writer clusters, sqlite for dashboard visibility: <doc:StorageAndDashboard>
- Tune pruning votes for remote workers: <doc:SamplersAndPruners>
- Keep GPU workers busy between report calls: <doc:GPUAndMLX>
