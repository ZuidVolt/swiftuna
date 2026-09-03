public import Distributed
import Foundation
import Swiftuna

/// Runs `nTrials` trials against `coordinator` across `workers` local workers.
///
/// This is the distributed equivalent of `Study.optimize`: one call replaces
/// the manual checkout/report/tell loop. Each worker checks out trials,
/// evaluates `objective`, and records the returned value as completed.
///
/// A free function rather than a coordinator method because `objective` is a
/// closure: distributed methods only accept `Codable` parameters, so the
/// driver must live outside the actor. Workers run in the caller's process;
/// only `ask`/`report`/`tell` cross the actor boundary, so this also works
/// against a remote coordinator proxy.
///
/// - Parameters:
///   - coordinator: Coordinator to check trials out from.
///   - nTrials: Total trials to run.
///   - workers: Parallel worker tasks. Defaults to 4.
///   - objective: Evaluates a checked-out trial (via its `trial` and
///     `report`) and returns the objective value. If `objective` settles
///     the trial itself (e.g. `prune()`), the driver's completion tell is
///     skipped for that trial.
/// - Throws: The first error thrown by `objective` or the coordinator.
///   A throwing trial is recorded as `.fail` before the error propagates;
///   earlier trials stay recorded.
///
/// ```swift
/// try await optimize(coordinator: coordinator, nTrials: 20, workers: 4) { ctx in
///     let x = ctx.trial.double("x") ?? 0.0
///     return x * x
/// }
/// ```
public func optimize<ActorSystem>(
    coordinator: StudyCoordinator<ActorSystem>,
    nTrials: Int,
    workers: Int = 4,
    objective: @Sendable @escaping (DistributedTrialContext<ActorSystem>) async throws -> Double
) async throws where ActorSystem: DistributedActorSystem<any Codable> {
    try await optimize(coordinator: coordinator, nTrials: nTrials, workers: workers) { ctx in
        [try await objective(ctx)]
    }
}

/// Runs `nTrials` multi-objective trials, one objective vector per trial.
///
/// The vector length must match the study's `directions`; a mismatch throws
/// and the trial stays in flight for a corrected retry.
public func optimize<ActorSystem>(
    coordinator: StudyCoordinator<ActorSystem>,
    nTrials: Int,
    workers: Int = 4,
    objective: @Sendable @escaping (DistributedTrialContext<ActorSystem>) async throws -> [Double]
) async throws where ActorSystem: DistributedActorSystem<any Codable> {
    precondition(nTrials >= 0, "nTrials must be non-negative")
    precondition(workers >= 1, "workers must be at least 1")
    try await withThrowingTaskGroup(of: Void.self) { group in
        for worker in 0..<workers {
            let count = nTrials / workers + (worker < nTrials % workers ? 1 : 0)
            group.addTask {
                try await runWorker(coordinator: coordinator, nTrials: count, objective: objective)
            }
        }
        try await group.waitForAll()
    }
}

/// Runs a worker loop against `coordinator`: check out, evaluate, record.
///
/// This is the remote worker's whole program. Point it at a coordinator proxy
/// resolved over any transport and it behaves exactly like an `optimize` worker:
///
/// ```swift
/// let coordinator = try StudyCoordinator.resolve(id: coordinatorID, using: client)
/// try await runWorker(coordinator: coordinator) { ctx in
///     evaluate(ctx.trial)
/// }
/// ```
///
/// - Parameters:
///   - coordinator: Coordinator (local or remote proxy) to pull trials from.
///   - nTrials: Trials to run. `nil` (default) loops until the search space is
///     exhausted, which is the normal mode for long-lived remote workers.
///   - objective: Evaluates a checked-out trial and returns the objective value.
///     Trials it settles itself (e.g. `prune()`) skip the completion tell.
/// - Throws: Objective errors (after recording the trial as `.fail`) and
///   coordinator errors other than `searchSpaceExhausted` and `tooManyInFlight`.
///   A full coordinator sleeps the worker briefly and retries; exhaustion ends
///   the loop normally only when `nTrials` is `nil`.
public func runWorker<ActorSystem>(
    coordinator: StudyCoordinator<ActorSystem>,
    nTrials: Int? = nil,
    objective: @Sendable @escaping (DistributedTrialContext<ActorSystem>) async throws -> Double
) async throws where ActorSystem: DistributedActorSystem<any Codable> {
    try await runWorker(coordinator: coordinator, nTrials: nTrials) { ctx in
        [try await objective(ctx)]
    }
}

/// Runs a multi-objective worker loop against `coordinator`.
///
/// Same contract as the single-objective variant, with one objective vector
/// per trial matching the study's `directions`.
public func runWorker<ActorSystem>(
    coordinator: StudyCoordinator<ActorSystem>,
    nTrials: Int? = nil,
    objective: @Sendable @escaping (DistributedTrialContext<ActorSystem>) async throws -> [Double]
) async throws where ActorSystem: DistributedActorSystem<any Codable> {
    if let nTrials {
        precondition(nTrials >= 0, "nTrials must be non-negative")
        for _ in 0..<nTrials {
            try await runOneTrial(coordinator: coordinator, objective: objective)
        }
        return
    }
    while true {
        do {
            try await runOneTrial(coordinator: coordinator, objective: objective)
        } catch SwiftunaDistributedError.searchSpaceExhausted {
            return
        } catch SwiftunaDistributedError.tooManyInFlight {
            try await Task.sleep(for: .milliseconds(100))
        }
    }
}

/// Checks out, evaluates, and records a single trial.
private func runOneTrial<ActorSystem>(
    coordinator: StudyCoordinator<ActorSystem>,
    objective: @Sendable @escaping (DistributedTrialContext<ActorSystem>) async throws -> [Double]
) async throws where ActorSystem: DistributedActorSystem<any Codable> {
    let ctx = try await DistributedTrialContext.checkout(from: coordinator)
    do {
        let values = try await objective(ctx)
        do {
            try await coordinator.tell(
                DistributedTrialResult(trialNumber: ctx.trialNumber, values: values))
        } catch SwiftunaDistributedError.trialAlreadyFinished {
            // Objective settled the trial itself (prune/fail).
        }
    } catch {
        try? await ctx.fail()
        throw error
    }
}
