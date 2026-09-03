public import Distributed
import Foundation
public import Swiftuna

/// A worker-side handle for one checked-out trial.
///
/// `DistributedTrialContext` removes the `trialNumber` bookkeeping from worker
/// loops: `report` and `tell` already know which trial they belong to.
///
/// ```swift
/// let ctx = try await DistributedTrialContext.checkout(from: coordinator)
/// let x = ctx.spec.double("x") ?? 0.0
/// for epoch in 1...3 {
///     if try await ctx.report(step: epoch, value: x * x) {
///         try await ctx.prune()
///         break
///     }
/// }
/// try await ctx.tell(value: x * x)
/// ```
///
/// Constructing (via ``checkout(from:)``) and every method below are plain
/// local calls — only the underlying `ask`/`report`/`tell` cross the actor
/// boundary — so the context itself never needs to satisfy the actor system's
/// `Codable` requirement.
public struct DistributedTrialContext<ActorSystem>: Sendable
where ActorSystem: DistributedActorSystem<any Codable> {
    /// The checked-out trial specification.
    public let spec: DistributedTrialSpec

    private let coordinator: StudyCoordinator<ActorSystem>

    /// The checked-out trial number.
    public var trialNumber: Int { spec.trialNumber }

    public init(spec: DistributedTrialSpec, coordinator: StudyCoordinator<ActorSystem>) {
        self.spec = spec
        self.coordinator = coordinator
    }

    /// Checks out the next trial from `coordinator` wrapped in a context.
    public static func checkout(from coordinator: StudyCoordinator<ActorSystem>) async throws -> Self {
        Self(spec: try await coordinator.ask(), coordinator: coordinator)
    }

    /// Reports an intermediate value. Returns `true` if the pruner recommends stopping.
    @discardableResult
    public func report(step: Int, value: Double) async throws -> Bool {
        try await coordinator.report(trialNumber: spec.trialNumber, step: step, value: value)
    }

    /// Records the final outcome with a prebuilt result.
    public func tell(_ result: DistributedTrialResult) async throws {
        try await coordinator.tell(result)
    }

    /// Records a single-objective completed outcome with typed keys.
    public func tell(
        value: Double,
        constraintPairs: [(ConstraintKey, Double)] = [],
        userAttrPairs: [(String, any AttributeConvertible)] = []
    ) async throws {
        try await tell(
            DistributedTrialResult(
                trialNumber: spec.trialNumber,
                value: value,
                constraintPairs: constraintPairs,
                userAttrPairs: userAttrPairs
            ))
    }

    /// Records the trial as pruned.
    public func prune() async throws {
        try await tell(.pruned(trialNumber: spec.trialNumber))
    }

    /// Records the trial as failed.
    public func fail() async throws {
        try await tell(.failed(trialNumber: spec.trialNumber))
    }
}
