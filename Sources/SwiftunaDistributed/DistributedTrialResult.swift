import Foundation
public import Swiftuna

/// The evaluated result of a trial returned by a worker to the coordinator.
public struct DistributedTrialResult: Sendable, Codable {
    /// The trial number matching the requested ``DistributedTrialSpec/trialNumber``.
    public let trialNumber: Int

    /// Evaluated objective value(s). Empty for pruned or failed trials.
    public let values: [Double]

    /// The resulting state of the trial execution.
    public let state: TrialState

    /// Evaluated constraints ($v \le 0.0$ indicates satisfaction).
    public let constraints: [String: Double]

    /// Metadata attributes attached by the worker during execution.
    public let userAttrs: [String: String]

    /// Creates a single-objective completed trial result.
    public init(
        trialNumber: Int,
        value: Double,
        constraints: [String: Double] = [:],
        userAttrs: [String: String] = [:]
    ) {
        self.trialNumber = trialNumber
        self.values = [value]
        self.state = .complete
        self.constraints = constraints
        self.userAttrs = userAttrs
    }

    /// Creates a multi-objective trial result with explicit state.
    public init(
        trialNumber: Int,
        values: [Double],
        state: TrialState = .complete,
        constraints: [String: Double] = [:],
        userAttrs: [String: String] = [:]
    ) {
        self.trialNumber = trialNumber
        self.values = values
        self.state = state
        self.constraints = constraints
        self.userAttrs = userAttrs
    }

    /// Factory method for creating an early-pruned trial result.
    public static func pruned(trialNumber: Int) -> DistributedTrialResult {
        DistributedTrialResult(
            trialNumber: trialNumber,
            values: [],
            state: .pruned
        )
    }

    /// Factory method for creating a failed trial result.
    public static func failed(trialNumber: Int) -> DistributedTrialResult {
        DistributedTrialResult(
            trialNumber: trialNumber,
            values: [],
            state: .fail
        )
    }
}
