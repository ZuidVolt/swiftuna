import Foundation
public import Swiftuna

/// The evaluated result of a trial returned by a worker to the coordinator.
public struct DistributedTrialResult: Sendable, Codable {
    /// The trial number matching the requested ``DistributedTrial/trialNumber``.
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

    /// Creates a single-objective completed trial result with typed keys.
    ///
    /// Distinct `constraintPairs` / `userAttrPairs` labels keep this overload
    /// unambiguous next to the `[String: ...]` dictionary initializers.
    ///
    /// - Parameters:
    ///   - constraintPairs: (`ConstraintKey`, value) pairs. For `ConstraintKeyProtocol`
    ///     types, wrap the name: `ConstraintKey(MyKey.name)`.
    ///   - userAttrPairs: (name, value) pairs serialized via ``AttributeConvertible``.
    ///     For typed keys, pass `MyKey.name` as the name.
    public init(
        trialNumber: Int,
        value: Double,
        constraintPairs: [(ConstraintKey, Double)],
        userAttrPairs: [(String, any AttributeConvertible)]
    ) {
        self.init(
            trialNumber: trialNumber,
            values: [value],
            state: .complete,
            constraints: Dictionary(uniqueKeysWithValues: constraintPairs.map { ($0.0.name, $0.1) }),
            userAttrs: Dictionary(uniqueKeysWithValues: userAttrPairs.map { ($0.0, $0.1.toAttributeString()) })
        )
    }

    /// Creates a multi-objective trial result with typed keys.
    public init(
        trialNumber: Int,
        values: [Double],
        state: TrialState = .complete,
        constraintPairs: [(ConstraintKey, Double)],
        userAttrPairs: [(String, any AttributeConvertible)]
    ) {
        self.init(
            trialNumber: trialNumber,
            values: values,
            state: state,
            constraints: Dictionary(uniqueKeysWithValues: constraintPairs.map { ($0.0.name, $0.1) }),
            userAttrs: Dictionary(uniqueKeysWithValues: userAttrPairs.map { ($0.0, $0.1.toAttributeString()) })
        )
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
