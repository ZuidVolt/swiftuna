public import Foundation

/// Lifecycle state of an optimization trial.
public enum TrialState: Int32, Sendable, CaseIterable, Hashable {
    /// The trial is actively being evaluated by an objective function.
    case running = 0

    /// The trial finished evaluation successfully with valid objective value(s).
    case complete = 1

    /// The trial was stopped early by a ``Pruner`` before completing all iterations.
    case pruned = 2

    /// The trial is pre-enqueued and waiting to be processed by a worker.
    case waiting = 3

    /// The trial encountered an unhandled error or exception during evaluation.
    case fail = 4
}

/// An immutable, evaluated record of a trial persisted in storage.
///
/// `PersistedTrial` captures all parameter suggestions, intermediate evaluations,
/// user-defined metadata attributes, mathematical constraint values, and timestamps.
///
/// ### Example
/// ```swift
/// if let best = try study.bestTrial {
///     print("Best Trial #\(best.number) achieved value \(best.value ?? 0.0)")
///     print("Parameters: \(best.params)")
/// }
/// ```
public struct PersistedTrial: Sendable {
    /// Zero-based sequential trial index within the study.
    public let number: Int

    /// Final lifecycle state of the trial.
    public let state: TrialState

    /// Primary objective value for single-objective trials (or first value for multi-objective).
    public let value: Double?

    /// Array of objective values matching the study's ``Study/directions``.
    public let values: [Double]

    /// Dictionary of hyperparameter names and their evaluated numerical values.
    public let params: [String: Double]

    /// User-defined string metadata attributes attached to the trial.
    public let userAttrs: [String: String]

    /// Mathematical constraint evaluation values ($v \le 0.0$ indicates satisfaction).
    public let constraints: [String: Double]

    /// Intermediate step-by-step values reported via ``Trial/report(_:step:pruneIfWorse:)``.
    public let intermediateValues: [Int: Double]

    /// Timestamp when trial execution began.
    public let datetimeStart: Date?

    /// Timestamp when trial execution completed or failed.
    public let datetimeComplete: Date?

    /// Indicates whether all mathematical constraints are satisfied ($v \le 0.0$).
    public var isFeasible: Bool {
        constraints.values.allSatisfy { $0 <= 0.0 }
    }

    /// Execution duration of the trial in Swift native `Duration`.
    public var duration: Duration? {
        guard let start = datetimeStart, let end = datetimeComplete else { return nil }
        let diff = end.timeIntervalSince(start)
        return .seconds(diff)
    }

    /// Creates a persisted trial record.
    public init(
        number: Int,
        state: TrialState,
        value: Double?,
        values: [Double] = [],
        params: [String: Double],
        userAttrs: [String: String] = [:],
        constraints: [String: Double] = [:],
        intermediateValues: [Int: Double] = [:],
        datetimeStart: Date? = nil,
        datetimeComplete: Date? = nil
    ) {
        self.number = number
        self.state = state
        self.value = value
        self.values = values.isEmpty ? (value.map { [$0] } ?? []) : values
        self.params = params
        self.userAttrs = userAttrs
        self.constraints = constraints
        self.intermediateValues = intermediateValues
        self.datetimeStart = datetimeStart
        self.datetimeComplete = datetimeComplete
    }

    /// Accesses a strongly-typed user attribute value using an ``AttributeKey``.
    public subscript<K: AttributeKey>(_ key: K.Type) -> K.Value? {
        guard let str = userAttrs[K.name] else { return nil }
        return K.Value.fromAttributeString(str)
    }

    /// Accesses a constraint evaluation value using a ``ConstraintKey``.
    public subscript<K: ConstraintKey>(_ key: K.Type) -> Double? {
        constraints[K.name]
    }

    /// Accesses a raw string user attribute by string name.
    public subscript(_ key: String) -> String? {
        userAttrs[key]
    }

    /// Deserializes a user attribute to a specific ``AttributeConvertible`` type by string name.
    ///
    /// - Parameters:
    ///   - type: The target type to convert to.
    ///   - key: The attribute key string.
    /// - Returns: Deserialized value, or `nil` if missing or decoding failed.
    public func userAttr<T: AttributeConvertible>(as type: T.Type, key: String) -> T? {
        guard let str = userAttrs[key] else { return nil }
        return T.fromAttributeString(str)
    }
}

// MARK: - Sequence Analytics Extensions (Optuna Calibration Parity)

extension Sequence where Element == PersistedTrial {
    /// Filters trials to only those with ``TrialState/complete`` state and valid objective value(s).
    public func completed() -> [PersistedTrial] {
        filter { $0.state == .complete && $0.value != nil }
    }

    /// Filters trials to only those with ``TrialState/pruned`` state.
    public func pruned() -> [PersistedTrial] {
        filter { $0.state == .pruned }
    }

    /// Filters trials to only those where all constraint values are feasible ($v \le 0.0$).
    public func feasible() -> [PersistedTrial] {
        filter(\.isFeasible)
    }

    /// Filters trials to only those where at least one constraint is violated ($v > 0.0$).
    public func infeasible() -> [PersistedTrial] {
        filter { !$0.isFeasible }
    }

    /// Returns the optimal completed and feasible trial according to `direction`, or `nil` if none exist.
    ///
    /// - Parameter direction: The optimization direction (defaults to ``Direction/minimize``).
    /// - Returns: The best feasible completed trial.
    public func bestFeasible(direction: Direction = .minimize) -> PersistedTrial? {
        feasible().completed().min { a, b in
            guard let valA = a.value, let valB = b.value else { return false }
            return direction == .minimize ? (valA < valB) : (valA > valB)
        }
    }

    /// Sorts trials by primary objective value.
    ///
    /// - Parameter ascending: If `true` (default), sorts lowest to highest; if `false`, highest to lowest.
    public func sortedByValue(ascending: Bool = true) -> [PersistedTrial] {
        self.filter { $0.value != nil }.sorted {
            ascending ? ($0.value! < $1.value!) : ($0.value! > $1.value!)
        }
    }

    /// Returns the top `n` trials sorted by objective value.
    ///
    /// - Parameters:
    ///   - n: Number of trials to return.
    ///   - ascending: If `true` (default), lowest values first; if `false`, highest values first.
    public func top(_ n: Int, ascending: Bool = true) -> [PersistedTrial] {
        Array(sortedByValue(ascending: ascending).prefix(n))
    }

    /// Filters trials whose primary objective value falls within `tolerance` of `baselineValue`.
    public func within(tolerance: Double, of baselineValue: Double) -> [PersistedTrial] {
        filter {
            guard let val = $0.value else { return false }
            return val <= baselineValue + tolerance
        }
    }

    /// Extracts all values recorded for a specific parameter across this sequence of trials.
    ///
    /// - Parameter paramName: Name of the hyperparameter.
    /// - Returns: Array of sampled numerical values.
    public func values(for paramName: String) -> [Double] {
        compactMap { $0.params[paramName] }
    }

    /// Computes the empirical minimum and maximum bounds observed for each parameter across this sequence of trials.
    ///
    /// - Returns: Dictionary mapping parameter names to their observed `ClosedRange<Double>`.
    public func parameterIntervals() -> [String: ClosedRange<Double>] {
        var minValues: [String: Double] = [:]
        var maxValues: [String: Double] = [:]

        for trial in self {
            for (paramName, paramVal) in trial.params {
                minValues[paramName] = Swift.min(minValues[paramName] ?? paramVal, paramVal)
                maxValues[paramName] = Swift.max(maxValues[paramName] ?? paramVal, paramVal)
            }
        }

        var intervals: [String: ClosedRange<Double>] = [:]
        for (paramName, minVal) in minValues {
            if let maxVal = maxValues[paramName] {
                intervals[paramName] = minVal...maxVal
            }
        }
        return intervals
    }
}
