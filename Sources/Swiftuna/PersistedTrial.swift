import Foundation

public enum TrialState: Int32, Sendable, CaseIterable, Hashable {
    case running = 0
    case complete = 1
    case pruned = 2
    case waiting = 3
    case fail = 4
}

public struct PersistedTrial: Sendable {
    public let number: Int
    public let state: TrialState
    public let value: Double?
    public let values: [Double]
    public let params: [String: Double]
    public let userAttrs: [String: String]
    public let constraints: [String: Double]

    public var isFeasible: Bool {
        constraints.values.allSatisfy { $0 <= 0.0 }
    }

    public init(
        number: Int,
        state: TrialState,
        value: Double?,
        values: [Double] = [],
        params: [String: Double],
        userAttrs: [String: String] = [:],
        constraints: [String: Double] = [:]
    ) {
        self.number = number
        self.state = state
        self.value = value
        self.values = values.isEmpty ? (value.map { [$0] } ?? []) : values
        self.params = params
        self.userAttrs = userAttrs
        self.constraints = constraints
    }

    public subscript<K: AttributeKey>(_ key: K.Type) -> K.Value? {
        guard let str = userAttrs[K.name] else { return nil }
        return K.Value.fromAttributeString(str)
    }

    public subscript<K: ConstraintKey>(_ key: K.Type) -> Double? {
        constraints[K.name]
    }

    public subscript(_ key: String) -> String? {
        userAttrs[key]
    }

    public func userAttr<T: AttributeConvertible>(as type: T.Type, key: String) -> T? {
        guard let str = userAttrs[key] else { return nil }
        return T.fromAttributeString(str)
    }
}

// MARK: - Sequence Analytics Extensions (Optuna Calibration Parity)

extension Sequence where Element == PersistedTrial {
    public func completed() -> [PersistedTrial] {
        filter { $0.state == .complete && $0.value != nil }
    }

    public func pruned() -> [PersistedTrial] {
        filter { $0.state == .pruned }
    }

    /// Filters trials to only those where all constraint values are <= 0.0.
    public func feasible() -> [PersistedTrial] {
        filter(\.isFeasible)
    }

    /// Filters trials to only those where at least one constraint is violated (> 0.0).
    public func infeasible() -> [PersistedTrial] {
        filter { !$0.isFeasible }
    }

    /// Returns the optimal completed and feasible trial, or nil if no completed trial is feasible.
    public func bestFeasible(direction: Direction = .minimize) -> PersistedTrial? {
        feasible().completed().min { a, b in
            guard let valA = a.value, let valB = b.value else { return false }
            return direction == .minimize ? (valA < valB) : (valA > valB)
        }
    }

    public func sortedByValue(ascending: Bool = true) -> [PersistedTrial] {
        self.filter { $0.value != nil }.sorted {
            ascending ? ($0.value! < $1.value!) : ($0.value! > $1.value!)
        }
    }

    public func top(_ n: Int, ascending: Bool = true) -> [PersistedTrial] {
        Array(sortedByValue(ascending: ascending).prefix(n))
    }

    public func within(tolerance: Double, of baselineValue: Double) -> [PersistedTrial] {
        filter {
            guard let val = $0.value else { return false }
            return val <= baselineValue + tolerance
        }
    }

    public func values(for paramName: String) -> [Double] {
        compactMap { $0.params[paramName] }
    }

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
