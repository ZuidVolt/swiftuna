public import Foundation

/// Lifecycle state of an optimization trial.
public enum TrialState: Int32, Sendable, CaseIterable, Hashable, Codable {
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
///     print("Optimizer: \(best.params["optimizer"]?.asString ?? "none")")
///     print("Learning rate: \(best.params["lr"]?.asDouble ?? 0.0)")
/// }
/// ```
public struct PersistedTrial: Sendable, Codable {
    /// Zero-based sequential trial index within the study.
    public let number: Int

    /// Final lifecycle state of the trial.
    public let state: TrialState

    /// Primary objective value for single-objective trials (or first value for multi-objective).
    public let value: Double?

    /// Array of objective values matching the study's ``Study/directions``.
    public let values: [Double]

    /// Dictionary of hyperparameter names and their strongly-typed evaluated values.
    public let params: [String: ParameterValue]

    private let _internalParams: [String: Double]?

    /// Dictionary of hyperparameter names and their raw mathematical internal floats.
    public var internalParams: [String: Double] {
        _internalParams ?? params.compactMapValues { $0.asDouble }
    }

    /// User-defined string metadata attributes attached to the trial.
    public let userAttrs: [String: String]

    /// Mathematical constraint evaluation values ($v \le 0.0$ indicates satisfaction).
    public let constraints: [String: Double]

    /// Intermediate step-by-step values reported via ``Trial/report(_:step:pruneIfWorse:)-(Double,_,_)``.
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

    /// Creates a persisted trial record with strongly-typed parameter values.
    public init(
        number: Int,
        state: TrialState,
        value: Double?,
        values: [Double] = [],
        params: [String: ParameterValue],
        internalParams: [String: Double] = [:],
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
        self._internalParams = internalParams.isEmpty ? nil : internalParams
        self.userAttrs = userAttrs
        self.constraints = constraints
        self.intermediateValues = intermediateValues
        self.datetimeStart = datetimeStart
        self.datetimeComplete = datetimeComplete
    }

    /// Backwards-compatible convenience initializer taking numerical parameters.
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
        self.init(
            number: number,
            state: state,
            value: value,
            values: values,
            params: params.mapValues { .double($0) },
            internalParams: params,
            userAttrs: userAttrs,
            constraints: constraints,
            intermediateValues: intermediateValues,
            datetimeStart: datetimeStart,
            datetimeComplete: datetimeComplete
        )
    }

    // MARK: - Codable Implementation

    private enum CodingKeys: String, CodingKey {
        case number
        case state
        case values
        case params
        case paramValues = "param_values"
        case userAttrs = "user_attrs"
        case constraints
        case intermediateValues = "intermediate_values"
        case datetimeStart = "datetime_start"
        case datetimeComplete = "datetime_complete"
    }

    /// Decodes a persisted trial directly from Optuna or Rustuna serialized JSON representation.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.number = try container.decode(Int.self, forKey: .number)

        let stateInt = try container.decode(Int32.self, forKey: .state)
        self.state = TrialState(rawValue: stateInt) ?? .fail

        let values = try container.decodeIfPresent([Double].self, forKey: .values) ?? []
        self.values = values
        self.value = values.first

        let rawParams = try container.decodeIfPresent([String: Double].self, forKey: .params) ?? [:]
        let paramValues = try container.decodeIfPresent([String: ParameterValue].self, forKey: .paramValues)
        self.params = paramValues ?? rawParams.mapValues { .double($0) }
        self._internalParams = rawParams.isEmpty ? nil : rawParams

        self.userAttrs = try container.decodeIfPresent([String: String].self, forKey: .userAttrs) ?? [:]
        self.constraints = try container.decodeIfPresent([String: Double].self, forKey: .constraints) ?? [:]
        self.intermediateValues = try container.decodeIfPresent([Int: Double].self, forKey: .intermediateValues) ?? [:]

        let dtStartStr = try container.decodeIfPresent(String.self, forKey: .datetimeStart)
        self.datetimeStart = dtStartStr.flatMap { parseOptunaDate($0) }

        let dtCompleteStr = try container.decodeIfPresent(String.self, forKey: .datetimeComplete)
        self.datetimeComplete = dtCompleteStr.flatMap { parseOptunaDate($0) }
    }

    /// Encodes this persisted trial record into standard Optuna JSON representation.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(number, forKey: .number)
        try container.encode(state.rawValue, forKey: .state)
        try container.encode(values, forKey: .values)
        try container.encode(internalParams, forKey: .params)
        try container.encode(params, forKey: .paramValues)
        try container.encode(userAttrs, forKey: .userAttrs)
        try container.encode(constraints, forKey: .constraints)
        try container.encode(intermediateValues, forKey: .intermediateValues)
        let isoFormatter = ISO8601DateFormatter()
        try container.encodeIfPresent(datetimeStart.map { isoFormatter.string(from: $0) }, forKey: .datetimeStart)
        try container.encodeIfPresent(datetimeComplete.map { isoFormatter.string(from: $0) }, forKey: .datetimeComplete)
    }

    // MARK: - Subscripts

    /// Accesses a strongly-typed user attribute value using an ``AttributeKey``.
    public subscript<V>(key: AttributeKey<V>) -> V? {
        guard let str = userAttrs[key.name] else { return nil }
        return V.fromAttributeString(str)
    }

    /// Accesses a strongly-typed user attribute value using an ``AttributeKeyProtocol``.
    public subscript<K: AttributeKeyProtocol>(_ key: K.Type) -> K.Value? {
        guard let str = userAttrs[K.name] else { return nil }
        return K.Value.fromAttributeString(str)
    }

    /// Accesses a constraint evaluation value using a ``ConstraintKey``.
    public subscript(constraint key: ConstraintKey) -> Double? {
        constraints[key.name]
    }

    /// Accesses a constraint evaluation value using a ``ConstraintKeyProtocol``.
    public subscript<K: ConstraintKeyProtocol>(constraint key: K.Type) -> Double? {
        constraints[K.name]
    }

    /// Accesses a constraint evaluation value using a ``ConstraintKeyProtocol`` directly.
    public subscript<K: ConstraintKeyProtocol>(_ key: K.Type) -> Double? {
        constraints[K.name]
    }

    /// Returns the numerical floating-point value (`Double`) for the given parameter, if present.
    ///
    /// ### Example
    /// ```swift
    /// let lr: Double = bestTrial.double("lr") ?? 0.001
    /// ```
    public func double(_ name: String) -> Double? {
        params[name]?.asDouble ?? internalParams[name]
    }

    /// Returns the 32-bit floating-point value (`Float`) for the given parameter, if present.
    ///
    /// ### Example
    /// ```swift
    /// let lr: Float = bestTrial.float("lr") ?? 0.001
    /// ```
    public func float(_ name: String) -> Float? {
        double(name).map(Float.init)
    }

    /// Returns the integer value (`Int`) for the given parameter, if present.
    ///
    /// ### Example
    /// ```swift
    /// let batchSize: Int = bestTrial.int("batch_size") ?? 32
    /// ```
    public func int(_ name: String) -> Int? {
        params[name]?.asInt
    }

    /// Returns the string value for the given parameter, if present.
    ///
    /// ### Example
    /// ```swift
    /// let tag: String = bestTrial.string("tag") ?? "default"
    /// ```
    public func string(_ name: String) -> String? {
        params[name]?.asString
    }

    /// Returns the boolean value for the given parameter, if present.
    public func bool(_ name: String) -> Bool? {
        params[name]?.asBool
    }

    /// Deserializes a categorical parameter directly to a Swift Enum conforming to `RawRepresentable`.
    ///
    /// ### Example
    /// ```swift
    /// let act = bestTrial.param("activation", as: Activation.self)
    /// ```
    public func param<T: RawRepresentable>(_ name: String, as: T.Type) -> T? where T.RawValue == String {
        guard let s = params[name]?.asString else { return nil }
        return T(rawValue: s)
    }

    /// Deserializes a categorical parameter to a Swift Enum conforming to `RawRepresentable` with type inference.
    ///
    /// ### Example
    /// ```swift
    /// let act: Activation? = bestTrial.param("activation")
    /// ```
    public func param<T: RawRepresentable>(_ name: String) -> T? where T.RawValue == String {
        param(name, as: T.self)
    }

    /// Accesses a raw string user attribute by string name.
    public subscript(_ key: String) -> String? {
        userAttrs[key]
    }

    /// Deserializes a user attribute to a specific ``AttributeConvertible`` type by string name.
    public func userAttr<T: AttributeConvertible>(as type: T.Type, key: String) -> T? {
        guard let str = userAttrs[key] else { return nil }
        return T.fromAttributeString(str)
    }
}

// MARK: - Sequence Operations

extension Sequence where Element == PersistedTrial {
    /// Filters trials to only those with ``TrialState/complete`` state and valid objective value(s).
    public func completed() -> [PersistedTrial] {
        filter { $0.state == .complete && $0.value != nil }
    }

    /// Filters trials to only those with ``TrialState/pruned`` state.
    public func pruned() -> [PersistedTrial] {
        filter { $0.state == .pruned }
    }

    /// Filters trials to only those with ``TrialState/fail`` state.
    public func failed() -> [PersistedTrial] {
        filter { $0.state == .fail }
    }

    /// Filters trials to only those where all constraint values are feasible ($v \le 0.0$).
    public func feasible() -> [PersistedTrial] {
        filter(\.isFeasible)
    }

    /// Filters trials to only those where at least one constraint is violated ($v > 0.0$).
    public func infeasible() -> [PersistedTrial] {
        filter { !$0.isFeasible }
    }

    /// Returns the optimal completed trial according to `direction`, or `nil` if none exist.
    ///
    /// - Parameter direction: The optimization direction (defaults to ``Direction/minimize``).
    /// - Returns: The best completed trial in this sequence.
    ///
    /// ### Example
    /// ```swift
    /// let bestAdam = study.trials.completed()
    ///     .filter(param: "optimizer", equals: OptimizerChoice.adam)
    ///     .best()
    /// ```
    public func best(direction: Direction = .minimize) -> PersistedTrial? {
        completed().min { a, b in
            guard let valA = a.value, let valB = b.value else { return false }
            return direction == .minimize ? (valA < valB) : (valA > valB)
        }
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

    /// Filters trials that have a specific strongly-typed user attribute value.
    ///
    /// ### Example
    /// ```swift
    /// let resnetTrials = study.trials.filter(where: .architecture, equals: "ResNet-50")
    /// ```
    public func filter<V: Equatable>(where key: AttributeKey<V>, equals value: V) -> [PersistedTrial] {
        filter { $0[key] == value }
    }

    /// Filters trials that evaluated a categorical hyperparameter equal to the specified Swift enum.
    ///
    /// ### Example
    /// ```swift
    /// let adamTrials = study.trials.filter(param: "optimizer", equals: OptimizerChoice.adamw)
    /// ```
    public func filter<T: RawRepresentable & Equatable>(param name: String, equals value: T) -> [PersistedTrial]
    where T.RawValue == String {
        filter { trial in
            trial.param(name, as: T.self) == value
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
        compactMap { $0.params[paramName]?.asDouble ?? $0.internalParams[paramName] }
    }

    /// Computes the empirical minimum and maximum bounds observed for each parameter across this sequence of trials.
    ///
    /// - Returns: Dictionary mapping parameter names to their observed `ClosedRange<Double>`.
    public func parameterIntervals() -> [String: ClosedRange<Double>] {
        var minValues: [String: Double] = [:]
        var maxValues: [String: Double] = [:]

        for trial in self {
            for (paramName, paramVal) in trial.internalParams {
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

internal func parseOptunaDate(_ str: String) -> Date? {
    if let d = try? Date(str, strategy: .iso8601) {
        return d
    }
    // SQLite format "yyyy-MM-dd HH:mm:ss.SSSSSS" -> normalize to ISO8601 "yyyy-MM-ddTHH:mm:ss.SSSSSSZ"
    var isoStr = str.replacingOccurrences(of: " ", with: "T")
    if !isoStr.hasSuffix("Z"), !isoStr.contains("+") {
        isoStr.append("Z")
    }
    return try? Date(isoStr, strategy: .iso8601)
}
