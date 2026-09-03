import Foundation
internal import LibRustuna

/// An active trial representing a single evaluation step of an objective function.
///
/// `Trial` is passed to the objective closure in ``Study/optimize(nTrials:timeout:objective:)-3gyl5`` or retrieved directly
/// via ``Study/ask()``. It provides interfaces to suggest hyperparameters, report intermediate steps for early stopping,
/// and record user-defined metadata or mathematical constraints.
///
/// > Important:
/// > `Trial` is a non-copyable type (`~Copyable`). It must be completed either by returning from the optimization closure
/// > or by passing it to ``Study/tell(consuming:value:state:)``. Double-use or use-after-consume is prevented at compile time.
///
/// ### Example
/// ```swift
/// study.optimize(nTrials: 50) { trial in
///     let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
///     let layers = try trial.suggest("layers", in: 1...5)
///     return evaluateModel(layers: layers, lr: lr)
/// }
/// ```
public struct Trial: ~Copyable {
    private var raw: OpaquePointer?
    public let number: Int
    internal weak var studyRef: Study?

    public struct IntermediateStep: Sendable {
        public let step: Int
        public let value: Double
    }

    internal var intermediateSteps: ContiguousArray<IntermediateStep> = []

    internal init(raw: OpaquePointer?, study: Study? = nil) {
        self.raw = raw
        self.studyRef = study
        if let raw {
            self.number = Int(rustuna_trial_get_number(raw))
        } else {
            self.number = 0
        }
    }

    deinit {
        if let raw {
            rustuna_trial_free(raw)
        }
    }

    package mutating func takeHandle() -> OpaquePointer? {
        let h = raw
        raw = nil
        return h
    }

    /// Suggests a floating-point parameter value from a continuous or discretized range.
    ///
    /// The value is sampled according to the study's active ``Sampler`` (e.g. ``TPESampler``, ``QMCSampler``).
    ///
    /// - Parameters:
    ///   - name: A parameter name.
    ///   - range: Closed range `[lowerBound, upperBound]` defining the endpoints of suggested values. Both bounds are included.
    ///   - step: A discretization step. If specified, the parameter value is rounded to a multiple of this step.
    ///   - log: If `true`, suggest values from a log scale. Incompatible with `step`.
    /// - Returns: A suggested `Double` value.
    /// - Throws: ``SwiftunaError/handleExpired(_:)`` if the trial handle is invalid,
    ///           or ``SwiftunaError/invalidRange(_:)`` if the range bounds are non-finite or inverted.
    ///
    /// ### Example
    /// ```swift
    /// let lr = try trial.suggest("learning_rate", in: 1e-5...1e-1, log: true)
    /// let weightDecay = try trial.suggest("weight_decay", in: 0.0...0.1, step: 0.01)
    /// ```
    public mutating func suggest(
        _ name: String,
        in range: ClosedRange<Double>,
        step: Double? = nil,
        log: Bool = false
    ) throws(SwiftunaError) -> Double {
        guard let raw else {
            throw SwiftunaError.handleExpired("Trial handle is expired or invalid")
        }
        guard !range.lowerBound.isNaN, !range.upperBound.isNaN,
            range.lowerBound.isFinite, range.upperBound.isFinite,
            range.lowerBound <= range.upperBound
        else {
            throw SwiftunaError.invalidRange("Range bounds must be finite and lowerBound <= upperBound: \(range)")
        }

        var outVal: Double = 0.0
        let stepVal = step ?? 0.0
        let status = name.withCString { cName in
            rustuna_trial_suggest_float(
                raw,
                cName,
                range.lowerBound,
                range.upperBound,
                stepVal,
                log,
                &outVal
            )
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Float suggestion failed for '\(name)'")
        }
        return outVal
    }

    /// Suggests a 32-bit floating-point parameter value (`Float`) from a continuous or discretized range.
    ///
    /// Ideal for direct interop with GPU tensor frameworks like Apple MLX, PyTorch, or Metal Shading Language.
    ///
    /// - Parameters:
    ///   - name: A parameter name.
    ///   - range: Closed range `[lowerBound, upperBound]` defining the endpoints of suggested values.
    ///   - step: A discretization step. When specified, suggested values will be quantized to multiples of `step`.
    ///   - log: If `true`, suggest values from a log scale. Incompatible with `step != nil`.
    /// - Returns: A suggested `Float` value.
    /// - Throws: ``SwiftunaError/handleExpired(_:)`` if the trial handle is invalid,
    ///           or ``SwiftunaError/invalidRange(_:)`` if the range is invalid.
    ///
    /// ### Example
    /// ```swift
    /// let lr: Float = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
    /// let dropout: Float = try trial.suggest("dropout", in: 0.0...0.5, step: 0.05)
    /// ```
    public mutating func suggest(
        _ name: String,
        in range: ClosedRange<Float>,
        step: Float? = nil,
        log: Bool = false
    ) throws(SwiftunaError) -> Float {
        let dVal = try suggest(
            name,
            in: Double(range.lowerBound)...Double(range.upperBound),
            step: step.map(Double.init),
            log: log
        )
        return Float(dVal)
    }

    /// Suggests an integer parameter value from a discrete range.
    ///
    /// - Parameters:
    ///   - name: A parameter name.
    ///   - range: Closed range `[lowerBound, upperBound]` defining the endpoints of suggested values. Both bounds are included.
    ///   - step: A discretization step. Defaults to `1`.
    ///   - log: If `true`, suggest values from a log scale. Incompatible with `step > 1`.
    /// - Returns: A suggested `Int` value.
    /// - Throws: ``SwiftunaError/handleExpired(_:)`` if the trial handle is invalid,
    ///           or ``SwiftunaError/invalidRange(_:)`` if the integer range is empty.
    ///
    /// ### Example
    /// ```swift
    /// let batchSize = try trial.suggest("batch_size", in: 16...128, step: 16)
    /// let epochs = try trial.suggest("epochs", in: 5...50)
    /// ```
    public mutating func suggest(
        _ name: String,
        in range: ClosedRange<Int>,
        step: Int = 1,
        log: Bool = false
    ) throws(SwiftunaError) -> Int {
        guard let raw else {
            throw SwiftunaError.handleExpired("Trial handle is expired or invalid")
        }
        guard !range.isEmpty else {
            throw SwiftunaError.invalidRange("Integer range cannot be empty: \(range)")
        }

        var outVal: Int64 = 0
        let status = name.withCString { cName in
            rustuna_trial_suggest_int(
                raw,
                cName,
                Int64(range.lowerBound),
                Int64(range.upperBound),
                Int64(step),
                log,
                &outVal
            )
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Int suggestion failed for '\(name)'")
        }
        return Int(outVal)
    }

    /// Suggests a categorical parameter value from a collection of candidates.
    ///
    /// - Parameters:
    ///   - name: A parameter name.
    ///   - choices: Parameter value candidates.
    /// - Returns: A suggested value from `choices`.
    /// - Throws: ``SwiftunaError/handleExpired(_:)`` if the trial handle is invalid,
    ///           or ``SwiftunaError/emptyChoices(_:)`` if `choices` is empty.
    ///
    /// ### Example
    /// ```swift
    /// enum Activation: String, CaseIterable, Sendable { case relu, gelu, silu }
    /// let activation = try trial.suggest("activation", choices: Activation.allCases) // Inferred as Activation
    /// let batchSize = try trial.suggest("batch_size", choices: [32, 64, 128])         // Inferred as Int
    /// let optimizer = try trial.suggest("optimizer", choices: ["adamw", "sgd", "adam"])
    /// ```
    public mutating func suggest<T: Equatable>(
        _ name: String,
        choices: [T]
    ) throws(SwiftunaError) -> T {
        guard let raw else {
            throw SwiftunaError.handleExpired("Trial handle is expired or invalid")
        }
        guard !choices.isEmpty else {
            throw SwiftunaError.emptyChoices("Choices cannot be empty for parameter '\(name)'")
        }

        var chosenIdx: Int = 0
        let stringChoices: [String] = choices.map { choice in
            if let r = choice as? (any RawRepresentable), let s = r.rawValue as? String {
                return s
            }
            return String(describing: choice)
        }

        let status = name.withCString { cName in
            withCStrings(stringChoices) { buf in
                rustuna_trial_suggest_categorical(
                    raw,
                    cName,
                    buf.baseAddress,
                    choices.count,
                    &chosenIdx
                )
            }
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(
                fallbackCode: status, context: "Categorical suggestion failed for '\(name)'")
        }
        return choices[chosenIdx]
    }

    // MARK: - User Attributes

    private var localAttrs: [String: String] = [:]

    public mutating func setUserAttr(
        _ key: String,
        value: some AttributeConvertible
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Trial handle is expired or invalid")
        }
        let strVal = value.toAttributeString()
        let status = key.withCString { cKey in
            strVal.withCString { cVal in
                rustuna_trial_set_user_attr(raw, cKey, cVal)
            }
        }
        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to set user attribute '\(key)'")
        }
        localAttrs[key] = strVal
    }

    public subscript<V>(key: AttributeKey<V>) -> V? {
        mutating get {
            guard let str = localAttrs[key.name] else { return nil }
            return V.fromAttributeString(str)
        }
        set {
            if let newValue {
                try? setUserAttr(key.name, value: newValue)
            } else {
                localAttrs.removeValue(forKey: key.name)
            }
        }
    }

    public subscript<K: AttributeKeyProtocol>(_ key: K.Type) -> K.Value? {
        mutating get {
            guard let str = localAttrs[K.name] else { return nil }
            return K.Value.fromAttributeString(str)
        }
        set {
            if let newValue {
                try? setUserAttr(K.name, value: newValue)
            } else {
                localAttrs.removeValue(forKey: K.name)
            }
        }
    }

    public subscript(userAttr key: String) -> String? {
        mutating get {
            localAttrs[key]
        }
        set {
            if let newValue {
                try? setUserAttr(key, value: newValue)
            } else {
                localAttrs.removeValue(forKey: key)
            }
        }
    }

    public subscript(_ key: String) -> String? {
        mutating get {
            localAttrs[key]
        }
        set {
            if let newValue {
                try? setUserAttr(key, value: newValue)
            } else {
                localAttrs.removeValue(forKey: key)
            }
        }
    }

    public var userAttrs: [String: String] {
        localAttrs
    }

    // MARK: - Constraints

    private var localConstraints: [String: Double] = [:]

    /// Sets a single mathematical constraint on the trial.
    ///
    /// - Parameters:
    ///   - name: The unique identifier for the constraint.
    ///   - value: The constraint evaluation. Values `<= 0.0` indicate satisfaction (feasibility);
    ///            values `> 0.0` indicate a violation of magnitude `value`.
    /// - Throws:
    ///   - `SwiftunaError.invalidArgument` if `value` is NaN.
    ///   - `SwiftunaError.attrOverwriteNotAllowed` if this constraint key was already set on the trial.
    public mutating func setConstraint(
        _ name: String,
        value: Double
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Trial handle is expired or invalid")
        }
        try validateConstraint(name: name, value: value)
        guard !localConstraints.keys.contains(name) else {
            throw SwiftunaError.attrOverwriteNotAllowed("Constraint '\(name)' is already set on trial #\(number)")
        }

        let status = name.withCString { cName in
            rustuna_trial_set_constraint(raw, cName, value)
        }
        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to set constraint '\(name)'")
        }
        localConstraints[name] = value
    }

    /// Sets a mathematical constraint using a ``ConstraintKey`` and a floating-point evaluation.
    ///
    /// Values `<= 0.0` indicate constraint satisfaction (feasibility). Values `> 0.0` indicate violation magnitude.
    ///
    /// ### Example
    /// ```swift
    /// extension ConstraintKey { static let maxLatency = ConstraintKey("max_latency_ms") }
    /// try trial.setConstraint(.maxLatency, value: latencyMs - 25.0)
    /// ```
    public mutating func setConstraint(
        _ key: ConstraintKey,
        value: some BinaryFloatingPoint
    ) throws(SwiftunaError) {
        try setConstraint(key.name, value: Double(value))
    }

    /// Sets a mathematical constraint using a ``ConstraintKey`` and an integer evaluation (such as parameter count).
    ///
    /// Values `<= 0` indicate constraint satisfaction (feasibility). Values `> 0` indicate violation magnitude.
    ///
    /// ### Example
    /// ```swift
    /// extension ConstraintKey { static let maxParams = ConstraintKey("max_params_10k") }
    /// try trial.setConstraint(.maxParams, value: model.totalParameters - 10_000)
    /// ```
    public mutating func setConstraint(
        _ key: ConstraintKey,
        value: some BinaryInteger
    ) throws(SwiftunaError) {
        try setConstraint(key.name, value: Double(value))
    }

    public mutating func setConstraint<K: ConstraintKeyProtocol>(
        _ key: K.Type,
        value: Double
    ) throws(SwiftunaError) {
        try setConstraint(K.name, value: value)
    }

    /// Sets multiple constraints on the trial in batch.
    public mutating func setConstraints(
        _ constraints: [String: Double]
    ) throws(SwiftunaError) {
        for (name, value) in constraints {
            try setConstraint(name, value: value)
        }
    }

    /// Returns a copy of constraints set on this active trial.
    public var constraints: [String: Double] {
        localConstraints
    }

    public subscript(constraint key: ConstraintKey) -> Double? {
        mutating get {
            localConstraints[key.name]
        }
        set {
            if let newValue {
                try? setConstraint(key.name, value: newValue)
            } else {
                localConstraints.removeValue(forKey: key.name)
            }
        }
    }

    public subscript<K: ConstraintKeyProtocol>(constraint key: K.Type) -> Double? {
        mutating get {
            self[constraint: ConstraintKey(K.name)]
        }
        set {
            self[constraint: ConstraintKey(K.name)] = newValue
        }
    }

    public subscript<K: ConstraintKeyProtocol>(_ key: K.Type) -> Double? {
        mutating get {
            self[constraint: ConstraintKey(K.name)]
        }
        set {
            self[constraint: ConstraintKey(K.name)] = newValue
        }
    }

    // MARK: - Intermediate Reporting & Pruning

    /// Reports an intermediate objective value for the given step (e.g. epoch number).
    ///
    /// Intermediate values are utilized by pruners (such as ``MedianPruner``, ``SuccessiveHalvingPruner``,
    /// or ``HyperbandPruner``) to terminate unpromising trials early.
    ///
    /// - Parameters:
    ///   - value: The intermediate evaluation value (e.g. loss, validation error, or reward).
    ///   - step: The step index (e.g. epoch number or iteration).
    ///   - pruneIfWorse: If `true`, immediately evaluates the study pruner and throws ``SwiftunaError/trialPruned(reason:)``
    ///                   if early stopping is recommended.
    /// - Throws: ``SwiftunaError/trialPruned(reason:)`` when `pruneIfWorse` is `true` and pruning is triggered.
    public mutating func report(
        _ value: Double,
        step: Int,
        pruneIfWorse: Bool = false
    ) throws(SwiftunaError) {
        intermediateSteps.append(IntermediateStep(step: step, value: value))

        if pruneIfWorse {
            if try shouldPrune {
                throw SwiftunaError.trialPruned(reason: "Pruned at step \(step) with value \(value)")
            }
        }
    }

    /// Reports an intermediate 32-bit floating point objective value (`Float`) for the given step.
    public mutating func report(
        _ value: Float,
        step: Int,
        pruneIfWorse: Bool = false
    ) throws(SwiftunaError) {
        try report(Double(value), step: step, pruneIfWorse: pruneIfWorse)
    }

    /// Evaluates whether the study pruner recommends early stopping based on the latest reported value.
    ///
    /// Useful for evaluating pruning explicitly without throwing immediately, enabling custom cleanup, checkpointing, or logging.
    ///
    /// ### Example
    /// ```swift
    /// try trial.report(valLoss, step: epoch)
    /// if try trial.shouldPrune {
    ///     saveCheckpoint()
    ///     try trial.prune()
    /// }
    /// ```
    public var shouldPrune: Bool {
        get throws(SwiftunaError) {
            guard let study = studyRef else { return false }
            guard let last = intermediateSteps.last else { return false }
            return try study.pruner.shouldPrune(
                study: study,
                trialNumber: number,
                step: last.step,
                currentValue: last.value
            )
        }
    }

    /// Manually prunes the trial, throwing ``SwiftunaError/trialPruned(reason:)``.
    ///
    /// ### Example
    /// ```swift
    /// if try trial.shouldPrune {
    ///     try trial.prune(reason: "Validation loss diverged at epoch \(epoch)")
    /// }
    /// ```
    public func prune(reason: String? = nil) throws(SwiftunaError) -> Never {
        let step = intermediateSteps.last?.step ?? 0
        let val = intermediateSteps.last?.value ?? 0.0
        throw SwiftunaError.trialPruned(reason: reason ?? "Pruned at step \(step) with value \(val)")
    }

    /// Suggests the parameter described by a declarative ``SearchParam``.
    ///
    /// This is the shared per-type sampling path: `Trial` closures,
    /// `SearchSpaceParams`, and the distributed coordinator all funnel through
    /// this method, so a distribution behaves identically everywhere.
    ///
    /// - Returns: The parameter name and its sampled ``ParameterValue``.
    public mutating func suggest(_ param: SearchParam) throws(SwiftunaError) -> (String, ParameterValue) {
        switch param {
        case .float(let name, let lower, let upper, let log, let step):
            let value = try suggest(name, in: lower...upper, step: step, log: log)
            return (name, .double(value))
        case .int(let name, let lower, let upper, let step, let log):
            let value = try suggest(name, in: lower...upper, step: step, log: log)
            return (name, .int(value))
        case .categorical(let name, let choices):
            let value = try suggest(name, choices: choices)
            return (name, value)
        }
    }
}

/// One declarative parameter in a search space.
///
/// Unlike an `AskFunction` closure, `SearchParam` values are plain data:
/// `Codable` for logging and transmission, `Equatable` for testing.
/// See `SearchSpaceParams` (SwiftunaDistributed) for the ordered collection.
public enum SearchParam: Sendable, Codable, Equatable {
    /// Floating-point range, mirroring `Trial.suggest(_:in:step:log:)`.
    case float(name: String, lower: Double, upper: Double, log: Bool = false, step: Double? = nil)
    /// Integer range, mirroring `Trial.suggest(_:in:step:log:)`.
    case int(name: String, lower: Int, upper: Int, step: Int = 1, log: Bool = false)
    /// Categorical choices, mirroring `Trial.suggest(_:choices:)`.
    case categorical(name: String, choices: [ParameterValue])

    /// The parameter name suggested by this descriptor.
    public var name: String {
        switch self {
        case .float(let name, _, _, _, _): return name
        case .int(let name, _, _, _, _): return name
        case .categorical(let name, _): return name
        }
    }
}

/// Validates a constraint value the same way for local and distributed trials.
///
/// - Throws: `SwiftunaError.invalidArgument` if `value` is NaN.
package func validateConstraint(name: String, value: Double) throws(SwiftunaError) {
    guard !value.isNaN else {
        throw SwiftunaError.invalidArgument("Constraint value for '\(name)' cannot be NaN")
    }
}

private func withCStrings<R>(
    _ strings: [String],
    _ body: (UnsafeBufferPointer<UnsafePointer<CChar>?>) throws -> R
) rethrows -> R {
    func recurse(index: Int, ptrs: inout [UnsafePointer<CChar>?]) throws -> R {
        if index == strings.count {
            return try ptrs.withUnsafeBufferPointer(body)
        }
        return try strings[index].withCString { cStr in
            ptrs.append(cStr)
            defer { ptrs.removeLast() }
            return try recurse(index: index + 1, ptrs: &ptrs)
        }
    }
    var ptrs: [UnsafePointer<CChar>?] = []
    ptrs.reserveCapacity(strings.count)
    return try recurse(index: 0, ptrs: &ptrs)
}
