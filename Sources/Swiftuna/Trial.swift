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

    internal mutating func takeHandle() -> OpaquePointer? {
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
        let cStrings: [UnsafePointer<CChar>?] = choices.map { choice in
            String(describing: choice).withCString { cStr in
                UnsafePointer(strdup(cStr))
            }
        }
        defer {
            for ptr in cStrings {
                if let ptr {
                    free(UnsafeMutableRawPointer(mutating: ptr))
                }
            }
        }
        let status = name.withCString { cName in
            cStrings.withUnsafeBufferPointer { buf in
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

    public mutating func setUserAttr<K: AttributeKey>(
        _ key: K.Type,
        value: K.Value
    ) throws(SwiftunaError) {
        try setUserAttr(K.name, value: value.toAttributeString())
    }

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

    public subscript<K: AttributeKey>(_ key: K.Type) -> K.Value? {
        mutating get {
            guard let str = localAttrs[K.name] else { return nil }
            return K.Value.fromAttributeString(str)
        }
        set {
            if let newValue {
                try? setUserAttr(K.self, value: newValue)
            } else {
                localAttrs.removeValue(forKey: K.name)
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
        guard !value.isNaN else {
            throw SwiftunaError.invalidArgument("Constraint value for '\(name)' cannot be NaN")
        }
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

    /// Sets a strongly-typed constraint on the trial using a `ConstraintKey`.
    public mutating func setConstraint<K: ConstraintKey>(
        _ key: K.Type,
        value: Double
    ) throws(SwiftunaError) {
        try setConstraint(K.name, value: value)
    }

    /// Sets multiple constraints on the trial in batch.
    ///
    /// A trial is feasible when every constraint value is `<= 0.0`, and infeasible when any constraint is `> 0.0`.
    /// Constraint values of `NaN` are rejected.
    ///
    /// - Parameters:
    ///   - constraints: A dictionary mapping constraint names to their evaluated values.
    /// - Throws: ``SwiftunaError/invalidArgument(_:)`` if any constraint value is NaN.
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

    public subscript<K: ConstraintKey>(_ key: K.Type) -> Double? {
        localConstraints[K.name]
    }

    public subscript(constraint name: String) -> Double? {
        localConstraints[name]
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
    ///
    /// ### Example
    /// ```swift
    /// for epoch in 1...50 {
    ///     let loss = evaluateEpoch(epoch)
    ///     try trial.report(loss, step: epoch, pruneIfWorse: true)
    /// }
    /// ```
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

    /// Evaluates whether the study pruner recommends pruning based on the latest reported value.
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
}
