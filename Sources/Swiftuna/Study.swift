internal import Foundation
internal import LibRustuna

/// Manages trial limits and monotonic execution deadlines for optimization loops.
internal struct OptimizationBudget: Sendable {
    let nTrials: Int?
    let deadline: ContinuousClock.Instant?
    private let clock = ContinuousClock()

    init(nTrials: Int?, timeout: Duration?) throws(SwiftunaError) {
        guard nTrials != nil || timeout != nil else {
            throw SwiftunaError.invalidArgument("Either nTrials or timeout must be specified in optimize")
        }
        self.nTrials = nTrials
        let c = ContinuousClock()
        self.deadline = timeout.map { c.now + $0 }
    }

    var isExhausted: Bool {
        if let deadline, clock.now >= deadline {
            return true
        }
        return false
    }

    func shouldStop(submitted: Int) -> Bool {
        if let nTrials, submitted >= nTrials {
            return true
        }
        return isExhausted
    }
}

/// An optimization study coordinating trials, samplers, pruners, and storage.
///
/// A study corresponds to an optimization task over a search space. It maintains trial history,
/// samples parameter candidates using an active ``Sampler``, prunes unpromising trials via a ``Pruner``,
/// and persists results across in-memory or SQLite storage backends.
///
/// Studies are typically created using ``Swiftuna/createStudy(name:direction:storage:sampler:pruner:loadIfExists:)``.
///
/// ### Example
/// ```swift
/// let study = try Swiftuna.createStudy(name: "hyperparameter_search", direction: .minimize)
/// try study.optimize(nTrials: 100) { trial in
///     let x = try trial.suggest("x", in: -5.0...5.0)
///     return x * x
/// }
/// print("Best value: \(try study.bestValue)")
/// ```
public final class Study: @unchecked Sendable {
    private var raw: OpaquePointer?
    package var rawHandle: OpaquePointer? { raw }
    public let name: String
    public let directions: [Direction]
    public let pruner: any Pruner
    public let storage: StorageBackend

    public var direction: Direction {
        directions.first ?? .minimize
    }

    internal init(
        raw: OpaquePointer?,
        name: String,
        directions: [Direction],
        pruner: any Pruner = NopPruner(),
        storage: StorageBackend = .inMemory
    ) {
        self.raw = raw
        self.name = name
        self.directions = directions
        self.pruner = pruner
        self.storage = storage
    }

    internal convenience init(
        raw: OpaquePointer?,
        name: String,
        direction: Direction,
        pruner: any Pruner = NopPruner(),
        storage: StorageBackend = .inMemory
    ) {
        self.init(raw: raw, name: name, directions: [direction], pruner: pruner, storage: storage)
    }

    /// Synchronizes and formats the underlying SQLite database for full compatibility
    /// with `optuna-dashboard` and Python Optuna.
    public func syncWithOptunaDashboard() {
        guard case .sqlite(let path) = storage else { return }
        // Formatting the database can take milliseconds: worth one span when
        // instrumented, invisible otherwise.
        let span = SwiftunaTelemetry.shared.span(
            name: "swiftuna.dashboard_sync",
            attributes: ["study.name": .string(name)]
        )
        StorageBackend.syncWithOptunaDashboard(at: path)
        span?.end(status: .ok)
    }

    deinit {
        syncWithOptunaDashboard()
        if let raw {
            rustuna_study_free(raw)
        }
    }

    /// Creates a new trial from which hyperparameters can be suggested.
    ///
    /// This method is part of the manual ask-and-tell interface, serving as an alternative to ``optimize(nTrials:timeout:objective:)-3gyl5``
    /// when you need fine-grained control over execution flow, custom distributed runners, or batch evaluations.
    ///
    /// - Returns: An active, non-copyable ``Trial`` object.
    /// - Throws: ``SwiftunaError/handleExpired(_:)`` if the study handle is invalid,
    ///           or ``SwiftunaError/searchSpaceExhausted(_:)`` if a discrete sampler (e.g. ``GridSampler``) has exhausted all parameter combinations.
    ///
    /// ### Example
    /// ```swift
    /// var trial = try study.ask()
    /// let x = try trial.suggest("x", in: -10.0...10.0)
    /// let loss = evaluate(x)
    /// try study.tell(consuming: trial, value: loss)
    /// ```
    public func ask() throws(SwiftunaError) -> Trial {
        try checkNotInCallback()
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        var trialPtr: OpaquePointer?
        let status = rustuna_study_ask(raw, &trialPtr)

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to ask for next trial")
        }
        return Trial(raw: trialPtr, study: self)
    }

    /// Serializes enqueue-then-ask pairs across drivers sharing this study.
    ///
    /// Enqueue fixes the *next* ask, so the pair must not interleave with
    /// other drivers' pairs. Plain `NSLock`: `Synchronization.Mutex.withLock`
    /// erases throws to `any Error`, which does not compose with this file's
    /// typed `throws(SwiftunaError)` — the lock/unlock pair keeps error
    /// types exact. The critical section holds two FFI calls and never the
    /// evaluation.
    private let askSlot = NSLock()

    /// Checks out the next trial with `params` fixed, atomically.
    ///
    /// Enqueue-then-ask under ``askSlot``: concurrent drivers on one study
    /// can never receive each other's configurations. Partial dicts allowed —
    /// omitted params fall back to the study's ``Sampler``, mirroring
    /// Optuna's relative/independent split.
    ///
    /// Refuses checkout from inside a sampler callback (``SwiftunaError/reentrantAsk(_:)``):
    /// a trial checked out mid-suggestion would leak unfinished and scramble
    /// queue pairing. See `samplerCallbackDepth` in `CallbackSampler.swift`.
    /// Refuses trial checkout from inside a sampler callback.
    ///
    /// A trial checked out mid-suggestion would leak unfinished and scramble
    /// queue pairing. One check shared by ``ask()`` and ``askEnqueued(_:)``.
    private func checkNotInCallback() throws(SwiftunaError) {
        if Thread.current.threadDictionary[samplerCallbackDepthKey] != nil {
            throw SwiftunaError.reentrantAsk(
                "Cannot check out a trial from inside a sampler callback: the trial would leak unfinished and queue pairing would scramble.")
        }
    }

    func askEnqueued(_ params: [String: ParameterValue]) throws(SwiftunaError) -> Trial {
        try checkNotInCallback()
        askSlot.lock()
        defer { askSlot.unlock() }
        try enqueue(params)
        return try ask()
    }

    /// Finishes a trial created with ``ask()``, recording its objective value and final state.
    ///
    /// - Parameters:
    ///   - trial: The active ``Trial`` to finish. This parameter is consumed to prevent reuse.
    ///   - value: The final evaluated objective value.
    ///   - state: The final state to assign to the trial (defaults to ``TrialState/complete``).
    /// - Throws: ``SwiftunaError`` if recording the trial results to storage fails.
    public func tell(
        consuming trial: consuming Trial,
        value: Double,
        state: TrialState = .complete
    ) throws(SwiftunaError) {
        try tell(consuming: trial, values: [value], state: state)
    }

    /// Finishes a trial created with ``ask()`` without objective values (e.g. for failed or pruned trials).
    ///
    /// - Parameters:
    ///   - trial: The active ``Trial`` to finish. This parameter is consumed to prevent reuse.
    ///   - state: The final state to assign to the trial (e.g. ``TrialState/pruned`` or ``TrialState/fail``).
    /// - Throws: ``SwiftunaError`` if recording the trial results to storage fails.
    public func tell(
        consuming trial: consuming Trial,
        state: TrialState
    ) throws(SwiftunaError) {
        try tell(consuming: trial, values: [], state: state)
    }

    /// Finishes a multi-objective trial created with ``ask()``, recording multiple objective values.
    ///
    /// - Parameters:
    ///   - trial: The active ``Trial`` to finish. This parameter is consumed to prevent reuse.
    ///   - values: The sequence of evaluated objective values matching the study's ``directions``.
    ///   - state: The final state to assign to the trial (defaults to ``TrialState/complete``).
    /// - Throws: ``SwiftunaError`` if recording the trial results to storage fails.
    public func tell(
        consuming trial: consuming Trial,
        values: [Double],
        state: TrialState = .complete
    ) throws(SwiftunaError) {
        let trialNumber = trial.number
        let intermediateSteps = Dictionary(
            uniqueKeysWithValues: trial.intermediateSteps.map { ($0.step, $0.value) }
        )
        _ = trial.takeHandle()  // Release from trial deinit

        try tellRecorded(
            trialNumber: trialNumber,
            state: state,
            values: values,
            intermediateSteps: intermediateSteps
        )
    }

    /// Records a detached trial outcome in storage.
    ///
    /// Shared by ``tell(consuming:values:state:)`` and `SwiftunaDistributed`'s
    /// coordinator so both paths serialize intermediates and call FFI identically.
    /// The caller owns handle lifecycle: the trial handle must already be
    /// detached (or never read again) before calling.
    package func tellRecorded(
        trialNumber: Int,
        state: TrialState,
        values: [Double],
        intermediateSteps: [Int: Double]
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        var intermediateJsonStr: String?
        if !intermediateSteps.isEmpty {
            let stepMap = Dictionary(uniqueKeysWithValues: intermediateSteps.map { (String($0.key), $0.value) })
            if let data = try? JSONEncoder().encode(stepMap) {
                intermediateJsonStr = String(data: data, encoding: .utf8)
            }
        }

        let status: Int32 = withOptionalCString(intermediateJsonStr) { cIntermediate in
            if values.isEmpty {
                return rustuna_study_tell_multi(
                    raw,
                    UInt32(trialNumber),
                    state.rawValue,
                    nil,
                    0,
                    cIntermediate
                )
            }
            return values.withUnsafeBufferPointer { buf in
                rustuna_study_tell_multi(
                    raw,
                    UInt32(trialNumber),
                    state.rawValue,
                    buf.baseAddress,
                    values.count,
                    cIntermediate
                )
            }
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(
                fallbackCode: status, context: "Failed to tell multi-objective trial #\(trialNumber)")
        }
    }

    /// Optimizes a multi-objective function until `nTrials` or `timeout` is reached.
    ///
    /// - Parameters:
    ///   - nTrials: The maximum number of trials to run. At least one of `nTrials` or `timeout` must be specified.
    ///   - timeout: The maximum duration to allow optimization to proceed. Trials already running will complete gracefully.
    ///   - objective: A closure taking an `inout Trial` and returning an array of objective values matching the study's ``directions``.
    /// - Throws: ``SwiftunaError`` if optimization encounters an error or an invalid argument is provided.
    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        objective: (inout Trial) throws(SwiftunaError) -> [Double]
    ) throws(SwiftunaError) {
        defer {
            syncWithOptunaDashboard()
        }
        let budget = try OptimizationBudget(nTrials: nTrials, timeout: timeout)
        let clock = ContinuousClock()
        var iteration = 0

        while !budget.shouldStop(submitted: iteration) {
            let trial: Trial
            do {
                trial = try ask()
            } catch SwiftunaError.searchSpaceExhausted {
                break
            }

            iteration += 1
            var activeTrial = trial
            let trialNum = activeTrial.number
            let startTime = clock.now
            // One call guards every allocation below: the tracer lock, the
            // attribute dict, and the trial span all stay dead on the
            // disabled path.
            let span = SwiftunaTelemetry.shared.trialSpan(
                study: name, trialNumber: trialNum, distributed: false)
            // Threads the ambient span into the trial so `report` heartbeats
            // and pruner votes land on it as events.
            activeTrial.telemetrySpan = span

            let evalResult: Result<[Double], SwiftunaError>
            do {
                let vals = try objective(&activeTrial)
                evalResult = .success(vals)
            } catch {
                evalResult = .failure(error)
            }

            let elapsed = clock.now - startTime
            // Sampled hyperparameters on every terminal path: pruned and
            // failed trials map the search space just as much as completed
            // ones. Behind the same `span?` chain, so nothing new runs
            // when uninstrumented.
            for (paramName, paramValue) in activeTrial.suggestedParams {
                span?.setAttribute("param.\(paramName)", value: paramValue.telemetryAttribute)
            }
            switch evalResult {
            case .success(let vals):
                span?.setAttribute("trial.status", value: "complete")
                // Raw double, never formatted: String(format:) follows the
                // device locale and must not touch telemetry.
                span?.setAttribute(
                    "trial.duration_ms",
                    value: .double(Double(elapsed.components.attoseconds) / 1e15))
                span?.end(status: .ok)
                try tell(consuming: activeTrial, values: vals, state: .complete)
            case .failure(let err):
                switch err {
                case .trialPruned:
                    span?.setAttribute("trial.status", value: "pruned")
                    span?.end(status: .ok)
                    try tell(consuming: activeTrial, values: [], state: .pruned)
                default:
                    span?.setAttribute("trial.status", value: "failed")
                    span?.end(status: .error(err.description))
                    try tell(consuming: activeTrial, values: [], state: .fail)
                    throw err
                }
            }
        }
    }

    /// Optimizes a single-objective function until `nTrials` or `timeout` is reached.
    ///
    /// - Parameters:
    ///   - nTrials: The maximum number of trials to run. At least one of `nTrials` or `timeout` must be specified.
    ///   - timeout: The maximum duration to allow optimization to proceed. Trials already running will complete gracefully.
    ///   - objective: A closure taking an `inout Trial` and returning a single evaluated `Double`.
    /// - Throws: ``SwiftunaError`` if optimization fails.
    ///
    /// ### Example
    /// ```swift
    /// try study.optimize(nTrials: 100) { trial in
    ///     let x = try trial.suggest("x", in: -10.0...10.0)
    ///     return x * x
    /// }
    /// ```
    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        objective: (inout Trial) throws(SwiftunaError) -> Double
    ) throws(SwiftunaError) {
        try optimize(nTrials: nTrials, timeout: timeout) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            [try objective(&trial)]
        }
    }

    /// Concurrently optimizes an objective function using structured Swift concurrency (`TaskGroup`).
    ///
    /// - Parameters:
    ///   - nTrials: The maximum number of trials to run. At least one of `nTrials` or `timeout` must be specified.
    ///   - timeout: The maximum duration to allow optimization to proceed.
    ///   - concurrency: Maximum number of concurrent tasks. Defaults to `ProcessInfo.processInfo.activeProcessorCount`.
    ///   - objective: An async closure taking an `inout Trial` and returning a single evaluated `Double`.
    /// - Throws: Any error thrown during concurrent evaluation.
    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        concurrency: Int? = nil,
        objective: @Sendable @escaping (inout Trial) async throws -> Double
    ) async throws {
        let budget = try OptimizationBudget(nTrials: nTrials, timeout: timeout)
        let maxConcurrent = max(1, concurrency ?? ProcessInfo.processInfo.activeProcessorCount)
        let initialLimit = nTrials != nil ? min(maxConcurrent, nTrials!) : maxConcurrent

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                var submitted = 0

                func spawnNextWorker() {
                    if budget.shouldStop(submitted: submitted) { return }

                    submitted += 1
                    group.addTask {
                        let trial: Trial
                        do {
                            trial = try self.ask()
                        } catch SwiftunaError.searchSpaceExhausted {
                            return
                        }
                        var activeTrial = trial
                        let evalResult: Result<Double, any Error>
                        do {
                            let val = try await objective(&activeTrial)
                            evalResult = .success(val)
                        } catch {
                            evalResult = .failure(error)
                        }

                        switch evalResult {
                        case .success(let val):
                            try self.tell(consuming: activeTrial, value: val, state: .complete)
                        case .failure(let err):
                            if case SwiftunaError.trialPruned = err {
                                try self.tell(consuming: activeTrial, values: [], state: .pruned)
                            } else {
                                try self.tell(consuming: activeTrial, values: [], state: .fail)
                                throw err
                            }
                        }
                    }
                }

                for _ in 0..<initialLimit {
                    spawnNextWorker()
                }

                while (try await group.next()) != nil {
                    spawnNextWorker()
                }
            }
        } catch let err as SwiftunaError {
            throw err
        } catch {
            throw SwiftunaError.unexpected("Concurrent optimization failed: \(error)")
        }
    }

    public func optimize<E: Error>(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        objective: (inout Trial) throws(E) -> Double
    ) throws {
        var caughtError: E?
        try optimize(nTrials: nTrials, timeout: timeout) { (trial: inout Trial) throws(SwiftunaError) -> Double in
            do {
                return try objective(&trial)
            } catch let err as SwiftunaError {
                throw err
            } catch let err as E {
                caughtError = err
                throw SwiftunaError.objectiveError("\(err)")
            } catch {
                throw SwiftunaError.objectiveError("\(error)")
            }
        }
        if let err = caughtError {
            throw err
        }
    }

    /// The best completed trial in a single-objective study, or `nil` if no completed trials exist.
    ///
    /// - Throws: ``SwiftunaError/unsupportedMultiObjective`` if called on a multi-objective study (use ``bestTrials`` instead).
    public var bestTrial: PersistedTrial? {
        get throws(SwiftunaError) {
            guard directions.count <= 1 else {
                throw SwiftunaError.unsupportedMultiObjective
            }
            guard let raw else {
                throw SwiftunaError.handleExpired("Study handle is invalid")
            }

            var ptPtr: OpaquePointer?
            let status = rustuna_study_get_best_trial(raw, &ptPtr)

            if status != 0 {
                return nil
            }
            guard let pt = ptPtr else { return nil }
            defer { rustuna_persisted_trial_free(pt) }

            return parsePersistedTrial(pt)
        }
    }

    /// Returns the best completed and feasible trial (where all constraints <= 0.0), or `nil` if no completed trial is feasible.
    ///
    /// - Throws: ``SwiftunaError/unsupportedMultiObjective`` if called on a multi-objective study (use ``bestTrials`` instead).
    public var bestFeasibleTrial: PersistedTrial? {
        get throws(SwiftunaError) {
            guard directions.count <= 1 else {
                throw SwiftunaError.unsupportedMultiObjective
            }
            return try trials.bestFeasible(direction: direction)
        }
    }

    /// The Pareto frontier of non-dominated trials in a multi-objective study.
    ///
    /// - Throws: ``SwiftunaError`` if querying trials from storage fails.
    public var bestTrials: [PersistedTrial] {
        get throws(SwiftunaError) {
            guard let raw else {
                throw SwiftunaError.handleExpired("Study handle is invalid")
            }

            var trialsPtr: UnsafeMutablePointer<OpaquePointer?>?
            var count: Int = 0
            let status = rustuna_study_get_best_trials(raw, &trialsPtr, &count)

            if status != 0 {
                throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to get Pareto best trials")
            }

            return parseTrialsBuffer(trialsPtr, count: count)
        }
    }

    /// The best objective value achieved among completed trials in a single-objective study.
    ///
    /// - Throws: ``SwiftunaError/noTrialsFound`` if no completed trials are found in the study,
    ///           or ``SwiftunaError/unsupportedMultiObjective`` if called on a multi-objective study.
    public var bestValue: Double {
        get throws(SwiftunaError) {
            guard let bt = try bestTrial, let val = bt.value else {
                throw SwiftunaError.noTrialsFound
            }
            return val
        }
    }

    /// The suggested parameter dictionary of the best completed trial in a single-objective study.
    ///
    /// Returns a dictionary of hyperparameter names and their evaluated values for the best trial.
    ///
    /// - Throws: ``SwiftunaError/noTrialsFound`` if no completed trials are found in the study,
    ///           or ``SwiftunaError/unsupportedMultiObjective`` if called on a multi-objective study.
    public var bestParams: [String: ParameterValue] {
        get throws(SwiftunaError) {
            guard let bt = try bestTrial else {
                throw SwiftunaError.noTrialsFound
            }
            return bt.params
        }
    }

    /// Returns a dictionary of hyperparameter names and their internal mathematical float values.
    public var bestInternalParams: [String: Double] {
        get throws(SwiftunaError) {
            guard let bt = try bestTrial else {
                throw SwiftunaError.noTrialsFound
            }
            return bt.internalParams
        }
    }

    /// Returns all trials recorded in the study regardless of state.
    ///
    /// - Throws: ``SwiftunaError`` if reading from storage fails.
    public var trials: [PersistedTrial] {
        get throws(SwiftunaError) {
            try trials(where: Set(TrialState.allCases))
        }
    }

    /// Fetches trials filtered by their `TrialState` at the storage layer.
    ///
    /// This avoids allocating discarded trial objects across the C ABI.
    /// Pass `since` with the known trial count for incremental refresh: only
    /// the unseen tail is converted and transferred, so refresh costs O(new)
    /// instead of O(history). Trial numbers are dense from 0.
    public func trials(
        where states: Set<TrialState>,
        since trialNumber: Int = 0
    ) throws(SwiftunaError) -> [PersistedTrial] {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is invalid")
        }
        guard !states.isEmpty else { return [] }

        let mask = states.reduce(UInt32(0)) { $0 | (1 << $1.rawValue) }
        var cJson: UnsafeMutablePointer<CChar>?
        var jsonLen: Int = 0
        let status = rustuna_study_get_trials_json_since(
            raw, mask, UInt32(max(0, trialNumber)), &cJson, &jsonLen)

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to get filtered trials")
        }
        guard let cJson, jsonLen > 0 else { return [] }
        defer { rustuna_string_free(cJson) }

        let data = Data(bytesNoCopy: cJson, count: jsonLen, deallocator: .none)
        return (try? Self.trialDecoder.decode([PersistedTrial].self, from: data)) ?? []
    }

    /// Fetches trials filtered by variadic `TrialState`s.
    public func trials(where states: TrialState...) throws(SwiftunaError) -> [PersistedTrial] {
        try trials(where: Set(states))
    }

    /// Copies this study to a destination storage backend, replicating all trials, directions, and attributes.
    ///
    /// - Parameters:
    ///   - destination: The target storage backend to copy to.
    ///   - newName: Optional target study name. If nil, uses the source study name.
    /// - Returns: An active `Study` connected to the destination storage backend.
    /// - Throws:
    ///   - `SwiftunaError.duplicatedStudy` if a study with the target name already exists in the destination.
    @discardableResult
    public func copy(
        to destination: StorageBackend,
        as newName: String? = nil
    ) throws(SwiftunaError) -> Study {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or closed")
        }

        var outStudy: OpaquePointer?
        let targetName = newName ?? name

        let status = targetName.withCString { cName in
            withOptionalCString(destination.pathString) { cPath in
                rustuna_study_copy(raw, destination.rawStorageType, cPath, cName, &outStudy)
            }
        }

        guard status == 0, let outStudy else {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to copy study '\(name)'")
        }

        return Study(
            raw: outStudy,
            name: targetName,
            directions: directions,
            storage: destination
        )
    }

    private func parseTrialsBuffer(
        _ rawBuffer: UnsafeMutablePointer<OpaquePointer?>?,
        count: Int
    ) -> [PersistedTrial] {
        guard let rawBuffer, count > 0 else { return [] }
        defer { rustuna_trials_buffer_free(rawBuffer, count) }

        var results: [PersistedTrial] = []
        results.reserveCapacity(count)

        for i in 0..<count {
            if let trialPtr = rawBuffer[i] {
                results.append(parsePersistedTrial(trialPtr))
            }
        }
        return results
    }

    private static let trialDecoder = JSONDecoder()

    private func parsePersistedTrial(_ trialPtr: OpaquePointer) -> PersistedTrial {
        var cJson: UnsafeMutablePointer<CChar>?
        guard rustuna_persisted_trial_get_json(trialPtr, &cJson) == 0, let cJson else {
            return PersistedTrial(
                number: 0, state: .fail, value: nil, values: [],
                params: [:], internalParams: [:], userAttrs: [:]
            )
        }
        defer { rustuna_string_free(cJson) }

        let len = strlen(cJson)
        let data = Data(bytesNoCopy: cJson, count: len, deallocator: .none)
        return (try? Self.trialDecoder.decode(PersistedTrial.self, from: data))
            ?? PersistedTrial(
                number: 0, state: .fail, value: nil, values: [],
                params: [:], internalParams: [:], userAttrs: [:]
            )
    }

    // MARK: - Study User Attributes & Analytics

    public func setUserAttr<V>(
        _ key: AttributeKey<V>,
        value: V
    ) throws(SwiftunaError) {
        try setUserAttr(key.name, value: value.toAttributeString())
    }

    public func setUserAttr<K: AttributeKeyProtocol>(
        _ key: K.Type,
        value: K.Value
    ) throws(SwiftunaError) {
        try setUserAttr(K.name, value: value.toAttributeString())
    }

    public func setUserAttr(
        _ key: String,
        value: some AttributeConvertible
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }
        let strVal = value.toAttributeString()
        let status = key.withCString { cKey in
            strVal.withCString { cVal in
                rustuna_study_set_user_attr(raw, cKey, cVal)
            }
        }
        if status != 0 {
            throw SwiftunaError.fromLastError(
                fallbackCode: status, context: "Failed to set study user attribute '\(key)'")
        }
    }

    public func userAttr<V>(_ key: AttributeKey<V>) throws(SwiftunaError) -> V? {
        guard let str = try userAttr(key.name) else { return nil }
        return V.fromAttributeString(str)
    }

    public func userAttr<K: AttributeKeyProtocol>(_ key: K.Type) throws(SwiftunaError) -> K.Value? {
        guard let str = try userAttr(K.name) else { return nil }
        return K.Value.fromAttributeString(str)
    }

    public func userAttr(_ key: String) throws(SwiftunaError) -> String? {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }
        var outPtr: UnsafeMutablePointer<CChar>?
        let status = key.withCString { cKey in
            rustuna_study_get_user_attr(raw, cKey, &outPtr)
        }
        if status != 0 {
            throw SwiftunaError.fromLastError(
                fallbackCode: status, context: "Failed to get study user attribute '\(key)'")
        }
        guard let outPtr else { return nil }
        defer { rustuna_string_free(outPtr) }
        return String(cString: outPtr)
    }

    public subscript<V>(key: AttributeKey<V>) -> V? {
        get {
            try? userAttr(key)
        }
        set {
            if let newValue {
                try? setUserAttr(key, value: newValue)
            }
        }
    }

    public subscript<K: AttributeKeyProtocol>(_ key: K.Type) -> K.Value? {
        get {
            guard let str = try? userAttr(K.name) else { return nil }
            return K.Value.fromAttributeString(str)
        }
        set {
            if let newValue {
                try? setUserAttr(K.name, value: newValue.toAttributeString())
            }
        }
    }

    public subscript(userAttr key: String) -> String? {
        get {
            try? userAttr(key)
        }
        set {
            if let newValue {
                try? setUserAttr(key, value: newValue)
            }
        }
    }

    public subscript(_ key: String) -> String? {
        get {
            try? userAttr(key)
        }
        set {
            if let newValue {
                try? setUserAttr(key, value: newValue)
            }
        }
    }

    public func parameterIntervals(
        tolerance: Double = 0.01
    ) throws(SwiftunaError) -> [String: ClosedRange<Double>] {
        let best = try bestValue
        return try trials.completed().within(tolerance: tolerance, of: best).parameterIntervals()
    }

    // MARK: - Trial Enqueueing & Hyperparameter Importance

    /// Enqueues a trial configuration with specified parameter values and optional user attributes.
    ///
    /// Pre-queued configurations will be evaluated by the study in FIFO order prior to stochastic sampling.
    /// Any parameter omitted from `params` will be sampled dynamically by the study's ``Sampler``.
    ///
    /// Heterogeneous literals and variables just work: when every value
    /// converts exactly to ``ParameterValue`` (primitives and `ParameterValue`
    /// itself), the typed encoding is used automatically with no JSON round
    /// trip. Only exotic `AttributeConvertible` types fall back to the JSON
    /// form (~2x the cost, measured in release).
    ///
    /// - Parameters:
    ///   - params: Parameter names and fixed values to evaluate.
    ///   - userAttrs: Optional user-defined metadata associated with the enqueued trial.
    /// - Returns: `self` to support fluent call chaining.
    /// - Throws: ``SwiftunaError`` if parameter serialization fails or study handle is invalid.
    ///
    /// ### Example
    /// ```swift
    /// try study.enqueue(["learning_rate": 0.01, "batch_size": 32])
    /// ```
    @discardableResult
    public func enqueue(
        _ params: [String: any AttributeConvertible],
        userAttrs: [String: String] = [:]
    ) throws(SwiftunaError) -> Self {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        // Fast path: every value converts exactly, so skip JSON entirely.
        // One conversion pass (~0.3µs) to save ~1µs of serialization.
        var typed: [String: ParameterValue] = [:]
        typed.reserveCapacity(params.count)
        var allExact = true
        for (key, value) in params {
            guard let pv = ParameterValue(exactly: value) else {
                allExact = false
                break
            }
            typed[key] = pv
        }
        if allExact {
            return try enqueueTyped(params: typed, raw: raw, userAttrs: userAttrs)
        }

        var jsonDict: [String: Any] = [:]
        for (k, v) in params {
            let str = v.toAttributeString()
            if let i = Int(str) {
                jsonDict[k] = i
            } else if let d = Double(str) {
                jsonDict[k] = d
            } else if str == "true" {
                jsonDict[k] = true
            } else if str == "false" {
                jsonDict[k] = false
            } else {
                jsonDict[k] = str
            }
        }

        guard let paramsData = try? JSONSerialization.data(withJSONObject: jsonDict, options: []),
            let paramsJson = String(data: paramsData, encoding: .utf8)
        else {
            throw SwiftunaError.invalidArgument("Failed to serialize params dictionary to JSON")
        }

        let userAttrsJson = try encodeUserAttrsJson(userAttrs)

        let status = paramsJson.withCString { cParams in
            withOptionalCString(userAttrsJson) { cAttrs in
                rustuna_study_enqueue_trial(raw, cParams, cAttrs)
            }
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to enqueue trial")
        }

        return self
    }

    /// Encodes user attributes as JSON, or `nil` when empty.
    private func encodeUserAttrsJson(_ userAttrs: [String: String]) throws(SwiftunaError) -> String? {
        guard !userAttrs.isEmpty else { return nil }
        guard let attrsData = try? JSONEncoder().encode(userAttrs),
            let str = String(data: attrsData, encoding: .utf8)
        else {
            throw SwiftunaError.invalidArgument("Failed to serialize userAttrs to JSON")
        }
        return str
    }

    /// Enqueues a trial configuration with strongly-typed parameter values.
    ///
    /// Same queue semantics as the `[String: any AttributeConvertible]`
    /// overload but without the JSON round trip on either side: values travel typed through
    /// `rustuna_study_enqueue_typed`. Prefer this overload whenever values
    /// are already ``ParameterValue`` (e.g. from a custom sampler) — it is
    /// the fastest enqueue path. Integer values must fit the `f64` mantissa
    /// exactly (true for all hyperparameter ranges).
    ///
    /// - Parameters:
    ///   - params: Parameter names and fixed values to evaluate.
    ///   - userAttrs: Optional user-defined metadata associated with the enqueued trial.
    /// - Returns: `self` to support fluent call chaining.
    /// - Throws: ``SwiftunaError`` if the study handle is invalid.
    ///
    /// ### Example
    /// ```swift
    /// try study.enqueue(["learning_rate": .double(0.01), "batch_size": .int(32)])
    /// ```
    @discardableResult
    public func enqueue(
        _ params: [String: ParameterValue],
        userAttrs: [String: String] = [:]
    ) throws(SwiftunaError) -> Self {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }
        return try enqueueTyped(params: params, raw: raw, userAttrs: userAttrs)
    }

    /// Shared typed-enqueue core: parallel C arrays straight into
    /// `rustuna_study_enqueue_typed`, no JSON anywhere.
    private func enqueueTyped(
        params: [String: ParameterValue],
        raw: OpaquePointer,
        userAttrs: [String: String]
    ) throws(SwiftunaError) -> Self {
        var names: [String] = []
        var kinds: [UInt8] = []
        var nums: [Double] = []
        var strs: [String] = []
        names.reserveCapacity(params.count)
        kinds.reserveCapacity(params.count)
        nums.reserveCapacity(params.count)
        strs.reserveCapacity(params.count)
        for (key, value) in params {
            names.append(key)
            switch value {
            case .int(let i):
                kinds.append(0)
                nums.append(Double(i))
                strs.append("")
            case .double(let d):
                kinds.append(1)
                nums.append(d)
                strs.append("")
            case .string(let s):
                kinds.append(2)
                nums.append(0)
                strs.append(s)
            case .bool(let b):
                kinds.append(3)
                nums.append(b ? 1.0 : 0.0)
                strs.append("")
            }
        }

        let userAttrsJson = try encodeUserAttrsJson(userAttrs)

        // Nested recursions keep each pointer array alive for the call, with
        // no concatenated scratch array and no index arithmetic. Empty input
        // yields nil bases with count 0, which the typed entry point accepts.
        let status = withCStrings(names) { namePtrs in
            withCStrings(strs) { strPtrs in
                kinds.withUnsafeBufferPointer { kb in
                    nums.withUnsafeBufferPointer { fb in
                        withOptionalCString(userAttrsJson) { cAttrs in
                            rustuna_study_enqueue_typed(
                                raw,
                                namePtrs.baseAddress, kb.baseAddress, fb.baseAddress,
                                strPtrs.baseAddress, names.count, cAttrs
                            )
                        }
                    }
                }
            }
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to enqueue trial")
        }

        return self
    }

    /// Evaluates hyperparameter importance scores using the PED-ANOVA (Partial Dependence ANOVA) algorithm.
    ///
    /// Computes how much each hyperparameter influences the objective function variance across completed trials.
    /// Requires at least 2 completed trials in the study.
    ///
    /// - Parameters:
    ///   - normalize: If `true`, importance scores are normalized so their sum equals `1.0`. Defaults to `true`.
    ///   - params: Optional subset of parameter names to evaluate. If `nil`, all evaluated parameters are included.
    /// - Returns: A `Result` containing a dictionary mapping parameter names to importance values,
    ///   or ``SwiftunaError/noCompletedTrial`` if fewer than 2 completed trials exist.
    ///
    /// ### Example
    /// ```swift
    /// if case .success(let importances) = study.paramImportances() {
    ///     for (param, score) in importances.sorted(by: { $0.value > $1.value }) {
    ///         print("\(param): \(String(format: "%.2f%%", score * 100))")
    ///     }
    /// }
    /// ```
    public func paramImportances(
        normalize: Bool = true,
        params: [String]? = nil
    ) -> Result<[String: Double], SwiftunaError> {
        guard let raw else {
            return .failure(.handleExpired("Study handle is expired or invalid"))
        }

        let completedTrials: [PersistedTrial]
        do {
            completedTrials = try trials.completed()
        } catch {
            return .failure(error)
        }
        guard completedTrials.count >= 2 else {
            return .failure(.noCompletedTrial)
        }

        var paramsJsonStr: String?
        if let params {
            if let data = try? JSONEncoder().encode(params),
                let str = String(data: data, encoding: .utf8)
            {
                paramsJsonStr = str
            }
        }

        var outJsonPtr: UnsafeMutablePointer<CChar>?
        let status = withOptionalCString(paramsJsonStr) { cParams in
            rustuna_study_get_param_importances(raw, normalize, cParams, &outJsonPtr)
        }

        if status != 0 {
            return .failure(.fromLastError(fallbackCode: status, context: "Failed to evaluate parameter importances"))
        }

        guard let outJsonPtr else {
            return .success([:])
        }
        defer { rustuna_string_free(outJsonPtr) }

        let len = strlen(outJsonPtr)
        let data = Data(bytesNoCopy: outJsonPtr, count: len, deallocator: .none)
        guard let dict = try? Self.trialDecoder.decode([String: Double].self, from: data) else {
            let jsonString = String(cString: outJsonPtr)
            return .failure(.unexpected("Failed to decode parameter importance payload: \(jsonString)"))
        }

        return .success(dict)
    }

    // MARK: - Trial Injection & Seeding

    /// Injects an already-evaluated historical trial or expert baseline into the study.
    ///
    /// Injected trials immediately update the surrogate model (e.g. ``TPESampler``) and Pareto frontier,
    /// allowing warm-starting from previous experiments or prior domain knowledge.
    ///
    /// - Parameter trial: A completed or pruned ``PersistedTrial`` object.
    /// - Throws: ``SwiftunaError`` if the trial's objective values count mismatches study directions or storage fails.
    ///
    /// ### Example
    /// ```swift
    /// let baseline = PersistedTrial(
    ///     number: 0,
    ///     state: .complete,
    ///     values: [0.42],
    ///     params: ["learning_rate": 0.001, "batch_size": 64]
    /// )
    /// try study.addTrial(baseline)
    /// ```
    public func addTrial(_ trial: PersistedTrial) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        // Fast upfront Swift validation
        if trial.state == .complete, trial.values.count != directions.count {
            throw SwiftunaError.invalidArgument(
                "Trial has \(trial.values.count) objective values, expected \(directions.count) for study '\(name)'"
            )
        }

        struct AddTrialJSONPayload: Encodable {
            let state: Int32
            let values: [Double]?
            let params: [String: Double]
            let intermediate_values: [String: Double]?
            let user_attrs: [String: String]?
            let system_attrs: [String: String]?
        }

        let intMap =
            trial.intermediateValues.isEmpty
            ? nil
            : Dictionary(
                uniqueKeysWithValues: trial.intermediateValues.map { (String($0.key), $0.value) }
            )

        let payload = AddTrialJSONPayload(
            state: trial.state.rawValue,
            values: trial.state == .complete ? trial.values : nil,
            params: trial.internalParams,
            intermediate_values: intMap,
            user_attrs: trial.userAttrs.isEmpty ? nil : trial.userAttrs,
            system_attrs: nil
        )

        guard let data = try? JSONEncoder().encode(payload),
            let jsonStr = String(data: data, encoding: .utf8)
        else {
            throw SwiftunaError.invalidArgument("Failed to encode trial payload to JSON")
        }

        let status = jsonStr.withCString { cStr in
            rustuna_study_add_trial_json(raw, cStr)
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to add trial to study '\(name)'")
        }
    }

    /// Injects multiple historical trials into the study in batch.
    public func addTrials(_ trials: [PersistedTrial]) throws(SwiftunaError) {
        for trial in trials {
            try addTrial(trial)
        }
    }
}
