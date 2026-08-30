internal import Foundation
internal import LibRustuna

public final class Study: @unchecked Sendable {
    private var raw: OpaquePointer?
    public let name: String
    public let directions: [Direction]
    public let pruner: any Pruner

    public var direction: Direction {
        directions.first ?? .minimize
    }

    internal init(
        raw: OpaquePointer?,
        name: String,
        directions: [Direction],
        pruner: any Pruner = NopPruner()
    ) {
        self.raw = raw
        self.name = name
        self.directions = directions
        self.pruner = pruner
    }

    internal convenience init(
        raw: OpaquePointer?,
        name: String,
        direction: Direction,
        pruner: any Pruner = NopPruner()
    ) {
        self.init(raw: raw, name: name, directions: [direction], pruner: pruner)
    }

    deinit {
        if let raw {
            rustuna_study_free(raw)
        }
    }

    public func ask() throws(SwiftunaError) -> Trial {
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

    public func tell(
        consuming trial: consuming Trial,
        value: Double,
        state: TrialState = .complete
    ) throws(SwiftunaError) {
        try tell(consuming: trial, values: [value], state: state)
    }

    public func tell(
        consuming trial: consuming Trial,
        state: TrialState
    ) throws(SwiftunaError) {
        try tell(consuming: trial, values: [], state: state)
    }

    public func tell(
        consuming trial: consuming Trial,
        values: [Double],
        state: TrialState = .complete
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        let trialNumber = trial.number
        let intermediateSteps = trial.intermediateSteps
        _ = trial.takeHandle()  // Release from trial deinit

        var intermediateJsonStr: String?
        if !intermediateSteps.isEmpty {
            let stepMap = Dictionary(uniqueKeysWithValues: intermediateSteps.map { (String($0.step), $0.value) })
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

    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        objective: (inout Trial) throws(SwiftunaError) -> [Double]
    ) throws(SwiftunaError) {
        guard nTrials != nil || timeout != nil else {
            throw SwiftunaError.invalidArgument("Either nTrials or timeout must be specified in optimize")
        }

        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now + $0 }
        var iteration = 0

        while true {
            if let nTrials, iteration >= nTrials {
                break
            }
            if let deadline, clock.now >= deadline {
                break
            }

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
            let span = SwiftunaTelemetry.shared.tracer.startSpan(
                name: "swiftuna.trial",
                attributes: ["study.name": name, "trial.number": String(trialNum)]
            )

            let evalResult: Result<[Double], SwiftunaError>
            do {
                let vals = try objective(&activeTrial)
                evalResult = .success(vals)
            } catch {
                evalResult = .failure(error)
            }

            let elapsed = clock.now - startTime
            switch evalResult {
            case .success(let vals):
                span.setAttribute("trial.status", value: "complete")
                span.setAttribute(
                    "trial.duration_ms", value: String(format: "%.2f", Double(elapsed.components.attoseconds) / 1e15))
                span.end(status: .ok)
                try tell(consuming: activeTrial, values: vals, state: .complete)
            case .failure(let err):
                switch err {
                case .trialPruned:
                    span.setAttribute("trial.status", value: "pruned")
                    span.end(status: .ok)
                    try tell(consuming: activeTrial, values: [], state: .pruned)
                default:
                    span.setAttribute("trial.status", value: "failed")
                    span.end(status: .error(err.description))
                    try tell(consuming: activeTrial, values: [], state: .fail)
                    throw err
                }
            }
        }
    }

    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        objective: (inout Trial) throws(SwiftunaError) -> Double
    ) throws(SwiftunaError) {
        try optimize(nTrials: nTrials, timeout: timeout) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            [try objective(&trial)]
        }
    }

    public func optimize(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        concurrency: Int? = nil,
        objective: @Sendable @escaping (inout Trial) async throws -> Double
    ) async throws {
        guard nTrials != nil || timeout != nil else {
            throw SwiftunaError.invalidArgument("Either nTrials or timeout must be specified in optimize")
        }

        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now + $0 }
        let maxConcurrent = max(1, concurrency ?? ProcessInfo.processInfo.activeProcessorCount)
        let initialLimit = nTrials != nil ? min(maxConcurrent, nTrials!) : maxConcurrent

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                var submitted = 0

                func spawnNextWorker() {
                    if let nTrials, submitted >= nTrials { return }
                    if let deadline, clock.now >= deadline { return }

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

    /// Returns the best completed and feasible trial (where all constraints <= 0.0), or nil if no completed trial is feasible.
    ///
    /// Throws `SwiftunaError.unsupportedMultiObjective` if called on a multi-objective study (use `bestTrials` instead).
    public var bestFeasibleTrial: PersistedTrial? {
        get throws(SwiftunaError) {
            guard directions.count <= 1 else {
                throw SwiftunaError.unsupportedMultiObjective
            }
            return try trials.bestFeasible(direction: direction)
        }
    }

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

    public var bestValue: Double {
        get throws(SwiftunaError) {
            guard let bt = try bestTrial, let val = bt.value else {
                throw SwiftunaError.noTrialsFound
            }
            return val
        }
    }

    public var bestParams: [String: Double] {
        get throws(SwiftunaError) {
            guard let bt = try bestTrial else {
                throw SwiftunaError.noTrialsFound
            }
            return bt.params
        }
    }

    public var trials: [PersistedTrial] {
        get throws(SwiftunaError) {
            try trials(where: Set(TrialState.allCases))
        }
    }

    /// Fetches trials filtered by their `TrialState` at the storage layer.
    ///
    /// This avoids allocating discarded trial objects across the C ABI.
    public func trials(where states: Set<TrialState>) throws(SwiftunaError) -> [PersistedTrial] {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is invalid")
        }
        guard !states.isEmpty else { return [] }

        let mask = states.reduce(UInt32(0)) { $0 | (1 << $1.rawValue) }
        var trialsPtr: UnsafeMutablePointer<OpaquePointer?>?
        var count: Int = 0
        let status = rustuna_study_get_trials_filtered(raw, mask, &trialsPtr, &count)

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to get filtered trials")
        }

        return parseTrialsBuffer(trialsPtr, count: count)
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
            directions: directions
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

    private struct PersistedTrialJSONPayload: Decodable {
        let number: Int
        let state: Int32
        let values: [Double]
        let params: [String: Double]
        let user_attrs: [String: String]
        let constraints: [String: Double]
        let intermediate_values: [String: Double]
        let datetime_start: String?
        let datetime_complete: String?
    }

    private func parsePersistedTrial(_ trialPtr: OpaquePointer) -> PersistedTrial {
        var cJson: UnsafeMutablePointer<CChar>?
        guard rustuna_persisted_trial_get_json(trialPtr, &cJson) == 0, let cJson else {
            return PersistedTrial(number: 0, state: .fail, value: nil, values: [], params: [:], userAttrs: [:])
        }
        defer { rustuna_string_free(cJson) }

        guard let data = String(cString: cJson).data(using: .utf8),
            let payload = try? JSONDecoder().decode(PersistedTrialJSONPayload.self, from: data)
        else {
            return PersistedTrial(number: 0, state: .fail, value: nil, values: [], params: [:], userAttrs: [:])
        }

        let intermediateValues = Dictionary(
            uniqueKeysWithValues: payload.intermediate_values.compactMap { k, v in
                Int(k).map { ($0, v) }
            })
        let state = TrialState(rawValue: payload.state) ?? .fail
        let dtStart = payload.datetime_start.flatMap { parseOptunaDate($0) }
        let dtComplete = payload.datetime_complete.flatMap { parseOptunaDate($0) }

        return PersistedTrial(
            number: payload.number,
            state: state,
            value: payload.values.first,
            values: payload.values,
            params: payload.params,
            userAttrs: payload.user_attrs,
            constraints: payload.constraints,
            intermediateValues: intermediateValues,
            datetimeStart: dtStart,
            datetimeComplete: dtComplete
        )
    }

    private func parseOptunaDate(_ str: String) -> Date? {
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

    // MARK: - Study User Attributes & Analytics

    public func setUserAttr<K: AttributeKey>(
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

    public func userAttr<K: AttributeKey>(_ key: K.Type) throws(SwiftunaError) -> K.Value? {
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

    public subscript<K: AttributeKey>(_ key: K.Type) -> K.Value? {
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

    @discardableResult
    public func enqueue(
        _ params: [String: any AttributeConvertible],
        userAttrs: [String: String] = [:]
    ) throws(SwiftunaError) -> Self {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
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

        let userAttrsJson: String?
        if !userAttrs.isEmpty {
            guard let attrsData = try? JSONEncoder().encode(userAttrs),
                let str = String(data: attrsData, encoding: .utf8)
            else {
                throw SwiftunaError.invalidArgument("Failed to serialize userAttrs to JSON")
            }
            userAttrsJson = str
        } else {
            userAttrsJson = nil
        }

        let status = paramsJson.withCString { cParams in
            if let userAttrsJson {
                return userAttrsJson.withCString { cAttrs in
                    rustuna_study_enqueue_trial(raw, cParams, cAttrs)
                }
            }
            return rustuna_study_enqueue_trial(raw, cParams, nil)
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to enqueue trial")
        }

        return self
    }

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
                let str = String(data: data, encoding: .utf8) {
                paramsJsonStr = str
            }
        }

        var outJsonPtr: UnsafeMutablePointer<CChar>?
        let status: Int32
        if let paramsJsonStr {
            status = paramsJsonStr.withCString { cParams in
                rustuna_study_get_param_importances(raw, normalize, cParams, &outJsonPtr)
            }
        } else {
            status = rustuna_study_get_param_importances(raw, normalize, nil, &outJsonPtr)
        }

        if status != 0 {
            return .failure(.fromLastError(fallbackCode: status, context: "Failed to evaluate parameter importances"))
        }

        guard let outJsonPtr else {
            return .success([:])
        }
        defer { rustuna_string_free(outJsonPtr) }

        let jsonString = String(cString: outJsonPtr)
        guard let data = jsonString.data(using: .utf8),
            let dict = try? JSONDecoder().decode([String: Double].self, from: data)
        else {
            return .failure(.unexpected("Failed to decode parameter importance payload: \(jsonString)"))
        }

        return .success(dict)
    }

    // MARK: - Trial Injection & Seeding

    /// Injects an already-evaluated historical trial or expert baseline into the study.
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

        let intMap = trial.intermediateValues.isEmpty ? nil : Dictionary(
            uniqueKeysWithValues: trial.intermediateValues.map { (String($0.key), $0.value) }
        )

        let payload = AddTrialJSONPayload(
            state: trial.state.rawValue,
            values: trial.state == .complete ? trial.values : nil,
            params: trial.params,
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
