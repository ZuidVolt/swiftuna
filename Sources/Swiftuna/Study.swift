internal import Foundation
internal import LibRustuna

public final class Study: @unchecked Sendable {
    private var raw: OpaquePointer?
    public let name: String
    public let directions: [Direction]
    public var direction: Direction {
        directions.first ?? .minimize
    }

    internal init(raw: OpaquePointer?, name: String, directions: [Direction]) {
        self.raw = raw
        self.name = name
        self.directions = directions
    }

    internal convenience init(raw: OpaquePointer?, name: String, direction: Direction) {
        self.init(raw: raw, name: name, directions: [direction])
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
        return Trial(raw: trialPtr)
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
        values: [Double],
        state: TrialState = .complete
    ) throws(SwiftunaError) {
        guard let raw else {
            throw SwiftunaError.handleExpired("Study handle is expired or invalid")
        }

        let trialNumber = trial.number
        _ = trial.takeHandle()  // Release from trial deinit

        let status: Int32
        if values.isEmpty {
            status = rustuna_study_tell_multi(raw, UInt32(trialNumber), state.rawValue, nil, 0)
        } else {
            status = values.withUnsafeBufferPointer { buf in
                rustuna_study_tell_multi(raw, UInt32(trialNumber), state.rawValue, buf.baseAddress, values.count)
            }
        }

        if status != 0 {
            throw SwiftunaError.fromLastError(
                fallbackCode: status, context: "Failed to tell multi-objective trial #\(trialNumber)")
        }
    }

    public func optimize(
        nTrials: Int,
        objective: (inout Trial) throws(SwiftunaError) -> [Double]
    ) throws(SwiftunaError) {
        for _ in 0..<nTrials {
            var trial = try ask()
            let evalResult: Result<[Double], SwiftunaError>
            do {
                let vals = try objective(&trial)
                evalResult = .success(vals)
            } catch {
                evalResult = .failure(error)
            }
            switch evalResult {
            case .success(let vals):
                try tell(consuming: trial, values: vals, state: .complete)
            case .failure(let err):
                switch err {
                case .trialPruned:
                    try tell(consuming: trial, values: [], state: .pruned)
                default:
                    try tell(consuming: trial, values: [], state: .fail)
                    throw err
                }
            }
        }
    }

    public func optimize(
        nTrials: Int,
        objective: (inout Trial) throws(SwiftunaError) -> Double
    ) throws(SwiftunaError) {
        try optimize(nTrials: nTrials) { (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            [try objective(&trial)]
        }
    }

    public func optimize(
        nTrials: Int,
        concurrency: Int? = nil,
        objective: @Sendable @escaping (inout Trial) async throws(SwiftunaError) -> Double
    ) async throws(SwiftunaError) {
        let maxConcurrent = max(1, concurrency ?? ProcessInfo.processInfo.activeProcessorCount)
        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                var submitted = 0

                func spawnNextWorker() {
                    guard submitted < nTrials else { return }
                    submitted += 1
                    group.addTask {
                        var trial = try self.ask()
                        let evalResult: Result<Double, SwiftunaError>
                        do {
                            let val = try await objective(&trial)
                            evalResult = .success(val)
                        } catch let err as SwiftunaError {
                            evalResult = .failure(err)
                        } catch {
                            evalResult = .failure(.unexpected("\(error)"))
                        }

                        switch evalResult {
                        case .success(let val):
                            try self.tell(consuming: trial, value: val, state: .complete)
                        case .failure(let err):
                            switch err {
                            case .trialPruned:
                                try self.tell(consuming: trial, value: 0.0, state: .pruned)
                            default:
                                try self.tell(consuming: trial, value: 0.0, state: .fail)
                                throw err
                            }
                        }
                    }
                }

                for _ in 0..<min(maxConcurrent, nTrials) {
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
        nTrials: Int,
        objective: (inout Trial) throws(E) -> Double
    ) throws {
        for _ in 0..<nTrials {
            var trial = try ask()
            let evalResult: Result<Double, E>
            do {
                let val = try objective(&trial)
                evalResult = .success(val)
            } catch {
                evalResult = .failure(error)
            }
            switch evalResult {
            case .success(let val):
                try tell(consuming: trial, value: val, state: .complete)
            case .failure(let err):
                try tell(consuming: trial, value: 0.0, state: .fail)
                throw err
            }
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
            return try trials.completed().feasible().bestFeasible(direction: direction)
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
            guard let raw else {
                throw SwiftunaError.handleExpired("Study handle is invalid")
            }

            var trialsPtr: UnsafeMutablePointer<OpaquePointer?>?
            var count: Int = 0
            let status = rustuna_study_get_trials(raw, &trialsPtr, &count)

            if status != 0 {
                throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to get study trials")
            }

            return parseTrialsBuffer(trialsPtr, count: count)
        }
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

    private func parsePersistedTrial(_ trialPtr: OpaquePointer) -> PersistedTrial {
        let num = Int(rustuna_persisted_trial_get_number(trialPtr))
        let stateRaw = rustuna_persisted_trial_get_state(trialPtr)
        let state = TrialState(rawValue: stateRaw) ?? .fail

        var valsPtr: UnsafeMutablePointer<Double>?
        var valsCount: Int = 0
        var values: [Double] = []
        if rustuna_persisted_trial_get_values(trialPtr, &valsPtr, &valsCount) {
            if let valsPtr, valsCount > 0 {
                values = Array(UnsafeBufferPointer(start: valsPtr, count: valsCount))
                rustuna_values_buffer_free(valsPtr, valsCount)
            }
        }

        var jsonPtr: UnsafeMutablePointer<CChar>?
        let jsonStatus = rustuna_persisted_trial_get_params_json(trialPtr, &jsonPtr)
        var params: [String: Double] = [:]

        if jsonStatus == 0, let jsonPtr {
            defer { rustuna_string_free(jsonPtr) }
            let jsonString = String(cString: jsonPtr)
            if let data = jsonString.data(using: .utf8),
                let parsed = try? JSONDecoder().decode([String: Double].self, from: data)
            {
                params = parsed
            }
        }

        var userAttrsJsonPtr: UnsafeMutablePointer<CChar>?
        let userAttrsStatus = rustuna_persisted_trial_get_user_attrs_json(trialPtr, &userAttrsJsonPtr)
        var userAttrs: [String: String] = [:]

        if userAttrsStatus == 0, let userAttrsJsonPtr {
            defer { rustuna_string_free(userAttrsJsonPtr) }
            let jsonString = String(cString: userAttrsJsonPtr)
            if let data = jsonString.data(using: .utf8),
                let parsed = try? JSONDecoder().decode([String: String].self, from: data)
            {
                userAttrs = parsed
            }
        }

        var constraintsJsonPtr: UnsafeMutablePointer<CChar>?
        let constraintsStatus = rustuna_persisted_trial_get_constraints_json(trialPtr, &constraintsJsonPtr)
        var constraints: [String: Double] = [:]

        if constraintsStatus == 0, let constraintsJsonPtr {
            defer { rustuna_string_free(constraintsJsonPtr) }
            let jsonString = String(cString: constraintsJsonPtr)
            if let data = jsonString.data(using: .utf8),
                let parsed = try? JSONDecoder().decode([String: Double].self, from: data)
            {
                constraints = parsed
            }
        }

        return PersistedTrial(
            number: num,
            state: state,
            value: values.first,
            values: values,
            params: params,
            userAttrs: userAttrs,
            constraints: constraints
        )
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
            } else {
                return rustuna_study_enqueue_trial(raw, cParams, nil)
            }
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
                let str = String(data: data, encoding: .utf8)
            {
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
}
