import Foundation
internal import LibRustuna

public struct Trial: ~Copyable {
    private var raw: OpaquePointer?
    public let number: Int

    internal init(raw: OpaquePointer?) {
        self.raw = raw
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
              range.lowerBound <= range.upperBound else {
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
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Categorical suggestion failed for '\(name)'")
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
}
