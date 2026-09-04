import Foundation

/// A strongly typed, heterogeneous hyperparameter value evaluated by an optimization study.
///
/// Unlike internal mathematical floats, `ParameterValue` preserves the true domain
/// representation of the parameter, including categorical string choices and integer steps.
///
/// ### Example
/// ```swift
/// if let opt = trial.params["optimizer"]?.asString {
///     print("Suggested optimizer: \(opt)")
/// }
/// ```
public enum ParameterValue: Sendable, CustomStringConvertible, Codable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)

    // MARK: - Typed Accessors

    /// Returns the integer value if this parameter is an integer, or converts a whole float.
    public var asInt: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return d.rounded() == d ? Int(d) : nil
        case .bool(let b): return b ? 1 : 0
        case .string(let s): return Int(s)
        }
    }

    /// Returns the numerical floating-point representation of this parameter.
    public var asDouble: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        case .bool(let b): return b ? 1.0 : 0.0
        case .string(let s): return Double(s)
        }
    }

    /// Returns the string representation or categorical label.
    public var asString: String? {
        switch self {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        }
    }

    /// Returns the boolean value if this parameter represents a boolean.
    public var asBool: Bool? {
        switch self {
        case .bool(let b): return b
        case .int(let i): return i == 0 ? false : (i == 1 ? true : nil)
        case .double(let d): return d == 0.0 ? false : (d == 1.0 ? true : nil)
        case .string(let s): return Bool(s)
        }
    }

    // MARK: - Telemetry

    /// Maps this value to a typed span attribute, preserving numeric types
    /// for backend aggregation instead of stringifying them.
    public var telemetryAttribute: TelemetryAttribute {
        switch self {
        case .int(let i): return .int(i)
        case .double(let d): return .double(d)
        case .string(let s): return .string(s)
        case .bool(let b): return .bool(b)
        }
    }

    // MARK: - CustomStringConvertible

    public var description: String {        switch self {
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .string(let s): return s
        case .bool(let b): return String(b)
        }
    }

    // MARK: - Initializer

    public init<T: Equatable>(_ value: T) {
        if let p = value as? ParameterValue {
            self = p
        } else if let i = value as? Int {
            self = .int(i)
        } else if let d = value as? Double {
            self = .double(d)
        } else if let f = value as? Float {
            self = .double(Double(f))
        } else if let b = value as? Bool {
            self = .bool(b)
        } else if let s = value as? String {
            self = .string(s)
        } else if let r = value as? (any RawRepresentable), let s = r.rawValue as? String {
            self = .string(s)
        } else {
            self = .string(String(describing: value))
        }
    }

    // MARK: - Codable

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid parameter value")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .bool(let b): try container.encode(b)
        }
    }
}

// MARK: - ExpressibleBy Literals

extension ParameterValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension ParameterValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension ParameterValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension ParameterValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension ParameterValue: Equatable {
    public static func == (lhs: ParameterValue, rhs: ParameterValue) -> Bool {
        switch (lhs, rhs) {
        case (.int(let a), .int(let b)):
            return a == b
        case (.double(let a), .double(let b)):
            return a == b
        case (.string(let a), .string(let b)):
            return a == b
        case (.bool(let a), .bool(let b)):
            return a == b
        case (.int(let a), .double(let b)):
            return Double(a) == b
        case (.double(let a), .int(let b)):
            return a == Double(b)
        default:
            return false
        }
    }
}

extension ParameterValue: Hashable {
    public func hash(into hasher: inout Hasher) {
        if let d = asDouble {
            hasher.combine(d)
        } else if let s = asString {
            hasher.combine(s)
        } else if let b = asBool {
            hasher.combine(b)
        }
    }
}

extension ParameterValue: Comparable {
    public static func < (lhs: ParameterValue, rhs: ParameterValue) -> Bool {
        if let ld = lhs.asDouble, let rd = rhs.asDouble {
            return ld < rd
        }
        return lhs.description < rhs.description
    }
}

/// Uniform typed reading of hyperparameter values.
///
/// `PersistedTrial`, `DistributedTrial`, and any future carrier conform
/// once, so readers behave identically everywhere instead of being
/// reimplemented per carrier.
public protocol ParamReadable {
    /// Returns the 64-bit floating-point value for the parameter, if present.
    func double(_ name: String) -> Double?

    /// Returns the 32-bit floating-point value for the parameter, if present.
    func float(_ name: String) -> Float?

    /// Returns the integer value for the parameter, if present.
    func int(_ name: String) -> Int?

    /// Returns the string value for the parameter, if present.
    func string(_ name: String) -> String?

    /// Returns the boolean value for the parameter, if present.
    func bool(_ name: String) -> Bool?

    /// Deserializes a categorical parameter to a `RawRepresentable` enum.
    func param<T: RawRepresentable>(_ name: String, as: T.Type) -> T? where T.RawValue == String
}
