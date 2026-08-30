import Foundation

/// A type that can be converted to and from a string representation stored in Rustuna's attribute storage engine.
///
/// Primitives including `String`, `Int`, `Int64`, `UInt64`, `Double`, `Float`, and `Bool` conform automatically.
/// Any `RawRepresentable` enum whose `RawValue` is `AttributeConvertible` also conforms automatically.
public protocol AttributeConvertible: Sendable {
    /// Deserializes an instance from its raw string representation stored in the database.
    static func fromAttributeString(_ raw: String) -> Self?

    /// Serializes this instance into a string representation for storage.
    func toAttributeString() -> String
}

extension String: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> String? { raw }
    @inlinable
    public func toAttributeString() -> String { self }
}

extension Int: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Int? { Int(raw) }
    @inlinable
    public func toAttributeString() -> String { String(self) }
}

extension Int64: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Int64? { Int64(raw) }
    @inlinable
    public func toAttributeString() -> String { String(self) }
}

extension UInt64: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> UInt64? { UInt64(raw) }
    @inlinable
    public func toAttributeString() -> String { String(self) }
}

extension Double: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Double? { Double(raw) }
    @inlinable
    public func toAttributeString() -> String { String(self) }
}

extension Float: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Float? { Float(raw) }
    @inlinable
    public func toAttributeString() -> String { String(self) }
}

extension Bool: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Bool? {
        if raw == "true" { return true }
        if raw == "false" { return false }
        return nil
    }
    @inlinable
    public func toAttributeString() -> String { self ? "true" : "false" }
}

/// Automatic conformance for RawRepresentable enums whose RawValue is AttributeConvertible.
extension AttributeConvertible where Self: RawRepresentable, RawValue: AttributeConvertible {
    @inlinable
    public static func fromAttributeString(_ raw: String) -> Self? {
        guard let rawVal = RawValue.fromAttributeString(raw) else { return nil }
        return Self(rawValue: rawVal)
    }

    @inlinable
    public func toAttributeString() -> String {
        rawValue.toAttributeString()
    }
}

/// A statically typed, compile-time key for user attributes on studies and trials.
///
/// Unlike Python's untyped string dictionary approach (`trial.set_user_attr("key", value)`),
/// `AttributeKey` allows Swift developers to declare typed schema keys that enforce both the attribute
/// name and its value type at compile time.
///
/// ### Example
/// ```swift
/// public enum ArchitectureTag: AttributeKey {
///     public typealias Value = String
///     public static let name = "architecture"
/// }
///
/// public enum MaxEpochs: AttributeKey {
///     public typealias Value = Int
///     public static let name = "max_epochs"
/// }
///
/// // Type-safe subscript write on an active trial:
/// trial[ArchitectureTag.self] = "ResNet-50"
/// trial[MaxEpochs.self] = 100
///
/// // Type-safe read on a completed trial:
/// let arch: String? = bestTrial[ArchitectureTag.self]
/// let epochs: Int? = bestTrial[MaxEpochs.self]
/// ```
public protocol AttributeKey: Sendable {
    /// The Swift type associated with this attribute key.
    associatedtype Value: AttributeConvertible

    /// The unique string identifier used when storing the attribute in SQLite or memory.
    static var name: String { get }
}

/// A wrapper that enables storing any `Codable & Sendable` Swift structure as a JSON-encoded user attribute.
///
/// Use `CodableAttribute` when an attribute contains nested or complex data that exceeds primitive scalar types.
///
/// ### Example
/// ```swift
/// public struct ModelConfig: Codable, Sendable {
///     public let layers: [Int]
///     public let dropout: Double
/// }
///
/// public enum ConfigKey: AttributeKey {
///     public typealias Value = CodableAttribute<ModelConfig>
///     public static let name = "model_config"
/// }
///
/// let config = ModelConfig(layers: [64, 128, 64], dropout: 0.2)
/// trial[ConfigKey.self] = CodableAttribute(config)
///
/// if let saved = bestTrial[ConfigKey.self]?.value {
///     print("Saved layers: \(saved.layers)")
/// }
/// ```
public struct CodableAttribute<T: Codable & Sendable>: AttributeConvertible {
    /// The decoded value instance.
    public let value: T

    /// Wraps a Codable value for attribute storage.
    public init(_ value: T) {
        self.value = value
    }

    public static func fromAttributeString(_ raw: String) -> CodableAttribute<T>? {
        guard let data = raw.data(using: .utf8),
            let val = try? JSONDecoder().decode(T.self, from: data)
        else {
            return nil
        }
        return Self(val)
    }

    public func toAttributeString() -> String {
        guard let data = try? JSONEncoder().encode(value),
            let str = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return str
    }
}
