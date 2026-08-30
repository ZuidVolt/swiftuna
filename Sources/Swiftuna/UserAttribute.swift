import Foundation

/// A type that can be converted to and from a string representation stored in Rustuna's attribute engine.
public protocol AttributeConvertible: Sendable {
    static func fromAttributeString(_ raw: String) -> Self?
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
public protocol AttributeKey: Sendable {
    associatedtype Value: AttributeConvertible
    static var name: String { get }
}

/// Dynamic JSON wrapper for arbitrary Codable payloads stored as user attributes.
public struct CodableAttribute<T: Codable & Sendable>: AttributeConvertible {
    public let value: T

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
