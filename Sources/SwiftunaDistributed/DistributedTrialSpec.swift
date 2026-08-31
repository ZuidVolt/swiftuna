import Foundation
public import Swiftuna

/// The serialized specification of an active hyperparameter trial dispatched over the network to a remote worker.
public struct DistributedTrialSpec: Sendable, Codable, Identifiable, Hashable {
    /// The unique trial number within the study.
    public var id: Int { trialNumber }

    /// The unique trial number within the study.
    public let trialNumber: Int

    /// Dictionary of sampled hyperparameter values.
    public let params: [String: ParameterValue]

    /// Creates a distributed trial specification.
    public init(trialNumber: Int, params: [String: ParameterValue]) {
        self.trialNumber = trialNumber
        self.params = params
    }

    /// Returns the 64-bit floating point value for the parameter, if present.
    public func double(_ name: String) -> Double? {
        params[name]?.asDouble
    }

    /// Returns the 32-bit floating point value (`Float`) for the parameter, if present.
    public func float(_ name: String) -> Float? {
        params[name]?.asDouble.map(Float.init)
    }

    /// Returns the integer value for the parameter, if present.
    public func int(_ name: String) -> Int? {
        params[name]?.asInt
    }

    /// Returns the string value for the parameter, if present.
    public func string(_ name: String) -> String? {
        params[name]?.asString
    }

    /// Returns the boolean value for the parameter, if present.
    public func bool(_ name: String) -> Bool? {
        params[name]?.asBool
    }

    /// Deserializes a categorical parameter to a Swift enum conforming to `RawRepresentable`.
    public func param<T: RawRepresentable>(_ name: String, as: T.Type = T.self) -> T? where T.RawValue == String {
        guard let s = params[name]?.asString else { return nil }
        return T(rawValue: s)
    }
}
