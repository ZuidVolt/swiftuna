import Foundation
public import Swiftuna

/// A trial checked out to a remote worker, as returned by ``StudyCoordinator/ask()``.
///
/// This is the trial from the worker's perspective: sampled values to evaluate,
/// without the noncopyable `Trial` itself, which can never cross a process
/// boundary. Read values through the ``ParamReadable`` accessors.
public struct DistributedTrial: Sendable, Codable, Identifiable, Hashable, ParamReadable {
    /// The unique trial number within the study.
    public var id: Int { trialNumber }

    /// The unique trial number within the study.
    public let trialNumber: Int

    /// Dictionary of sampled hyperparameter values.
    public let params: [String: ParameterValue]

    /// Name of the study that produced this trial, for span attributes.
    ///
    /// Lets worker-side `swiftuna.trial` spans emit `study.name` with dashboard
    /// parity to local runs, without the worker ever touching the study.
    public let studyName: String

    /// W3C `traceparent` from the coordinator's sample span, if any.
    ///
    /// Opaque: forwarded untouched, never parsed. `nil` when the coordinator
    /// runs uninstrumented or the backend has no context support.
    public let traceParent: String?

    /// Creates a distributed trial.
    public init(
        trialNumber: Int,
        params: [String: ParameterValue],
        studyName: String = "",
        traceParent: String? = nil
    ) {
        self.trialNumber = trialNumber
        self.params = params
        self.studyName = studyName
        self.traceParent = traceParent
    }

    private enum CodingKeys: String, CodingKey {
        case trialNumber, params, studyName, traceParent
    }

    /// Decodes older payloads that predate `studyName`/`traceParent`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        trialNumber = try container.decode(Int.self, forKey: .trialNumber)
        params = try container.decode([String: ParameterValue].self, forKey: .params)
        studyName = try container.decodeIfPresent(String.self, forKey: .studyName) ?? ""
        traceParent = try container.decodeIfPresent(String.self, forKey: .traceParent)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(trialNumber, forKey: .trialNumber)
        try container.encode(params, forKey: .params)
        try container.encode(studyName, forKey: .studyName)
        try container.encodeIfPresent(traceParent, forKey: .traceParent)
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
