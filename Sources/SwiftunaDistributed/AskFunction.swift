import Foundation
public import Swiftuna

/// The ask function sampled by the coordinator on each ``StudyCoordinator/ask()``.
///
/// Named for what it does — the suggest-phase of an objective, run
/// coordinator-side because `Trial` cannot cross a process boundary.
/// For static spaces prefer declarative ``SearchParam`` values via
/// `AskFunction(params:customizing:)`; use this closure form for the
/// procedural remainder.
public struct AskFunction: Sendable {
    public typealias SamplerClosure = @Sendable (inout Trial) throws -> [String: ParameterValue]
    private let sampler: SamplerClosure

    /// Creates an ask function from a sampling closure.
    public init(_ sampler: @escaping SamplerClosure) {
        self.sampler = sampler
    }

    /// Creates an ask function from declarative params, no hatch.
    public init(_ params: SearchParam...) {
        self.init(params: SearchSpaceParams(params))
    }

    /// Creates an ask function from declarative params, no hatch.
    public init(_ params: [SearchParam]) {
        self.init(params: SearchSpaceParams(params))
    }

    /// Samples hyperparameters on the given active trial.
    public func sample(trial: inout Trial) throws -> [String: ParameterValue] {
        try sampler(&trial)
    }
}
