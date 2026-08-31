import Foundation
public import Swiftuna

/// Defines the hyperparameter search space sampled by the coordinator on each ``StudyCoordinator/ask()``.
public struct SearchSpace: Sendable {
    public typealias SamplerClosure = @Sendable (inout Trial) throws -> [String: ParameterValue]
    private let sampler: SamplerClosure

    /// Creates a search space from a sampling closure.
    public init(_ sampler: @escaping SamplerClosure) {
        self.sampler = sampler
    }

    /// Samples hyperparameters on the given active trial.
    public func sample(trial: inout Trial) throws -> [String: ParameterValue] {
        try sampler(&trial)
    }
}
