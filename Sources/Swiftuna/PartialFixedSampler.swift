import Foundation

/// A sampler wrapper that fixes a specified subset of hyperparameters to constant values while
/// delegating remaining free parameters to an underlying base sampler.
///
/// `PartialFixedSampler` implements Optuna's `PartialFixedSampler` pattern, allowing specific
/// dimensions to remain static across trials while letting free parameters be optimized by either
/// a Rust-backed engine (e.g. ``TPESampler``, ``QMCSampler``, ``NSGAIISampler``) or a pure Swift
/// strategy (e.g. ``CMASampler``).
///
/// ### Example
/// ```swift
/// // Fix batch_size and optimizer while optimizing learning rate with TPESampler:
/// let baseSampler = TPESampler(seed: 42)
/// let partialSampler = PartialFixedSampler(
///     fixedParams: [
///         "batch_size": 32,
///         "optimizer": "adam"
///     ],
///     baseSampler: baseSampler
/// )
///
/// let study = try Swiftuna.createStudy(sampler: partialSampler)
/// try study.optimize(nTrials: 20) { trial in
///     let lr = try trial.suggest("lr", in: 1e-4...1e-1, log: true)
///     let batch = try trial.suggest("batch_size", from: [16, 32, 64])
///     let opt = try trial.suggest("optimizer", from: ["adam", "sgd"])
///     return evaluateModel(lr: lr, batch: batch, opt: opt)
/// }
/// ```
public struct PartialFixedSampler: CustomSampler, Sendable {
    /// The fixed parameter assignments to inject into each trial.
    public let fixedParams: [String: ParameterValue]

    /// The underlying delegate sampler handling unfixed parameters.
    public let delegate: DelegateSampler

    /// Delegate sampler choices supported by ``PartialFixedSampler``.
    public enum DelegateSampler: Sendable {
        /// A Rust-backed sampler engine (e.g. ``TPESampler``, ``QMCSampler``, ``NSGAIISampler``).
        case rust(any Sampler)
        /// A pure Swift custom suggestion algorithm (e.g. ``CMASampler``).
        case custom(any CustomSampler)
    }

    public var retainsParameterHistory: Bool {
        switch delegate {
        case .rust:
            return true
        case .custom(let custom):
            return custom.retainsParameterHistory
        }
    }

    public var underlyingSampler: (any Sampler)? {
        switch delegate {
        case .rust(let sampler):
            return sampler
        case .custom:
            return nil
        }
    }

    /// Initializes a partial fixed sampler wrapping a Rust-backed ``Sampler``.
    ///
    /// - Parameters:
    ///   - fixedParams: Dictionary of parameter names and their fixed values.
    ///   - baseSampler: The delegate sampler for un-fixed parameters.
    public init(
        fixedParams: [String: ParameterValue],
        baseSampler: any Sampler
    ) {
        self.fixedParams = fixedParams
        self.delegate = .rust(baseSampler)
    }

    /// Initializes a partial fixed sampler wrapping a Swift ``CustomSampler``.
    ///
    /// - Parameters:
    ///   - fixedParams: Dictionary of parameter names and their fixed values.
    ///   - baseSampler: The delegate sampler for un-fixed parameters.
    public init(
        fixedParams: [String: ParameterValue],
        baseSampler: any CustomSampler
    ) {
        self.fixedParams = fixedParams
        self.delegate = .custom(baseSampler)
    }

    /// Proposes the next trial configuration, overlaying fixed parameters.
    public func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        switch delegate {
        case .rust:
            // Returning fixedParams causes Swiftuna to enqueue them;
            // all other suggested parameters automatically fall back to the study's Rust sampler.
            return fixedParams
        case .custom(let custom):
            var params = try custom.sample(history: history, trialNumber: trialNumber)
            for (k, v) in fixedParams {
                params[k] = v
            }
            return params
        }
    }
}
