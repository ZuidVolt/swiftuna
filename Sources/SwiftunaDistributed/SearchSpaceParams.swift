import Foundation
public import Swiftuna

/// One declarative parameter in a search space.
///
/// Unlike a ``SearchSpace`` closure, `SearchParam` values are plain data:
/// `Codable` for logging and transmission, `Equatable` for testing.
public enum SearchParam: Sendable, Codable, Equatable {
    /// Floating-point range, mirroring `Trial.suggest(_:in:step:log:)`.
    case float(name: String, lower: Double, upper: Double, log: Bool = false, step: Double? = nil)
    /// Integer range, mirroring `Trial.suggest(_:in:step:log:)`.
    case int(name: String, lower: Int, upper: Int, step: Int = 1, log: Bool = false)
    /// Categorical choices, mirroring `Trial.suggest(_:choices:)`.
    case categorical(name: String, choices: [ParameterValue])

    /// The parameter name suggested by this descriptor.
    public var name: String {
        switch self {
        case .float(let name, _, _, _, _): return name
        case .int(let name, _, _, _, _): return name
        case .categorical(let name, _): return name
        }
    }

    /// Suggests this parameter on `trial`, returning its name and sampled value.
    public func sample(trial: inout Trial) throws -> (String, ParameterValue) {
        switch self {
        case .float(let name, let lower, let upper, let log, let step):
            let value = try trial.suggest(name, in: lower...upper, step: step, log: log)
            return (name, .double(value))
        case .int(let name, let lower, let upper, let step, let log):
            let value = try trial.suggest(name, in: lower...upper, step: step, log: log)
            return (name, .int(value))
        case .categorical(let name, let choices):
            let value = try trial.suggest(name, choices: choices)
            return (name, value)
        }
    }
}

/// A declarative, `Codable` search space: an ordered list of ``SearchParam``.
public struct SearchSpaceParams: Sendable, Codable, Equatable {
    /// Descriptors sampled in order on each `ask()`.
    public var params: [SearchParam]

    public init(_ params: [SearchParam]) {
        self.params = params
    }

    public init(_ params: SearchParam...) {
        self.params = params
    }

    /// Samples every descriptor, returning name-to-value mappings.
    ///
    /// - Throws: `SwiftunaError.invalidArgument` on duplicate parameter names.
    public func sample(trial: inout Trial) throws -> [String: ParameterValue] {
        let names = params.map(\.name)
        if Set(names).count != names.count {
            throw SwiftunaError.invalidArgument("Duplicate parameter names in search space: \(names)")
        }
        return Dictionary(uniqueKeysWithValues: try params.map { try $0.sample(trial: &trial) })
    }
}

extension SearchSpace {
    /// Creates a space from declarative `params` plus an optional procedural hatch.
    ///
    /// `customize` runs after the declarative params on the same trial, so one
    /// space mixes both styles: declare the static dims, compute the rest
    /// (conditional spaces, derived values) in code. On name collision the
    /// procedural entries win.
    public init(params: SearchSpaceParams, customizing customize: SamplerClosure? = nil) {
        self.init { trial in
            var dict = try params.sample(trial: &trial)
            if let customize {
                for (key, value) in try customize(&trial) {
                    dict[key] = value
                }
            }
            return dict
        }
    }
}
