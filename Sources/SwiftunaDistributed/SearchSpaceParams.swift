import Foundation
public import Swiftuna

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
        return Dictionary(uniqueKeysWithValues: try params.map { try trial.suggest($0) })
    }
}

extension AskFunction {
    /// Creates an ask function from declarative `params` plus an optional procedural hatch.
    ///
    /// `customize` runs after the declarative params on the same trial, so one
    /// function mixes both styles: declare the static dims, compute the rest
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
