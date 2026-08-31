import Foundation

/// A compile-time, strongly-typed key for trial constraints.
///
/// In mathematical optimization, constraints enforce requirements beyond the objective function, such as
/// memory limits, maximum inference latency, or safety bounds.
///
/// ### Constraint Formulation
/// Swiftuna follows the standard optimization convention where **a constraint is satisfied if and only if its value is less than or equal to zero**:
///
/// $$c_i(x) \le 0.0 \implies \text{Feasible}$$
/// $$c_i(x) > 0.0 \implies \text{Infeasible (violation magnitude)}$$
///
/// When constraints are specified:
/// - In ``TPESampler``, trials are partitioned by feasibility before building Parzen density models.
/// - In ``NSGAIISampler``, constrained-domination ensures feasible solutions strictly dominate infeasible ones,
///   and infeasible solutions are ranked by total constraint violation magnitude.
///
/// ### Example (Dot-Syntax Recommended)
/// ```swift
/// extension ConstraintKey {
///     static let maxLatency = ConstraintKey("max_latency_ms")
///     static let maxMemory = ConstraintKey("memory_mb_limit")
/// }
///
/// try study.optimize(nTrials: 100) { trial in
///     let batchSize = try trial.suggest("batch_size", in: 16...512)
///     let (loss, memMB, latencyMs) = evaluateModel(batchSize: batchSize)
///
///     // Enforce memMB <= 4096 and latencyMs <= 50.0:
///     trial[constraint: .maxMemory] = memMB - 4096.0
///     trial[constraint: .maxLatency] = latencyMs - 50.0
///
///     return loss
/// }
/// ```
public struct ConstraintKey: Sendable, Hashable, ExpressibleByStringLiteral {
    /// The unique string identifier for the constraint.
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public init(stringLiteral value: String) {
        self.name = value
    }
}

/// A protocol enabling custom types to serve as constraint keys.
public protocol ConstraintKeyProtocol: Sendable {
    static var name: String { get }
}
