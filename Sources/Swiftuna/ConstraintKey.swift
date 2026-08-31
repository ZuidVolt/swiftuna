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
///
/// ### Example
/// ```swift
/// public enum MemoryLimit: ConstraintKey {
///     public static let name = "memory_mb_limit"
/// }
///
/// public enum MaxLatency: ConstraintKey {
///     public static let name = "max_latency_ms"
/// }
///
/// try study.optimize(nTrials: 100) { trial in
///     let batchSize = try trial.suggest("batch_size", in: 16...512)
///     let (loss, memMB, latencyMs) = evaluateModel(batchSize: batchSize)
///
///     // Enforce memMB <= 4096 and latencyMs <= 50.0:
///     trial[constraint: MemoryLimit.self] = memMB - 4096.0
///     trial[constraint: MaxLatency.self] = latencyMs - 50.0
///
///     return loss
/// }
/// ```
public protocol ConstraintKey: Sendable {
    /// The unique string name identifier used to record this constraint in trial records.
    static var name: String { get }
}
