import Foundation

/// A compile-time, strongly-typed key for trial constraints.
///
/// Types conforming to `ConstraintKey` allow defining static constraint names that eliminate
/// stringly-typed typos while retaining zero runtime overhead.
///
/// ### Example
/// ```swift
/// public enum MaxLatency: ConstraintKey {
///     public static let name = "max_latency"
/// }
///
/// // Type-safe constraint setting:
/// try trial.setConstraint(MaxLatency.self, value: latency - 15.0)
///
/// // Subscript access:
/// trial[constraint: MaxLatency.self] = latency - 15.0
/// ```
public protocol ConstraintKey: Sendable {
    /// The unique string name identifier used to record this constraint in trial records.
    static var name: String { get }
}
