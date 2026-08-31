import Foundation

/// The optimization direction for a single objective in a study.
///
/// A study can optimize for either cost/loss minimization (``Direction/minimize``) or performance/accuracy
/// maximization (``Direction/maximize``).
///
/// For multi-objective optimization, pass an array of `Direction` values corresponding to each objective
/// to ``createStudy(name:directions:storage:sampler:pruner:loadIfExists:)``.
///
/// ### Examples
///
/// Single-objective minimization:
/// ```swift
/// let study = try Swiftuna.createStudy(
///     name: "loss_minimization",
///     direction: .minimize
/// )
/// ```
///
/// Multi-objective optimization (e.g., maximize accuracy while minimizing latency):
/// ```swift
/// let study = try Swiftuna.createStudy(
///     name: "accuracy_vs_latency",
///     directions: [.maximize, .minimize]
/// )
/// ```
public enum Direction: Int32, Sendable, CaseIterable, Hashable {
    /// Directs the optimizer to find parameter configurations that produce the lowest objective value.
    ///
    /// Typical use cases include minimizing loss functions, prediction error (RMSE/MAE), execution latency,
    /// energy consumption, or memory footprint.
    case minimize = 0

    /// Directs the optimizer to find parameter configurations that produce the highest objective value.
    ///
    /// Typical use cases include maximizing validation accuracy, F1-score, reward in reinforcement learning,
    /// throughput (requests/sec), or return on investment.
    case maximize = 1
}
