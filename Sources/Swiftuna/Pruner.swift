import Foundation

/// Protocol for deciding whether an active trial should be early-stopped based on intermediate values.
public protocol Pruner: Sendable {
    /// Evaluates whether the given trial should be pruned at `step`.
    func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool
}

/// No-operation pruner that never early-stops trials (default).
public struct NopPruner: Pruner {
    public init() {}

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) -> Bool {
        false
    }
}

/// Pruner using the median stopping rule.
///
/// Prunes an active trial if its intermediate value at a step is worse than the median (50th percentile)
/// of intermediate values reported by previous completed or pruned trials at the same step.
///
/// Under the hood, `MedianPruner` delegates to ``PercentilePruner`` with `percentile: 50.0`.
///
/// ### Example
/// ```swift
/// let pruner = MedianPruner(nStartupTrials: 5, nWarmupSteps: 10, intervalSteps: 2)
/// let study = try Swiftuna.createStudy(pruner: pruner)
/// ```
public struct MedianPruner: Pruner {
    public let underlying: PercentilePruner

    /// Number of initial trials executed without pruning to establish an initial performance baseline.
    public var nStartupTrials: Int { underlying.nStartupTrials }

    /// Number of initial steps within each trial before pruning evaluation begins.
    public var nWarmupSteps: Int { underlying.nWarmupSteps }

    /// Step interval at which pruning decisions are evaluated.
    public var intervalSteps: Int { underlying.intervalSteps }

    /// Initializes a Median pruner.
    ///
    /// - Parameters:
    ///   - nStartupTrials: Trials run before pruning starts. Defaults to `5`.
    ///   - nWarmupSteps: Steps within a trial before pruning starts. Defaults to `0`.
    ///   - intervalSteps: Step frequency for evaluating pruning. Defaults to `1`.
    public init(
        nStartupTrials: Int = 5,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1
    ) {
        self.underlying = PercentilePruner(
            percentile: 50.0,
            nStartupTrials: nStartupTrials,
            nWarmupSteps: nWarmupSteps,
            intervalSteps: intervalSteps
        )
    }

    @inline(__always)
    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        try underlying.shouldPrune(
            study: study,
            trialNumber: trialNumber,
            step: step,
            currentValue: currentValue
        )
    }
}

/// Pruner to keep trials whose intermediate values fall in the top percentile of historical trials.
///
/// Prunes an active trial if its intermediate value is worse than the given `percentile` among previous
/// completed and pruned trials at the same step.
///
/// ### Example
/// ```swift
/// // Keep only trials performing in the top 25% (prune bottom 75%)
/// let pruner = PercentilePruner(percentile: 25.0, nStartupTrials: 5)
/// ```
public struct PercentilePruner: Pruner {
    /// Target percentile threshold between 0.0 and 100.0.
    public let percentile: Double

    /// Number of initial trials run before pruning decisions take effect.
    public let nStartupTrials: Int

    /// Number of initial steps within each trial before pruning evaluation begins.
    public let nWarmupSteps: Int

    /// Step interval at which pruning decisions are evaluated.
    public let intervalSteps: Int

    /// Initializes a Percentile pruner.
    ///
    /// - Parameters:
    ///   - percentile: Percentile threshold between 0.0 and 100.0 (e.g. `25.0` for top 25%).
    ///   - nStartupTrials: Trials run before pruning starts. Defaults to `5`.
    ///   - nWarmupSteps: Steps within each trial before pruning starts. Defaults to `0`.
    ///   - intervalSteps: Step frequency for evaluating pruning. Defaults to `1`.
    public init(
        percentile: Double = 50.0,
        nStartupTrials: Int = 5,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1
    ) {
        self.percentile = min(100.0, max(0.0, percentile))
        self.nStartupTrials = max(0, nStartupTrials)
        self.nWarmupSteps = max(0, nWarmupSteps)
        self.intervalSteps = max(1, intervalSteps)
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        if step < nWarmupSteps || step % intervalSteps != 0 {
            return false
        }

        let allTrials = try study.trials
        let previousTrials = allTrials.filter { $0.number < trialNumber && ($0.state == .complete || $0.state == .pruned) }

        if previousTrials.count < nStartupTrials {
            return false
        }

        let valuesAtStep: [Double] = previousTrials.compactMap { $0.intermediateValues[step] }
        guard !valuesAtStep.isEmpty else {
            return false
        }

        let sortedValues = valuesAtStep.sorted()
        let index = Int(Double(sortedValues.count - 1) * (percentile / 100.0))
        let threshold = sortedValues[min(max(0, index), sortedValues.count - 1)]

        if study.direction == .minimize {
            return currentValue > threshold
        }
        return currentValue < threshold
    }
}

/// Pruner that prunes immediately if an intermediate value crosses absolute predefined thresholds.
///
/// Evaluates whether the reported value exceeds `upper` or drops below `lower`. Also prunes `NaN` evaluations.
///
/// ### Example
/// ```swift
/// // Prune immediately if loss exceeds 100.0 or drops below 0.0
/// let pruner = ThresholdPruner(lower: 0.0, upper: 100.0)
/// ```
public struct ThresholdPruner: Pruner {
    /// Lower bound threshold. If an intermediate value is `< lower`, the trial is pruned.
    public let lower: Double?

    /// Upper bound threshold. If an intermediate value is `> upper`, the trial is pruned.
    public let upper: Double?

    /// Initializes a Threshold pruner.
    ///
    /// - Parameters:
    ///   - lower: Optional lower bound cutoff.
    ///   - upper: Optional upper bound cutoff.
    public init(lower: Double? = nil, upper: Double? = nil) {
        self.lower = lower
        self.upper = upper
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) -> Bool {
        if currentValue.isNaN {
            return true
        }
        if let upper, currentValue > upper {
            return true
        }
        if let lower, currentValue < lower {
            return true
        }
        return false
    }
}

/// Asynchronous Successive Halving Algorithm (ASHA) pruner.
///
/// Allocates resources across geometric rungs scaling by `reductionFactor` ($\eta$). At each rung,
/// only the top $1 / \eta$ fraction of trials are promoted to continue evaluation to the next rung.
///
/// Unlike synchronous successive halving, ASHA evaluates trials asynchronously without blocking workers,
/// making it ideal for distributed or concurrent optimization runs.
///
/// ### Example
/// ```swift
/// let pruner = SuccessiveHalvingPruner(minResource: 1, reductionFactor: 4)
/// let study = try Swiftuna.createStudy(pruner: pruner)
/// ```
public struct SuccessiveHalvingPruner: Pruner {
    /// Minimum resource allocation (e.g. initial epoch count or step) before trials encounter the first rung.
    public let minResource: Int

    /// Reduction factor $\eta$ governing the promotion rate ($1 / \eta$) and rung progression spacing.
    public let reductionFactor: Int

    /// Initial early stopping rate exponent determining the first rung index.
    public let minEarlyStoppingRate: Int

    /// Minimum number of trials that must reach a rung before pruning evaluations take effect.
    public let bootstrapCount: Int

    /// Optional predicate filtering which trials belong to this evaluation arm (used by ``HyperbandPruner``).
    public let trialFilter: (@Sendable (PersistedTrial) -> Bool)?

    /// Initializes an Asynchronous Successive Halving (ASHA) pruner.
    ///
    /// - Parameters:
    ///   - minResource: Minimum resource step before the first rung. Defaults to `1`.
    ///   - reductionFactor: Promotion divisor $\eta$. Defaults to `4`.
    ///   - minEarlyStoppingRate: Initial rung rate exponent. Defaults to `0`.
    ///   - bootstrapCount: Trials needed at a rung before pruning begins. Defaults to `0`.
    ///   - trialFilter: Optional closure to isolate trials by bracket.
    public init(
        minResource: Int = 1,
        reductionFactor: Int = 4,
        minEarlyStoppingRate: Int = 0,
        bootstrapCount: Int = 0,
        trialFilter: (@Sendable (PersistedTrial) -> Bool)? = nil
    ) {
        self.minResource = max(1, minResource)
        self.reductionFactor = max(2, reductionFactor)
        self.minEarlyStoppingRate = max(0, minEarlyStoppingRate)
        self.bootstrapCount = max(0, bootstrapCount)
        self.trialFilter = trialFilter
    }

    /// Computes the step index for the given rung index $k$.
    public func rungStep(at index: Int) -> Int {
        var r = minResource
        let totalRate = minEarlyStoppingRate + index
        for _ in 0..<totalRate {
            r *= reductionFactor
        }
        return r
    }

    /// Checks if a given step matches any rung.
    public func isRung(step: Int) -> Bool {
        guard step >= minResource else { return false }
        var r = rungStep(at: 0)
        while r <= step {
            if r == step {
                return true
            }
            r *= reductionFactor
        }
        return false
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        guard isRung(step: step) else {
            return false
        }

        let allTrials = try study.trials
        let completedOrPruned = allTrials.filter {
            $0.number < trialNumber
                && ($0.state == .complete || $0.state == .pruned)
                && (trialFilter?($0) ?? true)
        }

        let valuesAtRung = completedOrPruned.compactMap { $0.intermediateValues[step] }
        guard valuesAtRung.count >= bootstrapCount && !valuesAtRung.isEmpty else {
            return false
        }

        let numPromoted = max(1, valuesAtRung.count / reductionFactor)

        let sortedValues = study.direction == .minimize
            ? valuesAtRung.sorted(by: <)
            : valuesAtRung.sorted(by: >)

        let cutoffThreshold = sortedValues[numPromoted - 1]

        if study.direction == .minimize {
            return currentValue > cutoffThreshold
        } else {
            return currentValue < cutoffThreshold
        }
    }
}

/// Hyperband pruner managing multiple brackets of SuccessiveHalvingPruner.
///
/// Hyperband addresses the exploration vs. exploitation trade-off by running several
/// ``SuccessiveHalvingPruner`` brackets with varying aggressive early stopping configurations.
/// Trials are deterministically assigned to brackets based on `trialNumber % nBrackets`, ensuring
/// 100% stateless and concurrency-safe bracket partitioning.
///
/// ### Example
/// ```swift
/// let pruner = HyperbandPruner(minResource: 1, maxResource: 81, reductionFactor: 3)
/// let study = try Swiftuna.createStudy(pruner: pruner)
/// ```
public struct HyperbandPruner: Pruner {
    /// Minimum resource allocation (initial rung step).
    public let minResource: Int

    /// Maximum resource allocation cap for the most promising trials.
    public let maxResource: Int

    /// Reduction factor $\eta$ governing rung progression and bracket laddering.
    public let reductionFactor: Int

    /// Minimum number of trials required at each rung before pruning begins.
    public let bootstrapCount: Int

    /// Total number of brackets managed by this Hyperband instance.
    public let nBrackets: Int
    private let pruners: [SuccessiveHalvingPruner]

    /// Initializes a Hyperband pruner.
    ///
    /// - Parameters:
    ///   - minResource: Minimum resource step. Defaults to `1`.
    ///   - maxResource: Maximum resource step. Defaults to `80`.
    ///   - reductionFactor: Resource scaling factor $\eta$. Defaults to `3`.
    ///   - bootstrapCount: Trials required before pruning. Defaults to `0`.
    public init(
        minResource: Int = 1,
        maxResource: Int = 80,
        reductionFactor: Int = 3,
        bootstrapCount: Int = 0
    ) {
        let minR = max(1, minResource)
        let maxR = max(minR, maxResource)
        let eta = max(2, reductionFactor)

        self.minResource = minR
        self.maxResource = maxR
        self.reductionFactor = eta
        self.bootstrapCount = max(0, bootstrapCount)

        var sMax = 0
        var resourceCap = minR
        while resourceCap * eta <= maxR {
            resourceCap *= eta
            sMax += 1
        }
        let totalBrackets = sMax + 1
        self.nBrackets = totalBrackets

        var bracketPruners: [SuccessiveHalvingPruner] = []
        for s in 0..<totalBrackets {
            var minRBracket = minR
            for _ in 0..<s {
                minRBracket *= eta
            }
            bracketPruners.append(
                SuccessiveHalvingPruner(
                    minResource: minRBracket,
                    reductionFactor: eta,
                    minEarlyStoppingRate: s,
                    bootstrapCount: bootstrapCount,
                    trialFilter: { $0.number % totalBrackets == s }
                )
            )
        }
        self.pruners = bracketPruners
    }

    /// Determines the bracket index assigned to a trial.
    public func bracket(for trialNumber: Int) -> Int {
        trialNumber % nBrackets
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        let b = bracket(for: trialNumber)
        let pruner = pruners[b]
        return try pruner.shouldPrune(
            study: study,
            trialNumber: trialNumber,
            step: step,
            currentValue: currentValue
        )
    }
}

import Synchronization

/// Pruner that wraps another pruner to provide a patience grace period, or acts as a standalone
/// early-stopping monitor based on stagnation.
///
/// `PatientPruner` operates in two distinct modes:
/// 1. **Wrapped Mode** (`wrappedPruner != nil`): Suppresses pruning signals from `wrappedPruner` until
///    the underlying pruner votes to prune for `patience` consecutive steps.
/// 2. **Standalone Mode** (`wrappedPruner == nil`): Monitors objective value improvements, pruning the trial
///    if it fails to improve upon its historical best value by at least `minDelta` within `patience` steps.
///
/// ### Example
/// ```swift
/// // Tolerate up to 3 consecutive prune signals from MedianPruner before actually stopping
/// let base = MedianPruner(nStartupTrials: 5)
/// let pruner = PatientPruner(wrappedPruner: base, patience: 3)
/// ```
public struct PatientPruner: Pruner {
    /// The underlying pruner whose prune decisions are delayed. If `nil`, operates in standalone mode.
    public let wrappedPruner: (any Pruner)?

    /// The number of consecutive prune votes or unimproved steps tolerated before pruning triggers.
    public let patience: Int

    /// Minimum absolute change in objective value considered a meaningful improvement (standalone mode only).
    public let minDelta: Double

    private struct TrialState: Sendable {
        var consecutivePruneVotes: Int = 0
        var bestValue: Double? = nil
        var consecutiveUnimproved: Int = 0
    }

    private final class StateBox: Sendable {
        let states = Mutex<[Int: TrialState]>([:])
    }
    private let storage = StateBox()

    /// Initializes a Patient pruner.
    ///
    /// - Parameters:
    ///   - wrappedPruner: Optional base pruner to wrap with patience.
    ///   - patience: Consecutive steps tolerated before pruning triggers.
    ///   - minDelta: Minimum improvement threshold for standalone mode. Defaults to `0.0`.
    public init(
        wrappedPruner: (any Pruner)? = nil,
        patience: Int,
        minDelta: Double = 0.0
    ) {
        self.wrappedPruner = wrappedPruner
        self.patience = max(0, patience)
        self.minDelta = max(0.0, minDelta)
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        if let wrapped = wrappedPruner {
            let baseWantsPrune = try wrapped.shouldPrune(
                study: study,
                trialNumber: trialNumber,
                step: step,
                currentValue: currentValue
            )

            return storage.states.withLock { states in
                var state = states[trialNumber] ?? TrialState()
                if baseWantsPrune {
                    state.consecutivePruneVotes += 1
                    states[trialNumber] = state
                    return state.consecutivePruneVotes > patience
                } else {
                    state.consecutivePruneVotes = 0
                    states[trialNumber] = state
                    return false
                }
            }
        }

        // Standalone patience mode
        return storage.states.withLock { states in
            var state = states[trialNumber] ?? TrialState()
            guard let best = state.bestValue else {
                state.bestValue = currentValue
                state.consecutiveUnimproved = 0
                states[trialNumber] = state
                return false
            }

            let improved: Bool
            if study.direction == .minimize {
                improved = currentValue <= best - minDelta
            } else {
                improved = currentValue >= best + minDelta
            }

            if improved {
                state.bestValue = currentValue
                state.consecutiveUnimproved = 0
                states[trialNumber] = state
                return false
            } else {
                state.consecutiveUnimproved += 1
                states[trialNumber] = state
                return state.consecutiveUnimproved > patience
            }
        }
    }
}
