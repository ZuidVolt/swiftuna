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

/// Pruner that prunes if the trial's best intermediate value is worse than the median (50th percentile)
/// of intermediate values of previous trials at the same step.
public struct MedianPruner: Pruner {
    public let underlying: PercentilePruner

    public var nStartupTrials: Int { underlying.nStartupTrials }
    public var nWarmupSteps: Int { underlying.nWarmupSteps }
    public var intervalSteps: Int { underlying.intervalSteps }

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

/// Pruner that prunes if the trial's intermediate value is worse than the given percentile of previous trials.
public struct PercentilePruner: Pruner {
    public let percentile: Double
    public let nStartupTrials: Int
    public let nWarmupSteps: Int
    public let intervalSteps: Int

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

/// Pruner that prunes immediately if the intermediate value exceeds absolute predefined thresholds.
public struct ThresholdPruner: Pruner {
    public let lower: Double?
    public let upper: Double?

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
/// Allocates resources across rungs scaling by `reductionFactor` (η). At each rung,
/// only the top 1 / η fraction of trials are promoted to continue to the next rung.
public struct SuccessiveHalvingPruner: Pruner {
    public let minResource: Int
    public let reductionFactor: Int
    public let minEarlyStoppingRate: Int
    public let bootstrapCount: Int
    public let trialFilter: (@Sendable (PersistedTrial) -> Bool)?

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

    /// Computes the rung step for index k.
    public func rungStep(at index: Int) -> Int {
        var r = minResource
        for _ in 0..<(minEarlyStoppingRate + index) {
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
/// Deterministically assigns trials to brackets based on `trialNumber % nBrackets`, ensuring
/// 100% stateless and concurrency-safe bracket partitioning.
public struct HyperbandPruner: Pruner {
    public let minResource: Int
    public let maxResource: Int
    public let reductionFactor: Int
    public let bootstrapCount: Int

    public let nBrackets: Int
    private let pruners: [SuccessiveHalvingPruner]

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

/// Pruner that adds patience (grace period) to an underlying pruner, or acts as a standalone
/// early-stopping pruner based on lack of objective improvement.
public struct PatientPruner: Pruner {
    public let wrappedPruner: (any Pruner)?
    public let patience: Int
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
