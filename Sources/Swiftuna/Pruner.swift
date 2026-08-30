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

/// Pruner that prunes if the trial's best intermediate value is worse than the median of
/// intermediate values of previous trials at the same step.
public struct MedianPruner: Pruner {
    public let nStartupTrials: Int
    public let nWarmupSteps: Int
    public let intervalSteps: Int

    public init(
        nStartupTrials: Int = 5,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1
    ) {
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
        let median: Double
        let count = sortedValues.count
        if count % 2 == 1 {
            median = sortedValues[count / 2]
        } else {
            median = (sortedValues[count / 2 - 1] + sortedValues[count / 2]) / 2.0
        }

        if study.direction == .minimize {
            return currentValue > median
        } else {
            return currentValue < median
        }
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
        } else {
            return currentValue < threshold
        }
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
