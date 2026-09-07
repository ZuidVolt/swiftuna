import Synchronization

/// Specifies resource allocation bounds (e.g. `minResource` or `maxResource`) for rungs in early-stopping algorithms.
///
/// Supports automatic heuristic inference from initial completed trials (`.auto`) or explicit step bounds (`.step(Int)`).
public enum ResourceBound: ExpressibleByIntegerLiteral, Sendable, Equatable {
    /// Automatically infers the resource bound dynamically based on the maximum step observed in completed trials.
    case auto
    /// An explicit static step bound.
    case step(Int)

    public init(integerLiteral value: Int) {
        self = .step(value)
    }

    /// Returns the explicit step value, or `nil` if `.auto`.
    public var stepValue: Int? {
        switch self {
        case .auto: return nil
        case .step(let s): return s
        }
    }
}

/// Checks whether `step` is the first reported step in the current interval bucket `[warmup + k * interval, warmup + (k+1) * interval)`.
@inline(always)
internal func isFirstInIntervalStep(
    step: Int,
    reportedSteps: some Collection<Int>,
    warmup: Int,
    interval: Int
) -> Bool {
    guard step >= warmup else { return false }
    let bucket = (step - warmup) / interval
    let minInBucket = warmup + bucket * interval
    return !reportedSteps.contains { $0 >= minInBucket && $0 < step }
}

/// Standard IEEE 802.3 CRC32 hashing replicating Python's `binascii.crc32` exactly.
internal func optunaCRC32(_ string: String) -> UInt32 {
    var crc: UInt32 = 0xFFFF_FFFF
    for byte in string.utf8 {
        crc ^= UInt32(byte)
        for _ in 0..<8 {
            let mask = (crc & 1) != 0 ? UInt32(0xEDB8_8320) : 0
            crc = (crc >> 1) ^ mask
        }
    }
    return ~crc
}

/// Protocol for deciding whether an active trial should be early-stopped based on intermediate values.
public protocol Pruner: Sendable {
    /// Evaluates whether the given trial should be pruned at `step`.
    func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool
}

extension Pruner {
    /// Backwards-compatible evaluation without explicit intermediate history.
    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double
    ) throws(SwiftunaError) -> Bool {
        try shouldPrune(
            study: study,
            trialNumber: trialNumber,
            step: step,
            currentValue: currentValue,
            intermediateValues: [step: currentValue]
        )
    }
}

/// No-operation pruner that never early-stops trials (default).
public struct NopPruner: Pruner {
    public init() {}

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) -> Bool {
        false
    }
}

/// Pruner using the median stopping rule.
///
/// Prunes an active trial if its best intermediate value up to the current step is worse than the median (50th percentile)
/// of intermediate values reported by previous completed trials at the same step.
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

    /// Minimum number of reported trials at a step required before pruning decisions take effect.
    public var nMinTrials: Int { underlying.nMinTrials }

    /// Initializes a Median pruner.
    ///
    /// - Parameters:
    ///   - nStartupTrials: Trials run before pruning starts. Defaults to `5`.
    ///   - nWarmupSteps: Steps within a trial before pruning starts. Defaults to `0`.
    ///   - intervalSteps: Step frequency for evaluating pruning. Defaults to `1`.
    ///   - nMinTrials: Minimum completed trials required at a step before pruning. Defaults to `1`.
    public init(
        nStartupTrials: Int = 5,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1,
        nMinTrials: Int = 1
    ) {
        self.underlying = PercentilePruner(
            percentile: 50.0,
            nStartupTrials: nStartupTrials,
            nWarmupSteps: nWarmupSteps,
            intervalSteps: intervalSteps,
            nMinTrials: nMinTrials
        )
    }

    @inline(always)
    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool {
        try underlying.shouldPrune(
            study: study,
            trialNumber: trialNumber,
            step: step,
            currentValue: currentValue,
            intermediateValues: intermediateValues
        )
    }
}

/// Pruner to keep trials whose best intermediate values fall in the top percentile of historical trials.
///
/// Prunes an active trial if its best intermediate value up to the current step is worse than the given `percentile`
/// among previous completed trials at the same step.
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

    /// Minimum number of completed trials required at a step before pruning is evaluated.
    public let nMinTrials: Int

    /// Initializes a Percentile pruner.
    ///
    /// - Parameters:
    ///   - percentile: Percentile threshold between 0.0 and 100.0 (e.g. `25.0` for top 25%).
    ///   - nStartupTrials: Trials run before pruning starts. Defaults to `5`.
    ///   - nWarmupSteps: Steps within each trial before pruning starts. Defaults to `0`.
    ///   - intervalSteps: Step frequency for evaluating pruning. Defaults to `1`.
    ///   - nMinTrials: Minimum completed trials at a step required to judge pruning. Defaults to `1`.
    public init(
        percentile: Double = 50.0,
        nStartupTrials: Int = 5,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1,
        nMinTrials: Int = 1
    ) {
        precondition(
            percentile >= 0.0 && percentile <= 100.0, "Percentile must be between 0.0 and 100.0, got \(percentile).")
        precondition(nStartupTrials >= 0, "nStartupTrials must be non-negative, got \(nStartupTrials).")
        precondition(nWarmupSteps >= 0, "nWarmupSteps must be non-negative, got \(nWarmupSteps).")
        precondition(intervalSteps >= 1, "intervalSteps must be at least 1, got \(intervalSteps).")
        precondition(nMinTrials >= 1, "nMinTrials must be at least 1, got \(nMinTrials).")

        self.percentile = percentile
        self.nStartupTrials = nStartupTrials
        self.nWarmupSteps = nWarmupSteps
        self.intervalSteps = intervalSteps
        self.nMinTrials = nMinTrials
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool {
        if step < nWarmupSteps {
            return false
        }
        if !isFirstInIntervalStep(
            step: step, reportedSteps: intermediateValues.keys, warmup: nWarmupSteps, interval: intervalSteps)
        {
            return false
        }

        // Determine the trial's best intermediate result over all steps up to `step`
        let bestIntermediateResult: Double
        if study.direction == .minimize {
            bestIntermediateResult = intermediateValues.values.min() ?? currentValue
        } else {
            bestIntermediateResult = intermediateValues.values.max() ?? currentValue
        }

        if bestIntermediateResult.isNaN {
            return true
        }

        // Fetch completed trials across CFFI using state mask to avoid serializing pending/failed/running trials
        let completedTrials = try study.trials(where: [.complete])
        let previousTrials = completedTrials.filter { $0.number < trialNumber }

        if previousTrials.count < nStartupTrials {
            return false
        }

        let valuesAtStep: [Double] = previousTrials.compactMap { $0.intermediateValues[step] }.filter { !$0.isNaN }
        guard valuesAtStep.count >= nMinTrials else {
            return false
        }

        let sortedValues = valuesAtStep.sorted()
        let effectivePercentile = study.direction == .maximize ? (100.0 - percentile) : percentile
        let threshold: Double
        if sortedValues.count == 1 {
            threshold = sortedValues[0]
        } else {
            let virtualIdx = Double(sortedValues.count - 1) * (effectivePercentile / 100.0)
            let low = Int(virtualIdx.rounded(.down))
            let high = Int(virtualIdx.rounded(.up))
            let frac = virtualIdx - Double(low)
            threshold = sortedValues[low] + frac * (sortedValues[high] - sortedValues[low])
        }

        if study.direction == .minimize {
            return bestIntermediateResult > threshold
        }
        return bestIntermediateResult < threshold
    }
}

/// Pruner that prunes immediately if an intermediate value crosses absolute predefined thresholds.
///
/// Evaluates whether the reported value exceeds `upper` or drops below `lower`. Also prunes `NaN` evaluations.
/// Supports warmup step suppression and interval checking matching Python Optuna.
///
/// ### Example
/// ```swift
/// // Prune immediately if loss exceeds 100.0 or drops below 0.0 after 5 warmup steps
/// let pruner = ThresholdPruner(lower: 0.0, upper: 100.0, nWarmupSteps: 5)
/// ```
public struct ThresholdPruner: Pruner {
    /// Lower bound threshold. If an intermediate value is `< lower`, the trial is pruned.
    public let lower: Double?

    /// Upper bound threshold. If an intermediate value is `> upper`, the trial is pruned.
    public let upper: Double?

    /// Number of initial steps within each trial before pruning evaluation begins.
    public let nWarmupSteps: Int

    /// Step interval at which pruning decisions are evaluated.
    public let intervalSteps: Int

    /// Initializes a Threshold pruner.
    ///
    /// - Parameters:
    ///   - lower: Optional lower bound cutoff.
    ///   - upper: Optional upper bound cutoff.
    ///   - nWarmupSteps: Initial steps before pruning begins. Defaults to `0`.
    ///   - intervalSteps: Step interval between checks. Defaults to `1`.
    public init(
        lower: Double? = nil,
        upper: Double? = nil,
        nWarmupSteps: Int = 0,
        intervalSteps: Int = 1
    ) {
        precondition(lower != nil || upper != nil, "Either lower or upper threshold must be specified.")
        if let lower, let upper {
            precondition(
                lower <= upper, "Lower threshold (\(lower)) must be less than or equal to upper threshold (\(upper)).")
        }
        precondition(nWarmupSteps >= 0, "nWarmupSteps must be non-negative, got \(nWarmupSteps).")
        precondition(intervalSteps >= 1, "intervalSteps must be at least 1, got \(intervalSteps).")

        self.lower = lower
        self.upper = upper
        self.nWarmupSteps = nWarmupSteps
        self.intervalSteps = intervalSteps
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) -> Bool {
        if step < nWarmupSteps {
            return false
        }
        if !isFirstInIntervalStep(
            step: step, reportedSteps: intermediateValues.keys, warmup: nWarmupSteps, interval: intervalSteps)
        {
            return false
        }
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
/// let pruner = SuccessiveHalvingPruner(minResource: .auto, reductionFactor: 4)
/// let study = try Swiftuna.createStudy(pruner: pruner)
/// ```
public struct SuccessiveHalvingPruner: Pruner {
    /// Minimum resource allocation bound before trials encounter the first rung (supports `.auto` or explicit `.step(Int)`).
    public let minResource: ResourceBound

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
    ///   - minResource: Minimum resource bound before the first rung (defaults to `.auto` or integer literal).
    ///   - reductionFactor: Promotion divisor $\eta$. Defaults to `4`.
    ///   - minEarlyStoppingRate: Initial rung rate exponent. Defaults to `0`.
    ///   - bootstrapCount: Trials needed at a rung before pruning begins. Defaults to `0`.
    ///   - trialFilter: Optional closure to isolate trials by bracket.
    public init(
        minResource: ResourceBound = .step(1),
        reductionFactor: Int = 4,
        minEarlyStoppingRate: Int = 0,
        bootstrapCount: Int = 0,
        trialFilter: (@Sendable (PersistedTrial) -> Bool)? = nil
    ) {
        precondition(
            bootstrapCount == 0 || minResource != .auto,
            "bootstrapCount > 0 and minResource == .auto are mutually incompatible.")
        precondition(reductionFactor >= 2, "reductionFactor must be at least 2, got \(reductionFactor).")
        precondition(
            minEarlyStoppingRate >= 0, "minEarlyStoppingRate must be non-negative, got \(minEarlyStoppingRate).")
        precondition(bootstrapCount >= 0, "bootstrapCount must be non-negative, got \(bootstrapCount).")
        if let s = minResource.stepValue {
            precondition(s >= 1, "minResource must be at least 1, got \(s).")
        }

        self.minResource = minResource
        self.reductionFactor = reductionFactor
        self.minEarlyStoppingRate = minEarlyStoppingRate
        self.bootstrapCount = bootstrapCount
        self.trialFilter = trialFilter
    }

    /// Computes the step index for the given rung index $k$ using an explicit or resolved minimum resource.
    public func rungStep(at index: Int, effectiveMinResource: Int? = nil) -> Int {
        var r = effectiveMinResource ?? minResource.stepValue ?? 1
        let totalRate = minEarlyStoppingRate + index
        for _ in 0..<totalRate {
            r *= reductionFactor
        }
        return r
    }

    /// Checks if a given step matches any rung for the given minimum resource.
    public func isRung(step: Int, effectiveMinResource: Int? = nil) -> Bool {
        let baseMin = effectiveMinResource ?? minResource.stepValue ?? 1
        guard step >= baseMin else { return false }
        var r = rungStep(at: 0, effectiveMinResource: baseMin)
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
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool {
        let completedTrials = try study.trials(where: [.complete])

        // Resolve effective minimum resource
        let effectiveMin: Int
        if let explicit = minResource.stepValue {
            effectiveMin = explicit
        } else {
            let maxSteps = completedTrials.compactMap { $0.intermediateValues.keys.max() }
            guard let maxStep = maxSteps.max() else {
                return false  // No completed trials yet to estimate minResource
            }
            effectiveMin = max(maxStep / 100, 1)
        }

        guard isRung(step: step, effectiveMinResource: effectiveMin) else {
            return false
        }

        if currentValue.isNaN {
            return true
        }

        let competingTrials = completedTrials.filter {
            $0.number < trialNumber && (trialFilter?($0) ?? true)
        }

        var competingValues = competingTrials.compactMap { $0.intermediateValues[step] }.filter { !$0.isNaN }
        competingValues.append(currentValue)

        if competingValues.count <= bootstrapCount {
            return true
        }

        var promotableIdx = (competingValues.count / reductionFactor) - 1
        if promotableIdx == -1 {
            promotableIdx = 0
        }

        competingValues.sort()
        if study.direction == .maximize {
            return currentValue < competingValues[competingValues.count - 1 - promotableIdx]
        }
        return currentValue > competingValues[promotableIdx]
    }
}

/// Encapsulates precomputed Hyperband bracket ladder schedules and CRC32 bracket assignments.
internal struct HyperbandLadder: Sendable {
    let totalBrackets: Int
    let budgets: [Int]
    let totalBudget: Int

    init(minResource: Int, maxResource: Int, reductionFactor: Int) {
        var sMax = 0
        var resourceCap = minResource
        while resourceCap * reductionFactor <= maxResource {
            resourceCap *= reductionFactor
            sMax += 1
        }
        let total = sMax + 1
        self.totalBrackets = total

        var budgetsList: [Int] = []
        var totalB = 0
        for bracketId in 0..<total {
            let s = total - 1 - bracketId
            var etaPowS = 1
            for _ in 0..<s { etaPowS *= reductionFactor }
            let b = (total * etaPowS + s) / (s + 1)
            budgetsList.append(b)
            totalB += b
        }
        self.budgets = budgetsList
        self.totalBudget = totalB
    }

    func bracket(for trialNumber: Int, studyName: String?) -> Int {
        if let studyName, totalBudget > 0 {
            var n = Int(optunaCRC32("\(studyName)_\(trialNumber)") % UInt32(totalBudget))
            for (idx, b) in budgets.enumerated() {
                n -= b
                if n < 0 {
                    return idx
                }
            }
        }
        return trialNumber % totalBrackets
    }
}

/// Hyperband pruner managing multiple brackets of SuccessiveHalvingPruner.
///
/// Hyperband addresses the exploration vs. exploitation trade-off by running several
/// ``SuccessiveHalvingPruner`` brackets with varying aggressive early stopping configurations.
/// Supports `.auto` maximum resource detection based on initial completed trials.
///
/// ### Example
/// ```swift
/// let pruner = HyperbandPruner(minResource: 1, maxResource: .auto, reductionFactor: 3)
/// let study = try Swiftuna.createStudy(pruner: pruner)
/// ```
public struct HyperbandPruner: Pruner {
    /// Minimum resource allocation (initial rung step).
    public let minResource: Int

    /// Maximum resource allocation cap for the most promising trials (supports `.auto` or explicit `.step(Int)`).
    public let maxResource: ResourceBound

    /// Reduction factor $\eta$ governing rung progression and bracket laddering.
    public let reductionFactor: Int

    /// Minimum number of trials required at each rung before pruning begins.
    public let bootstrapCount: Int

    /// Precomputed static bracket schedule when `maxResource` is explicit.
    internal let staticLadder: HyperbandLadder?

    /// Initializes a Hyperband pruner.
    ///
    /// - Parameters:
    ///   - minResource: Minimum resource step. Defaults to `1`.
    ///   - maxResource: Maximum resource step bound. Defaults to `80` (or `.auto`).
    ///   - reductionFactor: Resource scaling factor $\eta$. Defaults to `3`.
    ///   - bootstrapCount: Trials required before pruning. Defaults to `0`.
    public init(
        minResource: Int = 1,
        maxResource: ResourceBound = 80,
        reductionFactor: Int = 3,
        bootstrapCount: Int = 0
    ) {
        precondition(
            bootstrapCount == 0 || maxResource != .auto,
            "bootstrapCount > 0 and maxResource == .auto are mutually incompatible.")
        precondition(minResource >= 1, "minResource must be at least 1, got \(minResource).")
        precondition(reductionFactor >= 2, "reductionFactor must be at least 2, got \(reductionFactor).")
        precondition(bootstrapCount >= 0, "bootstrapCount must be non-negative, got \(bootstrapCount).")
        if let m = maxResource.stepValue {
            precondition(m >= minResource, "maxResource (\(m)) must be >= minResource (\(minResource)).")
            self.staticLadder = HyperbandLadder(minResource: minResource, maxResource: m, reductionFactor: reductionFactor)
        } else {
            self.staticLadder = nil
        }

        self.minResource = minResource
        self.maxResource = maxResource
        self.reductionFactor = reductionFactor
        self.bootstrapCount = bootstrapCount
    }

    /// Total number of brackets managed by this Hyperband instance, or `nil` if `maxResource` is `.auto` and uninitialized.
    public var nBrackets: Int? {
        staticLadder?.totalBrackets
    }

    /// Determines the bracket index assigned to a trial for a given or resolved total bracket count.
    public func bracket(for trialNumber: Int, studyName: String? = nil, nBrackets: Int? = nil) -> Int {
        if let staticLadder {
            return staticLadder.bracket(for: trialNumber, studyName: studyName)
        }
        guard let total = nBrackets, total > 0 else {
            return 0
        }
        return trialNumber % total
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool {
        let ladder: HyperbandLadder
        if let staticLadder {
            ladder = staticLadder
        } else {
            let completed = try study.trials(where: [.complete])
            let maxSteps = completed.compactMap { $0.intermediateValues.keys.max() }
            guard let maxStep = maxSteps.max() else {
                return false  // No completed trials yet to estimate maxResource
            }
            ladder = HyperbandLadder(
                minResource: minResource,
                maxResource: maxStep + 1,
                reductionFactor: reductionFactor
            )
        }

        let b = ladder.bracket(for: trialNumber, studyName: study.name)
        let capturedLadder = ladder
        let capturedStudyName = study.name
        let pruner = SuccessiveHalvingPruner(
            minResource: .step(minResource),
            reductionFactor: reductionFactor,
            minEarlyStoppingRate: b,
            bootstrapCount: bootstrapCount
        ) { trial in
            capturedLadder.bracket(for: trial.number, studyName: capturedStudyName) == b
        }
        return try pruner.shouldPrune(
            study: study,
            trialNumber: trialNumber,
            step: step,
            currentValue: currentValue,
            intermediateValues: intermediateValues
        )
    }
}

/// Pruner that monitors intermediate values and prunes if the improvement in intermediate values
/// after a patience period is less than a threshold.
///
/// In wrapped mode (`wrappedPruner != nil`), pruning signals from the underlying pruner are gated
/// by stagnation: trials that continue making progress by at least `minDelta` are never pruned.
///
/// In standalone mode (`wrappedPruner == nil`), the trial is early-stopped as soon as it stagnates
/// for `patience` consecutive steps.
///
/// `PatientPruner` is completely stateless over `intermediateValues`, eliminating lock contention across workers.
///
/// ### Example
/// ```swift
/// // Tolerate up to 3 unimproved steps before delegating to MedianPruner
/// let base = MedianPruner(nStartupTrials: 5)
/// let pruner = PatientPruner(wrappedPruner: base, patience: 3)
/// ```
public struct PatientPruner: Pruner {
    /// The underlying pruner whose prune decisions are gated by stagnation. If `nil`, operates in standalone mode.
    public let wrappedPruner: (any Pruner)?

    /// Number of consecutive unimproved steps tolerated before pruning triggers or delegates.
    public let patience: Int

    /// Minimum absolute change in objective value considered a meaningful improvement.
    public let minDelta: Double

    /// Initializes a Patient pruner.
    ///
    /// - Parameters:
    ///   - wrappedPruner: Optional base pruner to wrap with patience.
    ///   - patience: Consecutive steps tolerated before pruning triggers.
    ///   - minDelta: Minimum improvement threshold. Defaults to `0.0`.
    public init(
        wrappedPruner: (any Pruner)? = nil,
        patience: Int,
        minDelta: Double = 0.0
    ) {
        precondition(patience >= 0, "patience must be non-negative, got \(patience).")
        precondition(minDelta >= 0.0, "minDelta must be non-negative, got \(minDelta).")
        self.wrappedPruner = wrappedPruner
        self.patience = patience
        self.minDelta = minDelta
    }

    public func shouldPrune(
        study: Study,
        trialNumber: Int,
        step: Int,
        currentValue: Double,
        intermediateValues: [Int: Double]
    ) throws(SwiftunaError) -> Bool {
        let sortedEntries = intermediateValues.sorted { $0.key < $1.key }
        // Do not prune if number of steps to determine are insufficient (matching Optuna: steps.count <= patience + 1)
        if sortedEntries.count <= patience + 1 {
            return false
        }

        let splitIdx = sortedEntries.count - patience - 1
        let scoresBefore = sortedEntries[..<splitIdx].lazy.map(\.value).filter { !$0.isNaN }
        let scoresAfter = sortedEntries[splitIdx...].lazy.map(\.value).filter { !$0.isNaN }

        guard let minOrMaxBefore = (study.direction == .minimize ? scoresBefore.min() : scoresBefore.max()),
            let minOrMaxAfter = (study.direction == .minimize ? scoresAfter.min() : scoresAfter.max())
        else {
            return false
        }

        let maybePrune: Bool
        if study.direction == .minimize {
            maybePrune = minOrMaxBefore + minDelta < minOrMaxAfter
        } else {
            maybePrune = minOrMaxBefore - minDelta > minOrMaxAfter
        }

        if maybePrune {
            if let wrapped = wrappedPruner {
                return try wrapped.shouldPrune(
                    study: study,
                    trialNumber: trialNumber,
                    step: step,
                    currentValue: currentValue,
                    intermediateValues: intermediateValues
                )
            }
            return true
        }
        return false
    }
}
