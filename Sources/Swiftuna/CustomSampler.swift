internal import LibRustuna

/// History snapshot handed to a ``CustomSampler`` on every trial.
///
/// `all` is the full known history (including trials that predate the
/// driver); `new` is the unseen tail since the sampler's last call — fold
/// that instead of re-scanning `all`, or suggest cost grows quadratically.
/// `best` is the best completed trial (direction-aware), precomputed so the
/// common "best-so-far plus perturbation" sampler needs no scan at all.
/// `nil` for multi-objective studies, which have no scalar best.
public struct StudyHistory: Sendable {
    /// Every known trial, oldest first.
    public let all: [PersistedTrial]
    /// Trials unseen since the sampler's last call (usually exactly one).
    /// On the first call there is no last call, so `new == all`: cold-start
    /// samplers see the pre-existing history in full.
    public let new: [PersistedTrial]
    /// Best completed trial, or `nil` when none exists or the study is
    /// multi-objective.
    public let best: PersistedTrial?

    public init(all: [PersistedTrial], newSince count: Int, directions: [Direction]) {
        self.all = all
        self.new = Array(all.suffix(max(0, all.count - count)))
        self.best = Self.best(of: all, directions: directions)
    }

    /// Precomputed best without scanning: the driver's running-best path.
    ///
    /// The driver folds each finished trial into a cached best (O(1) per
    /// trial) instead of rescanning history per snapshot (O(history), which
    /// totals quadratic over a run). Same ordering as the scanning init by
    /// construction — both go through ``best(of:directions:)`` and the fold
    /// below replaces only on strict improvement (first-wins on ties, matching
    /// `min`/`max` with strict predicates).
    package init(all: [PersistedTrial], newSince count: Int, best: PersistedTrial?) {
        self.all = all
        self.new = Array(all.suffix(max(0, all.count - count)))
        self.best = best
    }
}

/// Best-trial ordering shared by the scanning init and the driver's fold.
extension StudyHistory {
    /// Scalar value a trial competes with (direction-aware missing default).
    fileprivate static func bestValue(of trial: PersistedTrial, direction: Direction) -> Double {
        switch direction {
        case .minimize:
            return trial.values.first ?? .infinity
        case .maximize:
            return trial.values.first ?? -.infinity
        }
    }

    /// Strict improvement test. First-wins on ties, matching `min`/`max`
    /// with strict predicates in ``best(of:directions:)``.
    fileprivate static func isBetter(value a: Double, than b: Double, direction: Direction) -> Bool {
        switch direction {
        case .minimize:
            return a < b
        case .maximize:
            return a > b
        }
    }

    /// Best completed trial by scan. Single-objective only; `nil` otherwise.
    /// Predicates are the original init's, factored — not reinterpreted.
    package static func best(of trials: [PersistedTrial], directions: [Direction]) -> PersistedTrial? {
        guard directions.count == 1, let direction = directions.first else { return nil }
        let complete = trials.lazy.filter { $0.state == .complete }
        switch direction {
        case .minimize:
            return complete.min {
                isBetter(
                    value: bestValue(of: $0, direction: direction),
                    than: bestValue(of: $1, direction: direction),
                    direction: direction)
            }
        case .maximize:
            return complete.max {
                bestValue(of: $0, direction: direction) < bestValue(of: $1, direction: direction)
            }
        }
    }

    /// Folds one finished trial into a running best in O(1). Only completed
    /// trials can take the lead; anything else leaves the incumbent alone.
    package static func fold(_ trial: PersistedTrial, into current: PersistedTrial?, directions: [Direction])
        -> PersistedTrial?
    {
        // NOTE: testFoldMatchesScanOnTies pins this to the scanning init's
        // ordering; keep the two in lockstep.
        guard trial.state == .complete,
            directions.count == 1, let direction = directions.first
        else { return current }
        guard let current else { return trial }
        return isBetter(
            value: bestValue(of: trial, direction: direction),
            than: bestValue(of: current, direction: direction),
            direction: direction) ? trial : current
    }
}

/// A custom parameter-suggestion strategy in Swift.
///
/// One method: read `history`, return the next configuration. The driver
/// fixes the returned params ahead of `ask` (typed enqueue, no JSON);
/// params you omit fall back to the study's Rust ``Sampler``, mirroring
/// Optuna's relative/independent split. For per-suggestion control inside
/// the trial (conditional spaces), use ``CallbackSampler`` instead.
///
/// A throwing `sample` aborts the run with the original error: no trial
/// exists yet at sample time, so there is nothing to record, and silent
/// fallback would corrupt the experiment.
///
/// Thread-safety: `sample` runs serially inside the driver's loop today.
/// Keep state in the conforming type (a final class with a lock travels
/// well) rather than relying on call order from multiple drivers.
///
/// ### Example
/// ```swift
/// struct HillClimb: CustomSampler {
///     let step: Double
///     func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
///         guard let bx = history.best?.params["x"]?.asDouble else {
///             return ["x": .double(Double.random(in: -10.0...10.0))]
///         }
///         return ["x": .double(min(10.0, max(-10.0, bx + Double.random(in: -step...step))))]
///     }
/// }
/// try study.optimize(nTrials: 50, using: HillClimb(step: 1.0)) { trial in
///     let x = try trial.suggest("x", in: -10.0...10.0)
///     return x * x
/// }
/// ```
public protocol CustomSampler: Sendable {
    /// Whether the optimization driver should retain full parameter dictionaries in the Swift `StudyHistory` buffer.
    ///
    /// When `true` (the default), each historical trial in `StudyHistory` retains its full hyperparameter dictionary.
    /// When `false` (e.g. for algorithms like `CMASampler` that track their own candidate vectors or only need objective values),
    /// the driver stores an empty parameter dictionary in the history buffer, reducing memory usage by ~50%
    /// at large trial counts while Rustuna's storage still retains all parameters.
    var retainsParameterHistory: Bool { get }

    /// Whether the sampler has completely exhausted its search space.
    ///
    /// When `true`, ``Study/ask()`` and the optimization loop terminate early with
    /// ``SwiftunaError/searchSpaceExhausted(_:)``. Defaults to `false`.
    var isExhausted: Bool { get }

    /// Underlying Rustuna ``Sampler`` used for background trial checkout.
    ///
    /// Defaults to `self as? (any Sampler)` if conforming, or `nil` (which defaults to `RandomSampler`).
    var underlyingSampler: (any Sampler)? { get }

    /// Proposes the next trial configuration from history.
    ///
    /// - Parameters:
    ///   - history: Full and incremental history plus the precomputed best.
    ///   - trialNumber: Zero-based index of the trial being configured.
    /// - Returns: Fixed parameter values. Omitted params are Rust-sampled.
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue]
}

extension CustomSampler {
    public var retainsParameterHistory: Bool { true }
    public var isExhausted: Bool { false }
    public var underlyingSampler: (any Sampler)? { self as? (any Sampler) }
}

/// A custom suggestion closure: the function form of ``CustomSampler``.
public typealias CustomSuggestClosure = @Sendable (StudyHistory, Int) throws -> [String: ParameterValue]

/// Adapts a closure to ``CustomSampler`` so drivers implement one loop.
private struct ClosureCustomSampler: CustomSampler {
    let body: CustomSuggestClosure
    func sample(history: StudyHistory, trialNumber: Int) throws -> [String: ParameterValue] {
        try body(history, trialNumber)
    }
}

/// Classifies a caught error without tripping the ownership verifier.
///
/// Kept out of line: `as?` casts combined with consuming trial handles in
/// generic throwing contexts crash SIL verification (Xcode 16 beta), so the
/// borrow lives here, far from any `consume`.
private func isTrialPruned(_ error: any Error) -> Bool {
    guard let serr = error as? SwiftunaError else { return false }
    if case .trialPruned = serr { return true }
    return false
}

/// See ``isTrialPruned(_:)``.
private func isSearchSpaceExhausted(_ error: any Error) -> Bool {
    guard let serr = error as? SwiftunaError else { return false }
    if case .searchSpaceExhausted = serr { return true }
    return false
}

extension Study {
    /// Optimizes with a custom Swift sampler: suggest, fix, evaluate, record.
    ///
    /// Each iteration reads history (accumulated locally, O(1) amortized —
    /// no refetch; best folded incrementally, never rescanned), asks the
    /// sampler, atomically fixes and checks out the
    /// trial (`askEnqueued`, safe across drivers sharing the study),
    /// evaluates, and records. History params are exact: the sampler's fixed
    /// dict merged over everything the objective actually suggested
    /// (including Rust-sampled rest), so partial fixing stays truthful.
    ///
    /// Failed objectives record `.fail` like
    /// ``optimize(nTrials:timeout:objective:)-3gyl5``; sampler throws abort
    /// the run with the original error — no trial exists yet at suggest
    /// time, so there is nothing to record, and silent fallback would
    /// corrupt the experiment.
    ///
    /// - Parameters:
    ///   - nTrials: Maximum trials. At least one of `nTrials`/`timeout` required.
    ///   - timeout: Maximum duration.
    ///   - sampler: Custom suggestion strategy.
    ///   - objective: Evaluates a checked-out trial, one value per direction.
    ///     Generic over the thrown error, so user objectives keep their own
    ///     error types (mirroring the generic `optimize` overload).
    public func optimize<E: Error>(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        using sampler: any CustomSampler,
        objective: (inout Trial) throws(E) -> [Double]
    ) throws {
        let budget = try OptimizationBudget(nTrials: nTrials, timeout: timeout)
        let clock = ContinuousClock()
        var history = try trials
        var seen = history.count
        var iteration = 0
        // One scan up front; the fold at the bottom of the loop keeps this
        // current in O(1) per trial, so snapshots never rescan.
        var runningBest = StudyHistory.best(of: history, directions: directions)

        let retainsParams = sampler.retainsParameterHistory

        while !budget.shouldStop(submitted: iteration) {
            // Snapshot lives only for the sample call: `all` shares the
            // array buffer with `history`, and sharing it past this scope
            // would force a full copy on the append below (CoW) — O(history)
            // per trial, quadratic over the run. A sampler that retains
            // history pays that copy itself, once, by choice.
            let fixed: [String: ParameterValue]
            do {
                let snap = StudyHistory(all: history, newSince: seen, best: runningBest)
                fixed = try sampler.sample(history: snap, trialNumber: history.count)
            } catch {
                if isSearchSpaceExhausted(error) { break }
                throw error
            }
            // Mark everything seen *before* suggesting: the next call's `new`
            // is exactly what completed since this one.
            seen = history.count
            // Abort loudly: nothing to record, original error preserved.

            let trial: Trial
            do {
                trial = try askEnqueued(fixed)
            } catch {
                if isSearchSpaceExhausted(error) { break }
                throw error
            }
            iteration += 1
            var activeTrial = trial
            activeTrial.trackSuggestions = true
            let trialNum = activeTrial.number
            let startTime = clock.now
            let span = SwiftunaTelemetry.shared.trialSpan(
                study: name, trialNumber: trialNum, distributed: false)
            activeTrial.telemetrySpan = span

            // Straight-line complete path; the catch below covers only the
            // objective, where the trial is provably live (no consume has
            // run), so partial suggestions merge freely there.
            let vals: [Double]
            do {
                vals = try objective(&activeTrial)
            } catch {
                try handleObjectiveError(
                    error,
                    trial: activeTrial,
                    trialNum: trialNum,
                    fixed: fixed,
                    span: span,
                    history: &history,
                    retainsParams: retainsParams
                )
                continue
            }

            try recordCompletedTrial(
                trial: activeTrial,
                trialNum: trialNum,
                vals: vals,
                fixed: fixed,
                startTime: startTime,
                clock: clock,
                span: span,
                history: &history,
                runningBest: &runningBest,
                retainsParams: retainsParams
            )
        }
    }

    /// Single-objective custom-sampler optimization.
    public func optimize<E: Error>(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        using sampler: any CustomSampler,
        objective: (inout Trial) throws(E) -> Double
    ) throws {
        var caughtError: E?
        try optimize(nTrials: nTrials, timeout: timeout, using: sampler) {
            (trial: inout Trial) throws(SwiftunaError) -> [Double] in
            do {
                return [try objective(&trial)]
            } catch let err as SwiftunaError {
                throw err
            } catch let err as E {
                caughtError = err
                throw SwiftunaError.objectiveError("\(err)")
            } catch {
                throw SwiftunaError.objectiveError("\(error)")
            }
        }
        if let err = caughtError {
            throw err
        }
    }

    /// Multi-objective custom-sampler optimization from a closure.
    public func optimize<E: Error>(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        using suggest: @escaping CustomSuggestClosure,
        objective: (inout Trial) throws(E) -> [Double]
    ) throws {
        try optimize(
            nTrials: nTrials, timeout: timeout,
            using: ClosureCustomSampler(body: suggest), objective: objective)
    }

    /// Single-objective custom-sampler optimization from a closure.
    public func optimize<E: Error>(
        nTrials: Int? = nil,
        timeout: Duration? = nil,
        using suggest: @escaping CustomSuggestClosure,
        objective: (inout Trial) throws(E) -> Double
    ) throws {
        try optimize(
            nTrials: nTrials, timeout: timeout,
            using: ClosureCustomSampler(body: suggest),
            objective: objective
        )
    }

    // MARK: - Inlined Pipeline Helpers

    @inline(always)
    private func handleObjectiveError(
        _ error: any Error,
        trial: consuming Trial,
        trialNum: Int,
        fixed: [String: ParameterValue],
        span: (any TelemetrySpan)?,
        history: inout [PersistedTrial],
        retainsParams: Bool
    ) throws {
        let partial = fixed.merging(trial.suggestedParams) { _, new in new }
        for (paramName, paramValue) in partial {
            span?.setAttribute("param.\(paramName)", value: paramValue.telemetryAttribute)
        }
        let isPruned = isTrialPruned(error)
        let state: TrialState = isPruned ? .pruned : .fail
        span?.setAttribute("trial.status", value: isPruned ? "pruned" : "failed")
        if isPruned {
            span?.end(status: .ok)
        } else {
            span?.end(status: .error(String(describing: error)))
        }
        try tell(consuming: trial, values: [], state: state)
        history.append(
            PersistedTrial(
                number: trialNum, state: state, value: nil, params: retainsParams ? partial : [:]))
        if !isPruned {
            throw error
        }
    }

    @inline(always)
    private func recordCompletedTrial(
        trial: consuming Trial,
        trialNum: Int,
        vals: [Double],
        fixed: [String: ParameterValue],
        startTime: ContinuousClock.Instant,
        clock: ContinuousClock,
        span: (any TelemetrySpan)?,
        history: inout [PersistedTrial],
        runningBest: inout PersistedTrial?,
        retainsParams: Bool
    ) throws {
        let recorded = fixed.merging(trial.suggestedParams) { _, new in new }
        let elapsed = clock.now - startTime
        for (paramName, paramValue) in recorded {
            span?.setAttribute("param.\(paramName)", value: paramValue.telemetryAttribute)
        }
        span?.setAttribute("trial.status", value: "complete")
        span?.setAttribute(
            "trial.duration_ms",
            value: .double(Double(elapsed.components.attoseconds) / 1e15))
        span?.end(status: .ok)
        try tell(consuming: trial, values: vals, state: .complete)

        let finishedHistory = PersistedTrial(
            number: trialNum, state: .complete, value: nil, values: vals,
            params: retainsParams ? recorded : [:])
        history.append(finishedHistory)

        if directions.count == 1, let direction = directions.first {
            let trialVal = vals.first ?? 0.0
            let isLead: Bool
            if let current = runningBest {
                let curVal = StudyHistory.bestValue(of: current, direction: direction)
                isLead = StudyHistory.isBetter(value: trialVal, than: curVal, direction: direction)
            } else {
                isLead = true
            }
            if isLead {
                runningBest =
                    retainsParams
                    ? finishedHistory
                    : PersistedTrial(
                        number: trialNum, state: .complete, value: nil, values: vals, params: recorded)
            }
        }
    }
}
