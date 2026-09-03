public import Distributed
import Foundation
public import Swiftuna
internal import LibRustuna

/// A distributed actor that coordinates hyperparameter optimization across remote workers.
///
/// Workers call ``ask()`` to retrieve configurations, ``report(trialNumber:step:value:)`` to stream
/// metrics and receive early-stopping recommendations, and ``tell(_:)`` to record final outcomes.
public distributed actor StudyCoordinator<ActorSystem> where ActorSystem: DistributedActorSystem<any Codable> {
    private let study: Study
    private let searchSpace: SearchSpace
    private var inFlight: [Int: InFlightTrial] = [:]
    private var completedCount: Int = 0
    private var finishedCount: Int = 0
    private let maxInFlight: Int
    private let leasePolicy: LeasePolicy?
    private var finishedNumbers: Set<Int> = []
    private var expiredNumbers: Set<Int> = []

    /// Creates a coordinator for `study`, sampling from `searchSpace`.
    ///
    /// The coordinator assumes exclusive access to `study` while trials are in
    /// flight: touching the study directly (e.g. `study.trials`, `study.ask()`)
    /// concurrently with coordinator operations races on the same storage
    /// handle. Compile-time enforcement is not expressible in Swift 6
    /// (`Study` is a shared reference type, so neither `consuming` nor
    /// `~Escapable` can prevent aliasing), so this is a documented contract.
    /// Read through ``trials()`` / ``bestTrial()`` / ``bestTrials()`` instead.
    /// Direct reads are safe again once ``inFlightCount()`` returns zero.
    ///
    /// - Parameters:
    ///   - maxInFlight: Maximum trials handed out concurrently. `ask()`
    ///     throws ``SwiftunaDistributedError/tooManyInFlight(_:)`` past this limit.
    ///     Defaults to `Int.max` (unbounded, matching pre-cap behavior).
    ///     Values below 1 are clamped to 1.
    ///   - leasePolicy: Trial lease policy. `nil` (default) disables leases,
    ///     preserving exact pre-lease behavior.
    public init(
        study: Study,
        searchSpace: SearchSpace,
        actorSystem: ActorSystem,
        maxInFlight: Int = .max,
        leasePolicy: LeasePolicy? = nil
    ) {
        self.study = study
        self.searchSpace = searchSpace
        self.maxInFlight = max(1, maxInFlight)
        self.leasePolicy = leasePolicy
        self.actorSystem = actorSystem
    }

    /// Runs `body` against the study, wrapping any failure as a study error.
    private func wrap<T>(_ body: () throws -> T) throws -> T {
        do {
            return try body()
        } catch {
            throw SwiftunaDistributedError.studyError(String(describing: error))
        }
    }

    /// Dispatches the next hyperparameter configuration to an active worker.
    ///
    /// - Throws: ``SwiftunaDistributedError/tooManyInFlight(_:)`` when the
    ///   in-flight cap is reached, or ``SwiftunaDistributedError/searchSpaceExhausted``
    ///   when a discrete sampler (e.g. ``GridSampler``) has no untried configurations left.
    public distributed func ask() throws -> DistributedTrialSpec {
        reapExpired()
        guard inFlight.count < maxInFlight else {
            throw SwiftunaDistributedError.tooManyInFlight(inFlight.count)
        }
        do {
            var trial = try study.ask()
            let params = try searchSpace.sample(trial: &trial)
            let trialNumber = trial.number
            guard let handle = trial.takeHandle() else {
                throw SwiftunaDistributedError.studyError("Failed to take handle for trial #\(trialNumber)")
            }
            inFlight[trialNumber] = InFlightTrial(rawHandle: handle, trialNumber: trialNumber)
            return DistributedTrialSpec(trialNumber: trialNumber, params: params)
        } catch SwiftunaError.searchSpaceExhausted {
            throw SwiftunaDistributedError.searchSpaceExhausted
        } catch {
            throw SwiftunaDistributedError.studyError(String(describing: error))
        }
    }

    /// Reports intermediate progress for early stopping. Returns `true` if the pruner recommends stopping.
    ///
    /// Reporting the same `step` twice keeps the latest value; earlier values for
    /// that step are overwritten and never reach storage.
    public distributed func report(trialNumber: Int, step: Int, value: Double) throws -> Bool {
        if let active = inFlight[trialNumber] {
            // Heartbeat first: a live worker must not be reaped by its own report.
            active.leasedAt = .now
        }
        reapExpired()
        guard let active = inFlight[trialNumber] else {
            throw terminalError(for: trialNumber)
        }
        active.intermediateSteps[step] = value
        active.leasedAt = .now
        return try wrap {
            try study.pruner.shouldPrune(
                study: study,
                trialNumber: trialNumber,
                step: step,
                currentValue: value
            )
        }
    }

    /// Records the final trial evaluation outcome from a worker.
    ///
    /// The in-flight entry is removed only after the outcome is durably recorded.
    /// If recording fails, the entry (and its trial handle) is retained so the
    /// worker can retry the same `tell` with corrected values.
    ///
    /// - Throws: ``SwiftunaDistributedError/invalidConstraint(_:)`` when a
    ///   constraint value is NaN (mirroring ``Trial/setConstraint(_:value:)``).
    public distributed func tell(_ result: DistributedTrialResult) throws {
        guard let active = inFlight[result.trialNumber] else {
            throw terminalError(for: result.trialNumber)
        }

        for (name, value) in result.constraints where value.isNaN {
            throw SwiftunaDistributedError.invalidConstraint(
                "Constraint '\(name)' value cannot be NaN (trial #\(result.trialNumber))")
        }

        guard let handle = active.rawHandle else {
            throw SwiftunaDistributedError.studyError("Trial #\(result.trialNumber) handle already freed")
        }

        // Set constraints on handle
        for (name, val) in result.constraints {
            _ = name.withCString { cName in
                rustuna_trial_set_constraint(handle, cName, val)
            }
        }

        // Set user attrs on handle
        for (name, val) in result.userAttrs {
            _ = name.withCString { cName in
                val.withCString { cVal in
                    rustuna_trial_set_user_attr(handle, cName, cVal)
                }
            }
        }

        try wrap {
            try study.tellRecorded(
                trialNumber: result.trialNumber,
                state: result.state,
                values: result.values,
                intermediateSteps: active.intermediateSteps
            )
        }

        inFlight.removeValue(forKey: result.trialNumber)
        if let handle = active.takeHandle() {
            rustuna_trial_free(handle)
        }
        retire(result.trialNumber, expired: false)

        finishedCount += 1
        if result.state == .complete {
            completedCount += 1
        }
    }

    /// Reaps trials whose lease lapsed. Runs piggybacked on `ask`/`report`;
    /// there are no background timers.
    private func reapExpired() {
        guard let policy = leasePolicy else { return }
        let now = ContinuousClock.now
        for (number, active) in inFlight where now - active.leasedAt > .seconds(policy.timeoutSeconds) {
            switch policy.onExpiry {
            case .failTrial:
                try? study.tellRecorded(
                    trialNumber: number,
                    state: .fail,
                    values: [],
                    intermediateSteps: active.intermediateSteps
                )
            }
            inFlight.removeValue(forKey: number)
            if let handle = active.takeHandle() {
                rustuna_trial_free(handle)
            }
            retire(number, expired: true)
            finishedCount += 1
        }
    }

    /// Specific terminal error for a trial number that is no longer in flight.
    private func terminalError(for trialNumber: Int) -> SwiftunaDistributedError {
        if expiredNumbers.contains(trialNumber) {
            return .leaseExpired(trialNumber)
        }
        if finishedNumbers.contains(trialNumber) {
            return .trialAlreadyFinished(trialNumber)
        }
        return .trialNotFound(trialNumber)
    }

    /// Remembers a retired trial number, trimming both sets past the cap.
    ///
    /// Retirement is approximate: past `retiredNumberCap` entries both sets are
    /// cleared, so a very old duplicate `tell` may report `trialNotFound`
    /// instead of its specific terminal error.
    private func retire(_ trialNumber: Int, expired: Bool) {
        if expired {
            expiredNumbers.insert(trialNumber)
        } else {
            finishedNumbers.insert(trialNumber)
        }
        if finishedNumbers.count + expiredNumbers.count > retiredNumberCap {
            finishedNumbers.removeAll(keepingCapacity: true)
            expiredNumbers.removeAll(keepingCapacity: true)
        }
    }

    /// Returns the best overall trial evaluated so far.
    ///
    /// - Throws: ``SwiftunaDistributedError/studyError(_:)`` on multi-objective
    ///   studies (use ``bestTrials()`` for the Pareto frontier instead).
    public distributed func bestTrial() throws -> PersistedTrial? {
        try wrap { try study.bestTrial }
    }

    /// Returns the Pareto frontier of non-dominated trials in a multi-objective study.
    public distributed func bestTrials() throws -> [PersistedTrial] {
        try wrap { try study.bestTrials }
    }

    /// Returns all trials recorded in the study regardless of state.
    public distributed func trials() throws -> [PersistedTrial] {
        try wrap { try study.trials }
    }

    /// Returns trials filtered by lifecycle state.
    public distributed func trials(where states: Set<TrialState>) throws -> [PersistedTrial] {
        try wrap { try study.trials(where: states) }
    }

    /// Evaluates hyperparameter importance scores using PED-ANOVA.
    ///
    /// - Parameters:
    ///   - normalize: If `true`, scores sum to `1.0`.
    ///   - params: Optional subset of parameter names. `nil` includes all evaluated parameters.
    public distributed func paramImportances(normalize: Bool, params: [String]?) throws -> [String: Double] {
        try wrap { try study.paramImportances(normalize: normalize, params: params).get() }
    }

    /// Sets a study-level user attribute readable by every worker.
    public distributed func setUserAttr(_ key: String, value: String) throws {
        try wrap { try study.setUserAttr(key, value: value) }
    }

    /// Reads a study-level user attribute, or `nil` when unset.
    public distributed func userAttr(_ key: String) throws -> String? {
        try wrap { try study.userAttr(key) }
    }

    /// Returns the count of active in-flight trials currently being evaluated by workers.
    public distributed func inFlightCount() -> Int {
        inFlight.count
    }

    /// Returns the number of trials that finished with `.complete` state.
    public distributed func completedTrialsCount() -> Int {
        completedCount
    }

    /// Returns the number of trials in any terminal state (`.complete`, `.pruned`, or `.fail`).
    public distributed func finishedTrialsCount() -> Int {
        finishedCount
    }
}
