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

    public init(study: Study, searchSpace: SearchSpace, actorSystem: ActorSystem) {
        self.study = study
        self.searchSpace = searchSpace
        self.actorSystem = actorSystem
    }

    /// Dispatches the next hyperparameter configuration to an active worker.
    public distributed func ask() throws -> DistributedTrialSpec {
        do {
            var trial = try study.ask()
            let params = try searchSpace.sample(trial: &trial)
            let trialNumber = trial.number
            guard let handle = trial.takeHandle() else {
                throw SwiftunaDistributedError.studyError("Failed to take handle for trial #\(trialNumber)")
            }
            inFlight[trialNumber] = InFlightTrial(rawHandle: handle, trialNumber: trialNumber)
            return DistributedTrialSpec(trialNumber: trialNumber, params: params)
        } catch {
            throw SwiftunaDistributedError.studyError(String(describing: error))
        }
    }

    /// Reports intermediate progress for early stopping. Returns `true` if the pruner recommends stopping.
    public distributed func report(trialNumber: Int, step: Int, value: Double) throws -> Bool {
        guard let active = inFlight[trialNumber] else {
            throw SwiftunaDistributedError.trialNotFound(trialNumber)
        }
        active.intermediateSteps[step] = value
        do {
            return try study.pruner.shouldPrune(
                study: study,
                trialNumber: trialNumber,
                step: step,
                currentValue: value
            )
        } catch {
            throw SwiftunaDistributedError.studyError(String(describing: error))
        }
    }

    /// Records the final trial evaluation outcome from a worker.
    public distributed func tell(_ result: DistributedTrialResult) throws {
        guard let active = inFlight.removeValue(forKey: result.trialNumber) else {
            throw SwiftunaDistributedError.trialNotFound(result.trialNumber)
        }

        guard let handle = active.takeHandle() else {
            throw SwiftunaDistributedError.studyError("Trial #\(result.trialNumber) handle already freed")
        }
        defer { rustuna_trial_free(handle) }

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

        // Finish in study via C FFI
        let intermediate = active.intermediateSteps
        var intermediateJsonStr: String?
        if !intermediate.isEmpty {
            let stepMap = Dictionary(uniqueKeysWithValues: intermediate.map { (String($0.key), $0.value) })
            if let data = try? JSONEncoder().encode(stepMap) {
                intermediateJsonStr = String(data: data, encoding: .utf8)
            }
        }

        guard let rawStudy = study.rawHandle else {
            throw SwiftunaDistributedError.studyError("Study handle is expired or invalid")
        }

        let values = result.values
        let state = result.state

        let status: Int32 = intermediateJsonStr.withOptionalCString { cIntermediate in
            if values.isEmpty {
                return rustuna_study_tell_multi(
                    rawStudy,
                    UInt32(result.trialNumber),
                    state.rawValue,
                    nil,
                    0,
                    cIntermediate
                )
            }
            return values.withUnsafeBufferPointer { buf in
                rustuna_study_tell_multi(
                    rawStudy,
                    UInt32(result.trialNumber),
                    state.rawValue,
                    buf.baseAddress,
                    values.count,
                    cIntermediate
                )
            }
        }

        if status != 0 {
            throw SwiftunaDistributedError.studyError("Failed to tell trial #\(result.trialNumber): status \(status)")
        }

        completedCount += 1
    }

    /// Returns the best overall trial evaluated so far.
    public distributed func bestTrial() throws -> PersistedTrial? {
        do {
            return try study.bestTrial
        } catch {
            throw SwiftunaDistributedError.studyError(String(describing: error))
        }
    }

    /// Returns the count of active in-flight trials currently being evaluated by workers.
    public distributed func inFlightCount() -> Int {
        inFlight.count
    }

    /// Returns the number of completed trials.
    public distributed func completedTrialsCount() -> Int {
        completedCount
    }
}

fileprivate extension Optional where Wrapped == String {
    func withOptionalCString<R>(_ body: (UnsafePointer<CChar>?) throws -> R) rethrows -> R {
        if let self {
            return try self.withCString { try body($0) }
        } else {
            return try body(nil)
        }
    }
}
