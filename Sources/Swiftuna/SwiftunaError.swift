import Foundation
internal import LibRustuna

/// Represents all error conditions that can arise during study creation, sampling, optimization, or storage operations.
public enum SwiftunaError: Error, CustomStringConvertible, Sendable, Equatable {
    /// An invalid argument was passed to an API method (e.g. empty names, invalid bounds, or mismatched objective counts).
    case invalidArgument(String)

    /// An unrecoverable internal error occurred within the underlying Rust engine.
    case panic(String)

    /// The user-provided objective function threw an unhandled error during trial evaluation.
    case objectiveError(String)

    /// A failure occurred inside the parameter sampling algorithm (e.g. invalid distribution configuration or internal sampler state).
    case samplerError(String)

    /// A persistence error occurred during read/write operations against the storage backend (e.g. SQLite locking or corrupted journal).
    case storageError(String)

    /// A study with the specified name already exists in the given storage backend and `loadIfExists` was `false`.
    case duplicatedStudy(String)

    /// The requested study name was not found in the storage backend.
    case studyNotFound(String)

    /// The requested trial ID or number was not found in the study record.
    case trialNotFound(String)

    /// The trial was discarded before completion without reporting valid objective values.
    case trialDiscarded

    /// The requested user attribute or system attribute key does not exist on the study or trial.
    case attrNotFound(String)

    /// The pre-enqueued trial queue has no remaining parameter configurations to pop.
    case trialQueueEmpty

    /// An attempt was made to overwrite an existing attribute key when overwrites are forbidden.
    case attrOverwriteNotAllowed(String)

    /// The objective values returned by the evaluation closure are invalid (e.g. NaN, infinite, or count mismatching study directions).
    case invalidObjectiveValues(String)

    /// An operation was attempted on a trial that has already transitioned to a terminal state (`complete`, `pruned`, or `fail`).
    case trialAlreadyFinished(String)

    /// The requested search space configuration is not supported by the active sampler.
    case unsupportedSearchSpace(String)

    /// Multi-objective optimization was attempted with a sampler or pruner that only supports single-objective studies.
    case unsupportedMultiObjective

    /// An operation requiring at least one successfully completed trial (such as `bestTrial` or `bestValue`) was called on a study with no completed trials.
    case noCompletedTrial

    /// The suggested distribution for a parameter conflicts with a previous distribution recorded for the same parameter name.
    case incompatibleDistribution(String)

    /// An invalid fixed parameter value was supplied during partial trial enqueuing.
    case invalidFixedParam(String)

    /// An optional dependency or backend feature required for this operation is not available.
    case missingDependency(String)

    /// An unexpected internal condition or unmapped status code was encountered.
    case unexpected(String)

    /// A numeric distribution range is invalid (e.g. `lowerBound > upperBound`, non-finite bounds, or invalid log step).
    case invalidRange(String)

    /// An empty collection of choices was provided to categorical parameter suggestion.
    case emptyChoices(String)

    /// No trials exist in the study.
    case noTrialsFound

    /// The underlying C pointer handle for a study, trial, or sampler has expired or been consumed.
    case handleExpired(String)

    /// The trial was stopped early by an active ``Pruner``.
    case trialPruned(reason: String? = nil)

    /// A discrete or grid sampler has evaluated every possible parameter combination in the search space.
    case searchSpaceExhausted(String)

    public static func from(code: Int32, message: String) -> Self {
        let errCode = RustunaErrorCode(rawValue: code)
        switch errCode {
        case RUSTUNA_SUCCESS: return .unexpected("Success code passed to error handler: \(message)")
        case RUSTUNA_ERR_INVALID_ARGUMENT: return .invalidArgument(message)
        case RUSTUNA_ERR_EMPTY_CHOICES: return .emptyChoices(message)
        case RUSTUNA_ERR_INVALID_RANGE: return .invalidRange(message)
        case RUSTUNA_ERR_PANIC: return .panic(message)
        case RUSTUNA_ERR_OBJECTIVE: return .objectiveError(message)
        case RUSTUNA_ERR_SAMPLER: return .samplerError(message)
        case RUSTUNA_ERR_STORAGE: return .storageError(message)
        case RUSTUNA_ERR_DUPLICATED_STUDY: return .duplicatedStudy(message)
        case RUSTUNA_ERR_STUDY_NOT_FOUND: return .studyNotFound(message)
        case RUSTUNA_ERR_TRIAL_NOT_FOUND: return .trialNotFound(message)
        case RUSTUNA_ERR_TRIAL_DISCARDED: return .trialDiscarded
        case RUSTUNA_ERR_ATTR_NOT_FOUND: return .attrNotFound(message)
        case RUSTUNA_ERR_TRIAL_QUEUE_EMPTY: return .trialQueueEmpty
        case RUSTUNA_ERR_ATTR_OVERWRITE_NOT_ALLOWED: return .attrOverwriteNotAllowed(message)
        case RUSTUNA_ERR_INVALID_OBJECTIVE_VALUES: return .invalidObjectiveValues(message)
        case RUSTUNA_ERR_TRIAL_ALREADY_FINISHED: return .trialAlreadyFinished(message)
        case RUSTUNA_ERR_UNSUPPORTED_SEARCH_SPACE: return .unsupportedSearchSpace(message)
        case RUSTUNA_ERR_UNSUPPORTED_MULTI_OBJECTIVE: return .unsupportedMultiObjective
        case RUSTUNA_ERR_NO_COMPLETED_TRIAL: return .noCompletedTrial
        case RUSTUNA_ERR_INCOMPATIBLE_DISTRIBUTION: return .incompatibleDistribution(message)
        case RUSTUNA_ERR_INVALID_FIXED_PARAM: return .invalidFixedParam(message)
        case RUSTUNA_ERR_MISSING_DEPENDENCY: return .missingDependency(message)
        case RUSTUNA_ERR_UNEXPECTED: return .unexpected(message)
        case RUSTUNA_ERR_IMPORTANCE_EVALUATOR: return .samplerError(message)
        case RUSTUNA_ERR_SEARCH_SPACE_EXHAUSTED: return .searchSpaceExhausted(message)
        default: return .unexpected("Unknown error code \(code): \(message)")
        }
    }

    internal static func fromLastError(fallbackCode: Int32 = -2, context: String) -> Self {
        var code: Int32 = 0
        var cMsg: UnsafeMutablePointer<CChar>?
        let hasError = rustuna_take_last_error(&code, &cMsg)
        if hasError != 0 {
            let msg = cMsg.map { String(cString: $0) } ?? context
            if let cMsg {
                rustuna_string_free(cMsg)
            }
            let effectiveCode = code != 0 ? code : fallbackCode
            return from(code: effectiveCode, message: msg)
        } else {
            return from(code: fallbackCode, message: context)
        }
    }

    public var description: String {
        switch self {
        case .invalidArgument(let m): "Invalid argument: \(m)"
        case .panic(let m): "Rust panic: \(m)"
        case .objectiveError(let m): "Objective error: \(m)"
        case .samplerError(let m): "Sampler error: \(m)"
        case .storageError(let m): "Storage error: \(m)"
        case .duplicatedStudy(let m): "Duplicated study: \(m)"
        case .studyNotFound(let m): "Study not found: \(m)"
        case .trialNotFound(let m): "Trial not found: \(m)"
        case .trialDiscarded: "Trial discarded"
        case .attrNotFound(let m): "Attribute not found: \(m)"
        case .trialQueueEmpty: "Trial queue empty"
        case .attrOverwriteNotAllowed(let m): "Attribute overwrite not allowed: \(m)"
        case .invalidObjectiveValues(let m): "Invalid objective values: \(m)"
        case .trialAlreadyFinished(let m): "Trial already finished: \(m)"
        case .unsupportedSearchSpace(let m): "Unsupported search space: \(m)"
        case .unsupportedMultiObjective: "Multi-objective optimization is not supported by this configuration"
        case .noCompletedTrial: "No completed trial found in study"
        case .incompatibleDistribution(let m): "Incompatible distribution: \(m)"
        case .invalidFixedParam(let m): "Invalid fixed parameter: \(m)"
        case .missingDependency(let m): "Missing dependency: \(m)"
        case .unexpected(let m): "Unexpected error: \(m)"
        case .invalidRange(let m): "Invalid distribution range: \(m)"
        case .emptyChoices(let m): "Empty choices provided: \(m)"
        case .noTrialsFound: "No trials found in study"
        case .handleExpired(let m): "Handle expired or invalid: \(m)"
        case .trialPruned(let r): "Trial pruned: \(r ?? "No reason given")"
        case .searchSpaceExhausted(let m): "Search space exhausted: \(m)"
        }
    }
}
