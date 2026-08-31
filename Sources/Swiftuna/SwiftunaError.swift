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
        switch code {
        case -1: return .invalidArgument(message)
        case -99: return .panic(message)
        case 1: return .objectiveError(message)
        case 2: return .samplerError(message)
        case 3: return .storageError(message)
        case 4: return .duplicatedStudy(message)
        case 5: return .studyNotFound(message)
        case 6: return .trialNotFound(message)
        case 7: return .trialDiscarded
        case 8: return .attrNotFound(message)
        case 9: return .trialQueueEmpty
        case 10: return .attrOverwriteNotAllowed(message)
        case 11: return .invalidObjectiveValues(message)
        case 12: return .trialAlreadyFinished(message)
        case 13: return .unsupportedSearchSpace(message)
        case 14: return .unsupportedMultiObjective
        case 15: return .noCompletedTrial
        case 16: return .incompatibleDistribution(message)
        case 17: return .invalidFixedParam(message)
        case 18: return .missingDependency(message)
        case 19: return .unexpected(message)
        case 21: return .searchSpaceExhausted(message)
        default: return .unexpected("Code \(code): \(message)")
        }
    }

    internal static func fromLastError(fallbackCode: Int32 = -2, context: String) -> Self {
        let code = rustuna_last_error_code()
        let effectiveCode = code != 0 ? code : fallbackCode
        let msg = rustuna_last_error_message().map { String(cString: $0) } ?? context
        return from(code: effectiveCode, message: msg)
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
