import Foundation

internal import LibRustuna

public enum SwiftunaError: Error, CustomStringConvertible, Sendable, Equatable {
    case invalidArgument(String)
    case panic(String)
    case objectiveError(String)
    case samplerError(String)
    case storageError(String)
    case duplicatedStudy(String)
    case studyNotFound(String)
    case trialNotFound(String)
    case trialDiscarded
    case attrNotFound(String)
    case trialQueueEmpty
    case attrOverwriteNotAllowed(String)
    case invalidObjectiveValues(String)
    case trialAlreadyFinished(String)
    case unsupportedSearchSpace(String)
    case unsupportedMultiObjective
    case noCompletedTrial
    case incompatibleDistribution(String)
    case invalidFixedParam(String)
    case missingDependency(String)
    case unexpected(String)
    case invalidRange(String)
    case emptyChoices(String)
    case noTrialsFound
    case handleExpired(String)
    case trialPruned(reason: String? = nil)

    public static func from(code: Int32, message: String) -> SwiftunaError {
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
        default: return .unexpected("Code \(code): \(message)")
        }
    }

    internal static func fromLastError(fallbackCode: Int32 = -2, context: String) -> SwiftunaError {
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
        }
    }
}
