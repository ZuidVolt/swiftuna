import Foundation

/// Errors that can occur during distributed hyperparameter orchestration.
public enum SwiftunaDistributedError: Error, Sendable, Codable, CustomStringConvertible {
    case trialNotFound(Int)
    case trialAlreadyFinished(Int)
    case leaseExpired(Int)
    case studyError(String)

    public var description: String {
        switch self {
        case .trialNotFound(let num): return "Trial #\(num) not found or not currently active"
        case .trialAlreadyFinished(let num): return "Trial #\(num) was already completed or reported"
        case .leaseExpired(let num): return "Lease for trial #\(num) has expired"
        case .studyError(let msg): return "Study error: \(msg)"
        }
    }
}
