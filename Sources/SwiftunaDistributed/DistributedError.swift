import Foundation

/// Errors that can occur during distributed hyperparameter orchestration.
public enum SwiftunaDistributedError: Error, Sendable, Codable, CustomStringConvertible {
    case trialNotFound(Int)
    case trialAlreadyFinished(Int)
    case leaseExpired(Int)
    case searchSpaceExhausted
    case tooManyInFlight(Int)
    case invalidConstraint(String)
    case objectiveCountMismatch(expected: Int, got: Int)
    case studyError(String)

    public var description: String {
        switch self {
        case .trialNotFound(let num): return "Trial #\(num) not found or not currently active"
        case .trialAlreadyFinished(let num): return "Trial #\(num) was already completed or reported"
        case .leaseExpired(let num): return "Lease for trial #\(num) has expired"
        case .searchSpaceExhausted: return "Search space exhausted: no untried configurations remain"
        case .tooManyInFlight(let count): return "Too many in-flight trials (\(count)): coordinator is at capacity"
        case .invalidConstraint(let msg): return "Invalid constraint: \(msg)"
        case .objectiveCountMismatch(let expected, let got):
            return "Expected \(expected) objective value(s), got \(got)"
        case .studyError(let msg): return "Study error: \(msg)"
        }
    }

    /// Whether retrying the same call can succeed.
    ///
    /// Only `tooManyInFlight` qualifies: wait for a slot (or use
    /// `ask(waitingUpTo:)`) and retry. Every other case needs a different
    /// call — a corrected payload, a new trial, or operator intervention —
    /// so blind retry loops should stop on them.
    public var isRetryable: Bool {
        switch self {
        case .tooManyInFlight: return true
        case .trialNotFound, .trialAlreadyFinished, .leaseExpired,
            .searchSpaceExhausted, .invalidConstraint, .objectiveCountMismatch, .studyError:
            return false
        }
    }
}
