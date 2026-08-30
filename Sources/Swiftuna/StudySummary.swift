import Foundation

/// Lightweight snapshot metadata describing an existing study in a storage backend.
public struct StudySummary: Identifiable, Sendable, Equatable {
    public let id: Int
    public let name: String
    public let directions: [Direction]
    public let userAttrs: [String: String]
    public let systemAttrs: [String: String]
    public let trialCount: Int

    public var direction: Direction {
        directions.first ?? .minimize
    }

    public init(
        id: Int,
        name: String,
        directions: [Direction],
        userAttrs: [String: String] = [:],
        systemAttrs: [String: String] = [:],
        trialCount: Int = 0
    ) {
        self.id = id
        self.name = name
        self.directions = directions
        self.userAttrs = userAttrs
        self.systemAttrs = systemAttrs
        self.trialCount = trialCount
    }
}

// Internal decodable payload matching Rust's SerializableStudySummary
internal struct StudySummaryPayload: Decodable {
    let id: Int
    let name: String
    let directions: [Int32]
    let user_attrs: [String: String]
    let system_attrs: [String: String]
    let trial_count: Int

    func toStudySummary() -> StudySummary {
        let dirs = directions.map { Direction(rawValue: $0) ?? .minimize }
        return StudySummary(
            id: id,
            name: name,
            directions: dirs,
            userAttrs: user_attrs,
            systemAttrs: system_attrs,
            trialCount: trial_count
        )
    }
}

extension Sequence where Element == StudySummary {
    /// Filters study summaries having at least `count` trials.
    public func minTrials(_ count: Int) -> [StudySummary] {
        filter { $0.trialCount >= count }
    }

    /// Sorts study summaries by their trial count.
    public func sortedByTrialCount(descending: Bool = true) -> [StudySummary] {
        sorted { descending ? ($0.trialCount > $1.trialCount) : ($0.trialCount < $1.trialCount) }
    }

    /// Finds a study summary with the matching name.
    public func named(_ name: String) -> StudySummary? {
        first { $0.name == name }
    }
}
