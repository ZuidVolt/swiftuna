import Foundation

/// Lightweight snapshot metadata describing an existing study in a storage backend.
///
/// `StudySummary` is retrieved via ``StorageBackend/studies()`` or ``getStudies(in:)`` without loading
/// full trial histories into memory, making it fast and efficient for inspecting databases or dashboard backends.
///
/// ### Example
/// ```swift
/// let storage = StorageBackend.sqlite(path: "experiments.db")
/// let summaries = try Swiftuna.getStudies(in: storage)
///
/// // Find active studies with at least 50 trials
/// let matureStudies = summaries.minTrials(50).sortedByTrialCount()
/// for study in matureStudies {
///     print("Study \(study.name) (ID: \(study.id)) has \(study.trialCount) trials")
/// }
/// ```
public struct StudySummary: Identifiable, Sendable, Equatable {
    /// The unique numerical identifier of the study within the storage database.
    public let id: Int

    /// The unique string name identifier of the study.
    public let name: String

    /// The optimization directions (e.g. `[.minimize]` or `[.maximize, .minimize]`).
    public let directions: [Direction]

    /// User-defined string metadata attributes attached to the study.
    public let userAttrs: [String: String]

    /// System metadata attributes attached to the study.
    public let systemAttrs: [String: String]

    /// Total number of trials (including running, complete, pruned, waiting, and failed) recorded in the study.
    public let trialCount: Int

    /// Primary direction of optimization (first element in ``directions``, defaulting to ``Direction/minimize``).
    public var direction: Direction {
        directions.first ?? .minimize
    }

    /// Initializes a study summary record.
    ///
    /// - Parameters:
    ///   - id: Unique integer study ID.
    ///   - name: Unique study name.
    ///   - directions: Array of optimization directions.
    ///   - userAttrs: Key-value user attributes. Defaults to `[:]`.
    ///   - systemAttrs: Key-value system attributes. Defaults to `[:]`.
    ///   - trialCount: Total trial count. Defaults to `0`.
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
    ///
    /// - Parameter count: Minimum number of trials required.
    /// - Returns: Filtered array of study summaries.
    public func minTrials(_ count: Int) -> [StudySummary] {
        filter { $0.trialCount >= count }
    }

    /// Sorts study summaries by their trial count.
    ///
    /// - Parameter descending: If `true` (default), sorts from highest trial count to lowest.
    /// - Returns: Sorted array of study summaries.
    public func sortedByTrialCount(descending: Bool = true) -> [StudySummary] {
        sorted { descending ? ($0.trialCount > $1.trialCount) : ($0.trialCount < $1.trialCount) }
    }

    /// Finds the first study summary matching the given name.
    ///
    /// - Parameter name: The study name identifier to search for.
    /// - Returns: The matching ``StudySummary``, or `nil` if not found.
    public func named(_ name: String) -> StudySummary? {
        first { $0.name == name }
    }
}
