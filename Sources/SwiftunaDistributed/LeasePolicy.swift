import Foundation

/// Cap for retired trial-number sets on the coordinator (see `retire`).
let retiredNumberCap = 4096

/// What the coordinator does with a trial whose lease expires.
public enum LeaseExpiryAction: String, Sendable, Codable {
    /// Records the trial as `.fail` with its reported intermediates, freeing the slot.
    case failTrial
}

/// Policy governing trial leases on the coordinator.
///
/// A lease starts at `ask()` and is renewed by every `report()` heartbeat. When
/// no heartbeat arrives within `timeoutSeconds`, the next `ask()` or `report()`
/// call reaps the trial (no background timers) and applies `onExpiry`.
///
/// `nil` (the default) disables leases entirely, preserving exact pre-lease behavior.
public struct LeasePolicy: Sendable, Codable {
    /// Seconds without a heartbeat before a trial is reaped.
    public var timeoutSeconds: Double

    /// Action applied to expired trials during reaping.
    public var onExpiry: LeaseExpiryAction

    public init(timeoutSeconds: Double = 300, onExpiry: LeaseExpiryAction = .failTrial) {
        self.timeoutSeconds = max(0.001, timeoutSeconds)
        self.onExpiry = onExpiry
    }
}
