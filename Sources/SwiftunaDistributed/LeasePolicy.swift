import Foundation

/// Cap for retired trial-number sets on the coordinator (see `retire`).
let retiredNumberCap = 4_096

/// Policy governing trial leases on the coordinator.
///
/// A lease starts at `ask()` and is renewed by every `report()` heartbeat. When
/// no heartbeat arrives within `timeout`, the next `ask()` or `report()`
/// call reaps the trial as `.fail` (no background timers).
///
/// `nil` (the default) disables leases entirely, preserving exact pre-lease behavior.
public struct LeasePolicy: Sendable, Codable {
    /// Time without a heartbeat before a trial is reaped as failed.
    public var timeout: Duration

    public init(timeout: Duration = .seconds(300)) {
        self.timeout = timeout
    }
}
