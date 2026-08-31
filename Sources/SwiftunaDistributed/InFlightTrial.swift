import Foundation
import Swiftuna
internal import LibRustuna

package final class InFlightTrial: @unchecked Sendable {
    package var rawHandle: OpaquePointer?
    package let trialNumber: Int
    package var intermediateSteps: [Int: Double] = [:]
    package var leasedAt: ContinuousClock.Instant

    package init(rawHandle: OpaquePointer, trialNumber: Int, leasedAt: ContinuousClock.Instant = .now) {
        self.rawHandle = rawHandle
        self.trialNumber = trialNumber
        self.leasedAt = leasedAt
    }

    deinit {
        if let h = rawHandle {
            rustuna_trial_free(h)
        }
    }

    package func takeHandle() -> OpaquePointer? {
        let h = rawHandle
        rawHandle = nil
        return h
    }
}
