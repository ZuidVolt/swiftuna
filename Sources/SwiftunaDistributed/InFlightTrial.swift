import Foundation
import Swiftuna
internal import LibRustuna

/// An owned Rustuna trial handle. Noncopyable so the handle has exactly one
/// owner; `take()` moves it out, and whatever remains is freed in `deinit`.
package struct TrialHandle: ~Copyable, @unchecked Sendable {
    package var raw: OpaquePointer?

    package init(_ raw: OpaquePointer?) {
        self.raw = raw
    }

    deinit {
        if let raw {
            rustuna_trial_free(raw)
        }
    }

    package mutating func take() -> OpaquePointer? {
        let h = raw
        raw = nil
        return h
    }
}

package final class InFlightTrial: @unchecked Sendable {
    package var handle: TrialHandle
    package let trialNumber: Int
    package var intermediateSteps: [Int: Double] = [:]
    package var leasedAt: ContinuousClock.Instant

    package init(rawHandle: OpaquePointer, trialNumber: Int, leasedAt: ContinuousClock.Instant = .now) {
        self.handle = TrialHandle(rawHandle)
        self.trialNumber = trialNumber
        self.leasedAt = leasedAt
    }

    package func takeHandle() -> OpaquePointer? {
        handle.take()
    }
}
