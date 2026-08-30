import Foundation
internal import LibRustuna

public protocol Sampler: Sendable {
    func makeRawHandle() -> OpaquePointer?
}

public struct TPESampler: Sampler {
    public let seed: UInt64?

    public init(seed: UInt64? = nil) {
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        if let s = seed {
            return rustuna_sampler_tpe_new(s, true)
        } else {
            return rustuna_sampler_tpe_new(0, false)
        }
    }
}

public struct RandomSampler: Sampler {
    public let seed: UInt64?

    public init(seed: UInt64? = nil) {
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        if let s = seed {
            return rustuna_sampler_random_new(s, true)
        } else {
            return rustuna_sampler_random_new(0, false)
        }
    }
}
