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
        }
        return rustuna_sampler_tpe_new(0, false)
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
        }
        return rustuna_sampler_random_new(0, false)
    }
}

public struct NSGAIISampler: Sampler {
    public let populationSize: Int
    public let mutationProb: Double?
    public let crossoverProb: Double
    public let swappingProb: Double
    public let seed: UInt64?

    public init(
        populationSize: Int = 50,
        mutationProb: Double? = nil,
        crossoverProb: Double = 0.9,
        swappingProb: Double = 0.5,
        seed: UInt64? = nil
    ) {
        self.populationSize = populationSize
        self.mutationProb = mutationProb
        self.crossoverProb = crossoverProb
        self.swappingProb = swappingProb
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        var rawPtr: OpaquePointer?
        let mutProbVal = mutationProb ?? -1.0
        let status = rustuna_sampler_nsgaii_new(
            populationSize,
            mutProbVal,
            crossoverProb,
            swappingProb,
            seed ?? 0,
            &rawPtr
        )
        if status != 0 {
            return nil
        }
        return rawPtr
    }
}

/// A Quasi-Monte Carlo Sampler that generates low-discrepancy Sobol sequences.
///
/// QMC systematically minimizes search space voids and clusters across continuous and
/// categorical parameters using Antonov-Saleev Gray code and Joe-Kuo direction numbers
/// up to 1,024 dimensions.
public struct QMCSampler: Sampler {
    public let seed: UInt64?

    /// Creates a QMC sampler.
    ///
    /// - Parameter seed: Optional seed for the fallback random sampler used when
    ///   parameters fall outside the joint search space.
    public init(seed: UInt64? = nil) {
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        rustuna_sampler_qmc_new(seed ?? 0, seed != nil)
    }
}
