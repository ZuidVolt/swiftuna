#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

/// Protocol for CMA-ES pseudo-random number generators.
package protocol CMAPRNGProtocol: Sendable {
    /// Generates a uniform random Double in $(0, 1)$.
    mutating func nextUniform() -> Double
    /// Generates a standard normal random variable $\mathcal{N}(0, 1)$.
    mutating func nextGaussian() -> Double
}

// MARK: - Fast Modern PRNG (Xoshiro256+ / SplitMix64)

/// Fast, high-quality pseudo-random number generator for production CMA-ES runs.
package struct FastPRNG: CMAPRNGProtocol {
    private var s0: UInt64
    private var s1: UInt64
    private var s2: UInt64
    private var s3: UInt64
    private var cachedGaussian: Double?

    package init(seed: UInt64 = 42) {
        var sm = seed
        func splitMix() -> UInt64 {
            sm &+= 0x9e37_79b9_7f4a_7c15
            var z = sm
            z = (z ^ (z &>> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z &>> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z &>> 31)
        }
        self.s0 = splitMix()
        self.s1 = splitMix()
        self.s2 = splitMix()
        self.s3 = splitMix()
        self.cachedGaussian = nil
    }

    private mutating func nextUInt64() -> UInt64 {
        let result = s0 &+ s3
        let t = s1 &<< 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = (s3 &<< 45) | (s3 &>> 19)
        return result
    }

    package mutating func nextUniform() -> Double {
        // Generate uniform Double in (0, 1)
        let raw = nextUInt64() &>> 11
        let u = (Double(raw) + 0.5) / 9007199254740992.0
        return min(max(u, 1e-15), 1.0 - 1e-15)
    }

    package mutating func nextGaussian() -> Double {
        if let cached = cachedGaussian {
            cachedGaussian = nil
            return cached
        }
        let u1 = nextUniform()
        let u2 = nextUniform()
        let r = (-2.0 * log(u1)).squareRoot()
        let theta = 2.0 * Double.pi * u2
        cachedGaussian = r * sin(theta)
        return r * cos(theta)
    }
}

// MARK: - NumPy-Compatible MT19937 PRNG

/// 32-bit Mersenne Twister replicating NumPy's `np.random.RandomState(seed)`
/// exactly, ensuring 1-to-1 bit-for-bit test fixture reproducibility.
package struct NumpyMT19937PRNG: CMAPRNGProtocol {
    private var mt: [UInt32] = [UInt32](repeating: 0, count: 624)
    private var mti: Int = 625
    private var cachedGaussian: Double?

    package init(seed: UInt32 = 42) {
        mt[0] = seed
        for i in 1..<624 {
            let prev = mt[i - 1]
            mt[i] = 1_812_433_253 &* (prev ^ (prev &>> 30)) &+ UInt32(i)
        }
        mti = 624
        cachedGaussian = nil
    }

    private mutating func nextUInt32() -> UInt32 {
        if mti >= 624 {
            for i in 0..<624 {
                let y = (mt[i] & 0x8000_0000) | (mt[(i + 1) % 624] & 0x7fff_ffff)
                var nextVal = mt[(i + 397) % 624] ^ (y &>> 1)
                if (y & 1) != 0 {
                    nextVal ^= 0x9908_b0df
                }
                mt[i] = nextVal
            }
            mti = 0
        }

        var y = mt[mti]
        mti += 1

        y ^= (y &>> 11)
        y ^= (y &<< 7) & 0x9d2c_5680
        y ^= (y &<< 15) & 0xefc6_0000
        y ^= (y &>> 18)

        return y
    }

    package mutating func nextUniform() -> Double {
        // NumPy random_sample generates 53-bit float:
        let a = nextUInt32() &>> 5
        let b = nextUInt32() &>> 6
        let u = (Double(a) * 67108864.0 + Double(b)) / 9007199254740992.0
        return min(max(u, 1e-15), 1.0 - 1e-15)
    }

    package mutating func nextGaussian() -> Double {
        // NumPy legacy standard normal: Marsaglia polar method
        if let cached = cachedGaussian {
            cachedGaussian = nil
            return cached
        }
        var x1 = 0.0
        var x2 = 0.0
        var r2 = 0.0
        repeat {
            x1 = 2.0 * nextUniform() - 1.0
            x2 = 2.0 * nextUniform() - 1.0
            r2 = x1 * x1 + x2 * x2
        } while r2 >= 1.0 || r2 == 0.0

        let f = (-2.0 * log(r2) / r2).squareRoot()
        cachedGaussian = f * x1
        return f * x2
    }
}
