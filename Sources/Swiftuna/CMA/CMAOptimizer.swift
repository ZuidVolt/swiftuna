#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

/// State payload for checkpoint serialization.
package struct CMACheckpoint: Codable, Sendable {
    package let dim: Int
    package let generation: Int
    package let mean: [Double]
    package let sigma: Double
    package let pSigma: [Double]
    package let pc: [Double]
    package let cov: [Double]
}

/// A move-only (`~Copyable`), zero-allocation Active CMA-ES optimizer.
///
/// Implements Hansen (2016) Active CMA-ES with negative covariance updates,
/// matching CyberAgent's `cmaes._cma.py` in Optuna.
package struct CMAOptimizer: ~Copyable {
    package let dim: Int
    package let populationSize: Int
    package let mu: Int
    package let muEff: Double
    package let cc: Double
    package let c1: Double
    package let cmu: Double
    package let cSigma: Double
    package let dSigma: Double
    package let cm: Double
    package let chiN: Double
    package let weights: [Double]

    // Dynamic state
    package var mean: [Double]
    package var sigma: Double
    package var pSigma: [Double]
    package var pc: [Double]
    package var C: FlatMatrix
    package var generation: Int

    // Bounds and resampling
    package var bounds: [(lower: Double, upper: Double)]?
    package var maxResampling: Int

    // Eigensystem cache
    private var eigensystem: JacobiEigensystem
    private var needsEigenUpdate: Bool

    package init(
        mean: [Double],
        sigma: Double,
        bounds: [(lower: Double, upper: Double)]? = nil,
        populationSize: Int? = nil,
        maxResampling: Int? = nil
    ) {
        let n = mean.count
        precondition(n > 0, "Dimension must be positive")
        precondition(sigma > 0, "Sigma must be positive")

        self.dim = n
        let popSize = populationSize ?? (4 + Int(floor(3.0 * log(Double(n)))))
        self.populationSize = popSize
        let muVal = popSize / 2
        self.mu = muVal

        // Hansen (2016) Eq. 49: raw weights
        var weightsPrime = [Double](repeating: 0.0, count: popSize)
        for i in 0..<popSize {
            weightsPrime[i] = log((Double(popSize) + 1.0) / 2.0) - log(Double(i + 1))
        }

        var sumWPos = 0.0
        var sumWSqPos = 0.0
        for i in 0..<muVal {
            sumWPos += weightsPrime[i]
            sumWSqPos += weightsPrime[i] * weightsPrime[i]
        }
        let muEffVal = (sumWPos * sumWPos) / sumWSqPos
        self.muEff = muEffVal

        var sumWNeg = 0.0
        var sumWSqNeg = 0.0
        for i in muVal..<popSize {
            sumWNeg += abs(weightsPrime[i])
            sumWSqNeg += weightsPrime[i] * weightsPrime[i]
        }
        let muEffMinus = (sumWNeg * sumWNeg) / (sumWSqNeg > 0 ? sumWSqNeg : 1.0)

        let alphaCov = 2.0
        let c1Val = alphaCov / (pow(Double(n) + 1.3, 2.0) + muEffVal)
        self.c1 = c1Val

        let cmuNumerator = alphaCov * (muEffVal - 2.0 + 1.0 / muEffVal)
        let cmuDenominator = pow(Double(n) + 2.0, 2.0) + alphaCov * muEffVal / 2.0
        let cmuVal = min(1.0 - c1Val - 1e-8, cmuNumerator / cmuDenominator)
        self.cmu = cmuVal

        let minAlpha = min(
            1.0 + c1Val / cmuVal,
            1.0 + (2.0 * muEffMinus) / (muEffVal + 2.0),
            (1.0 - c1Val - cmuVal) / (Double(n) * cmuVal)
        )

        // Eq. 53: normalized positive and negative weights
        var finalWeights = [Double](repeating: 0.0, count: popSize)
        for i in 0..<popSize {
            if weightsPrime[i] >= 0.0 {
                finalWeights[i] = weightsPrime[i] / sumWPos
            } else {
                finalWeights[i] = minAlpha * weightsPrime[i] / sumWNeg
            }
        }
        self.weights = finalWeights
        self.cm = 1.0

        // Step-size control constants (Eq. 55)
        let cSigmaVal = (muEffVal + 2.0) / (Double(n) + muEffVal + 5.0)
        self.cSigma = cSigmaVal
        let dSigmaTerm = max(0.0, ((muEffVal - 1.0) / (Double(n) + 1.0)).squareRoot() - 1.0)
        self.dSigma = 1.0 + 2.0 * dSigmaTerm + cSigmaVal

        // Rank-one cumulation constant (Eq. 56)
        self.cc = (4.0 + muEffVal / Double(n)) / (Double(n) + 4.0 + 2.0 * muEffVal / Double(n))

        // E||N(0, I)|| expectation
        self.chiN = Double(n).squareRoot() * (1.0 - 1.0 / (4.0 * Double(n)) + 1.0 / (21.0 * pow(Double(n), 2.0)))

        self.mean = mean
        self.sigma = sigma
        self.pSigma = [Double](repeating: 0.0, count: n)
        self.pc = [Double](repeating: 0.0, count: n)
        self.C = FlatMatrix.identity(dim: n)
        self.generation = 0
        self.bounds = bounds
        self.maxResampling = maxResampling ?? (10 * n)

        self.eigensystem = JacobiEigensystem(dim: n)
        self.needsEigenUpdate = true
    }

    /// Checks if candidate vector is within specified bounds.
    package func isFeasible(_ x: [Double]) -> Bool {
        guard let bounds else { return true }
        for i in 0..<dim {
            if x[i] < bounds[i].lower || x[i] > bounds[i].upper {
                return false
            }
        }
        return true
    }

    /// Clips out-of-bounds candidate vector to boundaries.
    package func clipToBounds(_ x: [Double]) -> [Double] {
        guard let bounds else { return x }
        var clipped = x
        for i in 0..<dim {
            if clipped[i] < bounds[i].lower {
                clipped[i] = bounds[i].lower
            } else if clipped[i] > bounds[i].upper {
                clipped[i] = bounds[i].upper
            }
        }
        return clipped
    }

    /// Ensures eigensystem $C = B D^2 B^T$ is freshly computed.
    private mutating func updateEigensystemIfNeeded() {
        guard needsEigenUpdate else { return }
        C.enforceSymmetry()
        eigensystem.decompose(matrix: C)
        needsEigenUpdate = false
    }

    /// Samples a candidate solution vector $x \in \mathbb{R}^D$ from $\mathcal{N}(m, \sigma^2 C)$.
    package mutating func ask(rng: inout some CMAPRNGProtocol) -> [Double] {
        updateEigensystemIfNeeded()

        // Rejection resampling loop
        for _ in 0..<maxResampling {
            let x = sampleSingleSolution(rng: &rng)
            if isFeasible(x) {
                return x
            }
        }

        // Resampling exceeded: draw one more and clip to bounds
        let x = sampleSingleSolution(rng: &rng)
        return clipToBounds(x)
    }

    private mutating func sampleSingleSolution(rng: inout some CMAPRNGProtocol) -> [Double] {
        // z ~ N(0, I)
        var y = [Double](repeating: 0.0, count: dim)
        // y = B * (D * z)
        for j in 0..<dim {
            let zj = rng.nextGaussian()
            let dj = eigensystem.D[j]
            let scaledZ = dj * zj
            for i in 0..<dim {
                y[i] += eigensystem.B[i, j] * scaledZ
            }
        }

        // x = mean + sigma * y
        var x = [Double](repeating: 0.0, count: dim)
        for i in 0..<dim {
            x[i] = mean[i] + sigma * y[i]
        }
        return x
    }

    /// Updates CMA-ES distribution from evaluated solutions in the generation.
    ///
    /// - Parameter solutions: Tuple of parameter point and evaluated objective value.
    package mutating func tell(_ solutions: [(point: [Double], value: Double)]) {
        precondition(solutions.count == populationSize, "Solutions count must match populationSize")

        generation += 1

        // Sort solutions ascending by objective value (minimization)
        var sorted = solutions
        sorted.sort { $0.value < $1.value }

        updateEigensystemIfNeeded()

        // Compute normalized displacement vectors: y_k = (x_k - mean) / sigma
        var yK = [[Double]](repeating: [], count: populationSize)
        for k in 0..<populationSize {
            var y = [Double](repeating: 0.0, count: dim)
            let pt = sorted[k].point
            for i in 0..<dim {
                y[i] = (pt[i] - mean[i]) / sigma
            }
            yK[k] = y
        }

        // Selection and recombination: y_w = sum_{i=0}^{mu-1} w_i * y_i
        var yW = [Double](repeating: 0.0, count: dim)
        for i in 0..<mu {
            let wi = weights[i]
            let yi = yK[i]
            for d in 0..<dim {
                yW[d] += wi * yi[d]
            }
        }

        // Mean update: m = m + c_m * sigma * y_w
        for i in 0..<dim {
            mean[i] += cm * sigma * yW[i]
        }

        // Step-size control: C^(-1/2) * y_w = B * diag(1/D) * B^T * y_w
        let btYw = eigensystem.B.multiplyTransposed(vec: yW)
        var scaledBtYw = [Double](repeating: 0.0, count: dim)
        for i in 0..<dim {
            scaledBtYw[i] = btYw[i] / eigensystem.D[i]
        }
        let cInvSqrtYw = eigensystem.B.multiply(vec: scaledBtYw)

        let pSigmaFactor = (cSigma * (2.0 - cSigma) * muEff).squareRoot()
        var pSigmaNormSq = 0.0
        for i in 0..<dim {
            pSigma[i] = (1.0 - cSigma) * pSigma[i] + pSigmaFactor * cInvSqrtYw[i]
            pSigmaNormSq += pSigma[i] * pSigma[i]
        }
        let normPSigma = pSigmaNormSq.squareRoot()

        // Update sigma
        sigma *= exp((cSigma / dSigma) * (normPSigma / chiN - 1.0))
        sigma = min(sigma, 1e32)

        // Covariance matrix adaptation
        let hSigmaCondLeft = normPSigma / (1.0 - pow(1.0 - cSigma, 2.0 * Double(generation + 1))).squareRoot()
        let hSigmaCondRight = (1.4 + 2.0 / (Double(dim) + 1.0)) * chiN
        let hSigma = (hSigmaCondLeft < hSigmaCondRight) ? 1.0 : 0.0

        let pcFactor = hSigma * (cc * (2.0 - cc) * muEff).squareRoot()
        for i in 0..<dim {
            pc[i] = (1.0 - cc) * pc[i] + pcFactor * yW[i]
        }

        let deltaHSigma = (1.0 - hSigma) * cc * (2.0 - cc)

        // Active CMA negative weight re-scaling (Eq. 46)
        var wIO = weights
        for k in 0..<populationSize {
            if weights[k] < 0.0 {
                let btYk = eigensystem.B.multiplyTransposed(vec: yK[k])
                var normSq = 0.0
                for i in 0..<dim {
                    let term = btYk[i] / eigensystem.D[i]
                    normSq += term * term
                }
                wIO[k] = weights[k] * (Double(dim) / (normSq + 1e-8))
            }
        }

        // Covariance decay factor
        var sumWeights = 0.0
        for w in weights {
            sumWeights += w
        }
        let decay = 1.0 + c1 * deltaHSigma - c1 - cmu * sumWeights
        C.scale(decay)

        // Rank-one update: c1 * pc * pc^T
        C.addOuterProductSymmetric(vec: pc, multiplier: c1)

        // Rank-mu update: cmu * sum(w_io_k * y_k * y_k^T)
        for k in 0..<populationSize {
            let mult = cmu * wIO[k]
            C.addOuterProductSymmetric(vec: yK[k], multiplier: mult)
        }

        needsEigenUpdate = true
    }

    /// Exports current state for checkpoint persistence.
    package func exportCheckpoint() -> CMACheckpoint {
        CMACheckpoint(
            dim: dim,
            generation: generation,
            mean: mean,
            sigma: sigma,
            pSigma: pSigma,
            pc: pc,
            cov: Array(C.buffer)
        )
    }

    /// Restores optimizer state from a checkpoint.
    package mutating func restoreCheckpoint(_ ckpt: CMACheckpoint) {
        precondition(ckpt.dim == dim, "Checkpoint dimension mismatch")
        self.generation = ckpt.generation
        self.mean = ckpt.mean
        self.sigma = ckpt.sigma
        self.pSigma = ckpt.pSigma
        self.pc = ckpt.pc
        self.C = FlatMatrix(dim: dim, buffer: ContiguousArray(ckpt.cov))
        self.needsEigenUpdate = true
    }
}
