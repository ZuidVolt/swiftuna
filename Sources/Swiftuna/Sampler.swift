import Foundation
internal import LibRustuna

/// A strategy for proposing parameter values across optimization trials.
public protocol Sampler: Sendable {
    /// Creates an opaque pointer to the underlying Rustuna sampler engine.
    func makeRawHandle() -> OpaquePointer?
}

/// Sampler using the Tree-structured Parzen Estimator (TPE) algorithm.
///
/// On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) `l(x)` to
/// parameter values associated with the best-performing objective observations, and another
/// GMM `g(x)` to the remaining observations. It selects the candidate `x` maximizing the ratio `l(x) / g(x)`.
///
/// For multi-objective optimization, TPE (MOTPE) uses non-domination ranks and hypervolume
/// contributions to partition observations into the good and poor models.
///
/// > Note:
/// > Mathematical constraints set via ``Trial/setConstraints(_:)``
/// > are respected during observation partitioning: feasible trials are prioritized over infeasible ones,
/// > and infeasible trials are ordered by their total violation magnitude.
///
/// ### Example
/// ```swift
/// let sampler = TPESampler(seed: 42)
/// let study = try Swiftuna.createStudy(sampler: sampler)
/// try study.optimize(nTrials: 100) { trial in
///     let x = try trial.suggest("x", in: -10.0...10.0)
///     return x * x
/// }
/// ```
public struct TPESampler: Sampler {
    /// Optional seed for the pseudorandom number generator. If `nil`, a non-deterministic seed is generated.
    public let seed: UInt64?

    /// Joint-sampling mode. `nil` (default) selects automatically, matching
    /// Optuna: multivariate for single-objective studies, independent for
    /// multi-objective studies.
    public let multivariate: Bool?

    /// Completed trials collected before TPE engages (earlier trials sample
    /// randomly). Defaults to `10`, matching the engine.
    public let nStartupTrials: Int

    /// Initializes a TPE sampler.
    ///
    /// - Parameters:
    ///   - seed: Seed for random number generation. If `nil`, a random seed is selected.
    ///   - multivariate: `nil` for automatic selection, `true` to force joint
    ///     sampling, `false` to force independent sampling.
    ///   - nStartupTrials: Completed trials before TPE engages. Defaults to `10`.
    public init(seed: UInt64? = nil, multivariate: Bool? = nil, nStartupTrials: Int = 10) {
        self.seed = seed
        self.multivariate = multivariate
        self.nStartupTrials = nStartupTrials
    }

    public func makeRawHandle() -> OpaquePointer? {
        // -1 auto, 0 independent, 1 joint.
        let multivariateFlag = Int8(multivariate.map { $0 ? 1 : 0 } ?? -1)
        // Negative counts would wrap as huge usize across the ABI.
        let startup = max(0, nStartupTrials)
        if let s = seed {
            return rustuna_sampler_tpe_full(s, true, multivariateFlag, startup)
        }
        return rustuna_sampler_tpe_full(0, false, multivariateFlag, startup)
    }
}

/// Sampler using uniform random search across the parameter space.
///
/// Evaluates parameter values independently and uniformly at random without utilizing historical trial feedback.
///
/// ### Example
/// ```swift
/// let sampler = RandomSampler(seed: 123)
/// let study = try Swiftuna.createStudy(sampler: sampler)
/// ```
public struct RandomSampler: Sampler {
    /// Optional seed for the pseudorandom number generator.
    public let seed: UInt64?

    /// Initializes a random sampler.
    ///
    /// - Parameter seed: Optional seed for reproducibility. If `nil`, a non-deterministic seed is selected.
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

/// Sampler using the NSGA-II (Non-dominated Sorting Genetic Algorithm II) evolutionary algorithm.
///
/// NSGA-II is designed for multi-objective optimization. It maintains a population of candidates across
/// generations, using fast non-dominated sorting to rank solutions and crowding distance to preserve diversity
/// along the Pareto-optimal frontier.
///
/// > Note:
/// > When mathematical constraints are set via ``Trial/setConstraints(_:)``, NSGA-II applies
/// > constrained-domination: feasible solutions strictly dominate infeasible ones, and infeasible solutions
/// > are ranked according to their constraint violation sums.
///
/// ### Example
/// ```swift
/// let sampler = NSGAIISampler(populationSize: 40, crossoverProb: 0.9, seed: 42)
/// let study = try Swiftuna.createStudy(directions: [.minimize, .minimize], sampler: sampler)
/// try study.optimize(nTrials: 100) { trial in
///     let x = try trial.suggest("x", in: -5.0...5.0)
///     let y = try trial.suggest("y", in: -5.0...5.0)
///     return [x * x, (x - 2.0) * (x - 2.0) + y * y]
/// }
/// ```
public struct NSGAIISampler: Sampler {
    /// Number of candidate individuals maintained in each generation.
    public let populationSize: Int

    /// Probability of mutating each parameter of a candidate individual.
    public let mutationProb: Double?

    /// Probability of performing crossover between two parent individuals.
    public let crossoverProb: Double

    /// Probability of swapping each parameter value during genetic crossover.
    public let swappingProb: Double

    /// Optional seed for deterministic reproducibility.
    public let seed: UInt64?

    /// Initializes an NSGA-II multi-objective genetic sampler.
    ///
    /// - Parameters:
    ///   - populationSize: Number of individuals per generation. Defaults to `50`.
    ///   - mutationProb: Probability of mutating parameters. If `nil`, defaults to `1.0 / nParams`.
    ///   - crossoverProb: Probability of crossover between two parents. Defaults to `0.9`.
    ///   - swappingProb: Probability of swapping parameter values during crossover. Defaults to `0.5`.
    ///   - seed: Optional seed for reproducibility.
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

/// A Quasi-Monte Carlo (QMC) Sampler that generates low-discrepancy Sobol sequences.
///
/// QMC systematically minimizes search space voids and clusters across continuous and
/// categorical parameters using Antonov-Saleev Gray code and Joe-Kuo direction numbers
/// up to 1,024 dimensions. It provides faster space-filling convergence than pure uniform random sampling.
///
/// ### Example
/// ```swift
/// let sampler = QMCSampler(seed: 42)
/// let study = try Swiftuna.createStudy(
///     name: "qmc_exploration",
///     sampler: sampler
/// )
/// ```
public struct QMCSampler: Sampler {
    /// Optional seed for deterministic reproducibility and scrambling.
    public let seed: UInt64?

    /// Creates a QMC Sobol sequence sampler.
    ///
    /// - Parameter seed: Optional seed for the fallback random sampler used when
    ///   parameters fall outside the joint search space or when scrambling is enabled.
    public init(seed: UInt64? = nil) {
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        rustuna_sampler_qmc_new(seed ?? 0, seed != nil)
    }
}

/// Exhaustive grid search sampler over a discrete, integer, or categorical parameter grid.
///
/// Precomputes the Cartesian product of all provided parameter domains and evaluates each
/// combination without replacement. When an optional seed is provided, the evaluation order
/// of the grid points is shuffled deterministically.
///
/// If an optimization loop attempts to evaluate more trials than the total number of grid combinations,
/// ``Study/optimize(nTrials:timeout:objective:)-3gyl5`` will terminate early, and manual ``Study/ask()``
/// will throw ``SwiftunaError/searchSpaceExhausted(_:)``.
///
/// ### Example
/// ```swift
/// let searchSpace: [String: GridSampler.ValueList] = [
///     "learning_rate": [0.001, 0.01, 0.1],
///     "batch_size": [16, 32, 64],
///     "optimizer": .init(categorical: ["adam", "sgd"])
/// ]
///
/// let sampler = GridSampler(searchSpace: searchSpace, seed: 123)
/// let study = try Swiftuna.createStudy(sampler: sampler)
/// try study.optimize(nTrials: 18) { trial in
///     let lr = try trial.suggest("learning_rate", in: 0.001...0.1)
///     let batch = try trial.suggest("batch_size", in: 16...64)
///     let opt = try trial.suggest("optimizer", from: ["adam", "sgd"])
///     return trainModel(lr: lr, batch: batch, opt: opt)
/// }
/// ```
public struct GridSampler: Sampler {
    /// Helper structure to define discrete parameter domains for grid search.
    public struct ValueList: ExpressibleByArrayLiteral, Sendable {
        /// Numerical representations of the parameter options.
        public let values: [Double]

        /// Creates a parameter grid domain from a floating-point array literal.
        public init(arrayLiteral elements: Double...) {
            self.values = elements
        }

        /// Creates a parameter grid domain from an array of `Double` values.
        public init(_ elements: [Double]) {
            self.values = elements
        }

        /// Creates a parameter grid domain from an array of `Int` values.
        public init(_ elements: [Int]) {
            self.values = elements.map(Double.init)
        }

        /// Creates a parameter grid domain from an array of categorical `String` choices.
        public init(categorical: [String]) {
            self.values = (0..<categorical.count).map(Double.init)
        }
    }

    /// The discrete parameter domains forming the grid Cartesian product.
    public let searchSpace: [String: [Double]]

    /// Optional seed for deterministic shuffling of the grid evaluation sequence.
    public let seed: UInt64?

    /// Initializes a GridSampler with typed ``ValueList`` domains.
    ///
    /// - Parameters:
    ///   - searchSpace: Dictionary mapping parameter names to their list of valid discrete values.
    ///   - seed: Optional seed for deterministic shuffling.
    public init(searchSpace: [String: ValueList], seed: UInt64? = nil) {
        self.searchSpace = searchSpace.mapValues { $0.values }
        self.seed = seed
    }

    /// Initializes a GridSampler with raw `[Double]` domains.
    ///
    /// - Parameters:
    ///   - searchSpace: Dictionary mapping parameter names to their list of floating-point values.
    ///   - seed: Optional seed for deterministic shuffling.
    public init(searchSpace: [String: [Double]], seed: UInt64? = nil) {
        self.searchSpace = searchSpace
        self.seed = seed
    }

    public func makeRawHandle() -> OpaquePointer? {
        guard let data = try? JSONSerialization.data(withJSONObject: searchSpace, options: []),
            let jsonStr = String(data: data, encoding: .utf8)
        else {
            return nil
        }

        return jsonStr.withCString { cStr in
            var raw: OpaquePointer?
            let code = rustuna_sampler_grid_new(cStr, seed ?? 0, seed != nil, &raw)
            return code == 0 ? raw : nil
        }
    }
}
