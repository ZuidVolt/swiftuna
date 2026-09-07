import Foundation
internal import LibRustuna
import Synchronization

/// Sampler that performs dynamic exhaustive search over discrete, integer, categorical,
/// or conditional parameter spaces using a prefix decision tree.
///
/// Unlike ``GridSampler``, which requires upfront static Cartesian product declarations,
/// `BruteForceSampler` dynamically discovers parameter distributions and branching conditions
/// during execution. It traverses a decision tree, tracking explored, running, and unexpanded
/// branches, and terminates automatically with ``SwiftunaError/searchSpaceExhausted(_:)``
/// once all paths have been evaluated.
///
/// ### Example
/// ```swift
/// let sampler = BruteForceSampler(seed: 42)
/// let study = try Swiftuna.createStudy(sampler: sampler)
///
/// try study.optimize(nTrials: 100) { trial in
///     let model = try trial.suggest("model", choices: ["linear", "mlp"])
///     if model == "linear" {
///         let reg = try trial.suggest("reg", in: 0.1...0.3, step: 0.1)
///         return evaluateLinear(reg: reg)
///     } else {
///         let layers = try trial.suggest("layers", in: 1...2)
///         return evaluateMLP(layers: layers)
///     }
/// }
/// ```
public final class BruteForceSampler: CustomSampler, Sendable {
    /// Optional seed for deterministic candidate exploration order.
    public let seed: UInt64?

    /// If `true`, the sampler does not skip candidate branches currently being evaluated by concurrent workers.
    public let avoidPrematureStop: Bool

    /// If `true`, uses NumPy-compatible MT19937 PRNG and choice distribution for exact parity validation.
    public let useNumpyPRNG: Bool

    /// Thread-safe internal state protected by a Swift 6 Mutex.
    private let mutex: Mutex<BruteForceState>

    /// Initializes a BruteForceSampler.
    ///
    /// - Parameters:
    ///   - seed: Optional seed for reproducible candidate traversal order.
    ///   - avoidPrematureStop: If `true`, does not avoid candidate branches currently evaluated by concurrent workers.
    ///   - searchSpace: Optional upfront parameter search space. If omitted, discovered dynamically at runtime.
    ///   - useNumpyPRNG: If `true`, uses NumPy-compatible MT19937 PRNG for exact parity validation.
    public init(
        seed: UInt64? = nil,
        avoidPrematureStop: Bool = false,
        searchSpace: [String: [ParameterValue]]? = nil,
        useNumpyPRNG: Bool = false
    ) {
        self.seed = seed
        self.avoidPrematureStop = avoidPrematureStop
        self.useNumpyPRNG = useNumpyPRNG
        self.mutex = Mutex(
            BruteForceState(
                seed: seed,
                avoidPrematureStop: avoidPrematureStop,
                searchSpace: searchSpace,
                useNumpyPRNG: useNumpyPRNG
            )
        )
    }

    /// Retains parameter history to verify and synchronize tree state against completed trials.
    public var retainsParameterHistory: Bool {
        true
    }

    /// Returns `true` if all candidate branches in the decision tree have been evaluated.
    public var isExhausted: Bool {
        mutex.withLock { $0.isExhausted }
    }

    /// Proposes the next configuration or throws ``SwiftunaError/searchSpaceExhausted(_:)`` if all paths are explored.
    public func sample(
        history: StudyHistory,
        trialNumber: Int
    ) throws -> [String: ParameterValue] {
        try mutex.withLock { state in
            state.synchronize(with: history)

            if state.isExhausted {
                throw SwiftunaError.searchSpaceExhausted("Brute-force search space fully explored")
            }

            // Check if root has no unexpanded branches left
            if !state.tree.isAnyExpandable(nodeIndex: 0, excludeRunning: !state.avoidPrematureStop)
                && state.inFlightPaths.isEmpty
            {
                state.isExhausted = true
                throw SwiftunaError.searchSpaceExhausted("Brute-force search space fully explored")
            }

            // If static searchSpace was pre-configured, sample all dimensions directly:
            if let searchSpace = state.searchSpace, !searchSpace.isEmpty {
                var result: [String: ParameterValue] = [:]
                var curIdx = 0
                for (name, values) in searchSpace {
                    let cands = values.enumerated().map { Double($0.offset) }
                    state.tree.expand(nodeIndex: curIdx, paramName: name, candidates: cands)
                    guard
                        let chosenCand = state.tree.sampleChild(
                            nodeIndex: curIdx,
                            excludeRunning: !state.avoidPrematureStop,
                            rng: &state.rng
                        )
                    else {
                        state.isExhausted = true
                        throw SwiftunaError.searchSpaceExhausted("Brute-force search space fully explored")
                    }
                    let idx = Int(chosenCand.rounded())
                    result[name] = values[idx]
                    curIdx = state.tree.getOrCreateChild(nodeIndex: curIdx, paramName: name, candidate: chosenCand)
                }
                return result
            }

            // In define-by-run mode, parameters are sampled dynamically inside trial.suggest callbacks
            return [:]
        }
    }

    /// Underlying callback sampler that drives define-by-run suggestions.
    public var underlyingSampler: (any Sampler)? {
        makeCallbackSampler()
    }

    /// Creates the underlying ``CallbackSampler`` wiring the decision tree to suggestion callbacks.
    public func makeCallbackSampler() -> CallbackSampler {
        let onFloat: CallbackSampler.FloatFn = { [self] name, low, high, step, log, trialNumber in
            guard let step = step, step > 0 else {
                return nil
            }
            var candidates: [Double] = []
            let lowDec = Decimal(string: String(low)) ?? Decimal(low)
            let highDec = Decimal(string: String(high)) ?? Decimal(high)
            let stepDec = Decimal(string: String(step)) ?? Decimal(step)
            var cur = lowDec
            while cur <= highDec {
                candidates.append((cur as NSDecimalNumber).doubleValue)
                cur += stepDec
            }
            return mutex.withLock { state in
                state.selectCandidate(paramName: name, candidates: candidates, trialNumber: trialNumber)
            }
        }

        let onInt: CallbackSampler.IntFn = { [self] name, low, high, step, log, trialNumber in
            let s = max(1, step)
            var candidates: [Double] = []
            var cur = low
            while cur <= high {
                candidates.append(Double(cur))
                cur += s
            }
            let chosen = mutex.withLock { state in
                state.selectCandidate(paramName: name, candidates: candidates, trialNumber: trialNumber)
            }
            return Int64(chosen.rounded())
        }

        let onCategorical: CallbackSampler.CategoricalFn = { [self] name, choices, trialNumber in
            guard !choices.isEmpty else { return nil }
            let candidates = (0..<choices.count).map(Double.init)
            let chosen = mutex.withLock { state in
                state.selectCandidate(paramName: name, candidates: candidates, trialNumber: trialNumber)
            }
            let idx = Int(chosen.rounded())
            return (idx >= 0 && idx < choices.count) ? idx : 0
        }

        return CallbackSampler(
            onFloat: onFloat,
            onInt: onInt,
            onCategorical: onCategorical
        )
    }
}
