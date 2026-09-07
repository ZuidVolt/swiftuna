import Foundation
import Synchronization

/// Node representation in the brute-force decision tree.
internal struct DecisionNode: Sendable {
    var paramName: String?
    var value: Double?
    var isLeaf: Bool = false
    var isRunning: Bool = false
    var candidateValues: [Double] = []
    var children: [Double: Int] = [:]
}

/// A value-type prefix decision tree tracking explored, running, and unexpanded parameter branches.
internal struct DecisionTree: Sendable {
    var nodes: [DecisionNode]

    init() {
        self.nodes = [DecisionNode(paramName: nil, value: nil)]
    }

    /// Expands `nodeIndex` with the given parameter name and candidate options.
    mutating func expand(nodeIndex: Int, paramName: String, candidates: [Double]) {
        if nodes[nodeIndex].candidateValues.isEmpty {
            nodes[nodeIndex].paramName = paramName
            nodes[nodeIndex].candidateValues = candidates
        }
    }

    /// Gets or creates a child node for `candidate` under `nodeIndex`.
    mutating func getOrCreateChild(nodeIndex: Int, paramName: String, candidate: Double) -> Int {
        if let existing = nodes[nodeIndex].children[candidate] {
            return existing
        }
        let newIdx = nodes.count
        nodes.append(DecisionNode(paramName: paramName, value: candidate, isLeaf: false, isRunning: false))
        nodes[nodeIndex].children[candidate] = newIdx
        return newIdx
    }

    /// Counts unexpanded candidate branches in the subtree rooted at `nodeIndex`.
    func countUnexpanded(nodeIndex: Int, excludeRunning: Bool) -> Int {
        let node = nodes[nodeIndex]
        if node.isLeaf {
            return 0
        }
        if node.candidateValues.isEmpty {
            return (excludeRunning && node.isRunning) ? 0 : 1
        }
        var count = 0
        for cand in node.candidateValues {
            if let childIdx = node.children[cand] {
                count += countUnexpanded(nodeIndex: childIdx, excludeRunning: excludeRunning)
            } else {
                // Not yet visited child node is an unexpanded choice
                count += 1
            }
        }
        return count
    }

    /// Returns whether any unexpanded branches exist in the subtree with short-circuiting traversal.
    func isAnyExpandable(nodeIndex: Int = 0, excludeRunning: Bool) -> Bool {
        let node = nodes[nodeIndex]
        if node.isLeaf {
            return false
        }
        if node.candidateValues.isEmpty {
            return !(excludeRunning && node.isRunning)
        }
        for cand in node.candidateValues {
            if let childIdx = node.children[cand] {
                if isAnyExpandable(nodeIndex: childIdx, excludeRunning: excludeRunning) {
                    return true
                }
            } else {
                return true
            }
        }
        return false
    }

    /// Blended uniform sampling with flat uniform sampling (alpha = 0.5) matching Optuna's `sample_child`.
    mutating func sampleChild(
        nodeIndex: Int,
        excludeRunning: Bool,
        rng: inout BruteForcePRNG
    ) -> Double? {
        let node = nodes[nodeIndex]
        guard !node.candidateValues.isEmpty else { return nil }

        let candidates = node.candidateValues
        var unexpandedCounts = [Double]()
        unexpandedCounts.reserveCapacity(candidates.count)

        for cand in candidates {
            if let childIdx = node.children[cand] {
                unexpandedCounts.append(Double(countUnexpanded(nodeIndex: childIdx, excludeRunning: excludeRunning)))
            } else {
                unexpandedCounts.append(1.0)
            }
        }

        let totalUnexpanded = unexpandedCounts.reduce(0.0, +)
        if totalUnexpanded == 0 {
            return nil
        }

        let positiveCount = unexpandedCounts.reduce(0.0) { $0 + ($1 > 0 ? 1.0 : 0.0) }
        let alpha = 0.5

        var weights = [Double](repeating: 0.0, count: candidates.count)
        for i in 0..<candidates.count {
            let wOrig = unexpandedCounts[i] / totalUnexpanded
            let wFlat = unexpandedCounts[i] > 0 ? (1.0 / positiveCount) : 0.0
            weights[i] = (1.0 - alpha) * wOrig + alpha * wFlat
        }

        // Prioritize non-running and unexpanded candidates if available
        if excludeRunning {
            var hasNonRunningPositive = false
            for i in 0..<candidates.count {
                if weights[i] > 0 {
                    if let childIdx = node.children[candidates[i]] {
                        if !nodes[childIdx].isRunning {
                            hasNonRunningPositive = true
                            break
                        }
                    } else {
                        hasNonRunningPositive = true
                        break
                    }
                }
            }
            if hasNonRunningPositive {
                for i in 0..<candidates.count {
                    if let childIdx = node.children[candidates[i]], nodes[childIdx].isRunning {
                        weights[i] = 0.0
                    }
                }
            }
        }

        let sumWeights = weights.reduce(0.0, +)
        guard sumWeights > 0 else {
            return candidates.first
        }

        let u = rng.nextUniform()
        var cumulative = 0.0
        for i in 0..<candidates.count {
            cumulative += weights[i] / sumWeights
            if u < cumulative || i == candidates.count - 1 {
                return candidates[i]
            }
        }
        return candidates.last
    }
}

/// PRNG abstraction supporting default high-performance SplitMix64 or NumPy MT19937.
internal enum BruteForcePRNG: Sendable {
    case splitMix(SplitMix64)
    case numpy(NumpyMT19937PRNG)

    mutating func nextUniform() -> Double {
        switch self {
        case .splitMix(var sm):
            let raw = sm.next() & 0x1f_ffff_ffff_ffff
            let val = Double(raw) / Double(0x20_0000_0000_0000)
            self = .splitMix(sm)
            return val
        case .numpy(var np):
            let val = np.nextUniform()
            self = .numpy(np)
            return val
        }
    }
}

/// Internal thread-safe state container for ``BruteForceSampler``.
internal struct BruteForceState: Sendable {
    var tree: DecisionTree = DecisionTree()
    var inFlightPaths: [Int: [Int]] = [:]
    var isExhausted: Bool = false
    var rng: BruteForcePRNG
    var avoidPrematureStop: Bool
    var searchSpace: [String: [ParameterValue]]?

    init(
        seed: UInt64? = nil,
        avoidPrematureStop: Bool = false,
        searchSpace: [String: [ParameterValue]]? = nil,
        useNumpyPRNG: Bool = false
    ) {
        if useNumpyPRNG {
            self.rng = .numpy(NumpyMT19937PRNG(seed: UInt32(truncatingIfNeeded: seed ?? 42)))
        } else {
            self.rng = .splitMix(SplitMix64(seed: seed ?? 42))
        }
        self.avoidPrematureStop = avoidPrematureStop
        self.searchSpace = searchSpace
    }

    /// Finalizes in-flight trials up to `trialNumber`.
    mutating func finalizePreceding(upTo trialNumber: Int) {
        let keysToFinalize = inFlightPaths.keys.filter { $0 < trialNumber }
        for key in keysToFinalize {
            if let path = inFlightPaths.removeValue(forKey: key), let lastNode = path.last {
                tree.nodes[lastNode].isLeaf = true
                for idx in path {
                    tree.nodes[idx].isRunning = false
                }
            }
        }
        if !tree.isAnyExpandable(nodeIndex: 0, excludeRunning: false) && inFlightPaths.isEmpty {
            isExhausted = true
        }
    }

    /// Synchronizes tree state with finished trials from ``StudyHistory``.
    mutating func synchronize(with history: StudyHistory) {
        if inFlightPaths.isEmpty {
            return
        }
        for trial in history.all.reversed() {
            if let path = inFlightPaths.removeValue(forKey: trial.number), let lastNode = path.last {
                tree.nodes[lastNode].isLeaf = true
                for idx in path {
                    tree.nodes[idx].isRunning = false
                }
            }
            if inFlightPaths.isEmpty {
                break
            }
        }
        if inFlightPaths.isEmpty && !tree.isAnyExpandable(nodeIndex: 0, excludeRunning: false) {
            isExhausted = true
        }
    }

    /// Selects the next candidate value for `paramName` from `candidates` during define-by-run suggestions.
    mutating func selectCandidate(
        paramName: String,
        candidates: [Double],
        trialNumber: Int
    ) -> Double {
        finalizePreceding(upTo: trialNumber)

        let excludeRunning = !avoidPrematureStop

        // Current parent node for this trial
        let path = inFlightPaths[trialNumber] ?? [0]
        let parentIdx = path.last ?? 0

        // Expand parent with candidate set if needed
        tree.expand(nodeIndex: parentIdx, paramName: paramName, candidates: candidates)

        let chosen: Double
        if let cand = tree.sampleChild(nodeIndex: parentIdx, excludeRunning: excludeRunning, rng: &rng) {
            chosen = cand
        } else {
            // Space from this parent is exhausted; pick fallback deterministically so active trial can finish
            isExhausted = true
            chosen = candidates.first ?? 0.0
        }

        let childIdx = tree.getOrCreateChild(nodeIndex: parentIdx, paramName: paramName, candidate: chosen)
        tree.nodes[childIdx].isRunning = true

        var newPath = path
        newPath.append(childIdx)
        inFlightPaths[trialNumber] = newPath

        return chosen
    }
}

/// SplitMix64 PRNG for deterministic candidate shuffling and sampling.
internal struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z &>> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z &>> 31)
    }
}
