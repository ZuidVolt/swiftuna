import Foundation
import PropertyBased
import Testing
@testable import Swiftuna

@Suite("Enqueue, Importance & Pipeline Property Tests")
struct EnqueueAndImportancePropertyTests {

    @Test("Enqueue invariants: kinds × FIFO × partial fixing × user attributes round-trip")
    func testEnqueueInvariants() async throws {
        await propertyCheck(
            input: Gen<Double>.double(in: -100.0...100.0),
            Gen<Int>.int(in: -1000...1000),
            Gen<Bool>.bool()
        ) { targetWeight, targetStep, targetFlag in
            do {
                let study = try Swiftuna.createStudy(direction: .minimize)

                // Enqueue Trial 0: Fully specified heterogeneous configuration with user attributes
                let optChoice = targetFlag ? "sgd" : "adam"
                try study.enqueue(
                    [
                        "weight": .double(targetWeight),
                        "step": .int(targetStep),
                        "opt": .string(optChoice),
                        "flag": .bool(targetFlag),
                    ],
                    userAttrs: ["tag": "baseline_\(targetStep)"]
                )

                // Enqueue Trial 1: Partially specified configuration (only step fixed)
                try study.enqueue(["step": .int(targetStep + 10)])

                // Ask Trial 0: Verify exact parameter fixing & FIFO identity
                var trial0 = try study.ask()
                #expect(trial0.number == 0)
                let w0 = try trial0.suggest("weight", in: -10000.0...10000.0)
                let s0 = try trial0.suggest("step", in: -10000...10000)
                let o0 = try trial0.suggest("opt", choices: ["adam", "sgd"])
                let f0 = try trial0.suggest("flag", choices: [true, false])

                #expect(abs(w0 - targetWeight) < 1e-12)
                #expect(s0 == targetStep)
                #expect(o0 == optChoice)
                #expect(f0 == targetFlag)
                try study.tell(consuming: trial0, value: 1.0)

                // Ask Trial 1: Partially fixed parameter matches; unfixed parameter samples successfully
                var trial1 = try study.ask()
                #expect(trial1.number == 1)
                let s1 = try trial1.suggest("step", in: -10000...10000)
                let w1 = try trial1.suggest("weight", in: -10000.0...10000.0)
                #expect(s1 == targetStep + 10)
                #expect((-10000.0...10000.0).contains(w1))
                try study.tell(consuming: trial1, value: 2.0)

                // Verify storage read-back and attribute preservation
                let trials = try study.trials
                #expect(trials.count == 2)
                #expect(trials[0].userAttrs["tag"] == "baseline_\(targetStep)")
                #expect(trials[0].params["flag"]?.asBool == targetFlag)
                #expect(trials[0].params["opt"]?.asString == optChoice)
                #expect(trials[1].params["step"]?.asInt == targetStep + 10)
            } catch {
                Issue.record("Enqueue invariant violation: \(error)")
            }
        }
    }

    @Test("Sequence sorting and top-N pipeline mathematical invariants")
    func testSequencePipelineInvariants() async throws {
        await propertyCheck(input: Gen<Int>.int(in: 3...15)) { count in
            var mockTrials: [PersistedTrial] = []
            for i in 0..<count {
                let val = Double((i * 37 + 11) % 43)
                let trial = PersistedTrial(
                    number: i,
                    state: .complete,
                    value: val,
                    params: ["p1": val * 2.0, "p2": -val]
                )
                mockTrials.append(trial)
            }

            // Invariant 1: sortedByValue is monotonic non-decreasing
            let sortedAsc = mockTrials.sortedByValue(ascending: true)
            #expect(sortedAsc.count == count)
            for i in 0..<(sortedAsc.count - 1) {
                #expect(sortedAsc[i].value! <= sortedAsc[i + 1].value!)
            }

            // Invariant 2: sortedByValue(ascending: false) is monotonic non-increasing
            let sortedDesc = mockTrials.sortedByValue(ascending: false)
            for i in 0..<(sortedDesc.count - 1) {
                #expect(sortedDesc[i].value! >= sortedDesc[i + 1].value!)
            }

            // Invariant 3: top(k) length is min(k, count)
            let k = 3
            let topK = mockTrials.top(k)
            #expect(topK.count == min(k, count))
            #expect(topK.first!.value == sortedAsc.first!.value)

            // Invariant 4: parameterIntervals bounds contain all trial values
            let intervals = mockTrials.parameterIntervals()
            let p1Range = intervals["p1"]!
            let p2Range = intervals["p2"]!

            for trial in mockTrials {
                #expect(p1Range.contains(trial.params["p1"]!.asDouble!))
                #expect(p2Range.contains(trial.params["p2"]!.asDouble!))
            }

            // Invariant 5: within(tolerance:of:) bounds
            let minVal = sortedAsc.first!.value!
            let tol = 10.0
            let withinTol = mockTrials.within(tolerance: tol, of: minVal)
            for trial in withinTol {
                #expect(trial.value! <= minVal + tol)
                #expect(trial.value! >= minVal)
            }
        }
    }

    @Test("PED-ANOVA sensitivity ordering and normalization invariants")
    func testPedAnovaSensitivityInvariants() async throws {
        await propertyCheck(input: Gen<Double>.double(in: 20.0...80.0)) { ratio in
            do {
                let study = try Swiftuna.createStudy(
                    name: "sens_test_\(Int(ratio))",
                    sampler: TPESampler(seed: 101)
                )

                // Optimize anisotropic quadratic objective: ratio * x1^2 + x2^2
                try study.optimize(nTrials: 35) { (trial: inout Trial) throws(SwiftunaError) -> Double in
                    let x1 = try trial.suggest("high_sensitivity", in: -5.0...5.0)
                    let x2 = try trial.suggest("low_sensitivity", in: -5.0...5.0)
                    return ratio * (x1 * x1) + (x2 * x2)
                }

                let importances = try study.paramImportances(normalize: true).get()

                let iHigh = try #require(importances["high_sensitivity"])
                let iLow = try #require(importances["low_sensitivity"])

                // Invariant 1: High sensitivity parameter dominates importance
                #expect(iHigh > iLow)

                // Invariant 2: Both importances are non-negative
                #expect(iHigh >= 0.0)
                #expect(iLow >= 0.0)

                // Invariant 3: Normalized importances sum to 1.0 (within numerical tolerance)
                #expect(abs((iHigh + iLow) - 1.0) < 1e-4)
            } catch {
                Issue.record("PED-ANOVA sensitivity property violation: \(error)")
            }
        }
    }
}
