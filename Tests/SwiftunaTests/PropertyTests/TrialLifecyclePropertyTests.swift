import Foundation
import PropertyBased
import Testing
@testable import Swiftuna

@Suite("Trial Lifecycle Invariant Tests")
struct TrialLifecyclePropertyTests {

    @Test("Trial numbers increase monotonically, un-told drops are safe, and bestValue tracks optimal across directions")
    func testTrialLifecycleInvariants() async throws {
        await propertyCheck(
            input: Gen<Int>.int(in: 4...12)
        ) { nTrials in
            do {
                let study = try Swiftuna.createStudy(direction: .minimize)
                var toldValues: [Double] = []

                for expectedNumber in 0..<nTrials {
                    // Intentionally drop the trial un-told on trial index 1 when nTrials >= 5
                    if expectedNumber == 1 && nTrials >= 5 {
                        let unToldTrial = try study.ask()
                        #expect(unToldTrial.number == expectedNumber)
                        // Dropped here cleanly without tell
                    } else {
                        let trial = try study.ask()
                        #expect(trial.number == expectedNumber)

                        let val = Double((expectedNumber * 17) % 31) - 15.0
                        toldValues.append(val)
                        try study.tell(consuming: trial, value: val)
                    }
                }

                let allTrials = try study.trials
                #expect(allTrials.count == nTrials)
                #expect(allTrials.completed().count == toldValues.count)

                let bestVal = try study.bestValue
                let expectedBest = toldValues.min()!
                #expect(bestVal == expectedBest)
            } catch {
                Issue.record("Lifecycle invariant violation: \(error)")
            }
        }
    }
}
