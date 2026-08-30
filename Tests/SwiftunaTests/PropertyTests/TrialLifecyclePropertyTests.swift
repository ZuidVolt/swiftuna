import Foundation
import PropertyBased
import Testing
@testable import Swiftuna

@Suite("Trial Lifecycle Invariant Tests")
struct TrialLifecyclePropertyTests {

    @Test("Trial numbers increase monotonically and bestValue tracks the minimum")
    func testMonotonicTrialNumbersAndBestValue() async throws {
        await propertyCheck(input: Gen<Int>.int(in: 3...10)) { nTrials in
            do {
                let study = try Swiftuna.createStudy(direction: .minimize)
                var observedValues: [Double] = []

                for expectedNumber in 0..<nTrials {
                    let trial = try study.ask()
                    #expect(trial.number == expectedNumber)

                    let val = Double((expectedNumber * 17) % 31)
                    observedValues.append(val)
                    try study.tell(consuming: trial, value: val)
                }

                let allTrials = try study.trials
                #expect(allTrials.count == nTrials)

                let bestVal = try study.bestValue
                let expectedMin = observedValues.min()!
                #expect(bestVal == expectedMin)
            } catch {
                Issue.record("Lifecycle invariant violation: \(error)")
            }
        }
    }

    @Test("Dropping an un-told trial cleans up Rust resources safely")
    func testDroppedTrialSafety() throws {
        let study = try Swiftuna.createStudy(direction: .minimize)
        do {
            let unToldTrial = try study.ask()
            #expect(unToldTrial.number == 0)
        }
        // Study should remain fully operational and capable of asking for next trial
        let nextTrial = try study.ask()
        #expect(nextTrial.number == 1)
        try study.tell(consuming: nextTrial, value: 42.0)

        let bestVal = try study.bestValue
        #expect(bestVal == 42.0)
    }
}
