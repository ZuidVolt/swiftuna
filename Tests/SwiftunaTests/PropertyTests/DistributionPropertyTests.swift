import Foundation
import PropertyBased
import Testing
@testable import Swiftuna

@Suite("Distribution Property Tests")
struct DistributionPropertyTests {

    @Test("Float suggestions strictly adhere to generated ClosedRange bounds")
    func testFloatSuggestionBounds() async throws {
        await propertyCheck(
            input: Gen<Double>.double(in: -100.0...0.0),
            Gen<Double>.double(in: 0.1...100.0)
        ) { low, high in
            do {
                let study = try Swiftuna.createStudy(direction: .minimize)
                var trial = try study.ask()
                let range: ClosedRange<Double> = low...high
                let val = try trial.suggest("x", in: range)
                try study.tell(consuming: trial, value: val)

                #expect(val >= range.lowerBound)
                #expect(val <= range.upperBound)
            } catch {
                Issue.record("Unexpected suggestion failure: \(error)")
            }
        }
    }

    @Test("Integer suggestions strictly adhere to generated integer range bounds")
    func testIntSuggestionBounds() async throws {
        await propertyCheck(
            input: Gen<Int>.int(in: -50...0),
            Gen<Int>.int(in: 1...50)
        ) { low, high in
            do {
                let study = try Swiftuna.createStudy(direction: .minimize)
                var trial = try study.ask()
                let range: ClosedRange<Int> = low...high
                let val = try trial.suggest("k", in: range)
                try study.tell(consuming: trial, value: Double(val))

                #expect(val >= range.lowerBound)
                #expect(val <= range.upperBound)
            } catch {
                Issue.record("Unexpected int suggestion failure: \(error)")
            }
        }
    }

    @Test("Categorical suggestions always choose an existing element from choices")
    func testCategoricalChoicesContainment() async throws {
        let choices = ["alpha", "beta", "gamma", "delta", "epsilon"]
        await propertyCheck(input: Gen<Int>.int(in: 1...5)) { n in
            do {
                let subChoices = Array(choices.prefix(n))
                let study = try Swiftuna.createStudy(direction: .minimize)
                var trial = try study.ask()
                let chosen = try trial.suggest("opt", choices: subChoices)
                try study.tell(consuming: trial, value: 1.0)

                #expect(subChoices.contains(chosen))
            } catch {
                Issue.record("Unexpected categorical suggestion failure: \(error)")
            }
        }
    }

    @Test("Empty categorical choices cleanly throw error without crashing")
    func testEmptyCategoricalChoices() async throws {
        let empty: [String] = []
        let study = try Swiftuna.createStudy(direction: .minimize)
        var trial = try study.ask()
        #expect(throws: SwiftunaError.self) {
            _ = try trial.suggest("empty", choices: empty)
        }
    }
}
