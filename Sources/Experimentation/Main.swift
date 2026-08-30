import Foundation
import Swiftuna

@main
struct ExperimentationApp {
    static func main() throws {
        print("Creating Swiftuna study with TPE sampler...")
        let sampler = TPESampler(seed: 42)
        let study = try Swiftuna.createStudy(name: "quadratic_test", direction: .minimize, sampler: sampler)

        try study.optimize(nTrials: 20) { trial in
            let x = try trial.suggest("x", in: -10.0...10.0)
            let y = try trial.suggest("y", in: -10.0...10.0)
            let loss = pow(x - 2.0, 2) + pow(y + 5.0, 2)
            print("Trial #\(trial.number): x=\(x), y=\(y) -> loss=\(loss)")
            return loss
        }

        let bestVal = try study.bestValue
        let bestParams = try study.bestParams
        print("Optimization completed! Best value: \(bestVal), Best params: \(bestParams)")
    }
}
