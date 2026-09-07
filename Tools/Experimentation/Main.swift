import Foundation
import Swiftuna

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

let D = 50
let N_BENCH_TRIALS = 250000
let SEED: UInt64 = 42

func logMsg(_ msg: String) {
    print(msg)
    fflush(stdout)
}

@inline(__always)
func rosenbrock(_ xs: [Double]) -> Double {
    var total = 0.0
    for i in 0..<(xs.count - 1) {
        let diff1 = xs[i + 1] - xs[i] * xs[i]
        let diff2 = 1.0 - xs[i]
        total += 100.0 * diff1 * diff1 + diff2 * diff2
    }
    return total
}

func getPeakMemoryMB() -> Double {
    var rusage = rusage()
    getrusage(RUSAGE_SELF, &rusage)
    #if canImport(Darwin)
        return Double(rusage.ru_maxrss) / (1024.0 * 1024.0)
    #else
        return Double(rusage.ru_maxrss) / 1024.0
    #endif
}

@main
struct ExperimentationApp {
    static func main() throws {
        let nTrials = CommandLine.arguments.count > 1 ? (Int(CommandLine.arguments[1]) ?? 100) : 100

        logMsg("============================================================")
        logMsg(" Swiftuna CMA-ES Benchmark: D=\(D), N=\(nTrials) Trials")
        logMsg("============================================================")

        let dimensions: [CMAParamDimension] = (0..<D).map { i in
            .continuous(name: "x_\(i)", lower: -5.0, upper: 5.0, log: false)
        }
        let sampler = CMASampler(dimensions: dimensions, seed: SEED)
        let study = try Swiftuna.createStudy(
            name: "swiftuna_cma_\(UUID().uuidString)",
            direction: .minimize,
            sampler: sampler
        )

        let clock = ContinuousClock()
        let start = clock.now

        try study.optimize(nTrials: nTrials) { trial in
            var xs = [Double]()
            xs.reserveCapacity(D)
            for i in 0..<D {
                xs.append(try trial.suggest("x_\(i)", in: -5.0...5.0))
            }
            return rosenbrock(xs)
        }

        let duration = clock.now - start
        let elapsed = Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
        let throughput = Double(nTrials) / elapsed
        let latencyUs = (elapsed * 1e6) / Double(nTrials)
        let peakRss = getPeakMemoryMB()
        let bestValue = try study.bestTrial?.value ?? Double.nan

        logMsg(String(format: "Elapsed Time:       %.3f s", elapsed))
        logMsg(String(format: "Throughput:         %.1f trials/s", throughput))
        logMsg(String(format: "Latency per trial:  %.2f µs", latencyUs))
        logMsg(String(format: "Best Value:         %.6e", bestValue))
        logMsg(String(format: "Peak RSS:           %.2f MB", peakRss))
        logMsg("============================================================")
    }
}
