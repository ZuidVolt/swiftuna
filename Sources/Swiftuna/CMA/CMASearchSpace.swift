#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

/// Defines the numerical boundary and distribution specification for one dimension.
public enum CMAParamDimension: Sendable, Equatable {
    case continuous(name: String, lower: Double, upper: Double, log: Bool = false)
    case discrete(name: String, lower: Int, upper: Int, step: Int = 1)
    case steppedFloat(name: String, lower: Double, upper: Double, step: Double)

    public var name: String {
        switch self {
        case .continuous(let n, _, _, _): return n
        case .discrete(let n, _, _, _): return n
        case .steppedFloat(let n, _, _, _): return n
        }
    }
}

/// Normalizes parameters into the unit hypercube $[0, 1]^D$ following Optuna's SearchSpaceTransform.
package struct CMASearchSpace: Sendable, Equatable {
    package let dimensions: [CMAParamDimension]
    package let dimensionNames: [String]

    package var count: Int { dimensions.count }

    package init(dimensions: [CMAParamDimension]) {
        self.dimensions = dimensions.sorted { $0.name < $1.name }
        self.dimensionNames = self.dimensions.map(\.name)
    }

    /// Transforms an external parameter dictionary into a normalized $[0, 1]^D$ coordinate.
    package func transform(_ params: [String: ParameterValue]) -> [Double]? {
        var coords = [Double](repeating: 0.0, count: count)
        for (i, dim) in dimensions.enumerated() {
            guard let p = params[dim.name] else { return nil }
            switch dim {
            case .continuous(_, let low, let high, let logScale):
                guard let val = p.asDouble else { return nil }
                if logScale {
                    let logLow = log(low)
                    let logHigh = log(high)
                    coords[i] = (log(val) - logLow) / (logHigh - logLow)
                } else {
                    coords[i] = (val - low) / (high - low)
                }
            case .discrete(_, let low, let high, _):
                guard let intVal = p.asInt else { return nil }
                // Half-step expansion matching Optuna
                let lowD = Double(low) - 0.5
                let highD = Double(high) + 0.5
                coords[i] = (Double(intVal) - lowD) / (highD - lowD)
            case .steppedFloat(_, let low, let high, _):
                guard let val = p.asDouble else { return nil }
                coords[i] = (val - low) / (high - low)
            }
            coords[i] = min(max(coords[i], 0.0), 1.0)
        }
        return coords
    }

    /// Untransforms a normalized $[0, 1]^D$ coordinate back to external ParameterValue representations.
    package func untransform(_ point: [Double]) -> [String: ParameterValue] {
        precondition(point.count == count, "Point dimension must match search space")
        var result: [String: ParameterValue] = [:]
        for (i, dim) in dimensions.enumerated() {
            let u = min(max(point[i], 0.0), 1.0)
            switch dim {
            case .continuous(let name, let low, let high, let logScale):
                if logScale {
                    let logLow = log(low)
                    let logHigh = log(high)
                    let val = exp(logLow + u * (logHigh - logLow))
                    result[name] = .double(val)
                } else {
                    let val = low + u * (high - low)
                    result[name] = .double(val)
                }
            case .discrete(let name, let low, let high, let step):
                let lowD = Double(low) - 0.5
                let highD = Double(high) + 0.5
                let rawVal = lowD + u * (highD - lowD)
                var rounded = Int(round(rawVal))
                if step > 1 {
                    rounded = low + ((rounded - low) / step) * step
                }
                let clamped = min(max(rounded, low), high)
                result[name] = .int(clamped)
            case .steppedFloat(let name, let low, let high, let step):
                let raw = low + u * (high - low)
                let stepped = low + round((raw - low) / step) * step
                let clamped = min(max(stepped, low), high)
                result[name] = .double(clamped)
            }
        }
        return result
    }
}
