import Foundation
internal import LibRustuna

public func createStudy<S: Sampler>(
    name: String = "default",
    direction: Direction = .minimize,
    sampler: S
) throws(SwiftunaError) -> Study {
    let rawSampler = sampler.makeRawHandle()
    defer {
        if let rawSampler {
            rustuna_sampler_free(rawSampler)
        }
    }

    var studyPtr: OpaquePointer?
    let status = name.withCString { cName in
        rustuna_study_new(
            cName,
            direction.rawValue,
            rawSampler,
            &studyPtr
        )
    }

    if status != 0 {
        throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to create study '\(name)'")
    }

    return Study(raw: studyPtr, name: name, direction: direction)
}

public func createStudy(
    name: String = "default",
    direction: Direction = .minimize
) throws(SwiftunaError) -> Study {
    try createStudy(name: name, direction: direction, sampler: TPESampler())
}
