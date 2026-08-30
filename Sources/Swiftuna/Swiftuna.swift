import Foundation
internal import LibRustuna

/// Creates a single-objective study with optional storage and custom sampler.
public func createStudy<S: Sampler>(
    name: String = "default",
    direction: Direction = .minimize,
    storage: StorageBackend = .inMemory,
    sampler: S,
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    try createStudy(
        name: name,
        directions: [direction],
        storage: storage,
        sampler: sampler,
        loadIfExists: loadIfExists
    )
}

/// Creates a single-objective study with default TPESampler.
public func createStudy(
    name: String = "default",
    direction: Direction = .minimize,
    storage: StorageBackend = .inMemory,
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    try createStudy(
        name: name,
        direction: direction,
        storage: storage,
        sampler: TPESampler(),
        loadIfExists: loadIfExists
    )
}

@inline(__always)
internal func withOptionalCString<R>(
    _ string: String?,
    _ body: (UnsafePointer<CChar>?) throws -> R
) rethrows -> R {
    if let string {
        return try string.withCString { try body($0) }
    } else {
        return try body(nil)
    }
}

/// Creates a multi-objective study with custom sampler.
public func createStudy<S: Sampler>(
    name: String = "default",
    directions: [Direction],
    storage: StorageBackend = .inMemory,
    sampler: S,
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    let rawSampler = sampler.makeRawHandle()
    defer {
        if let rawSampler {
            rustuna_sampler_free(rawSampler)
        }
    }

    var studyPtr: OpaquePointer?
    let dirInts: [Int32] = directions.map(\.rawValue)

    let status = name.withCString { cName in
        dirInts.withUnsafeBufferPointer { dirBuf in
            withOptionalCString(storage.pathString) { cPath in
                rustuna_study_create_full(
                    cName,
                    dirBuf.baseAddress,
                    directions.count,
                    storage.rawStorageType,
                    cPath,
                    loadIfExists,
                    rawSampler,
                    &studyPtr
                )
            }
        }
    }

    if status != 0 {
        throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to create study '\(name)'")
    }

    return Study(raw: studyPtr, name: name, directions: directions)
}

/// Creates a multi-objective study with default NSGAIISampler (if directions.count > 1) or TPESampler.
public func createStudy(
    name: String = "default",
    directions: [Direction],
    storage: StorageBackend = .inMemory,
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    if directions.count > 1 {
        return try createStudy(
            name: name,
            directions: directions,
            storage: storage,
            sampler: NSGAIISampler(),
            loadIfExists: loadIfExists
        )
    } else {
        return try createStudy(
            name: name,
            directions: directions,
            storage: storage,
            sampler: TPESampler(),
            loadIfExists: loadIfExists
        )
    }
}

/// Loads an existing study from persistent storage.
public func loadStudy<S: Sampler>(
    name: String,
    storage: StorageBackend,
    sampler: S
) throws(SwiftunaError) -> Study {
    let rawSampler = sampler.makeRawHandle()
    defer {
        if let rawSampler {
            rustuna_sampler_free(rawSampler)
        }
    }

    var studyPtr: OpaquePointer?
    let status: Int32 = name.withCString { cName in
        withOptionalCString(storage.pathString) { cPath in
            rustuna_study_load(
                cName,
                storage.rawStorageType,
                cPath,
                rawSampler,
                &studyPtr
            )
        }
    }

    if status != 0 {
        throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to load study '\(name)'")
    }

    return Study(raw: studyPtr, name: name, directions: [.minimize])
}

/// Loads an existing study from persistent storage with default TPESampler.
public func loadStudy(
    name: String,
    storage: StorageBackend
) throws(SwiftunaError) -> Study {
    try loadStudy(name: name, storage: storage, sampler: TPESampler())
}

