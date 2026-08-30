public import Foundation
internal import LibRustuna

/// Creates a single-objective study with optional storage, custom sampler, and pruner.
///
/// - Parameters:
///   - name: The unique study identifier. Defaults to `"default"`.
///   - direction: Direction of optimization (either ``Direction/minimize`` or ``Direction/maximize``). Defaults to ``Direction/minimize``.
///   - storage: Storage backend to persist study data. Defaults to ``StorageBackend/inMemory``.
///   - sampler: Sampler algorithm for parameter suggestions (e.g. ``TPESampler``, ``QMCSampler``, ``GridSampler``).
///   - pruner: Pruner algorithm for early stopping (e.g. ``MedianPruner``, ``HyperbandPruner``). Defaults to ``NopPruner``.
///   - loadIfExists: If `true`, reloads an existing study with matching `name` from `storage` instead of throwing an error. Defaults to `false`.
/// - Returns: An active ``Study`` instance ready for optimization.
/// - Throws: ``SwiftunaError/duplicatedStudy(_:)`` if a study with `name` already exists and `loadIfExists` is `false`.
///
/// ### Example
/// ```swift
/// let study = try Swiftuna.createStudy(
///     name: "hyperband_tuning",
///     direction: .minimize,
///     storage: .sqlite(path: "experiments.db"),
///     sampler: TPESampler(),
///     pruner: HyperbandPruner(),
///     loadIfExists: true
/// )
/// ```
public func createStudy<S: Sampler>(
    name: String = "default",
    direction: Direction = .minimize,
    storage: StorageBackend = .inMemory,
    sampler: S,
    pruner: any Pruner = NopPruner(),
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    try createStudy(
        name: name,
        directions: [direction],
        storage: storage,
        sampler: sampler,
        pruner: pruner,
        loadIfExists: loadIfExists
    )
}

/// Creates a single-objective study with default TPESampler and optional pruner.
///
/// - Parameters:
///   - name: The unique study identifier. Defaults to `"default"`.
///   - direction: Direction of optimization (either ``Direction/minimize`` or ``Direction/maximize``). Defaults to ``Direction/minimize``.
///   - storage: Storage backend to persist study data. Defaults to ``StorageBackend/inMemory``.
///   - pruner: Pruner algorithm for early stopping (defaults to ``NopPruner``).
///   - loadIfExists: If `true`, reloads an existing study with matching `name` from `storage` instead of throwing an error. Defaults to `false`.
/// - Returns: An active ``Study`` instance ready for optimization.
/// - Throws: ``SwiftunaError/duplicatedStudy(_:)`` if a study with `name` already exists and `loadIfExists` is `false`.
public func createStudy(
    name: String = "default",
    direction: Direction = .minimize,
    storage: StorageBackend = .inMemory,
    pruner: any Pruner = NopPruner(),
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    try createStudy(
        name: name,
        directions: [direction],
        storage: storage,
        pruner: pruner,
        loadIfExists: loadIfExists
    )
}

@inline(always)
internal func withOptionalCString<R>(
    _ string: String?,
    _ body: (UnsafePointer<CChar>?) throws -> R
) rethrows -> R {
    if let string {
        return try string.withCString { try body($0) }
    }
    return try body(nil)
}

/// Creates a multi-objective study with custom sampler and optional pruner.
public func createStudy<S: Sampler>(
    name: String = "default",
    directions: [Direction],
    storage: StorageBackend = .inMemory,
    sampler: S,
    pruner: any Pruner = NopPruner(),
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

    return Study(raw: studyPtr, name: name, directions: directions, pruner: pruner)
}

/// Creates a multi-objective study with default NSGAIISampler (if directions.count > 1) or TPESampler.
public func createStudy(
    name: String = "default",
    directions: [Direction],
    storage: StorageBackend = .inMemory,
    pruner: any Pruner = NopPruner(),
    loadIfExists: Bool = false
) throws(SwiftunaError) -> Study {
    if directions.count > 1 {
        return try createStudy(
            name: name,
            directions: directions,
            storage: storage,
            sampler: NSGAIISampler(),
            pruner: pruner,
            loadIfExists: loadIfExists
        )
    }
    return try createStudy(
        name: name,
        directions: directions,
        storage: storage,
        sampler: TPESampler(),
        pruner: pruner,
        loadIfExists: loadIfExists
    )
}

/// Loads an existing study from persistent storage with custom sampler and pruner.
public func loadStudy<S: Sampler>(
    name: String,
    storage: StorageBackend,
    sampler: S,
    pruner: any Pruner = NopPruner()
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

    return Study(raw: studyPtr, name: name, directions: [.minimize], pruner: pruner)
}

/// Loads an existing study from persistent storage with default TPESampler and optional pruner.
public func loadStudy(
    name: String,
    storage: StorageBackend,
    pruner: any Pruner = NopPruner()
) throws(SwiftunaError) -> Study {
    try loadStudy(name: name, storage: storage, sampler: TPESampler(), pruner: pruner)
}

/// Constructs an already-evaluated historical trial that can be injected into a study via `study.addTrial`.
public func createTrial(
    state: TrialState = .complete,
    value: Double? = nil,
    values: [Double] = [],
    params: [String: Double] = [:],
    userAttrs: [String: String] = [:],
    constraints: [String: Double] = [:],
    intermediateValues: [Int: Double] = [:],
    datetimeStart: Date? = nil,
    datetimeComplete: Date? = nil
) -> PersistedTrial {
    PersistedTrial(
        number: 0,
        state: state,
        value: value,
        values: values,
        params: params,
        userAttrs: userAttrs,
        constraints: constraints,
        intermediateValues: intermediateValues,
        datetimeStart: datetimeStart,
        datetimeComplete: datetimeComplete
    )
}

// MARK: - Study Lifecycle & Storage Operations

/// Copies a study to a destination storage backend, replicating all trials, directions, and attributes.
@discardableResult
public func copyStudy(
    from study: Study,
    to destination: StorageBackend,
    as newName: String? = nil
) throws(SwiftunaError) -> Study {
    try study.copy(to: destination, as: newName)
}

/// Copies a study directly between two storage backends without an active Study instance.
public func copyStudy(
    fromName: String,
    fromStorage: StorageBackend,
    toName: String? = nil,
    toStorage: StorageBackend
) throws(SwiftunaError) {
    let targetName = toName ?? fromName

    let status = fromName.withCString { cSrcName in
        targetName.withCString { cDestName in
            withOptionalCString(fromStorage.pathString) { cSrc in
                withOptionalCString(toStorage.pathString) { cDest in
                    rustuna_storage_copy_study(
                        fromStorage.rawStorageType,
                        cSrc,
                        cSrcName,
                        toStorage.rawStorageType,
                        cDest,
                        cDestName
                    )
                }
            }
        }
    }

    if status != 0 {
        throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to copy study '\(fromName)'")
    }
}

/// Returns all studies stored in the specified storage backend.
public func getStudies(in storage: StorageBackend) throws(SwiftunaError) -> [StudySummary] {
    try storage.studies()
}

/// Deletes a study from the specified storage backend by name.
public func deleteStudy(named name: String, in storage: StorageBackend) throws(SwiftunaError) {
    try storage.deleteStudy(named: name)
}
