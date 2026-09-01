public import Foundation
import LibRustuna

/// Defines the persistence storage engine for Swiftuna studies.
///
/// Swiftuna supports three storage engines:
/// - ``StorageBackend/inMemory``: Ephemeral, maximum performance, zero disk I/O.
/// - ``StorageBackend/sqlite(path:)``: SQLite3 database file, 100% byte-compatible with Python Optuna and `optuna-dashboard`.
/// - ``StorageBackend/journal(path:)``: Append-only lockless journal file optimized for massive concurrency.
///
/// ### Examples
///
/// Creating an in-memory study:
/// ```swift
/// let study = try Swiftuna.createStudy(storage: .inMemory)
/// ```
///
/// Persisting to a SQLite database (inspectable via `optuna-dashboard sqlite:///experiments.db`):
/// ```swift
/// let storage = StorageBackend.sqlite(path: "experiments.db")
/// let study = try Swiftuna.createStudy(
///     name: "production_study",
///     storage: storage,
///     loadIfExists: true
/// )
/// ```
///
/// High-throughput concurrent logging using lockless journal storage:
/// ```swift
/// let storage = StorageBackend.journal(path: "hpc_cluster.log")
/// let study = try Swiftuna.createStudy(name: "cluster_eval", storage: storage)
/// ```
public enum StorageBackend: Sendable, Equatable {
    /// Pure volatile in-memory storage.
    ///
    /// Trials are stored in RAM within the Rustuna runtime. Provides zero disk I/O overhead
    /// and fastest iteration speed for single-process jobs.
    case inMemory

    /// SQLite3 database storage.
    ///
    /// Data is stored in a standard SQLite file schema identical to Python Optuna (`RDBStorage`).
    /// You can visualize running studies in real-time by starting Optuna Dashboard:
    /// ```bash
    /// pip install optuna-dashboard
    /// optuna-dashboard sqlite:///experiments.db
    /// ```
    case sqlite(path: String)

    /// High-throughput lockless append-only journal storage.
    ///
    /// Recommended for distributed environments, NFS network filesystems, or thousands of concurrent workers
    /// where SQLite database write-locks could introduce lock contention.
    case journal(path: String)

    /// Convenience initializer creating a SQLite storage backend from a `Foundation.URL`.
    ///
    /// - Parameter url: File URL pointing to the SQLite database file.
    public static func sqlite(url: URL) -> Self {
        .sqlite(path: url.path(percentEncoded: false))
    }

    /// Convenience initializer creating a Journal storage backend from a `Foundation.URL`.
    ///
    /// - Parameter url: File URL pointing to the journal log file.
    public static func journal(url: URL) -> Self {
        .journal(path: url.path(percentEncoded: false))
    }

    internal var rawStorageType: Int32 {
        switch self {
        case .inMemory:
            return 0
        case .sqlite:
            return 1
        case .journal:
            return 2
        }
    }

    internal var pathString: String? {
        switch self {
        case .inMemory:
            return nil
        case .sqlite(let path), .journal(let path):
            return path
        }
    }
}

// MARK: - Storage Lifecycle Operations

extension StorageBackend {
    /// Retrieves metadata summaries for all studies present in this storage backend.
    ///
    /// - Returns: An array of ``StudySummary`` descriptors.
    /// - Throws: ``SwiftunaError/storageError(_:)`` if reading from storage fails.
    ///
    /// ### Example
    /// ```swift
    /// let storage = StorageBackend.sqlite(path: "experiments.db")
    /// for summary in try storage.studies() {
    ///     print("Study '\(summary.name)' has \(summary.trialCount) trials")
    /// }
    /// ```
    public func studies() throws(SwiftunaError) -> [StudySummary] {
        var jsonPtr: UnsafeMutablePointer<CChar>?
        let status = withOptionalCString(pathString) { cPath in
            rustuna_storage_get_studies_json(rawStorageType, cPath, &jsonPtr)
        }

        guard status == 0, let jsonPtr else {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to retrieve studies from storage")
        }
        defer { rustuna_string_free(jsonPtr) }

        let jsonStr = String(cString: jsonPtr)
        guard let data = jsonStr.data(using: .utf8),
            let payloads = try? JSONDecoder().decode([StudySummaryPayload].self, from: data)
        else {
            throw SwiftunaError.storageError("Failed to decode study summaries JSON payload")
        }
        return payloads.map { $0.toStudySummary() }
    }

    /// Deletes a study and all of its associated trials and attributes from this storage backend.
    ///
    /// - Parameter name: Name identifier of the study to delete.
    /// - Throws: ``SwiftunaError/studyNotFound(_:)`` if no study with `name` exists,
    ///           or ``SwiftunaError/storageError(_:)`` on storage failure.
    public func deleteStudy(named name: String) throws(SwiftunaError) {
        let status = name.withCString { cName in
            withOptionalCString(pathString) { cPath in
                rustuna_storage_delete_study(rawStorageType, cPath, cName)
            }
        }
        if status != 0 {
            throw SwiftunaError.fromLastError(fallbackCode: status, context: "Failed to delete study '\(name)'")
        }
    }

    /// Deletes a study using its ``StudySummary`` descriptor.
    ///
    /// - Parameter summary: The summary representation of the study to delete.
    /// - Throws: ``SwiftunaError`` if deletion fails.
    public func deleteStudy(_ summary: StudySummary) throws(SwiftunaError) {
        try deleteStudy(named: summary.name)
    }

    /// Synchronizes and formats SQLite tables to ensure 100% binary compatibility with
    /// Python Optuna (`optuna.storages.RDBStorage`) and `optuna-dashboard`.
    ///
    /// This formats internal system category labels into valid JSON, JSON-encodes string attributes,
    /// and aggregates mathematical constraints into Optuna-standard JSON arrays using SQLite's native JSON engine.
    ///
    /// - Parameter path: The file system path to the SQLite database.
    public static func syncWithOptunaDashboard(at path: String) {
        _ = path.withCString { cPath in
            rustuna_storage_sync_optuna_dashboard(cPath)
        }
    }
}
