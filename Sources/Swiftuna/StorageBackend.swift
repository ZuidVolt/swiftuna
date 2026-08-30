public import Foundation

/// Defines the underlying persistence storage engine for a Swiftuna study.
public enum StorageBackend: Sendable, Equatable {
    /// Pure volatile in-memory storage (fastest, zero disk I/O).
    case inMemory

    /// SQLite3 database storage (100% byte-compatible with Python Optuna and `optuna-dashboard`).
    case sqlite(path: String)

    /// High-throughput lockless append-only journal storage.
    case journal(path: String)

    /// Convenience initializer using a `Foundation.URL` for SQLite.
    public static func sqlite(url: URL) -> StorageBackend {
        .sqlite(path: url.path(percentEncoded: false))
    }

    /// Convenience initializer using a `Foundation.URL` for Journal.
    public static func journal(url: URL) -> StorageBackend {
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

import LibRustuna

extension StorageBackend {
    /// Returns all studies stored in this storage backend.
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
              let payloads = try? JSONDecoder().decode([StudySummaryPayload].self, from: data) else {
            throw SwiftunaError.storageError("Failed to decode study summaries JSON payload")
        }
        return payloads.map { $0.toStudySummary() }
    }

    /// Deletes a study from this storage backend by name.
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

    /// Deletes a study using its summary reference.
    public func deleteStudy(_ summary: StudySummary) throws(SwiftunaError) {
        try deleteStudy(named: summary.name)
    }
}
