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
