# Type-safe user attributes and metadata

Declare compile-time schemas for trial metadata, custom enums, and Codable structures.

## Overview

In Python Optuna, trial attributes are untyped string dictionaries (`trial.set_user_attr("key", value)`). While simple, untyped strings can introduce silent typos, runtime casting crashes, and broken assumptions when metadata schemas evolve.

Swiftuna introduces **type-safe attribute schemas** via ``AttributeKey`` and ``AttributeConvertible``:
- Key names and value types are verified at compile time with autocomplete support.
- Native Swift subscript syntax works uniformly on ``Trial``, ``Study``, and ``PersistedTrial``.
- Swift `RawRepresentable` enums convert automatically without boilerplate.
- Complex `Codable` structs serialize to JSON and deserialize safely using ``CodableAttribute``.

---

## 1. Declaring scalar attribute keys

Create empty enum types conforming to ``AttributeKey`` to declare schema keys:

```swift
import Swiftuna

// Primitive scalar keys
public enum GitCommitHash: AttributeKey {
    public typealias Value = String
    public static let name = "git_commit"
}

public enum TotalEpochs: AttributeKey {
    public typealias Value = Int
    public static let name = "total_epochs"
}

public enum EarlyStopped: AttributeKey {
    public typealias Value = Bool
    public static let name = "early_stopped"
}
```

---

## 2. Writing and reading typed attributes

Use typed subscript syntax on active trials or completed trial records:

```swift
try study.optimize(nTrials: 50) { trial in
    let lr = try trial.suggest("lr", in: 1e-4...1e-1)
    
    // Type-safe writes on active trial
    trial[GitCommitHash.self] = "a1b2c3d"
    trial[TotalEpochs.self] = 25
    trial[EarlyStopped.self] = false
    
    return train(lr: lr)
}

// Type-safe reads on completed trial
if let best = try study.bestTrial {
    let commit: String? = best[GitCommitHash.self]
    let epochs: Int? = best[TotalEpochs.self]
    let stopped: Bool? = best[EarlyStopped.self]
    print("Best model from commit \(commit ?? "unknown") ran for \(epochs ?? 0) epochs")
}
```

---

## 3. Custom Swift enums as attributes

Any `RawRepresentable` enum whose `RawValue` conforms to ``AttributeConvertible`` (like `String` or `Int`) conforms automatically:

```swift
public enum ModelArchitecture: String, AttributeConvertible, Sendable {
    case resnet18
    case resnet50
    case convnextSmall
}

public enum ArchitectureKey: AttributeKey {
    public typealias Value = ModelArchitecture
    public static let name = "architecture"
}

// Write enum directly to active trial
trial[ArchitectureKey.self] = .convnextSmall

// Read back strongly-typed enum from completed trial
let arch: ModelArchitecture? = bestTrial[ArchitectureKey.self]
```

---

## 4. Nested structures with CodableAttribute

When you need to store structured metadata (such as profiling summaries, hardware configurations, or tensor shapes), wrap any `Codable & Sendable` struct in ``CodableAttribute``:

```swift
public struct HardwareStats: Codable, Sendable {
    public let peakMemoryMB: Double
    public let gpuUtilization: Double
    public let deviceName: String
}

public enum HardwareStatsKey: AttributeKey {
    public typealias Value = CodableAttribute<HardwareStats>
    public static let name = "hardware_stats"
}

// Write structured JSON payload
let stats = HardwareStats(peakMemoryMB: 3412.5, gpuUtilization: 0.94, deviceName: "Apple M3 Max")
trial[HardwareStatsKey.self] = CodableAttribute(stats)

// Read payload back from trial record
if let savedStats = bestTrial[HardwareStatsKey.self]?.value {
    print("Peak Memory: \(savedStats.peakMemoryMB) MB on \(savedStats.deviceName)")
}
```
