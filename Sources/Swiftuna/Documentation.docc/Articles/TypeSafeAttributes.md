# Type-Safe User Attributes & Metadata

Define compile-time schemas for trial metadata, custom enums, and arbitrary Codable structures.

## Overview

In Python Optuna, user attributes are untyped string dictionaries (`trial.set_user_attr("key", value)`). While simple, untyped strings can introduce runtime bugs, silent typos, and deserialization errors in production pipelines.

Swiftuna introduces **statically typed schemas** via ``AttributeKey`` and ``AttributeConvertible``, providing:
- Compile-time checking of key names and value types
- Native Swift subscript access on ``Trial``, ``Study``, and ``PersistedTrial``
- JSON serialization of complex Swift `Codable` structs via ``CodableAttribute``

---

## 1. Declaring Type-Safe Attribute Keys

Define keys conforming to ``AttributeKey``:

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

## 2. Writing & Reading Typed Attributes

Use standard Swift subscript syntax on active trials or studies:

```swift
try study.optimize(nTrials: 50) { trial in
    let lr = try trial.suggest("lr", in: 1e-4...1e-1)
    
    // Type-safe writes on active trial
    trial[GitCommitHash.self] = "a1b2c3d"
    trial[TotalEpochs.self] = 25
    trial[EarlyStopped.self] = false
    
    return train(lr: lr)
}

// Type-safe reads on completed trials
if let best = try study.bestTrial {
    let commit: String? = best[GitCommitHash.self]
    let epochs: Int? = best[TotalEpochs.self]
    let stopped: Bool? = best[EarlyStopped.self]
    print("Best model from commit \(commit ?? "unknown") ran for \(epochs ?? 0) epochs")
}
```

---

## 3. Custom Swift Enums as Attributes

Any `RawRepresentable` enum whose `RawValue` conforms to ``AttributeConvertible`` automatically conforms to ``AttributeConvertible``:

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

// Write enum directly
trial[ArchitectureKey.self] = .convnextSmall

// Read back enum directly
let arch: ModelArchitecture? = bestTrial[ArchitectureKey.self]
```

---

## 4. Complex Structures with `CodableAttribute`

For complex nested payloads (such as dataset partitions, hardware profiling metrics, or layer configurations), wrap any `Codable & Sendable` type in ``CodableAttribute``:

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

// Writing complex struct
let stats = HardwareStats(peakMemoryMB: 3412.5, gpuUtilization: 0.94, deviceName: "Apple M3 Max")
trial[HardwareStatsKey.self] = CodableAttribute(stats)

// Reading complex struct back
if let savedStats = bestTrial[HardwareStatsKey.self]?.value {
    print("Peak Memory: \(savedStats.peakMemoryMB) MB on \(savedStats.deviceName)")
}
```
