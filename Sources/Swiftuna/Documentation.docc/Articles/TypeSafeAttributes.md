# Type-safe user attributes and metadata

Declare compile-time schemas for trial metadata, custom enums, and Codable structures with autocomplete.

## Overview

In Python Optuna, trial attributes are untyped string dictionaries (`trial.set_user_attr("key", value)`). While simple, untyped strings can introduce silent typos, runtime casting crashes, and broken assumptions when metadata schemas evolve.

Swiftuna introduces **type-safe attribute schemas** via ``AttributeKey`` and ``AttributeConvertible``:
- Key names and value types are verified at compile time with dot-syntax autocomplete support.
- Native Swift subscript syntax works uniformly on ``Trial``, ``Study``, and ``PersistedTrial``.
- Swift `RawRepresentable` enums convert automatically without boilerplate.
- Complex `Codable` structs serialize to JSON and deserialize safely using ``CodableAttribute``.
- Untyped string access is always available when dynamic behavior is preferred.

---

## 1. Declaring attribute keys with dot-syntax

Extend ``AttributeKey`` to declare schema keys with exact value types:

```swift
import Swiftuna

// Declare static keys for dot-syntax autocomplete
extension AttributeKey where Value == String {
    static let gitCommit = AttributeKey<String>("git_commit")
}

extension AttributeKey where Value == Int {
    static let totalEpochs = AttributeKey<Int>("total_epochs")
}

extension AttributeKey where Value == Bool {
    static let earlyStopped = AttributeKey<Bool>("early_stopped")
}
```

---

## 2. Writing and reading typed attributes

Use dot-syntax subscripts directly on active trials or completed trial records:

```swift
try study.optimize(nTrials: 50) { trial in
    let lr = try trial.suggest("lr", in: 1e-4...1e-1)
    
    // Type-safe writes on active trial with dot-syntax
    trial[.gitCommit] = "a1b2c3d"
    trial[.totalEpochs] = 25
    trial[.earlyStopped] = false
    
    return train(lr: lr)
}

// Type-safe reads on completed trial
if let best = try study.bestTrial {
    let commit: String? = best[.gitCommit]
    let epochs: Int? = best[.totalEpochs]
    let stopped: Bool? = best[.earlyStopped]
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

extension AttributeKey where Value == ModelArchitecture {
    static let architecture = AttributeKey<ModelArchitecture>("architecture")
}

// Write enum directly to active trial
trial[.architecture] = .convnextSmall

// Read back strongly-typed enum from completed trial
let arch: ModelArchitecture? = bestTrial[.architecture]
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

extension AttributeKey where Value == CodableAttribute<HardwareStats> {
    static let hardwareStats = AttributeKey<CodableAttribute<HardwareStats>>("hardware_stats")
}

// Write structured JSON payload
let stats = HardwareStats(peakMemoryMB: 3412.5, gpuUtilization: 0.94, deviceName: "Apple M3 Max")
trial[.hardwareStats] = CodableAttribute(stats)

// Read payload back from trial record
if let savedStats = bestTrial[.hardwareStats]?.value {
    print("Peak Memory: \(savedStats.peakMemoryMB) MB on \(savedStats.deviceName)")
}
```

---

## 5. Dynamic untyped attributes

When dynamic or exploratory behavior is preferred (matching Python's `trial.set_user_attr`), use string subscripting:

```swift
trial["dataset_tag"] = "imagenet_v2"
trial[userAttr: "experiment_id"] = "exp_42"

let tag: String? = bestTrial["dataset_tag"]
```
