// swift-tools-version: 6.4

import Foundation
import PackageDescription

let rustConfig = ProcessInfo.processInfo.environment["RUST_CONFIGURATION"] ?? "debug"
let rustLibDir = "crates/rustuna-ffi/target/\(rustConfig)"

let linkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L\(rustLibDir)", "-lrustuna_ffi"])
]

let swiftSettings: [SwiftSetting] = [
    .swiftLanguageMode(.v6),
    .defaultIsolation(nil),
    .enableUpcomingFeature("ExistentialAny"),
    .enableUpcomingFeature("InternalImportsByDefault"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .treatAllWarnings(as: .error),
]

let package = Package(
    name: "Swiftuna",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "Swiftuna",
            targets: ["Swiftuna"]
        ),
        .executable(
            name: "SwiftunaMigrator",
            targets: ["SwiftunaMigrator"]
        ),
        .executable(
            name: "SwiftunaParity",
            targets: ["SwiftunaParity"]
        ),
        .executable(
            name: "SwiftunaBench",
            targets: ["SwiftunaBench"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/x-sheep/swift-property-based.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-docc-plugin.git", from: "1.4.3")
    ],
    targets: [
        .target(
            name: "LibRustuna",
            publicHeadersPath: "include",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "Swiftuna",
            dependencies: ["LibRustuna"],
            resources: [.process("Swiftuna.docc")],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaMigrator",
            dependencies: ["Swiftuna"],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaParity",
            dependencies: ["Swiftuna"],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaBench",
            dependencies: ["Swiftuna"],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "Experimentation",
            dependencies: ["Swiftuna"],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .testTarget(
            name: "SwiftunaTests",
            dependencies: ["Swiftuna", .product(name: "PropertyBased", package: "swift-property-based")],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
    ]
)
