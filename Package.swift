// swift-tools-version: 6.3

import PackageDescription

// LibRustuna is a prebuilt Rust staticlib shipped as an SE-0482
// `staticLibrary` artifact bundle (no unsafeFlags, SPM-Index compatible).
// Variants: macos-arm64, linux-x86_64, linux-aarch64. Rebuild via
// Tools/build-artifactbundle.py; sqlite is statically bundled inside the .a.
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
        .library(
            name: "SwiftunaDistributed",
            targets: ["SwiftunaDistributed"]
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
        .library(
            name: "SwiftunaBenchKit",
            targets: ["SwiftunaBenchKit"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/x-sheep/swift-property-based.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-docc-plugin.git", from: "1.4.3"),
    ],
    targets: [
        .binaryTarget(
            name: "LibRustuna",
            path: "LibRustuna.artifactbundle"
        ),
        .target(
            name: "Swiftuna",
            dependencies: ["LibRustuna"],
            resources: [.process("Documentation.docc")],
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "SwiftunaMigrator",
            dependencies: ["Swiftuna"],
            path: "Tools/SwiftunaMigrator",
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "SwiftunaParity",
            dependencies: ["Swiftuna"],
            path: "Tools/SwiftunaParity",
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "SwiftunaBench",
            dependencies: ["SwiftunaBenchKit"],
            path: "Tools/SwiftunaBench",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "SwiftunaBenchKit",
            dependencies: ["Swiftuna", "SwiftunaDistributed", "LibRustuna"],
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "Experimentation",
            dependencies: ["Swiftuna"],
            path: "Tools/Experimentation",
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "SwiftunaBenchKitTests",
            dependencies: ["SwiftunaBenchKit"],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "SwiftunaTests",
            dependencies: ["Swiftuna", .product(name: "PropertyBased", package: "swift-property-based")],
            swiftSettings: swiftSettings
        ),
        .target(
            name: "SwiftunaDistributed",
            dependencies: ["Swiftuna"],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "SwiftunaDistributedTests",
            dependencies: ["SwiftunaDistributed", "Swiftuna"],
            swiftSettings: swiftSettings
        ),
    ]
)
