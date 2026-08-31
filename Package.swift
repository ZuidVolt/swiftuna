// swift-tools-version: 6.3

import PackageDescription

// Vendored Rust staticlibs — checked in via Tools/package-binaries.py
// Always release binary (apple-m1 / generic, bundled sqlite, strip -x / strip --strip-unneeded)
let vendoredMacDir = "Sources/LibRustuna/artifacts/macos-arm64"
#if arch(x86_64)
    let vendoredLinuxDir = "Sources/LibRustuna/artifacts/linux-x86_64"
#elseif arch(arm64)
    let vendoredLinuxDir = "Sources/LibRustuna/artifacts/linux-aarch64"
#else
    let vendoredLinuxDir = "Sources/LibRustuna/artifacts/linux-x86_64"
#endif

let linkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L\(vendoredMacDir)", "-lrustuna_ffi"], .when(platforms: [.macOS])),
    .unsafeFlags(["-L\(vendoredLinuxDir)", "-lrustuna_ffi"], .when(platforms: [.linux])),
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
        .package(url: "https://github.com/apple/swift-docc-plugin.git", from: "1.4.3"),
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
            resources: [.process("Documentation.docc")],
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaMigrator",
            dependencies: ["Swiftuna"],
            path: "Tools/SwiftunaMigrator",
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaParity",
            dependencies: ["Swiftuna"],
            path: "Tools/SwiftunaParity",
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "SwiftunaBench",
            dependencies: ["Swiftuna"],
            path: "Tools/SwiftunaBench",
            swiftSettings: swiftSettings,
            linkerSettings: linkerSettings
        ),
        .executableTarget(
            name: "Experimentation",
            dependencies: ["Swiftuna"],
            path: "Tools/Experimentation",
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
