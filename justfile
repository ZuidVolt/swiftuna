export RUST_CONFIGURATION := env("RUST_CONFIGURATION", "debug")
SOURCE := "Sources/"
BUILD_DIR := ".build/arm64-apple-macosx"

build-ffi:
    cargo build --manifest-path crates/rustuna-ffi/Cargo.toml

build-ffi-release:
    RUSTFLAGS="-C target-cpu=native -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml
    strip -x crates/rustuna-ffi/target/release/librustuna_ffi.a

build: build-ffi
    swift build

build-release: build-ffi-release
    RUST_CONFIGURATION=release swift build -c release

test: build-ffi
    swift test

test-prop: build-ffi
    swift test --filter PropertyTests

parity: build-ffi
    swift run SwiftunaParity

update-parity-corpus: build-ffi
    cargo run --manifest-path crates/rustuna-ffi/Cargo.toml --bin generate_traces
    uv run python tools/generate_python_traces.py
    swift run SwiftunaParity

bench: build-ffi-release
    RUST_CONFIGURATION=release swift run -c release SwiftunaBench

migrate-check:
    swift run SwiftunaMigrator

clean:
    cargo clean --manifest-path crates/rustuna-ffi/Cargo.toml
    swift package clean

reset:
    rm -rf .build Package.resolved
    swift package reset
    swift package resolve
