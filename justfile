export RUST_CONFIGURATION := env("RUST_CONFIGURATION", "debug")
SOURCE := "Sources/"
BUILD_DIR := ".build/arm64-apple-macosx"

build-ffi:
    cargo build --manifest-path crates/rustuna-ffi/Cargo.toml

# macOS M-series (arm64): apple-m1 covers M1..M4 family; Linux: generic (portable x86_64/aarch64)
build-ffi-release:
    RUSTFLAGS="-C target-cpu=apple-m1 -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml
    strip -x crates/rustuna-ffi/target/release/librustuna_ffi.a

build-ffi-release-linux:
    RUSTFLAGS="-C target-cpu=generic -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml --target x86_64-unknown-linux-gnu
    strip -x crates/rustuna-ffi/target/x86_64-unknown-linux-gnu/release/librustuna_ffi.a

build-ffi-release-linux-system-sqlite:
    RUSTFLAGS="-C target-cpu=generic -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml --target x86_64-unknown-linux-gnu --no-default-features --features sqlite-system
    strip -x crates/rustuna-ffi/target/x86_64-unknown-linux-gnu/release/librustuna_ffi.a

build: build-ffi
    bash -c 'set -o pipefail; swift build 2>&1 | awk "/Failed frontend command:/{skip=1; next} /error: Build failed/{skip=0} !skip"'

build-release: build-ffi-release
    bash -c 'set -o pipefail; RUST_CONFIGURATION=release swift build -c release 2>&1 | awk "/Failed frontend command:/{skip=1; next} /error: Build failed/{skip=0} !skip"'

test: build-ffi
    bash -c 'set -o pipefail; swift test 2>&1 | awk "/Failed frontend command:/{skip=1; next} /error: Build failed/{skip=0} !skip"'

test-prop: build-ffi
    bash -c 'set -o pipefail; swift test --filter PropertyTests 2>&1 | awk "/Failed frontend command:/{skip=1; next} /error: Build failed/{skip=0} !skip"'

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

# Build static Swift-DocC documentation for static hosting (GitHub Pages)
docs-build: build-ffi
    swift package --allow-writing-to-directory ./docs \
        generate-documentation --target Swiftuna --disable-indexing --transform-for-static-hosting --hosting-base-path swiftuna --output-path ./docs

# Start a local web server to preview DocC documentation with live reload
docs-preview port="8080": build-ffi
    -kill -9 $$(lsof -ti :{{ port }}) 2>/dev/null || true
    swift package --disable-sandbox preview-documentation --target Swiftuna --port {{ port }}
