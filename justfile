SOURCE := "Sources/"
BUILD_DIR := ".build/arm64-apple-macosx"

build-ffi:
    cargo build --manifest-path crates/rustuna-ffi/Cargo.toml

build-ffi-release:
    RUSTFLAGS="-C target-cpu=apple-m1 -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml
    strip -x crates/rustuna-ffi/target/release/librustuna_ffi.a
    mv crates/rustuna-ffi/target/release/librustuna_ffi.a Sources/LibRustuna/artifacts/macos-arm64/librustuna_ffi.a

build:
    swift build

build-release: build-ffi-release
    swift build -c release

test:
    swift test

test-ffi:
    cargo test --manifest-path crates/rustuna-ffi/Cargo.toml

test-all: test-ffi test

test-prop:
    swift test --filter PropertyTests

parity:
    swift run SwiftunaParity

update-parity-corpus:
    cargo run --manifest-path crates/rustuna-ffi/Cargo.toml --bin generate_traces
    uv run python Tools/generate_python_traces.py
    swift run SwiftunaParity

package-binaries:
    python3 Tools/package-binaries.py

package-binaries-check:
    python3 Tools/package-binaries.py --check

bench: build-ffi-release
    swift run -c release SwiftunaBench

microbench: build-ffi-release
    swift run -c release --package-path tmp_microbench

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
docs-build:
    swift package --allow-writing-to-directory ./docs \
        generate-documentation --target Swiftuna --disable-indexing --transform-for-static-hosting --hosting-base-path swiftuna --output-path ./docs

# Start a local web server to preview DocC documentation with live reload
docs-preview port="8080":
 -kill -9 $(lsof -ti :{{ port }}) 2>/dev/null || true
 swift package --disable-sandbox preview-documentation --target Swiftuna --port {{ port }}
