#!/usr/bin/env python3
"""
Assemble/verify LibRustuna.artifactbundle (SE-0482 staticLibrary).

The .a files live DIRECTLY in the bundle — build recipes
(just build-ffi-release, Tools/package-binaries.py, CI) write them there.
This script owns everything else:

  python3 Tools/build-artifactbundle.py            # sync headers + info.json, verify .a present
  python3 Tools/build-artifactbundle.py --check    # verify bundle in sync (CI gate)
  python3 Tools/build-artifactbundle.py --zip      # also emit LibRustuna.artifactbundle.zip + checksum

Layout (Intel macOS intentionally omitted — deprecated, Swift 6.5 drops it):
  LibRustuna.artifactbundle/
    info.json
    manifest.json                     (inputs hash + per-artifact sha256, via generate_manifest.py)
    include/rustuna.h + module.modulemap  (copied from Sources/LibRustuna/include/)
    macos-arm64/librustuna_ffi.a          (arm64-apple-macosx)
    linux-x86_64/librustuna_ffi.a         (x86_64-unknown-linux-gnu)
    linux-aarch64/librustuna_ffi.a        (aarch64-unknown-linux-gnu)

The .a files are fully static LLVM objects (Mach-O/ELF, no bitcode by
design — see crates/rustuna-ffi/Cargo.toml note on LTO); sqlite is
statically bundled inside (nm shows T _sqlite3_open / T sqlite3_open),
so the SE-0482 audit (libc-only external deps) passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_INCLUDE = ROOT / "Sources" / "LibRustuna" / "include"
BUNDLE = ROOT / "LibRustuna.artifactbundle"

VERSION = "0.1.0"

# (bundle subdir, supported triple) — .a files are written here directly
# by the build recipes; this script never copies them.
VARIANTS = [
    ("macos-arm64", "arm64-apple-macosx"),
    ("linux-x86_64", "x86_64-unknown-linux-gnu"),
    ("linux-aarch64", "aarch64-unknown-linux-gnu"),
]

HEADERS = ["rustuna.h", "module.modulemap"]


def info_json() -> dict:
    return {
        "schemaVersion": "1.0",
        "artifacts": {
            "LibRustuna": {
                "version": VERSION,
                "type": "staticLibrary",
                "variants": [
                    {
                        "path": f"{subdir}/librustuna_ffi.a",
                        "supportedTriples": [triple],
                        "staticLibraryMetadata": {
                            "headerPaths": ["include"],
                            "moduleMapPath": "include/module.modulemap",
                        },
                    }
                    for subdir, triple in VARIANTS
                ],
            }
        },
    }


def build() -> bool:
    ok = True
    for subdir, triple in VARIANTS:
        lib = BUNDLE / subdir / "librustuna_ffi.a"
        if not lib.exists():
            print(f"  ✘ missing {lib.relative_to(ROOT)} ({triple})", file=sys.stderr)
            print("    build it first: just build-ffi-release  /  python3 Tools/package-binaries.py",
                  file=sys.stderr)
            ok = False
    for h in HEADERS:
        if not (SRC_INCLUDE / h).exists():
            print(f"  ✘ missing {SRC_INCLUDE / h}", file=sys.stderr)
            ok = False
    if not ok:
        return False

    (BUNDLE / "include").mkdir(parents=True, exist_ok=True)
    for h in HEADERS:
        shutil.copy2(SRC_INCLUDE / h, BUNDLE / "include" / h)
        print(f"  ✓ include/{h}")
    for subdir, triple in VARIANTS:
        lib = BUNDLE / subdir / "librustuna_ffi.a"
        sz = lib.stat().st_size
        print(f"  ✓ {subdir}/librustuna_ffi.a  {sz / 1_000_000:.1f} MB  ({triple})")
    (BUNDLE / "info.json").write_text(json.dumps(info_json(), indent=2) + "\n")
    print(f"  ✓ info.json  (staticLibrary, {len(VARIANTS)} variants)")
    return True


def check() -> bool:
    if not BUNDLE.is_dir():
        print(f"  ✘ {BUNDLE.name}/ missing — run without --check first", file=sys.stderr)
        return False
    want = json.dumps(info_json(), indent=2) + "\n"
    try:
        got = (BUNDLE / "info.json").read_text()
    except OSError:
        print("  ✘ info.json missing", file=sys.stderr)
        return False
    if got != want:
        print("  ✘ info.json stale — re-run Tools/build-artifactbundle.py", file=sys.stderr)
        return False
    ok = True
    for subdir, triple in VARIANTS:
        a = BUNDLE / subdir / "librustuna_ffi.a"
        if not a.exists():
            print(f"  ✘ missing {a.relative_to(ROOT)} ({triple})", file=sys.stderr)
            ok = False
    for h in HEADERS:
        a, b = BUNDLE / "include" / h, SRC_INCLUDE / h
        if a.read_bytes() != b.read_bytes():
            print(f"  ✘ include/{h} stale — re-run", file=sys.stderr)
            ok = False
    if ok:
        print("  ✓ bundle in sync")
    return ok


def zip_bundle() -> bool:
    out = ROOT / "LibRustuna.artifactbundle.zip"
    if out.exists():
        out.unlink()
    subprocess.run(
        ["zip", "-qr", str(out), "LibRustuna.artifactbundle"], cwd=str(ROOT), check=True
    )
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"  ✓ {out.name}  {out.stat().st_size / 1_000_000:.1f} MB")
    print(f"  checksum: {h}")
    print("  Package.swift remote form:")
    print(f'    .binaryTarget(name: "LibRustuna", url: "<release-url>/{out.name}", checksum: "{h}")')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble LibRustuna.artifactbundle")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    if args.check:
        return 0 if check() else 1
    if not build():
        return 1
    if args.zip and not zip_bundle():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
