#!/usr/bin/env python3
"""Hash inputs for vendored librustuna_ffi.a — used by CI to skip rebuilds."""

import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERNS = [
    "crates/rustuna-ffi/Cargo.toml",
    "crates/rustuna-ffi/Cargo.lock",
    "Sources/LibRustuna/include/rustuna.h",
    "Sources/LibRustuna/include/module.modulemap",
    "Sources/LibRustuna/shim.c",
]


def compute_inputs_hash() -> str:
    patterns = list(PATTERNS)
    for p in ROOT.glob("crates/rustuna-ffi/src/**/*.rs"):
        patterns.append(str(p.relative_to(ROOT)))
    for d in [
        "ref/rustuna/rustuna_core",
        "ref/rustuna/rustuna_storage",
        "ref/rustuna/rustuna_sampler",
        "ref/rustuna/rustuna_importance",
    ]:
        for p in pathlib.Path(ROOT / d).rglob("*.rs"):
            try:
                patterns.append(str(p.relative_to(ROOT)))
            except ValueError:
                patterns.append(str(p))
        for p in pathlib.Path(ROOT / d).rglob("Cargo.toml"):
            try:
                patterns.append(str(p.relative_to(ROOT)))
            except ValueError:
                patterns.append(str(p))
    h = hashlib.sha256()
    for p in sorted(set(patterns)):
        pp = ROOT / p if not pathlib.Path(p).is_absolute() else pathlib.Path(p)
        if pp.is_file():
            h.update(str(p).encode())
            h.update(b"\0")
            h.update(hashlib.sha256(pp.read_bytes()).digest())
    h.update(os.environ.get("RUSTFLAGS", "").encode())
    try:
        h.update(subprocess.check_output(["rustc", "--version", "--verbose"]))
    except (OSError, subprocess.SubprocessError):
        pass  # rustc not available — hash without it
    return h.hexdigest()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-manifest",
        action="store_true",
        help="exit 0 if manifest matches, 1 if stale",
    )
    ap.add_argument("--print", action="store_true", help="print hash")
    args = ap.parse_args()

    cur = compute_inputs_hash()
    if args.print or not args.check_manifest:
        print(cur)

    if args.check_manifest:
        try:
            j = json.loads(
                (ROOT / "Sources/LibRustuna/artifacts/manifest.json").read_text()
            )
            prev = j.get("inputs_hash", "")
        except (OSError, json.JSONDecodeError):
            prev = ""
        # also require artifact exists
        has_artifact = (
            ROOT / "Sources/LibRustuna/artifacts/linux-x86_64/librustuna_ffi.a"
        ).exists()
        if cur == prev and has_artifact:
            print(f"up to date: {cur}", file=sys.stderr if "sys" in dir() else None)
            return 0
        else:
            print(
                f"stale: cur={cur} prev={prev}",
                file=sys.stderr if "sys" in dir() else None,
            )
            return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
