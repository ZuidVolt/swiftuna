#!/usr/bin/env python3
"""Generate LibRustuna.artifactbundle/manifest.json from current inputs."""

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Single hashing implementation lives in hash_rustuna.py — no local copy.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ffi_util import sqlite_defined_symbols
from hash_rustuna import compute_inputs_hash

inputs_hash = compute_inputs_hash()


def file_hash(p):
    try:
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def sqlite_defs(p: pathlib.Path) -> int:
    """Count defined _sqlite3_* symbols: 0 means the slice links system
    SQLite (macOS policy); >0 means bundled (Linux policy). Derived, not
    hardcoded, so the manifest reports what is actually shipped."""
    return sqlite_defined_symbols(p)


manifest = {
    "version": 1,
    "inputs_hash": inputs_hash,
    "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip()
    if shutil.which("rustc")
    else "",
    # NOTE: no generated_at — the file must be byte-deterministic so CI
    # only commits when content actually changes.
    "artifacts": {},
}
for arch, path in [
    ("macos-arm64", "LibRustuna.artifactbundle/macos-arm64/librustuna_ffi.a"),
    ("linux-x86_64", "LibRustuna.artifactbundle/linux-x86_64/librustuna_ffi.a"),
    ("linux-aarch64", "LibRustuna.artifactbundle/linux-aarch64/librustuna_ffi.a"),
]:
    p = pathlib.Path(path)
    if p.exists():
        ndefs = sqlite_defs(p)
        manifest["artifacts"][arch] = {
            "file": str(p),
            "sha256": file_hash(p),
            "size": p.stat().st_size,
            # Provenance is measured: "system" (macOS links libsqlite3,
            # zero bundled defs) vs "bundled" (Linux hermetic).
            "sqlite": "system" if ndefs == 0 else "bundled",
            "sqlite_defined_symbols": ndefs,
        }
pathlib.Path("LibRustuna.artifactbundle/manifest.json").write_text(
    json.dumps(manifest, indent=2)
)
print(json.dumps(manifest, indent=2))
