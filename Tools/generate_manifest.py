#!/usr/bin/env python3
"""Generate LibRustuna.artifactbundle/manifest.json from current inputs."""

import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from datetime import timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Reuse same hashing logic as hash_rustuna.py
try:
    from hash_rustuna import compute_inputs_hash

    inputs_hash = compute_inputs_hash()
except ImportError:
    # fallback inline (kept for standalone use)
    patterns = [
        "crates/rustuna-ffi/Cargo.toml",
        "crates/rustuna-ffi/Cargo.lock",
        "Sources/LibRustuna/include/rustuna.h",
        "Sources/LibRustuna/include/module.modulemap",
        "Sources/LibRustuna/shim.c",
    ]
    for p in ROOT.glob("crates/rustuna-ffi/src/**/*.rs"):
        patterns.append(str(p.relative_to(ROOT)))
    for d in [
        "ref/rustuna/rustuna_core",
        "ref/rustuna/rustuna_storage",
        "ref/rustuna/rustuna_sampler",
        "ref/rustuna/rustuna_importance",
    ]:
        for p in (ROOT / d).rglob("*.rs"):
            patterns.append(str(p.relative_to(ROOT)))
        for p in (ROOT / d).rglob("Cargo.toml"):
            patterns.append(str(p.relative_to(ROOT)))

    h = hashlib.sha256()
    for p in sorted(set(patterns)):
        pp = ROOT / p
        if pp.is_file():
            h.update(str(p).encode())
            h.update(b"\0")
            h.update(hashlib.sha256(pp.read_bytes()).digest())
    h.update(os.environ.get("RUSTFLAGS", "").encode())
    try:
        h.update(subprocess.check_output(["rustc", "--version", "--verbose"]))
    except (OSError, subprocess.SubprocessError):
        pass  # rustc not available
    inputs_hash = h.hexdigest()


def file_hash(p):
    try:
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def sqlite_defs(p: pathlib.Path) -> int:
    """Count defined _sqlite3_* symbols: 0 means the slice links system
    SQLite (macOS policy); >0 means bundled (Linux policy). Derived, not
    hardcoded, so the manifest reports what is actually shipped."""
    try:
        out = subprocess.check_output(
            ["nm", "-g", str(p)], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return -1
    n = 0
    for line in out.splitlines():
        # nm -g lists undefined U refs too — count DEFINED only.
        parts = line.split()
        if len(parts) >= 3 and len(parts[-2]) == 1 and parts[-2].upper() != "U":
            # Mach-O prefixes C symbols with `_`, ELF does not.
            if parts[-1].lstrip("_").startswith("sqlite3_"):
                n += 1
    return n


manifest = {
    "version": 1,
    "inputs_hash": inputs_hash,
    "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip()
    if shutil.which("rustc")
    else "",
    "generated_at": datetime.datetime.now(timezone.utc)
    .isoformat()
    .replace("+00:00", "Z"),
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
