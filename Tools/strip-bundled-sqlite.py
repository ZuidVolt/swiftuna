#!/usr/bin/env python3
"""
Excise the bundled SQLite amalgamation object from a macOS librustuna_ffi.a.

Why: on Apple platforms the OS ships SQLite (SDK .tbd, always linkable),
so the macOS bundle slice links the system library instead
(Package.swift: .linkedLibrary("sqlite3")). This removes ~35% of the
archive, eliminates all bundled `_sqlite3_*` definitions (no
duplicate-symbol / silent-version-substitution hazard with other packages
vendoring SQLite), and keeps one transparent provenance per variant:
macOS = system, Linux = bundled (hermetic — no -dev package needed).

The Rust side is unaffected by construction: rusqlite and the raw
`sqlite3_*` externs in rustuna_storage_sync_optuna_dashboard keep 53
undefined refs, which the final link satisfies from libsqlite3.dylib.
All 53 exist in the macOS SDK stub (verified) and system SQLite ships
JSON1 (json_quote/json_valid/json_group_array), which the dashboard
sync SQL requires.

Usage:
  python3 Tools/strip-bundled-sqlite.py <path-to-librustuna_ffi.a>

Exits non-zero if no sqlite object is found or any `_sqlite3_` definition
remains afterwards.
"""

from __future__ import annotations

import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ffi_util import sqlite_defined_names


def members(archive: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["ar", "t", str(archive)], capture_output=True, text=True, check=True
    ).stdout
    return out.splitlines()


def sqlite_defs(archive: pathlib.Path) -> set[str]:
    return sqlite_defined_names(archive)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: strip-bundled-sqlite.py <librustuna_ffi.a>", file=sys.stderr)
        return 2
    archive = pathlib.Path(sys.argv[1])
    c_objs = [
        m for m in members(archive) if m.endswith("-sqlite3.o") and "rcgu" not in m
    ]
    if not c_objs:
        print(
            f"  ✘ no bundled sqlite amalgamation object in {archive}", file=sys.stderr
        )
        return 1
    for m in c_objs:
        subprocess.run(["ar", "d", str(archive), m], check=True)
        print(f"  ✓ excised {m}")
    remaining = sqlite_defs(archive)
    if remaining:
        print(
            f"  ✘ {len(remaining)} _sqlite3_ definitions remain: "
            f"{sorted(remaining)[:5]}…",
            file=sys.stderr,
        )
        return 1
    print("  ✓ 0 _sqlite3_ definitions remain (refs resolve to system libsqlite3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
