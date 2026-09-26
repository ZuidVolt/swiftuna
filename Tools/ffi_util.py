#!/usr/bin/env python3
"""Shared helpers for the LibRustuna packaging scripts.

Single home for the two bits of logic every script needs so the bundle
invariants can't drift apart again:
- `sqlite_defined_symbols(archive)`: count of DEFINED `_sqlite3_*` symbols.
  0 means the slice links system SQLite (macOS policy); >0 means bundled
  (Linux hermetic policy). Mach-O prefixes C symbols with `_`, ELF does
  not — both are accepted. `nm -g` also lists undefined `U` refs, which
  are excluded: the Rust objects legitimately keep undefined sqlite refs
  that the final link resolves to the system library.
"""

from __future__ import annotations

import pathlib
import subprocess


def _is_sqlite_sym(tok: str) -> bool:
    return tok.lstrip("_").startswith("sqlite3_")


def _nm_global(archive: str | pathlib.Path) -> str:
    """May raise OSError / CalledProcessError if `nm` is unavailable."""
    return subprocess.run(
        ["nm", "-g", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
    ).stdout


def _iter_defined(nm_output: str):
    """Yield names of globally-DEFINED symbols (`nm -g` also lists
    undefined `U` refs, which are excluded: the Rust objects legitimately
    keep undefined sqlite refs that the final link resolves elsewhere)."""
    for line in nm_output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[-2]) == 1 and parts[-2].upper() != "U":
            yield parts[-1]


def sqlite_defined_names(archive: str | pathlib.Path) -> set[str]:
    """Names of defined sqlite symbols (Mach-O `_` prefix kept as-is)."""
    return {
        name for name in _iter_defined(_nm_global(archive)) if _is_sqlite_sym(name)
    }


def sqlite_defined_symbols(archive: str | pathlib.Path) -> int:
    """Return defined sqlite symbol count, or -1 if `nm` is unavailable."""
    try:
        return len(sqlite_defined_names(archive))
    except (OSError, subprocess.SubprocessError):
        return -1
