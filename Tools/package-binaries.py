#!/usr/bin/env python3
"""
Vendored binary packager for Swiftuna.

- Builds macOS arm64 natively (apple-m1, bundled sqlite, strip -x)
- Builds Linux x86_64 (+ try aarch64) via Apple `container` (swift:6.3-jammy + rustup stable, generic, bundled)
  Swift image has no cargo — installs rustup stable (1.98) fresh; rust image would need update.
- Vendors into Sources/LibRustuna/artifacts/<platform>/
- Uses Apple container CLI (`container`) via subprocess — no Docker, no YAML

Usage:
  python Tools/package-binaries.py            # full (macOS + linux x86_64 + try aarch64)
  python Tools/package-binaries.py --skip-linux
  python Tools/package-binaries.py --linux-only
  python Tools/package-binaries.py --verbose
  python Tools/package-binaries.py --check      # verify existing artifacts + show sizes

Requires: cargo, rustup, Apple `container` (macOS 26+, Apple Silicon) for Linux.
  Install: `container system start` must succeed; image swift:6.3-jammy (fallback ubuntu:22.04) pulled on demand.
  macOS part works without container.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATE = ROOT / "crates" / "rustuna-ffi"
ARTIFACTS = ROOT / "Sources" / "LibRustuna" / "artifacts"
MACOS_DIR = ARTIFACTS / "macos-arm64"
LINUX_X86_DIR = ARTIFACTS / "linux-x86_64"
LINUX_ARM_DIR = ARTIFACTS / "linux-aarch64"

CONTAINER_IMAGE = "docker.io/library/swift:6.3-jammy"
CONTAINER_IMAGE_FALLBACK = "docker.io/library/ubuntu:22.04"
CONTAINER_RUST_TOOLCHAIN = "stable"


def log(msg: str, *, verbose: bool = True):
    if verbose:
        print(msg, flush=True)


def run(cmd: list[str], *, cwd: pathlib.Path | None = None, env=None, check: bool = True) -> subprocess.CompletedProcess:
    pretty = " ".join(cmd)
    print(f"  $ {pretty}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def run_checked(cmd: list[str], **kw) -> bool:
    try:
        run(cmd, **kw)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✘ failed ({e.returncode}): {' '.join(cmd)}", file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"  ✘ not found: {e}", file=sys.stderr)
        return False


def ensure_rust_targets():
    need = ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
    try:
        out = subprocess.run(["rustup", "target", "list", "--installed"], capture_output=True, text=True, check=True)
        installed = set(out.stdout.split())
    except Exception:
        installed = set()
    for t in need:
        if t not in installed:
            log(f"  rustup target add {t} …")
            run(["rustup", "target", "add", t])


def build_macos() -> bool:
    print("\n● macOS arm64  (apple-m1, bundled sqlite)", flush=True)
    ok = run_checked(
        ["cargo", "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")],
        env={**__import__("os").environ, "RUSTFLAGS": "-C target-cpu=apple-m1 -C embed-bitcode=no"},
    )
    if not ok:
        return False
    src = CRATE / "target" / "release" / "librustuna_ffi.a"
    if not src.exists():
        print(f"  ✘ missing {src}", file=sys.stderr)
        return False
    try:
        run(["strip", "-x", str(src)])
    except Exception as e:
        print(f"  ⚠ strip -x failed: {e}", file=sys.stderr)
    MACOS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, MACOS_DIR / "librustuna_ffi.a")
    sz = (MACOS_DIR / "librustuna_ffi.a").stat().st_size
    print(f"  ✓ {MACOS_DIR / 'librustuna_ffi.a'}  {sz / 1_000_000:.1f} MB")
    try:
        out = subprocess.run(["file", str(MACOS_DIR / "librustuna_ffi.a")], capture_output=True, text=True, check=True)
        print(f"    {out.stdout.strip()[:120]}")
    except Exception:
        pass
    return True


# ---- Apple container helpers ----

def container_available() -> bool:
    # True only if system is running — check status output, not just binary presence
    try:
        r = subprocess.run(["container", "system", "status"], capture_output=True, text=True)
        out = (r.stdout + r.stderr).lower()
        if r.returncode == 0 and "running" in out:
            return True
        # fallback: if status mentions "not running" or XPC error, not available
        if "not running" in out or "xpc connection" in out or "connection invalid" in out:
            return False
        return r.returncode == 0 and "running" in out
    except FileNotFoundError:
        return False
    except Exception:
        return False


def ensure_container_system() -> bool:
    if container_available():
        return True
    print("  → starting container system …", flush=True)
    try:
        # --enable-kernel-install avoids interactive prompt for kata kernel (first start)
        proc = subprocess.run(["container", "system", "start", "--enable-kernel-install"], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  container system start: {proc.stdout.strip()} {proc.stderr.strip()}", flush=True)
            # fallback without flag
            proc = subprocess.run(["container", "system", "start"], capture_output=True, text=True)
            print(f"  container system start (fallback): {proc.stdout.strip()} {proc.stderr.strip()}", flush=True)
        # poll up to 40s (kernel download can be large)
        for _ in range(20):
            time.sleep(2)
            if container_available():
                print("  ✓ container system running", flush=True)
                return True
        print("  ✘ container system did not become ready (poll timeout)", file=sys.stderr)
        try:
            r = subprocess.run(["container", "system", "status"], capture_output=True, text=True)
            print(f"  status: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
        except Exception:
            pass
        return False
    except FileNotFoundError:
        print("  ✘ container not found (install from developer.apple.com, macOS 26+ Apple Silicon)", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ✘ container system start failed: {e}", file=sys.stderr)
        return False


def build_linux_via_container(target: str, out_dir: pathlib.Path, *, verbose: bool) -> bool:
    """
    Build for `target` inside Apple `container`. Uses x86_64 (amd64 via Rosetta) for all Linux targets
    per user request (“please use x86-64 for container as well”).
    """
    # x86_64 via Rosetta (linux/amd64), aarch64 native (linux/arm64) for correct linker flags
    platform_map = {
        "x86_64-unknown-linux-gnu": "linux/amd64",
        "aarch64-unknown-linux-gnu": "linux/arm64",
    }
    platform = platform_map.get(target, "linux/amd64")
    print(f"\n● Linux {target}  (generic, bundled sqlite)  platform={platform}  via container", flush=True)

    if not ensure_container_system():
        print("  ✘ container system not running — skip Linux build", file=sys.stderr)
        print("    hint: run `container system start` then re-run", file=sys.stderr)
        return False

    # Pull image explicitly (container does not auto-pull on run in some versions)
    # Use FQDN as required by container registry; try primary then fallback (swift image may not exist on Hub)
    effective_image = CONTAINER_IMAGE
    print(f"  container image pull {effective_image} ({platform}) …", flush=True)
    pull_ok = run_checked(["container", "image", "pull", "--platform", platform, effective_image])
    if not pull_ok:
        pull_ok = run_checked(["container", "image", "pull", effective_image])
    if not pull_ok and CONTAINER_IMAGE_FALLBACK != effective_image:
        print(f"  trying fallback {CONTAINER_IMAGE_FALLBACK} …", flush=True)
        pull_ok = run_checked(["container", "image", "pull", "--platform", platform, CONTAINER_IMAGE_FALLBACK]) or run_checked(
            ["container", "image", "pull", CONTAINER_IMAGE_FALLBACK]
        )
        if pull_ok:
            effective_image = CONTAINER_IMAGE_FALLBACK
    if not pull_ok:
        # If pull failed due to XPC not running, try to start once more
        if ensure_container_system():
            print("  retry pull after container system start …", flush=True)
            pull_ok = run_checked(["container", "image", "pull", "--platform", platform, effective_image]) or run_checked(
                ["container", "image", "pull", effective_image]
            )

    extra_pkgs = "file"  # native arm64 container has native gcc; no cross needed
    container_script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl build-essential pkg-config libsqlite3-dev ca-certificates {extra_pkgs} > /dev/null
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain {CONTAINER_RUST_TOOLCHAIN} > /dev/null
  export PATH="$HOME/.cargo/bin:$PATH"
else
  # rust image has old cargo (1.82) — update to stable for float_next_up_down
  if command -v rustup >/dev/null 2>&1; then rustup update --no-self-update > /dev/null 2>&1 || true; fi
  export PATH="$HOME/.cargo/bin:$PATH"
fi
export PATH="$HOME/.cargo/bin:$PATH"
# ensure stable toolchain is default
if command -v rustup >/dev/null 2>&1; then rustup default {CONTAINER_RUST_TOOLCHAIN} > /dev/null 2>&1 || true; fi
rustup target add {target} > /dev/null
echo "  cargo {target} …"
RUSTFLAGS="-C target-cpu=generic -C embed-bitcode=no" cargo build --release --manifest-path crates/rustuna-ffi/Cargo.toml --target {target}
echo "  strip …"
strip --strip-unneeded crates/rustuna-ffi/target/{target}/release/librustuna_ffi.a || strip crates/rustuna-ffi/target/{target}/release/librustuna_ffi.a || true
ls -lh crates/rustuna-ffi/target/{target}/release/librustuna_ffi.a || true
file crates/rustuna-ffi/target/{target}/release/librustuna_ffi.a || echo "file not available"
"""

    # container run: use --platform linux/amd64 (Rosetta) for all per request
    cmd = [
        "container", "run", "--rm",
        "--platform", platform,
        "-v", f"{ROOT}:/workspace",
        "-w", "/workspace",
        effective_image,
        "bash", "-c", container_script,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✘ Linux {target} build failed (exit {e.returncode})", file=sys.stderr)
        if target == "aarch64-unknown-linux-gnu":
            print("    aarch64 is best-effort — x86_64 is the required one", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("  ✘ container not found (install from developer.apple.com)", file=sys.stderr)
        return False

    src = CRATE / "target" / target / "release" / "librustuna_ffi.a"
    if not src.exists():
        print(f"  ✘ missing after container build: {src}", file=sys.stderr)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / "librustuna_ffi.a")
    sz = (out_dir / "librustuna_ffi.a").stat().st_size
    print(f"  ✓ {out_dir / 'librustuna_ffi.a'}  {sz / 1_000_000:.1f} MB")
    return True


def check_artifacts():
    print("\nArtifacts:", flush=True)
    for p in [MACOS_DIR / "librustuna_ffi.a", LINUX_X86_DIR / "librustuna_ffi.a", LINUX_ARM_DIR / "librustuna_ffi.a"]:
        if p.exists():
            sz = p.stat().st_size
            try:
                f = subprocess.run(["file", str(p)], capture_output=True, text=True, check=True).stdout.strip().split("\n")[0]
            except Exception:
                f = "?"
            print(f"  ✓ {p.relative_to(ROOT)}  {sz / 1_000_000:.1f} MB  {f[:80]}")
        else:
            print(f"  ✘ missing {p.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Package vendored librustuna_ffi.a for Swiftuna (Apple container)")
    ap.add_argument("--skip-linux", action="store_true", help="skip Linux builds")
    ap.add_argument("--linux-only", action="store_true", help="only Linux builds")
    ap.add_argument("--check", action="store_true", help="only check existing artifacts")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.check:
        check_artifacts()
        return 0

    start = time.time()
    if args.verbose:
        ensure_rust_targets()

    ok = True
    if not args.linux_only and not build_macos():
        ok = False
        print("  ✘ macOS build failed", file=sys.stderr)
    if not args.skip_linux:
        if not build_linux_via_container("x86_64-unknown-linux-gnu", LINUX_X86_DIR, verbose=args.verbose):
            ok = False
            print("  ✘ Linux x86_64 failed (required)", file=sys.stderr)
        a_ok = build_linux_via_container("aarch64-unknown-linux-gnu", LINUX_ARM_DIR, verbose=args.verbose)
        if not a_ok:
            print("  ⚠ Linux aarch64 failed — continuing (x86_64 is the required one)", file=sys.stderr)

    check_artifacts()
    dur = time.time() - start
    print(f"\nDone in {dur:.1f}s  {'✓' if ok else '✘ check logs'}")
    print("Next: swift build -c release  →  just bench")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
