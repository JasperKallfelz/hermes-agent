#!/usr/bin/env python3
"""Verify and reproducibly regenerate the reviewed Pi production npm lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

NPM_VERSION = "11.17.0"
# Registry-installed npm and npm copied from the digest-pinned Node image are
# separately reviewed artifacts. Their package trees are not byte-identical.
NPM_TREE_SHA256 = "1951e4981b80b2ffd23f8f6c26f91598b15e3892a94108f9b91a321c0f805716"
IMAGE_NPM_TREE_SHA256 = "f845429e8c8fa262101900dd8eddaf43193ed9c0db51ab25085c6bedefc51370"
_TREE_HASHES = {
    "registry": NPM_TREE_SHA256,
    "image": IMAGE_NPM_TREE_SHA256,
}


def npm_package_root(npm_cli: Path) -> Path:
    resolved = npm_cli.expanduser().resolve(strict=True)
    if resolved.name != "npm-cli.js" or resolved.parent.name != "bin":
        raise RuntimeError("npm must resolve to the reviewed npm-cli.js package entrypoint")
    root = resolved.parent.parent
    metadata = json.loads((root / "package.json").read_text(encoding="utf-8"))
    if metadata.get("name") != "npm" or metadata.get("version") != NPM_VERSION:
        raise RuntimeError(f"npm must be exactly {NPM_VERSION}")
    return root


def canonical_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            payload = b"L\0" + os.readlink(path).encode("utf-8")
        elif path.is_file():
            payload = b"F\0" + path.read_bytes()
        else:
            continue
        digest.update(relative + b"\0" + payload + b"\0")
    return digest.hexdigest()


def verify_npm_tree(npm_cli: Path, *, source: str = "registry") -> Path:
    try:
        expected = _TREE_HASHES[source]
    except KeyError as exc:
        raise RuntimeError(f"unknown reviewed npm tree source: {source}") from exc
    root = npm_package_root(npm_cli)
    actual = canonical_tree_hash(root)
    if actual != expected:
        raise RuntimeError(
            f"npm {NPM_VERSION} {source} package tree does not match the Hermes review"
        )
    return root


def verify_runtime_bundle(assets: Path) -> None:
    manifest_path = assets / "runtime-integrity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("asset_sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("runtime-integrity.json lacks the complete asset map")
    shipped = {
        path.name
        for path in assets.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    if set(expected) != shipped:
        raise RuntimeError("runtime-integrity.json does not enumerate the full Pi asset bundle")
    for name, reviewed_hash in expected.items():
        path = assets / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != reviewed_hash:
            raise RuntimeError(f"Pi runtime asset failed integrity verification: {name}")


def verify_lock_integrities(lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    packages = lock.get("packages")
    if lock.get("lockfileVersion") != 3 or not isinstance(packages, dict):
        raise RuntimeError("Pi package-lock.json must use npm lockfileVersion 3")
    missing = sorted(
        name for name, entry in packages.items()
        if name and isinstance(entry, dict) and not entry.get("link")
        and str(entry.get("resolved") or "").startswith("https://")
        and not str(entry.get("integrity") or "").startswith("sha512-")
    )
    if missing:
        raise RuntimeError(
            "remote production npm entries lack sha512 integrity: " + ", ".join(missing)
        )


def regenerate(
    assets: Path,
    npm_cli: Path,
    *,
    cache: Path | None = None,
    offline: bool = False,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="hermes-pi-lock-") as raw_temp:
        temp = Path(raw_temp)
        shutil.copyfile(assets / "package.json", temp / "package.json")
        npm_cache = cache or (temp / "npm-cache")
        command = [
            "node", str(npm_cli), "install", "--package-lock-only", "--ignore-scripts",
            "--no-audit", "--no-fund", "--cache", str(npm_cache),
            "--logs-dir", str(temp / "logs"),
        ]
        if offline:
            command.append("--offline")
        process = subprocess.run(
            command, cwd=temp, text=True, capture_output=True, timeout=300,
        )
        if process.returncode:
            raise RuntimeError(f"npm lock regeneration failed: {process.stderr.strip()}")
        generated = temp / "package-lock.json"
        verify_lock_integrities(generated)
        return generated.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npm", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--npm-tree-source",
        choices=tuple(_TREE_HASHES),
        default="registry",
    )
    parser.add_argument("--verify-bundle", action="store_true")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent
    npm_cli = args.npm.expanduser().resolve(strict=True)
    verify_npm_tree(npm_cli, source=args.npm_tree_source)
    if args.verify_bundle:
        verify_runtime_bundle(assets)
    lock_path = assets / "package-lock.json"
    verify_lock_integrities(lock_path)
    if args.integrity_only:
        return 0
    generated = regenerate(
        assets, npm_cli, cache=args.cache, offline=args.offline,
    )
    if args.write:
        lock_path.write_bytes(generated)
    elif generated != lock_path.read_bytes():
        raise RuntimeError("Pi package-lock.json drifted from exact npm 11.17.0 output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
