#!/usr/bin/env python3
"""Generate complete Pi runtime SBOM and supplemental source provenance."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCKED_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)(?:\s|\\|$)"
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required provenance environment value is absent: {name}")
    return value


def _read_nonempty(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"required runtime evidence is empty: {path.name}")
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_read_nonempty(path))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runtime evidence is malformed JSON: {path.name}") from exc


def _normalize_python_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _property(category: str, **extra: str) -> list[dict[str, str]]:
    values = {"hermes:category": category, **extra}
    return [{"name": name, "value": value} for name, value in values.items()]


def _component(
    *,
    component_type: str,
    name: str,
    version: str,
    purl: str,
    category: str,
    **properties: str,
) -> dict[str, Any]:
    if not name or not version:
        raise RuntimeError(f"{category} component lacks name or version")
    return {
        "type": component_type,
        "name": name,
        "version": version,
        "purl": purl,
        "properties": _property(category, **properties),
    }


def _image_metadata(evidence: Path, runtime_image_id: str) -> tuple[str, str]:
    payload = _load_json(evidence / "image-inspect.json")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("image inspection evidence must contain exactly one image")
    info = payload[0]
    if info.get("Id") != runtime_image_id or not _IMAGE_ID.fullmatch(runtime_image_id):
        raise RuntimeError("image inspection evidence does not match the immutable runtime ID")
    image_os = info.get("Os")
    architecture = info.get("Architecture")
    if image_os != "linux" or architecture != "arm64":
        raise RuntimeError("Pi evidence target must be the reviewed linux/arm64 platform")
    return image_os, architecture


def _dpkg_components(evidence: Path) -> list[dict[str, Any]]:
    components = []
    for line in _read_nonempty(evidence / "dpkg-packages.tsv").splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not all(fields):
            raise RuntimeError("malformed dpkg package evidence")
        name, version, architecture = fields
        components.append(_component(
            component_type="library",
            name=name,
            version=version,
            purl=(
                f"pkg:deb/debian/{quote(name, safe='')}@{quote(version, safe='')}"
                f"?arch={quote(architecture, safe='')}"
            ),
            category="image-dpkg",
            package_manager="dpkg",
        ))
    return components


def _npm_components(evidence: Path) -> list[dict[str, Any]]:
    lock = _load_json(evidence / "package-lock.json")
    if not isinstance(lock, dict):
        raise RuntimeError("image npm lock evidence is malformed")
    packages = lock.get("packages")
    if lock.get("lockfileVersion") != 3 or not isinstance(packages, dict):
        raise RuntimeError("image npm lock evidence is malformed")
    components = []
    pi_present = False
    for package_path, entry in packages.items():
        if not isinstance(package_path, str) or not isinstance(entry, dict):
            raise RuntimeError("image npm lock contains malformed package metadata")
        version = entry.get("version")
        if not isinstance(version, str) or not version:
            if package_path:
                raise RuntimeError(f"image npm lock package lacks a version: {package_path}")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            name = package_path.rsplit("node_modules/", 1)[-1]
        if name == "@earendil-works/pi-coding-agent" and version == "0.84.3":
            pi_present = True
        components.append(_component(
            component_type="library",
            name=name,
            version=version,
            purl=f"pkg:npm/{quote(name, safe='/')}@{quote(version, safe='')}",
            category="pi-npm-lock",
            package_manager="npm",
            lock_path=package_path or "<root>",
        ))
    if not pi_present:
        raise RuntimeError("image npm lock does not contain the reviewed Pi package")
    return components


def _verification_components(evidence: Path) -> list[dict[str, Any]]:
    requirements = _read_nonempty(evidence / "verification-requirements.txt")
    locked = {
        _normalize_python_name(match.group(1)): match.group(2)
        for line in requirements.splitlines()
        if (match := _LOCKED_REQUIREMENT.match(line.strip()))
    }
    installed = _load_json(evidence / "verify-python-packages.json")
    if not locked or not isinstance(installed, list) or not installed:
        raise RuntimeError("hash-locked verification Python evidence is absent")
    observed: dict[str, str] = {}
    components = []
    for entry in installed:
        if not isinstance(entry, dict):
            raise RuntimeError("verification Python package evidence is malformed")
        name = entry.get("name")
        version = entry.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("verification Python package lacks name or version")
        normalized = _normalize_python_name(name)
        if normalized in observed:
            raise RuntimeError(f"duplicate verification Python package: {name}")
        observed[normalized] = version
        components.append(_component(
            component_type="library",
            name=name,
            version=version,
            purl=f"pkg:pypi/{quote(normalized, safe='')}@{quote(version, safe='')}",
            category="verify-python-lock",
            package_manager="uv",
        ))
    if observed != locked:
        missing = sorted(set(locked) - set(observed))
        unexpected = sorted(set(observed) - set(locked))
        mismatched = sorted(
            name for name in set(locked) & set(observed)
            if locked[name] != observed[name]
        )
        raise RuntimeError(
            "verification Python environment does not exactly match its hash lock "
            f"(missing={missing}, unexpected={unexpected}, mismatched={mismatched})"
        )
    return components


def _tool_versions(evidence: Path) -> dict[str, str]:
    versions = {
        "node": _read_nonempty(evidence / "node-version.txt"),
        "npm": _read_nonempty(evidence / "npm-version.txt"),
        "python": _read_nonempty(evidence / "python-version.txt"),
        "uv": _read_nonempty(evidence / "uv-version.txt"),
    }
    if not versions["node"].startswith("v"):
        raise RuntimeError("image Node version evidence is malformed")
    if versions["npm"] != "11.17.0":
        raise RuntimeError("image npm version is not the reviewed npm 11.17.0")
    if not versions["python"].startswith("Python "):
        raise RuntimeError("image Python version evidence is malformed")
    if versions["uv"] != "uv 0.11.6":
        raise RuntimeError("image uv version is not the reviewed uv 0.11.6")
    return versions


def _tool_components(versions: dict[str, str]) -> list[dict[str, Any]]:
    normalized = {
        "node": versions["node"].removeprefix("v"),
        "npm": versions["npm"],
        "python": versions["python"].removeprefix("Python "),
        "uv": versions["uv"].removeprefix("uv "),
    }
    return [
        _component(
            component_type="application",
            name=name,
            version=version,
            purl=f"pkg:generic/{name}@{quote(version, safe='')}",
            category="runtime-tool",
            source="immutable-image",
        )
        for name, version in normalized.items()
    ]


def generate(evidence: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_image_id = _required_env("RUNTIME_IMAGE_ID")
    repository = _required_env("SOURCE_REPOSITORY")
    revision = _required_env("SOURCE_REVISION")
    source_ref = _required_env("SOURCE_REF")
    run_id = _required_env("WORKFLOW_RUN_ID")
    run_attempt = _required_env("WORKFLOW_RUN_ATTEMPT")
    image_os, architecture = _image_metadata(evidence, runtime_image_id)
    docker_version = _load_json(evidence / "docker-version.json")
    if not isinstance(docker_version, dict):
        raise RuntimeError("Docker version evidence is malformed")
    docker_client = (docker_version.get("Client") or {}).get("Version")
    docker_server = (docker_version.get("Server") or {}).get("Version")
    if not isinstance(docker_client, str) or not isinstance(docker_server, str):
        raise RuntimeError("Docker client/server version evidence is incomplete")
    buildkit = _read_nonempty(evidence / "buildkit-version.txt")
    versions = _tool_versions(evidence)
    categories = {
        "image-dpkg": _dpkg_components(evidence),
        "pi-npm-lock": _npm_components(evidence),
        "runtime-tool": _tool_components(versions),
        "verify-python-lock": _verification_components(evidence),
    }
    if any(not components for components in categories.values()):
        raise RuntimeError("one or more required Pi SBOM categories are absent")
    components = [
        component
        for category in sorted(categories)
        for component in categories[category]
    ]
    components.sort(
        key=lambda item: (
            next(prop["value"] for prop in item["properties"] if prop["name"] == "hermes:category"),
            item["purl"],
            item["name"],
        )
    )
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:" + str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "/".join((repository, revision, runtime_image_id)),
        )),
        "version": 1,
        "metadata": {
            "timestamp": generated,
            "component": {
                "type": "container",
                "name": "hermes-pi-runtime",
                "version": runtime_image_id,
                "properties": _property(
                    "runtime-image", target_os=image_os, target_architecture=architecture,
                ),
            },
        },
        "components": components,
    }
    provenance = {
        "schema": "hermes.pi.source-provenance.v2",
        "generated_at": generated,
        "runtime_image_id": runtime_image_id,
        "source_repository": repository,
        "source_revision": revision,
        "source_ref": source_ref,
        "source_date_epoch": 1704067200,
        "target": {"os": image_os, "architecture": architecture},
        "tools": {
            "docker_client": docker_client,
            "docker_server": docker_server,
            "buildkit": buildkit,
            **versions,
        },
        "workflow_run_id": run_id,
        "workflow_run_attempt": run_attempt,
    }
    return sbom, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve(strict=True)
    sbom, provenance = generate(evidence)
    (evidence / "sbom.cdx.json").write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (evidence / "source-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
