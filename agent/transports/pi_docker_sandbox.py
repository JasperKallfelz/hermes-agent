"""Hard Docker containment for interactive and one-shot Pi coding work."""
from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.transports.pi_rpc_client import (
    APPROVED_PROVIDER_ENV,
    PINNED_PI_VERSION,
    redact_protocol_text,
)

_REPO_DIGEST_REF = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_IMAGE_ID_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIXED_VERIFY_EXECUTABLES = {
    "pytest": "/opt/hermes/verify-venv/bin/pytest",
    "python": "/opt/hermes/verify-venv/bin/python",
    "python3": "/opt/hermes/verify-venv/bin/python",
    "ruff": "/opt/hermes/verify-venv/bin/ruff",
    "ty": "/opt/hermes/verify-venv/bin/ty",
    "npm": "/usr/local/bin/npm",
}
_NPM_VERIFY_SCRIPTS = frozenset({"build", "lint", "test", "typecheck"})
_DEFAULT_EGRESS_DESTINATIONS: frozenset[str] = frozenset()
_CONTAINER_ROLES = frozenset({"runtime", "proxy"})
_NETWORK_ROLE = "network"
_EGRESS_DESTINATION = re.compile(
    r"^(?=.{1,253}(?::[0-9]{1,5})?$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?::[0-9]{1,5})?$"
)
_REVIEWED_IMAGE_LABELS = {
    "org.hermes.pi.version": PINNED_PI_VERSION,
    "org.hermes.pi.tarball.sha512": "62bda9f4fb9badb15999810f608f82f0a9993fdc6516e2c39c01bae11b54d190e0add8976166becbb586c893b93a5a8f9623a454231df48cd5b333b7fedd03d3",
    "org.hermes.pi.package-json.sha256": "6c0bfa4558a0de32fb54fa97ca73397753ec7d4ad7d0ba607790920168de751f",
    "org.hermes.pi.cli.sha256": "1c3a5094b54aae9ae98c66516ce8c6578140363d081471ca7e91f9cb8c23dc8a",
    "org.hermes.pi.shrinkwrap.sha256": "df18751951cef578a1e5d809b33774bb385f69539f197bade0e70f1944aa3de7",
    "org.hermes.pi.manifest.sha256": "9fef223236a34bcab3a6926dcc1f3cd1f4d3504e3ce5eb37ac840b5ea0b280f2",
    "org.hermes.pi.egress-proxy.sha256": "0d8f3d724b70d141072fc51c42ccd6cc17296a5775d0bf68c2496b3de7882aab",
    "org.hermes.pi.bridge.sha256": "26058182f2aa31e5e3fd46160143e07cd3f6d506b8898970755128d2faac6bf5",
    "org.hermes.pi.registry-npm-tree.sha256": "1951e4981b80b2ffd23f8f6c26f91598b15e3892a94108f9b91a321c0f805716",
    "org.hermes.pi.image-npm-tree.sha256": "f845429e8c8fa262101900dd8eddaf43193ed9c0db51ab25085c6bedefc51370",
    "org.hermes.pi.verification-requirements.sha256": "18ceee9cfc1b05a75456c272adb20dfda896bd46064f105d0357898db9911857",
}
# Host-side authenticity anchor.  This is intentionally outside pi_assets so a
# newly reviewed deterministic build can be added without changing the Docker
# build context (and therefore without recursively changing the image ID).
# The sole entry is the reviewed post-repair linux/arm64 image produced by two
# independent clean no-cache builds with identical immutable IDs.
_REVIEWED_IMAGE_IDS = frozenset({
    "sha256:e89f45110e9277902bafbf49009e842bc9e38180e668fea8a6ff3dcdb2dd2cdf",
})
_REVIEWED_IMAGE_OS = "linux"
_REVIEWED_IMAGE_ARCHITECTURE = "arm64"


class PiSandboxError(RuntimeError):
    """The required hard-containment boundary is unavailable or invalid."""


def _canonical_path(path: str | Path, *, strict: bool) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=strict)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PiSandboxError(f"Pi containment path is unavailable: {exc}") from exc


def _validate_mount_source(path: Path, *, description: str) -> Path:
    """Reject Docker ``--mount`` field delimiters after canonicalization."""
    serialized = str(path)
    if "," in serialized or any(ord(character) < 32 or ord(character) == 127 for character in serialized):
        raise PiSandboxError(
            f"Pi {description} path cannot be serialized safely as a Docker bind mount"
        )
    return path


def _bind_mount(source: Path, destination: str, *, readonly: bool = False) -> str:
    _validate_mount_source(source, description="bind-mount source")
    mount = f"type=bind,src={source},dst={destination}"
    return mount + (",readonly" if readonly else "")


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_workspace_boundary(
    worktree: str | Path,
    *,
    config_dir: str | Path,
    session_dir: str | Path,
    binding_root: str | Path | None = None,
    sensitive_roots: tuple[str | Path, ...] | list[str | Path] = (),
) -> Path:
    """Canonicalize and reject every workspace/control-plane alias.

    A shared ancestor is harmless (for example ``~/.hermes/worktrees`` and
    ``~/.hermes/runtimes``).  The assigned workspace itself must not equal,
    contain, or be contained by a mounted/control root.
    """
    workspace = _canonical_path(worktree, strict=True)
    if not workspace.is_dir():
        raise PiSandboxError("Pi workspace must be an existing directory")
    home = Path.home().expanduser().resolve(strict=False)
    if workspace == Path(workspace.anchor) or workspace == home:
        raise PiSandboxError("Pi workspace must be a narrow project/worktree, not home or filesystem root")

    # Control leaves may be about to be provisioned. ``strict=False`` still
    # resolves every existing symlinked ancestor, which is what closes alias
    # and symlink overlap before those directories are created.
    config = _canonical_path(config_dir, strict=False)
    sessions = _canonical_path(session_dir, strict=False)
    binding = (
        _canonical_path(binding_root, strict=False)
        if binding_root is not None
        else None
    )
    leaves = [config, sessions, *([binding] if binding is not None else [])]
    for index, left in enumerate(leaves):
        for right in leaves[index + 1 :]:
            if _paths_overlap(left, right):
                raise PiSandboxError("Pi config, session, and binding roots must be disjoint")

    controls = [*leaves]
    controls.extend(
        _canonical_path(root, strict=False)
        for root in sensitive_roots
    )
    for control in controls:
        if _paths_overlap(workspace, control):
            raise PiSandboxError(
                f"Pi workspace overlaps a sensitive runtime/control root: {control}"
            )
    return workspace


def is_immutable_image_ref(image: object) -> bool:
    """Return whether *image* is an immutable registry digest or local ID."""
    value = str(image or "")
    return bool(_REPO_DIGEST_REF.fullmatch(value) or _IMAGE_ID_REF.fullmatch(value))


def _unprivileged_identity() -> tuple[int, int]:
    """Map the host identity into Docker without ever selecting container root."""
    host_uid = os.getuid() if hasattr(os, "getuid") else 65532
    host_gid = os.getgid() if hasattr(os, "getgid") else 65532
    uid = host_uid if host_uid > 0 else 65532
    gid = host_gid if host_gid > 0 else 65532
    return uid, gid


@dataclass(frozen=True)
class DockerLimits:
    cpus: float = 2.0
    memory: str = "2g"
    pids: int = 256
    tmpfs_size: str = "256m"


class PiDockerSandbox:
    """Build exact ``docker run`` commands for one assigned worktree."""

    _active_lock = threading.RLock()
    _active_identities: set[str] = set()

    def __init__(
        self,
        *,
        image: str,
        worktree: str | Path,
        config_dir: str | Path,
        session_dir: str | Path,
        extension_file: str | Path,
        read_only_workspace: bool = False,
        provider_env_names: set[str] | frozenset[str] | None = None,
        egress_allowlist: set[str] | frozenset[str] | None = None,
        limits: DockerLimits | None = None,
        docker_bin: str = "docker",
        container_name: str | None = None,
        resource_identity: str | None = None,
        binding_root: str | Path | None = None,
        sensitive_roots: tuple[str | Path, ...] | list[str | Path] = (),
    ) -> None:
        if not is_immutable_image_ref(image):
            raise PiSandboxError(
                "Pi Docker image must be an exact repo@sha256 digest or sha256 image ID"
            )
        self.image = str(image)
        self.worktree = validate_workspace_boundary(
            worktree,
            config_dir=config_dir,
            session_dir=session_dir,
            binding_root=binding_root,
            sensitive_roots=sensitive_roots,
        )
        self.config_dir = _canonical_path(config_dir, strict=True)
        self.session_dir = _canonical_path(session_dir, strict=True)
        self.binding_root = (
            _canonical_path(binding_root, strict=False)
            if binding_root is not None
            else None
        )
        self.extension_file = _canonical_path(extension_file, strict=True)
        for description, path in (
            ("workspace", self.worktree),
            ("config", self.config_dir),
            ("session", self.session_dir),
            ("extension", self.extension_file),
            ("binding", self.binding_root),
        ):
            if path is not None:
                _validate_mount_source(path, description=description)
        self.read_only_workspace = read_only_workspace
        requested_env = set(provider_env_names or ())
        unknown_env = sorted(requested_env - set(APPROVED_PROVIDER_ENV))
        if unknown_env:
            raise PiSandboxError(
                "unapproved Pi provider environment variable names: "
                + ", ".join(unknown_env)
            )
        self.provider_env_names = sorted(requested_env)
        requested_egress = set(_DEFAULT_EGRESS_DESTINATIONS)
        requested_egress.update(str(value).strip().lower() for value in (egress_allowlist or ()))
        invalid_egress = []
        for value in sorted(requested_egress):
            host = value.rsplit(":", 1)[0] if ":" in value else value
            try:
                ipaddress.ip_address(host)
                is_ip = True
            except ValueError:
                is_ip = False
            if (
                not _EGRESS_DESTINATION.fullmatch(value)
                or "*" in value or "://" in value or is_ip
                or (":" in value and not 1 <= int(value.rsplit(":", 1)[1]) <= 65535)
            ):
                invalid_egress.append(value)
        if invalid_egress:
            raise PiSandboxError(
                "Pi egress destinations must be exact DNS hostnames with optional ports: "
                + ", ".join(invalid_egress)
            )
        self.egress_allowlist = tuple(sorted(requested_egress))
        self.limits = limits or DockerLimits()
        self.docker_bin = docker_bin
        self.container_name = container_name or f"hermes-pi-{uuid.uuid4().hex[:20]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", self.container_name):
            raise PiSandboxError("invalid managed Pi container name")
        resource_id = hashlib.sha256(self.container_name.encode("ascii")).hexdigest()[:20]
        self.network_name = f"hermes-pi-net-{resource_id}"
        self.proxy_name = f"hermes-pi-proxy-{resource_id}"
        self.resource_identity = hashlib.sha256(
            str(resource_identity or self.container_name).encode("utf-8")
        ).hexdigest()
        self.created_at = int(time.time())
        self._egress_prepared = False
        if not self.extension_file.is_file() or not self.extension_file.is_relative_to(self.config_dir):
            raise PiSandboxError("the Hermes Pi extension must be inside the dedicated config directory")

    @classmethod
    def active_resource_identities(cls) -> frozenset[str]:
        with cls._active_lock:
            return frozenset(cls._active_identities)

    def mark_active(self) -> None:
        with self._active_lock:
            self._active_identities.add(self.resource_identity)

    def inspect(self, *, timeout: float = 15.0) -> dict[str, Any]:
        if shutil.which(self.docker_bin) is None:
            raise PiSandboxError("Docker CLI is unavailable")
        try:
            proc = subprocess.run(
                [self.docker_bin, "image", "inspect", self.image],
                text=True, capture_output=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PiSandboxError(f"Docker image inspection failed: {exc}") from exc
        if proc.returncode:
            raise PiSandboxError(redact_protocol_text(proc.stderr or "pinned Pi image is unavailable"))
        try:
            payload = json.loads(proc.stdout)
            if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
                raise TypeError("image inspection must return exactly one object")
            info = payload[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise PiSandboxError("Docker returned malformed image metadata") from exc
        raw_repo_digests = info.get("RepoDigests")
        if raw_repo_digests is None:
            repo_digests: set[str] = set()
        elif (
            not isinstance(raw_repo_digests, list)
            or any(not isinstance(value, str) for value in raw_repo_digests)
        ):
            raise PiSandboxError("Docker returned malformed image repository-digest metadata")
        else:
            repo_digests = set(raw_repo_digests)
        inspected_id = info.get("Id")
        if not isinstance(inspected_id, str) or not _IMAGE_ID_REF.fullmatch(inspected_id):
            raise PiSandboxError("Docker returned a malformed immutable image ID")
        if _REPO_DIGEST_REF.fullmatch(self.image):
            if self.image not in repo_digests:
                raise PiSandboxError(
                    "local Docker image metadata does not contain the configured repository digest"
                )
        elif inspected_id != self.image:
            raise PiSandboxError("local Docker image ID does not match the configured exact ID")
        if inspected_id not in _REVIEWED_IMAGE_IDS:
            raise PiSandboxError("Pi image ID is not in the host-side reviewed image allowlist")
        image_os = info.get("Os")
        architecture = info.get("Architecture")
        if not isinstance(image_os, str) or not isinstance(architecture, str):
            raise PiSandboxError("Docker image platform metadata is missing or malformed")
        if image_os != _REVIEWED_IMAGE_OS or architecture != _REVIEWED_IMAGE_ARCHITECTURE:
            raise PiSandboxError(
                "Pi image platform does not match reviewed linux/arm64 runtime"
            )
        config = info.get("Config")
        if not isinstance(config, dict):
            raise PiSandboxError("Docker returned malformed image configuration metadata")
        labels = config.get("Labels")
        if not isinstance(labels, dict):
            raise PiSandboxError("Docker returned malformed image label metadata")
        for label, expected in _REVIEWED_IMAGE_LABELS.items():
            if str(labels.get(label) or "") != expected:
                raise PiSandboxError(f"Pi image reviewed-artifact label is missing or invalid: {label}")
        return info

    @property
    def _allowlist_digest(self) -> str:
        return hashlib.sha256(",".join(self.egress_allowlist).encode("ascii")).hexdigest()

    def _run_docker(self, args: list[str], *, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.docker_bin, *args], text=True, capture_output=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PiSandboxError(f"Docker egress setup failed: {exc}") from exc

    def _resource_labels(self, role: str) -> list[str]:
        labels = {
            "org.hermes.pi.managed": "true",
            "org.hermes.pi.owner": self.container_name,
            "org.hermes.pi.identity": self.resource_identity,
            "org.hermes.pi.created-at": str(self.created_at),
            "org.hermes.pi.role": role,
        }
        result: list[str] = []
        for key, value in labels.items():
            result.extend(["--label", f"{key}={value}"])
        return result

    def prepare_egress(self, *, timeout: float = 30.0) -> None:
        """Create a private internal network and a separately confined proxy."""
        if not self.egress_allowlist:
            raise PiSandboxError(
                "Pi egress allowlist is empty; configure exact required provider hostnames"
            )
        if self._egress_prepared:
            self.inspect_egress(timeout=min(timeout, 15.0))
            return
        self.inspect(timeout=min(timeout, 15.0))
        try:
            created = self._run_docker([
                "network", "create", "--internal",
                *self._resource_labels("network"),
                self.network_name,
            ], timeout=timeout)
            if created.returncode:
                raise PiSandboxError(redact_protocol_text(created.stderr or "could not create Pi internal network"))
            allowlist = ",".join(self.egress_allowlist)
            uid, gid = _unprivileged_identity()
            proxy = self._run_docker([
            "run", "--detach", "--rm", "--init",
            "--name", self.proxy_name,
            "--network", self.network_name,
            "--network-alias", "pi-egress-proxy",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--pids-limit=64", "--memory=128m", "--cpus=0.5",
            "--user", f"{uid}:{gid}",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
            *self._resource_labels("proxy"),
            "--label", f"org.hermes.pi.egress.allowlist-sha256={self._allowlist_digest}",
            "--env", f"HERMES_PI_PROXY_ALLOWLIST={allowlist}",
            "--entrypoint", "python3", self.image, "/opt/hermes/pi_egress_proxy.py",
            ], timeout=timeout)
            if proxy.returncode:
                raise PiSandboxError(redact_protocol_text(proxy.stderr or "could not start Pi egress proxy"))
            connected = self._run_docker(
                ["network", "connect", "bridge", self.proxy_name], timeout=timeout
            )
            if connected.returncode:
                raise PiSandboxError(redact_protocol_text(connected.stderr or "could not attach Pi proxy egress"))
            self._egress_prepared = True
            self.inspect_egress(timeout=min(timeout, 15.0))
        except Exception:
            self.cleanup(timeout=min(timeout, 10.0))
            raise

    def inspect_egress(self, *, timeout: float = 15.0) -> dict[str, Any]:
        network_proc = self._run_docker(
            ["network", "inspect", self.network_name], timeout=timeout
        )
        proxy_proc = self._run_docker(
            ["container", "inspect", self.proxy_name], timeout=timeout
        )
        if network_proc.returncode or proxy_proc.returncode:
            raise PiSandboxError("Pi egress network or proxy is unavailable")
        try:
            network = json.loads(network_proc.stdout)[0]
            proxy = json.loads(proxy_proc.stdout)[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise PiSandboxError("Docker returned malformed Pi egress metadata") from exc
        network_labels = network.get("Labels") or {}
        if network.get("Internal") is not True:
            raise PiSandboxError("Pi network is not Docker Internal=true")
        if network_labels.get("org.hermes.pi.owner") != self.container_name:
            raise PiSandboxError("Pi network ownership label is invalid")
        if network_labels.get("org.hermes.pi.managed") != "true":
            raise PiSandboxError("Pi network managed-resource label is invalid")
        if network_labels.get("org.hermes.pi.identity") != self.resource_identity:
            raise PiSandboxError("Pi network session/task identity label is invalid")
        if network_labels.get("org.hermes.pi.created-at") != str(self.created_at):
            raise PiSandboxError("Pi network creation-time label is invalid")
        config = proxy.get("Config") or {}
        host_config = proxy.get("HostConfig") or {}
        labels = config.get("Labels") or {}
        networks = ((proxy.get("NetworkSettings") or {}).get("Networks") or {})
        env = set(config.get("Env") or [])
        expected_env = f"HERMES_PI_PROXY_ALLOWLIST={','.join(self.egress_allowlist)}"
        uid, gid = _unprivileged_identity()
        if config.get("Image") != self.image:
            raise PiSandboxError("Pi proxy does not use the configured immutable image")
        if labels.get("org.hermes.pi.owner") != self.container_name:
            raise PiSandboxError("Pi proxy ownership label is invalid")
        if labels.get("org.hermes.pi.managed") != "true":
            raise PiSandboxError("Pi proxy managed-resource label is invalid")
        if labels.get("org.hermes.pi.identity") != self.resource_identity:
            raise PiSandboxError("Pi proxy session/task identity label is invalid")
        if labels.get("org.hermes.pi.created-at") != str(self.created_at):
            raise PiSandboxError("Pi proxy creation-time label is invalid")
        if labels.get("org.hermes.pi.egress.allowlist-sha256") != self._allowlist_digest:
            raise PiSandboxError("Pi proxy allowlist identity is invalid")
        if expected_env not in env:
            raise PiSandboxError("Pi proxy allowlist configuration is invalid")
        if config.get("User") != f"{uid}:{gid}":
            raise PiSandboxError("Pi proxy must run as the expected non-root identity")
        if host_config.get("Binds") or host_config.get("Mounts") or proxy.get("Mounts"):
            raise PiSandboxError("Pi proxy must not expose host filesystem mounts")
        if self.network_name not in networks or "bridge" not in networks:
            raise PiSandboxError("Pi proxy network attachments are incomplete")
        if host_config.get("ReadonlyRootfs") is not True:
            raise PiSandboxError("Pi proxy root filesystem is not read-only")
        return {"network": network, "proxy": proxy}

    def _base_run_args(
        self,
        *,
        workspace_read_only: bool | None = None,
        network_none: bool = False,
        include_provider_env: bool = True,
    ) -> list[str]:
        read_only = self.read_only_workspace if workspace_read_only is None else workspace_read_only
        uid, gid = _unprivileged_identity()
        workspace_mount = _bind_mount(
            self.worktree, "/workspace", readonly=read_only,
        )
        args = [
            self.docker_bin, "run", "--rm", "--init", "--interactive",
            "--name", self.container_name,
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            f"--pids-limit={self.limits.pids}", f"--memory={self.limits.memory}",
            f"--cpus={self.limits.cpus}", "--user", f"{uid}:{gid}",
            *self._resource_labels("runtime"),
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.limits.tmpfs_size}",
            "--tmpfs", "/home/pi:rw,nosuid,nodev,size=32m",
            "--mount", workspace_mount,
            "--mount", _bind_mount(self.config_dir, "/pi-config", readonly=True),
            "--mount", _bind_mount(self.session_dir, "/pi-sessions"),
            "--workdir", "/workspace",
            "--env", "HOME=/home/pi", "--env", "PI_CODING_AGENT_DIR=/pi-config",
            "--env", "PI_CODING_AGENT_SESSION_DIR=/pi-sessions",
            "--env", "PI_OFFLINE=1", "--env", "PI_SKIP_VERSION_CHECK=1",
            "--env", "PI_TELEMETRY=0", "--env", "HERMES_PI_WORKSPACE=/workspace",
        ]
        if network_none:
            args.extend(["--network", "none"])
        else:
            proxy_url = "http://pi-egress-proxy:8080"
            args.extend([
                "--network", self.network_name,
                "--env", f"HTTP_PROXY={proxy_url}",
                "--env", f"HTTPS_PROXY={proxy_url}",
                "--env", f"http_proxy={proxy_url}",
                "--env", f"https_proxy={proxy_url}",
                "--env", "NODE_USE_ENV_PROXY=1",
                "--env", "NO_PROXY=", "--env", "no_proxy=",
            ])
        if include_provider_env:
            for name in self.provider_env_names:
                args.extend(["--env", name])
        return args

    def pi_launch_command(self, pi_args: list[str]) -> list[str]:
        return [*self._base_run_args(), "--entrypoint", "pi", self.image, *pi_args]

    def verify_version(self, *, timeout: float = 20.0) -> None:
        self.inspect(timeout=timeout)
        args = [
            *self._base_run_args(
                workspace_read_only=True, network_none=True, include_provider_env=False
            ),
            "--entrypoint", "pi", self.image, "--version",
        ]
        try:
            proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        finally:
            self.cleanup(timeout=min(timeout, 10.0))
        if proc.returncode or proc.stdout.strip() != PINNED_PI_VERSION:
            raise PiSandboxError(f"pinned image did not execute Pi {PINNED_PI_VERSION}")

    def cleanup(self, *, timeout: float = 10.0) -> None:
        """Best-effort daemon-side cleanup after the attached CLI is retired."""
        try:
            subprocess.run(
                [self.docker_bin, "rm", "--force", self.container_name],
                text=True, capture_output=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run(
                [self.docker_bin, "rm", "--force", self.proxy_name],
                text=True, capture_output=True, timeout=timeout, check=False,
            )
            subprocess.run(
                [self.docker_bin, "network", "rm", self.network_name],
                text=True, capture_output=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._egress_prepared = False
        with self._active_lock:
            self._active_identities.discard(self.resource_identity)

    @classmethod
    def sweep_stale_resources(
        cls,
        *,
        docker_bin: str = "docker",
        now: int | None = None,
        minimum_age_seconds: int = 3600,
        active_identities: set[str] | frozenset[str] | None = None,
        timeout: float = 20.0,
    ) -> dict[str, int]:
        """Remove only old, labeled, inactive Hermes Pi resources.

        Unlabeled resources, running runtime containers, proxies whose owner
        still exists, non-empty networks, recent resources, and explicitly
        active identities are never touched.
        """
        cutoff_now = int(time.time()) if now is None else int(now)
        active = set(active_identities or ())

        def run(args: list[str]) -> subprocess.CompletedProcess[str]:
            try:
                return subprocess.run(
                    [docker_bin, *args], text=True, capture_output=True,
                    timeout=timeout, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PiSandboxError(f"Docker stale-resource sweep failed: {exc}") from exc

        names_proc = run(["container", "ls", "-a", "--format", "{{.Names}}"])
        if names_proc.returncode != 0:
            return {"containers": 0, "networks": 0}
        known_names = {name for name in names_proc.stdout.splitlines() if name}
        ids_proc = run([
            "container", "ls", "-a", "--filter", "label=org.hermes.pi.managed=true",
            "--format", "{{.ID}}",
        ])
        removed_containers = 0
        if ids_proc.returncode == 0:
            ids = [value for value in ids_proc.stdout.splitlines() if value]
            if ids:
                inspected = run(["container", "inspect", *ids])
                try:
                    rows = json.loads(inspected.stdout) if inspected.returncode == 0 else []
                except json.JSONDecodeError:
                    rows = []
                for row in rows if isinstance(rows, list) else []:
                    labels = ((row.get("Config") or {}).get("Labels") or {})
                    state = row.get("State") or {}
                    identity = str(labels.get("org.hermes.pi.identity") or "")
                    owner = str(labels.get("org.hermes.pi.owner") or "")
                    role = str(labels.get("org.hermes.pi.role") or "")
                    try:
                        created = int(labels.get("org.hermes.pi.created-at") or 0)
                    except (TypeError, ValueError):
                        continue
                    if (
                        labels.get("org.hermes.pi.managed") != "true"
                        or role not in _CONTAINER_ROLES
                        or not identity or identity in active
                        or not owner
                        or created <= 0 or cutoff_now - created < minimum_age_seconds
                        or bool(state.get("Running") and role == "runtime")
                        or (role == "proxy" and owner in known_names)
                    ):
                        continue
                    target = str(row.get("Id") or "")
                    if target and run(["rm", "--force", target]).returncode == 0:
                        removed_containers += 1

        networks_proc = run([
            "network", "ls", "--filter", "label=org.hermes.pi.managed=true",
            "--format", "{{.ID}}",
        ])
        removed_networks = 0
        if networks_proc.returncode == 0:
            for network_id in networks_proc.stdout.splitlines():
                if not network_id:
                    continue
                inspected = run(["network", "inspect", network_id])
                try:
                    row = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else {}
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
                labels = row.get("Labels") or {}
                identity = str(labels.get("org.hermes.pi.identity") or "")
                owner = str(labels.get("org.hermes.pi.owner") or "")
                role = str(labels.get("org.hermes.pi.role") or "")
                try:
                    created = int(labels.get("org.hermes.pi.created-at") or 0)
                except (TypeError, ValueError):
                    continue
                if (
                    labels.get("org.hermes.pi.managed") != "true"
                    or role != _NETWORK_ROLE
                    or not identity or identity in active
                    or not owner or owner in known_names
                    or created <= 0 or cutoff_now - created < minimum_age_seconds
                    or bool(row.get("Containers"))
                ):
                    continue
                if run(["network", "rm", network_id]).returncode == 0:
                    removed_networks += 1
        return {"containers": removed_containers, "networks": removed_networks}

    @staticmethod
    def validate_verification_argv(argv: list[str]) -> str:
        if not argv:
            raise PiSandboxError("verification command is empty")
        if argv[0] != Path(argv[0]).name:
            raise PiSandboxError("verification executables must use a reviewed bare tool name")
        program = argv[0]
        executable = _FIXED_VERIFY_EXECUTABLES.get(program)
        if executable is None:
            raise PiSandboxError(
                f"verification program {program!r} is not in the fixed reviewed-image allowlist"
            )
        for argument in argv[1:]:
            if any(character in argument for character in ("\x00", "\r", "\n")):
                raise PiSandboxError("verification arguments contain control characters")
            option_value = argument.split("=", 1)[1] if "=" in argument else argument
            if Path(option_value).is_absolute() or option_value.startswith("file:"):
                raise PiSandboxError("verification arguments must not name absolute paths")
            if ".." in Path(option_value).parts:
                raise PiSandboxError("verification arguments must not traverse outside the workspace")
        if program in {"python", "python3"}:
            if len(argv) < 3 or argv[1] != "-m" or argv[2] not in {"compileall", "pytest"}:
                raise PiSandboxError(
                    "Python verification is limited to python3 -m pytest|compileall"
                )
        elif program == "ruff":
            if len(argv) < 2 or argv[1] != "check":
                raise PiSandboxError("Ruff verification is limited to ruff check")
        elif program == "ty":
            if len(argv) < 2 or argv[1] != "check":
                raise PiSandboxError("ty verification is limited to ty check")
        elif program == "npm":
            if len(argv) < 2:
                raise PiSandboxError("npm verification requires test or a reviewed run script")
            if argv[1] == "test" and len(argv) == 2:
                pass
            elif (
                len(argv) == 3
                and argv[1] == "run"
                and argv[2] in _NPM_VERIFY_SCRIPTS
            ):
                pass
            else:
                raise PiSandboxError(
                    "npm verification is limited to test or run build|lint|test|typecheck"
                )
        return executable

    def _verification_run_args(self) -> list[str]:
        uid, gid = _unprivileged_identity()
        return [
            self.docker_bin, "run", "--rm", "--init",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            f"--pids-limit={self.limits.pids}", f"--memory={self.limits.memory}",
            f"--cpus={self.limits.cpus}", "--user", f"{uid}:{gid}",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.limits.tmpfs_size}",
            "--tmpfs", "/home/pi:rw,nosuid,nodev,size=32m",
            "--mount", _bind_mount(self.worktree, "/workspace", readonly=True),
            "--workdir", "/workspace", "--network", "none",
            "--env", "HOME=/home/pi",
        ]

    def run_verification(self, argv: list[str], *, timeout: float) -> dict[str, Any]:
        executable = self.validate_verification_argv(argv)
        command = [
            *self._verification_run_args(),
            "--entrypoint", executable, self.image, *argv[1:],
        ]
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
            return {
                "argv": argv,
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": redact_protocol_text(proc.stdout, limit=16 * 1024),
                "stderr": redact_protocol_text(proc.stderr, limit=16 * 1024),
                "sandbox": "docker",
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "argv": argv,
                "ok": False,
                "error": f"verification timed out after {timeout:.0f}s",
                "stdout": redact_protocol_text(exc.stdout or "", limit=16 * 1024),
                "stderr": redact_protocol_text(exc.stderr or "", limit=16 * 1024),
                "sandbox": "docker",
            }
        finally:
            self.cleanup(timeout=min(timeout, 10.0))
