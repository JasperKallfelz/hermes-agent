"""Authenticity, packaging, sandbox, and protocol checks for experimental Pi."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.pi_runtime import managed_pi_paths, verify_managed_pi
from agent.transports.pi_docker_sandbox import PiDockerSandbox, validate_workspace_boundary
from agent.transports.pi_rpc_client import PiRpcClient, redact_protocol_text


@dataclass
class PiDoctorReport:
    ok: bool
    executable: str | None = None
    version: str | None = None
    rpc_ok: bool = False
    package_ok: bool = False
    extension_ok: bool = False
    docker_ok: bool = False
    checks: list[str] = field(default_factory=list)
    error: str | None = None


def check_pi_runtime(
    *,
    cwd: str,
    timeout: float = 10.0,
    hermes_home: str | Path | None = None,
    docker_image: str | None = None,
    egress_allowlist: set[str] | frozenset[str] | None = None,
) -> PiDoctorReport:
    client: PiRpcClient | None = None
    sandbox: PiDockerSandbox | None = None
    checks: list[str] = []
    try:
        paths = managed_pi_paths(hermes_home=hermes_home, provision=False)
        executable = verify_managed_pi(paths, run_version=False)
        checks.extend([
            "managed path", "package version", "package integrity",
            "managed executable integrity",
        ])
        packaged_assets = Path(__file__).with_name("pi_assets")
        required_assets = (
            packaged_assets / "hermes_bridge.ts",
            packaged_assets / "Dockerfile",
            packaged_assets / "build-image.sh",
            packaged_assets / "egress_proxy.py",
            packaged_assets / "package.json",
            packaged_assets / "package-lock.json",
            packaged_assets / "runtime-integrity.json",
            packaged_assets / "verification-requirements.in",
            packaged_assets / "verification-requirements.txt",
            packaged_assets / "verify_npm_lock.py",
        )
        if not all(asset.is_file() for asset in required_assets):
            raise RuntimeError("Hermes Pi runtime assets are missing from the installed package")
        checks.extend(["packaged extension", "packaged Dockerfile", "packaged build helper"])
        if not docker_image:
            raise RuntimeError(
                "interactive Pi requires an exact Docker repo digest or local sha256 image ID"
            )
        if not paths.config_dir.is_dir() or not paths.session_root.is_dir() or not paths.extension_file.is_file():
            raise RuntimeError("managed Pi config/session/extension directories have not been provisioned")
        probe_session = paths.session_root / "doctor"
        validate_workspace_boundary(
            cwd,
            config_dir=paths.config_dir,
            session_dir=probe_session,
            binding_root=paths.binding_root,
            sensitive_roots=(paths.root, paths.session_root, paths.home_dir),
        )
        probe_session.mkdir(parents=True, exist_ok=True, mode=0o700)
        sweep = getattr(PiDockerSandbox, "sweep_stale_resources", None)
        active = getattr(PiDockerSandbox, "active_resource_identities", None)
        if callable(sweep):
            sweep(
                active_identities=active() if callable(active) else frozenset(),
                timeout=timeout,
            )
        sandbox = PiDockerSandbox(
            image=docker_image,
            worktree=cwd,
            config_dir=paths.config_dir,
            session_dir=probe_session,
            extension_file=paths.extension_file,
            read_only_workspace=True,
            egress_allowlist=egress_allowlist,
            binding_root=paths.binding_root,
            sensitive_roots=(paths.root, paths.session_root, paths.home_dir),
        )
        sandbox.verify_version(timeout=timeout)
        mark_active = getattr(sandbox, "mark_active", None)
        if callable(mark_active):
            mark_active()
        sandbox.prepare_egress(timeout=timeout)
        checks.extend(["Docker immutable reference", "Docker security profile", "container Pi version"])

        # The schema handshake uses the same read-only Docker boundary as an
        # interactive session. Doctor is ephemeral; application turns never
        # use --no-session.
        pi_args = [
            "--mode", "rpc", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-themes", "--no-context-files",
            "--no-approve", "--offline", "--no-session",
            "--tools", "read,edit,write",
        ]
        client = PiRpcClient(
            None,
            cwd=cwd,
            launch_command=sandbox.pi_launch_command(pi_args),
            allowed_provider_vars=set(),
        )
        sandbox.inspect_egress(timeout=timeout)
        checks.extend(["internal egress network", "allowlisted proxy identity"])
        state = client.request("get_state", timeout=timeout)
        models = client.request("get_available_models", timeout=timeout)
        if state.get("command") != "get_state" or not isinstance(state.get("data"), dict):
            raise RuntimeError("Pi get_state response schema is incompatible")
        if models.get("command") != "get_available_models" or not isinstance(models.get("data"), dict):
            raise RuntimeError("Pi model-catalog response schema is incompatible")
        checks.extend(["RPC handshake", "state schema", "model schema"])
        return PiDoctorReport(
            True, executable, "0.84.3", rpc_ok=True, package_ok=True,
            extension_ok=True, docker_ok=True, checks=checks,
        )
    except Exception as exc:
        return PiDoctorReport(False, checks=checks, error=redact_protocol_text(exc))
    finally:
        if client:
            client.close(force=True)
        if sandbox:
            sandbox.cleanup(timeout=min(timeout, 10.0))
