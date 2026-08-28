"""Hermes adapter and authenticity boundary for experimental Pi RPC."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.external_runtime.finalize import finalize_external_turn
from agent.transports.pi_rpc_client import (
    sanitized_pi_environment,
    PINNED_PI_VERSION,
    redact_protocol_text,
    redact_protocol_value,
)
from agent.transports.pi_rpc_session import PiRpcSession
from agent.transports.pi_docker_sandbox import (
    PiDockerSandbox,
    validate_workspace_boundary,
)

logger = logging.getLogger(__name__)

PINNED_PACKAGE_JSON_SHA256 = "6c0bfa4558a0de32fb54fa97ca73397753ec7d4ad7d0ba607790920168de751f"
PINNED_CLI_SHA256 = "1c3a5094b54aae9ae98c66516ce8c6578140363d081471ca7e91f9cb8c23dc8a"
PINNED_NPM_INTEGRITY = "sha512-Yr2p9PubrbFZmYEPYI+C8KmZP9xlFuLDnAG64RtU0ZDgrdiXYWa+y7WGyJO5OlqPliOkVCMd9IzVszO3/t0D0w=="
PINNED_SHRINKWRAP_SHA256 = "df18751951cef578a1e5d809b33774bb385f69539f197bade0e70f1944aa3de7"
PINNED_MANIFEST_SHA256 = "9fef223236a34bcab3a6926dcc1f3cd1f4d3504e3ce5eb37ac840b5ea0b280f2"
PINNED_BRIDGE_SHA256 = "26058182f2aa31e5e3fd46160143e07cd3f6d506b8898970755128d2faac6bf5"
PINNED_TARBALL_SHA512 = "62bda9f4fb9badb15999810f608f82f0a9993fdc6516e2c39c01bae11b54d190e0add8976166becbb586c893b93a5a8f9623a454231df48cd5b333b7fedd03d3"


@dataclass(frozen=True)
class ManagedPiPaths:
    root: Path
    executable: Path
    package_json: Path
    package_lock: Path
    config_dir: Path
    session_root: Path
    binding_root: Path
    home_dir: Path
    extension_file: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def managed_pi_paths(*, hermes_home: str | Path | None = None, provision: bool = False) -> ManagedPiPaths:
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    root = Path(hermes_home).expanduser().resolve() / "runtimes" / "pi" / PINNED_PI_VERSION
    package = root / "node_modules" / "@earendil-works" / "pi-coding-agent"
    executable = root / "node_modules" / ".bin" / ("pi.cmd" if os.name == "nt" else "pi")
    config_dir = root / "managed-config"
    session_root = root / "sessions"
    binding_root = root / "bindings"
    home_dir = root / "home"
    extension_file = config_dir / "hermes_bridge.ts"
    if provision:
        for directory in (config_dir, session_root, binding_root, home_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                directory.chmod(0o700)
            except OSError:
                pass
        packaged = Path(__file__).with_name("transports") / "pi_assets" / "hermes_bridge.ts"
        if not extension_file.is_file() or _sha256(extension_file) != _sha256(packaged):
            shutil.copyfile(packaged, extension_file)
            try:
                extension_file.chmod(0o600)
            except OSError:
                pass
    return ManagedPiPaths(
        root=root,
        executable=executable,
        package_json=package / "package.json",
        package_lock=root / "package-lock.json",
        config_dir=config_dir,
        session_root=session_root,
        binding_root=binding_root,
        home_dir=home_dir,
        extension_file=extension_file,
    )


def verify_managed_pi(paths: ManagedPiPaths, *, run_version: bool = True) -> str:
    try:
        root = paths.root.resolve(strict=True)
        executable = paths.executable.resolve(strict=True)
        package_json = paths.package_json.resolve(strict=True)
        package_lock = paths.package_lock.resolve(strict=True)
        shrinkwrap = (package_json.parent / "npm-shrinkwrap.json").resolve(strict=True)
        cli_artifact = (package_json.parent / "dist" / "bundle" / "cli.js").resolve(strict=True)
        extension = paths.extension_file.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"managed Pi {PINNED_PI_VERSION} installation is incomplete: {exc}") from exc
    if not all(path.is_file() for path in (
        executable, package_json, package_lock, shrinkwrap, cli_artifact, extension
    )):
        raise RuntimeError("managed Pi executable or package metadata is missing")
    if not all(path.is_relative_to(root) for path in (
        executable, package_json, package_lock, shrinkwrap, cli_artifact, extension
    )):
        raise RuntimeError("managed Pi installation resolves outside its pinned runtime directory")
    secure_paths = {
        root, package_json, package_lock, shrinkwrap, executable, cli_artifact, extension,
        paths.executable.parent.resolve(strict=True),
    }
    for target in (executable, package_json, package_lock, shrinkwrap, cli_artifact, extension):
        parent = target.parent
        while parent != root:
            secure_paths.add(parent)
            parent = parent.parent
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    for path in secure_paths:
        path_stat = path.stat()
        mode = path_stat.st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"managed Pi path is group/world writable: {path.name}")
        if current_uid is not None and path_stat.st_uid not in {0, current_uid}:
            raise RuntimeError(f"managed Pi path has an untrusted owner: {path.name}")
    try:
        metadata = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("managed Pi package metadata is unreadable") from exc
    if metadata.get("name") != "@earendil-works/pi-coding-agent" or metadata.get("version") != PINNED_PI_VERSION:
        raise RuntimeError(f"managed Pi package must be exactly {PINNED_PI_VERSION}")
    try:
        lock = json.loads(package_lock.read_text(encoding="utf-8"))
        packages = lock.get("packages") or {}
        lock_root = packages.get("") or {}
        lock_entry = packages.get("node_modules/@earendil-works/pi-coding-agent") or {}
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise RuntimeError("managed Pi npm integrity metadata is unreadable") from exc
    if (
        lock_root.get("dependencies", {}).get("@earendil-works/pi-coding-agent") != PINNED_PI_VERSION
        or lock_entry.get("version") != PINNED_PI_VERSION
        or lock_entry.get("integrity") != PINNED_NPM_INTEGRITY
        or _sha256(package_lock) != PINNED_MANIFEST_SHA256
        or _sha256(shrinkwrap) != PINNED_SHRINKWRAP_SHA256
    ):
        raise RuntimeError("managed Pi npm tarball integrity does not match the Hermes pin")
    if _sha256(package_json) != PINNED_PACKAGE_JSON_SHA256 or _sha256(cli_artifact) != PINNED_CLI_SHA256:
        raise RuntimeError("managed Pi package integrity does not match the Hermes pin")
    if _sha256(extension) != PINNED_BRIDGE_SHA256:
        raise RuntimeError("managed Pi bridge integrity does not match the Hermes pin")
    if run_version:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(paths.home_dir),
            "PI_CODING_AGENT_DIR": str(paths.config_dir),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        proc = subprocess.run(
            [str(executable), "--version"], text=True, capture_output=True,
            timeout=10, env=env,
        )
        if proc.returncode or proc.stdout.strip() != PINNED_PI_VERSION:
            raise RuntimeError(f"managed Pi executable did not report {PINNED_PI_VERSION}")
    return str(executable)


def install_managed_pi(
    *,
    hermes_home: str | Path | None = None,
    npm_bin: str = "npm",
    force: bool = False,
) -> ManagedPiPaths:
    """Install the reviewed Pi tarball and its shrinkwrapped dependencies."""
    paths = managed_pi_paths(hermes_home=hermes_home, provision=False)
    if paths.root.exists() and not force:
        verify_managed_pi(paths)
        return paths
    npm = shutil.which(npm_bin)
    if npm is None:
        raise RuntimeError("the reviewed npm 11.17.0 runtime is required to install Pi")
    from agent.transports.pi_assets.verify_npm_lock import (
        verify_npm_tree,
    )

    try:
        verify_npm_tree(Path(npm), source="registry")
    except Exception as exc:
        raise RuntimeError(f"unreviewed npm runtime rejected: {exc}") from exc
    paths.root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{PINNED_PI_VERSION}-install-", dir=paths.root.parent))
    backup: Path | None = None
    try:
        (staging / "npm-home").mkdir(mode=0o700)
        safe_env = sanitized_pi_environment(
            os.environ,
            allowed_provider_vars=set(),
            runtime_values={"HOME": str(staging / "npm-home")},
        )
        proc = subprocess.run(
            [npm, "pack", f"@earendil-works/pi-coding-agent@{PINNED_PI_VERSION}",
             "--pack-destination", str(staging)],
            cwd=staging, env=safe_env, text=True, capture_output=True, timeout=300,
        )
        if proc.returncode:
            raise RuntimeError(f"npm pack failed: {redact_protocol_text(proc.stderr)}")
        tarball = staging / f"earendil-works-pi-coding-agent-{PINNED_PI_VERSION}.tgz"
        if not tarball.is_file():
            raise RuntimeError("npm did not produce the expected pinned Pi tarball")
        digest = hashlib.sha512(tarball.read_bytes()).hexdigest()
        if digest != PINNED_TARBALL_SHA512:
            raise RuntimeError("downloaded Pi tarball integrity does not match the Hermes pin")
        assets = Path(__file__).with_name("transports") / "pi_assets"
        shutil.copyfile(assets / "package.json", staging / "package.json")
        shutil.copyfile(assets / "package-lock.json", staging / "package-lock.json")
        ci = subprocess.run(
            [npm, "ci", "--omit=dev", "--no-audit", "--no-fund"],
            cwd=staging, env=safe_env, text=True, capture_output=True, timeout=900,
        )
        if ci.returncode:
            raise RuntimeError(f"npm ci failed: {redact_protocol_text(ci.stderr)}")
        package = staging / "node_modules" / "@earendil-works" / "pi-coding-agent"
        if _sha256(package / "npm-shrinkwrap.json") != PINNED_SHRINKWRAP_SHA256:
            raise RuntimeError("Pi shrinkwrap integrity does not match the Hermes review")
        if (
            _sha256(package / "package.json") != PINNED_PACKAGE_JSON_SHA256
            or _sha256(package / "dist" / "bundle" / "cli.js") != PINNED_CLI_SHA256
        ):
            raise RuntimeError("installed Pi package content does not match the Hermes review")
        config_dir = staging / "managed-config"
        for directory in (config_dir, staging / "sessions", staging / "bindings", staging / "home"):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        packaged_extension = Path(__file__).with_name("transports") / "pi_assets" / "hermes_bridge.ts"
        shutil.copyfile(packaged_extension, config_dir / "hermes_bridge.ts")
        if paths.root.exists():
            backup = paths.root.with_name(f".{paths.root.name}.backup-{time.time_ns()}")
            os.replace(paths.root, backup)
        os.replace(staging, paths.root)
        installed = managed_pi_paths(hermes_home=hermes_home, provision=False)
        verify_managed_pi(installed)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        return installed
    except Exception:
        if backup is not None and backup.exists():
            if paths.root.exists():
                os.replace(paths.root, staging)
            os.replace(backup, paths.root)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def launch_managed_pi_auth(
    *,
    hermes_home: str | Path | None = None,
    acknowledge_trusted_local: bool = False,
) -> int:
    """Run the separate elevated local auth flow after explicit acknowledgement."""
    if not acknowledge_trusted_local:
        raise RuntimeError(
            "Pi authentication is not contained like normal execution; rerun only after "
            "explicitly acknowledging the elevated trusted-local operation"
        )
    paths = managed_pi_paths(hermes_home=hermes_home, provision=True)
    executable = verify_managed_pi(paths)
    env = sanitized_pi_environment(
        os.environ,
        allowed_provider_vars=set(),
        runtime_values={
            "HOME": str(paths.home_dir),
            "PI_CODING_AGENT_DIR": str(paths.config_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(paths.session_root),
            "PI_TELEMETRY": "0",
        },
    )
    env.pop("PI_OFFLINE", None)
    return subprocess.call([
        executable, "--no-extensions", "--no-skills", "--no-prompt-templates",
        "--no-themes", "--no-context-files", "--no-session",
    ], env=env, cwd=paths.root)


def managed_pi_executable(configured: str | None = None) -> str:
    if configured:
        raise RuntimeError("arbitrary Pi executable paths are not allowed; use the managed pinned runtime")
    paths = managed_pi_paths(provision=True)
    return verify_managed_pi(paths)


def _safe_callback(callback: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args, **kwargs)
    except Exception:
        logger.debug("Pi display callback raised", exc_info=True)


def make_pi_event_bridge(agent: Any) -> Callable[[dict[str, Any]], None]:
    """Project live Pi deltas/tools/status into the existing Hermes UI hooks."""
    started: dict[str, tuple[str, dict[str, Any]]] = {}
    fire_and_forget_times: deque[float] = deque()
    fire_and_forget_methods = {"notify", "setStatus", "setWidget", "setTitle", "set_editor_text"}

    def on_event(event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "extension_ui_request" and event.get("method") in fire_and_forget_methods:
            now = time.monotonic()
            while fire_and_forget_times and fire_and_forget_times[0] <= now - 5.0:
                fire_and_forget_times.popleft()
            if len(fire_and_forget_times) >= 20:
                return
            fire_and_forget_times.append(now)
        if kind == "message_update":
            update = event.get("assistantMessageEvent") or {}
            if not isinstance(update, dict):
                return
            delta = str(update.get("delta") or "")
            if update.get("type") == "text_delta":
                _safe_callback(getattr(agent, "_fire_stream_delta", None), delta)
            elif update.get("type") == "thinking_delta":
                _safe_callback(getattr(agent, "_fire_reasoning_delta", None), delta)
            return
        if kind == "tool_execution_start":
            native = str(event.get("toolCallId") or "")
            call_id = f"pi_{native}"
            name = str(event.get("toolName") or "tool")
            raw_args = event.get("args")
            args_value = redact_protocol_value(raw_args, max_string=4000)
            args: dict[str, Any] = args_value if isinstance(args_value, dict) else {}
            started[call_id] = (name, args)
            _safe_callback(getattr(agent, "tool_progress_callback", None), "tool.started", name, None, args)
            _safe_callback(getattr(agent, "tool_start_callback", None), call_id, name, args)
            return
        if kind == "tool_execution_update":
            name = str(event.get("toolName") or "tool")
            preview = redact_protocol_text(event.get("partialResult") or "", limit=1000)
            _safe_callback(getattr(agent, "tool_progress_callback", None), "tool.progress", name, preview, None)
            return
        if kind == "tool_execution_end":
            call_id = f"pi_{event.get('toolCallId') or ''}"
            name, args = started.pop(call_id, (str(event.get("toolName") or "tool"), {}))
            result = redact_protocol_text(event.get("result") or "", limit=4000)
            is_error = bool(event.get("isError"))
            _safe_callback(
                getattr(agent, "tool_progress_callback", None),
                "tool.completed", name, None, None, is_error=is_error, result=result,
            )
            _safe_callback(getattr(agent, "tool_complete_callback", None), call_id, name, args, result)
            return
        if kind in {"compaction_start", "compaction_end", "auto_retry_start", "auto_retry_end", "queue_update"}:
            _safe_callback(
                getattr(agent, "status_callback", None),
                "lifecycle", f"Pi {kind.replace('_', ' ')}",
            )
        _safe_callback(
            getattr(agent, "event_callback", None),
            f"runtime:pi:{kind}",
            {"runtime": "pi_rpc", "event": redact_protocol_value(event, max_string=4000)},
        )

    return on_event


def make_pi_approval_handler(agent: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Map Pi confirm UI to Hermes' established dangerous-action prompt.

    Free-form/select/editor requests are not needed by the pinned bridge and
    stay fail-closed. Gateway/cron sessions without an installed interactive
    callback also deny immediately.
    """
    def handle(event: dict[str, Any]) -> dict[str, Any]:
        method = str(event.get("method") or "")
        if method in {"select", "input", "editor"}:
            clarify = getattr(agent, "clarify_callback", None)
            if not callable(clarify):
                return {"cancelled": True}
            title = redact_protocol_text(event.get("title") or "Pi runtime input", limit=1000)
            choices: list[str] | None = None
            if method == "select":
                raw_options = event.get("options")
                if not isinstance(raw_options, list):
                    return {"cancelled": True}
                choices = [
                    redact_protocol_text(option, limit=1000)
                    for option in raw_options[:100]
                    if isinstance(option, str)
                ]
                if not choices:
                    return {"cancelled": True}
            elif method == "input" and event.get("placeholder"):
                title += "\n" + redact_protocol_text(event.get("placeholder"), limit=1000)
            elif method == "editor" and event.get("prefill"):
                title += "\nCurrent text:\n" + redact_protocol_text(event.get("prefill"), limit=4000)
            try:
                answer = clarify(title, choices, multi_select=False)
            except Exception:
                logger.warning("Pi clarification mediation failed closed", exc_info=True)
                return {"cancelled": True}
            if answer is None or not str(answer).strip():
                return {"cancelled": True}
            value = redact_protocol_text(answer, limit=16 * 1024)
            if choices is not None and value not in choices:
                return {"cancelled": True}
            return {"value": value}
        if method != "confirm":
            return {"cancelled": True}
        try:
            from tools.approval import prompt_dangerous_approval
            from tools.terminal_tool import _get_approval_callback

            callback = _get_approval_callback()
            if callback is None:
                return {"confirmed": False}
            raw_timeout = event.get("timeout")
            timeout = max(1, min(int(raw_timeout or 30_000) // 1000, 60))
            verdict = prompt_dangerous_approval(
                redact_protocol_text(event.get("message") or "Pi dangerous action", limit=4000),
                redact_protocol_text(event.get("title") or "Pi runtime approval", limit=1000),
                timeout_seconds=timeout,
                allow_permanent=False,
                approval_callback=callback,
            )
            return {"confirmed": verdict in {"once", "session", "always"}}
        except Exception:
            logger.warning("Pi approval mediation failed closed", exc_info=True)
            return {"confirmed": False}

    return handle


def _session_slug(value: Any) -> str:
    return hashlib.sha256(str(value or "anonymous").encode("utf-8")).hexdigest()[:32]


def get_or_create_pi_session(agent: Any) -> PiRpcSession:
    """Return a digest-verified, read-only-workspace Pi adapter."""
    session = getattr(agent, "_pi_rpc_session", None)
    if session is not None:
        return session
    from agent.runtime_cwd import resolve_agent_cwd
    from hermes_cli.config import load_config

    config = load_config()
    pi_cfg = ((config.get("agent") or {}).get("pi_rpc") or {}) if isinstance(config, dict) else {}
    if not isinstance(pi_cfg, dict) or pi_cfg.get("enabled") is not True:
        raise RuntimeError("Pi RPC is experimental and disabled; set agent.pi_rpc.enabled: true")
    paths = managed_pi_paths(provision=False)
    cwd = str(
        Path(
            getattr(agent, "session_cwd", None) or resolve_agent_cwd()
        ).resolve(strict=True)
    )
    owned = paths.session_root / _session_slug(getattr(agent, "session_id", None))
    validate_workspace_boundary(
        cwd,
        config_dir=paths.config_dir,
        session_dir=owned,
        binding_root=paths.binding_root,
        sensitive_roots=(paths.root, paths.session_root, paths.home_dir),
    )
    paths = managed_pi_paths(provision=True)
    # Keep the host installation authenticity check, but never execute it for
    # an interactive turn. The container image is the containment boundary.
    verify_managed_pi(paths, run_version=False)
    image = str(pi_cfg.get("docker_image") or "").strip()
    if not image:
        raise RuntimeError(
            "interactive Pi requires agent.pi_rpc.docker_image as an exact immutable reference"
        )
    owned = paths.session_root / _session_slug(getattr(agent, "session_id", None))
    owned.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = str(pi_cfg.get("interactive_mode") or "inspect")
    if mode == "implement":
        raise RuntimeError("interactive Pi implement mode is unavailable; use coding_task with Docker containment")
    if mode not in {"review", "inspect", "plan"}:
        raise RuntimeError("interactive Pi mode must be review, inspect, or plan")
    provider_vars = pi_cfg.get("provider_env_allowlist") or []
    allowed_provider_vars = {str(name) for name in provider_vars if isinstance(name, str)}
    egress_allowlist = {
        str(host) for host in (pi_cfg.get("egress_allowlist") or [])
        if isinstance(host, str)
    }
    sweep = getattr(PiDockerSandbox, "sweep_stale_resources", None)
    active = getattr(PiDockerSandbox, "active_resource_identities", None)
    if callable(sweep):
        sweep(active_identities=active() if callable(active) else frozenset())
    sandbox = PiDockerSandbox(
        image=image,
        worktree=cwd,
        config_dir=paths.config_dir,
        session_dir=owned,
        extension_file=paths.extension_file,
        read_only_workspace=True,
        provider_env_names=allowed_provider_vars,
        egress_allowlist=egress_allowlist,
        resource_identity=f"interactive:{getattr(agent, 'session_id', None) or 'anonymous'}",
        binding_root=paths.binding_root,
        sensitive_roots=(paths.root, paths.session_root, paths.home_dir),
    )
    sandbox.verify_version(timeout=min(float(pi_cfg.get("turn_timeout_seconds") or 3600), 20.0))
    mark_active = getattr(sandbox, "mark_active", None)
    if callable(mark_active):
        mark_active()
    binding_dir = paths.binding_root / "interactive" / _session_slug(
        getattr(agent, "session_id", None)
    )
    binding_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        session = PiRpcSession(
            executable=None,
            cwd=cwd,
            config_dir=paths.config_dir,
            session_dir=owned,
            binding_path=binding_dir / "binding.json",
            event_callback=make_pi_event_bridge(agent),
            approval_handler=make_pi_approval_handler(agent),
            silence_timeout=float(pi_cfg.get("silence_timeout_seconds") or 300),
            turn_timeout=float(pi_cfg.get("turn_timeout_seconds") or 3600),
            mode=mode,
            sandbox=sandbox,
            allowed_provider_vars=allowed_provider_vars,
        )
    except Exception:
        sandbox.cleanup()
        raise
    agent._pi_rpc_session = session
    return session


def select_pi_model(agent: Any, provider: str, model_id: str):
    """Apply one exact native-catalog model and synchronize Hermes metadata."""
    provider = str(provider or "").strip()
    model_id = str(model_id or "").strip()
    if not provider or not model_id:
        raise ValueError("Pi model selection requires provider/model syntax")
    session = get_or_create_pi_session(agent)
    matches = [
        item for item in session.available_models()
        if isinstance(item, dict)
        and str(item.get("provider") or "") == provider
        and str(item.get("id") or item.get("modelId") or "") == model_id
    ]
    if len(matches) != 1:
        raise ValueError("Pi model must exactly match one provider/model in the native catalog")
    state = session.set_model(provider, model_id)
    if state.model_provider != provider or state.model_id != model_id:
        raise RuntimeError("Pi did not retain the exact requested provider/model")
    agent.provider = provider
    agent.model = model_id
    return state


def parse_pi_model_selection(model_input: str, explicit_provider: str = "") -> tuple[str, str]:
    """Resolve the one Pi model grammar shared by CLI, gateway, and TUI."""
    target = str(model_input or "").strip()
    provider = str(explicit_provider or "").strip()
    if provider and target:
        return provider, target
    if not provider and "/" in target:
        provider, target = (part.strip() for part in target.split("/", 1))
        if provider and target:
            return provider, target
    raise ValueError(
        "Pi model selection requires provider/model or <model> --provider <provider>"
    )


def format_pi_reconciliation_report(report: dict[str, Any]) -> str:
    """Render concise recovery state plus explicit safe next commands."""
    action = str(report.get("action") or "inspect")
    lines = [
        f"Pi reconciliation {action}: prompt={report.get('prompt_state') or 'unknown'}",
        (
            f"cursor {report.get('previous_cursor') or 'none'} → "
            f"{report.get('cursor') or 'none'} · "
            f"model={((report.get('model') or {}).get('provider') or 'unknown')}/"
            f"{((report.get('model') or {}).get('id') or 'unknown')} · "
            f"thinking={report.get('thinking_level') or 'off'}"
        ),
    ]
    if action == "inspect":
        lines.append(
            "native state: "
            f"streaming={bool(report.get('is_streaming'))} · "
            f"compacting={bool(report.get('is_compacting'))} · "
            f"pending={int(report.get('pending_message_count') or 0)}"
        )
        if report.get("cursor_advanced"):
            lines.append(
                "Adopt unavailable: the native cursor advanced beyond Hermes' canonical "
                "checkpoint. Exact transcript recovery is not available."
            )
            lines.append(
                "Next: /runtime reconcile abandon archives the prior binding, loses "
                "native continuity, and requires independent approval."
            )
        else:
            lines.append(
                "Next: /runtime reconcile adopt (unchanged cursor)  or  "
                "/runtime reconcile abandon (archival context loss)"
            )
    elif action == "abandon":
        if report.get("runtime_ownership_retired") is False:
            lines.append(
                "CONTEXT LOSS: binding archived, but old runtime cleanup is unconfirmed. "
                "Do not allocate a fresh Pi session until stale-resource cleanup succeeds."
            )
        else:
            lines.append(
                "CONTEXT LOSS: binding archived and old runtime ownership retired; the slot "
                "is unbound until the next Pi run. Native continuity will not resume."
            )
        lines.append("The uncertain prompt was not replayed; independent approval is required.")
    return "\n".join(lines)


def run_pi_rpc_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> dict[str, Any]:
    """Run a native Pi turn; Pi owns native context and Hermes stores projection."""
    _ = effective_task_id
    try:
        session = get_or_create_pi_session(agent)
    except Exception as exc:
        from agent.external_runtime.types import ExternalTurnResult, TurnStatus

        turn = ExternalTurnResult(
            status=TurnStatus.ERROR,
            error=redact_protocol_text(exc),
            should_retire=True,
            phase="before_send",
            prompt_accepted=False,
        )
    else:
        turn = session.run_turn(
            original_user_message if original_user_message is not None else user_message
        )
        if turn.should_retire:
            try:
                session.close()
            except Exception as exc:
                logger.warning("Pi retirement cleanup failed", exc_info=True)
                if turn.error is None:
                    turn.error = redact_protocol_text(exc)
            finally:
                agent._pi_rpc_session = None
    return finalize_external_turn(
        agent, messages, turn, runtime="pi_rpc",
        original_user_message=original_user_message,
        should_review_memory=should_review_memory,
    )


def reconcile_pi_agent(agent: Any, action: str) -> dict[str, Any]:
    """Reconcile the active conversation's exact native Pi binding without prompting."""
    if getattr(agent, "api_mode", None) != "pi_rpc":
        raise RuntimeError("Pi reconciliation requires an active Pi runtime session")
    normalized = str(action or "").strip().lower()
    if normalized == "abandon":
        from agent.runtime_cwd import resolve_agent_cwd

        paths = managed_pi_paths(provision=False)
        cwd = str(Path(getattr(agent, "session_cwd", None) or resolve_agent_cwd()).resolve(strict=True))
        owned = paths.session_root / _session_slug(getattr(agent, "session_id", None))
        binding_path = (
            paths.binding_root / "interactive" /
            _session_slug(getattr(agent, "session_id", None)) / "binding.json"
        )
        live_session = getattr(agent, "_pi_rpc_session", None)
        if live_session is not None:
            live_binding = Path(getattr(live_session, "binding_path", "")).expanduser().resolve()
            live_session_root = Path(getattr(live_session, "session_dir", "")).expanduser().resolve()
            if live_binding != binding_path.resolve() or live_session_root != owned.resolve():
                raise RuntimeError(
                    "active Pi runtime ownership does not match this conversation binding"
                )
            report = live_session.reconcile("abandon")
        else:
            # A retired process has already released its runtime ownership.
            # Host-only archival deliberately remains usable with no Docker
            # daemon, image, or native transcript available.
            report = PiRpcSession.abandon_binding(
                cwd=cwd, session_dir=owned, binding_path=binding_path,
            )
            report["runtime_ownership_retired"] = True
        agent._pi_rpc_session = None
    else:
        session = get_or_create_pi_session(agent)
        report = session.reconcile(normalized)
    if normalized == "adopt":
        state = session._runtime_state()
        if state.model_id:
            agent.model = state.model_id
        if state.model_provider:
            agent.provider = state.model_provider
    safe = redact_protocol_value(report, max_string=16 * 1024)
    return safe if isinstance(safe, dict) else {"error": "invalid reconciliation result"}
