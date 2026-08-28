"""One-shot external coding missions with hard worktree containment."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from agent.pi_runtime import managed_pi_paths
from agent.external_runtime.types import ExternalTurnResult, TurnStatus
from agent.transports.pi_docker_sandbox import (
    PiDockerSandbox,
    PiSandboxError,
    is_immutable_image_ref,
    validate_workspace_boundary,
)
from agent.transports.pi_rpc_client import redact_protocol_text, redact_protocol_value
from tools.registry import registry

MAX_TIMEOUT = 3600
MAX_VERIFICATION_COMMANDS = 20
_RESUME_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_READ_ONLY_MODES = frozenset({"review", "inspect", "plan"})


class CodingTaskError(ValueError):
    pass


def _git(args: list[str], cwd: Path, timeout: int = 20, *, binary: bool = False) -> str | bytes:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True,
        text=not binary, timeout=timeout,
    )
    if proc.returncode:
        stderr = proc.stderr.decode("utf-8", errors="replace") if binary else proc.stderr
        raise CodingTaskError(redact_protocol_text(stderr or "git worktree validation failed"))
    return proc.stdout


def validate_isolated_worktree(cwd: str) -> Path:
    root = Path(cwd).expanduser().resolve(strict=True)
    if not root.is_dir() or _git(["rev-parse", "--show-toplevel"], root).strip() != str(root):
        raise CodingTaskError("cwd must be the canonical root of an explicit Git worktree")
    dot_git = root / ".git"
    if not dot_git.is_file() or dot_git.is_symlink():
        raise CodingTaskError("write-capable coding tasks require an isolated linked Git worktree")
    gitdir_line = dot_git.read_text(encoding="utf-8", errors="replace").strip()
    if not gitdir_line.startswith("gitdir:"):
        raise CodingTaskError("invalid linked-worktree metadata")
    raw_gitdir = Path(gitdir_line.split(":", 1)[1].strip())
    gitdir = (root / raw_gitdir).resolve(strict=True) if not raw_gitdir.is_absolute() else raw_gitdir.resolve(strict=True)
    common_raw = Path(str(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], root)).strip())
    common = common_raw.resolve(strict=True)
    registered = common / "worktrees"
    if gitdir == common or not gitdir.is_relative_to(registered.resolve(strict=True)):
        raise CodingTaskError("cwd is not a registered isolated linked worktree")
    worktree_record = gitdir / "gitdir"
    if not worktree_record.is_file():
        raise CodingTaskError("linked-worktree registration is incomplete")
    recorded_dot_git = Path(worktree_record.read_text(encoding="utf-8", errors="replace").strip()).resolve(strict=True)
    if recorded_dot_git != dot_git.resolve(strict=True):
        raise CodingTaskError("linked-worktree registration does not point back to cwd")
    return root


def _status_snapshot(root: Path) -> dict[str, str]:
    raw = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], root, binary=True)
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    result: dict[str, str] = {}
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        text = field.decode("utf-8", errors="surrogateescape")
        if len(text) < 4:
            continue
        status = text[:2]
        path = text[3:]
        result[path] = status
        if "R" in status or "C" in status:
            if index < len(fields) and fields[index]:
                original = fields[index].decode("utf-8", errors="surrogateescape")
                index += 1
                result[original] = status
    return result


def _dirty_fingerprints(root: Path, status: dict[str, str]) -> dict[str, str]:
    """Fingerprint dirty paths so an allowed pre-existing edit stays attributable."""
    fingerprints: dict[str, str] = {}
    for relative in status:
        path = root / relative
        digest = hashlib.sha256()
        try:
            if path.is_symlink():
                digest.update(b"symlink\0" + str(path.readlink()).encode("utf-8"))
            elif path.is_file():
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif path.exists():
                digest.update(b"directory")
            else:
                digest.update(b"missing")
        except OSError as exc:
            digest.update(f"unreadable:{exc.__class__.__name__}".encode("ascii"))
        fingerprints[relative] = digest.hexdigest()
    return fingerprints


def _configured_pi() -> tuple[dict[str, Any], str]:
    from hermes_cli.config import load_config

    config = load_config()
    pi_cfg = ((config.get("agent") or {}).get("pi_rpc") or {}) if isinstance(config, dict) else {}
    if not isinstance(pi_cfg, dict) or pi_cfg.get("enabled") is not True:
        raise CodingTaskError("Pi RPC is experimental and disabled (agent.pi_rpc.enabled must be true)")
    image = str(pi_cfg.get("docker_image") or "").strip()
    if not image:
        raise CodingTaskError(
            "agent.pi_rpc.docker_image must name an exact repo digest or local image ID"
        )
    return pi_cfg, image


def check_coding_task_requirements() -> bool:
    try:
        pi_cfg, image = _configured_pi()
        paths = managed_pi_paths(provision=False)
        return bool(
            is_immutable_image_ref(image)
            and pi_cfg.get("enabled") is True
            and shutil.which("docker")
            and paths.extension_file.is_file()
            and paths.executable.is_file()
        )
    except Exception:
        return False


def _task_id(goal: str, root: Path) -> str:
    seed = f"{root}\0{goal}\0{time.time_ns()}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:32]


def _verification(
    commands: list[str],
    sandbox: PiDockerSandbox,
    deadline: float,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    if len(commands) > MAX_VERIFICATION_COMMANDS:
        raise CodingTaskError(
            f"verification accepts at most {MAX_VERIFICATION_COMMANDS} commands"
        )
    for index, command in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for not_run in commands[index:]:
                outcomes.append({
                    "command": redact_protocol_text(not_run, limit=4096),
                    "ok": False,
                    "not_run": True,
                    "error": "not run: task timeout exhausted",
                    "sandbox": "docker",
                })
            break
        try:
            argv = shlex.split(command, posix=True)
            PiDockerSandbox.validate_verification_argv(argv)
            outcome = sandbox.run_verification(argv, timeout=min(remaining, 600.0))
            outcome["command"] = redact_protocol_text(command, limit=4096)
            safe_outcome = redact_protocol_value(outcome, max_string=16 * 1024)
            outcomes.append(safe_outcome if isinstance(safe_outcome, dict) else {
                "command": redact_protocol_text(command, limit=4096),
                "ok": False,
                "error": "verification returned an invalid result",
            })
        except (ValueError, PiSandboxError) as exc:
            outcomes.append({
                "command": redact_protocol_text(command, limit=4096),
                "ok": False,
                "error": redact_protocol_text(exc),
            })
    return outcomes


def coding_task(
    *,
    goal: str = "",
    context: str = "",
    runtime: str = "pi",
    cwd: str,
    model: str | None = None,
    thinking: str = "high",
    mode: str = "implement",
    verification: list[str] | None = None,
    resume_id: str | None = None,
    timeout_seconds: int = MAX_TIMEOUT,
    allow_dirty_paths: list[str] | None = None,
    action: str = "run",
    parent_agent: Any = None,
    session_factory: Callable[..., Any] | None = None,
    sandbox_factory: Callable[..., Any] = PiDockerSandbox,
) -> dict[str, Any]:
    if getattr(parent_agent, "api_mode", None) == "pi_rpc":
        raise CodingTaskError("coding_task is blocked inside Pi to prevent recursive spawning")
    runtime = runtime.lower().replace("_rpc", "")
    if runtime != "pi":
        raise CodingTaskError("only the contained Pi coding_task adapter is available")
    action = str(action or "run").strip().lower()
    if action not in {"run", "inspect", "adopt", "abandon"}:
        raise CodingTaskError("action must be run, inspect, adopt, or abandon")
    if action == "run" and not goal.strip():
        raise CodingTaskError("goal is required")
    if action != "run" and not resume_id:
        raise CodingTaskError("reconciliation actions require an explicit resume_id")
    verification = list(verification or [])
    if action == "run":
        if len(verification) > MAX_VERIFICATION_COMMANDS:
            raise CodingTaskError(
                f"verification accepts at most {MAX_VERIFICATION_COMMANDS} commands"
            )
        if mode not in _READ_ONLY_MODES | {"implement"}:
            raise CodingTaskError("invalid coding task mode")
    root = validate_isolated_worktree(cwd)
    timeout_seconds = max(1, min(int(timeout_seconds), MAX_TIMEOUT)) if action == "run" else MAX_TIMEOUT
    deadline = time.monotonic() + timeout_seconds

    task_name = resume_id or _task_id(goal, root)
    if not _RESUME_ID.fullmatch(task_name):
        raise CodingTaskError("resume_id contains unsafe characters")
    if action == "abandon":
        # Abandon is a host-side recovery primitive. It must remain available
        # even when Pi is disabled and Docker/the reviewed image/process are
        # gone. Only the binding, assigned worktree, and managed session-root
        # ownership are consulted.
        paths = managed_pi_paths(provision=False)
        owned_session = (paths.session_root / "coding-tasks" / task_name).resolve()
        task_root = (paths.session_root / "coding-tasks").resolve()
        if not owned_session.is_relative_to(task_root):
            raise CodingTaskError("resume_id escaped the managed task-session directory")
        binding_dir = paths.binding_root / "coding-tasks" / task_name
        if session_factory is None:
            from agent.transports.pi_rpc_session import PiRpcSession
            report = PiRpcSession.abandon_binding(
                cwd=root,
                session_dir=owned_session,
                binding_path=binding_dir / "binding.json",
            )
        else:
            session = session_factory(
                executable="pi-host-only-recovery",
                cwd=str(root),
                config_dir=paths.config_dir,
                session_dir=owned_session,
                binding_path=binding_dir / "binding.json",
                mode="inspect",
            )
            try:
                report = session.reconcile("abandon")
            finally:
                session.close()
        return {
            "runtime": runtime,
            "mode": "reconcile",
            "sandbox": "host-only",
            "resume_id": task_name,
            "status": "reconciled",
            "reconciliation": redact_protocol_value(report, max_string=16 * 1024),
            "verification": [],
            "verification_passed": False,
            "unresolved_blockers": [],
            "requires_independent_approval": True,
        }

    pi_cfg, image = _configured_pi()
    paths = managed_pi_paths(provision=False)
    owned_session = (paths.session_root / "coding-tasks" / task_name).resolve()
    task_root = (paths.session_root / "coding-tasks").resolve()
    if not owned_session.is_relative_to(task_root):
        raise CodingTaskError("resume_id escaped the managed task-session directory")
    try:
        validate_workspace_boundary(
            root,
            config_dir=paths.config_dir,
            session_dir=owned_session,
            binding_root=paths.binding_root,
            sensitive_roots=(paths.root, paths.session_root, paths.home_dir),
        )
    except PiSandboxError as exc:
        raise CodingTaskError(str(exc)) from exc
    paths = managed_pi_paths(provision=True)
    owned_session = (paths.session_root / "coding-tasks" / task_name).resolve()
    brief_path = owned_session / "brief.md"
    try:
        owned_session.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise CodingTaskError(
            f"could not initialize managed Pi session storage: {redact_protocol_text(exc)}"
        ) from exc
    provider_vars = {
        str(name) for name in (pi_cfg.get("provider_env_allowlist") or []) if isinstance(name, str)
    }
    egress_allowlist = {
        str(host) for host in (pi_cfg.get("egress_allowlist") or []) if isinstance(host, str)
    }
    try:
        if sandbox_factory is PiDockerSandbox:
            PiDockerSandbox.sweep_stale_resources(
                active_identities=PiDockerSandbox.active_resource_identities(),
            )
        sandbox_kwargs: dict[str, Any] = {
            "image": image,
            "worktree": root,
            "config_dir": paths.config_dir,
            "session_dir": owned_session,
            "extension_file": paths.extension_file,
            "read_only_workspace": (action != "run" or mode in _READ_ONLY_MODES),
            "provider_env_names": provider_vars,
            "egress_allowlist": egress_allowlist,
            "resource_identity": task_name,
        }
        if sandbox_factory is PiDockerSandbox:
            sandbox_kwargs.update({
                "binding_root": paths.binding_root,
                "sensitive_roots": (paths.root, paths.session_root, paths.home_dir),
            })
        sandbox = sandbox_factory(
            **sandbox_kwargs,
        )
    except Exception as exc:
        raise CodingTaskError(
            f"could not initialize contained Pi runtime: {redact_protocol_text(exc)}"
        ) from exc

    if session_factory is None:
        from agent.transports.pi_rpc_session import PiRpcSession

        session_factory = PiRpcSession
    binding_dir = paths.binding_root / "coding-tasks" / task_name
    try:
        binding_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        cleanup = getattr(sandbox, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
        raise CodingTaskError(
            f"could not initialize Pi control-plane binding: {redact_protocol_text(exc)}"
        ) from exc
    session = None
    turn = ExternalTurnResult(
        status=TurnStatus.ERROR,
        error="contained Pi runtime did not start",
        should_retire=True,
        phase="before_send",
        prompt_accepted=False,
    )
    checks: list[dict[str, Any]] = []
    before: dict[str, str] = {}
    before_fingerprints: dict[str, str] = {}
    after: dict[str, str] = {}
    introduced: list[str] = []
    snapshot_taken = False
    durable_settlement = False
    settlement_record: str | None = None
    session_closed = False
    try:
        session = session_factory(
            executable=None,
            cwd=str(root),
            config_dir=paths.config_dir,
            session_dir=owned_session,
            binding_path=binding_dir / "binding.json",
            silence_timeout=min(float(pi_cfg.get("silence_timeout_seconds") or 300), timeout_seconds),
            turn_timeout=timeout_seconds,
            mode="inspect" if action != "run" else mode,
            sandbox=sandbox,
            allowed_provider_vars=provider_vars,
        )
        try:
            # Reconciliation is a control-plane operation. Branch before every
            # run-only catalog, model, thinking, brief, dirt, and verification
            # action so repeated task fields cannot mutate uncertain state.
            if action != "run":
                report = session.reconcile(action)
                safe_report = redact_protocol_value(report, max_string=16 * 1024)
                inspect_blockers: list[str] = []
                if action == "inspect":
                    inspect_blockers = [
                        (
                            "adopt is unavailable because the native cursor advanced; "
                            "explicit archival abandon loses native continuity and requires "
                            "independent approval"
                            if bool(report.get("cursor_advanced"))
                            else "explicit adopt or archival abandon is still required"
                        )
                    ]
                return {
                    "runtime": runtime,
                    "mode": "reconcile",
                    "sandbox": "docker",
                    "resume_id": task_name,
                    "status": "reconciliation_required" if action == "inspect" else "reconciled",
                    "reconciliation": safe_report,
                    "verification": [],
                    "verification_passed": False,
                    "unresolved_blockers": inspect_blockers,
                    "requires_independent_approval": True,
                }

            sandbox.verify_version(timeout=min(20.0, timeout_seconds))
            mark_active = getattr(sandbox, "mark_active", None)
            if callable(mark_active):
                mark_active()
            before = _status_snapshot(root)
            before_fingerprints = _dirty_fingerprints(root, before)
            snapshot_taken = True
            allowed_dirty = set(allow_dirty_paths or ())
            unexpected_dirty = sorted(set(before) - allowed_dirty)
            if unexpected_dirty:
                raise CodingTaskError(
                    "assigned worktree has pre-existing dirty paths that were not explicitly allowed: "
                    + ", ".join(unexpected_dirty[:20])
                )
            brief = (
                f"# Goal\n\n{goal.strip()}\n\n# Context and constraints\n\n{context.strip()}\n\n"
                f"# Mode\n\n{mode}\n\n# Verification requested\n\n"
                + "\n".join(f"- `{command}`" for command in verification)
                + "\n"
            )
            if model:
                if "/" not in model or model.startswith("/") or model.endswith("/"):
                    raise CodingTaskError("Pi model must use provider/model syntax")
                provider, model_id = model.split("/", 1)
                catalog = session.available_models()
                matches = [
                    item for item in catalog
                    if isinstance(item, dict)
                    and str(item.get("provider") or "") == provider
                    and str(item.get("id") or item.get("modelId") or "") == model_id
                ]
                if len(matches) != 1:
                    raise CodingTaskError(
                        "Pi model must exactly match one provider/model in the native catalog"
                    )
            # Validate every caller-controlled selection before overwriting a
            # resumable brief or changing any native durable setting.
            try:
                brief_path.write_text(brief, encoding="utf-8")
                brief_path.chmod(0o600)
            except OSError as exc:
                raise CodingTaskError(
                    f"could not write the managed Pi task brief: {redact_protocol_text(exc)}"
                ) from exc
            if model:
                session.set_model(provider, model_id)
            if thinking:
                session.set_thinking_level(thinking)
            # The complete brief is both durably stored in the dedicated
            # native-session directory and sent verbatim; Pi never depends on
            # parent chat history.
            candidate = session.run_turn(brief)
            if (
                not isinstance(candidate, ExternalTurnResult)
                or not isinstance(candidate.status, TurnStatus)
            ):
                raise RuntimeError("Pi runtime returned an invalid turn result")
            turn = candidate
            if turn.status is TurnStatus.COMPLETED and turn.error:
                raise RuntimeError(turn.error)
            if turn.status is TurnStatus.COMPLETED:
                # Freeze the assigned worktree before independent host
                # verification.  The writable/model-bearing Pi process must not
                # be able to race evidence collection or mutate after it passes.
                session.close()
                session_closed = True
                checks = _verification(verification, sandbox, deadline)
                after = _status_snapshot(root)
                after_fingerprints = _dirty_fingerprints(root, after)
                introduced = sorted(
                    path for path in set(before) | set(after)
                    if before.get(path) != after.get(path)
                    or before_fingerprints.get(path) != after_fingerprints.get(path)
                )
                verification_passed = bool(checks) and all(
                    bool(item.get("ok")) for item in checks
                )
                finalize_coding_task = getattr(session, "finalize_coding_task", None)
                if not callable(finalize_coding_task):
                    raise RuntimeError(
                        "Pi session does not expose the coding_task durability fence"
                    )
                settlement_record = finalize_coding_task(
                    turn=turn,
                    evidence={
                        "resume_id": task_name,
                        "mode": mode,
                        "required_verification_count": len(verification),
                        "verification": checks,
                        "verification_passed": verification_passed,
                        "workspace": {
                            "pre_existing_dirty_paths": sorted(before),
                            "changed_paths": introduced,
                        },
                    },
                )
                if not isinstance(settlement_record, str) or not settlement_record:
                    raise RuntimeError(
                        "Pi coding_task durability fence returned no settlement record"
                    )
                durable_settlement = True
        except Exception as exc:
            if turn.status is TurnStatus.COMPLETED and turn.prompt_accepted is True:
                fail_projection = getattr(session, "finalize_projection", None)
                if callable(fail_projection):
                    try:
                        fail_projection(persisted=False, turn=turn)
                    except Exception:
                        pass
                turn.status = TurnStatus.RECONCILIATION_REQUIRED
                turn.reconciliation_required = True
                turn.error = (
                    "contained Pi runtime completed, but durable coding_task settlement "
                    f"failed: {redact_protocol_text(exc)}"
                )
            else:
                turn = ExternalTurnResult(
                    status=TurnStatus.ERROR,
                    error=f"contained Pi runtime failed: {redact_protocol_text(exc)}",
                    should_retire=True,
                    phase="before_send",
                    prompt_accepted=False,
                )
        finally:
            if not session_closed:
                try:
                    session.close()
                except Exception as exc:
                    if turn.status is TurnStatus.COMPLETED:
                        turn.status = TurnStatus.RECONCILIATION_REQUIRED
                        turn.reconciliation_required = True
                        turn.error = (
                            "contained Pi runtime cleanup failed before durable settlement: "
                            f"{redact_protocol_text(exc)}"
                        )
                    elif not turn.error:
                        turn.error = (
                            "contained Pi runtime cleanup failed: "
                            f"{redact_protocol_text(exc)}"
                        )
    except Exception as exc:
        if turn.status is TurnStatus.COMPLETED and not durable_settlement:
            turn.status = TurnStatus.RECONCILIATION_REQUIRED
            turn.reconciliation_required = True
            turn.error = (
                "contained Pi runtime completed, but durable coding_task settlement "
                f"failed: {redact_protocol_text(exc)}"
            )
        else:
            turn = ExternalTurnResult(
                status=TurnStatus.ERROR,
                error=f"contained Pi runtime failed: {redact_protocol_text(exc)}",
                should_retire=True,
                phase="before_send",
                prompt_accepted=False,
            )
        checks = []
    finally:
        cleanup = getattr(sandbox, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
    if len(checks) < len(verification):
        for command in verification[len(checks):]:
            checks.append({
                "command": redact_protocol_text(command, limit=4096),
                "ok": False,
                "not_run": True,
                "error": "not run: Pi turn did not complete successfully",
                "sandbox": "docker",
            })
    if not after:
        after = _status_snapshot(root)
    if not snapshot_taken:
        before = dict(after)
        before_fingerprints = _dirty_fingerprints(root, before)
    after_fingerprints = _dirty_fingerprints(root, after)
    if not introduced:
        introduced = sorted(
            path for path in set(before) | set(after)
            if before.get(path) != after.get(path)
            or before_fingerprints.get(path) != after_fingerprints.get(path)
        )
    status_value = turn.status.value
    blockers = [redact_protocol_text(turn.error)] if turn.error else []
    if turn.status is not TurnStatus.COMPLETED:
        blockers.append(f"Pi turn did not complete successfully (status: {status_value})")
    if mode in _READ_ONLY_MODES and introduced:
        blockers.append("read-only Pi task changed the assigned worktree")
    verification_passed = bool(checks) and all(bool(item.get("ok")) for item in checks)
    if not verification:
        blockers.append("no trusted verification commands were configured")
    elif not verification_passed:
        blockers.append("one or more requested verification commands failed or were not run")
    return {
        "runtime": runtime,
        "mode": mode,
        "sandbox": "docker",
        "session_id": turn.session_id,
        "session_file": turn.session_file,
        "resume_id": task_name,
        "cursor": turn.cursor,
        "status": status_value,
        "final_response": redact_protocol_text(turn.final_text, limit=100_000),
        "pre_existing_dirty_paths": sorted(before),
        "changed_paths": introduced,
        "verification": checks,
        "verification_passed": verification_passed,
        "durable_settlement": durable_settlement,
        "settlement_record": settlement_record,
        "unresolved_blockers": blockers,
        "requires_independent_approval": True,
    }


CODING_TASK_SCHEMA = {
    "name": "coding_task",
    "description": (
        "Run a complete coding mission through a contained external runtime in an explicitly "
        "assigned isolated Git worktree. Results always require independent Hermes verification."
    ),
    "parameters": {"type": "object", "properties": {
        "goal": {"type": "string"}, "context": {"type": "string"},
        "runtime": {"type": "string", "enum": ["pi"]},
        "cwd": {"type": "string"}, "model": {"type": "string"},
        "thinking": {"type": "string", "enum": ["off", "minimal", "low", "medium", "high", "xhigh", "max"]},
        "mode": {"type": "string", "enum": ["implement", "review", "inspect", "plan"]},
        "verification": {
            "type": "array", "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_VERIFICATION_COMMANDS,
            "description": (
                "Required for action=run; every command is captured before durable settlement."
            ),
        },
        "resume_id": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["run", "inspect", "adopt", "abandon"],
            "description": (
                "Reconciliation actions never send a prompt. Adopt works only when the native "
                "cursor did not advance; abandon archives the prior binding and loses native "
                "conversation continuity."
            ),
        },
        "allow_dirty_paths": {"type": "array", "items": {"type": "string"}},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT},
    }, "required": ["runtime", "cwd"]},
}

registry.register(
    name="coding_task", toolset="coding", schema=CODING_TASK_SCHEMA,
    handler=lambda args, **kw: json.dumps(
        coding_task(**args, parent_agent=kw.get("parent_agent")),
        ensure_ascii=False,
        sort_keys=True,
    ),
    check_fn=check_coding_task_requirements,
    emoji="🛠️", max_result_size_chars=100_000,
)
