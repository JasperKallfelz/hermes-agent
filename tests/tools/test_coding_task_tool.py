import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.transports.pi_docker_sandbox as pi_docker_sandbox
from agent.external_runtime.binding import load_binding
from agent.external_runtime.types import ExternalTurnResult, TurnStatus
from agent.transports.pi_docker_sandbox import (
    _REVIEWED_IMAGE_LABELS,
    _REVIEWED_IMAGE_IDS,
    PiDockerSandbox,
    PiSandboxError,
)
from agent.transports.pi_rpc_session import PiRpcSession
from tools.coding_task_tool import (
    CodingTaskError,
    _verification,
    coding_task,
    validate_isolated_worktree,
)
from tools.registry import registry


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)
    return proc.stdout.strip()


@pytest.fixture
def linked_worktree(tmp_path):
    main = tmp_path / "main"
    work = tmp_path / "assigned"
    main.mkdir()
    _git(main, "init")
    _git(main, "config", "user.email", "test@example.invalid")
    _git(main, "config", "user.name", "Test")
    (main / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(main, "add", "tracked.txt")
    _git(main, "commit", "-m", "base")
    _git(main, "worktree", "add", "-b", "task", str(work))
    return work


def test_parent_checkout_is_rejected(tmp_path):
    with pytest.raises((CodingTaskError, FileNotFoundError)):
        validate_isolated_worktree(str(tmp_path))


def test_recursion_is_blocked_before_runtime_launch(tmp_path):
    parent = type("Agent", (), {"api_mode": "pi_rpc"})()
    with pytest.raises(CodingTaskError, match="recursive"):
        coding_task(goal="x", runtime="pi", cwd=str(tmp_path), parent_agent=parent)


def test_registered_linked_worktree_is_accepted(linked_worktree):
    assert validate_isolated_worktree(str(linked_worktree)) == linked_worktree.resolve()


def test_preexisting_dirty_paths_are_refused(monkeypatch, tmp_path, linked_worktree):
    _patch_runtime(monkeypatch, tmp_path)
    (linked_worktree / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    result = coding_task(
        goal="x", runtime="pi", cwd=str(linked_worktree),
        session_factory=_Session, sandbox_factory=_Sandbox,
    )
    assert result["status"] == "error"
    assert any("pre-existing dirty" in blocker for blocker in result["unresolved_blockers"])


class _Sandbox:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def verify_version(self, **kwargs):
        pass

    def run_verification(self, argv, timeout):
        return {"argv": argv, "ok": True, "exit_code": 0, "stdout": "ok", "stderr": ""}

    def cleanup(self, **kwargs):
        self.cleaned = True


class _Session:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set_model(self, provider, model):
        pass

    def available_models(self):
        return [{"provider": "openai", "id": "gpt-test"}]

    def set_thinking_level(self, level):
        pass

    def run_turn(self, brief):
        return ExternalTurnResult(
            final_text="done", session_id="native", session_file="session.jsonl", cursor="leaf"
        )

    def finalize_coding_task(self, *, turn, evidence):
        if evidence["required_verification_count"] <= 0:
            raise RuntimeError("host verification evidence is required")
        assert len(evidence["verification"]) == evidence["required_verification_count"]
        return "coding-task-settlement-test.json"

    def close(self):
        pass


def _patch_runtime(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    config = root / "config"
    sessions = root / "sessions"
    bindings = root / "bindings"
    home = root / "home"
    config.mkdir(parents=True)
    sessions.mkdir()
    bindings.mkdir()
    home.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("export default function () {}\n", encoding="utf-8")
    paths = SimpleNamespace(
        root=root, config_dir=config, session_root=sessions, binding_root=bindings,
        home_dir=home,
        extension_file=extension,
    )
    monkeypatch.setattr(
        "tools.coding_task_tool._configured_pi",
        lambda: ({"enabled": True}, "registry.invalid/pi@sha256:" + "a" * 64),
    )
    monkeypatch.setattr("tools.coding_task_tool.managed_pi_paths", lambda provision=True: paths)


def test_read_only_task_uses_read_only_mount_and_empty_verification_fails(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)
    _Sandbox.instances.clear()
    result = coding_task(
        goal="review", runtime="pi", cwd=str(linked_worktree), mode="review",
        session_factory=_Session, sandbox_factory=_Sandbox,
    )
    assert _Sandbox.instances[-1].kwargs["read_only_workspace"] is True
    assert result["changed_paths"] == []
    assert result["verification_passed"] is False
    assert result["status"] == "reconciliation_required"
    assert result["durable_settlement"] is False
    assert "no trusted verification" in result["unresolved_blockers"][-1]


def test_verification_runs_after_completed_turn(monkeypatch, tmp_path, linked_worktree):
    _patch_runtime(monkeypatch, tmp_path)
    order = []

    class OrderedSession(_Session):
        def run_turn(self, brief):
            order.append("native-completed")
            return super().run_turn(brief)

        def close(self):
            order.append("pi-closed")

        def finalize_coding_task(self, *, turn, evidence):
            order.append("durable-settlement")
            return super().finalize_coding_task(turn=turn, evidence=evidence)

    class OrderedSandbox(_Sandbox):
        def run_verification(self, argv, timeout):
            order.append("host-verification")
            return super().run_verification(argv, timeout)

    result = coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], session_factory=OrderedSession,
        sandbox_factory=OrderedSandbox,
    )
    assert result["verification_passed"] is True
    assert result["verification"][0]["argv"] == ["pytest", "-q"]
    assert result["durable_settlement"] is True
    assert order == [
        "native-completed", "pi-closed", "host-verification", "durable-settlement",
    ]


def test_completed_turn_fails_closed_when_coding_task_settlement_cannot_land(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)

    class SettlementFailure(_Session):
        def finalize_coding_task(self, *, turn, evidence):
            assert turn.status is TurnStatus.COMPLETED
            assert len(evidence["verification"]) == 1
            raise OSError("binding disk unavailable")

    result = coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], session_factory=SettlementFailure,
        sandbox_factory=_Sandbox,
    )

    assert result["status"] == "reconciliation_required"
    assert result["durable_settlement"] is False
    assert result["verification_passed"] is True
    assert any("durable coding_task settlement" in item for item in result["unresolved_blockers"])


def test_real_pi_session_coding_task_settles_and_same_resume_id_runs_again(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)

    class ResumableClient:
        prompt_calls = []

        def __init__(self, **kwargs):
            extra = kwargs["extra_args"]
            session_dir = Path(extra[extra.index("--session-dir") + 1])
            if "--session" in extra:
                self.session_file = Path(extra[extra.index("--session") + 1])
                stored = json.loads(self.session_file.read_text(encoding="utf-8"))
                self.session_id = stored["session_id"]
                self.cursor = stored["cursor"]
                self.thinking = stored.get("thinking")
            else:
                self.session_id = extra[extra.index("--session-id") + 1]
                self.session_file = session_dir / f"{self.session_id}.jsonl"
                self.cursor = None
                self.thinking = None
                self._save()
            self.events = iter(())

        def _save(self):
            self.session_file.write_text(
                json.dumps({
                    "session_id": self.session_id,
                    "cursor": self.cursor,
                    "thinking": self.thinking,
                }) + "\n",
                encoding="utf-8",
            )

        def start(self):
            pass

        def is_alive(self):
            return True

        def drain_events(self):
            return []

        def request(self, kind, **kwargs):
            if kind == "get_state":
                return {"data": {
                    "sessionId": self.session_id,
                    "sessionFile": str(self.session_file),
                    "thinkingLevel": self.thinking,
                }}
            if kind == "get_entries":
                entries = [{"id": self.cursor}] if self.cursor else []
                return {"data": {"leafId": self.cursor, "entries": entries}}
            if kind == "set_thinking_level":
                self.thinking = kwargs["level"]
                self._save()
                return {"success": True}
            if kind == "prompt":
                self.__class__.prompt_calls.append(kwargs["message"])
                self.cursor = f"leaf-{len(self.__class__.prompt_calls)}"
                self._save()
                self.events = iter([
                    {"type": "agent_start"},
                    {"type": "message_end", "message": {"content": "done"}},
                    {"type": "agent_settled"},
                ])
                return {"success": True, "_hermes_sequence": 0}
            raise AssertionError(kind)

        def next_event(self, timeout=1):
            return next(self.events)

        def close(self, **_kwargs):
            pass

    def real_session_factory(**kwargs):
        kwargs.pop("sandbox")
        kwargs["executable"] = "pi"
        kwargs["client_factory"] = ResumableClient
        return PiRpcSession(**kwargs)

    ResumableClient.prompt_calls = []
    first = coding_task(
        goal="first", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], resume_id="durable-resume",
        session_factory=real_session_factory, sandbox_factory=_Sandbox,
    )
    binding_path = (
        tmp_path / "runtime" / "bindings" / "coding-tasks" /
        "durable-resume" / "binding.json"
    )
    first_binding = load_binding(binding_path)
    settlement_files = sorted(binding_path.parent.glob("coding-task-settlement-*.json"))

    assert first["status"] == "completed"
    assert first["durable_settlement"] is True
    assert first_binding.prompt_state == "settled"
    assert first_binding.prompt_request_id is None
    assert len(settlement_files) == 1
    evidence = json.loads(settlement_files[0].read_text(encoding="utf-8"))
    assert evidence["required_verification_count"] == 1
    assert evidence["native_result"]["cursor"] == "leaf-1"

    second = coding_task(
        goal="second", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], resume_id="durable-resume",
        session_factory=real_session_factory, sandbox_factory=_Sandbox,
    )
    second_binding = load_binding(binding_path)

    assert second["status"] == "completed"
    assert second["durable_settlement"] is True
    assert second["session_id"] == first["session_id"]
    assert second_binding.prompt_state == "settled"
    assert second_binding.last_entry_id == "leaf-2"
    assert len(ResumableClient.prompt_calls) == 2
    assert len(list(binding_path.parent.glob("coding-task-settlement-*.json"))) == 2


def test_binding_is_host_only_and_outside_pi_session_mount(monkeypatch, tmp_path, linked_worktree):
    _patch_runtime(monkeypatch, tmp_path)
    captured = {}

    class CapturingSession(_Session):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], resume_id="safe-resume",
        session_factory=CapturingSession, sandbox_factory=_Sandbox,
    )
    binding = Path(captured["binding_path"])
    session_dir = Path(captured["session_dir"])
    assert not binding.is_relative_to(session_dir)
    assert binding.parent.parent.parent == tmp_path / "runtime" / "bindings"


def test_runtime_exception_returns_redacted_failure_and_always_cleans_up(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)
    secret = "sk-" + "s" * 40

    class ExplodingSession(_Session):
        closed = False

        def run_turn(self, brief):
            raise RuntimeError(f"provider rejected {secret}")

        def close(self):
            self.__class__.closed = True

    _Sandbox.instances.clear()
    result = coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], session_factory=ExplodingSession,
        sandbox_factory=_Sandbox,
    )
    assert result["status"] == "error"
    assert result["verification_passed"] is False
    assert any("status: error" in blocker for blocker in result["unresolved_blockers"])
    assert secret not in json.dumps(result)
    assert ExplodingSession.closed is True
    assert _Sandbox.instances[-1].cleaned is True


def test_more_than_twenty_verification_commands_is_rejected_before_launch(tmp_path):
    with pytest.raises(CodingTaskError, match="at most 20"):
        coding_task(
            goal="inspect", runtime="pi", cwd=str(tmp_path),
            verification=["pytest -q"] * 21,
        )


@pytest.mark.parametrize(
    "status",
    [
        TurnStatus.PARTIAL,
        TurnStatus.INTERRUPTED,
        TurnStatus.ERROR,
        TurnStatus.RECONCILIATION_REQUIRED,
    ],
)
def test_every_noncompleted_turn_has_explicit_blocker(
    monkeypatch, tmp_path, linked_worktree, status
):
    _patch_runtime(monkeypatch, tmp_path)

    class NonCompletedSession(_Session):
        def run_turn(self, brief):
            return ExternalTurnResult(status=status, error=None)

    result = coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=["pytest -q"], session_factory=NonCompletedSession,
        sandbox_factory=_Sandbox,
    )
    assert len(result["verification"]) == 1
    assert result["verification"][0]["not_run"] is True
    assert result["verification_passed"] is False
    assert f"Pi turn did not complete successfully (status: {status.value})" in result["unresolved_blockers"]
    assert any("failed or were not run" in blocker for blocker in result["unresolved_blockers"])


def test_deadline_exhaustion_emits_evidence_for_every_remaining_command(monkeypatch):
    monkeypatch.setattr("tools.coding_task_tool.time.monotonic", lambda: 100.0)
    outcomes = _verification(
        ["pytest -q", "ruff check .", "ty check"], _Sandbox(), deadline=99.0
    )
    assert [item["command"] for item in outcomes] == [
        "pytest -q", "ruff check .", "ty check",
    ]
    assert all(item["ok"] is False and item["not_run"] is True for item in outcomes)


def test_invalid_model_does_not_overwrite_resumable_brief(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)
    brief = tmp_path / "sessions" / "coding-tasks" / "resume-safe" / "brief.md"
    brief.parent.mkdir(parents=True)
    brief.write_text("original brief\n", encoding="utf-8")

    class CatalogSession(_Session):
        set_calls = []

        def available_models(self):
            return [{"provider": "anthropic", "id": "claude-reviewed"}]

        def set_model(self, provider, model):
            self.set_calls.append((provider, model))

        def run_turn(self, brief):
            raise AssertionError("invalid model must not send a prompt")

    result = coding_task(
        goal="replacement", runtime="pi", cwd=str(linked_worktree),
        resume_id="resume-safe", model="openai/not-in-catalog",
        verification=["pytest -q"], session_factory=CatalogSession,
        sandbox_factory=_Sandbox,
    )
    assert result["status"] == "error"
    assert brief.read_text(encoding="utf-8") == "original brief\n"
    assert CatalogSession.set_calls == []


@pytest.mark.parametrize("action", ["inspect", "adopt", "abandon"])
def test_coding_task_reconciliation_action_never_sends_prompt(
    monkeypatch, tmp_path, linked_worktree, action
):
    _patch_runtime(monkeypatch, tmp_path)
    calls = []

    class ReconcileSession(_Session):
        def reconcile(self, requested):
            calls.append(("reconcile", requested))
            return {"action": requested, "replayed": False}

        def run_turn(self, brief):
            calls.append(("prompt", brief))
            raise AssertionError("reconciliation must not prompt")

    result = coding_task(
        runtime="pi", cwd=str(linked_worktree), resume_id="uncertain",
        action=action, session_factory=ReconcileSession, sandbox_factory=_Sandbox,
    )
    assert calls == [("reconcile", action)]
    assert result["reconciliation"]["replayed"] is False


def test_reconciliation_ignores_every_run_only_field_without_mutation(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)
    task_dir = tmp_path / "sessions" / "coding-tasks" / "uncertain"
    task_dir.mkdir(parents=True)
    brief = task_dir / "brief.md"
    brief.write_text("authoritative original\n", encoding="utf-8")
    (linked_worktree / "tracked.txt").write_text("already dirty\n", encoding="utf-8")
    calls = []

    class ControlOnlySandbox(_Sandbox):
        def verify_version(self, **kwargs):
            raise AssertionError("reconciliation must not inspect the image")

        def run_verification(self, argv, timeout):
            raise AssertionError("reconciliation must not verify")

    class ControlOnlySession(_Session):
        def available_models(self):
            raise AssertionError("reconciliation must not query models")

        def set_model(self, provider, model):
            raise AssertionError("reconciliation must not set a model")

        def set_thinking_level(self, level):
            raise AssertionError("reconciliation must not set thinking")

        def run_turn(self, brief):
            raise AssertionError("reconciliation must not prompt")

        def reconcile(self, requested):
            calls.append(requested)
            return {"action": requested, "replayed": False}

    result = coding_task(
        goal="must be ignored", context="ignored", runtime="pi",
        cwd=str(linked_worktree), resume_id="uncertain", action="inspect",
        model="not/even-queried", thinking="invalid-and-ignored", mode="invalid-and-ignored",
        verification=["git status"] * 21, allow_dirty_paths=["wrong"],
        timeout_seconds="not-an-integer",
        session_factory=ControlOnlySession, sandbox_factory=ControlOnlySandbox,
    )
    assert calls == ["inspect"]
    assert result["verification"] == []
    assert brief.read_text(encoding="utf-8") == "authoritative original\n"


def test_coding_task_abandon_requires_no_pi_config_image_or_docker(
    monkeypatch, tmp_path, linked_worktree
):
    _patch_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "tools.coding_task_tool._configured_pi",
        lambda: (_ for _ in ()).throw(AssertionError("must not read Pi runtime config")),
    )
    calls = []

    class HostOnlySession(_Session):
        def reconcile(self, action):
            calls.append(action)
            return {"action": action, "slot_bound": False, "replayed": False}

    result = coding_task(
        runtime="pi", cwd=str(linked_worktree), resume_id="uncertain",
        action="abandon", model="ignored/model", thinking="ignored",
        verification=["git status"], session_factory=HostOnlySession,
        sandbox_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not initialize Docker")
        ),
    )
    assert calls == ["abandon"]
    assert result["sandbox"] == "host-only"
    assert result["reconciliation"]["slot_bound"] is False


def test_registered_coding_task_dispatch_returns_success_string(monkeypatch):
    expected = {"status": "completed", "final_response": "ok"}
    monkeypatch.setattr("tools.coding_task_tool.coding_task", lambda **_kwargs: expected)
    result = registry.dispatch("coding_task", {"runtime": "pi", "cwd": "/assigned"})
    assert isinstance(result, str)
    assert json.loads(result) == expected
    assert "tool_result_contract" not in result


def test_verification_evidence_is_recursively_redacted(monkeypatch, tmp_path, linked_worktree):
    _patch_runtime(monkeypatch, tmp_path)
    secret = "sk-" + "z" * 40

    class SecretSandbox(_Sandbox):
        def run_verification(self, argv, timeout):
            return {
                "argv": argv, "ok": False, "exit_code": 1,
                "stdout": {"nested": secret}, "stderr": secret,
            }

    result = coding_task(
        goal="inspect", runtime="pi", cwd=str(linked_worktree), mode="inspect",
        verification=[f"pytest --token={secret}"], session_factory=_Session,
        sandbox_factory=SecretSandbox,
    )
    assert secret not in json.dumps(result["verification"])


def test_rejected_verification_command_is_redacted(tmp_path):
    secret = "sk-" + "v" * 40
    sandbox = SimpleNamespace()

    outcomes = _verification(
        [f"unreviewed-tool --token={secret}"],
        sandbox,
        deadline=float("inf"),
    )

    assert secret not in json.dumps(outcomes)
    assert outcomes[0]["ok"] is False


def test_docker_command_has_hardening_and_only_three_mount_roots(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="registry.invalid/pi@sha256:" + "b" * 64,
        worktree=work, config_dir=config, session_dir=sessions, extension_file=extension,
    )
    argv = sandbox.pi_launch_command(["--mode", "rpc"])
    joined = " ".join(argv)
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    user_index = argv.index("--user")
    assert int(argv[user_index + 1].split(":", 1)[0]) > 0
    assert "--name" in argv
    assert "docker.sock" not in joined
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert mounts == [
        f"type=bind,src={work.resolve()},dst=/workspace",
        f"type=bind,src={config.resolve()},dst=/pi-config,readonly",
        f"type=bind,src={sessions.resolve()},dst=/pi-sessions",
    ]
    assert f"--network {sandbox.network_name}" in joined
    assert "HTTPS_PROXY=http://pi-egress-proxy:8080" in joined
    assert "NODE_USE_ENV_PROXY=1" in joined


def test_root_host_identity_never_selects_container_root(tmp_path, monkeypatch):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    monkeypatch.setattr(pi_docker_sandbox.os, "getuid", lambda: 0)
    monkeypatch.setattr(pi_docker_sandbox.os, "getgid", lambda: 0)
    sandbox = PiDockerSandbox(
        image="sha256:" + "9" * 64,
        worktree=work,
        config_dir=config,
        session_dir=sessions,
        extension_file=extension,
    )

    argv = sandbox._base_run_args(network_none=True)
    assert argv[argv.index("--user") + 1] == "65532:65532"


def test_base_run_args_read_only_override_keeps_workspace_and_config_readonly(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "e" * 64,
        worktree=work, config_dir=config, session_dir=sessions,
        extension_file=extension,
    )
    argv = sandbox._base_run_args(
        workspace_read_only=True, network_none=True, include_provider_env=False,
    )
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert mounts == [
        f"type=bind,src={work.resolve()},dst=/workspace,readonly",
        f"type=bind,src={config.resolve()},dst=/pi-config,readonly",
        f"type=bind,src={sessions.resolve()},dst=/pi-sessions",
    ]
    assert all(
        mount.endswith(",readonly")
        for mount in mounts
        if "dst=/workspace" in mount or "dst=/pi-config" in mount
    )


def test_read_only_workspace_symlink_does_not_add_host_target_mount(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    host_secrets = tmp_path / "host-secrets"
    for directory in (work, config, sessions, host_secrets):
        directory.mkdir()
    (host_secrets / "token").write_text("do-not-mount", encoding="utf-8")
    (work / "escape").symlink_to(host_secrets, target_is_directory=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "9" * 64,
        worktree=work, config_dir=config, session_dir=sessions,
        extension_file=extension, read_only_workspace=True,
    )
    argv = sandbox.pi_launch_command(["--mode", "rpc"])
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert mounts == [
        f"type=bind,src={work.resolve()},dst=/workspace,readonly",
        f"type=bind,src={config.resolve()},dst=/pi-config,readonly",
        f"type=bind,src={sessions.resolve()},dst=/pi-sessions",
    ]
    assert all(str(host_secrets.resolve()) not in mount for mount in mounts)


@pytest.mark.parametrize("path_kind", ["workspace", "config", "session", "extension", "binding"])
def test_every_dynamic_mount_class_rejects_comma_field_injection_before_docker(
    tmp_path, monkeypatch, path_kind,
):
    paths = {
        "workspace": tmp_path / "work",
        "config": tmp_path / "config",
        "session": tmp_path / "sessions",
        "binding": tmp_path / "bindings",
    }
    if path_kind in paths:
        paths[path_kind] = tmp_path / f"{path_kind},bind-nonrecursive"
    for directory in paths.values():
        directory.mkdir()
    extension_name = (
        "hermes_bridge,bind-nonrecursive.ts"
        if path_kind == "extension"
        else "hermes_bridge.ts"
    )
    extension = paths["config"] / extension_name
    extension.write_text("x", encoding="utf-8")
    docker_calls = []
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: docker_calls.append((args, kwargs)),
    )
    with pytest.raises(PiSandboxError, match="cannot be serialized safely"):
        PiDockerSandbox(
            image=next(iter(_REVIEWED_IMAGE_IDS)),
            worktree=paths["workspace"],
            config_dir=paths["config"],
            session_dir=paths["session"],
            binding_root=paths["binding"],
            extension_file=extension,
        )
    assert docker_calls == []


@pytest.mark.parametrize("delimiter", ["\n", "\r", "\x1f", "\x7f"])
def test_verification_mount_rejects_control_delimiters(tmp_path, delimiter):
    work = tmp_path / f"verification{delimiter}field"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    with pytest.raises(PiSandboxError, match="cannot be serialized safely"):
        PiDockerSandbox(
            image=next(iter(_REVIEWED_IMAGE_IDS)),
            worktree=work,
            config_dir=config,
            session_dir=sessions,
            extension_file=extension,
        )


def test_spaces_and_unicode_are_safe_in_all_dynamic_mount_paths(tmp_path):
    work = tmp_path / "work safe 雪"
    config = tmp_path / "config safe 雪"
    sessions = tmp_path / "sessions safe 雪"
    bindings = tmp_path / "bindings safe 雪"
    for directory in (work, config, sessions, bindings):
        directory.mkdir()
    extension = config / "bridge safe 雪.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image=next(iter(_REVIEWED_IMAGE_IDS)),
        worktree=work,
        config_dir=config,
        session_dir=sessions,
        binding_root=bindings,
        extension_file=extension,
    )
    runtime_mounts = [
        argv
        for index, argv in enumerate(sandbox.pi_launch_command(["--mode", "rpc"]))
        if index > 0
        and sandbox.pi_launch_command(["--mode", "rpc"])[index - 1] == "--mount"
    ]
    verification_mounts = [
        argv
        for index, argv in enumerate(sandbox._verification_run_args())
        if index > 0 and sandbox._verification_run_args()[index - 1] == "--mount"
    ]
    assert len(runtime_mounts) == 3
    assert len(verification_mounts) == 1
    assert all(" safe 雪" in mount for mount in runtime_mounts)
    assert "work safe 雪" in verification_mounts[0]


def test_real_linked_worktree_with_mount_field_name_is_rejected(tmp_path, monkeypatch):
    main = tmp_path / "main"
    victim = tmp_path / "victim,bind-nonrecursive"
    main.mkdir()
    _git(main, "init")
    _git(main, "config", "user.email", "test@example.invalid")
    _git(main, "config", "user.name", "Test")
    (main / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(main, "add", "tracked.txt")
    _git(main, "commit", "-m", "base")
    _git(main, "worktree", "add", "-b", "victim", str(victim))
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    config.mkdir()
    sessions.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    docker_calls = []
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: docker_calls.append((args, kwargs)),
    )
    with pytest.raises(PiSandboxError, match="cannot be serialized safely"):
        PiDockerSandbox(
            image=next(iter(_REVIEWED_IMAGE_IDS)),
            worktree=victim,
            config_dir=config,
            session_dir=sessions,
            extension_file=extension,
        )
    assert docker_calls == []


@pytest.mark.parametrize(
    "workspace_case",
    ["descendant", "ancestor", "equal", "alias", "symlink", "runtime-descendant"],
)
def test_sandbox_constructor_rejects_workspace_control_root_overlap_before_docker(
    tmp_path, monkeypatch, workspace_case
):
    runtime = tmp_path / "runtime"
    config = runtime / "managed-config"
    sessions = runtime / "sessions" / "owned"
    bindings = runtime / "bindings"
    home = runtime / "home"
    safe = tmp_path / "work"
    for directory in (config, sessions, bindings, home, safe):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    if workspace_case == "descendant":
        workspace = config / "project"
        workspace.mkdir()
    elif workspace_case == "ancestor":
        workspace = tmp_path
    elif workspace_case == "equal":
        workspace = config
    elif workspace_case == "alias":
        workspace = runtime / ".." / "runtime" / "managed-config"
    elif workspace_case == "symlink":
        workspace = tmp_path / "workspace-link"
        workspace.symlink_to(config, target_is_directory=True)
    else:
        workspace = runtime / "unmounted-project"
        workspace.mkdir()

    docker_calls = []
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: docker_calls.append((args, kwargs)),
    )
    with pytest.raises(PiSandboxError, match="overlaps"):
        PiDockerSandbox(
            image="sha256:" + "a" * 64,
            worktree=workspace,
            config_dir=config,
            session_dir=sessions,
            binding_root=bindings,
            sensitive_roots=(runtime, sessions.parent, home),
            extension_file=extension,
        )
    assert docker_calls == []


def test_sandbox_constructor_allows_hermes_worktree_sibling_of_runtime(tmp_path):
    hermes = tmp_path / ".hermes"
    runtime = hermes / "runtimes" / "pi" / "0.84.3"
    work = hermes / "worktrees" / "assigned"
    config = runtime / "managed-config"
    sessions = runtime / "sessions" / "owned"
    bindings = runtime / "bindings"
    home = runtime / "home"
    for directory in (work, config, sessions, bindings, home):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "b" * 64,
        worktree=work,
        config_dir=config,
        session_dir=sessions,
        binding_root=bindings,
        sensitive_roots=(runtime, sessions.parent, home),
        extension_file=extension,
    )
    assert sandbox.worktree == work.resolve()


def test_verification_allowlist_rejects_shells_and_inline_code():
    with pytest.raises(PiSandboxError):
        PiDockerSandbox.validate_verification_argv(["sh", "-c", "touch /host"])
    with pytest.raises(PiSandboxError):
        PiDockerSandbox.validate_verification_argv(["python", "-c", "print(1)"])


def test_tag_only_image_reference_is_rejected(tmp_path):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    with pytest.raises(PiSandboxError, match="exact repo@sha256"):
        PiDockerSandbox(
            image="hermes-pi:0.84.3", worktree=tmp_path / "work",
            config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
            extension_file=extension,
        )


@pytest.mark.parametrize("reference_kind", ["repo", "id"])
def test_image_inspection_accepts_both_immutable_reference_kinds(
    tmp_path, monkeypatch, reference_kind
):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    image_id = next(iter(_REVIEWED_IMAGE_IDS))
    repo_digest = "registry.invalid/pi@sha256:" + "a" * 64
    image = repo_digest if reference_kind == "repo" else image_id
    metadata = [{
        "Id": image_id,
        "RepoDigests": [repo_digest],
        "Os": "linux",
        "Architecture": "arm64",
        "Config": {"Labels": dict(_REVIEWED_IMAGE_LABELS)},
    }]
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(metadata), stderr=""
        ),
    )
    sandbox = PiDockerSandbox(
        image=image, worktree=tmp_path / "work", config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions", extension_file=extension,
    )
    assert sandbox.inspect()["Id"] == image_id


@pytest.mark.parametrize("reference_kind", ["repo", "id"])
def test_copied_labels_cannot_substitute_an_unreviewed_image_id(
    tmp_path, monkeypatch, reference_kind,
):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    arbitrary_id = "sha256:" + "d" * 64
    repo_digest = "registry.invalid/pi@sha256:" + "e" * 64
    image = repo_digest if reference_kind == "repo" else arbitrary_id
    metadata = [{
        "Id": arbitrary_id,
        "RepoDigests": [repo_digest],
        "Os": "linux",
        "Architecture": "arm64",
        "Config": {"Labels": dict(_REVIEWED_IMAGE_LABELS)},
    }]
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(metadata), stderr="",
        ),
    )
    sandbox = PiDockerSandbox(
        image=image,
        worktree=tmp_path / "work",
        config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    with pytest.raises(PiSandboxError, match="host-side reviewed image allowlist"):
        sandbox.inspect()


@pytest.mark.parametrize(
    ("image_os", "architecture"),
    [("linux", "amd64"), ("windows", "arm64")],
)
def test_reviewed_image_id_with_platform_mismatch_fails_closed(
    tmp_path, monkeypatch, image_os, architecture,
):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    image_id = next(iter(_REVIEWED_IMAGE_IDS))
    metadata = [{
        "Id": image_id,
        "RepoDigests": [],
        "Os": image_os,
        "Architecture": architecture,
        "Config": {"Labels": dict(_REVIEWED_IMAGE_LABELS)},
    }]
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(metadata), stderr="",
        ),
    )
    sandbox = PiDockerSandbox(
        image=image_id,
        worktree=tmp_path / "work",
        config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    with pytest.raises(PiSandboxError, match="linux/arm64"):
        sandbox.inspect()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"Id": 7, "RepoDigests": [], "Os": "linux", "Architecture": "arm64"},
        {"Id": next(iter(_REVIEWED_IMAGE_IDS)), "RepoDigests": "not-a-list"},
        {
            "Id": next(iter(_REVIEWED_IMAGE_IDS)), "RepoDigests": [],
            "Os": None, "Architecture": "arm64", "Config": {"Labels": {}},
        },
        {
            "Id": next(iter(_REVIEWED_IMAGE_IDS)), "RepoDigests": [],
            "Os": "linux", "Architecture": "arm64", "Config": [],
        },
    ],
)
def test_malformed_image_metadata_fails_closed(tmp_path, monkeypatch, metadata):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    image_id = next(iter(_REVIEWED_IMAGE_IDS))
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps([metadata]), stderr="",
        ),
    )
    sandbox = PiDockerSandbox(
        image=image_id,
        worktree=tmp_path / "work",
        config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    with pytest.raises(PiSandboxError, match="malformed|missing"):
        sandbox.inspect()


def test_exact_local_image_id_mismatch_fails_closed(tmp_path, monkeypatch):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "Id": "sha256:" + "1" * 64, "RepoDigests": [],
                "Config": {"Labels": {"org.hermes.pi.version": "0.84.3"}},
            }]),
            stderr="",
        ),
    )
    sandbox = PiDockerSandbox(
        image="sha256:" + "2" * 64, worktree=tmp_path / "work",
        config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    with pytest.raises(PiSandboxError, match="does not match"):
        sandbox.inspect()


def test_image_with_version_only_but_missing_reviewed_artifact_labels_fails(tmp_path, monkeypatch):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    image = next(iter(_REVIEWED_IMAGE_IDS))
    monkeypatch.setattr("agent.transports.pi_docker_sandbox.shutil.which", lambda _name: "/docker")
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps([{
                "Id": image, "RepoDigests": [],
                "Os": "linux", "Architecture": "arm64",
                "Config": {"Labels": {"org.hermes.pi.version": "0.84.3"}},
            }]), stderr="",
        ),
    )
    sandbox = PiDockerSandbox(
        image=image, worktree=tmp_path / "work", config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions", extension_file=extension,
    )
    with pytest.raises(PiSandboxError, match="reviewed-artifact label"):
        sandbox.inspect()


def test_egress_inspection_requires_internal_network_and_exact_proxy_identity(tmp_path, monkeypatch):
    for name in ("work", "config", "sessions"):
        (tmp_path / name).mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    image = "sha256:" + "7" * 64
    sandbox = PiDockerSandbox(
        image=image, worktree=tmp_path / "work", config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions", extension_file=extension,
    )
    proxy_uid, proxy_gid = pi_docker_sandbox._unprivileged_identity()
    network = {
        "Internal": True,
        "Labels": {
            "org.hermes.pi.owner": sandbox.container_name,
            "org.hermes.pi.managed": "true",
            "org.hermes.pi.identity": sandbox.resource_identity,
            "org.hermes.pi.created-at": str(sandbox.created_at),
        },
    }
    proxy = {
        "Config": {
            "Image": image,
            "User": f"{proxy_uid}:{proxy_gid}",
                "Labels": {
                    "org.hermes.pi.owner": sandbox.container_name,
                    "org.hermes.pi.managed": "true",
                    "org.hermes.pi.identity": sandbox.resource_identity,
                    "org.hermes.pi.created-at": str(sandbox.created_at),
                    "org.hermes.pi.egress.allowlist-sha256": sandbox._allowlist_digest,
            },
            "Env": [f"HERMES_PI_PROXY_ALLOWLIST={','.join(sandbox.egress_allowlist)}"],
        },
        "HostConfig": {"ReadonlyRootfs": True, "Binds": None, "Mounts": None},
        "Mounts": [],
        "NetworkSettings": {"Networks": {sandbox.network_name: {}, "bridge": {}}},
    }

    def inspect(args, **_kwargs):
        payload = network if args[:2] == ["network", "inspect"] else proxy
        return SimpleNamespace(returncode=0, stdout=json.dumps([payload]), stderr="")

    monkeypatch.setattr(sandbox, "_run_docker", inspect)
    assert sandbox.inspect_egress()["network"]["Internal"] is True
    network["Internal"] = False
    with pytest.raises(PiSandboxError, match="Internal=true"):
        sandbox.inspect_egress()


def test_verification_is_networkless_and_does_not_inherit_provider_env(tmp_path, monkeypatch):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="registry.invalid/pi@sha256:" + "c" * 64,
        worktree=work, config_dir=config, session_dir=sessions, extension_file=extension,
        provider_env_names={"OPENAI_API_KEY"},
    )
    seen = {}

    def run(argv, **kwargs):
        if len(argv) > 1 and argv[1] == "run":
            seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.transports.pi_docker_sandbox.subprocess.run", run)
    sandbox.run_verification(["pytest", "-q"], timeout=1)
    assert "--network" in seen["argv"] and "none" in seen["argv"]
    assert "OPENAI_API_KEY" not in seen["argv"]
    assert f"src={work.resolve()},dst=/workspace,readonly" in " ".join(seen["argv"])
    joined = " ".join(seen["argv"])
    assert "dst=/pi-config" not in joined
    assert "dst=/pi-sessions" not in joined
    assert "PI_CODING_AGENT_DIR" not in joined
    assert "PI_CODING_AGENT_SESSION_DIR" not in joined
    assert "--entrypoint /opt/hermes/verify-venv/bin/pytest" in joined


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "status"],
        ["npx", "vitest"],
        ["/usr/local/bin/pytest", "-q"],
        ["pytest", "../outside"],
        ["pytest", "--rootdir=/etc"],
        ["python3", "script.py"],
        ["python3", "-c", "print('unsafe')"],
        ["npm", "exec", "vitest"],
        ["npm", "run", "postinstall"],
        ["npm", "test", "--script-shell=./owned-shell"],
    ],
)
def test_verification_argv_rejects_escape_and_arbitrary_execution(argv):
    with pytest.raises(PiSandboxError):
        PiDockerSandbox.validate_verification_argv(argv)


@pytest.mark.parametrize(
    ("argv", "executable"),
    [
        (["pytest", "-q"], "/opt/hermes/verify-venv/bin/pytest"),
        (["python3", "-m", "pytest", "-q"], "/opt/hermes/verify-venv/bin/python"),
        (["ruff", "check", "."], "/opt/hermes/verify-venv/bin/ruff"),
        (["ty", "check"], "/opt/hermes/verify-venv/bin/ty"),
        (["npm", "run", "typecheck"], "/usr/local/bin/npm"),
    ],
)
def test_verification_argv_resolves_only_fixed_image_executables(argv, executable):
    assert PiDockerSandbox.validate_verification_argv(argv) == executable


@pytest.mark.parametrize("name", ["HERMES_GATEWAY_TOKEN", "DISCORD_BOT_TOKEN", "AWS_SESSION_TOKEN"])
def test_unapproved_provider_environment_names_are_rejected(tmp_path, name):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    with pytest.raises(PiSandboxError, match="unapproved"):
        PiDockerSandbox(
            image="sha256:" + "d" * 64,
            worktree=tmp_path / "work", config_dir=tmp_path / "config",
            session_dir=tmp_path / "sessions", extension_file=extension,
            provider_env_names={name},
        )


def test_partial_egress_setup_always_runs_cleanup(tmp_path, monkeypatch):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "f" * 64, worktree=tmp_path / "work",
        config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
        extension_file=extension, egress_allowlist={"allowed.example"},
    )
    monkeypatch.setattr(sandbox, "inspect", lambda **_kwargs: {})
    calls = []

    def run(args, **_kwargs):
        calls.append(args)
        failed = args[:2] == ["network", "connect"]
        return SimpleNamespace(returncode=int(failed), stdout="", stderr="connect failed")

    cleaned = []
    monkeypatch.setattr(sandbox, "_run_docker", run)
    monkeypatch.setattr(sandbox, "cleanup", lambda **_kwargs: cleaned.append(True))
    with pytest.raises(PiSandboxError, match="connect failed"):
        sandbox.prepare_egress()
    assert cleaned == [True]
    create = next(args for args in calls if args[:2] == ["network", "create"])
    proxy = next(args for args in calls if args[:1] == ["run"])
    for argv in (create, proxy):
        joined = " ".join(argv)
        assert "org.hermes.pi.managed=true" in joined
        assert f"org.hermes.pi.identity={sandbox.resource_identity}" in joined
        assert f"org.hermes.pi.created-at={sandbox.created_at}" in joined
    assert proxy[proxy.index("--user") + 1].split(":", 1)[0] != "0"


def test_empty_egress_allowlist_fails_before_docker_access(tmp_path, monkeypatch):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "1" * 64, worktree=tmp_path / "work",
        config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    monkeypatch.setattr(
        sandbox, "_run_docker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Docker must not be touched for an empty allowlist")
        ),
    )
    with pytest.raises(PiSandboxError, match="allowlist is empty"):
        sandbox.prepare_egress()


def test_proxy_inspect_failure_after_setup_always_runs_cleanup(tmp_path, monkeypatch):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "2" * 64, worktree=tmp_path / "work",
        config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
        extension_file=extension, egress_allowlist={"allowed.example"},
    )
    monkeypatch.setattr(sandbox, "inspect", lambda **_kwargs: {})
    monkeypatch.setattr(
        sandbox, "_run_docker",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        sandbox, "inspect_egress",
        lambda **_kwargs: (_ for _ in ()).throw(PiSandboxError("proxy inspect failed")),
    )
    cleaned = []
    monkeypatch.setattr(sandbox, "cleanup", lambda **_kwargs: cleaned.append(True))
    with pytest.raises(PiSandboxError, match="proxy inspect failed"):
        sandbox.prepare_egress()
    assert cleaned == [True]


def test_verification_timeout_always_runs_cleanup(tmp_path, monkeypatch):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "a" * 64, worktree=tmp_path / "work",
        config_dir=tmp_path / "config", session_dir=tmp_path / "sessions",
        extension_file=extension,
    )
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 1))
        ),
    )
    cleaned = []
    monkeypatch.setattr(sandbox, "cleanup", lambda **_kwargs: cleaned.append(True))
    result = sandbox.run_verification(["pytest", "-q"], timeout=1)
    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert cleaned == [True]


def test_stale_sweep_removes_only_old_labeled_inactive_resources(monkeypatch):
    calls = []

    def completed(stdout="", returncode=0):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    def run(argv, **_kwargs):
        args = argv[1:]
        calls.append(args)
        if args[:3] == ["container", "ls", "-a"] and "label=org.hermes.pi.managed=true" not in args:
            return completed("active-owner\n")
        if args[:3] == ["container", "ls", "-a"]:
            return completed(
                "old-proxy\nlive-runtime\nrecent-proxy\nactive-stopped\n"
                "active-proxy\nunknown-role\n"
            )
        if args[:2] == ["container", "inspect"]:
            return completed(json.dumps([
                {
                    "Id": "old-proxy", "State": {"Running": True},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "old",
                        "org.hermes.pi.owner": "missing-owner", "org.hermes.pi.role": "proxy",
                        "org.hermes.pi.created-at": "1",
                    }},
                },
                {
                    "Id": "live-runtime", "State": {"Running": True},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "live",
                        "org.hermes.pi.owner": "active-owner", "org.hermes.pi.role": "runtime",
                        "org.hermes.pi.created-at": "1",
                    }},
                },
                {
                    "Id": "recent-proxy", "State": {"Running": False},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "recent",
                        "org.hermes.pi.owner": "missing-owner", "org.hermes.pi.role": "proxy",
                        "org.hermes.pi.created-at": "9950",
                    }},
                },
                {
                    "Id": "active-stopped", "State": {"Running": False},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "active",
                        "org.hermes.pi.owner": "missing-owner", "org.hermes.pi.role": "runtime",
                        "org.hermes.pi.created-at": "1",
                    }},
                },
                {
                    "Id": "active-proxy", "State": {"Running": True},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "cross-process",
                        "org.hermes.pi.owner": "active-owner", "org.hermes.pi.role": "proxy",
                        "org.hermes.pi.created-at": "1",
                    }},
                },
                {
                    "Id": "unknown-role", "State": {"Running": False},
                    "Config": {"Labels": {
                        "org.hermes.pi.managed": "true", "org.hermes.pi.identity": "future",
                        "org.hermes.pi.owner": "missing-owner", "org.hermes.pi.role": "future-role",
                        "org.hermes.pi.created-at": "1",
                    }},
                },
            ]))
        if args[:2] == ["network", "ls"]:
            return completed("old-empty\nold-busy\nactive-network\nunknown-network\n")
        if args[:2] == ["network", "inspect"]:
            network_id = args[2]
            return completed(json.dumps([{
                "Id": network_id,
                "Labels": {
                    "org.hermes.pi.managed": "true",
                    "org.hermes.pi.identity": (
                        "active" if network_id == "active-network" else network_id
                    ),
                    "org.hermes.pi.owner": "missing-owner",
                    "org.hermes.pi.role": (
                        "future-network" if network_id == "unknown-network" else "network"
                    ),
                    "org.hermes.pi.created-at": "1",
                },
                "Containers": {} if network_id == "old-empty" else {"member": {}},
            }]))
        return completed()

    monkeypatch.setattr("agent.transports.pi_docker_sandbox.subprocess.run", run)
    result = PiDockerSandbox.sweep_stale_resources(
        now=10_000, minimum_age_seconds=100, active_identities={"active"},
    )
    assert result == {"containers": 1, "networks": 1}
    removals = [
        args for args in calls
        if args and (args[0] == "rm" or args[:2] == ["network", "rm"])
    ]
    assert ["rm", "--force", "old-proxy"] in removals
    assert ["network", "rm", "old-empty"] in removals
    assert all(
        "live-runtime" not in args
        and "recent-proxy" not in args
        and "active-stopped" not in args
        and "active-proxy" not in args
        and "unknown-role" not in args
        and "unknown-network" not in args
        and "active-network" not in args
        for args in removals
    )


def test_stale_sweep_aborts_when_owner_census_fails(monkeypatch):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv[1:])
        return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")

    monkeypatch.setattr("agent.transports.pi_docker_sandbox.subprocess.run", run)
    assert PiDockerSandbox.sweep_stale_resources(now=10_000) == {
        "containers": 0,
        "networks": 0,
    }
    assert calls == [["container", "ls", "-a", "--format", "{{.Names}}"]]


def test_live_sandbox_identity_registry_tracks_cleanup(tmp_path, monkeypatch):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="sha256:" + "a" * 64,
        worktree=tmp_path / "work", config_dir=tmp_path / "config",
        session_dir=tmp_path / "sessions", extension_file=extension,
        resource_identity="active-test",
    )
    sandbox.mark_active()
    assert sandbox.resource_identity in PiDockerSandbox.active_resource_identities()
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    sandbox.cleanup()
    assert sandbox.resource_identity not in PiDockerSandbox.active_resource_identities()


@pytest.mark.parametrize("host", ["*", "https://api.example.com", "10.0.0.1", "example.com:99999"])
def test_broad_or_non_hostname_egress_rules_are_rejected(tmp_path, host):
    for directory in (tmp_path / "work", tmp_path / "config", tmp_path / "sessions"):
        directory.mkdir()
    extension = tmp_path / "config" / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    with pytest.raises(PiSandboxError, match="exact DNS hostnames"):
        PiDockerSandbox(
            image="sha256:" + "d" * 64,
            worktree=tmp_path / "work", config_dir=tmp_path / "config",
            session_dir=tmp_path / "sessions", extension_file=extension,
            egress_allowlist={host},
        )


def test_docker_cleanup_targets_only_managed_container_name(tmp_path, monkeypatch):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    extension = config / "hermes_bridge.ts"
    extension.write_text("x", encoding="utf-8")
    sandbox = PiDockerSandbox(
        image="registry.invalid/pi@sha256:" + "d" * 64,
        worktree=work, config_dir=config, session_dir=sessions,
        extension_file=extension, container_name="hermes-pi-test",
    )
    calls = []
    monkeypatch.setattr(
        "agent.transports.pi_docker_sandbox.subprocess.run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    sandbox.cleanup()
    assert calls == [
        ["docker", "rm", "--force", "hermes-pi-test"],
        ["docker", "rm", "--force", sandbox.proxy_name],
        ["docker", "network", "rm", sandbox.network_name],
    ]
