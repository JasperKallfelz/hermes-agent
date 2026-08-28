import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.external_runtime.binding import (
    RuntimeBinding,
    load_binding,
    save_binding,
    worktree_identity,
)
from agent.external_runtime.types import ExternalTurnResult, TurnStatus
from agent.transports.pi_rpc_client import (
    PINNED_PI_VERSION,
    PiProtocolError,
    PiRequestUncertainError,
    PiRpcError,
)
from agent.transports.pi_rpc_session import PiReconciliationRequired, PiRpcSession


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        extra = kwargs.get("extra_args") or []
        session_dir = Path(extra[extra.index("--session-dir") + 1])
        session_id = extra[extra.index("--session-id") + 1] if "--session-id" in extra else "s"
        self.session_id = session_id
        self.session_file = session_dir / f"{session_id}.jsonl"
        self.session_file.write_text("{}\n", encoding="utf-8")
        self.events = iter([
            {"type": "agent_start"},
            {"type": "message_start"},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "x"}},
            {"type": "agent_end"},
            {"type": "message_end", "message": {"content": "answer"}},
            {"type": "agent_settled"},
        ])
    def start(self): pass
    def is_alive(self): return True
    def drain_events(self): return []
    def request(self, kind, **kwargs):
        if kind == "get_state":
            return {"data": {"sessionId": self.session_id, "sessionFile": str(self.session_file)}}
        if kind == "get_entries": return {"data": {"leafId": "e", "entries": []}}
        return {"id": "accepted", "success": True}
    def next_event(self, timeout=1): return next(self.events)
    def close(self, **kwargs): pass


def _bound_session(tmp_path, **updates: Any):
    sessions = tmp_path / "sessions"
    bindings = tmp_path / "bindings"
    sessions.mkdir(exist_ok=True)
    bindings.mkdir(exist_ok=True)
    native = sessions / "native.jsonl"
    native.write_text("{}\n", encoding="utf-8")
    values: dict[str, Any] = {
        "runtime": "pi_rpc",
        "native_session_file": str(native),
        "native_session_id": "resume-me",
        "last_entry_id": "e",
        "cwd": str(tmp_path.resolve()),
        "runtime_version": PINNED_PI_VERSION,
        "worktree": worktree_identity(tmp_path),
        "prompt_state": "settled",
    }
    values.update(updates)
    binding_path = bindings / "binding.json"
    save_binding(binding_path, RuntimeBinding(**values))
    return sessions, native, binding_path


def test_agent_end_does_not_complete_before_agent_settled(tmp_path):
    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=FakeClient)
    result = session.run_turn("go")
    assert result.final_text == "answer"
    assert result.cursor == "e"
    assert session.binding is not None
    assert session.binding.prompt_state == "native_completed"
    session.finalize_projection(persisted=True, turn=result)
    assert session.binding.prompt_state == "settled"
    assert session.binding_path.is_file()


class UncertainClient(FakeClient):
    def request(self, kind, **kwargs):
        if kind == "prompt":
            raise PiRequestUncertainError("pipe died after flush")
        return super().request(kind, **kwargs)


def test_uncertain_prompt_is_never_marked_replayable(tmp_path):
    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=UncertainClient)
    result = session.run_turn("do not duplicate")
    assert result.status is TurnStatus.RECONCILIATION_REQUIRED
    assert result.reconciliation_required is True
    assert session.binding is not None
    assert session.binding.prompt_state == "reconciliation_required"
    with pytest.raises(PiReconciliationRequired, match="may have been accepted"):
        PiRpcSession(
            executable="pi",
            cwd=str(tmp_path),
            session_dir=session.session_dir,
            binding_path=session.binding_path,
            client_factory=FakeClient,
        ).ensure_started()


def test_missing_fresh_native_state_preserves_checkpoint_and_request_metadata(tmp_path):
    sessions, native, binding_path = _bound_session(
        tmp_path,
        prompt_state="native_completed",
        last_entry_id="checkpoint-before",
        model={"provider": "reviewed", "id": "model"},
        thinking_level="high",
        prompt_request_id="prompt-7",
        prompt_digest="digest-7",
        turn_generation=7,
    )
    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=FakeClient,
    )
    turn = ExternalTurnResult(
        status=TurnStatus.COMPLETED,
        session_id="resume-me",
        session_file=str(native),
        cursor="native-after",
        prompt_accepted=True,
    )

    with pytest.raises(PiReconciliationRequired, match="fresh native completion state"):
        session.finalize_projection(persisted=True, turn=turn)

    rebound = load_binding(binding_path)
    assert rebound.prompt_state == "reconciliation_required"
    assert rebound.last_entry_id == "checkpoint-before"
    assert rebound.model == {"provider": "reviewed", "id": "model"}
    assert rebound.thinking_level == "high"
    assert rebound.prompt_request_id == "prompt-7"
    assert rebound.prompt_digest == "digest-7"
    assert rebound.turn_generation == 7


def test_advanced_cursor_repeated_inspect_and_adopt_refusal_then_archival_abandon(
    tmp_path,
):
    sessions, native, binding_path = _bound_session(
        tmp_path,
        last_entry_id="before",
        prompt_state="reconciliation_required",
        prompt_request_id="prompt-1",
        prompt_digest="digest",
    )
    calls = []

    class ReconcileClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter([])

        def request(self, kind, **kwargs):
            calls.append(kind)
            if kind == "get_state":
                return {"data": {
                    "sessionId": "resume-me", "sessionFile": str(native),
                    "model": {"provider": "openai", "id": "gpt-reviewed"},
                    "thinkingLevel": "high",
                }}
            if kind == "get_entries":
                return {"data": {"leafId": "after", "entries": [
                    {"id": "before", "type": "message", "role": "user", "content": "secret"},
                    {"id": "after", "type": "message", "role": "assistant", "content": "answer"},
                ]}}
            raise AssertionError(f"reconciliation sent forbidden RPC command: {kind}")

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=ReconcileClient,
    )
    inspected = session.reconcile("inspect")
    assert inspected["cursor"] == "after"
    assert inspected["cursor_advanced"] is True
    assert inspected["adopt_available"] is False
    assert inspected["entry_summary"]["count"] == 2
    assert "content" not in inspected["entry_summary"]["tail"][0]
    assert session.binding is not None
    assert session.binding.prompt_state == "reconciliation_required"

    unchanged = binding_path.read_bytes()
    for _attempt in range(2):
        again = session.reconcile("inspect")
        assert again["cursor_advanced"] is True
        with pytest.raises(PiReconciliationRequired, match="adopt is unavailable"):
            session.reconcile("adopt")
        assert binding_path.read_bytes() == unchanged
        assert session.binding is not None
        assert session.binding.prompt_state == "reconciliation_required"
        assert session.binding.last_entry_id == "before"

    report = session.reconcile("abandon")
    archive = binding_path.with_name(report["archived_binding"])
    assert archive.read_bytes() == unchanged
    assert report["context_loss"] is True
    assert not binding_path.exists()
    assert "prompt" not in calls


def test_first_cursor_advance_requires_archival_abandon_and_restart_allocates_fresh(
    tmp_path,
):
    sessions, native, binding_path = _bound_session(
        tmp_path,
        last_entry_id=None,
        prompt_state="reconciliation_required",
        prompt_request_id="prompt-first",
        prompt_digest="digest-first",
    )
    calls = []

    class FirstAdvanceClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter(())

        def request(self, kind, **_kwargs):
            calls.append(kind)
            if kind == "get_state":
                return {"data": {
                    "sessionId": "resume-me",
                    "sessionFile": str(native),
                }}
            if kind == "get_entries":
                return {"data": {
                    "leafId": "first-native-entry",
                    "entries": [{"id": "first-native-entry"}],
                }}
            raise AssertionError(kind)

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=FirstAdvanceClient,
    )
    old_id = session.native_session_id
    inspected = session.reconcile("inspect")
    assert inspected["previous_cursor"] is None
    assert inspected["cursor_advanced"] is True
    with pytest.raises(PiReconciliationRequired, match="adopt is unavailable"):
        session.reconcile("adopt")
    report = session.reconcile("abandon")
    assert report["replayed"] is False
    assert not binding_path.exists()
    assert "prompt" not in calls

    restarted = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=FakeClient,
    )
    result = restarted.run_turn("fresh after archival context loss")
    restarted.finalize_projection(persisted=True, turn=result)
    assert result.status is TurnStatus.COMPLETED
    assert result.session_id != old_id
    assert restarted.binding is not None
    assert restarted.binding.prompt_state == "settled"


def test_uncertain_prompt_abandon_archives_durable_context_and_cleans_up(tmp_path):
    sessions, native, binding_path = _bound_session(
        tmp_path, prompt_state="accepted", prompt_request_id="prompt-1", prompt_digest="digest"
    )
    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=lambda **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("abandon must not launch Pi")),
    )
    cleanup = []

    class OwnedClient:
        def close(self, *, force=False):
            cleanup.append(("client", force))

    session.client = OwnedClient()
    session.sandbox = SimpleNamespace(cleanup=lambda: cleanup.append(("sandbox", True)))
    original = binding_path.read_bytes()
    report = session.reconcile("abandon")
    archive = binding_path.with_name(report["archived_binding"])
    assert archive.read_bytes() == original
    assert report["archived_binding_sha256"]
    assert report["context_loss"] is True
    assert report["requires_independent_approval"] is True
    assert "CONTEXT LOSS" in report["warning"]
    assert report["replayed"] is False
    assert report["runtime_ownership_retired"] is True
    assert cleanup == [("client", True), ("sandbox", True)]
    assert not binding_path.exists()
    assert session.binding is None


def test_reconciliation_revalidates_version_before_launch(tmp_path):
    sessions, _native, binding_path = _bound_session(
        tmp_path, runtime_version="0.84.2", prompt_state="reconciliation_required"
    )

    def forbidden_launch(**_kwargs):
        raise AssertionError("invalid identity must fail before native launch")

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=forbidden_launch,
    )
    with pytest.raises(PiReconciliationRequired, match="version"):
        session.reconcile("inspect")


def test_resume_uses_exact_persisted_session_file(tmp_path):
    sessions, native, binding_path = _bound_session(tmp_path)
    captured = {}

    class ResumeClient(FakeClient):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            extra = kwargs["extra_args"]
            assert extra[extra.index("--session") + 1] == str(native)
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter([])

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=ResumeClient,
    )
    state = session.ensure_started()
    assert state.session_id == "resume-me"
    assert "--no-session" not in captured["extra_args"]


@pytest.mark.parametrize(
    ("binding_updates", "state_updates", "message"),
    [
        ({"last_entry_id": "old"}, {}, "cursor changed"),
        ({"model": {"provider": "p", "id": "old"}}, {"model": {"provider": "p", "id": "new"}}, "model changed"),
        ({"thinking_level": "high"}, {"thinkingLevel": "low"}, "thinking level changed"),
    ],
)
def test_resume_reconciles_all_durable_state_fields(
    tmp_path, binding_updates, state_updates, message
):
    sessions, native, binding_path = _bound_session(tmp_path, **binding_updates)

    class ChangedStateClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter([])

        def request(self, kind, **kwargs):
            if kind == "get_state":
                data = {"sessionId": self.session_id, "sessionFile": str(self.session_file)}
                data.update(state_updates)
                return {"data": data}
            if kind == "get_entries":
                return {"data": {"leafId": "e", "entries": []}}
            return {"id": "accepted", "success": True}

    with pytest.raises(PiReconciliationRequired, match=message):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=ChangedStateClient,
        ).ensure_started()


def test_changed_cwd_and_runtime_version_fail_before_launch(tmp_path):
    sessions, _native, binding_path = _bound_session(
        tmp_path, runtime_version="0.84.2"
    )
    with pytest.raises(PiReconciliationRequired, match="version"):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=FakeClient,
        ).ensure_started()

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PiReconciliationRequired, match="cwd"):
        PiRpcSession(
            executable="pi", cwd=str(other), session_dir=sessions,
            binding_path=binding_path, client_factory=FakeClient,
        ).ensure_started()


def test_foreign_native_session_id_fails_closed(tmp_path):
    sessions, native, binding_path = _bound_session(tmp_path)

    class ForeignClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "foreign"
            self.session_file = native
            self.events = iter([])

    with pytest.raises(PiProtocolError, match="foreign"):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=ForeignClient,
        ).ensure_started()


def test_missing_bound_session_file_fails_closed(tmp_path):
    sessions = tmp_path / "sessions"
    bindings = tmp_path / "bindings"
    sessions.mkdir()
    bindings.mkdir()
    binding_path = bindings / "binding.json"
    save_binding(
        binding_path,
        RuntimeBinding(
            runtime="pi_rpc",
            native_session_file=str(sessions / "gone.jsonl"),
            native_session_id="gone",
            last_entry_id=None,
            cwd=str(tmp_path.resolve()),
            runtime_version=PINNED_PI_VERSION,
            worktree=worktree_identity(tmp_path),
            prompt_state="settled",
        ),
    )
    with pytest.raises(PiReconciliationRequired, match="missing"):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=FakeClient,
        ).ensure_started()


def test_contained_session_rejects_binding_inside_any_container_mount(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    sandbox = type("Sandbox", (), {
        "worktree": work.resolve(),
        "config_dir": config.resolve(),
        "session_dir": sessions.resolve(),
    })()
    for binding_path in (
        work / "binding.json", config / "binding.json", sessions / "binding.json",
    ):
        with pytest.raises(PiRpcError, match="outside every container mount"):
            PiRpcSession(
                executable=None, cwd=str(work), config_dir=config,
                session_dir=sessions, binding_path=binding_path,
                sandbox=sandbox,
            )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("inspect", "read,edit,write"),
        ("implement", "read,edit,write"),
    ],
)
def test_model_tool_surface_never_contains_a_shell(tmp_path, mode, expected):
    sandbox = None
    binding_path = None
    if mode == "implement":
        work = tmp_path / "work"
        config = tmp_path / "config"
        sessions = tmp_path / "sessions"
        bindings = tmp_path / "bindings"
        for directory in (work, config, sessions, bindings):
            directory.mkdir()
        (config / "hermes_bridge.ts").write_text("stub", encoding="utf-8")
        sandbox = SimpleNamespace(
            worktree=work.resolve(), config_dir=config.resolve(),
            session_dir=sessions.resolve(),
        )
        cwd = work
        config_dir = config
        session_dir = sessions
        binding_path = bindings / "binding.json"
    else:
        cwd = tmp_path
        config_dir = None
        session_dir = None
    session = PiRpcSession(
        executable="pi", cwd=str(cwd), config_dir=config_dir,
        session_dir=session_dir, binding_path=binding_path,
        mode=mode, sandbox=sandbox,
    )
    args = session._pi_args()
    tools = args[args.index("--tools") + 1]
    assert tools == expected
    assert "bash" not in tools and "powershell" not in tools


def test_events_before_prompt_acceptance_cannot_settle_new_generation(tmp_path):
    class SequencedClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sequenced = iter([
                ({"type": "agent_start"}, 1),
                ({"type": "agent_settled"}, 2),
                ({"type": "agent_start"}, 4),
                ({"type": "message_end", "message": {"content": "new"}}, 5),
                ({"type": "agent_settled"}, 6),
            ])

        def request(self, kind, **kwargs):
            if kind == "prompt":
                return {"type": "response", "success": True, "_hermes_sequence": 3}
            return super().request(kind, **kwargs)

        def next_event_with_sequence(self, timeout=1):
            return next(self.sequenced)

    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=SequencedClient)
    result = session.run_turn("new generation")
    assert result.final_text == "new"
    assert result.status is TurnStatus.COMPLETED


def test_steer_is_generation_safe_and_follow_up_is_disabled(tmp_path):
    calls = []

    class ControlClient(FakeClient):
        def request(self, kind, **kwargs):
            calls.append((kind, kwargs))
            return super().request(kind, **kwargs)

    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=ControlClient)
    assert session.steer("idle redirect") is False
    session.ensure_started()
    session.state = session.state.RUNNING
    session._active_generation = 7
    assert session.steer("redirect") is True
    assert session.follow_up("then verify") is False
    assert session.compact("keep decisions").completed is True
    assert [kind for kind, _kwargs in calls if kind in {"steer", "follow_up", "compact"}] == [
        "steer", "compact",
    ]


def test_interrupt_sends_abort_then_waits_for_agent_settled(tmp_path):
    calls = []
    session = None

    class AbortClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.events = iter([
                {"type": "agent_start"},
                {"type": "message_end", "message": {"content": "stopped"}},
                {"type": "agent_settled"},
            ])
            self.first_event = True

        def request(self, kind, **kwargs):
            calls.append(kind)
            return super().request(kind, **kwargs)

        def next_event(self, timeout=1):
            nonlocal session
            event = next(self.events)
            if self.first_event:
                self.first_event = False
                assert session is not None
                session.request_interrupt()
            return event

    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=AbortClient)
    result = session.run_turn("long task")
    assert result.status is TurnStatus.INTERRUPTED
    assert result.final_text == "stopped"
    assert "abort" in calls
    assert session.binding is not None
    assert session.binding.prompt_state == "native_completed"
    session.finalize_projection(persisted=True, turn=result)
    assert session.binding.prompt_state == "settled"


def test_extension_ui_without_handler_is_denied(tmp_path):
    responses = []

    class UiClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.events = iter([
                {"type": "agent_start"},
                {"type": "extension_ui_request", "method": "confirm", "id": "ui-1"},
                {"type": "agent_settled"},
            ])

        def send_extension_ui_response(self, request_id, **response):
            responses.append((request_id, response))

    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=UiClient)
    assert session.run_turn("go").status is TurnStatus.COMPLETED
    assert responses == [("ui-1", {"confirmed": False})]


def test_event_silence_after_acceptance_requires_reconciliation(tmp_path):
    class SilentClient(FakeClient):
        def next_event(self, timeout=1):
            raise queue.Empty

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), client_factory=SilentClient,
        silence_timeout=0.05, turn_timeout=0.2,
    )
    result = session.run_turn("accepted once")
    assert result.status is TurnStatus.RECONCILIATION_REQUIRED
    assert result.reconciliation_required is True
    assert session.binding is not None
    assert session.binding.prompt_state == "reconciliation_required"


def test_arbitrary_client_start_failure_is_normalized_to_error(tmp_path):
    class StartFailure(FakeClient):
        def start(self):
            raise RuntimeError("start exploded")

    result = PiRpcSession(
        executable="pi", cwd=str(tmp_path), client_factory=StartFailure,
    ).run_turn("never sent")
    assert result.status is TurnStatus.ERROR
    assert result.should_retire is True
    assert "start exploded" in (result.error or "")


def test_arbitrary_egress_prepare_failure_is_normalized_to_error(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    bindings = tmp_path / "bindings"
    for directory in (work, config, sessions, bindings):
        directory.mkdir()
    (config / "hermes_bridge.ts").write_text("stub", encoding="utf-8")
    sandbox = SimpleNamespace(
        worktree=work.resolve(), config_dir=config.resolve(), session_dir=sessions.resolve(),
        prepare_egress=lambda: (_ for _ in ()).throw(RuntimeError("egress exploded")),
        cleanup=lambda **_kwargs: None,
    )
    result = PiRpcSession(
        executable=None, cwd=str(work), config_dir=config, session_dir=sessions,
        binding_path=bindings / "binding.json", sandbox=sandbox,
    ).run_turn("never sent")
    assert result.status is TurnStatus.ERROR
    assert result.reconciliation_required is False
    assert "egress exploded" in (result.error or "")


@pytest.mark.parametrize(
    ("failing_write", "expected"),
    [(1, TurnStatus.ERROR), (2, TurnStatus.RECONCILIATION_REQUIRED),
     (3, TurnStatus.RECONCILIATION_REQUIRED), (4, TurnStatus.COMPLETED)],
)
def test_binding_write_failures_respect_prompt_acceptance_boundary(
    tmp_path, monkeypatch, failing_write, expected
):
    import agent.transports.pi_rpc_session as session_module

    calls = []
    real_save = save_binding

    def phase_failure(path, binding):
        calls.append(binding.prompt_state)
        if len(calls) == failing_write:
            raise OSError(f"binding write {failing_write} failed")
        return real_save(path, binding)

    class RecordingClient(FakeClient):
        prompt_calls = 0

        def request(self, kind, **kwargs):
            if kind == "prompt":
                self.__class__.prompt_calls += 1
            return super().request(kind, **kwargs)

    monkeypatch.setattr(session_module, "save_binding", phase_failure)
    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), client_factory=RecordingClient,
    )
    result = session.run_turn("exactly once")
    assert result.status is expected
    if failing_write == 4:
        with pytest.raises(OSError, match="binding write 4 failed"):
            session.finalize_projection(persisted=True, turn=result)
        assert session.binding is not None
        assert session.binding.prompt_state == "reconciliation_required"
    assert RecordingClient.prompt_calls == (0 if failing_write == 1 else 1)
    if expected is TurnStatus.RECONCILIATION_REQUIRED:
        assert result.reconciliation_required is True
        assert session.binding is not None
        assert session.binding.prompt_state in {"sending", "reconciliation_required"}


@pytest.mark.parametrize(
    "state_update",
    [
        {"isStreaming": True},
        {"isCompacting": True},
        {"pendingMessageCount": 1},
    ],
)
def test_adopt_rejects_busy_native_state(tmp_path, state_update):
    sessions, native, binding_path = _bound_session(
        tmp_path, last_entry_id="e", prompt_state="accepted",
    )

    class BusyClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter([])

        def request(self, kind, **kwargs):
            if kind == "get_state":
                data = {"sessionId": "resume-me", "sessionFile": str(native)}
                data.update(state_update)
                return {"data": data}
            if kind == "get_entries":
                return {"data": {"leafId": "e", "entries": [{"id": "e"}]}}
            raise AssertionError(kind)

    with pytest.raises(PiReconciliationRequired, match="still streaming"):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=BusyClient,
        ).reconcile("adopt")


def test_adopt_rejects_cursor_without_previous_lineage(tmp_path):
    sessions, native, binding_path = _bound_session(
        tmp_path, last_entry_id="previous", prompt_state="accepted",
    )

    class ForkedClient(FakeClient):
        def __init__(self, *args, **kwargs):
            self.session_id = "resume-me"
            self.session_file = native
            self.events = iter([])

        def request(self, kind, **kwargs):
            if kind == "get_state":
                return {"data": {"sessionId": "resume-me", "sessionFile": str(native)}}
            if kind == "get_entries":
                return {"data": {"leafId": "fork", "entries": [{"id": "fork"}]}}
            raise AssertionError(kind)

    with pytest.raises(PiReconciliationRequired, match="previously persisted cursor"):
        PiRpcSession(
            executable="pi", cwd=str(tmp_path), session_dir=sessions,
            binding_path=binding_path, client_factory=ForkedClient,
        ).reconcile("adopt")


def test_abandon_succeeds_without_native_file_or_available_runtime(tmp_path):
    sessions, native, binding_path = _bound_session(
        tmp_path, prompt_state="accepted", last_entry_id=None,
    )
    native.unlink()
    work = tmp_path / "work"
    config = tmp_path / "config"
    work.mkdir()
    config.mkdir()
    sandbox = SimpleNamespace(
        worktree=work.resolve(), config_dir=config.resolve(), session_dir=sessions.resolve(),
        cleanup=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Docker unavailable")
        ),
    )
    session = PiRpcSession(
        executable=None, cwd=str(tmp_path), config_dir=config, session_dir=sessions,
        binding_path=binding_path, sandbox=sandbox,
        client_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("abandon must not launch Pi")
        ),
    )
    report = session.reconcile("abandon")
    assert report["slot_bound"] is False
    assert not binding_path.exists()


def test_next_run_after_abandon_allocates_and_binds_one_fresh_session(tmp_path):
    sessions, _native, binding_path = _bound_session(
        tmp_path, prompt_state="accepted", last_entry_id="durable-before",
    )
    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), session_dir=sessions,
        binding_path=binding_path, client_factory=FakeClient,
    )
    old_id = session.native_session_id
    report = session.reconcile("abandon")
    assert report["slot_bound"] is False
    result = session.run_turn("fresh turn")
    session.finalize_projection(persisted=True, turn=result)
    assert result.status is TurnStatus.COMPLETED
    assert result.session_id and result.session_id != old_id
    assert session.binding is not None
    assert session.binding.native_session_id == result.session_id
    assert binding_path.is_file()


def test_steer_generation_ownership_is_atomic_against_settlement(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    settled = threading.Event()

    class RacingClient(FakeClient):
        def request(self, kind, **kwargs):
            if kind == "steer":
                entered.set()
                assert release.wait(2)
            return super().request(kind, **kwargs)

    session = PiRpcSession(executable="pi", cwd=str(tmp_path), client_factory=RacingClient)
    session.ensure_started()
    session.state = session.state.RUNNING
    session._active_generation = 9
    result = []
    steer_thread = threading.Thread(
        target=lambda: result.append(session.steer("owned"))
    )

    def settle():
        with session._lock:
            session.state = session.state.IDLE
            session._active_generation = None
        settled.set()

    steer_thread.start()
    assert entered.wait(2)
    settle_thread = threading.Thread(target=settle)
    settle_thread.start()
    assert not settled.wait(0.05)
    release.set()
    steer_thread.join(2)
    settle_thread.join(2)
    assert result == [True]
    assert settled.is_set()


def test_uncertain_steer_never_requests_duplicate_fallback(tmp_path):
    class UncertainSteerClient(FakeClient):
        def request(self, kind, **kwargs):
            if kind == "steer":
                raise PiRequestUncertainError("flush uncertain")
            return super().request(kind, **kwargs)

    session = PiRpcSession(
        executable="pi", cwd=str(tmp_path), client_factory=UncertainSteerClient,
    )
    session.ensure_started()
    session.state = session.state.RUNNING
    session._active_generation = 10
    assert session.steer("possibly accepted") is True
