"""Conversation-loop integration for the runtime-neutral Pi branch."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from agent.external_runtime.binding import RuntimeBinding, save_binding, worktree_identity
from agent.external_runtime.finalize import finalize_external_turn
from agent.external_runtime.types import ExternalTurnResult, TurnStatus
from agent.pi_runtime import make_pi_approval_handler, make_pi_event_bridge
from agent.transports.pi_rpc_client import PINNED_PI_VERSION
from agent.transports.pi_rpc_session import PiRpcSession


class _PiSession:
    def __init__(self):
        self.binding = SimpleNamespace(prompt_state="settled")

    def run_turn(self, user_input):
        self.binding.prompt_state = "native_completed"
        return ExternalTurnResult(
            final_text=f"pi: {user_input}",
            projected_messages=[{"role": "assistant", "content": f"pi: {user_input}"}],
            session_id="pi-session",
            session_file="/managed/pi-session.jsonl",
            cursor="leaf-2",
            run_id="pi-session:1",
            tool_iterations=2,
            prompt_tokens=11,
            output_tokens=7,
            cache_read_tokens=3,
            cost=0.25,
        )

    def finalize_projection(self, *, persisted, turn):
        assert persisted is True
        assert turn.prompt_accepted is True
        assert self.binding.prompt_state == "native_completed"
        self.binding.prompt_state = "settled"

    def close(self):
        pass


def _agent():
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="pi_rpc",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _install_session(agent, session):
    agent._pi_rpc_session = session
    return session


def test_pi_turn_uses_neutral_branch_and_persists_projection(monkeypatch):
    session = _PiSession()
    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session",
        lambda agent: _install_session(agent, session),
    )
    agent = _agent()
    agent._iters_since_skill = 0
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("inspect this")

    assert result["final_response"] == "pi: inspect this"
    assert result["completed"] is True
    assert result["external_runtime"] == {
        "runtime": "pi_rpc",
        "session_id": "pi-session",
        "session_file": "/managed/pi-session.jsonl",
        "cursor": "leaf-2",
        "run_id": "pi-session:1",
        "status": "completed",
        "reconciliation_required": False,
        "phase": "after_send",
        "prompt_accepted": True,
        "projection_persisted": True,
    }
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "pi: inspect this"
    assert isinstance(result["messages"][-1]["timestamp"], float)
    assert agent._iters_since_skill == 2


def test_pi_turn_never_enters_hermes_model_client(monkeypatch):
    session = _PiSession()
    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session",
        lambda agent: _install_session(agent, session),
    )
    agent = _agent()
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = AssertionError("Hermes loop entered")
    with patch.object(agent, "_spawn_background_review", return_value=None):
        assert agent.run_conversation("hello")["completed"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_two_pi_turns_persist_exact_user_assistant_alternation(monkeypatch):
    session = _PiSession()
    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session",
        lambda agent: _install_session(agent, session),
    )
    agent = _agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        first = agent.run_conversation("first")
        second = agent.run_conversation(
            "second", conversation_history=first["messages"],
        )
    conversational = [
        message for message in second["messages"]
        if message.get("role") in {"user", "assistant"}
    ]
    assert [message["role"] for message in conversational[-4:]] == [
        "user", "assistant", "user", "assistant",
    ]
    assert conversational[-4]["content"] == "first"
    assert conversational[-3]["content"] == "pi: first"
    assert conversational[-2]["content"] == "second"
    assert conversational[-1]["content"] == "pi: second"


def test_pi_approval_mapping_is_fail_closed_without_interactive_callback(monkeypatch):
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None)
    handler = make_pi_approval_handler(object())
    assert handler({"method": "confirm", "message": "rm -rf x"}) == {"confirmed": False}
    assert handler({"method": "input"}) == {"cancelled": True}


def test_pi_confirm_uses_hermes_dangerous_action_callback(monkeypatch):
    calls = []

    def callback(command, description, **kwargs):
        calls.append((command, description, kwargs))
        return "once"

    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: callback)
    handler = make_pi_approval_handler(object())
    result = handler({
        "method": "confirm", "title": "Pi approval", "message": "dangerous command", "timeout": 5000
    })
    assert result == {"confirmed": True}
    assert calls[0][2]["allow_permanent"] is False


def test_pi_dialogs_use_interactive_clarification_and_validate_selection():
    calls = []
    agent = SimpleNamespace(
        clarify_callback=lambda question, choices, multi_select=False: (
            calls.append((question, choices, multi_select)) or (choices[0] if choices else "typed")
        )
    )
    handler = make_pi_approval_handler(agent)
    assert handler({"method": "select", "title": "Pick", "options": ["A", "B"]}) == {
        "value": "A"
    }
    assert handler({"method": "input", "title": "Name", "placeholder": "value"}) == {
        "value": "typed"
    }
    assert calls == [("Pick", ["A", "B"], False), ("Name\nvalue", None, False)]


def test_pi_fire_and_forget_ui_events_are_bounded_and_rate_limited():
    delivered = []
    agent = SimpleNamespace(event_callback=lambda name, payload: delivered.append((name, payload)))
    bridge = make_pi_event_bridge(agent)

    for index in range(25):
        bridge({
            "type": "extension_ui_request",
            "method": "setStatus",
            "id": str(index),
            "message": "x" * 10_000,
        })

    assert len(delivered) == 20
    assert all(name == "runtime:pi:extension_ui_request" for name, _payload in delivered)
    assert len(delivered[0][1]["event"]["message"]) == 4000


def test_public_pi_path_normalizes_acquisition_failure_before_send(monkeypatch):
    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session",
        lambda _agent: (_ for _ in ()).throw(RuntimeError("image inspection exploded")),
    )
    result = _agent().run_conversation("must remain replayable")
    assert result["completed"] is False
    assert result["external_runtime"]["status"] == "error"
    assert result["external_runtime"]["phase"] == "before_send"
    assert result["external_runtime"]["prompt_accepted"] is False
    assert "image inspection exploded" in result["error"]


def test_dispatcher_fence_marks_untyped_adapter_exception_uncertain(monkeypatch):
    monkeypatch.setattr(
        "agent.pi_runtime.run_pi_rpc_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatcher exploded")),
    )
    result = _agent().run_conversation("acceptance is unknown")
    assert result["external_runtime"]["status"] == "reconciliation_required"
    assert result["external_runtime"]["phase"] == "unknown_after_send"
    assert result["external_runtime"]["prompt_accepted"] is None
    assert result["external_runtime"]["reconciliation_required"] is True
    assert "dispatcher exploded" in result["error"]


def _finalizer_agent(session):
    return SimpleNamespace(
        _session_db=None,
        _pi_rpc_session=session,
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _interrupt_requested=False,
        _sync_external_memory_for_turn=lambda **_kwargs: None,
        _spawn_background_review=lambda **_kwargs: None,
    )


@pytest.mark.parametrize("prompt_state", ["sending", "accepted", "reconciliation_required"])
def test_pi_finalizer_never_settles_prohibited_binding_state(prompt_state):
    class Session:
        binding = SimpleNamespace(prompt_state=prompt_state)

        def finalize_projection(self, **_kwargs):
            raise AssertionError("prohibited binding state must not reach settlement")

    turn = ExternalTurnResult(
        final_text="native result",
        projected_messages=[{"role": "assistant", "content": "native result"}],
        status=TurnStatus.COMPLETED,
        prompt_accepted=True,
    )
    result = finalize_external_turn(
        _finalizer_agent(Session()), [], turn, runtime="pi_rpc"
    )

    assert result["completed"] is False
    assert result["external_runtime"]["status"] == "reconciliation_required"
    assert result["external_runtime"]["projection_persisted"] is True


def test_pi_finalizer_never_settles_unknown_acceptance():
    class Session:
        binding = SimpleNamespace(prompt_state="native_completed")

        def finalize_projection(self, **_kwargs):
            raise AssertionError("unknown acceptance must not reach settlement")

    turn = ExternalTurnResult(
        status=TurnStatus.RECONCILIATION_REQUIRED,
        reconciliation_required=True,
        prompt_accepted=None,
        error="acceptance unknown",
    )
    result = finalize_external_turn(
        _finalizer_agent(Session()), [], turn, runtime="pi_rpc"
    )

    assert result["external_runtime"]["status"] == "reconciliation_required"
    assert result["external_runtime"]["projection_persisted"] is False


def test_empty_pi_projection_is_not_persistence_or_settlement():
    calls = []

    class Session:
        binding = SimpleNamespace(prompt_state="native_completed")

        def finalize_projection(self, *, persisted, turn):
            calls.append(persisted)
            assert turn.prompt_accepted is True
            self.binding.prompt_state = "reconciliation_required"

    turn = ExternalTurnResult(status=TurnStatus.COMPLETED, prompt_accepted=True)
    result = finalize_external_turn(
        _finalizer_agent(Session()), [], turn, runtime="pi_rpc"
    )

    assert calls == [False]
    assert result["completed"] is False
    assert result["external_runtime"]["projection_persisted"] is False
    assert "nothing was persisted" in result["error"]


def test_public_dispatch_preserves_uncertain_binding_byte_for_byte_until_reconciliation(
    monkeypatch, tmp_path
):
    sessions = tmp_path / "sessions"
    bindings = tmp_path / "bindings"
    sessions.mkdir()
    bindings.mkdir()
    native = sessions / "native.jsonl"
    native.write_text("{}\n", encoding="utf-8")
    binding_path = bindings / "binding.json"
    save_binding(
        binding_path,
        RuntimeBinding(
            runtime="pi_rpc",
            native_session_file=str(native),
            native_session_id="uncertain-native",
            last_entry_id="checkpoint",
            cwd=str(tmp_path.resolve()),
            runtime_version=PINNED_PI_VERSION,
            worktree=worktree_identity(tmp_path),
            model={"provider": "reviewed", "id": "model"},
            thinking_level="high",
            prompt_state="reconciliation_required",
            prompt_request_id="prompt-original",
            prompt_digest="digest-original",
            turn_generation=7,
        ),
    )
    rpc_calls = []

    class ReconcileClient:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

        def request(self, kind, **_kwargs):
            rpc_calls.append(kind)
            if kind == "get_state":
                return {"data": {
                    "sessionId": "uncertain-native",
                    "sessionFile": str(native),
                    "model": {"provider": "reviewed", "id": "model"},
                    "thinkingLevel": "high",
                }}
            if kind == "get_entries":
                return {"data": {
                    "leafId": "checkpoint",
                    "entries": [{"id": "checkpoint", "type": "message"}],
                }}
            raise AssertionError(f"reconciliation sent forbidden RPC command: {kind}")

        def close(self, **_kwargs):
            pass

    session = PiRpcSession(
        executable="pi",
        cwd=str(tmp_path),
        session_dir=sessions,
        binding_path=binding_path,
        client_factory=ReconcileClient,
    )
    original = binding_path.read_bytes()
    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session", lambda _agent: session
    )
    agent = _agent()
    agent._pi_rpc_session = session

    result = agent.run_conversation("a second prompt must not settle the first")

    assert result["external_runtime"]["status"] == "reconciliation_required"
    assert result["external_runtime"]["prompt_accepted"] is False
    assert binding_path.read_bytes() == original
    assert session.binding is not None
    assert session.binding.prompt_request_id == "prompt-original"
    assert rpc_calls == []

    report = session.reconcile("adopt")
    assert report["prompt_state"] == "settled"
    assert binding_path.read_bytes() != original
    assert session.binding is not None
    assert session.binding.prompt_state == "settled"
    assert "prompt" not in rpc_calls


def test_retirement_close_failure_never_escapes_public_path(monkeypatch):
    class ClosingFailure:
        def run_turn(self, _user_input):
            return ExternalTurnResult(
                final_text="settled", should_retire=True,
                phase="after_send", prompt_accepted=True,
            )

        def close(self):
            raise RuntimeError("close exploded")

    monkeypatch.setattr(
        "agent.pi_runtime.get_or_create_pi_session", lambda _agent: ClosingFailure()
    )
    result = _agent().run_conversation("sent once")
    assert result["completed"] is False
    assert result["external_runtime"]["phase"] == "after_send"
    assert result["external_runtime"]["prompt_accepted"] is True
    assert "close exploded" in result["error"]
