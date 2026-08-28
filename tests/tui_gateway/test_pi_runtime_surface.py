from types import SimpleNamespace

import pytest

import tui_gateway.server as server


class _PiSession:
    def __init__(self):
        self.model = ("openai", "gpt-test")
        self.thinking = "medium"

    def available_models(self):
        return [
            {"provider": "openai", "id": "gpt-test"},
            {"provider": "anthropic", "id": "claude-test"},
            {"provider": "openrouter", "id": "anthropic/claude-sonnet"},
        ]

    def get_state(self):
        return SimpleNamespace(
            model_provider=self.model[0], model_id=self.model[1],
            thinking_level=self.thinking,
        )

    def set_model(self, provider, model_id):
        self.model = (provider, model_id)
        return self.get_state()

    def set_thinking_level(self, level):
        self.thinking = level
        return self.get_state()


@pytest.fixture
def pi_surface(monkeypatch):
    native = _PiSession()
    agent = SimpleNamespace(
        api_mode="pi_rpc", model="gpt-test", provider="openai",
        session_total_tokens=0,
    )
    session = {
        "agent": agent,
        "running": False,
        "history": [],
        "history_lock": __import__("threading").RLock(),
        "session_key": "pi-key",
    }
    server._sessions["pi-surface"] = session
    monkeypatch.setattr("agent.pi_runtime.get_or_create_pi_session", lambda value: native)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    yield native, agent, session
    server._sessions.pop("pi-surface", None)


def test_model_options_come_only_from_pi_rpc_catalog(pi_surface):
    response = server._methods["model.options"]("one", {"session_id": "pi-surface"})
    result = response["result"]
    assert result["runtime"] == "pi_rpc"
    assert [row["slug"] for row in result["providers"]] == [
        "anthropic",
        "openai",
        "openrouter",
    ]
    assert all(row["runtime"] == "pi_rpc" for row in result["providers"])
    assert all(row["slug"] != "pi" for row in result["providers"])


def test_pi_model_and_thinking_controls_use_native_rpc(pi_surface):
    native, agent, session = pi_surface
    result = server._apply_model_switch(
        "pi-surface", session, "claude-test --provider anthropic --session"
    )
    assert result["runtime"] == "pi_rpc"
    assert native.model == ("anthropic", "claude-test")
    assert (agent.provider, agent.model) == native.model

    response = server._methods["config.set"](
        "two", {"session_id": "pi-surface", "key": "reasoning", "value": "high"}
    )
    assert response["result"] == {"key": "reasoning", "value": "high", "runtime": "pi_rpc"}
    assert native.thinking == "high"


def test_pi_model_provider_slash_form_uses_the_shared_parser(pi_surface):
    native, agent, session = pi_surface
    result = server._apply_model_switch(
        "pi-surface", session, "anthropic/claude-test --session"
    )
    assert result["runtime"] == "pi_rpc"
    assert native.model == ("anthropic", "claude-test")
    assert (agent.provider, agent.model) == native.model


def test_pi_model_nested_slash_form_uses_the_shared_parser(pi_surface):
    native, agent, session = pi_surface
    result = server._apply_model_switch(
        "pi-surface",
        session,
        "openrouter/anthropic/claude-sonnet --session",
    )
    assert result["runtime"] == "pi_rpc"
    assert native.model == ("openrouter", "anthropic/claude-sonnet")
    assert (agent.provider, agent.model) == native.model


def test_pi_model_global_switch_is_rejected(pi_surface):
    _, _, session = pi_surface
    with pytest.raises(ValueError, match="scoped"):
        server._apply_model_switch(
            "pi-surface", session, "gpt-test --provider openai --global"
        )


def test_runtime_pi_slash_dispatch_preserves_fresh_session_contract(pi_surface):
    _native, agent, session = pi_surface
    calls = []

    class Worker:
        def run(self, command):
            calls.append(command)
            return "runtime: hermes → pi_rpc\nStart a new session with /new to apply it."

    session["slash_worker"] = Worker()
    response = server._methods["slash.exec"](
        "runtime", {"session_id": "pi-surface", "command": "runtime pi"}
    )
    assert calls == ["runtime pi"]
    assert "Start a new session" in response["result"]["output"]
    assert session["agent"] is agent


@pytest.mark.parametrize("action", ["inspect", "adopt", "abandon"])
def test_runtime_reconcile_dispatches_live_session_without_slash_worker(
    monkeypatch, pi_surface, action
):
    _native, agent, session = pi_surface
    calls = []
    session["slash_worker"] = SimpleNamespace(
        run=lambda _command: (_ for _ in ()).throw(AssertionError("must not use slash worker"))
    )
    monkeypatch.setattr(
        "agent.pi_runtime.reconcile_pi_agent",
        lambda selected, requested: calls.append((selected, requested)) or {
            "action": requested, "replayed": False,
        },
    )
    response = server._methods["slash.exec"](
        "reconcile",
        {"session_id": "pi-surface", "command": f"runtime reconcile {action}"},
    )
    assert calls == [(agent, action)]
    output = response["result"]["output"]
    assert f"Pi reconciliation {action}" in output
    assert "prompt=unknown" in output
