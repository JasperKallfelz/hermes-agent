import pytest

from hermes_cli.runtime_switch import get_current_runtime, parse_args, set_runtime


def test_legacy_migration_and_neutral_persistence():
    cfg = {"model": {"openai_runtime": "codex_app_server"}}
    assert get_current_runtime(cfg) == "codex_app_server"
    assert set_runtime(cfg, "pi_rpc") == "codex_app_server"
    assert cfg["agent"]["runtime"] == "pi_rpc"
    assert cfg["model"]["openai_runtime"] == "auto"


def test_runtime_aliases_do_not_create_model_provider():
    assert parse_args("pi") == ("pi_rpc", [])
    assert parse_args("codex") == ("codex_app_server", [])
    assert parse_args("wat")[0] is None


def test_global_pi_selection_does_not_rewrite_auxiliary_provider_resolution(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    config = {
        "agent": {"runtime": "pi_rpc", "pi_rpc": {"enabled": True}},
        "providers": {},
    }
    monkeypatch.setattr(runtime_provider, "load_config", lambda: config)
    monkeypatch.setattr(runtime_provider, "resolve_requested_provider", lambda requested=None: "anthropic-custom")
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_named_custom_runtime",
        lambda **kwargs: {
            "provider": "custom",
            "api_mode": "anthropic_messages",
            "base_url": "https://example.invalid",
            "api_key": "stub",
            "source": "test",
        },
    )
    result = runtime_provider.resolve_runtime_provider(requested="anthropic-custom")
    assert result["provider"] == "custom"
    assert result["api_mode"] == "anthropic_messages"
    assert "agent_runtime" not in result


def test_main_agent_pi_resolution_needs_no_inference_provider(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "pi_rpc", "pi_rpc": {"enabled": True}}},
    )
    result = runtime_provider.resolve_main_agent_runtime_provider()
    assert result["provider"] == "pi"
    assert result["api_mode"] == "pi_rpc"
    assert result["agent_runtime"] == "pi_rpc"


def test_main_agent_non_pi_resolution_preserves_provider_contract(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    provider_result = {
        "provider": "openai-codex",
        "api_mode": "codex_app_server",
    }
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return provider_result

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "hermes"}},
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolver)

    result = runtime_provider.resolve_main_agent_runtime_provider(
        requested="openai-codex",
        target_model="gpt-5.4",
    )

    assert calls == [
        {"requested": "openai-codex", "target_model": "gpt-5.4"},
    ]
    assert result == provider_result
    assert result is not provider_result
    assert "agent_runtime" not in result


def test_main_agent_forwards_non_none_explicit_provider_fields(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    calls = []
    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "hermes"}},
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: calls.append(kwargs) or {"provider": "custom"},
    )

    runtime_provider.resolve_main_agent_runtime_provider(
        requested="custom:test",
        target_model="test-model",
        explicit_api_key="test-key",
        explicit_base_url="https://example.invalid/v1",
    )

    assert calls == [
        {
            "requested": "custom:test",
            "target_model": "test-model",
            "explicit_api_key": "test-key",
            "explicit_base_url": "https://example.invalid/v1",
        }
    ]


def test_main_agent_omits_none_provider_fields(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    calls = []

    def legacy_resolver(*, requested=None):
        calls.append({"requested": requested})
        return {"provider": "legacy"}

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "hermes"}},
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", legacy_resolver)

    result = runtime_provider.resolve_main_agent_runtime_provider(requested="legacy")

    assert result == {"provider": "legacy"}
    assert calls == [{"requested": "legacy"}]


def test_main_agent_does_not_hide_resolver_type_error(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    def broken_resolver(*, requested=None, target_model=None):
        raise TypeError(f"resolver bug for {requested}/{target_model}")

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "hermes"}},
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", broken_resolver)

    with pytest.raises(TypeError, match="resolver bug for openrouter/anthropic/claude"):
        runtime_provider.resolve_main_agent_runtime_provider(
            requested="openrouter",
            target_model="anthropic/claude",
        )


def test_main_agent_config_failure_does_not_fall_back_to_provider(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    def fail_config():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(runtime_provider, "load_config", fail_config)
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not resolve provider")
        ),
    )
    with pytest.raises(RuntimeError, match="config unavailable"):
        runtime_provider.resolve_main_agent_runtime_provider()


def test_main_pi_ignores_unrelated_disabled_provider(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {
            "agent": {"runtime": "pi_rpc", "pi_rpc": {"enabled": True}},
            "providers": {"openrouter": {"enabled": False}},
        },
    )
    result = runtime_provider.resolve_main_agent_runtime_provider(requested="openrouter")
    assert result["api_mode"] == "pi_rpc"
    assert result["provider"] == "pi"


def test_disabled_pi_gate_fails_before_provider_resolution(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "pi_rpc", "pi_rpc": {"enabled": False}}},
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not resolve provider")),
    )
    with pytest.raises(RuntimeError, match="explicit agent.pi_rpc.enabled gate"):
        runtime_provider.resolve_main_agent_runtime_provider()


def test_global_pi_does_not_contaminate_auxiliary_codex_wire_mode(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"agent": {"runtime": "pi_rpc", "pi_rpc": {"enabled": True}}},
    )
    mode = runtime_provider._maybe_apply_codex_app_server_runtime(
        provider="openai-codex", api_mode="codex_responses",
        model_cfg={"openai_runtime": "codex_app_server"},
    )
    assert mode == "codex_responses"
