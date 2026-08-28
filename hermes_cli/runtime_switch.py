"""Neutral runtime selection; changes are effective in a fresh session."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

VALID_RUNTIMES = ("hermes", "codex_app_server", "pi_rpc")


@dataclass
class RuntimeStatus:
    success: bool
    old_value: str
    new_value: str | None
    message: str
    requires_new_session: bool = False


def get_current_runtime(config: dict) -> str:
    agent = config.get("agent") if isinstance(config, dict) else None
    value = str((agent or {}).get("runtime") or "").strip().lower() if isinstance(agent, dict) else ""
    if value in VALID_RUNTIMES:
        return value
    legacy = str(((config.get("model") or {}) if isinstance(config, dict) else {}).get("openai_runtime") or "")
    return "codex_app_server" if legacy == "codex_app_server" else "hermes"


def parse_args(value: str) -> tuple[str | None, list[str]]:
    raw = (value or "").strip().lower().replace("-", "_")
    aliases = {"auto": "hermes", "codex": "codex_app_server", "pi": "pi_rpc"}
    raw = aliases.get(raw, raw)
    if not raw:
        return None, []
    if raw not in VALID_RUNTIMES:
        return None, ["Unknown runtime. Use: hermes, codex, or pi"]
    return raw, []


def set_runtime(config: dict, runtime: str) -> str:
    if runtime not in VALID_RUNTIMES:
        raise ValueError(f"invalid runtime: {runtime}")
    old = get_current_runtime(config)
    config.setdefault("agent", {})["runtime"] = runtime
    # Keep the legacy input coherent for older Hermes versions.
    config.setdefault("model", {})["openai_runtime"] = (
        "codex_app_server" if runtime == "codex_app_server" else "auto"
    )
    return old


def apply(config: dict, runtime: str | None, *, persist_callback: Callable | None = None) -> RuntimeStatus:
    old = get_current_runtime(config)
    if runtime is None:
        return RuntimeStatus(True, old, old, f"runtime: {old}")
    if runtime == "pi_rpc":
        pi_settings = ((config.get("agent") or {}).get("pi_rpc") or {}) if isinstance(config, dict) else {}
        if not isinstance(pi_settings, dict) or pi_settings.get("enabled") is not True:
            return RuntimeStatus(
                False, old, None,
                "Cannot enable Pi RPC: experimental gate agent.pi_rpc.enabled is not true",
            )
        try:
            from agent.pi_runtime import managed_pi_executable
            executable = managed_pi_executable()
        except Exception as exc:
            return RuntimeStatus(False, old, None, f"Cannot enable Pi RPC: {exc}")
    else:
        executable = None
    if runtime == old:
        return RuntimeStatus(True, old, old, f"runtime already set to {old}")
    set_runtime(config, runtime)
    if persist_callback:
        persist_callback(config)
    detail = f"runtime: {old} → {runtime}"
    if executable:
        detail += f"\nPi: {executable} (experimental; authentication required separately)"
    return RuntimeStatus(True, old, runtime, detail, requires_new_session=True)
