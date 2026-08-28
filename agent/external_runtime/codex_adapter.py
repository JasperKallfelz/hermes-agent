"""Neutral adapter around the existing Codex app-server session."""
from __future__ import annotations

from typing import Any

from .types import CompactionResult, ExternalTurnResult, RuntimeState, TurnStatus


class CodexExternalRuntimeAdapter:
    """Keep Codex protocol details behind the shared external-runtime shape."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def ensure_started(self) -> RuntimeState:
        thread_id = self.session.ensure_started()
        return RuntimeState(runtime="codex_app_server", session_id=thread_id)

    def run_turn(self, user_input: object) -> ExternalTurnResult:
        native = self.session.run_turn(user_input=user_input)
        if getattr(native, "interrupted", False):
            status = TurnStatus.INTERRUPTED
        elif getattr(native, "error", None):
            status = TurnStatus.ERROR
        else:
            status = TurnStatus.COMPLETED
        return ExternalTurnResult(
            final_text=getattr(native, "final_text", ""),
            projected_messages=getattr(native, "projected_messages", []),
            status=status,
            error=getattr(native, "error", None),
            session_id=getattr(native, "thread_id", None),
            run_id=getattr(native, "turn_id", None),
            tool_iterations=getattr(native, "tool_iterations", 0),
            compacted=getattr(native, "compacted", False),
            should_retire=getattr(native, "should_retire", False),
            native={
                "token_usage_last": getattr(native, "token_usage_last", None),
                "token_usage_total": getattr(native, "token_usage_total", None),
                "model_context_window": getattr(native, "model_context_window", None),
            },
        )

    def steer(self, message: str) -> bool:
        return bool(self.session.request_steer(message))

    def follow_up(self, message: str) -> bool:
        return False

    def abort(self) -> None:
        self.session.request_interrupt()

    def set_model(self, provider: str, model_id: str) -> RuntimeState:
        raise NotImplementedError("Codex model selection is owned by its existing runtime config")

    def set_thinking_level(self, level: str) -> RuntimeState:
        raise NotImplementedError("Codex thinking selection is owned by its existing runtime config")

    def compact(self, instructions: str | None = None) -> CompactionResult:
        _ = instructions
        native = self.session.compact_thread()
        return CompactionResult(not bool(native.error), error=native.error)

    def get_state(self) -> RuntimeState:
        return self.ensure_started()

    def close(self) -> None:
        self.session.close()
