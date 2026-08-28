"""Single runtime-neutral conversation-loop branch point."""
from __future__ import annotations

from typing import Any

from agent.external_runtime.finalize import finalize_external_turn
from agent.external_runtime.types import ExternalTurnResult, TurnStatus
from agent.redact import redact_sensitive_text

EXTERNAL_API_MODES = frozenset({"codex_app_server", "pi_rpc"})


def dispatch_external_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool,
) -> dict[str, Any]:
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=should_review_memory,
        )
    if agent.api_mode == "pi_rpc":
        from agent.pi_runtime import run_pi_rpc_turn
        try:
            return run_pi_rpc_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                effective_task_id=effective_task_id,
                should_review_memory=should_review_memory,
            )
        except Exception as exc:
            # This is the final public-path fence. Acquisition, provisioning,
            # image inspection, adapter construction, and cleanup must never
            # leak a raw exception from AIAgent.run_conversation(). Because
            # run_pi_rpc_turn normally owns the send boundary. If an untyped
            # exception escapes it, this top-level fence has no evidence about
            # acceptance and must never authorize an automatic replay.
            turn = ExternalTurnResult(
                status=TurnStatus.RECONCILIATION_REQUIRED,
                error=redact_sensitive_text(str(exc), force=True),
                should_retire=True,
                reconciliation_required=True,
                phase="unknown_after_send",
                prompt_accepted=None,
            )
            return finalize_external_turn(
                agent,
                messages,
                turn,
                runtime="pi_rpc",
                original_user_message=original_user_message,
                should_review_memory=should_review_memory,
            )
    raise ValueError(f"not an external runtime: {agent.api_mode}")
