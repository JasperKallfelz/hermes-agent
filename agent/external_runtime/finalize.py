"""Shared projection/persistence boundary for all external agent runtimes."""
from __future__ import annotations

import logging
from typing import Any

from agent.redact import redact_sensitive_text
from .binding import SETTLEMENT_PROHIBITED_PROMPT_STATES
from .types import ExternalTurnResult, TurnStatus

logger = logging.getLogger(__name__)


def persist_external_projection(
    agent: Any,
    messages: list[dict[str, Any]],
    turn: ExternalTurnResult | Any,
    *,
    label: str,
) -> bool:
    """Append and flush a runtime projection exactly once through the agent DB."""
    projected = list(getattr(turn, "projected_messages", None) or [])
    if projected:
        from agent.message_metadata import append_message

        for projected_message in projected:
            append_message(messages, projected_message)
    if not projected or getattr(agent, "_session_db", None) is None:
        return True
    try:
        persisted = agent._flush_messages_to_session_db(messages)
    except Exception:
        logger.warning("%s projected-message flush failed", label, exc_info=True)
        return False
    if persisted is False:
        logger.warning(
            "%s turn was delivered but could not be persisted to the session DB (session=%s)",
            label,
            getattr(agent, "session_id", None),
        )
        return False
    return True


def build_external_result(
    messages: list[dict[str, Any]],
    turn: ExternalTurnResult,
    *,
    runtime: str,
    interrupted: bool | None = None,
    interrupt_message: str | None = None,
    agent_persisted: bool = True,
) -> dict[str, Any]:
    error = redact_sensitive_text(turn.error or "", force=True) or None
    was_interrupted = turn.interrupted if interrupted is None else interrupted
    result: dict[str, Any] = {
        "final_response": redact_sensitive_text(turn.final_text, force=True),
        "messages": messages,
        "api_calls": 1,
        "completed": turn.status.value == "completed" and error is None,
        "partial": turn.status.value != "completed" or error is not None,
        "interrupted": was_interrupted,
        "error": error,
        "agent_persisted": agent_persisted,
        "external_runtime": {
            "runtime": runtime,
            "session_id": turn.session_id,
            "session_file": turn.session_file,
            "cursor": turn.cursor,
            "run_id": turn.run_id,
            "status": turn.status.value,
            "reconciliation_required": turn.reconciliation_required,
            "phase": turn.phase,
            "prompt_accepted": turn.prompt_accepted,
        },
        "input_tokens": turn.prompt_tokens,
        "output_tokens": turn.output_tokens,
        "cache_read_tokens": turn.cache_read_tokens,
        "cache_write_tokens": turn.cache_write_tokens,
        "cost": turn.cost,
    }
    if interrupt_message:
        result["interrupt_message"] = redact_sensitive_text(interrupt_message, force=True)
    return result


def finalize_external_turn(
    agent: Any,
    messages: list[dict[str, Any]],
    turn: ExternalTurnResult,
    *,
    runtime: str,
    original_user_message: Any = None,
    should_review_memory: bool = False,
) -> dict[str, Any]:
    """Finalize a normalized external turn without replaying native history."""
    has_projection = bool(getattr(turn, "projected_messages", None))
    persisted = persist_external_projection(agent, messages, turn, label=runtime)
    if runtime == "pi_rpc":
        # ``persist_external_projection`` returns success when there is no work
        # to flush.  That is useful to its non-Pi callers, but an empty Pi
        # projection is not a durability boundary and must never settle a
        # possibly accepted native prompt.
        persisted = bool(has_projection and persisted)
        session = getattr(agent, "_pi_rpc_session", None)
        finalize_projection = getattr(session, "finalize_projection", None)
        binding = getattr(session, "binding", None)
        binding_state = getattr(binding, "prompt_state", None)
        settlement_prohibited = (
            bool(turn.reconciliation_required)
            or turn.prompt_accepted is None
            or binding_state in SETTLEMENT_PROHIBITED_PROMPT_STATES
        )
        settlement_expected = turn.prompt_accepted is True

        if settlement_expected and not settlement_prohibited and callable(finalize_projection):
            try:
                finalize_projection(persisted=persisted, turn=turn)
            except Exception:
                logger.warning("Pi projection settlement fence failed", exc_info=True)
                settlement_prohibited = True
        elif settlement_expected and not callable(finalize_projection):
            settlement_prohibited = True

        if settlement_expected and (not persisted or settlement_prohibited):
            turn.reconciliation_required = True
            turn.status = TurnStatus.RECONCILIATION_REQUIRED
            if turn.error is None:
                if not has_projection:
                    turn.error = (
                        "Pi completed the native turn without a canonical projection; "
                        "nothing was persisted and explicit reconciliation is required"
                    )
                elif settlement_prohibited:
                    turn.error = (
                        "Pi settlement was refused because prompt acceptance or binding "
                        "state is uncertain; explicit reconciliation is required"
                    )
                else:
                    turn.error = (
                        "Pi completed the turn but Hermes could not durably persist its "
                        "canonical projection; explicit reconciliation is required"
                    )
        elif turn.reconciliation_required or turn.prompt_accepted is None:
            # The turn already carries the fail-closed public result.  Crucially,
            # do not call the session finalizer: even a no-op/empty projection
            # must leave the prior uncertain binding byte-for-byte unchanged.
            turn.reconciliation_required = True
            turn.status = TurnStatus.RECONCILIATION_REQUIRED
    agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    should_review_skills = False
    if (
        getattr(agent, "_skill_nudge_interval", 0) > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in getattr(agent, "valid_tool_names", set())
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)
    if turn.final_text and not turn.interrupted and (should_review_memory or should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("external background review spawn raised", exc_info=True)

    interrupted = bool(turn.interrupted and getattr(agent, "_interrupt_requested", False))
    interrupt_message = getattr(agent, "_interrupt_message", None) if interrupted else None
    if interrupted:
        agent.clear_interrupt()
    result = build_external_result(
        messages, turn, runtime=runtime, interrupted=interrupted,
        # The inbound Hermes message is already owned by the agent persistence
        # path.  A failed *external projection* flush must not cause gateway
        # callers to append the user message a second time.
        interrupt_message=interrupt_message, agent_persisted=True,
    )
    result["external_runtime"]["projection_persisted"] = persisted
    return result
