from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TurnStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    RECONCILIATION_REQUIRED = "reconciliation_required"


@dataclass
class RuntimeState:
    runtime: str
    lifecycle: str = "idle"
    session_id: str | None = None
    session_file: str | None = None
    cursor: str | None = None
    model_provider: str | None = None
    model_id: str | None = None
    thinking_level: str | None = None
    is_streaming: bool = False
    is_compacting: bool = False
    pending_message_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ExternalTurnResult:
    final_text: str = ""
    projected_messages: list[dict[str, Any]] = field(default_factory=list)
    status: TurnStatus = TurnStatus.COMPLETED
    error: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    session_id: str | None = None
    session_file: str | None = None
    cursor: str | None = None
    run_id: str | None = None
    tool_iterations: int = 0
    compacted: bool = False
    should_retire: bool = False
    reconciliation_required: bool = False
    # ``prompt_accepted`` is deliberately tri-state. False proves that no
    # prompt bytes crossed the runtime boundary, True proves acceptance, and
    # None means transmission may have happened and replay is unsafe.
    phase: str = "after_send"
    prompt_accepted: bool | None = True
    native: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def interrupted(self) -> bool:
        return self.status == TurnStatus.INTERRUPTED

    @property
    def thread_id(self) -> str | None:
        return self.session_id

    @property
    def turn_id(self) -> str | None:
        return self.run_id

    @property
    def token_usage_last(self) -> dict[str, Any] | None:
        value = self.native.get("token_usage_last")
        return value if isinstance(value, dict) else None

    @property
    def token_usage_total(self) -> dict[str, Any] | None:
        value = self.native.get("token_usage_total")
        return value if isinstance(value, dict) else None

    @property
    def model_context_window(self) -> int | None:
        value = self.native.get("model_context_window")
        return int(value) if isinstance(value, int) else None


@dataclass
class CompactionResult:
    completed: bool
    cursor: str | None = None
    error: str | None = None
