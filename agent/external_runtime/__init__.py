"""Protocol-neutral external coding runtime contracts."""

from .base import ExternalAgentRuntime
from .dispatch import EXTERNAL_API_MODES, dispatch_external_turn
from .types import ExternalTurnResult, RuntimeState

__all__ = [
    "EXTERNAL_API_MODES", "ExternalAgentRuntime", "ExternalTurnResult",
    "RuntimeState", "dispatch_external_turn",
]
