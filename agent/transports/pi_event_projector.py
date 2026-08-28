"""Bounded projection of Pi 0.84.3 events into Hermes message records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent.transports.pi_rpc_client import PiProtocolError, redact_protocol_text, redact_protocol_value

MAX_PROJECTED_TURN_BYTES = 1024 * 1024
MAX_PREVIEW_CHARS = 16 * 1024


@dataclass
class PiProjection:
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_text: str | None = None
    tool_iteration: bool = False
    status: dict[str, Any] | None = None


def _call_id(event: dict[str, Any]) -> str:
    native = str(event.get("toolCallId") or event.get("tool_call_id") or "")
    if native:
        return f"pi_{native}"
    stable = json.dumps(event, ensure_ascii=False, sort_keys=True)
    return "pi_" + hashlib.sha256(stable.encode()).hexdigest()[:20]


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or item.get("thinking") or "")
            for item in value if isinstance(item, dict)
        )
    if isinstance(value, dict):
        return _content_text(value.get("content") or value.get("text") or "")
    return "" if value is None else str(value)


class PiEventProjector:
    """Assemble authoritative messages while enforcing a per-turn byte cap."""

    def __init__(
        self,
        *,
        max_preview_chars: int = MAX_PREVIEW_CHARS,
        max_turn_bytes: int = MAX_PROJECTED_TURN_BYTES,
    ) -> None:
        self.max_preview_chars = max_preview_chars
        self.max_turn_bytes = max_turn_bytes
        self._stored_bytes = 0
        self._text: dict[int, list[str]] = {}
        self._thinking: dict[int, list[str]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._emitted_tool_calls: set[str] = set()
        self.usage: dict[str, Any] = {}
        self.compacted = False

    def _bounded_text(self, value: Any, *, limit: int | None = None) -> str:
        text = redact_protocol_text(_content_text(value), limit=limit or self.max_preview_chars)
        encoded = text.encode("utf-8")
        if self._stored_bytes + len(encoded) > self.max_turn_bytes:
            raise PiProtocolError("Pi projected turn storage byte budget exceeded")
        self._stored_bytes += len(encoded)
        return text

    def _capture_usage(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        bounded = redact_protocol_value(value, max_string=min(self.max_preview_chars, 4096))
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self._stored_bytes + len(encoded) > self.max_turn_bytes:
            raise PiProtocolError("Pi projected turn storage byte budget exceeded")
        self._stored_bytes += len(encoded)
        self.usage = bounded

    def _bounded_value(self, value: Any) -> Any:
        bounded = redact_protocol_value(value, max_string=min(self.max_preview_chars, 4096))
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self._stored_bytes + len(encoded) > self.max_turn_bytes:
            raise PiProtocolError("Pi projected turn storage byte budget exceeded")
        self._stored_bytes += len(encoded)
        return bounded

    def project(self, event: dict[str, Any]) -> PiProjection:
        event = redact_protocol_value(event)
        kind = str(event.get("type") or "")
        self._capture_usage(event.get("usage"))
        if kind == "message_start":
            self._text = {}
            self._thinking = {}
            return PiProjection()
        if kind == "message_update":
            update = event.get("assistantMessageEvent") or {}
            if not isinstance(update, dict):
                return PiProjection()
            update_kind = str(update.get("type") or "")
            index = int(update.get("contentIndex") or 0)
            delta = self._bounded_text(update.get("delta") or update.get("text") or "")
            if update_kind in {"thinking_delta", "reasoning_delta"}:
                self._thinking.setdefault(index, []).append(delta)
            elif update_kind == "text_delta":
                self._text.setdefault(index, []).append(delta)
            return PiProjection()
        if kind == "message_end":
            message = event.get("message") or {}
            if not isinstance(message, dict) or message.get("role") not in {None, "assistant"}:
                return PiProjection()
            content = message.get("content")
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text") or ""))
                    elif part.get("type") == "thinking":
                        thinking_parts.append(str(part.get("thinking") or ""))
                    elif part.get("type") in {"toolCall", "tool_call"}:
                        native_id = str(
                            part.get("id") or part.get("toolCallId")
                            or part.get("tool_call_id") or ""
                        )
                        call_id = f"pi_{native_id}" if native_id else _call_id(part)
                        name = str(part.get("name") or part.get("toolName") or "tool")[:128]
                        raw_args = part.get("arguments", part.get("args", {}))
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                raw_args = {"input": raw_args}
                        args = self._bounded_value(raw_args or {})
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    args, ensure_ascii=False, sort_keys=True,
                                ),
                            },
                        })
                        self._emitted_tool_calls.add(call_id)
            text = self._bounded_text("".join(text_parts) if text_parts else "".join(
                "".join(self._text[index]) for index in sorted(self._text)
            ), limit=self.max_turn_bytes)
            thinking = self._bounded_text("".join(thinking_parts) if thinking_parts else "".join(
                "".join(self._thinking[index]) for index in sorted(self._thinking)
            ), limit=self.max_turn_bytes)
            result: dict[str, Any] = {"role": "assistant", "content": text}
            if thinking:
                result["reasoning"] = thinking
            if tool_calls:
                result["tool_calls"] = tool_calls
                if not text:
                    result["content"] = None
            self._capture_usage(message.get("usage"))
            return PiProjection(messages=[result], final_text=text)
        if kind == "tool_execution_start":
            call_id = _call_id(event)
            self._tools[call_id] = {
                "toolName": str(event.get("toolName") or "tool")[:128],
                "args": self._bounded_value(event.get("args") or {}),
            }
            return PiProjection()
        if kind == "tool_execution_update":
            return PiProjection(status={
                "kind": "tool_progress",
                "id": _call_id(event),
                "name": str(event.get("toolName") or "tool")[:128],
                "preview": self._bounded_text(event.get("partialResult") or ""),
            })
        if kind == "tool_execution_end":
            call_id = _call_id(event)
            started = self._tools.pop(call_id, {})
            name = str(event.get("toolName") or started.get("toolName") or "tool")[:128]
            event_args = event.get("args")
            args = (
                self._bounded_value(event_args)
                if event_args
                else started.get("args") or {}
            )
            output = self._bounded_text(event.get("result") or event.get("output") or "")
            if call_id not in self._emitted_tool_calls:
                raise PiProtocolError(
                    "Pi tool result has no authoritative assistant tool-call record"
                )
            tool = {"role": "tool", "tool_call_id": call_id, "content": output}
            return PiProjection(messages=[tool], tool_iteration=True)
        if kind in {
            "queue_update", "compaction_start", "compaction_end", "auto_retry_start",
            "auto_retry_end", "summarization_retry_scheduled",
            "summarization_retry_attempt_start", "summarization_retry_finished",
            "extension_error", "extension_ui_request",
        }:
            if kind == "compaction_end" and not event.get("aborted") and event.get("result"):
                self.compacted = True
                result = event.get("result") or {}
                if isinstance(result, dict):
                    self._capture_usage(result.get("usage"))
            return PiProjection(status={"kind": kind, "event": event})
        return PiProjection()
