import json
from pathlib import Path

import pytest

from agent.transports.pi_event_projector import PiEventProjector
from agent.transports.pi_rpc_client import PiProtocolError


def test_message_end_authoritative_and_tool_ids_stable():
    p = PiEventProjector()
    p.project({"type": "message_start"})
    p.project({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "draft"}})
    result = p.project({"type": "message_end", "message": {"content": [
        {"type": "text", "text": "final"},
        {"type": "toolCall", "id": "1", "name": "bash", "arguments": {"command": "pwd"}},
    ]}})
    assert result.final_text == "final"
    p.project({"type": "tool_execution_start", "toolCallId": "1", "toolName": "bash", "args": {"command": "pwd"}})
    done = p.project({"type": "tool_execution_end", "toolCallId": "1", "result": "ok"})
    assert result.messages[0]["tool_calls"][0]["id"] == "pi_1"
    assert done.messages == [{"role": "tool", "tool_call_id": "pi_1", "content": "ok"}]


def test_parallel_tool_calls_share_one_assistant_record_and_results_match_ids():
    p = PiEventProjector()
    p.project({"type": "tool_execution_start", "toolCallId": "a", "toolName": "read"})
    p.project({"type": "tool_execution_start", "toolCallId": "b", "toolName": "read"})
    assistant = p.project({"type": "message_end", "message": {"content": [
        {"type": "text", "text": "checking"},
        {"type": "toolCall", "id": "a", "name": "read", "arguments": {"path": "a"}},
        {"type": "toolCall", "id": "b", "name": "read", "arguments": {"path": "b"}},
    ]}})
    assert len(assistant.messages) == 1
    assert [call["id"] for call in assistant.messages[0]["tool_calls"]] == ["pi_a", "pi_b"]
    assert p.project({"type": "tool_execution_end", "toolCallId": "a", "result": "A"}).messages[0]["tool_call_id"] == "pi_a"
    assert p.project({"type": "tool_execution_end", "toolCallId": "b", "result": "B"}).messages[0]["tool_call_id"] == "pi_b"


def test_content_indexes_thinking_usage_and_compaction_are_projected():
    p = PiEventProjector()
    p.project({"type": "message_start"})
    p.project({"type": "message_update", "assistantMessageEvent": {
        "type": "thinking_delta", "contentIndex": 0, "delta": "why "
    }})
    p.project({"type": "message_update", "assistantMessageEvent": {
        "type": "text_delta", "contentIndex": 1, "delta": "hello"
    }})
    result = p.project({
        "type": "message_end",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
            "usage": {"input": 10, "output": 3, "cost": {"total": 0.1}},
        },
    })
    assert result.messages == [{"role": "assistant", "content": "answer", "reasoning": "reason"}]
    assert p.usage["input"] == 10
    p.project({"type": "compaction_end", "result": {"summary": "bounded"}})
    assert p.compacted is True


def test_projected_turn_has_aggregate_budget():
    p = PiEventProjector(max_turn_bytes=8, max_preview_chars=8)
    p.project({"type": "message_start"})
    p.project({"type": "message_update", "assistantMessageEvent": {
        "type": "text_delta", "delta": "12345678"
    }})
    with pytest.raises(PiProtocolError, match="storage byte budget"):
        p.project({"type": "message_end", "message": {"content": "12345678"}})


def test_tool_arguments_and_usage_count_toward_aggregate_budget():
    with pytest.raises(PiProtocolError, match="storage byte budget"):
        PiEventProjector(max_turn_bytes=32).project({
            "type": "tool_execution_start", "toolCallId": "1",
            "toolName": "read", "args": {"path": "x" * 100},
        })
    with pytest.raises(PiProtocolError, match="storage byte budget"):
        PiEventProjector(max_turn_bytes=32).project({
            "type": "message_end", "message": {
                "content": "", "usage": {"providerDetails": "x" * 100},
            },
        })


def test_sanitized_settled_fixture_uses_only_agent_settled_as_fence():
    records = [
        json.loads(line)
        for line in Path("tests/fixtures/pi_rpc/settled.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    projector = PiEventProjector()
    final = ""
    fences = 0
    for record in records:
        if record.get("type") == "response":
            assert record.get("command")
            continue
        projection = projector.project(record)
        final = projection.final_text or final
        fences += int(record.get("type") == "agent_settled")
    assert final == "done"
    assert fences == 1
