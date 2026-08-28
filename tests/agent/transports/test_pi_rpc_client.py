import json
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
import psutil

from agent.transports.pi_rpc_client import (
    LiteralLfJsonDecoder,
    PiProtocolError,
    PiRequestUncertainError,
    PiRpcClient,
    _Waiter,
    sanitized_pi_environment,
    redact_protocol_value,
)


def test_fragmented_utf8_and_unicode_line_separators_are_not_records():
    decoder = LiteralLfJsonDecoder()
    raw = json.dumps({"text": "café\u2028x\u2029y"}, ensure_ascii=False).encode() + b"\n"
    out = []
    for byte in raw:
        out.extend(decoder.feed(bytes([byte])))
    assert out == [{"text": "café\u2028x\u2029y"}]


def test_crlf_tolerated_but_partial_and_malformed_fail():
    assert LiteralLfJsonDecoder().feed(b'{"ok":true}\r\n') == [{"ok": True}]
    decoder = LiteralLfJsonDecoder(); decoder.feed(b'{"x":')
    with pytest.raises(PiProtocolError, match="partial"):
        decoder.finish()
    with pytest.raises(PiProtocolError, match="invalid"):
        LiteralLfJsonDecoder().feed(b"nope\n")


def test_record_bound_and_secret_environment_allowlist():
    with pytest.raises(PiProtocolError, match="size"):
        LiteralLfJsonDecoder(max_record_bytes=4).feed(b"12345")
    source = {"PATH": "/bin", "TELEGRAM_TOKEN": "no", "OPENAI_API_KEY": "explicit", "RANDOM": "no"}
    assert sanitized_pi_environment(source) == {
        "PATH": "/bin", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"
    }
    assert sanitized_pi_environment(source, allowed_provider_vars={"OPENAI_API_KEY"}) == {
        "PATH": "/bin", "OPENAI_API_KEY": "explicit",
        "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
    }


def test_response_correlation_requires_type_id_and_command(tmp_path):
    client = PiRpcClient("pi", cwd=str(tmp_path))
    inbox: queue.Queue[object] = queue.Queue(maxsize=1)
    client._responses["one"] = _Waiter("get_state", inbox)

    # An event carrying the same id is still an event, never a response.
    client._dispatch({"type": "agent_start", "id": "one"})
    assert client.next_event(timeout=0)["type"] == "agent_start"
    assert inbox.empty()

    client._dispatch({"type": "response", "id": "one", "command": "get_state", "success": True})
    response = cast(dict[str, Any], inbox.get_nowait())
    assert response["success"] is True
    client._dispatch({"type": "response", "id": "one", "command": "get_state", "success": True})
    assert "duplicate" in str(client.fatal_error)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"type": "response", "id": "foreign", "command": "get_state"}, "unknown"),
        ({"type": "response", "id": "one", "command": "get_entries"}, "did not match"),
    ],
)
def test_foreign_or_mismatched_response_is_fatal(tmp_path, record, message):
    client = PiRpcClient("pi", cwd=str(tmp_path))
    if record["id"] == "one":
        client._responses["one"] = _Waiter("get_state", queue.Queue(maxsize=1))
    client._dispatch(record)
    assert message in str(client.fatal_error)


def test_event_queue_enforces_aggregate_byte_budget(tmp_path):
    client = PiRpcClient("pi", cwd=str(tmp_path), max_event_queue_bytes=20)
    client._dispatch({"type": "event", "payload": "x" * 100})
    assert "byte budget" in str(client.fatal_error)


def test_recursive_redaction_happens_before_queueing(tmp_path):
    client = PiRpcClient("pi", cwd=str(tmp_path))
    client._dispatch({"type": "event", "nested": {"token": "sk-" + "x" * 40}})
    queued = client.next_event(timeout=0)
    assert "sk-" + "x" * 40 not in json.dumps(queued)


def test_key_aware_redaction_hides_opaque_values_at_any_depth():
    opaque = "dGhpcy1pcy1ub3QtcmVnZXgtc2hhcGVk"
    value = {
        "outer": [
            {"Authorization": opaque},
            {"credentials": {"arbitrary": opaque}},
            {"provider_api_key": opaque},
            {"safe": opaque},
        ],
        "refresh-token": opaque,
    }
    redacted = redact_protocol_value(value)
    assert redacted["outer"][0]["Authorization"] == "<redacted>"
    assert redacted["outer"][1]["credentials"] == "<redacted>"
    assert redacted["outer"][2]["provider_api_key"] == "<redacted>"
    assert redacted["refresh-token"] == "<redacted>"
    assert redacted["outer"][3]["safe"] == opaque


@pytest.mark.parametrize(
    "key",
    [
        "AuthorizationHeader", "auth_value", "authentication-state",
        "providerCredentialValue", "client_secret_value", "API Key Value",
        "private-key-material", "opaqueAccessTokenPayload",
    ],
)
def test_pattern_key_redaction_hides_opaque_and_base64_credentials(key):
    opaque = "VGhpcyBpcyBhbiBvcGFxdWUgY3JlZGVudGlhbCB2YWx1ZQ=="
    assert redact_protocol_value({key: {"nested": opaque}})[key] == "<redacted>"


@pytest.mark.parametrize(
    "key", ["token_count", "token-counts", "prompt_tokens", "outputTokens", "max_tokens"]
)
def test_usage_count_keys_remain_observable(key):
    assert redact_protocol_value({key: 17})[key] == 17


@pytest.mark.parametrize(
    "hostile",
    [
        "VGhpcyBpcyBhbiBvcGFxdWUgY3JlZGVudGlhbA==",
        {"value": "VGhpcyBpcyBhbiBvcGFxdWUgY3JlZGVudGlhbA=="},
        ["VGhpcyBpcyBhbiBvcGFxdWUgY3JlZGVudGlhbA=="],
        True,
        float("inf"),
        float("nan"),
    ],
)
@pytest.mark.parametrize("key", ["token_count", "prompt_tokens", "outputTokens"])
def test_usage_keys_reject_non_numeric_or_non_finite_opaque_values(key, hostile):
    assert redact_protocol_value({key: hostile})[key] == "<redacted>"


def _fixture(mode: str) -> list[str]:
    return [sys.executable, str(Path("tests/fixtures/pi_rpc/fake_pi_rpc.py").resolve()), mode]


def test_real_subprocess_interleaves_event_and_response(tmp_path):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture("echo"))
    response = client.request("get_state", timeout=2)
    assert response["command"] == "get_state"
    assert client.next_event(timeout=1)["type"] == "agent_start"
    client.close()


@pytest.mark.parametrize("mode, expected", [("malformed", "invalid"), ("partial", "partial")])
@pytest.mark.live_system_guard_bypass
def test_malformed_or_partial_stdout_retires_process(tmp_path, mode, expected):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture(mode))
    client.start()
    deadline = time.monotonic() + 3
    while client.fatal_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert expected in str(client.fatal_error)
    while client.process is not None and client.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.process is not None and client.process.poll() is not None


@pytest.mark.live_system_guard_bypass
def test_request_timeout_is_uncertain_and_retires_process(tmp_path):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture("hang"))
    with pytest.raises(PiRequestUncertainError, match="uncertain"):
        client.request("prompt", timeout=0.05, message="once")
    assert client.process is not None and client.process.poll() is not None


@pytest.mark.live_system_guard_bypass
def test_stderr_tail_is_bounded_and_redacted(tmp_path):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture("stderr"))
    client.start()
    deadline = time.monotonic() + 3
    while client.process is not None and client.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    tail = client.stderr_tail
    assert len(tail.encode("utf-8")) <= 32 * 1024
    assert "sk-" + "s" * 40 not in tail


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
@pytest.mark.live_system_guard_bypass
def test_close_kills_descendant_process_group(tmp_path):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture("child"))
    client.start()
    child_pid = int(client.next_event(timeout=2)["pid"])
    client.close(force=True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("Pi descendant survived client close")


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
@pytest.mark.live_system_guard_bypass
def test_descendant_is_killed_when_direct_child_exits_first(tmp_path):
    client = PiRpcClient(cwd=str(tmp_path), launch_command=_fixture("child_exit"))
    client.start()
    child_pid = int(client.next_event(timeout=2)["pid"])
    deadline = time.monotonic() + 3
    while client.process is not None and client.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.process is not None and client.process.poll() is not None
    client.close(force=True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not psutil.pid_exists(child_pid):
            break
        try:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.02)
    else:
        pytest.fail("Pi descendant survived after the direct child exited")
