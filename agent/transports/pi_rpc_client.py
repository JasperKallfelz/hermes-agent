"""Strict, bounded JSONL client for the pinned Pi Coding Agent RPC protocol."""
from __future__ import annotations

import codecs
import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from agent.redact import redact_sensitive_text

PINNED_PI_VERSION = "0.84.3"
MAX_RECORD_BYTES = 1024 * 1024
MAX_EVENT_QUEUE = 256
MAX_EVENT_QUEUE_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024

_SENSITIVE_VALUE_KEYS = frozenset({
    "accesskeyid", "accesstoken", "apikey", "authorization", "authtoken",
    "awsaccesskeyid", "awssecretaccesskey", "bearertoken", "clientsecret",
    "credential", "credentials", "idtoken", "oauth", "oauthtoken",
    "password", "privatekey", "refreshtoken", "secret", "secretaccesskey",
    "sessiontoken", "token",
})

_SAFE_USAGE_KEYS = frozenset({
    "cache_read_tokens", "cache_write_tokens", "input_tokens", "max_tokens",
    "output_tokens", "prompt_tokens", "reasoning_tokens", "token_count",
    "token_counts", "total_tokens",
})
_SENSITIVE_KEY_TOKENS = frozenset({
    "auth", "authentication", "authorization", "credential", "credentials",
    "oauth", "password", "secret", "secrets", "token", "tokens",
})


def _finite_usage_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sensitive_value_key(key: object, value: object = None) -> bool:
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    underscored = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if underscored in _SAFE_USAGE_KEYS:
        # Usage labels are observable only for actual finite counters.  A
        # hostile protocol payload must not smuggle opaque strings/objects
        # through a credential-shaped key merely by naming it token_count.
        return not _finite_usage_number(value)
    tokens = tuple(part for part in underscored.split("_") if part)
    normalized = "".join(tokens)
    return (
        normalized in _SENSITIVE_VALUE_KEYS
        or bool(set(tokens) & _SENSITIVE_KEY_TOKENS)
        or ("api" in tokens and "key" in tokens)
        or ("private" in tokens and "key" in tokens)
        or normalized.endswith((
        "accesstoken", "apikey", "authorization", "authtoken", "bearertoken",
        "clientsecret", "password", "privatekey", "refreshtoken",
        "secretaccesskey", "sessiontoken",
        ))
    )

# Provider credentials are never inherited incidentally. The caller must opt
# individual names in, and the managed Pi config/auth directory is preferred.
_BASE_ENV = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "WINDIR",
}
_PROVIDER_ENV = frozenset({
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN",
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "XAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION",
})
APPROVED_PROVIDER_ENV = _PROVIDER_ENV


class PiRpcError(RuntimeError):
    """Pi process, timeout, or command failure."""


class PiRequestUncertainError(PiRpcError):
    """The command bytes were flushed but no valid response was observed."""


class PiProtocolError(PiRpcError):
    """The child violated the pinned JSONL RPC contract."""


def sanitized_pi_environment(
    source: Mapping[str, str] | None = None,
    *,
    allowed_provider_vars: set[str] | frozenset[str] | None = None,
    runtime_values: dict[str, str] | None = None,
) -> dict[str, str]:
    source_values: Mapping[str, str] = os.environ if source is None else source
    opted_in = set(allowed_provider_vars or ()) & set(_PROVIDER_ENV)
    allowed = _BASE_ENV | opted_in
    result = {name: value for name, value in source_values.items() if name in allowed}
    for name, value in (runtime_values or {}).items():
        if name in {
            "HOME", "USERPROFILE", "PI_CODING_AGENT_DIR",
            "PI_CODING_AGENT_SESSION_DIR", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK",
            "PI_TELEMETRY", "HERMES_PI_WORKSPACE",
        }:
            result[name] = str(value)
    result.setdefault("PI_SKIP_VERSION_CHECK", "1")
    result.setdefault("PI_TELEMETRY", "0")
    return result


def redact_protocol_text(value: object, *, limit: int = MAX_STDERR_BYTES) -> str:
    try:
        redacted = redact_sensitive_text(str(value), force=True, redact_url_credentials=True)
    except Exception:
        return "<redacted>"
    return redacted[-limit:]


def redact_protocol_value(value: Any, *, max_string: int = 64 * 1024) -> Any:
    """Recursively redact every string field before queueing or persistence."""
    if isinstance(value, str):
        return redact_protocol_text(value, limit=max_string)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in list(value.items())[:512]:
            safe_key = redact_protocol_text(key, limit=256)
            redacted[safe_key] = (
                "<redacted>"
                if _sensitive_value_key(key, item)
                else redact_protocol_value(item, max_string=max_string)
            )
        return redacted
    if isinstance(value, list):
        return [redact_protocol_value(item, max_string=max_string) for item in value[:512]]
    if isinstance(value, tuple):
        return [redact_protocol_value(item, max_string=max_string) for item in value[:512]]
    return value


class LiteralLfJsonDecoder:
    """Incremental UTF-8 decoder splitting records on the literal LF byte."""

    def __init__(self, max_record_bytes: int = MAX_RECORD_BYTES) -> None:
        self.max_record_bytes = max_record_bytes
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        if not chunk:
            return []
        self._buffer.extend(chunk)
        records: list[dict[str, Any]] = []
        while True:
            boundary = self._buffer.find(b"\n")
            if boundary < 0:
                if len(self._buffer) > self.max_record_bytes:
                    raise PiProtocolError("Pi RPC record exceeds size limit")
                break
            raw = bytes(self._buffer[:boundary])
            del self._buffer[:boundary + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            if not raw:
                continue
            if len(raw) > self.max_record_bytes:
                raise PiProtocolError("Pi RPC record exceeds size limit")
            try:
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                text = decoder.decode(raw, final=True)
                value = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PiProtocolError(f"invalid Pi RPC JSONL record: {exc}") from exc
            if not isinstance(value, dict):
                raise PiProtocolError("Pi RPC records must be JSON objects")
            records.append(value)
        return records

    def finish(self) -> None:
        if self._buffer:
            raise PiProtocolError("partial Pi RPC record at process exit")


@dataclass
class _Waiter:
    command: str
    inbox: queue.Queue[object]


class PiRpcClient:
    """One authenticated Pi RPC subprocess with strict request correlation."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        runtime_env: dict[str, str] | None = None,
        allowed_provider_vars: set[str] | frozenset[str] | None = None,
        extra_args: list[str] | None = None,
        launch_command: list[str] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        max_event_queue: int = MAX_EVENT_QUEUE,
        max_event_queue_bytes: int = MAX_EVENT_QUEUE_BYTES,
    ) -> None:
        if launch_command is None and not executable:
            raise ValueError("executable or launch_command is required")
        self.executable = str(Path(executable).expanduser()) if executable else ""
        self.launch_command = list(launch_command) if launch_command else None
        self.cwd = str(Path(cwd).resolve(strict=True))
        self.env = sanitized_pi_environment(
            env, allowed_provider_vars=allowed_provider_vars, runtime_values=runtime_env
        )
        self.extra_args = list(extra_args or [])
        self.event_callback = event_callback
        self.events: queue.Queue[tuple[dict[str, Any], int, int]] = queue.Queue(maxsize=max_event_queue)
        self.max_event_queue_bytes = max_event_queue_bytes
        self._event_queue_bytes = 0
        self._responses: dict[str, _Waiter] = {}
        self._completed_ids: deque[str] = deque(maxlen=1024)
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stderr = bytearray()
        self._fatal: Exception | None = None
        self._record_sequence = 0
        self._closed = False
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def stderr_tail(self) -> str:
        with self._lock:
            data = bytes(self._stderr)
        return redact_protocol_text(data.decode("utf-8", errors="replace"))

    @property
    def fatal_error(self) -> Exception | None:
        return self._fatal

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if self._closed:
            raise PiRpcError("Pi RPC client is closed")
        args = self.launch_command or [
            self.executable,
            "--mode", "rpc", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-themes", "--no-context-files",
            "--no-approve", "--offline", *self.extra_args,
        ]
        try:
            self.process = subprocess.Popen(
                args, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                start_new_session=(os.name != "nt"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
        except OSError as exc:
            raise PiRpcError(f"could not launch managed Pi runtime: {exc}") from exc
        threading.Thread(target=self._read_stdout, daemon=True, name="pi-rpc-stdout").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="pi-rpc-stderr").start()

    def _set_fatal(self, exc: Exception) -> None:
        with self._lock:
            if self._fatal is not None:
                return
            safe = PiProtocolError(redact_protocol_text(exc)) if isinstance(exc, PiProtocolError) else PiRpcError(redact_protocol_text(exc))
            self._fatal = safe
            waiters = list(self._responses.values())
        for waiter in waiters:
            try:
                waiter.inbox.put_nowait(safe)
            except queue.Full:
                pass
        self._terminate_process_tree(graceful=False)

    def _read_stdout(self) -> None:
        decoder = LiteralLfJsonDecoder()
        try:
            assert self.process is not None and self.process.stdout is not None
            while True:
                chunk = self.process.stdout.read(64 * 1024)
                if not chunk:
                    decoder.finish()
                    if not self._closed:
                        self._set_fatal(PiRpcError("Pi RPC stdout closed unexpectedly"))
                    return
                for record in decoder.feed(chunk):
                    self._dispatch(record)
        except Exception as exc:
            self._set_fatal(exc if isinstance(exc, Exception) else PiRpcError(str(exc)))

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            chunk = self.process.stderr.read(4096)
            if not chunk:
                return
            with self._lock:
                self._stderr.extend(chunk)
                if len(self._stderr) > MAX_STDERR_BYTES:
                    del self._stderr[:-MAX_STDERR_BYTES]

    def _dispatch(self, raw_record: dict[str, Any]) -> None:
        record = redact_protocol_value(raw_record)
        with self._lock:
            self._record_sequence += 1
            sequence = self._record_sequence
        record_type = record.get("type")
        if record_type == "response":
            request_id = record.get("id")
            command = record.get("command")
            if not isinstance(request_id, str) or not request_id:
                self._set_fatal(PiProtocolError("Pi response omitted its request id"))
                return
            with self._lock:
                waiter = self._responses.get(request_id)
                duplicate = request_id in self._completed_ids
            if duplicate:
                self._set_fatal(PiProtocolError("Pi returned a duplicate response id"))
                return
            if waiter is None:
                self._set_fatal(PiProtocolError("Pi returned a response for unknown request id"))
                return
            if command != waiter.command:
                self._set_fatal(PiProtocolError("Pi response command did not match its request"))
                return
            with self._lock:
                self._completed_ids.append(request_id)
            record["_hermes_sequence"] = sequence
            try:
                waiter.inbox.put_nowait(record)
            except queue.Full:
                self._set_fatal(PiProtocolError("duplicate Pi response for active request"))
            return

        try:
            size = len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            self._set_fatal(PiProtocolError(f"Pi event was not JSON serializable: {exc}"))
            return
        with self._lock:
            if self._event_queue_bytes + size > self.max_event_queue_bytes:
                self._set_fatal(PiProtocolError("Pi RPC total event queue byte budget exceeded"))
                return
            self._event_queue_bytes += size
        try:
            self.events.put_nowait((record, size, sequence))
        except queue.Full:
            with self._lock:
                self._event_queue_bytes -= size
            self._set_fatal(PiProtocolError("Pi RPC event queue count limit exceeded"))
            return
        if self.event_callback:
            try:
                self.event_callback(record)
            except Exception:
                # Display callbacks cannot corrupt the protocol reader.
                pass

    def _send(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > MAX_RECORD_BYTES:
            raise PiProtocolError("Pi RPC command exceeds size limit")
        assert self.process is not None and self.process.stdin is not None
        with self._write_lock:
            self.process.stdin.write(encoded)
            self.process.stdin.flush()

    def request(
        self,
        command_type: str,
        *,
        timeout: float = 30.0,
        request_id: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        self.start()
        if self._fatal:
            raise PiRpcError(str(self._fatal))
        rid = request_id or uuid.uuid4().hex
        waiter = _Waiter(command_type, queue.Queue(maxsize=1))
        with self._lock:
            if rid in self._responses or rid in self._completed_ids:
                raise PiProtocolError("Pi RPC request id was reused")
            self._responses[rid] = waiter
        sent = False
        try:
            # Once the write begins, a short write/flush failure cannot prove
            # that Pi saw zero bytes. Classify the whole transmission window as
            # uncertain; conservative reconciliation is safer than replay.
            sent = True
            self._send({"id": rid, "type": command_type, **params})
            response = waiter.inbox.get(timeout=max(0.01, timeout))
            if isinstance(response, Exception):
                error_cls = PiRequestUncertainError if sent else PiRpcError
                raise error_cls(str(response)) from response
            if not isinstance(response, dict):
                raise PiProtocolError("Pi RPC response waiter received a non-object")
            typed_response = cast(dict[str, Any], response)
            if typed_response.get("success") is not True:
                raise PiRpcError(
                    redact_protocol_text(typed_response.get("error") or typed_response)
                )
            return typed_response
        except queue.Empty as exc:
            error: PiRpcError
            if sent:
                error = PiRequestUncertainError(f"Pi RPC request {command_type!r} timed out after acceptance became uncertain")
            else:
                error = PiRpcError(f"Pi RPC request {command_type!r} timed out")
            self._set_fatal(error)
            raise error from exc
        except (BrokenPipeError, OSError) as exc:
            error = PiRequestUncertainError("Pi RPC pipe failed after command transmission") if sent else PiRpcError("Pi RPC pipe failed before command transmission")
            self._set_fatal(error)
            raise error from exc
        finally:
            with self._lock:
                self._responses.pop(rid, None)

    def send_extension_ui_response(self, request_id: str, **response: Any) -> None:
        if not request_id:
            raise PiProtocolError("extension UI response requires an id")
        self._send({"type": "extension_ui_response", "id": request_id, **response})

    def next_event_with_sequence(self, timeout: float = 1.0) -> tuple[dict[str, Any], int]:
        if self._fatal:
            raise PiRpcError(str(self._fatal))
        try:
            record, size, sequence = self.events.get(timeout=timeout)
        except queue.Empty:
            if self._fatal:
                raise PiRpcError(str(self._fatal))
            if self.process and self.process.poll() is not None:
                error = PiRpcError(f"Pi exited with status {self.process.returncode}: {self.stderr_tail}")
                self._set_fatal(error)
                raise error
            raise
        with self._lock:
            self._event_queue_bytes = max(0, self._event_queue_bytes - size)
        return record, sequence

    def next_event(self, timeout: float = 1.0) -> dict[str, Any]:
        record, _sequence = self.next_event_with_sequence(timeout=timeout)
        return record

    def drain_events(self, *, limit: int = MAX_EVENT_QUEUE) -> list[dict[str, Any]]:
        drained: list[dict[str, Any]] = []
        for _ in range(limit):
            try:
                drained.append(self.next_event(timeout=0))
            except queue.Empty:
                break
        return drained

    def _terminate_process_tree(self, *, graceful: bool) -> None:
        process = self.process
        if process is None:
            return
        try:
            if graceful and process.poll() is None and process.stdin:
                process.stdin.close()
                process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            if os.name == "nt":
                # A direct child may exit after closing stdout while one of its
                # descendants remains alive.  taskkill /T must therefore run
                # even when poll() already reports that the parent exited.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=5, check=False,
                )
            else:
                # Even when Pi itself honored stdin EOF, descendants may still
                # hold the process group. Probe and reap the whole group.
                os.killpg(process.pid, 0)
                os.killpg(process.pid, signal.SIGTERM)
                group_deadline = time.monotonic() + 2.0
                while time.monotonic() < group_deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self, *, force: bool = False) -> None:
        self._closed = True
        self._terminate_process_tree(graceful=not force)

    def is_alive(self) -> bool:
        return bool(self.process and self.process.poll() is None and self._fatal is None)
