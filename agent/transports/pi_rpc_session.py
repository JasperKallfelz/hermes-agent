"""Persistent, crash-aware Pi RPC session state machine."""
from __future__ import annotations

import hashlib
import queue
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agent.external_runtime.binding import (
    BindingError,
    RuntimeBinding,
    UNSETTLED_PROMPT_STATES,
    load_binding,
    prompt_digest,
    save_binding,
    save_json_record,
    worktree_identity,
)
from agent.external_runtime.types import CompactionResult, ExternalTurnResult, RuntimeState, TurnStatus
from agent.transports.pi_docker_sandbox import PiDockerSandbox
from agent.transports.pi_event_projector import PiEventProjector
from agent.transports.pi_rpc_client import (
    PINNED_PI_VERSION,
    PiProtocolError,
    PiRequestUncertainError,
    PiRpcClient,
    PiRpcError,
    redact_protocol_text,
)


class PiSessionState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    RUNNING = "running"
    COMPACTING = "compacting"
    ABORTING = "aborting"
    DEAD = "dead"
    RESPAWNING = "respawning"
    RECONCILING = "reconciling"


class PiReconciliationRequired(PiRpcError):
    """A prior prompt has uncertain acceptance and must not be replayed."""


_READ_ONLY_MODES = frozenset({"review", "inspect", "plan"})


class PiRpcSession:
    def __init__(
        self,
        *,
        executable: str | None,
        cwd: str,
        config_dir: str | Path | None = None,
        session_dir: str | Path | None = None,
        binding_path: str | Path | None = None,
        client_factory: Callable[..., Any] = PiRpcClient,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        approval_handler: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        silence_timeout: float = 300.0,
        turn_timeout: float = 3600.0,
        mode: str = "inspect",
        sandbox: PiDockerSandbox | None = None,
        allowed_provider_vars: set[str] | frozenset[str] | None = None,
        native_session_id: str | None = None,
    ) -> None:
        self.executable = executable
        self.cwd = str(Path(cwd).resolve(strict=True))
        self.config_dir = Path(config_dir or (Path(self.cwd) / ".hermes-pi-config")).resolve()
        self.session_dir = Path(session_dir or (Path(self.cwd) / ".hermes-pi-sessions")).resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.binding_path = Path(binding_path or (self.session_dir / "binding.json"))
        self.client_factory = client_factory
        self.event_callback = event_callback
        self.approval_handler = approval_handler
        self.silence_timeout = max(0.05, silence_timeout)
        self.turn_timeout = max(self.silence_timeout, turn_timeout)
        self.mode = mode
        if mode not in _READ_ONLY_MODES | {"implement"}:
            raise ValueError("Pi mode must be implement, review, inspect, or plan")
        if mode == "implement" and sandbox is None:
            raise PiRpcError("write-capable Pi sessions require the hard Docker sandbox")
        self.sandbox = sandbox
        if self.sandbox:
            if binding_path is None:
                raise PiRpcError("contained Pi sessions require an explicit host-only binding path")
            binding_target = self.binding_path.expanduser().resolve()
            mounted_roots = (
                self.sandbox.worktree, self.sandbox.config_dir, self.sandbox.session_dir,
            )
            if any(
                binding_target == root or binding_target.is_relative_to(root)
                for root in mounted_roots
            ):
                raise PiRpcError("Hermes Pi binding must be outside every container mount")
            self.binding_path = binding_target
        self.allowed_provider_vars = frozenset(allowed_provider_vars or ())
        self.native_session_id = native_session_id or uuid.uuid4().hex
        self.binding: RuntimeBinding | None = None
        if self.binding_path.is_file():
            self.binding = load_binding(self.binding_path)
            self.native_session_id = self.binding.native_session_id
        self.client: PiRpcClient | None = None
        self.state = PiSessionState.STARTING
        self.native_state: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._interrupt = threading.Event()
        self._generation = self.binding.turn_generation if self.binding else 0
        self._active_generation: int | None = None
        # Fresh native completion state is intentionally process-local.  The
        # binding records ``native_completed`` while retaining Hermes' prior
        # canonical cursor.  If the process dies before the host durability
        # fence, exact projection data is gone and explicit reconciliation is
        # required instead of guessing from the native transcript.
        self._completed_native_state: RuntimeState | None = None

    @property
    def read_only(self) -> bool:
        return self.mode in _READ_ONLY_MODES

    def _host_session_path(self, native: str | None) -> str | None:
        if not native:
            return None
        path = Path(native)
        if self.sandbox and path.is_absolute() and path.is_relative_to(Path("/pi-sessions")):
            relative = path.relative_to("/pi-sessions")
            return str((self.session_dir / relative).resolve())
        return str(path.expanduser().resolve())

    def _native_session_path(self, host: str) -> str:
        path = Path(host).resolve(strict=True)
        if self.sandbox:
            return str(Path("/pi-sessions") / path.relative_to(self.session_dir.resolve(strict=True)))
        return str(path)

    def _extension_path(self) -> Path:
        candidate = self.config_dir / "hermes_bridge.ts"
        if candidate.is_file():
            return candidate
        packaged = Path(__file__).with_name("pi_assets") / "hermes_bridge.ts"
        return packaged.resolve(strict=True)

    def _pi_args(self) -> list[str]:
        extension = self._extension_path()
        if self.sandbox:
            extension_arg = "/pi-config/hermes_bridge.ts"
            session_dir_arg = "/pi-sessions"
        else:
            extension_arg = str(extension)
            session_dir_arg = str(self.session_dir)
        args = [
            "--mode", "rpc", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-themes", "--no-context-files",
            "--no-approve", "--offline", "--session-dir", session_dir_arg,
            "--extension", extension_arg,
        ]
        # Model-issued shell is intentionally unavailable in every Pi mode.
        # The model can use only structured, path-mediated file operations;
        # independent verification runs through a separate no-network profile.
        args.extend(["--tools", "read,edit,write"])
        if self.binding:
            args.extend(["--session", self._native_session_path(self.binding.native_session_file)])
        else:
            args.extend(["--session-id", self.native_session_id])
        return args

    def _new_client(self) -> PiRpcClient:
        pi_args = self._pi_args()
        runtime_env = {
            "HOME": str(self.config_dir.parent / "home"),
            "PI_CODING_AGENT_DIR": str(self.config_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(self.session_dir),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "HERMES_PI_WORKSPACE": self.cwd,
        }
        kwargs: dict[str, Any] = {
            "cwd": self.cwd,
            "event_callback": self.event_callback,
            "allowed_provider_vars": set(self.allowed_provider_vars),
        }
        if self.sandbox:
            self.sandbox.prepare_egress()
            kwargs["launch_command"] = self.sandbox.pi_launch_command(pi_args)
        else:
            kwargs.update({
                "executable": self.executable,
                "extra_args": pi_args[2:],  # client adds --mode rpc
                "runtime_env": runtime_env,
            })
        return self.client_factory(**kwargs)

    def _query_state(self) -> RuntimeState:
        assert self.client is not None
        self.native_state = self.client.request("get_state")
        entries = self.client.request("get_entries")
        data = self.native_state.get("data") or {}
        entry_data = entries.get("data") or {}
        if isinstance(data, dict):
            data = dict(data)
            data["leafId"] = entry_data.get("leafId") if isinstance(entry_data, dict) else None
        self.native_state = {"data": data}
        return self._runtime_state()

    def _validate_binding_identity(self, *, allow_uncertain: bool = False) -> RuntimeBinding:
        if self.binding is None:
            raise PiReconciliationRequired("no persisted Pi binding is available to reconcile")
        try:
            return self.binding.validate(
                assigned_cwd=self.cwd,
                session_root=self.session_dir,
                expected_runtime="pi_rpc",
                compatible_versions={PINNED_PI_VERSION},
                allow_uncertain=allow_uncertain,
            )
        except BindingError as exc:
            self.state = PiSessionState.RECONCILING
            raise PiReconciliationRequired(str(exc)) from exc

    def _validate_binding_host_ownership(self) -> RuntimeBinding:
        if self.binding is None:
            raise PiReconciliationRequired("no persisted Pi binding is available to reconcile")
        try:
            return self.binding.validate_host_ownership(
                assigned_cwd=self.cwd,
                session_root=self.session_dir,
                expected_runtime="pi_rpc",
                compatible_versions={PINNED_PI_VERSION},
                allow_uncertain=True,
            )
        except BindingError as exc:
            self.state = PiSessionState.RECONCILING
            raise PiReconciliationRequired(str(exc)) from exc

    def _start_bound_for_reconciliation(self) -> tuple[RuntimeState, dict[str, Any]]:
        binding = self._validate_binding_identity(allow_uncertain=True)
        if binding.prompt_state not in UNSETTLED_PROMPT_STATES:
            raise PiRpcError("the persisted Pi binding does not require reconciliation")
        self.state = PiSessionState.RECONCILING
        if self.client is None or not self.client.is_alive():
            self.client = self._new_client()
            self.client.start()
        assert self.client is not None
        self.native_state = self.client.request("get_state")
        entries = self.client.request("get_entries")
        data = self.native_state.get("data") or {}
        entry_data = entries.get("data") or {}
        if isinstance(data, dict):
            data = dict(data)
            data["leafId"] = entry_data.get("leafId") if isinstance(entry_data, dict) else None
        self.native_state = {"data": data}
        state = self._runtime_state()
        if state.session_id != binding.native_session_id:
            self._retire("Pi returned a foreign native session id during reconciliation")
            raise PiProtocolError("Pi returned a foreign native session id during reconciliation")
        if state.session_file != binding.native_session_file:
            self._retire("Pi returned a foreign session file during reconciliation")
            raise PiProtocolError("Pi returned a foreign session file during reconciliation")
        return state, entry_data if isinstance(entry_data, dict) else {}

    @staticmethod
    def _entry_summary(entry_data: dict[str, Any]) -> dict[str, Any]:
        raw_entries = entry_data.get("entries")
        entries = raw_entries if isinstance(raw_entries, list) else []
        tail: list[dict[str, Any]] = []
        for raw in entries[-10:]:
            if not isinstance(raw, dict):
                continue
            tail.append({
                "id": raw.get("id") or raw.get("entryId"),
                "type": raw.get("type"),
                "role": raw.get("role"),
            })
        return {"count": len(entries), "tail": tail}

    @classmethod
    def abandon_binding(
        cls,
        *,
        cwd: str | Path,
        session_dir: str | Path,
        binding_path: str | Path,
    ) -> dict[str, Any]:
        """Archive an uncertain binding using host metadata only.

        This control-plane operation intentionally does not construct a Pi
        client or sandbox, provision files, inspect an image, or invoke Docker.
        """
        assigned_cwd = str(Path(cwd).resolve(strict=True))
        session_root = Path(session_dir).resolve(strict=True)
        target = Path(binding_path)
        try:
            binding = load_binding(target).validate_host_ownership(
                assigned_cwd=assigned_cwd,
                session_root=session_root,
                expected_runtime="pi_rpc",
                compatible_versions={PINNED_PI_VERSION},
                allow_uncertain=True,
            )
        except (BindingError, OSError) as exc:
            raise PiReconciliationRequired(str(exc)) from exc
        if binding.prompt_state not in UNSETTLED_PROMPT_STATES:
            raise PiRpcError("the persisted Pi binding does not require reconciliation")
        try:
            original = target.read_bytes()
        except OSError as exc:
            raise PiReconciliationRequired(str(exc)) from exc
        archive = target.with_name(
            f"{target.name}.abandoned-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        )
        os.replace(target, archive)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return {
            "action": "abandon",
            "runtime": "pi_rpc",
            "session_id": binding.native_session_id,
            "session_file": binding.native_session_file,
            "cursor": binding.last_entry_id,
            "previous_cursor": binding.last_entry_id,
            "prompt_state": "abandoned",
            "archived_binding": archive.name,
            "archived_binding_sha256": hashlib.sha256(original).hexdigest(),
            "slot_bound": False,
            "context_loss": True,
            "requires_independent_approval": True,
            "warning": (
                "CONTEXT LOSS: the complete prior binding was archived, but native "
                "conversation continuity is intentionally discarded. The uncertain "
                "prompt was not replayed. Independent approval is required."
            ),
            "next_step": (
                "A fresh native session will be allocated on the next run; the archived "
                "native conversation will not be resumed."
            ),
            "replayed": False,
        }

    def reconcile(self, action: str) -> dict[str, Any]:
        """Inspect, adopt, or abandon one uncertain native prompt without replay."""
        normalized = str(action or "").strip().lower()
        if normalized not in {"inspect", "adopt", "abandon"}:
            raise ValueError("Pi reconciliation action must be inspect, adopt, or abandon")
        with self._lock:
            if normalized == "abandon":
                report = self.abandon_binding(
                    cwd=self.cwd,
                    session_dir=self.session_dir,
                    binding_path=self.binding_path,
                )
                # Archival is the authoritative no-replay transition.  Once it
                # lands, retire any live client/container ownership before the
                # slot can allocate a fresh native session.
                ownership_retired = self._retire("explicit archival abandon")
                self.binding = None
                self.native_session_id = uuid.uuid4().hex
                self.native_state = {}
                self._completed_native_state = None
                self._generation = 0
                self._active_generation = None
                self.state = PiSessionState.STARTING
                report["runtime_ownership_retired"] = ownership_retired
                if not ownership_retired:
                    report["cleanup_warning"] = (
                        "Old Pi runtime cleanup could not be confirmed. Do not trust a fresh "
                        "allocation until the labeled stale-resource sweep succeeds."
                    )
                    report["warning"] += " " + report["cleanup_warning"]
                return report

            state, entry_data = self._start_bound_for_reconciliation()
            assert self.binding is not None
            observed = {
                "action": normalized,
                "runtime": "pi_rpc",
                "session_id": state.session_id,
                "session_file": state.session_file,
                "cursor": state.cursor,
                "model": self._model_dict(state),
                "thinking_level": state.thinking_level,
                "is_streaming": state.is_streaming,
                "is_compacting": state.is_compacting,
                "pending_message_count": state.pending_message_count,
                "previous_cursor": self.binding.last_entry_id,
                "prompt_state": self.binding.prompt_state,
                "entry_summary": self._entry_summary(entry_data),
                "cursor_advanced": state.cursor != self.binding.last_entry_id,
                "adopt_available": state.cursor == self.binding.last_entry_id,
                "replayed": False,
            }
            if normalized == "inspect":
                return observed
            if normalized == "adopt":
                if state.is_streaming or state.is_compacting or state.pending_message_count:
                    raise PiReconciliationRequired(
                        "native Pi session is still streaming, compacting, or has pending messages"
                    )
                raw_entries = entry_data.get("entries")
                entries = raw_entries if isinstance(raw_entries, list) else []
                entry_ids = {
                    str(item.get("id") or item.get("entryId"))
                    for item in entries if isinstance(item, dict)
                    and (item.get("id") or item.get("entryId"))
                }
                previous = self.binding.last_entry_id
                if previous and previous != state.cursor and previous not in entry_ids:
                    raise PiReconciliationRequired(
                        "native Pi cursor lineage does not contain the previously persisted cursor"
                    )
                if state.cursor and state.cursor not in entry_ids:
                    raise PiReconciliationRequired(
                        "native Pi current cursor is absent from the durable entry lineage"
                    )
                if state.cursor != previous:
                    raise PiReconciliationRequired(
                        "Pi durable entries advanced beyond Hermes' canonical checkpoint; "
                        "adopt is unavailable without canonical transcript recovery. "
                        "Use explicit archival abandon only if native continuity may be lost"
                    )
                self._persist_prompt_state(
                    "settled", request_id=None, digest=None, state=state
                )
                self.state = PiSessionState.IDLE
                observed["prompt_state"] = "settled"
                return observed

            raise AssertionError("unreachable reconciliation action")

    def ensure_started(self) -> RuntimeState:
        with self._lock:
            if self.client is not None and self.client.is_alive():
                return self._runtime_state()
            if self.binding:
                self._validate_binding_identity()
            self.client = self._new_client()
            self.client.start()
            state = self._query_state()
            if state.session_id != self.native_session_id:
                self._retire("Pi returned a foreign native session id")
                raise PiProtocolError("Pi returned a foreign native session id")
            if self.binding:
                if state.session_file != self.binding.native_session_file:
                    self._retire("Pi resumed a different session file")
                    raise PiProtocolError("Pi resumed a different session file")
                if state.cursor != self.binding.last_entry_id:
                    self._retire("Pi durable cursor changed while Hermes was offline")
                    raise PiReconciliationRequired("Pi durable cursor changed; explicit reconciliation is required")
                if self._model_dict(state) != self.binding.model:
                    self._retire("Pi model changed while Hermes was offline")
                    raise PiReconciliationRequired("Pi model changed; explicit reconciliation is required")
                if state.thinking_level != self.binding.thinking_level:
                    self._retire("Pi thinking level changed while Hermes was offline")
                    raise PiReconciliationRequired(
                        "Pi thinking level changed; explicit reconciliation is required"
                    )
            else:
                if not state.session_file:
                    self._retire("Pi did not create a persistent session file")
                    raise PiProtocolError("Pi did not create a persistent session file")
                # Pi reports its future session path immediately but removes an
                # empty transcript at shutdown. Do not leave an orphan binding
                # merely because the process was inspected/model-listed. The
                # sending fence below creates the binding before any prompt
                # bytes cross stdin; a completed prompt makes the file durable.
            self.state = PiSessionState.IDLE
            return self._runtime_state()

    @staticmethod
    def _model_dict(state: RuntimeState) -> dict[str, str] | None:
        if not state.model_provider and not state.model_id:
            return None
        return {"provider": state.model_provider or "", "id": state.model_id or ""}

    def _runtime_state(self) -> RuntimeState:
        data = self.native_state.get("data") or self.native_state.get("state") or self.native_state
        if not isinstance(data, dict):
            data = {}
        model = data.get("model") or {}
        if not isinstance(model, dict):
            model = {}
        native_file = data.get("sessionFile") or data.get("session_file")
        return RuntimeState(
            runtime="pi_rpc", lifecycle=self.state.value,
            session_id=data.get("sessionId") or data.get("session_id"),
            session_file=self._host_session_path(native_file),
            cursor=data.get("lastEntryId") or data.get("leafId"),
            model_provider=model.get("provider"), model_id=model.get("id"),
            thinking_level=data.get("thinkingLevel"),
            is_streaming=bool(data.get("isStreaming")),
            is_compacting=bool(data.get("isCompacting")),
            pending_message_count=max(0, int(data.get("pendingMessageCount") or 0)),
            raw=data,
        )

    def get_state(self) -> RuntimeState:
        self.ensure_started()
        return self._query_state()

    def _persist_prompt_state(
        self,
        prompt_state: str,
        *,
        request_id: str | None,
        digest: str | None,
        state: RuntimeState | None = None,
    ) -> None:
        current = state
        if self.binding is None:
            if current is None:
                raise PiProtocolError("fresh Pi session state is required to create a binding")
            if not current.session_file or not current.session_id:
                raise PiProtocolError("Pi session identity was not initialized")
            # Settled control-only state on an empty Pi session is ephemeral;
            # avoid persisting a binding to a file Pi will delete on shutdown.
            if prompt_state == "settled" and not Path(current.session_file).is_file():
                return
            candidate = RuntimeBinding(
                runtime="pi_rpc",
                native_session_file=current.session_file,
                native_session_id=current.session_id,
                last_entry_id=current.cursor,
                cwd=self.cwd,
                runtime_version=PINNED_PI_VERSION,
                worktree=worktree_identity(self.cwd),
                model=self._model_dict(current),
                thinking_level=current.thinking_level,
                prompt_state=prompt_state,
                prompt_request_id=request_id,
                prompt_digest=digest,
                turn_generation=self._generation,
            )
        else:
            candidate = self.binding
        updates: dict[str, Any] = dict(
            prompt_state=prompt_state,
            prompt_request_id=request_id,
            prompt_digest=digest,
            turn_generation=self._generation,
        )
        if current is not None:
            updates.update(
                native_session_file=current.session_file or candidate.native_session_file,
                native_session_id=current.session_id or candidate.native_session_id,
                last_entry_id=current.cursor,
                model=self._model_dict(current),
                thinking_level=current.thinking_level,
            )
        candidate = candidate.with_updates(**updates)
        save_binding(self.binding_path, candidate)
        self.binding = candidate

    def _handle_extension_ui(self, event: dict[str, Any]) -> None:
        assert self.client is not None
        method = str(event.get("method") or "")
        if method not in {"select", "confirm", "input", "editor"}:
            return
        response = self.approval_handler(event) if self.approval_handler else None
        if not isinstance(response, dict):
            response = {"confirmed": False} if method == "confirm" else {"cancelled": True}
        self.client.send_extension_ui_response(str(event.get("id") or ""), **response)

    def run_turn(self, user_input: object) -> ExternalTurnResult:
        projected: list[dict[str, Any]] = []
        final_text = ""
        tools = 0
        request_id: str | None = None
        digest: str | None = None
        possible_acceptance = False
        self._completed_native_state = None
        if not isinstance(user_input, str):
            return ExternalTurnResult(
                status=TurnStatus.ERROR,
                error="Pi RPC does not support multimodal prompt content",
                should_retire=False,
                phase="before_send",
                prompt_accepted=False,
            )
        if self.binding is not None and self.binding.prompt_state in UNSETTLED_PROMPT_STATES:
            return ExternalTurnResult(
                status=TurnStatus.RECONCILIATION_REQUIRED,
                error="the previous Pi prompt has not crossed Hermes' persistence fence",
                should_retire=False,
                reconciliation_required=True,
                phase="before_send",
                prompt_accepted=False,
            )
        try:
            self.ensure_started()
        except PiReconciliationRequired as exc:
            return ExternalTurnResult(
                status=TurnStatus.RECONCILIATION_REQUIRED,
                error=redact_protocol_text(exc), should_retire=True,
                reconciliation_required=True,
                phase="before_send", prompt_accepted=False,
            )
        except Exception as exc:
            self._retire(redact_protocol_text(exc))
            return ExternalTurnResult(
                status=TurnStatus.ERROR,
                error=redact_protocol_text(exc),
                should_retire=True,
                phase="before_send", prompt_accepted=False,
            )
        assert self.client is not None
        stale = self.client.drain_events()
        # A prior settled marker cannot complete this generation. Draining the
        # whole idle queue makes the generation boundary explicit.
        _ = stale
        projector = PiEventProjector()
        saw_agent_start = False
        interrupted = False
        self.state = PiSessionState.RUNNING
        self._interrupt.clear()
        self._generation += 1
        request_id = f"prompt-{self._generation}-{uuid.uuid4().hex}"
        digest = prompt_digest(user_input)
        started_at = time.monotonic()
        try:
            # Persist the pre-transmission fence before a single prompt byte can
            # cross stdin. Any failure here is a deterministic pre-send ERROR.
            self._persist_prompt_state(
                "sending", request_id=request_id, digest=digest,
                state=self._runtime_state(),
            )
            # PiRpcClient conservatively treats the entire request call as a
            # possible transmission window. From here onward Hermes never
            # replays automatically, even when a later binding write fails.
            possible_acceptance = True
            accepted = self.client.request(
                "prompt", request_id=request_id, timeout=min(30.0, self.turn_timeout),
                message=user_input,
            )
            response_sequence = int(accepted.pop("_hermes_sequence", 0) or 0)
            self._persist_prompt_state(
                "accepted", request_id=request_id, digest=digest,
                state=self._runtime_state(),
            )
            with self._lock:
                self._active_generation = self._generation
            silence_deadline = time.monotonic() + self.silence_timeout
            wall_deadline = started_at + self.turn_timeout
            while True:
                now = time.monotonic()
                if now >= wall_deadline:
                    raise PiRequestUncertainError("Pi turn wall-clock deadline exceeded after prompt acceptance")
                if self._interrupt.is_set():
                    self.state = PiSessionState.ABORTING
                    self.client.request("abort", timeout=min(10.0, wall_deadline - now))
                    self._interrupt.clear()
                    interrupted = True
                try:
                    next_with_sequence = getattr(self.client, "next_event_with_sequence", None)
                    if callable(next_with_sequence):
                        event, event_sequence = next_with_sequence(
                            timeout=min(0.25, wall_deadline - now)
                        )
                        if event_sequence <= response_sequence:
                            continue
                    else:
                        event = self.client.next_event(timeout=min(0.25, wall_deadline - now))
                except queue.Empty:
                    if time.monotonic() >= silence_deadline:
                        raise PiRequestUncertainError("Pi RPC event silence deadline exceeded after prompt acceptance")
                    continue
                silence_deadline = time.monotonic() + self.silence_timeout
                if event.get("type") == "extension_ui_request":
                    self._handle_extension_ui(event)
                if event.get("type") == "agent_start":
                    saw_agent_start = True
                projection = projector.project(event)
                projected.extend(projection.messages)
                if projection.final_text is not None:
                    final_text = projection.final_text
                tools += int(projection.tool_iteration)
                if event.get("type") == "agent_settled":
                    if not saw_agent_start:
                        # Stale settle from a prior generation; never a fence.
                        continue
                    with self._lock:
                        self._active_generation = None
                    break
            state = self._query_state()
            self._completed_native_state = state
            # This durable marker is not settlement.  It retains the prior
            # canonical cursor/model/thinking fields while proving that the
            # in-process finalizer may use ``_completed_native_state`` after
            # host persistence succeeds.
            self._persist_prompt_state(
                "native_completed", request_id=request_id, digest=digest,
            )
            usage = projector.usage
            cost = usage.get("cost") or {}
            return ExternalTurnResult(
                final_text=final_text, projected_messages=projected,
                status=TurnStatus.INTERRUPTED if interrupted else TurnStatus.COMPLETED,
                session_id=state.session_id, session_file=state.session_file,
                cursor=state.cursor, run_id=f"{state.session_id}:{self._generation}",
                tool_iterations=tools, compacted=projector.compacted,
                prompt_tokens=int(usage.get("input") or 0),
                output_tokens=int(usage.get("output") or 0),
                cache_read_tokens=int(usage.get("cacheRead") or 0),
                cache_write_tokens=int(usage.get("cacheWrite") or 0),
                cost=float(cost.get("total") or 0.0) if isinstance(cost, dict) else 0.0,
                phase="after_send", prompt_accepted=True,
            )
        except Exception as exc:
            self._completed_native_state = None
            uncertain = possible_acceptance or isinstance(exc, PiRequestUncertainError)
            self.state = PiSessionState.RECONCILING if uncertain else PiSessionState.DEAD
            if uncertain:
                try:
                    self._persist_prompt_state(
                        "reconciliation_required",
                        request_id=request_id,
                        digest=digest,
                    )
                except Exception:
                    # The earlier durable ``sending`` fence, when writable,
                    # remains sufficient to block replay. A binding backend
                    # failure is itself reported as reconciliation-required.
                    pass
            self._retire(redact_protocol_text(exc))
            return ExternalTurnResult(
                status=TurnStatus.RECONCILIATION_REQUIRED if uncertain else TurnStatus.ERROR,
                error=redact_protocol_text(exc), should_retire=True,
                reconciliation_required=uncertain, projected_messages=projected,
                final_text=final_text, tool_iterations=tools,
                run_id=f"{self.native_session_id}:{self._generation}",
                phase="after_send" if uncertain else "before_send",
                prompt_accepted=None if uncertain else False,
            )
        finally:
            with self._lock:
                self._active_generation = None
            if self.state in {PiSessionState.RUNNING, PiSessionState.ABORTING}:
                self.state = PiSessionState.IDLE

    def _mark_reconciliation_required(self) -> None:
        binding = self.binding
        if binding is not None and binding.prompt_state == "native_completed":
            self._persist_prompt_state(
                "reconciliation_required",
                request_id=binding.prompt_request_id,
                digest=binding.prompt_digest,
            )
        self._completed_native_state = None
        self.state = PiSessionState.RECONCILING

    def _validate_native_completion(self, turn: ExternalTurnResult | None) -> RuntimeState:
        binding = self.binding
        if binding is None or binding.prompt_state != "native_completed":
            raise PiReconciliationRequired(
                "Pi settlement requires the exact native_completed durability boundary"
            )
        if (
            turn is None
            or turn.reconciliation_required
            or turn.prompt_accepted is not True
            or turn.status not in {TurnStatus.COMPLETED, TurnStatus.INTERRUPTED}
        ):
            raise PiReconciliationRequired(
                "Pi settlement requires a completed, certainly accepted native result"
            )
        state = self._completed_native_state
        if state is None:
            raise PiReconciliationRequired(
                "fresh native completion state is unavailable; explicit reconciliation is required"
            )
        if (
            state.session_id != turn.session_id
            or state.session_file != turn.session_file
            or state.cursor != turn.cursor
            or state.session_id != binding.native_session_id
            or state.session_file != binding.native_session_file
            or binding.turn_generation != self._generation
            or turn.run_id != f"{state.session_id}:{binding.turn_generation}"
        ):
            raise PiReconciliationRequired(
                "native completion identity changed before the durability fence"
            )
        return state

    def _settle_native_completion(self, turn: ExternalTurnResult) -> None:
        state = self._validate_native_completion(turn)
        try:
            self._persist_prompt_state(
                "settled", request_id=None, digest=None, state=state,
            )
        except Exception:
            try:
                self._mark_reconciliation_required()
            except Exception:
                pass
            raise
        self._completed_native_state = None
        self.state = (
            PiSessionState.IDLE
            if self.client is not None and self.client.is_alive()
            else PiSessionState.DEAD
        )

    def finalize_projection(
        self,
        *,
        persisted: bool,
        turn: ExternalTurnResult | None = None,
    ) -> None:
        """Settle only after Hermes durably accepts a non-empty projection."""
        with self._lock:
            if not persisted:
                if self.binding is not None and self.binding.prompt_state == "native_completed":
                    self._mark_reconciliation_required()
                return
            try:
                if turn is None:
                    raise PiReconciliationRequired(
                        "Pi settlement requires the exact completed turn result"
                    )
                self._settle_native_completion(turn)
            except Exception:
                if self.binding is not None and self.binding.prompt_state == "native_completed":
                    try:
                        self._mark_reconciliation_required()
                    except Exception:
                        pass
                raise

    def finalize_coding_task(
        self,
        *,
        turn: ExternalTurnResult,
        evidence: dict[str, Any],
    ) -> str:
        """Durably bind native completion to host verification before settlement."""
        with self._lock:
            try:
                self._validate_native_completion(turn)
                checks = evidence.get("verification")
                required = evidence.get("required_verification_count")
                if (
                    not isinstance(required, int)
                    or isinstance(required, bool)
                    or required <= 0
                    or not isinstance(checks, list)
                    or len(checks) != required
                    or any(
                        not isinstance(item, dict)
                        or not isinstance(item.get("command"), str)
                        or not isinstance(item.get("ok"), bool)
                        for item in checks
                    )
                    or not isinstance(evidence.get("verification_passed"), bool)
                    or evidence["verification_passed"]
                    != all(bool(item["ok"]) for item in checks)
                ):
                    raise PiReconciliationRequired(
                        "coding_task settlement requires evidence for every requested host verification"
                    )
                binding = self.binding
                if binding is None:
                    raise PiReconciliationRequired("Pi coding_task binding disappeared")
                request_id = binding.prompt_request_id or "missing-request"
                suffix = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
                record_path = self.binding_path.with_name(
                    f"coding-task-settlement-{self._generation}-{suffix}.json"
                )
                record = dict(evidence)
                record.update({
                    "schema_version": 1,
                    "runtime": "pi_rpc",
                    "native_result": {
                        "session_id": turn.session_id,
                        "session_file": turn.session_file,
                        "cursor": turn.cursor,
                        "run_id": turn.run_id,
                        "status": turn.status.value,
                        "final_text_sha256": hashlib.sha256(
                            turn.final_text.encode("utf-8")
                        ).hexdigest(),
                    },
                    "prompt_fence": {
                        "request_id": binding.prompt_request_id,
                        "prompt_digest": binding.prompt_digest,
                        "turn_generation": binding.turn_generation,
                        "prior_cursor": binding.last_entry_id,
                        "model": binding.model,
                        "thinking_level": binding.thinking_level,
                    },
                })
                save_json_record(record_path, record)
                self._settle_native_completion(turn)
                return record_path.name
            except Exception:
                if self.binding is not None and self.binding.prompt_state == "native_completed":
                    try:
                        self._mark_reconciliation_required()
                    except Exception:
                        pass
                raise

    def request_interrupt(self) -> None:
        self._interrupt.set()

    def steer(self, message: str) -> bool:
        cleaned = str(message or "").strip()
        if not cleaned:
            return False
        with self._lock:
            generation = self._active_generation
            client = self.client
            if (
                generation is None
                or self.state is not PiSessionState.RUNNING
                or client is None
                or not client.is_alive()
            ):
                return False
            # Generation ownership and the RPC transmission are one atomic
            # critical section. Settlement clears _active_generation under
            # this same lock, so it cannot win between validation and send.
            try:
                client.request("steer", message=cleaned)
            except PiRequestUncertainError:
                # The steering bytes may have crossed stdin. Report ownership
                # to callers so they never enqueue a duplicate fallback.
                return True
            return bool(
                self._active_generation == generation
                and self.state is PiSessionState.RUNNING
            )

    def follow_up(self, message: str) -> bool:
        """Native busy follow-up is disabled for Pi 0.84.3.

        That protocol version cannot atomically reject an idle queue request,
        so frontends must retain the durable Hermes turn queue exactly once.
        """
        _ = message
        return False

    def abort(self) -> None:
        self.request_interrupt()

    def set_model(self, provider: str, model_id: str) -> RuntimeState:
        self.ensure_started()
        assert self.client is not None
        self.client.request("set_model", provider=provider, modelId=model_id)
        state = self._query_state()
        self._persist_prompt_state("settled", request_id=None, digest=None, state=state)
        return state

    def set_thinking_level(self, level: str) -> RuntimeState:
        self.ensure_started()
        assert self.client is not None
        self.client.request("set_thinking_level", level=level)
        state = self._query_state()
        self._persist_prompt_state("settled", request_id=None, digest=None, state=state)
        return state

    def compact(self, instructions: str | None = None) -> CompactionResult:
        self.ensure_started()
        assert self.client is not None
        self.state = PiSessionState.COMPACTING
        try:
            self.client.request(
                "compact", timeout=self.turn_timeout,
                **({"customInstructions": instructions} if instructions else {}),
            )
            state = self._query_state()
            self._persist_prompt_state("settled", request_id=None, digest=None, state=state)
            return CompactionResult(True, cursor=state.cursor)
        except PiRpcError as exc:
            return CompactionResult(False, error=redact_protocol_text(exc))
        finally:
            self.state = PiSessionState.IDLE

    def available_models(self) -> list[dict[str, Any]]:
        self.ensure_started()
        assert self.client is not None
        result = self.client.request("get_available_models")
        data = result.get("data") or {}
        models = data.get("models") if isinstance(data, dict) else []
        return models if isinstance(models, list) else []

    def _retire(self, reason: str) -> bool:
        _ = reason
        retired = True
        if self.client:
            try:
                self.client.close(force=True)
            except Exception:
                retired = False
        self.client = None
        if self.sandbox:
            try:
                self.sandbox.cleanup()
            except Exception:
                retired = False
        if self.state != PiSessionState.RECONCILING:
            self.state = PiSessionState.DEAD
        return retired

    def close(self) -> None:
        if self.client:
            self.client.close()
        self.client = None
        if self.sandbox:
            self.sandbox.cleanup()
        self.state = PiSessionState.DEAD
