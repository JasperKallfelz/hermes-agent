"""Crash-safe bindings between a Hermes session and a native runtime session."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class BindingError(ValueError):
    """A persisted native-session binding cannot be trusted or resumed."""


PROMPT_STATES = frozenset({
    "idle",
    "sending",
    "accepted",
    "native_completed",
    "settled",
    "reconciliation_required",
})

# ``native_completed`` is deliberately still an unsettled state.  It proves
# that Pi reached its native settled event, but not that Hermes durably stored
# the canonical projection (or, for coding_task, its host verification
# evidence).  A process restart cannot reconstruct that projection from the Pi
# transcript, so every state in this set blocks another prompt until the
# in-process durability fence succeeds or a user explicitly reconciles it.
UNSETTLED_PROMPT_STATES = frozenset({
    "sending",
    "accepted",
    "native_completed",
    "reconciliation_required",
})

# These states have no completed native result that a durability fence may
# promote.  In particular, ``accepted`` alone is never evidence that the turn
# completed.
SETTLEMENT_PROHIBITED_PROMPT_STATES = frozenset({
    "sending",
    "accepted",
    "reconciliation_required",
})


def worktree_identity(cwd: str | Path) -> dict[str, Any]:
    """Return an identity that changes when a path is replaced or re-linked."""
    root = Path(cwd).expanduser().resolve(strict=True)
    stat = root.stat()
    identity: dict[str, Any] = {
        "realpath": str(root),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "git_worktree": False,
    }
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root, text=True, capture_output=True, timeout=10,
        )
        if probe.returncode or probe.stdout.strip() != "true":
            return identity
        identity["git_worktree"] = True
        for key, args in (
            ("git_dir", ["rev-parse", "--absolute-git-dir"]),
            ("git_common_dir", ["rev-parse", "--path-format=absolute", "--git-common-dir"]),
        ):
            proc = subprocess.run(
                ["git", *args], cwd=root, text=True, capture_output=True, timeout=10
            )
            if proc.returncode:
                raise BindingError(proc.stderr.strip() or "git worktree identity failed")
            identity[key] = str(Path(proc.stdout.strip()).resolve(strict=True))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BindingError(f"could not identify assigned Git worktree: {exc}") from exc
    return identity


def prompt_digest(message: object) -> str:
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RuntimeBinding:
    runtime: str
    native_session_file: str
    native_session_id: str
    last_entry_id: str | None
    cwd: str
    runtime_version: str
    worktree: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] | None = None
    thinking_level: str | None = None
    prompt_state: str = "idle"
    prompt_request_id: str | None = None
    prompt_digest: str | None = None
    turn_generation: int = 0

    def validate_host_ownership(
        self,
        *,
        assigned_cwd: str,
        session_root: str | Path,
        expected_runtime: str,
        compatible_versions: set[str] | None = None,
        allow_uncertain: bool = False,
    ) -> "RuntimeBinding":
        """Validate Hermes-owned identity without requiring native state to exist.

        This is the recovery boundary used by ``abandon``: a missing or corrupt
        native transcript must not prevent Hermes from archiving an uncertain
        binding, but a foreign worktree/runtime/path must still fail closed.
        """
        root = Path(assigned_cwd).expanduser().resolve(strict=True)
        cwd = Path(self.cwd).expanduser().resolve(strict=True)
        if cwd != root:
            raise BindingError("session binding cwd does not match the assigned worktree")
        if self.worktree != worktree_identity(root):
            raise BindingError("assigned worktree identity changed since the binding was written")

        sessions = Path(session_root).expanduser().resolve(strict=True)
        native = Path(self.native_session_file).expanduser()
        if native.is_symlink():
            raise BindingError("session binding file must not be a symlink")
        session = native.resolve(strict=False)
        if not session.is_relative_to(sessions):
            raise BindingError("session binding file is outside the managed session directory")
        if self.runtime != expected_runtime or not self.native_session_id:
            raise BindingError("foreign or incomplete runtime session binding")
        if compatible_versions is not None and self.runtime_version not in compatible_versions:
            raise BindingError("runtime version is incompatible with the persisted binding")
        if self.prompt_state not in PROMPT_STATES:
            raise BindingError("runtime binding has an invalid prompt acceptance state")
        if (
            not allow_uncertain
            and self.prompt_state in UNSETTLED_PROMPT_STATES
        ):
            raise BindingError(
                "the previous prompt may have been accepted; explicit reconciliation is required"
            )
        return self

    def validate(
        self,
        *,
        assigned_cwd: str,
        session_root: str | Path,
        expected_runtime: str,
        compatible_versions: set[str] | None = None,
        allow_uncertain: bool = False,
    ) -> "RuntimeBinding":
        self.validate_host_ownership(
            assigned_cwd=assigned_cwd,
            session_root=session_root,
            expected_runtime=expected_runtime,
            compatible_versions=compatible_versions,
            allow_uncertain=allow_uncertain,
        )
        session = Path(self.native_session_file).expanduser().resolve(strict=False)
        if not session.is_file():
            raise BindingError("session binding file is missing")
        return self

    def with_updates(self, **updates: Any) -> "RuntimeBinding":
        data = asdict(self)
        data.update(updates)
        return RuntimeBinding(**data)


def load_binding(path: str | Path) -> RuntimeBinding:
    try:
        source = Path(path)
        if source.is_symlink():
            raise BindingError("runtime binding must not be a symlink")
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("binding root must be an object")
        return RuntimeBinding(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise BindingError(f"invalid runtime binding: {exc}") from exc


def save_json_record(path: str | Path, value: Any) -> None:
    """Atomically persist one mode-0600 JSON control-plane record."""
    target = Path(path)
    if target.is_symlink():
        raise BindingError("runtime control-plane record must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_binding(path: str | Path, binding: RuntimeBinding) -> None:
    save_json_record(path, asdict(binding))
