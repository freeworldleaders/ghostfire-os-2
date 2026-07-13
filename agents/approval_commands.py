"""Authenticated owner command interface for agent approvals."""

from __future__ import annotations

import hmac
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4

from agents.approval import (
    AgentApprovalError,
    AgentApprovalGate,
)
from core.eventbus import EventBus


class ApprovalCommandAction(str, Enum):
    """Owner command actions exposed by the approval interface."""

    LIST = "list"
    GET = "get"
    APPROVE = "approve"
    DENY = "deny"
    CANCEL = "cancel"


class ApprovalCommandState(str, Enum):
    """Lifecycle states exposed by the command interface."""

    STOPPED = "stopped"
    RUNNING = "running"


class ApprovalCommandError(RuntimeError):
    """Base class for approval-command failures."""

    code = "approval_command_failed"
    public_message = "approval command failed"


class ApprovalCommandStateError(ApprovalCommandError):
    """Raised when the interface is unavailable."""

    code = "approval_interface_not_running"
    public_message = "approval command interface is not running"


class ApprovalCommandDisabledError(ApprovalCommandError):
    """Raised while owner commands remain in safe hold."""

    code = "approval_interface_disabled"
    public_message = "approval command interface is disabled"


class ApprovalCommandAuthenticationError(ApprovalCommandError):
    """Raised when the owner token is invalid."""

    code = "approval_unauthorized"
    public_message = "owner authorization failed"


class ApprovalCommandValidationError(ApprovalCommandError):
    """Raised when a command message is malformed."""

    code = "approval_invalid_request"
    public_message = "approval command request is invalid"


class ApprovalCommandOperationError(ApprovalCommandError):
    """Raised when the approval gate rejects an operation."""

    code = "approval_operation_failed"
    public_message = "approval command operation failed"


@dataclass(frozen=True, slots=True)
class ApprovalCommandRecord:
    """Immutable, redacted command audit record."""

    identifier: str
    action: str
    approval_id: str | None
    outcome: str
    error_code: str | None
    started_at: datetime
    completed_at: datetime

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (self.completed_at - self.started_at).total_seconds(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe command audit representation."""

        return {
            "identifier": self.identifier,
            "action": self.action,
            "approval_id": self.approval_id,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class AgentApprovalCommandInterface:
    """
    Token-authenticated command surface for owner approval decisions.

    The interface is disabled by default. Every command requires a dedicated
    owner token even when the transport itself is bound to loopback. Tokens,
    notes, and request attributes are never retained in audit history or
    telemetry.
    """

    _COMMON_FIELDS = frozenset(
        {"id", "type", "action", "token"}
    )
    _ACTION_FIELDS = {
        ApprovalCommandAction.LIST: frozenset({"status"}),
        ApprovalCommandAction.GET: frozenset({"approval_id"}),
        ApprovalCommandAction.APPROVE: frozenset(
            {"approval_id", "note"}
        ),
        ApprovalCommandAction.DENY: frozenset(
            {"approval_id", "note"}
        ),
        ApprovalCommandAction.CANCEL: frozenset(
            {"approval_id", "note"}
        ),
    }

    def __init__(
        self,
        approval_gate: AgentApprovalGate,
        *,
        enabled: bool = False,
        owner_token: str | None = None,
        history_limit: int = 100,
        max_note_length: int = 500,
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(approval_gate, AgentApprovalGate):
            raise TypeError(
                "approval_gate must be an AgentApprovalGate"
            )

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        if owner_token is not None:
            owner_token = _validate_text(
                owner_token,
                field_name="owner_token",
            )

        if enabled and owner_token is None:
            raise ValueError(
                "enabled approval commands require owner_token"
            )

        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError(
                "history_limit must be a positive integer"
            )

        if (
            isinstance(max_note_length, bool)
            or not isinstance(max_note_length, int)
            or max_note_length < 1
        ):
            raise ValueError(
                "max_note_length must be a positive integer"
            )

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._approval_gate = approval_gate
        self._enabled = enabled
        self._owner_token = owner_token
        self._history_limit = history_limit
        self._max_note_length = max_note_length
        self._event_bus = event_bus
        self._lock = RLock()
        self._state = ApprovalCommandState.STOPPED
        self._history: deque[ApprovalCommandRecord] = deque(
            maxlen=self._history_limit
        )
        self._command_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._authentication_failure_count = 0
        self._last_error_code: str | None = None

    @property
    def state(self) -> ApprovalCommandState:
        with self._lock:
            return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def command_count(self) -> int:
        with self._lock:
            return self._command_count

    def start(self) -> bool:
        """Start the command interface service."""

        with self._lock:
            if self._state is ApprovalCommandState.RUNNING:
                return False

            self._state = ApprovalCommandState.RUNNING
            self._last_error_code = None
            payload = self._status_payload_locked()

        self._publish(
            "ghostfire.approval_command.started",
            payload,
        )
        return True

    def stop(self) -> bool:
        """Stop the command interface service."""

        with self._lock:
            if self._state is ApprovalCommandState.STOPPED:
                return False

            self._state = ApprovalCommandState.STOPPED
            payload = self._status_payload_locked()

        self._publish(
            "ghostfire.approval_command.stopped",
            payload,
        )
        return True

    def health(self) -> bool:
        """Return whether the interface service is running."""

        with self._lock:
            return self._state is ApprovalCommandState.RUNNING

    def execute(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Authenticate and execute one owner approval command."""

        started_at = datetime.now(timezone.utc)
        command_id = uuid4().hex
        action_name = "unknown"
        approval_id: str | None = None

        try:
            normalized_message = self._validate_root(message)
            self._require_available()
            self._authenticate(normalized_message.get("token"))
            action = self._normalize_action(
                normalized_message.get("action")
            )
            action_name = action.value
            self._validate_fields(
                normalized_message,
                action,
            )

            if action is ApprovalCommandAction.LIST:
                status = normalized_message.get("status")
                requests = self._approval_gate.list_requests(
                    status=status,
                )
                data: Any = [
                    request.as_dict()
                    for request in requests
                ]
            else:
                approval_id = _validate_text(
                    normalized_message.get("approval_id"),
                    field_name="approval_id",
                )

                if action is ApprovalCommandAction.GET:
                    data = self._approval_gate.get(
                        approval_id
                    ).as_dict()
                else:
                    note = self._validate_note(
                        normalized_message.get("note", "")
                    )

                    if action is ApprovalCommandAction.APPROVE:
                        snapshot = self._approval_gate.approve(
                            approval_id,
                            decided_by=(
                                self._approval_gate.owner_identity
                            ),
                            note=note,
                        )
                    elif action is ApprovalCommandAction.DENY:
                        snapshot = self._approval_gate.deny(
                            approval_id,
                            decided_by=(
                                self._approval_gate.owner_identity
                            ),
                            note=note,
                        )
                    else:
                        snapshot = self._approval_gate.cancel(
                            approval_id,
                            decided_by=(
                                self._approval_gate.owner_identity
                            ),
                            note=note,
                        )

                    data = snapshot.as_dict()

            completed_at = datetime.now(timezone.utc)
            record = ApprovalCommandRecord(
                identifier=command_id,
                action=action_name,
                approval_id=approval_id,
                outcome="completed",
                error_code=None,
                started_at=started_at,
                completed_at=completed_at,
            )
            self._retain(record, success=True)
            self._publish(
                "ghostfire.approval_command.completed",
                record.as_dict(),
            )

            return {
                "status": "ok",
                "action": action_name,
                "data": data,
            }
        except ApprovalCommandError as exc:
            self._record_failure(
                command_id=command_id,
                action=action_name,
                approval_id=approval_id,
                started_at=started_at,
                error=exc,
            )
            raise
        except AgentApprovalError as exc:
            wrapped = ApprovalCommandOperationError(
                type(exc).__name__
            )
            self._record_failure(
                command_id=command_id,
                action=action_name,
                approval_id=approval_id,
                started_at=started_at,
                error=wrapped,
            )
            raise wrapped from exc
        except (TypeError, ValueError) as exc:
            wrapped = ApprovalCommandValidationError(
                type(exc).__name__
            )
            self._record_failure(
                command_id=command_id,
                action=action_name,
                approval_id=approval_id,
                started_at=started_at,
                error=wrapped,
            )
            raise wrapped from exc

    def handle(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a transport-safe response for one command."""

        try:
            return self.execute(message)
        except ApprovalCommandError as exc:
            return {
                "status": "error",
                "code": exc.code,
                "message": exc.public_message,
            }

    def history(self) -> tuple[ApprovalCommandRecord, ...]:
        """Return bounded redacted command history."""

        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe interface snapshot."""

        with self._lock:
            return {
                **self._status_payload_locked(),
                "enabled": self._enabled,
                "safe_hold": not self._enabled,
                "token_configured": self._owner_token is not None,
                "max_note_length": self._max_note_length,
            }

    def _validate_root(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            raise ApprovalCommandValidationError(
                "message must be a mapping"
            )

        normalized = dict(message)

        if normalized.get("type") != "approval":
            raise ApprovalCommandValidationError(
                "type must be approval"
            )

        return normalized

    def _require_available(self) -> None:
        with self._lock:
            if self._state is not ApprovalCommandState.RUNNING:
                raise ApprovalCommandStateError(
                    "interface is not running"
                )

        if not self._enabled:
            raise ApprovalCommandDisabledError(
                "interface is disabled"
            )

    def _authenticate(self, supplied_token: Any) -> None:
        if not isinstance(supplied_token, str):
            self._authentication_failed()
            raise ApprovalCommandAuthenticationError(
                "owner token is invalid"
            )

        expected = self._owner_token

        if expected is None or not hmac.compare_digest(
            supplied_token,
            expected,
        ):
            self._authentication_failed()
            raise ApprovalCommandAuthenticationError(
                "owner token is invalid"
            )

    def _authentication_failed(self) -> None:
        with self._lock:
            self._authentication_failure_count += 1

    def _validate_fields(
        self,
        message: Mapping[str, Any],
        action: ApprovalCommandAction,
    ) -> None:
        allowed = (
            self._COMMON_FIELDS
            | self._ACTION_FIELDS[action]
        )
        unknown = sorted(set(message) - allowed)

        if unknown:
            raise ApprovalCommandValidationError(
                "unknown fields: " + ", ".join(unknown)
            )

        if action in {
            ApprovalCommandAction.GET,
            ApprovalCommandAction.APPROVE,
            ApprovalCommandAction.DENY,
            ApprovalCommandAction.CANCEL,
        } and "approval_id" not in message:
            raise ApprovalCommandValidationError(
                "approval_id is required"
            )

    def _validate_note(self, note: Any) -> str:
        if not isinstance(note, str):
            raise ApprovalCommandValidationError(
                "note must be a string"
            )

        normalized = note.strip()

        if len(normalized) > self._max_note_length:
            raise ApprovalCommandValidationError(
                "note exceeds maximum length"
            )

        return normalized

    @staticmethod
    def _normalize_action(
        value: Any,
    ) -> ApprovalCommandAction:
        if not isinstance(value, str):
            raise ApprovalCommandValidationError(
                "action must be a string"
            )

        try:
            return ApprovalCommandAction(
                value.strip().lower()
            )
        except ValueError as exc:
            raise ApprovalCommandValidationError(
                "unsupported approval action"
            ) from exc

    def _record_failure(
        self,
        *,
        command_id: str,
        action: str,
        approval_id: str | None,
        started_at: datetime,
        error: ApprovalCommandError,
    ) -> None:
        record = ApprovalCommandRecord(
            identifier=command_id,
            action=action,
            approval_id=approval_id,
            outcome="failed",
            error_code=error.code,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        self._retain(record, success=False)
        self._publish(
            "ghostfire.approval_command.failed",
            record.as_dict(),
        )

    def _retain(
        self,
        record: ApprovalCommandRecord,
        *,
        success: bool,
    ) -> None:
        with self._lock:
            self._history.append(record)
            self._command_count += 1

            if success:
                self._success_count += 1
                self._last_error_code = None
            else:
                self._failure_count += 1
                self._last_error_code = record.error_code

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "healthy": (
                self._state is ApprovalCommandState.RUNNING
            ),
            "command_count": self._command_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "authentication_failure_count": (
                self._authentication_failure_count
            ),
            "history_size": len(self._history),
            "last_error_code": self._last_error_code,
        }

    def _publish(
        self,
        event_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self._event_bus is None:
            return

        self._event_bus.emit(
            event_name,
            deepcopy(dict(payload)),
            raise_exceptions=False,
        )


def _validate_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return normalized
