"""Owner-controlled one-time approval gate for GhostFire agents."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from core.eventbus import EventBus


class ApprovalStatus(str, Enum):
    """Lifecycle states for one approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ApprovalGateState(str, Enum):
    """Lifecycle states exposed by the approval gate."""

    STOPPED = "stopped"
    RUNNING = "running"


class AgentApprovalError(RuntimeError):
    """Base class for approval-gate failures."""


class ApprovalGateStateError(AgentApprovalError):
    """Raised when an operation conflicts with gate state."""


class ApprovalRequestError(AgentApprovalError):
    """Raised when an approval request is invalid or unavailable."""


class ApprovalIdentityError(AgentApprovalError):
    """Raised when a non-owner attempts an owner decision."""


class ApprovalCapacityError(AgentApprovalError):
    """Raised when the pending approval capacity is exhausted."""


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    """Immutable, redacted view of one approval record."""

    identifier: str
    fingerprint: str
    action: str
    agent_name: str
    agent_role: str
    resource: str
    mode: str
    attribute_names: tuple[str, ...]
    policy_rule: str | None
    policy_reason: str
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime
    decided_at: datetime | None
    consumed_at: datetime | None
    decided_by: str | None
    decision_note: str | None
    sequence: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe redacted approval representation."""

        return {
            "identifier": self.identifier,
            "fingerprint": self.fingerprint,
            "action": self.action,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "resource": self.resource,
            "mode": self.mode,
            "attribute_names": list(self.attribute_names),
            "policy_rule": self.policy_rule,
            "policy_reason": self.policy_reason,
            "status": self.status.value,
            "requested_at": self.requested_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "decided_at": (
                self.decided_at.isoformat()
                if self.decided_at is not None
                else None
            ),
            "consumed_at": (
                self.consumed_at.isoformat()
                if self.consumed_at is not None
                else None
            ),
            "decided_by": self.decided_by,
            "decision_note": self.decision_note,
            "sequence": self.sequence,
        }


@dataclass(slots=True)
class _ApprovalRecord:
    identifier: str
    fingerprint: str
    action: str
    agent_name: str
    agent_role: str
    resource: str
    mode: str
    attributes: Mapping[str, Any]
    policy_rule: str | None
    policy_reason: str
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime
    sequence: int
    decided_at: datetime | None = None
    consumed_at: datetime | None = None
    decided_by: str | None = None
    decision_note: str | None = None


class AgentApprovalGate:
    """
    Thread-safe owner approval gate with exact, one-time request binding.

    Requests are fingerprinted from the complete policy request, including
    protected attribute values. Public snapshots and telemetry expose only
    attribute names and the fingerprint. Approved requests are consumed once.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        history_limit: int = 100,
        max_pending: int = 100,
        approval_ttl_seconds: float = 300.0,
        owner_identity: str = "owner",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._history_limit = _validate_positive_int(
            history_limit,
            field_name="history_limit",
        )
        self._max_pending = _validate_positive_int(
            max_pending,
            field_name="max_pending",
        )

        if (
            isinstance(approval_ttl_seconds, bool)
            or not isinstance(approval_ttl_seconds, (int, float))
            or approval_ttl_seconds <= 0
        ):
            raise ValueError(
                "approval_ttl_seconds must be a positive number"
            )

        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")

        self._event_bus = event_bus
        self._approval_ttl_seconds = float(approval_ttl_seconds)
        self._owner_identity = _validate_text(
            owner_identity,
            field_name="owner_identity",
        )
        self._clock = clock or (
            lambda: datetime.now(timezone.utc)
        )
        self._lock = RLock()
        self._state = ApprovalGateState.STOPPED
        self._records: dict[str, _ApprovalRecord] = {}
        self._active_by_fingerprint: dict[str, str] = {}
        self._history: deque[ApprovalSnapshot] = deque(
            maxlen=self._history_limit
        )
        self._sequence = 0
        self._request_count = 0
        self._approval_count = 0
        self._denial_count = 0
        self._consumption_count = 0
        self._expiry_count = 0
        self._last_error: str | None = None

    @property
    def state(self) -> ApprovalGateState:
        with self._lock:
            return self._state

    @property
    def owner_identity(self) -> str:
        return self._owner_identity

    @property
    def pending_count(self) -> int:
        with self._lock:
            self._expire_locked(self._now())
            return self._pending_count_locked()

    def start(self) -> bool:
        """Start accepting approval requests and owner decisions."""

        with self._lock:
            if self._state is ApprovalGateState.RUNNING:
                return False

            self._state = ApprovalGateState.RUNNING
            self._last_error = None
            payload = self._status_payload_locked()

        self._publish("ghostfire.approval_gate.started", payload)
        return True

    def stop(self) -> bool:
        """Stop approval processing."""

        with self._lock:
            if self._state is ApprovalGateState.STOPPED:
                return False

            self._state = ApprovalGateState.STOPPED
            payload = self._status_payload_locked()

        self._publish("ghostfire.approval_gate.stopped", payload)
        return True

    def health(self) -> bool:
        """Return whether the gate accepts requests."""

        with self._lock:
            return self._state is ApprovalGateState.RUNNING

    def authorize_or_request(
        self,
        *,
        action: str,
        agent_name: str,
        agent_role: str,
        resource: str,
        mode: str,
        attributes: Mapping[str, Any] | None,
        policy_rule: str | None,
        policy_reason: str,
    ) -> ApprovalSnapshot:
        """
        Return a pending/denied request or consume an approved request.

        Replaying the exact execution after owner approval returns a CONSUMED
        snapshot. Any change to the protected request creates a new fingerprint
        and therefore requires a separate approval.
        """

        normalized = self._normalize_request(
            action=action,
            agent_name=agent_name,
            agent_role=agent_role,
            resource=resource,
            mode=mode,
            attributes=attributes,
            policy_rule=policy_rule,
            policy_reason=policy_reason,
        )
        fingerprint = _fingerprint_request(normalized)
        now = self._now()

        with self._lock:
            self._require_running_locked()
            expired = self._expire_locked(now)
            existing_id = self._active_by_fingerprint.get(
                fingerprint
            )

            if existing_id is not None:
                record = self._records[existing_id]

                if record.status is ApprovalStatus.APPROVED:
                    record.status = ApprovalStatus.CONSUMED
                    record.consumed_at = now
                    self._active_by_fingerprint.pop(
                        fingerprint,
                        None,
                    )
                    self._consumption_count += 1
                    snapshot = self._snapshot(record)
                    self._history.append(snapshot)
                    self._prune_locked()
                    event_name = (
                        "ghostfire.approval_gate.consumed"
                    )
                else:
                    snapshot = self._snapshot(record)
                    event_name = None
            else:
                if self._open_count_locked() >= self._max_pending:
                    self._last_error = (
                        "active approval capacity exhausted"
                    )
                    raise ApprovalCapacityError(
                        self._last_error
                    )

                self._sequence += 1
                record = _ApprovalRecord(
                    identifier=uuid4().hex,
                    fingerprint=fingerprint,
                    action=normalized["action"],
                    agent_name=normalized["agent_name"],
                    agent_role=normalized["agent_role"],
                    resource=normalized["resource"],
                    mode=normalized["mode"],
                    attributes=normalized["attributes"],
                    policy_rule=normalized["policy_rule"],
                    policy_reason=normalized["policy_reason"],
                    status=ApprovalStatus.PENDING,
                    requested_at=now,
                    expires_at=now + timedelta(
                        seconds=self._approval_ttl_seconds
                    ),
                    sequence=self._sequence,
                )
                self._records[record.identifier] = record
                self._active_by_fingerprint[
                    fingerprint
                ] = record.identifier
                self._request_count += 1
                self._last_error = None
                snapshot = self._snapshot(record)
                self._history.append(snapshot)
                self._prune_locked()
                event_name = (
                    "ghostfire.approval_gate.requested"
                )

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        if event_name is not None:
            self._publish(event_name, snapshot.as_dict())

        return snapshot

    def approve(
        self,
        identifier: str,
        *,
        decided_by: str,
        note: str = "",
    ) -> ApprovalSnapshot:
        """Approve one pending request as the configured owner."""

        return self._decide(
            identifier,
            decided_by=decided_by,
            note=note,
            status=ApprovalStatus.APPROVED,
        )

    def deny(
        self,
        identifier: str,
        *,
        decided_by: str,
        note: str = "",
    ) -> ApprovalSnapshot:
        """Deny one pending request as the configured owner."""

        return self._decide(
            identifier,
            decided_by=decided_by,
            note=note,
            status=ApprovalStatus.DENIED,
        )

    def cancel(
        self,
        identifier: str,
        *,
        decided_by: str,
        note: str = "",
    ) -> ApprovalSnapshot:
        """Cancel and clear a pending, approved, or denied request."""

        normalized_id = _validate_text(
            identifier,
            field_name="identifier",
        )
        owner = self._validate_owner(decided_by)
        normalized_note = _validate_note(note)
        now = self._now()

        with self._lock:
            self._require_running_locked()
            expired = self._expire_locked(now)
            record = self._require_record_locked(normalized_id)

            if record.status in {
                ApprovalStatus.CONSUMED,
                ApprovalStatus.CANCELLED,
                ApprovalStatus.EXPIRED,
            }:
                raise ApprovalRequestError(
                    f"approval cannot be cancelled from "
                    f"{record.status.value}"
                )

            record.status = ApprovalStatus.CANCELLED
            record.decided_at = now
            record.decided_by = owner
            record.decision_note = normalized_note
            self._active_by_fingerprint.pop(
                record.fingerprint,
                None,
            )
            snapshot = self._snapshot(record)
            self._history.append(snapshot)
            self._prune_locked()

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        self._publish(
            "ghostfire.approval_gate.cancelled",
            snapshot.as_dict(),
        )
        return snapshot

    def get(self, identifier: str) -> ApprovalSnapshot:
        """Return one redacted approval snapshot."""

        normalized_id = _validate_text(
            identifier,
            field_name="identifier",
        )

        with self._lock:
            expired = self._expire_locked(self._now())
            snapshot = self._snapshot(
                self._require_record_locked(normalized_id)
            )

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        return snapshot

    def list_requests(
        self,
        *,
        status: ApprovalStatus | str | None = None,
    ) -> tuple[ApprovalSnapshot, ...]:
        """Return redacted requests in creation order."""

        normalized_status = (
            _normalize_status(status)
            if status is not None
            else None
        )

        with self._lock:
            expired = self._expire_locked(self._now())
            snapshots = tuple(
                self._snapshot(record)
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.sequence,
                )
                if (
                    normalized_status is None
                    or record.status is normalized_status
                )
            )

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        return snapshots

    def history(self) -> tuple[ApprovalSnapshot, ...]:
        """Return bounded state-transition history."""

        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe redacted gate snapshot."""

        with self._lock:
            expired = self._expire_locked(self._now())
            requests = [
                self._snapshot(record).as_dict()
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.sequence,
                )
            ]
            payload = {
                **self._status_payload_locked(),
                "owner_identity": self._owner_identity,
                "approval_ttl_seconds": (
                    self._approval_ttl_seconds
                ),
                "max_pending": self._max_pending,
                "requests": requests,
            }

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        return payload

    def _decide(
        self,
        identifier: str,
        *,
        decided_by: str,
        note: str,
        status: ApprovalStatus,
    ) -> ApprovalSnapshot:
        normalized_id = _validate_text(
            identifier,
            field_name="identifier",
        )
        owner = self._validate_owner(decided_by)
        normalized_note = _validate_note(note)
        now = self._now()

        with self._lock:
            self._require_running_locked()
            expired = self._expire_locked(now)
            record = self._require_record_locked(normalized_id)

            if record.status is not ApprovalStatus.PENDING:
                raise ApprovalRequestError(
                    f"approval is not pending: "
                    f"{record.status.value}"
                )

            record.status = status
            record.decided_at = now
            record.decided_by = owner
            record.decision_note = normalized_note

            if status is ApprovalStatus.APPROVED:
                self._approval_count += 1
                event_name = (
                    "ghostfire.approval_gate.approved"
                )
            else:
                self._denial_count += 1
                event_name = (
                    "ghostfire.approval_gate.denied"
                )

            snapshot = self._snapshot(record)
            self._history.append(snapshot)
            self._prune_locked()

        for expired_snapshot in expired:
            self._publish(
                "ghostfire.approval_gate.expired",
                expired_snapshot.as_dict(),
            )

        self._publish(event_name, snapshot.as_dict())
        return snapshot

    def _normalize_request(
        self,
        *,
        action: str,
        agent_name: str,
        agent_role: str,
        resource: str,
        mode: str,
        attributes: Mapping[str, Any] | None,
        policy_rule: str | None,
        policy_reason: str,
    ) -> dict[str, Any]:
        if attributes is None:
            normalized_attributes: Mapping[str, Any] = (
                MappingProxyType({})
            )
        elif isinstance(attributes, Mapping):
            normalized_attributes = MappingProxyType(
                {
                    str(key): _freeze_value(value)
                    for key, value in attributes.items()
                }
            )
        else:
            raise TypeError(
                "attributes must be a mapping or None"
            )

        if policy_rule is not None:
            normalized_rule = _validate_text(
                policy_rule,
                field_name="policy_rule",
            )
        else:
            normalized_rule = None

        return {
            "action": _validate_text(
                action,
                field_name="action",
            ).lower(),
            "agent_name": _validate_text(
                agent_name,
                field_name="agent_name",
            ),
            "agent_role": _validate_text(
                agent_role,
                field_name="agent_role",
            ).lower(),
            "resource": _validate_text(
                resource,
                field_name="resource",
            ).lower(),
            "mode": _validate_text(
                mode,
                field_name="mode",
            ).lower(),
            "attributes": normalized_attributes,
            "policy_rule": normalized_rule,
            "policy_reason": _validate_text(
                policy_reason,
                field_name="policy_reason",
            ),
        }

    def _expire_locked(
        self,
        now: datetime,
    ) -> tuple[ApprovalSnapshot, ...]:
        expired: list[ApprovalSnapshot] = []

        for identifier in tuple(
            self._active_by_fingerprint.values()
        ):
            record = self._records.get(identifier)

            if record is None:
                continue

            if (
                record.status
                in {
                    ApprovalStatus.PENDING,
                    ApprovalStatus.APPROVED,
                }
                and now >= record.expires_at
            ):
                record.status = ApprovalStatus.EXPIRED
                record.decided_at = now
                record.decision_note = "approval expired"
                self._active_by_fingerprint.pop(
                    record.fingerprint,
                    None,
                )
                self._expiry_count += 1
                snapshot = self._snapshot(record)
                self._history.append(snapshot)
                expired.append(snapshot)

        if expired:
            self._prune_locked()

        return tuple(expired)

    def _pending_count_locked(self) -> int:
        return sum(
            record.status is ApprovalStatus.PENDING
            for record in self._records.values()
        )

    def _open_count_locked(self) -> int:
        return sum(
            record.status in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
                ApprovalStatus.DENIED,
            }
            for record in self._records.values()
        )

    def _require_running_locked(self) -> None:
        if self._state is not ApprovalGateState.RUNNING:
            raise ApprovalGateStateError(
                "approval gate is not running"
            )

    def _require_record_locked(
        self,
        identifier: str,
    ) -> _ApprovalRecord:
        try:
            return self._records[identifier]
        except KeyError as exc:
            raise ApprovalRequestError(
                f"approval request is not registered: {identifier}"
            ) from exc

    def _validate_owner(self, decided_by: str) -> str:
        owner = _validate_text(
            decided_by,
            field_name="decided_by",
        )

        if owner != self._owner_identity:
            self._last_error = (
                f"approval decision rejected for identity {owner!r}"
            )
            raise ApprovalIdentityError(self._last_error)

        return owner

    def _prune_locked(self) -> None:
        terminal = sorted(
            (
                record
                for record in self._records.values()
                if record.status in {
                    ApprovalStatus.CONSUMED,
                    ApprovalStatus.CANCELLED,
                    ApprovalStatus.EXPIRED,
                }
            ),
            key=lambda item: item.sequence,
        )
        excess = max(
            0,
            len(terminal) - self._history_limit,
        )

        for record in terminal[:excess]:
            self._records.pop(record.identifier, None)

    def _status_payload_locked(self) -> dict[str, Any]:
        counts = {
            status.value: 0
            for status in ApprovalStatus
        }

        for record in self._records.values():
            counts[record.status.value] += 1

        return {
            "state": self._state.value,
            "healthy": self._state is ApprovalGateState.RUNNING,
            "request_count": self._request_count,
            "approval_count": self._approval_count,
            "denial_count": self._denial_count,
            "consumption_count": self._consumption_count,
            "expiry_count": self._expiry_count,
            "pending_count": counts[
                ApprovalStatus.PENDING.value
            ],
            "active_count": self._open_count_locked(),
            "counts": counts,
            "history_size": len(self._history),
            "last_error": self._last_error,
        }

    @staticmethod
    def _snapshot(
        record: _ApprovalRecord,
    ) -> ApprovalSnapshot:
        return ApprovalSnapshot(
            identifier=record.identifier,
            fingerprint=record.fingerprint,
            action=record.action,
            agent_name=record.agent_name,
            agent_role=record.agent_role,
            resource=record.resource,
            mode=record.mode,
            attribute_names=tuple(
                sorted(record.attributes)
            ),
            policy_rule=record.policy_rule,
            policy_reason=record.policy_reason,
            status=record.status,
            requested_at=record.requested_at,
            expires_at=record.expires_at,
            decided_at=record.decided_at,
            consumed_at=record.consumed_at,
            decided_by=record.decided_by,
            decision_note=record.decision_note,
            sequence=record.sequence,
        )

    def _now(self) -> datetime:
        value = self._clock()

        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

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


def _fingerprint_request(request: Mapping[str, Any]) -> str:
    canonical = {
        key: _canonicalize(value)
        for key, value in request.items()
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item)
            for item in value
        ]

    if isinstance(value, (set, frozenset)):
        normalized = [
            _canonicalize(item)
            for item in value
        ]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                default=str,
            ),
        )

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return {
        "type": (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}"
        ),
        "repr": repr(value),
    }


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_value(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)

    if isinstance(value, set):
        return frozenset(
            _freeze_value(item)
            for item in value
        )

    return deepcopy(value)


def _validate_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _validate_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return normalized


def _validate_note(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("note must be a string")

    return value.strip()


def _normalize_status(
    value: ApprovalStatus | str,
) -> ApprovalStatus:
    if isinstance(value, ApprovalStatus):
        return value

    if isinstance(value, str):
        try:
            return ApprovalStatus(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported approval status: {value}"
            ) from exc

    raise TypeError(
        "status must be an ApprovalStatus or string"
    )
