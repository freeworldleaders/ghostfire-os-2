"""Deterministic execution policy for GhostFire AI agents."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fnmatch import fnmatchcase
from threading import RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from agents.approval import AgentApprovalGate, ApprovalStatus
from core.eventbus import EventBus


class PolicyAction(str, Enum):
    """Execution surfaces governed by policy rules."""

    AGENT_TASK = "agent_task"
    TOOL_INVOCATION = "tool_invocation"


class PolicyEffect(str, Enum):
    """Possible policy decisions."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyState(str, Enum):
    """Lifecycle states exposed by the policy engine."""

    STOPPED = "stopped"
    RUNNING = "running"


class ExecutionPolicyError(RuntimeError):
    """Base class for execution-policy failures."""


class PolicyRegistrationError(ExecutionPolicyError):
    """Raised when a policy rule is invalid or duplicated."""


class PolicyStateError(ExecutionPolicyError):
    """Raised when evaluation conflicts with policy-engine state."""


class PolicyDeniedError(ExecutionPolicyError):
    """Raised when policy explicitly denies an execution."""

    def __init__(self, decision: "PolicyDecision") -> None:
        self.decision = decision
        super().__init__(
            f"execution denied by policy: {decision.reason}"
        )


class PolicyApprovalRequiredError(ExecutionPolicyError):
    """Raised when execution requires owner approval."""

    def __init__(
        self,
        decision: "PolicyDecision",
        approval: Any | None = None,
    ) -> None:
        self.decision = decision
        self.approval = approval
        super().__init__(
            f"owner approval required: {decision.reason}"
        )


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """Immutable ordered execution-policy rule."""

    name: str
    effect: PolicyEffect
    actions: tuple[PolicyAction, ...]
    agents: tuple[str, ...]
    roles: tuple[str, ...]
    resources: tuple[str, ...]
    modes: tuple[str, ...]
    priority: int
    reason: str
    enabled: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe rule representation."""

        return {
            "name": self.name,
            "effect": self.effect.value,
            "actions": [
                action.value
                for action in self.actions
            ],
            "agents": list(self.agents),
            "roles": list(self.roles),
            "resources": list(self.resources),
            "modes": list(self.modes),
            "priority": self.priority,
            "reason": self.reason,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Immutable execution request evaluated by policy."""

    identifier: str
    action: PolicyAction
    agent_name: str
    agent_role: str
    resource: str
    mode: str
    attributes: Mapping[str, Any]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def create(
        cls,
        *,
        action: PolicyAction | str,
        agent_name: str,
        agent_role: str,
        resource: str,
        mode: str,
        attributes: Mapping[str, Any] | None = None,
        identifier: str | None = None,
    ) -> "PolicyRequest":
        normalized_attributes: Mapping[str, Any]

        if attributes is None:
            normalized_attributes = MappingProxyType({})
        elif isinstance(attributes, Mapping):
            normalized_attributes = MappingProxyType(
                {
                    key: _freeze_value(value)
                    for key, value in attributes.items()
                }
            )
        else:
            raise TypeError(
                "attributes must be a mapping or None"
            )

        return cls(
            identifier=(
                _validate_text(
                    identifier,
                    field_name="identifier",
                )
                if identifier is not None
                else uuid4().hex
            ),
            action=_normalize_action(action),
            agent_name=_validate_text(
                agent_name,
                field_name="agent_name",
            ),
            agent_role=_validate_text(
                agent_role,
                field_name="agent_role",
            ).lower(),
            resource=_validate_text(
                resource,
                field_name="resource",
            ).lower(),
            mode=_validate_text(
                mode,
                field_name="mode",
            ).lower(),
            attributes=normalized_attributes,
        )

    def as_dict(
        self,
        *,
        include_attributes: bool = False,
    ) -> dict[str, Any]:
        """
        Return a request representation.

        Attribute values are omitted by default so policy telemetry does not
        capture task payloads, tool arguments, or secrets.
        """

        payload = {
            "identifier": self.identifier,
            "action": self.action.value,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "resource": self.resource,
            "mode": self.mode,
            "attribute_names": sorted(self.attributes),
            "created_at": self.created_at.isoformat(),
        }

        if include_attributes:
            payload["attributes"] = _thaw_value(
                self.attributes
            )

        return payload


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Immutable result of one execution-policy evaluation."""

    request: PolicyRequest
    effect: PolicyEffect
    rule_name: str | None
    reason: str
    approval_id: str | None = None
    approved_by: str | None = None
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def allowed(self) -> bool:
        return self.effect is PolicyEffect.ALLOW

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, redacted decision representation."""

        return {
            "request": self.request.as_dict(),
            "effect": self.effect.value,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "allowed": self.allowed,
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass(slots=True)
class _RuleRecord:
    rule: PolicyRule
    sequence: int
    match_count: int = 0


class AgentExecutionPolicy:
    """
    Thread-safe, deterministic policy engine for agent and tool execution.

    Rules are evaluated by descending priority and then registration order.
    The first matching enabled rule wins. When no rule matches, the configured
    default effect is used. Approval-required decisions intentionally fail
    closed until a separate owner-approval subsystem is installed.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        approval_gate: AgentApprovalGate | None = None,
        history_limit: int = 100,
        default_effect: PolicyEffect | str = PolicyEffect.DENY,
    ) -> None:
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        if (
            approval_gate is not None
            and not isinstance(approval_gate, AgentApprovalGate)
        ):
            raise TypeError(
                "approval_gate must be an AgentApprovalGate or None"
            )

        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError(
                "history_limit must be a positive integer"
            )

        self._event_bus = event_bus
        self._approval_gate = approval_gate
        self._history_limit = history_limit
        self._default_effect = _normalize_effect(default_effect)
        self._lock = RLock()
        self._state = PolicyState.STOPPED
        self._records: dict[str, _RuleRecord] = {}
        self._sequence = 0
        self._history: deque[PolicyDecision] = deque(
            maxlen=self._history_limit
        )
        self._evaluation_count = 0
        self._allow_count = 0
        self._deny_count = 0
        self._approval_count = 0
        self._last_decision: PolicyDecision | None = None

    @property
    def state(self) -> PolicyState:
        with self._lock:
            return self._state

    @property
    def default_effect(self) -> PolicyEffect:
        return self._default_effect

    @property
    def evaluation_count(self) -> int:
        with self._lock:
            return self._evaluation_count

    def register_rule(
        self,
        name: str,
        effect: PolicyEffect | str,
        *,
        actions: Iterable[PolicyAction | str],
        agents: Iterable[str] = (),
        roles: Iterable[str] = (),
        resources: Iterable[str] = ("*",),
        modes: Iterable[str] = (),
        priority: int = 0,
        reason: str = "",
        enabled: bool = True,
    ) -> PolicyRule:
        """Register one deterministic policy rule."""

        normalized_name = _validate_text(
            name,
            field_name="name",
        )

        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer")

        if not isinstance(reason, str):
            raise TypeError("reason must be a string")

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        normalized_actions = _normalize_actions(actions)
        normalized_agents = _normalize_names(
            agents,
            field_name="agents",
            lower=False,
        )
        normalized_roles = _normalize_names(
            roles,
            field_name="roles",
            lower=True,
        )
        normalized_resources = _normalize_names(
            resources,
            field_name="resources",
            lower=True,
        )
        normalized_modes = _normalize_names(
            modes,
            field_name="modes",
            lower=True,
        )

        if not normalized_resources:
            raise PolicyRegistrationError(
                "resources cannot be empty"
            )

        rule = PolicyRule(
            name=normalized_name,
            effect=_normalize_effect(effect),
            actions=normalized_actions,
            agents=normalized_agents,
            roles=normalized_roles,
            resources=normalized_resources,
            modes=normalized_modes,
            priority=priority,
            reason=(
                reason.strip()
                or f"matched policy rule {normalized_name}"
            ),
            enabled=enabled,
        )

        with self._lock:
            if normalized_name in self._records:
                raise PolicyRegistrationError(
                    f"policy rule already registered: {normalized_name}"
                )

            self._sequence += 1
            self._records[normalized_name] = _RuleRecord(
                rule=rule,
                sequence=self._sequence,
            )

        self._publish(
            "ghostfire.execution_policy.rule_registered",
            rule.as_dict(),
        )
        return rule

    def unregister_rule(self, name: str) -> PolicyRule:
        """Remove and return one policy rule."""

        normalized = _validate_text(
            name,
            field_name="name",
        )

        with self._lock:
            try:
                record = self._records.pop(normalized)
            except KeyError as exc:
                raise PolicyRegistrationError(
                    f"policy rule is not registered: {normalized}"
                ) from exc

        self._publish(
            "ghostfire.execution_policy.rule_unregistered",
            record.rule.as_dict(),
        )
        return record.rule

    def start(self) -> bool:
        """Start policy evaluation."""

        with self._lock:
            if self._state is PolicyState.RUNNING:
                return False

            self._state = PolicyState.RUNNING
            payload = self._status_payload_locked()

        self._publish(
            "ghostfire.execution_policy.started",
            payload,
        )
        return True

    def stop(self) -> bool:
        """Stop policy evaluation."""

        with self._lock:
            if self._state is PolicyState.STOPPED:
                return False

            self._state = PolicyState.STOPPED
            payload = self._status_payload_locked()

        self._publish(
            "ghostfire.execution_policy.stopped",
            payload,
        )
        return True

    def health(self) -> bool:
        """Return whether the policy engine accepts evaluations."""

        with self._lock:
            return self._state is PolicyState.RUNNING

    def evaluate(
        self,
        request: PolicyRequest,
    ) -> PolicyDecision:
        """Evaluate one immutable request and retain its decision."""

        if not isinstance(request, PolicyRequest):
            raise TypeError("request must be a PolicyRequest")

        with self._lock:
            if self._state is not PolicyState.RUNNING:
                raise PolicyStateError(
                    "execution policy is not running"
                )

            records = sorted(
                self._records.values(),
                key=lambda record: (
                    -record.rule.priority,
                    record.sequence,
                ),
            )
            matched_record: _RuleRecord | None = None

            for record in records:
                if self._matches(record.rule, request):
                    matched_record = record
                    break

            if matched_record is None:
                effect = self._default_effect
                rule_name = None
                reason = (
                    "no policy rule matched; "
                    f"default effect is {effect.value}"
                )
            else:
                matched_record.match_count += 1
                effect = matched_record.rule.effect
                rule_name = matched_record.rule.name
                reason = matched_record.rule.reason

            decision = PolicyDecision(
                request=request,
                effect=effect,
                rule_name=rule_name,
                reason=reason,
            )
            self._history.append(decision)
            self._evaluation_count += 1
            self._last_decision = decision

            if effect is PolicyEffect.ALLOW:
                self._allow_count += 1
            elif effect is PolicyEffect.DENY:
                self._deny_count += 1
            else:
                self._approval_count += 1

        self._publish(
            "ghostfire.execution_policy.decision",
            decision.as_dict(),
        )
        return decision

    def authorize(
        self,
        *,
        action: PolicyAction | str,
        agent_name: str,
        agent_role: str,
        resource: str,
        mode: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate one request and fail closed unless allowed."""

        request = PolicyRequest.create(
            action=action,
            agent_name=agent_name,
            agent_role=agent_role,
            resource=resource,
            mode=mode,
            attributes=attributes,
        )
        decision = self.evaluate(request)

        if decision.effect is PolicyEffect.DENY:
            raise PolicyDeniedError(decision)

        if decision.effect is PolicyEffect.REQUIRE_APPROVAL:
            if self._approval_gate is None:
                raise PolicyApprovalRequiredError(decision)

            approval = self._approval_gate.authorize_or_request(
                action=request.action.value,
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                resource=request.resource,
                mode=request.mode,
                attributes=request.attributes,
                policy_rule=decision.rule_name,
                policy_reason=decision.reason,
            )

            if approval.status is ApprovalStatus.DENIED:
                denied = PolicyDecision(
                    request=request,
                    effect=PolicyEffect.DENY,
                    rule_name=decision.rule_name,
                    reason=(
                        "owner denied approval request "
                        f"{approval.identifier}"
                    ),
                    approval_id=approval.identifier,
                    approved_by=approval.decided_by,
                )
                self._replace_recorded_decision(
                    decision,
                    denied,
                )
                raise PolicyDeniedError(denied)

            if approval.status is ApprovalStatus.CONSUMED:
                approved = PolicyDecision(
                    request=request,
                    effect=PolicyEffect.ALLOW,
                    rule_name=decision.rule_name,
                    reason=(
                        "owner approval consumed for "
                        f"{approval.identifier}"
                    ),
                    approval_id=approval.identifier,
                    approved_by=approval.decided_by,
                )
                self._replace_recorded_decision(
                    decision,
                    approved,
                )
                return approved

            raise PolicyApprovalRequiredError(
                decision,
                approval,
            )

        return decision

    def _replace_recorded_decision(
        self,
        original: PolicyDecision,
        replacement: PolicyDecision,
    ) -> None:
        with self._lock:
            replaced = False

            for index in range(
                len(self._history) - 1,
                -1,
                -1,
            ):
                if self._history[index] is original:
                    self._history[index] = replacement
                    replaced = True
                    break

            if not replaced:
                self._history.append(replacement)

            if original.effect is PolicyEffect.ALLOW:
                self._allow_count = max(
                    0,
                    self._allow_count - 1,
                )
            elif original.effect is PolicyEffect.DENY:
                self._deny_count = max(
                    0,
                    self._deny_count - 1,
                )
            else:
                self._approval_count = max(
                    0,
                    self._approval_count - 1,
                )

            if replacement.effect is PolicyEffect.ALLOW:
                self._allow_count += 1
            elif replacement.effect is PolicyEffect.DENY:
                self._deny_count += 1
            else:
                self._approval_count += 1

            if self._last_decision is original:
                self._last_decision = replacement

    def list_rules(self) -> tuple[PolicyRule, ...]:
        """Return rules in effective evaluation order."""

        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: (
                    -record.rule.priority,
                    record.sequence,
                ),
            )
            return tuple(
                record.rule
                for record in records
            )

    def history(self) -> tuple[PolicyDecision, ...]:
        """Return bounded immutable decision history."""

        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe policy-engine snapshot."""

        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: (
                    -record.rule.priority,
                    record.sequence,
                ),
            )

            return {
                **self._status_payload_locked(),
                "default_effect": self._default_effect.value,
                "approval_gate_attached": (
                    self._approval_gate is not None
                ),
                "rules": [
                    {
                        **record.rule.as_dict(),
                        "match_count": record.match_count,
                    }
                    for record in records
                ],
                "last_decision": (
                    self._last_decision.as_dict()
                    if self._last_decision is not None
                    else None
                ),
            }

    @staticmethod
    def _matches(
        rule: PolicyRule,
        request: PolicyRequest,
    ) -> bool:
        if not rule.enabled:
            return False

        if request.action not in rule.actions:
            return False

        if rule.agents and request.agent_name not in rule.agents:
            return False

        if rule.roles and request.agent_role not in rule.roles:
            return False

        if rule.modes and request.mode not in rule.modes:
            return False

        return any(
            fnmatchcase(request.resource, pattern)
            for pattern in rule.resources
        )

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "healthy": self._state is PolicyState.RUNNING,
            "rule_count": len(self._records),
            "evaluation_count": self._evaluation_count,
            "allow_count": self._allow_count,
            "deny_count": self._deny_count,
            "approval_count": self._approval_count,
            "history_size": len(self._history),
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


def _normalize_actions(
    actions: Iterable[PolicyAction | str],
) -> tuple[PolicyAction, ...]:
    if isinstance(actions, str):
        raise TypeError(
            "actions must be an iterable of policy actions"
        )

    normalized: list[PolicyAction] = []

    for action in actions:
        item = _normalize_action(action)

        if item not in normalized:
            normalized.append(item)

    if not normalized:
        raise PolicyRegistrationError(
            "actions cannot be empty"
        )

    return tuple(normalized)


def _normalize_action(
    value: PolicyAction | str,
) -> PolicyAction:
    if isinstance(value, PolicyAction):
        return value

    if isinstance(value, str):
        try:
            return PolicyAction(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported policy action: {value}"
            ) from exc

    raise TypeError(
        "action must be a PolicyAction or string"
    )


def _normalize_effect(
    value: PolicyEffect | str,
) -> PolicyEffect:
    if isinstance(value, PolicyEffect):
        return value

    if isinstance(value, str):
        try:
            return PolicyEffect(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported policy effect: {value}"
            ) from exc

    raise TypeError(
        "effect must be a PolicyEffect or string"
    )


def _normalize_names(
    values: Iterable[str],
    *,
    field_name: str,
    lower: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(
            f"{field_name} must be an iterable of names"
        )

    normalized: list[str] = []

    for value in values:
        item = _validate_text(
            value,
            field_name=field_name[:-1] or field_name,
        )

        if lower:
            item = item.lower()

        if item not in normalized:
            normalized.append(item)

    return tuple(normalized)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_value(item)
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


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _thaw_value(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]

    if isinstance(value, frozenset):
        return [
            _thaw_value(item)
            for item in value
        ]

    return deepcopy(value)


def _validate_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return normalized
