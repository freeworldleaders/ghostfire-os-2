"""Secure, typed tool registry for GhostFire AI agents."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from core.eventbus import EventBus


ToolHandler = Callable[..., Any]


class ToolMode(str, Enum):
    """Operational modes used by the tool safety gate."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"


class ToolRegistryState(str, Enum):
    """Lifecycle states exposed by the tool registry."""

    STOPPED = "stopped"
    RUNNING = "running"


class AgentToolError(RuntimeError):
    """Base class for agent-tool failures."""


class ToolRegistrationError(AgentToolError):
    """Raised when a tool definition is invalid or duplicated."""


class ToolStateError(AgentToolError):
    """Raised when an operation conflicts with registry state."""


class ToolAuthorizationError(AgentToolError):
    """Raised when an agent is not authorized to use a tool."""


class ToolValidationError(AgentToolError):
    """Raised when tool arguments do not match the registered schema."""


class ToolExecutionError(AgentToolError):
    """Raised when a registered tool handler fails."""

    def __init__(
        self,
        result: "ToolResult",
        cause: Exception,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(
            f"tool {result.tool_name!r} failed invocation "
            f"{result.invocation_id!r}: "
            f"{type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Immutable public definition of one registered tool."""

    name: str
    description: str
    parameters: Mapping[str, tuple[type, ...]]
    required: tuple[str, ...]
    mode: ToolMode
    allowed_agents: tuple[str, ...]
    allowed_roles: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe tool definition."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                parameter: {
                    "types": [
                        accepted_type.__name__
                        for accepted_type in accepted_types
                    ],
                    "required": parameter in self.required,
                }
                for parameter, accepted_types
                in self.parameters.items()
            },
            "mode": self.mode.value,
            "allowed_agents": list(self.allowed_agents),
            "allowed_roles": list(self.allowed_roles),
        }


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Immutable invocation envelope supplied to registry telemetry."""

    identifier: str
    tool_name: str
    agent_name: str
    agent_role: str
    arguments: Mapping[str, Any]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def as_dict(
        self,
        *,
        include_arguments: bool = False,
    ) -> dict[str, Any]:
        """
        Return an invocation representation.

        Raw argument values are omitted by default so secrets do not enter
        telemetry or operational snapshots.
        """

        payload = {
            "identifier": self.identifier,
            "tool_name": self.tool_name,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "argument_names": sorted(self.arguments),
            "created_at": self.created_at.isoformat(),
        }

        if include_arguments:
            payload["arguments"] = _thaw_value(self.arguments)

        return payload


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Immutable result retained by the tool registry."""

    invocation_id: str
    tool_name: str
    agent_name: str
    status: str
    output: Any
    error: str | None
    started_at: datetime
    completed_at: datetime

    @property
    def duration_seconds(self) -> float:
        """Return elapsed execution time in seconds."""

        return max(
            0.0,
            (self.completed_at - self.started_at).total_seconds(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached result representation."""

        return {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "agent_name": self.agent_name,
            "status": self.status,
            "output": _thaw_value(self.output),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class _ToolRecord:
    definition: ToolDefinition
    handler: ToolHandler
    sequence: int
    invocation_count: int = 0
    failure_count: int = 0
    active_count: int = 0
    last_error: str | None = None


class AgentToolClient:
    """Agent-bound, least-privilege view of an AgentToolRegistry."""

    def __init__(
        self,
        registry: "AgentToolRegistry",
        *,
        agent_name: str,
        agent_role: str,
        allowed_tools: Iterable[str],
    ) -> None:
        if not isinstance(registry, AgentToolRegistry):
            raise TypeError("registry must be an AgentToolRegistry")

        if isinstance(allowed_tools, str):
            raise TypeError("allowed_tools must be an iterable of names")

        normalized_tools: list[str] = []

        for tool_name in allowed_tools:
            normalized = _normalize_tool_name(tool_name)

            if normalized not in normalized_tools:
                normalized_tools.append(normalized)

        self._registry = registry
        self._agent_name = _validate_text(
            agent_name,
            field_name="agent_name",
        )
        self._agent_role = _validate_text(
            agent_role,
            field_name="agent_role",
        ).lower()
        self._allowed_tools = tuple(normalized_tools)

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def agent_role(self) -> str:
        return self._agent_role

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return self._allowed_tools

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Invoke one authorized registered tool."""

        return self._registry.invoke(
            tool_name,
            arguments,
            agent_name=self._agent_name,
            agent_role=self._agent_role,
            client_allowlist=self._allowed_tools,
        )

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Return tools visible to this client."""

        return self._registry.list_authorized(
            agent_name=self._agent_name,
            agent_role=self._agent_role,
            client_allowlist=self._allowed_tools,
        )

    def supports(self, tool_name: str) -> bool:
        """Return whether this client can currently see a tool."""

        normalized = _normalize_tool_name(tool_name)
        return any(
            definition.name == normalized
            for definition in self.list_tools()
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe least-privilege client snapshot."""

        return {
            "agent_name": self._agent_name,
            "agent_role": self._agent_role,
            "allowed_tools": list(self._allowed_tools),
        }


class AgentToolRegistry:
    """
    Thread-safe registry for typed and permission-gated agent tools.

    Tool handlers execute synchronously in R1. The registry validates exact
    argument schemas, enforces client/agent/role allowlists, blocks mutating
    tools unless owner configuration explicitly enables them, and emits
    redacted telemetry.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        history_limit: int = 100,
        allow_mutating: bool = False,
    ) -> None:
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError("history_limit must be a positive integer")

        if not isinstance(allow_mutating, bool):
            raise TypeError("allow_mutating must be a boolean")

        self._event_bus = event_bus
        self._history_limit = history_limit
        self._allow_mutating = allow_mutating
        self._lock = RLock()
        self._state = ToolRegistryState.STOPPED
        self._records: dict[str, _ToolRecord] = {}
        self._sequence = 0
        self._history: deque[ToolResult] = deque(
            maxlen=self._history_limit
        )
        self._invocation_count = 0
        self._failure_count = 0
        self._active_count = 0
        self._last_error: str | None = None

    @property
    def state(self) -> ToolRegistryState:
        with self._lock:
            return self._state

    @property
    def invocation_count(self) -> int:
        with self._lock:
            return self._invocation_count

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    @property
    def allow_mutating(self) -> bool:
        return self._allow_mutating

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        parameters: Mapping[
            str,
            type | tuple[type, ...],
        ] | None = None,
        required: Iterable[str] = (),
        mode: ToolMode | str = ToolMode.READ_ONLY,
        allowed_agents: Iterable[str] = (),
        allowed_roles: Iterable[str] = (),
    ) -> ToolDefinition:
        """Register one typed tool and return its immutable definition."""

        normalized_name = _normalize_tool_name(name)

        if not callable(handler):
            raise TypeError("handler must be callable")

        if not isinstance(description, str):
            raise TypeError("description must be a string")

        normalized_parameters = _normalize_parameters(parameters or {})
        normalized_required = _normalize_name_iterable(
            required,
            field_name="required",
            normalizer=_validate_parameter_name,
        )

        unknown_required = tuple(
            parameter
            for parameter in normalized_required
            if parameter not in normalized_parameters
        )

        if unknown_required:
            raise ToolRegistrationError(
                "required parameters are not defined: "
                + ", ".join(unknown_required)
            )

        normalized_mode = _normalize_mode(mode)
        normalized_agents = _normalize_name_iterable(
            allowed_agents,
            field_name="allowed_agents",
            normalizer=lambda value: _validate_text(
                value,
                field_name="allowed_agent",
            ),
        )
        normalized_roles = _normalize_name_iterable(
            allowed_roles,
            field_name="allowed_roles",
            normalizer=lambda value: _validate_text(
                value,
                field_name="allowed_role",
            ).lower(),
        )

        definition = ToolDefinition(
            name=normalized_name,
            description=description.strip(),
            parameters=MappingProxyType(
                dict(normalized_parameters)
            ),
            required=normalized_required,
            mode=normalized_mode,
            allowed_agents=normalized_agents,
            allowed_roles=normalized_roles,
        )

        with self._lock:
            if normalized_name in self._records:
                raise ToolRegistrationError(
                    f"tool already registered: {normalized_name}"
                )

            self._sequence += 1
            self._records[normalized_name] = _ToolRecord(
                definition=definition,
                handler=handler,
                sequence=self._sequence,
            )

        self._publish(
            "ghostfire.agent_tool.registered",
            definition.as_dict(),
        )
        return definition

    def unregister(self, name: str) -> ToolDefinition:
        """Remove an inactive tool."""

        normalized = _normalize_tool_name(name)

        with self._lock:
            record = self._require_record_locked(normalized)

            if record.active_count:
                raise ToolStateError(
                    "an active tool cannot be unregistered"
                )

            self._records.pop(normalized)
            definition = record.definition

        self._publish(
            "ghostfire.agent_tool.unregistered",
            definition.as_dict(),
        )
        return definition

    def client(
        self,
        *,
        agent_name: str,
        agent_role: str,
        allowed_tools: Iterable[str],
    ) -> AgentToolClient:
        """Create a least-privilege client bound to one agent identity."""

        return AgentToolClient(
            self,
            agent_name=agent_name,
            agent_role=agent_role,
            allowed_tools=allowed_tools,
        )

    def start(self) -> bool:
        """Start accepting tool invocations."""

        with self._lock:
            if self._state is ToolRegistryState.RUNNING:
                return False

            self._state = ToolRegistryState.RUNNING
            self._last_error = None
            payload = self._status_payload_locked()

        self._publish("ghostfire.agent_tool_registry.started", payload)
        return True

    def stop(self) -> bool:
        """Stop an idle registry."""

        with self._lock:
            if self._state is ToolRegistryState.STOPPED:
                return False

            if self._active_count:
                raise ToolStateError(
                    "a registry with active invocations cannot be stopped"
                )

            self._state = ToolRegistryState.STOPPED
            payload = self._status_payload_locked()

        self._publish("ghostfire.agent_tool_registry.stopped", payload)
        return True

    def health(self) -> bool:
        """Return whether the registry accepts invocations."""

        with self._lock:
            return self._state is ToolRegistryState.RUNNING

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        agent_name: str,
        agent_role: str,
        client_allowlist: Iterable[str],
    ) -> ToolResult:
        """Validate, authorize, and execute one tool invocation."""

        normalized_name = _normalize_tool_name(name)
        normalized_agent = _validate_text(
            agent_name,
            field_name="agent_name",
        )
        normalized_role = _validate_text(
            agent_role,
            field_name="agent_role",
        ).lower()

        if arguments is None:
            argument_values: Mapping[str, Any] = {}
        elif isinstance(arguments, Mapping):
            argument_values = arguments
        else:
            raise TypeError("arguments must be a mapping or None")

        if isinstance(client_allowlist, str):
            raise TypeError(
                "client_allowlist must be an iterable of names"
            )

        normalized_client_allowlist = tuple(
            dict.fromkeys(
                _normalize_tool_name(tool_name)
                for tool_name in client_allowlist
            )
        )

        with self._lock:
            if self._state is not ToolRegistryState.RUNNING:
                raise ToolStateError("tool registry is not running")

            record = self._require_record_locked(normalized_name)

            try:
                self._authorize_locked(
                    record.definition,
                    agent_name=normalized_agent,
                    agent_role=normalized_role,
                    client_allowlist=normalized_client_allowlist,
                )
                validated_arguments = self._validate_arguments_locked(
                    record.definition,
                    argument_values,
                )
            except (
                ToolAuthorizationError,
                ToolValidationError,
            ) as exc:
                self._publish(
                    "ghostfire.agent_tool.invocation.denied",
                    {
                        "tool_name": normalized_name,
                        "agent_name": normalized_agent,
                        "agent_role": normalized_role,
                        "argument_names": sorted(argument_values),
                        "error_type": type(exc).__name__,
                    },
                )
                raise

            invocation = ToolInvocation(
                identifier=uuid4().hex,
                tool_name=normalized_name,
                agent_name=normalized_agent,
                agent_role=normalized_role,
                arguments=validated_arguments,
            )
            record.active_count += 1
            record.invocation_count += 1
            self._active_count += 1
            self._invocation_count += 1
            started_at = datetime.now(timezone.utc)

        self._publish(
            "ghostfire.agent_tool.invocation.started",
            {
                **invocation.as_dict(),
                "mode": record.definition.mode.value,
            },
        )

        try:
            output = record.handler(
                **_thaw_value(invocation.arguments)
            )
            frozen_output = _freeze_value(output)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc)
            error = f"{type(exc).__name__}: {exc}"
            result = ToolResult(
                invocation_id=invocation.identifier,
                tool_name=normalized_name,
                agent_name=normalized_agent,
                status="failed",
                output=None,
                error=error,
                started_at=started_at,
                completed_at=completed_at,
            )

            with self._lock:
                active_record = self._require_record_locked(
                    normalized_name
                )
                active_record.active_count -= 1
                active_record.failure_count += 1
                active_record.last_error = error
                self._active_count -= 1
                self._failure_count += 1
                self._last_error = error
                self._history.append(result)

            self._publish(
                "ghostfire.agent_tool.invocation.failed",
                {
                    "invocation_id": invocation.identifier,
                    "tool_name": normalized_name,
                    "agent_name": normalized_agent,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise ToolExecutionError(result, exc) from exc

        completed_at = datetime.now(timezone.utc)
        result = ToolResult(
            invocation_id=invocation.identifier,
            tool_name=normalized_name,
            agent_name=normalized_agent,
            status="completed",
            output=frozen_output,
            error=None,
            started_at=started_at,
            completed_at=completed_at,
        )

        with self._lock:
            active_record = self._require_record_locked(normalized_name)
            active_record.active_count -= 1
            active_record.last_error = None
            self._active_count -= 1
            self._last_error = None
            self._history.append(result)

        self._publish(
            "ghostfire.agent_tool.invocation.completed",
            {
                "invocation_id": invocation.identifier,
                "tool_name": normalized_name,
                "agent_name": normalized_agent,
                "status": "completed",
                "result_type": type(output).__name__,
            },
        )
        return result

    def get(self, name: str) -> ToolDefinition:
        """Return one immutable tool definition."""

        normalized = _normalize_tool_name(name)

        with self._lock:
            return self._require_record_locked(normalized).definition

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Return all definitions in registration order."""

        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: record.sequence,
            )
            return tuple(
                record.definition
                for record in records
            )

    def list_authorized(
        self,
        *,
        agent_name: str,
        agent_role: str,
        client_allowlist: Iterable[str],
    ) -> tuple[ToolDefinition, ...]:
        """Return definitions visible to one agent-bound client."""

        normalized_agent = _validate_text(
            agent_name,
            field_name="agent_name",
        )
        normalized_role = _validate_text(
            agent_role,
            field_name="agent_role",
        ).lower()

        if isinstance(client_allowlist, str):
            raise TypeError(
                "client_allowlist must be an iterable of names"
            )

        allowed = {
            _normalize_tool_name(tool_name)
            for tool_name in client_allowlist
        }
        visible: list[ToolDefinition] = []

        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: record.sequence,
            )

            for record in records:
                try:
                    self._authorize_locked(
                        record.definition,
                        agent_name=normalized_agent,
                        agent_role=normalized_role,
                        client_allowlist=tuple(allowed),
                    )
                except ToolAuthorizationError:
                    continue

                visible.append(record.definition)

        return tuple(visible)

    def history(self) -> tuple[ToolResult, ...]:
        """Return bounded immutable invocation history."""

        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe operational snapshot."""

        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: record.sequence,
            )

            return {
                **self._status_payload_locked(),
                "allow_mutating": self._allow_mutating,
                "tools": [
                    {
                        **record.definition.as_dict(),
                        "invocation_count": record.invocation_count,
                        "failure_count": record.failure_count,
                        "active_count": record.active_count,
                        "last_error": record.last_error,
                    }
                    for record in records
                ],
            }

    def _authorize_locked(
        self,
        definition: ToolDefinition,
        *,
        agent_name: str,
        agent_role: str,
        client_allowlist: tuple[str, ...],
    ) -> None:
        if definition.name not in client_allowlist:
            raise ToolAuthorizationError(
                f"client is not granted tool {definition.name!r}"
            )

        if (
            definition.allowed_agents
            and agent_name not in definition.allowed_agents
        ):
            raise ToolAuthorizationError(
                f"agent {agent_name!r} is not allowed to use "
                f"{definition.name!r}"
            )

        if (
            definition.allowed_roles
            and agent_role not in definition.allowed_roles
        ):
            raise ToolAuthorizationError(
                f"role {agent_role!r} is not allowed to use "
                f"{definition.name!r}"
            )

        if (
            definition.mode is ToolMode.MUTATING
            and not self._allow_mutating
        ):
            raise ToolAuthorizationError(
                "mutating tools are disabled by owner configuration"
            )

    @staticmethod
    def _validate_arguments_locked(
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        normalized_arguments: dict[str, Any] = {}

        for key, value in arguments.items():
            parameter = _validate_parameter_name(key)
            normalized_arguments[parameter] = value

        unknown = tuple(
            parameter
            for parameter in normalized_arguments
            if parameter not in definition.parameters
        )

        if unknown:
            raise ToolValidationError(
                "unknown parameters: " + ", ".join(unknown)
            )

        missing = tuple(
            parameter
            for parameter in definition.required
            if parameter not in normalized_arguments
        )

        if missing:
            raise ToolValidationError(
                "missing required parameters: " + ", ".join(missing)
            )

        for parameter, value in normalized_arguments.items():
            accepted_types = definition.parameters[parameter]

            if not _matches_types(value, accepted_types):
                expected = " or ".join(
                    accepted_type.__name__
                    for accepted_type in accepted_types
                )
                raise ToolValidationError(
                    f"parameter {parameter!r} must be {expected}"
                )

        return MappingProxyType(
            {
                key: _freeze_value(value)
                for key, value in normalized_arguments.items()
            }
        )

    def _require_record_locked(self, name: str) -> _ToolRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise ToolRegistrationError(
                f"tool is not registered: {name}"
            ) from exc

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "healthy": self._state is ToolRegistryState.RUNNING,
            "tool_count": len(self._records),
            "active_count": self._active_count,
            "invocation_count": self._invocation_count,
            "failure_count": self._failure_count,
            "history_size": len(self._history),
            "last_error": self._last_error,
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


def _normalize_tool_name(value: Any) -> str:
    normalized = _validate_text(
        value,
        field_name="tool_name",
    ).lower()

    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in normalized
    ):
        raise ValueError(
            "tool_name may contain only letters, numbers, dots, "
            "underscores, and hyphens"
        )

    return normalized


def _validate_parameter_name(value: Any) -> str:
    normalized = _validate_text(
        value,
        field_name="parameter_name",
    )

    if not normalized.isidentifier():
        raise ValueError(
            f"parameter_name is not a valid identifier: {normalized}"
        )

    return normalized


def _normalize_parameters(
    parameters: Mapping[
        str,
        type | tuple[type, ...],
    ],
) -> dict[str, tuple[type, ...]]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")

    normalized: dict[str, tuple[type, ...]] = {}

    for parameter, accepted in parameters.items():
        name = _validate_parameter_name(parameter)

        if isinstance(accepted, type):
            accepted_types = (accepted,)
        elif (
            isinstance(accepted, tuple)
            and accepted
            and all(
                isinstance(item, type)
                for item in accepted
            )
        ):
            accepted_types = accepted
        else:
            raise TypeError(
                f"parameter {name!r} must map to a type "
                "or non-empty tuple of types"
            )

        normalized[name] = tuple(dict.fromkeys(accepted_types))

    return normalized


def _normalize_name_iterable(
    values: Iterable[str],
    *,
    field_name: str,
    normalizer: Callable[[Any], str],
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be an iterable of names")

    normalized: list[str] = []

    for value in values:
        item = normalizer(value)

        if item not in normalized:
            normalized.append(item)

    return tuple(normalized)


def _normalize_mode(value: ToolMode | str) -> ToolMode:
    if isinstance(value, ToolMode):
        return value

    if isinstance(value, str):
        try:
            return ToolMode(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported tool mode: {value}"
            ) from exc

    raise TypeError("mode must be a ToolMode or string")


def _matches_types(
    value: Any,
    accepted_types: tuple[type, ...],
) -> bool:
    if (
        isinstance(value, bool)
        and bool not in accepted_types
        and any(
            accepted_type in {int, float}
            for accepted_type in accepted_types
        )
    ):
        return False

    return isinstance(value, accepted_types)


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
