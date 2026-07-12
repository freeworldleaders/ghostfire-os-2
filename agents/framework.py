"""Thread-safe AI agent execution framework for GhostFire OS."""

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

from agents.tools import AgentToolClient
from core.eventbus import EventBus


AgentHandler = Callable[["AgentTask", "AgentContext"], Any]


class AgentState(str, Enum):
    """Lifecycle states exposed by a GhostFire AI agent."""

    REGISTERED = "registered"
    STARTING = "starting"
    ONLINE = "online"
    BUSY = "busy"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"


class AgentFrameworkError(RuntimeError):
    """Base class for AI agent framework failures."""


class AgentRegistrationError(AgentFrameworkError):
    """Raised when agent registration is invalid."""


class AgentStateError(AgentFrameworkError):
    """Raised when an operation is incompatible with agent state."""


class AgentCapabilityError(AgentFrameworkError):
    """Raised when no agent can satisfy a requested capability."""


class AgentExecutionError(AgentFrameworkError):
    """Raised when an agent handler fails."""

    def __init__(
        self,
        result: "AgentResult",
        cause: Exception,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(
            f"agent {result.agent_name!r} failed task "
            f"{result.task_id!r}: {type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class AgentTask:
    """Immutable unit of work submitted to an AI agent."""

    identifier: str
    capability: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def create(
        cls,
        capability: str,
        payload: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        identifier: str | None = None,
    ) -> "AgentTask":
        normalized_capability = _validate_text(
            capability,
            field_name="capability",
        ).lower()
        normalized_identifier = (
            _validate_text(identifier, field_name="identifier")
            if identifier is not None
            else uuid4().hex
        )

        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping or None")

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        return cls(
            identifier=normalized_identifier,
            capability=normalized_capability,
            payload=_freeze_mapping(payload or {}),
            metadata=_freeze_mapping(metadata or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe task representation."""

        return {
            "identifier": self.identifier,
            "capability": self.capability,
            "payload": deepcopy(dict(self.payload)),
            "metadata": deepcopy(dict(self.metadata)),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Immutable execution context supplied to an agent handler."""

    agent_name: str
    role: str
    capabilities: tuple[str, ...]
    memory: Mapping[str, Any]
    task_metadata: Mapping[str, Any]
    tools: AgentToolClient | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe execution-context representation."""

        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "memory": deepcopy(dict(self.memory)),
            "task_metadata": deepcopy(dict(self.task_metadata)),
            "tools": (
                self.tools.snapshot()
                if self.tools is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Immutable result produced by an AI agent execution."""

    task_id: str
    agent_name: str
    capability: str
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
        """Return a JSON-safe result representation."""

        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "capability": self.capability,
            "status": self.status,
            "output": deepcopy(self.output),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class Agent:
    """
    Thread-safe AI agent runtime with lifecycle, memory, and history.

    Handlers are synchronous in R1. Multiple callers may execute the same
    online agent concurrently; state reports BUSY while any task is active.
    """

    def __init__(
        self,
        name: str,
        *,
        role: str = "general",
        capabilities: Iterable[str] = ("status",),
        handler: AgentHandler | None = None,
        tool_client: AgentToolClient | None = None,
        event_bus: EventBus | None = None,
        history_limit: int = 100,
        memory_limit: int = 100,
    ) -> None:
        self._name = _validate_text(name, field_name="name")
        self._role = _validate_text(role, field_name="role").lower()
        self._capabilities = _normalize_capabilities(capabilities)

        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable or None")

        if (
            tool_client is not None
            and not isinstance(tool_client, AgentToolClient)
        ):
            raise TypeError(
                "tool_client must be an AgentToolClient or None"
            )

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._history_limit = _validate_positive_int(
            history_limit,
            field_name="history_limit",
        )
        self._memory_limit = _validate_positive_int(
            memory_limit,
            field_name="memory_limit",
        )
        self._handler = handler or self._default_handler
        self._tool_client = tool_client
        self._event_bus = event_bus
        self._lock = RLock()
        self._state = AgentState.REGISTERED
        self._history: deque[AgentResult] = deque(
            maxlen=self._history_limit
        )
        self._memory: dict[str, Any] = {}
        self._active_count = 0
        self._execution_count = 0
        self._failure_count = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._capabilities

    @property
    def state(self) -> AgentState:
        with self._lock:
            return self._state

    @property
    def execution_count(self) -> int:
        with self._lock:
            return self._execution_count

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def supports(self, capability: str) -> bool:
        """Return whether this agent advertises a capability."""

        normalized = _validate_text(
            capability,
            field_name="capability",
        ).lower()
        return normalized in self._capabilities

    def start(self) -> bool:
        """Move the agent into the online state."""

        with self._lock:
            if self._state in {AgentState.ONLINE, AgentState.BUSY}:
                return False

            if self._active_count:
                raise AgentStateError(
                    "an agent with active tasks cannot be started"
                )

            self._state = AgentState.STARTING
            self._publish(
                "ghostfire.agent.starting",
                self._status_payload_locked(),
            )
            self._last_error = None
            self._state = AgentState.ONLINE
            payload = self._status_payload_locked()

        print(f"{self._name} online")
        self._publish("ghostfire.agent.started", payload)
        return True

    def run(self) -> bool:
        """Compatibility alias for the original Agent.run contract."""

        return self.start()

    def stop(self) -> bool:
        """Stop an idle agent."""

        with self._lock:
            if self._state in {
                AgentState.REGISTERED,
                AgentState.STOPPED,
            }:
                if self._state is AgentState.REGISTERED:
                    self._state = AgentState.STOPPED
                    payload = self._status_payload_locked()
                    self._publish(
                        "ghostfire.agent.stopped",
                        payload,
                    )
                    return True

                return False

            if self._active_count:
                raise AgentStateError(
                    "an agent with active tasks cannot be stopped"
                )

            self._state = AgentState.STOPPING
            self._publish(
                "ghostfire.agent.stopping",
                self._status_payload_locked(),
            )
            self._state = AgentState.STOPPED
            payload = self._status_payload_locked()

        self._publish("ghostfire.agent.stopped", payload)
        return True

    def health(self) -> bool:
        """Return whether the agent can accept work."""

        with self._lock:
            return self._state in {
                AgentState.ONLINE,
                AgentState.BUSY,
            }

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute one task and retain an immutable result."""

        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask")

        if task.capability not in self._capabilities:
            raise AgentCapabilityError(
                f"agent {self._name!r} does not support "
                f"{task.capability!r}"
            )

        with self._lock:
            if self._state not in {
                AgentState.ONLINE,
                AgentState.BUSY,
            }:
                raise AgentStateError(
                    f"agent {self._name!r} is not online"
                )

            self._active_count += 1
            self._state = AgentState.BUSY
            started_at = datetime.now(timezone.utc)
            context = AgentContext(
                agent_name=self._name,
                role=self._role,
                capabilities=self._capabilities,
                memory=_freeze_mapping(self._memory),
                task_metadata=_freeze_mapping(task.metadata),
                tools=self._tool_client,
            )
            started_payload = {
                "agent": self._name,
                "role": self._role,
                "task": task.as_dict(),
                "active_count": self._active_count,
            }

        self._publish(
            "ghostfire.agent.task.started",
            started_payload,
        )

        try:
            output = self._handler(task, context)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc)
            error = f"{type(exc).__name__}: {exc}"
            result = AgentResult(
                task_id=task.identifier,
                agent_name=self._name,
                capability=task.capability,
                status="failed",
                output=None,
                error=error,
                started_at=started_at,
                completed_at=completed_at,
            )

            with self._lock:
                self._active_count -= 1
                self._failure_count += 1
                self._last_error = error
                self._history.append(result)
                self._state = (
                    AgentState.BUSY
                    if self._active_count
                    else AgentState.DEGRADED
                )
                payload = self._result_payload_locked(result)

            self._publish(
                "ghostfire.agent.task.failed",
                payload,
            )
            raise AgentExecutionError(result, exc) from exc

        completed_at = datetime.now(timezone.utc)
        result = AgentResult(
            task_id=task.identifier,
            agent_name=self._name,
            capability=task.capability,
            status="completed",
            output=deepcopy(output),
            error=None,
            started_at=started_at,
            completed_at=completed_at,
        )

        with self._lock:
            self._active_count -= 1
            self._execution_count += 1
            self._history.append(result)
            self._state = (
                AgentState.BUSY
                if self._active_count
                else AgentState.ONLINE
            )
            payload = self._result_payload_locked(result)

        self._publish(
            "ghostfire.agent.task.completed",
            payload,
        )
        return result

    def set_memory(self, key: str, value: Any) -> None:
        """Store a bounded memory entry."""

        normalized_key = _validate_text(key, field_name="key")

        with self._lock:
            if normalized_key in self._memory:
                self._memory.pop(normalized_key)

            self._memory[normalized_key] = deepcopy(value)

            while len(self._memory) > self._memory_limit:
                oldest_key = next(iter(self._memory))
                self._memory.pop(oldest_key)

            payload = {
                "agent": self._name,
                "key": normalized_key,
                "memory_size": len(self._memory),
            }

        self._publish(
            "ghostfire.agent.memory.updated",
            payload,
        )

    def get_memory(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a copy of one memory entry."""

        normalized_key = _validate_text(key, field_name="key")

        with self._lock:
            return deepcopy(
                self._memory.get(normalized_key, default)
            )

    def delete_memory(self, key: str) -> bool:
        """Delete one memory entry."""

        normalized_key = _validate_text(key, field_name="key")

        with self._lock:
            existed = normalized_key in self._memory

            if existed:
                self._memory.pop(normalized_key)

            payload = {
                "agent": self._name,
                "key": normalized_key,
                "memory_size": len(self._memory),
            }

        if existed:
            self._publish(
                "ghostfire.agent.memory.deleted",
                payload,
            )

        return existed

    def clear_memory(self) -> int:
        """Clear memory and return the number of entries removed."""

        with self._lock:
            removed = len(self._memory)
            self._memory.clear()
            payload = {
                "agent": self._name,
                "removed_count": removed,
            }

        self._publish(
            "ghostfire.agent.memory.cleared",
            payload,
        )
        return removed

    def memory_snapshot(self) -> dict[str, Any]:
        """Return a detached memory snapshot."""

        with self._lock:
            return deepcopy(self._memory)

    def history(self) -> tuple[AgentResult, ...]:
        """Return immutable execution history."""

        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe agent snapshot."""

        with self._lock:
            return self._status_payload_locked()

    def _default_handler(
        self,
        task: AgentTask,
        context: AgentContext,
    ) -> dict[str, Any]:
        return {
            "agent": self._name,
            "role": context.role,
            "capability": task.capability,
            "payload": deepcopy(dict(task.payload)),
        }

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "role": self._role,
            "capabilities": list(self._capabilities),
            "state": self._state.value,
            "healthy": self._state in {
                AgentState.ONLINE,
                AgentState.BUSY,
            },
            "active_count": self._active_count,
            "execution_count": self._execution_count,
            "failure_count": self._failure_count,
            "history_size": len(self._history),
            "memory_size": len(self._memory),
            "tools": (
                self._tool_client.snapshot()
                if self._tool_client is not None
                else None
            ),
            "last_error": self._last_error,
        }

    def _result_payload_locked(
        self,
        result: AgentResult,
    ) -> dict[str, Any]:
        return {
            "agent": self._name,
            "role": self._role,
            "result": result.as_dict(),
            "state": self._state.value,
            "active_count": self._active_count,
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


def _normalize_capabilities(
    capabilities: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(capabilities, str):
        raise TypeError(
            "capabilities must be an iterable of names"
        )

    normalized: list[str] = []

    for capability in capabilities:
        value = _validate_text(
            capability,
            field_name="capability",
        ).lower()

        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("capabilities cannot be empty")

    return tuple(normalized)


def _freeze_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(deepcopy(dict(value)))


def _validate_positive_int(
    value: int,
    *,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 1:
        raise ValueError(f"{field_name} must be positive")

    return value


def _validate_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return normalized
