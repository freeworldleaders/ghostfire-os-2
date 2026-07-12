"""Deterministic task orchestration for GhostFire AI agents."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock, RLock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from agents.framework import AgentFrameworkError, AgentResult
from agents.registry import AgentRegistry
from core.eventbus import EventBus


class OrchestratorState(str, Enum):
    """Lifecycle states exposed by the task orchestrator."""

    STOPPED = "stopped"
    RUNNING = "running"
    EXECUTING = "executing"


class OrchestratedTaskState(str, Enum):
    """Execution states for one orchestrated task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class OrchestratorError(RuntimeError):
    """Base class for task-orchestrator failures."""


class OrchestratorStateError(OrchestratorError):
    """Raised when an operation conflicts with orchestrator state."""


class OrchestratorPlanError(OrchestratorError):
    """Raised when a task plan is invalid."""


class OrchestratorCapacityError(OrchestratorError):
    """Raised when the configured task capacity would be exceeded."""


@dataclass(frozen=True, slots=True)
class OrchestratedTask:
    """Immutable task definition managed by the orchestrator."""

    identifier: str
    capability: str
    payload: Mapping[str, Any]
    preferred_agent: str | None
    metadata: Mapping[str, Any]
    dependencies: tuple[str, ...]
    max_attempts: int
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def create(
        cls,
        capability: str,
        payload: Mapping[str, Any] | None = None,
        *,
        identifier: str | None = None,
        preferred_agent: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dependencies: Iterable[str] = (),
        max_attempts: int = 1,
    ) -> "OrchestratedTask":
        normalized_identifier = (
            _validate_text(identifier, field_name="identifier")
            if identifier is not None
            else uuid4().hex
        )
        normalized_capability = _validate_text(
            capability,
            field_name="capability",
        ).lower()
        normalized_preferred = (
            _validate_text(
                preferred_agent,
                field_name="preferred_agent",
            )
            if preferred_agent is not None
            else None
        )

        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping or None")

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        if isinstance(dependencies, str):
            raise TypeError("dependencies must be an iterable of task IDs")

        normalized_dependencies: list[str] = []

        for dependency in dependencies:
            normalized = _validate_text(
                dependency,
                field_name="dependency",
            )

            if normalized == normalized_identifier:
                raise OrchestratorPlanError(
                    "a task cannot depend on itself"
                )

            if normalized in normalized_dependencies:
                raise OrchestratorPlanError(
                    f"duplicate dependency: {normalized}"
                )

            normalized_dependencies.append(normalized)

        attempts = _validate_positive_int(
            max_attempts,
            field_name="max_attempts",
        )

        return cls(
            identifier=normalized_identifier,
            capability=normalized_capability,
            payload=_freeze_mapping(payload or {}),
            preferred_agent=normalized_preferred,
            metadata=_freeze_mapping(metadata or {}),
            dependencies=tuple(normalized_dependencies),
            max_attempts=attempts,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe task definition."""

        return {
            "identifier": self.identifier,
            "capability": self.capability,
            "payload": deepcopy(dict(self.payload)),
            "preferred_agent": self.preferred_agent,
            "metadata": deepcopy(dict(self.metadata)),
            "dependencies": list(self.dependencies),
            "max_attempts": self.max_attempts,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class OrchestratedTaskSnapshot:
    """Immutable external view of one managed task."""

    task: OrchestratedTask
    state: OrchestratedTaskState
    sequence: int
    attempts: int
    agent_name: str | None
    result: AgentResult | None
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe task snapshot."""

        return {
            "task": self.task.as_dict(),
            "state": self.state.value,
            "sequence": self.sequence,
            "attempts": self.attempts,
            "agent_name": self.agent_name,
            "result": (
                self.result.as_dict()
                if self.result is not None
                else None
            ),
            "error": self.error,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at is not None
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class OrchestrationRun:
    """Immutable summary of one deterministic execution pass."""

    identifier: str
    status: str
    task_ids: tuple[str, ...]
    counts: Mapping[str, int]
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
        """Return a JSON-safe run summary."""

        return {
            "identifier": self.identifier,
            "status": self.status,
            "task_ids": list(self.task_ids),
            "counts": dict(self.counts),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class _TaskRecord:
    task: OrchestratedTask
    sequence: int
    state: OrchestratedTaskState = OrchestratedTaskState.PENDING
    attempts: int = 0
    agent_name: str | None = None
    result: AgentResult | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AgentTaskOrchestrator:
    """
    Deterministic dependency-aware coordinator for AgentRegistry tasks.

    R1 executes synchronously, preserves submission order, retries through
    healthy capable agents, blocks dependents after terminal failures, and
    continues independent work.
    """

    _TERMINAL_STATES = {
        OrchestratedTaskState.COMPLETED,
        OrchestratedTaskState.FAILED,
        OrchestratedTaskState.BLOCKED,
        OrchestratedTaskState.CANCELLED,
    }
    _BLOCKING_STATES = {
        OrchestratedTaskState.FAILED,
        OrchestratedTaskState.BLOCKED,
        OrchestratedTaskState.CANCELLED,
    }

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        event_bus: EventBus | None = None,
        history_limit: int = 100,
        max_tasks: int = 1_000,
    ) -> None:
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an AgentRegistry")

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._history_limit = _validate_positive_int(
            history_limit,
            field_name="history_limit",
        )
        self._max_tasks = _validate_positive_int(
            max_tasks,
            field_name="max_tasks",
        )
        self._registry = registry
        self._event_bus = event_bus
        self._lock = RLock()
        self._execution_lock = Lock()
        self._state = OrchestratorState.STOPPED
        self._records: dict[str, _TaskRecord] = {}
        self._sequence = 0
        self._run_history: deque[OrchestrationRun] = deque(
            maxlen=self._history_limit
        )
        self._run_count = 0
        self._last_error: str | None = None

    @property
    def state(self) -> OrchestratorState:
        with self._lock:
            return self._state

    @property
    def run_count(self) -> int:
        with self._lock:
            return self._run_count

    @property
    def task_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def start(self) -> bool:
        """Start accepting execution requests."""

        with self._lock:
            if self._state in {
                OrchestratorState.RUNNING,
                OrchestratorState.EXECUTING,
            }:
                return False

            self._state = OrchestratorState.RUNNING
            self._last_error = None
            payload = self._status_payload_locked()

        self._publish("ghostfire.orchestrator.started", payload)
        return True

    def stop(self) -> bool:
        """Stop an idle orchestrator."""

        with self._lock:
            if self._state is OrchestratorState.STOPPED:
                return False

            if self._state is OrchestratorState.EXECUTING:
                raise OrchestratorStateError(
                    "an executing orchestrator cannot be stopped"
                )

            self._state = OrchestratorState.STOPPED
            payload = self._status_payload_locked()

        self._publish("ghostfire.orchestrator.stopped", payload)
        return True

    def health(self) -> bool:
        """Return whether the orchestrator can execute work."""

        with self._lock:
            return self._state in {
                OrchestratorState.RUNNING,
                OrchestratorState.EXECUTING,
            }

    def submit(
        self,
        capability: str,
        payload: Mapping[str, Any] | None = None,
        *,
        identifier: str | None = None,
        preferred_agent: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dependencies: Iterable[str] = (),
        max_attempts: int = 1,
    ) -> OrchestratedTaskSnapshot:
        """Create and atomically submit one task."""

        task = OrchestratedTask.create(
            capability,
            payload,
            identifier=identifier,
            preferred_agent=preferred_agent,
            metadata=metadata,
            dependencies=dependencies,
            max_attempts=max_attempts,
        )

        return self.submit_plan((task,))[0]

    def submit_plan(
        self,
        tasks: Iterable[OrchestratedTask],
    ) -> tuple[OrchestratedTaskSnapshot, ...]:
        """Atomically validate and submit a dependency plan."""

        if isinstance(tasks, (str, bytes)):
            raise TypeError("tasks must be an iterable of OrchestratedTask")

        proposed = tuple(tasks)

        if not proposed:
            raise OrchestratorPlanError("task plan cannot be empty")

        if not all(
            isinstance(task, OrchestratedTask)
            for task in proposed
        ):
            raise TypeError(
                "every plan item must be an OrchestratedTask"
            )

        with self._lock:
            if len(self._records) + len(proposed) > self._max_tasks:
                raise OrchestratorCapacityError(
                    "task plan exceeds configured capacity"
                )

            existing_ids = set(self._records)
            proposed_ids: set[str] = set()

            for task in proposed:
                if (
                    task.identifier in existing_ids
                    or task.identifier in proposed_ids
                ):
                    raise OrchestratorPlanError(
                        f"duplicate task ID: {task.identifier}"
                    )

                proposed_ids.add(task.identifier)

            available_ids = existing_ids | proposed_ids

            for task in proposed:
                missing = tuple(
                    dependency
                    for dependency in task.dependencies
                    if dependency not in available_ids
                )

                if missing:
                    raise OrchestratorPlanError(
                        f"task {task.identifier!r} has missing "
                        f"dependencies: {', '.join(missing)}"
                    )

            graph = {
                identifier: record.task.dependencies
                for identifier, record in self._records.items()
            }
            graph.update(
                {
                    task.identifier: task.dependencies
                    for task in proposed
                }
            )
            self._validate_acyclic(graph)

            snapshots: list[OrchestratedTaskSnapshot] = []

            for task in proposed:
                self._sequence += 1
                record = _TaskRecord(
                    task=task,
                    sequence=self._sequence,
                )
                self._records[task.identifier] = record
                snapshots.append(self._snapshot(record))

        for snapshot in snapshots:
            self._publish(
                "ghostfire.orchestrator.task.submitted",
                self._task_event_payload(snapshot),
            )

        return tuple(snapshots)

    def cancel(
        self,
        identifier: str,
        *,
        cascade: bool = False,
    ) -> tuple[OrchestratedTaskSnapshot, ...]:
        """Cancel a pending task and optionally its pending dependents."""

        normalized = _validate_text(
            identifier,
            field_name="identifier",
        )

        with self._lock:
            record = self._require_record_locked(normalized)

            if record.state is not OrchestratedTaskState.PENDING:
                raise OrchestratorStateError(
                    "only pending tasks can be cancelled"
                )

            target_ids = {normalized}

            if cascade:
                changed = True

                while changed:
                    changed = False

                    for candidate in self._records.values():
                        if (
                            candidate.state
                            is OrchestratedTaskState.PENDING
                            and candidate.task.identifier
                            not in target_ids
                            and any(
                                dependency in target_ids
                                for dependency
                                in candidate.task.dependencies
                            )
                        ):
                            target_ids.add(candidate.task.identifier)
                            changed = True

            cancelled: list[OrchestratedTaskSnapshot] = []
            completed_at = datetime.now(timezone.utc)

            for candidate in sorted(
                self._records.values(),
                key=lambda item: item.sequence,
            ):
                if candidate.task.identifier not in target_ids:
                    continue

                candidate.state = OrchestratedTaskState.CANCELLED
                candidate.error = "cancelled"
                candidate.completed_at = completed_at
                cancelled.append(self._snapshot(candidate))

        for snapshot in cancelled:
            self._publish(
                "ghostfire.orchestrator.task.cancelled",
                self._task_event_payload(snapshot),
            )

        return tuple(cancelled)

    def execute_pending(self) -> OrchestrationRun:
        """Execute the pending plan in deterministic dependency order."""

        if not self._execution_lock.acquire(blocking=False):
            raise OrchestratorStateError(
                "another orchestration run is already executing"
            )

        try:
            with self._lock:
                if self._state is not OrchestratorState.RUNNING:
                    raise OrchestratorStateError(
                        "orchestrator must be running"
                    )

                task_ids = tuple(
                    record.task.identifier
                    for record in sorted(
                        self._records.values(),
                        key=lambda item: item.sequence,
                    )
                    if record.state
                    is OrchestratedTaskState.PENDING
                )
                self._state = OrchestratorState.EXECUTING
                self._last_error = None
                run_id = uuid4().hex
                started_at = datetime.now(timezone.utc)

            self._publish(
                "ghostfire.orchestrator.run.started",
                {
                    "run_id": run_id,
                    "task_ids": list(task_ids),
                    "task_count": len(task_ids),
                },
            )

            unresolved = set(task_ids)

            while unresolved:
                progress = False

                for identifier in task_ids:
                    if identifier not in unresolved:
                        continue

                    with self._lock:
                        record = self._records[identifier]

                        if record.state is not OrchestratedTaskState.PENDING:
                            unresolved.discard(identifier)
                            progress = True
                            continue

                        dependency_states = {
                            dependency: self._records[dependency].state
                            for dependency in record.task.dependencies
                        }

                        blocking = tuple(
                            dependency
                            for dependency, state
                            in dependency_states.items()
                            if state in self._BLOCKING_STATES
                        )

                        if blocking:
                            record.state = OrchestratedTaskState.BLOCKED
                            record.error = (
                                "blocked by terminal dependency: "
                                + ", ".join(blocking)
                            )
                            record.completed_at = datetime.now(timezone.utc)
                            snapshot = self._snapshot(record)
                            unresolved.discard(identifier)
                            progress = True
                            action = ("blocked", snapshot)
                        elif all(
                            state is OrchestratedTaskState.COMPLETED
                            for state in dependency_states.values()
                        ):
                            record.state = OrchestratedTaskState.RUNNING
                            record.started_at = datetime.now(timezone.utc)
                            snapshot = self._snapshot(record)
                            action = ("execute", snapshot)
                        else:
                            action = None

                    if action is None:
                        continue

                    if action[0] == "blocked":
                        self._publish(
                            "ghostfire.orchestrator.task.blocked",
                            self._task_event_payload(action[1]),
                        )
                        continue

                    self._publish(
                        "ghostfire.orchestrator.task.started",
                        self._task_event_payload(action[1]),
                    )
                    self._execute_record(identifier)

                    unresolved.discard(identifier)
                    progress = True

                if not progress:
                    with self._lock:
                        for identifier in tuple(unresolved):
                            record = self._records[identifier]
                            record.state = OrchestratedTaskState.BLOCKED
                            record.error = (
                                "dependencies did not resolve"
                            )
                            record.completed_at = datetime.now(timezone.utc)
                            snapshot = self._snapshot(record)
                            unresolved.discard(identifier)

                            self._publish(
                                "ghostfire.orchestrator.task.blocked",
                                self._task_event_payload(snapshot),
                            )

            completed_at = datetime.now(timezone.utc)

            with self._lock:
                counts = self._counts_for_locked(task_ids)
                status = self._run_status(counts)
                run = OrchestrationRun(
                    identifier=run_id,
                    status=status,
                    task_ids=task_ids,
                    counts=MappingProxyType(dict(counts)),
                    started_at=started_at,
                    completed_at=completed_at,
                )
                self._run_history.append(run)
                self._run_count += 1
                self._state = OrchestratorState.RUNNING

                if status != "completed":
                    self._last_error = (
                        "one or more tasks did not complete"
                    )

            self._publish(
                "ghostfire.orchestrator.run.completed",
                run.as_dict(),
            )
            return run
        except Exception as exc:
            with self._lock:
                if self._state is OrchestratorState.EXECUTING:
                    self._state = OrchestratorState.RUNNING

                self._last_error = f"{type(exc).__name__}: {exc}"

            raise
        finally:
            self._execution_lock.release()

    def get_task(
        self,
        identifier: str,
    ) -> OrchestratedTaskSnapshot:
        """Return one immutable task snapshot."""

        normalized = _validate_text(
            identifier,
            field_name="identifier",
        )

        with self._lock:
            return self._snapshot(
                self._require_record_locked(normalized)
            )

    def list_tasks(self) -> tuple[OrchestratedTaskSnapshot, ...]:
        """Return task snapshots in submission order."""

        with self._lock:
            return tuple(
                self._snapshot(record)
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.sequence,
                )
            )

    def run_history(self) -> tuple[OrchestrationRun, ...]:
        """Return retained run summaries."""

        with self._lock:
            return tuple(self._run_history)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe orchestrator snapshot."""

        with self._lock:
            tasks = [
                self._snapshot(record).as_dict()
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.sequence,
                )
            ]
            counts = self._counts_for_locked(
                tuple(self._records)
            )
            history = [
                run.as_dict()
                for run in self._run_history
            ]

            return {
                "state": self._state.value,
                "healthy": self._state in {
                    OrchestratorState.RUNNING,
                    OrchestratorState.EXECUTING,
                },
                "task_count": len(tasks),
                "run_count": self._run_count,
                "counts": counts,
                "last_error": self._last_error,
                "tasks": tasks,
                "run_history": history,
            }

    def _execute_record(self, identifier: str) -> None:
        last_error: str | None = None

        while True:
            with self._lock:
                record = self._records[identifier]
                record.attempts += 1
                attempt = record.attempts
                task = record.task

            try:
                result = self._registry.dispatch(
                    task.capability,
                    task.payload,
                    preferred_agent=task.preferred_agent,
                    metadata={
                        **dict(task.metadata),
                        "orchestration_task_id": task.identifier,
                        "orchestration_attempt": attempt,
                    },
                )
            except AgentFrameworkError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

                if attempt < task.max_attempts:
                    self._publish(
                        "ghostfire.orchestrator.task.retrying",
                        {
                            "task_id": task.identifier,
                            "capability": task.capability,
                            "attempt": attempt,
                            "max_attempts": task.max_attempts,
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue

                with self._lock:
                    record = self._records[identifier]
                    record.state = OrchestratedTaskState.FAILED
                    record.error = last_error
                    record.completed_at = datetime.now(timezone.utc)
                    snapshot = self._snapshot(record)

                self._publish(
                    "ghostfire.orchestrator.task.failed",
                    self._task_event_payload(snapshot),
                )
                return

            with self._lock:
                record = self._records[identifier]
                record.state = OrchestratedTaskState.COMPLETED
                record.agent_name = result.agent_name
                record.result = result
                record.error = None
                record.completed_at = datetime.now(timezone.utc)
                snapshot = self._snapshot(record)

            self._publish(
                "ghostfire.orchestrator.task.completed",
                self._task_event_payload(snapshot),
            )
            return

    def _counts_for_locked(
        self,
        task_ids: tuple[str, ...],
    ) -> dict[str, int]:
        counts = {
            state.value: 0
            for state in OrchestratedTaskState
        }

        for identifier in task_ids:
            counts[self._records[identifier].state.value] += 1

        return counts

    @staticmethod
    def _run_status(counts: Mapping[str, int]) -> str:
        total = sum(counts.values())
        completed = counts[OrchestratedTaskState.COMPLETED.value]

        if total == completed:
            return "completed"

        if completed:
            return "partial"

        return "failed"

    def _require_record_locked(self, identifier: str) -> _TaskRecord:
        try:
            return self._records[identifier]
        except KeyError as exc:
            raise OrchestratorPlanError(
                f"task is not registered: {identifier}"
            ) from exc

    @staticmethod
    def _validate_acyclic(
        graph: Mapping[str, tuple[str, ...]],
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visited:
                return

            if identifier in visiting:
                raise OrchestratorPlanError(
                    f"dependency cycle detected at {identifier}"
                )

            visiting.add(identifier)

            for dependency in graph.get(identifier, ()):
                visit(dependency)

            visiting.remove(identifier)
            visited.add(identifier)

        for identifier in graph:
            visit(identifier)

    @staticmethod
    def _snapshot(
        record: _TaskRecord,
    ) -> OrchestratedTaskSnapshot:
        return OrchestratedTaskSnapshot(
            task=record.task,
            state=record.state,
            sequence=record.sequence,
            attempts=record.attempts,
            agent_name=record.agent_name,
            result=record.result,
            error=record.error,
            started_at=record.started_at,
            completed_at=record.completed_at,
        )

    @staticmethod
    def _task_event_payload(
        snapshot: OrchestratedTaskSnapshot,
    ) -> dict[str, Any]:
        return {
            "task_id": snapshot.task.identifier,
            "capability": snapshot.task.capability,
            "state": snapshot.state.value,
            "attempts": snapshot.attempts,
            "agent_name": snapshot.agent_name,
            "dependencies": list(snapshot.task.dependencies),
            "error": snapshot.error,
        }

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "task_count": len(self._records),
            "run_count": self._run_count,
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


def _validate_positive_int(value: Any, *, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(deepcopy(dict(value)))
