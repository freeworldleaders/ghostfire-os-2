"""Thread-safe task scheduling for GhostFire OS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event as ThreadEvent
from threading import RLock, Thread, current_thread
from time import monotonic
from typing import Any
from uuid import uuid4

from core.eventbus import EventBus


TaskCallback = Callable[[], Any]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """Opaque identifier returned when a task is scheduled."""

    identifier: str


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """Immutable external view of a scheduled task."""

    handle: TaskHandle
    name: str
    due_at: float
    interval_seconds: float | None
    sequence: int


@dataclass(slots=True)
class _TaskRecord:
    handle: TaskHandle
    name: str
    callback: TaskCallback
    due_at: float
    interval_seconds: float | None
    sequence: int


class SchedulerExecutionError(RuntimeError):
    """Raised after all due tasks run when one or more tasks failed."""

    def __init__(
        self,
        failures: tuple[tuple[ScheduledTask, Exception], ...],
    ) -> None:
        self.failures = failures
        super().__init__(f"{len(failures)} scheduled task(s) failed")


class Scheduler:
    """
    Thread-safe scheduler for one-time and recurring tasks.

    ``run_pending`` supports deterministic execution. ``start`` runs the same
    scheduler loop on a background worker.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        clock: Clock = monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._event_bus = event_bus
        self._clock = clock
        self._lock = RLock()
        self._tasks: dict[str, _TaskRecord] = {}
        self._sequence = 0

        self._stop_event = ThreadEvent()
        self._wake_event = ThreadEvent()
        self._thread: Thread | None = None
        self._poll_interval = 0.1

    def schedule_once(
        self,
        name: str,
        delay_seconds: float,
        callback: TaskCallback,
    ) -> TaskHandle:
        """Schedule a callback to execute once after a delay."""

        delay = self._validate_seconds(
            delay_seconds,
            field_name="delay_seconds",
            allow_zero=True,
        )

        return self._schedule(
            name=name,
            callback=callback,
            due_at=self._clock() + delay,
            interval_seconds=None,
        )

    def schedule_every(
        self,
        name: str,
        interval_seconds: float,
        callback: TaskCallback,
        *,
        start_immediately: bool = False,
    ) -> TaskHandle:
        """Schedule a callback using fixed-delay recurrence."""

        interval = self._validate_seconds(
            interval_seconds,
            field_name="interval_seconds",
            allow_zero=False,
        )

        initial_delay = 0.0 if start_immediately else interval

        return self._schedule(
            name=name,
            callback=callback,
            due_at=self._clock() + initial_delay,
            interval_seconds=interval,
        )

    def cancel(self, handle: TaskHandle) -> bool:
        """Cancel a task. Returns True when the task existed."""

        if not isinstance(handle, TaskHandle):
            raise TypeError("handle must be a TaskHandle")

        with self._lock:
            record = self._tasks.pop(handle.identifier, None)

        if record is None:
            return False

        self._wake_event.set()

        self._publish(
            "ghostfire.scheduler.task_cancelled",
            self._task_payload(self._snapshot(record)),
        )

        return True

    def run_pending(
        self,
        *,
        raise_exceptions: bool = True,
    ) -> list[Any]:
        """
        Execute all due tasks in deterministic order.

        A failed task cannot block later due tasks.
        """

        now = self._clock()

        with self._lock:
            due_records = sorted(
                (
                    record
                    for record in self._tasks.values()
                    if record.due_at <= now
                ),
                key=lambda record: (
                    record.due_at,
                    record.sequence,
                ),
            )

            executions: list[
                tuple[TaskCallback, ScheduledTask]
            ] = []

            for record in due_records:
                task = self._snapshot(record)
                executions.append((record.callback, task))

                if record.interval_seconds is None:
                    self._tasks.pop(record.handle.identifier, None)
                else:
                    record.due_at = now + record.interval_seconds

        results: list[Any] = []
        failures: list[tuple[ScheduledTask, Exception]] = []

        for callback, task in executions:
            self._publish(
                "ghostfire.scheduler.task_started",
                self._task_payload(task),
            )

            try:
                result = callback()
                results.append(result)

                payload = self._task_payload(task)
                payload["result_type"] = type(result).__name__

                self._publish(
                    "ghostfire.scheduler.task_completed",
                    payload,
                )
            except Exception as exc:
                failures.append((task, exc))

                payload = self._task_payload(task)
                payload["error_type"] = type(exc).__name__
                payload["error"] = str(exc)

                self._publish(
                    "ghostfire.scheduler.task_failed",
                    payload,
                )

        if failures and raise_exceptions:
            raise SchedulerExecutionError(tuple(failures))

        return results

    def start(
        self,
        *,
        poll_interval: float = 0.1,
    ) -> bool:
        """Start the background scheduler worker."""

        interval = self._validate_seconds(
            poll_interval,
            field_name="poll_interval",
            allow_zero=False,
        )

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._poll_interval = interval
            self._stop_event.clear()
            self._wake_event.clear()

            self._thread = Thread(
                target=self._worker_loop,
                name="GhostFireScheduler",
                daemon=True,
            )

            self._thread.start()

        self._publish(
            "ghostfire.scheduler.started",
            {"poll_interval": interval},
        )

        return True

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Stop the background scheduler worker."""

        wait_timeout = self._validate_seconds(
            timeout,
            field_name="timeout",
            allow_zero=False,
        )

        with self._lock:
            thread = self._thread

        if thread is None or not thread.is_alive():
            return False

        if thread is current_thread():
            raise RuntimeError(
                "scheduler worker cannot stop itself synchronously"
            )

        self._stop_event.set()
        self._wake_event.set()
        thread.join(wait_timeout)

        if thread.is_alive():
            raise RuntimeError(
                "scheduler worker did not stop before timeout"
            )

        with self._lock:
            self._thread = None

        self._publish(
            "ghostfire.scheduler.stopped",
            {},
        )

        return True

    def clear(self) -> int:
        """Remove all scheduled tasks and return the number removed."""

        with self._lock:
            removed_count = len(self._tasks)
            self._tasks.clear()

        self._wake_event.set()

        self._publish(
            "ghostfire.scheduler.cleared",
            {"removed_count": removed_count},
        )

        return removed_count

    def task_count(self) -> int:
        """Return the number of active scheduled tasks."""

        with self._lock:
            return len(self._tasks)

    def list_tasks(self) -> tuple[ScheduledTask, ...]:
        """Return immutable task snapshots in execution order."""

        with self._lock:
            records = sorted(
                self._tasks.values(),
                key=lambda record: (
                    record.due_at,
                    record.sequence,
                ),
            )

            return tuple(
                self._snapshot(record)
                for record in records
            )

    def seconds_until_next(self) -> float | None:
        """Return seconds until the next task, or None when empty."""

        with self._lock:
            if not self._tasks:
                return None

            due_at = min(
                record.due_at
                for record in self._tasks.values()
            )

        return max(0.0, due_at - self._clock())

    @property
    def is_running(self) -> bool:
        """Return whether the background worker is alive."""

        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _schedule(
        self,
        *,
        name: str,
        callback: TaskCallback,
        due_at: float,
        interval_seconds: float | None,
    ) -> TaskHandle:
        normalized_name = self._validate_name(name)

        if not callable(callback):
            raise TypeError("callback must be callable")

        handle = TaskHandle(uuid4().hex)

        with self._lock:
            self._sequence += 1

            record = _TaskRecord(
                handle=handle,
                name=normalized_name,
                callback=callback,
                due_at=due_at,
                interval_seconds=interval_seconds,
                sequence=self._sequence,
            )

            self._tasks[handle.identifier] = record

        self._wake_event.set()

        self._publish(
            "ghostfire.scheduler.task_scheduled",
            self._task_payload(self._snapshot(record)),
        )

        return handle

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_pending(raise_exceptions=False)

            wait_seconds = self.seconds_until_next()

            if wait_seconds is None:
                timeout = self._poll_interval
            else:
                timeout = min(
                    self._poll_interval,
                    max(0.001, wait_seconds),
                )

            self._wake_event.wait(timeout)
            self._wake_event.clear()

    def _publish(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        if self._event_bus is None:
            return

        self._event_bus.emit(
            event_name,
            payload,
            raise_exceptions=False,
        )

    @staticmethod
    def _snapshot(record: _TaskRecord) -> ScheduledTask:
        return ScheduledTask(
            handle=record.handle,
            name=record.name,
            due_at=record.due_at,
            interval_seconds=record.interval_seconds,
            sequence=record.sequence,
        )

    @staticmethod
    def _task_payload(task: ScheduledTask) -> dict[str, Any]:
        return {
            "task_id": task.handle.identifier,
            "task_name": task.name,
            "due_at": task.due_at,
            "interval_seconds": task.interval_seconds,
            "sequence": task.sequence,
        }

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("name cannot be empty")

        return normalized_name

    @staticmethod
    def _validate_seconds(
        value: float,
        *,
        field_name: str,
        allow_zero: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a number")

        normalized_value = float(value)

        if allow_zero:
            valid = normalized_value >= 0.0
        else:
            valid = normalized_value > 0.0

        if not valid:
            requirement = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"{field_name} must be {requirement}"
            )

        return normalized_value
