"""Dependency-aware service lifecycle management for GhostFire OS."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any

from core.eventbus import EventBus


ServiceCallback = Callable[[], Any]
HealthCallback = Callable[[], bool]


class ServiceState(str, Enum):
    """Lifecycle states exposed by the Service Manager."""

    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Immutable external snapshot of a managed service."""

    name: str
    state: ServiceState
    dependencies: tuple[str, ...]
    sequence: int
    start_count: int
    stop_count: int
    last_error: str | None


@dataclass(slots=True)
class _ServiceRecord:
    name: str
    start_callback: ServiceCallback
    stop_callback: ServiceCallback | None
    dependencies: tuple[str, ...]
    health_callback: HealthCallback | None
    sequence: int
    state: ServiceState = ServiceState.REGISTERED
    start_count: int = 0
    stop_count: int = 0
    last_error: str | None = None


class ServiceManagerError(RuntimeError):
    """Base class for Service Manager failures."""


class ServiceRegistrationError(ServiceManagerError):
    """Raised when service registration is invalid."""


class ServiceDependencyError(ServiceManagerError):
    """Raised when a dependency is missing or cyclic."""


class ServiceStartError(ServiceManagerError):
    """Raised when startup fails after rollback is attempted."""

    def __init__(
        self,
        service_name: str,
        cause: Exception,
        rollback_failures: tuple[tuple[str, Exception], ...],
    ) -> None:
        self.service_name = service_name
        self.cause = cause
        self.rollback_failures = rollback_failures

        message = f"service {service_name!r} failed to start: {cause}"

        if rollback_failures:
            message += (
                f"; {len(rollback_failures)} rollback stop(s) failed"
            )

        super().__init__(message)


class ServiceStopError(ServiceManagerError):
    """Raised after all requested stop operations are attempted."""

    def __init__(
        self,
        failures: tuple[tuple[str, Exception], ...],
    ) -> None:
        self.failures = failures
        super().__init__(
            f"{len(failures)} service stop operation(s) failed"
        )


class ServiceManager:
    """
    Thread-safe, dependency-aware lifecycle coordinator.

    Services start in topological dependency order and stop in reverse order.
    A startup failure rolls back only services started by that operation.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._event_bus = event_bus
        self._lock = RLock()
        self._services: dict[str, _ServiceRecord] = {}
        self._sequence = 0

    def register(
        self,
        name: str,
        start: ServiceCallback,
        *,
        stop: ServiceCallback | None = None,
        dependencies: Iterable[str] = (),
        health: HealthCallback | None = None,
    ) -> ServiceStatus:
        """Register a service and return its initial status."""

        normalized_name = self._validate_name(name)

        if not callable(start):
            raise TypeError("start must be callable")

        if stop is not None and not callable(stop):
            raise TypeError("stop must be callable or None")

        if health is not None and not callable(health):
            raise TypeError("health must be callable or None")

        normalized_dependencies = self._normalize_dependencies(
            dependencies
        )

        if normalized_name in normalized_dependencies:
            raise ServiceRegistrationError(
                "a service cannot depend on itself"
            )

        with self._lock:
            if normalized_name in self._services:
                raise ServiceRegistrationError(
                    f"service already registered: {normalized_name}"
                )

            self._sequence += 1

            record = _ServiceRecord(
                name=normalized_name,
                start_callback=start,
                stop_callback=stop,
                dependencies=normalized_dependencies,
                health_callback=health,
                sequence=self._sequence,
            )

            self._services[normalized_name] = record
            status = self._snapshot(record)

            self._publish(
                "ghostfire.service.registered",
                self._status_payload(status),
            )

            return status

    def unregister(self, name: str) -> ServiceStatus:
        """Remove a stopped service with no registered dependents."""

        normalized_name = self._validate_name(name)

        with self._lock:
            record = self._require_service_locked(normalized_name)

            if record.state in {
                ServiceState.STARTING,
                ServiceState.RUNNING,
                ServiceState.STOPPING,
            }:
                raise ServiceRegistrationError(
                    "running services cannot be unregistered"
                )

            dependents = self._direct_dependents_locked(normalized_name)

            if dependents:
                raise ServiceRegistrationError(
                    "service has registered dependents: "
                    + ", ".join(dependents)
                )

            status = self._snapshot(record)
            self._services.pop(normalized_name)

            self._publish(
                "ghostfire.service.unregistered",
                self._status_payload(status),
            )

            return status

    def start(self, name: str) -> ServiceStatus:
        """Start one service and its dependency chain."""

        normalized_name = self._validate_name(name)

        with self._lock:
            order = self._resolve_order_locked((normalized_name,))
            self._start_order_locked(order)
            return self._snapshot(
                self._services[normalized_name]
            )

    def start_all(self) -> tuple[ServiceStatus, ...]:
        """Start every registered service in dependency order."""

        with self._lock:
            names = tuple(self._services)
            order = self._resolve_order_locked(names)
            self._start_order_locked(order)

            return tuple(
                self._snapshot(self._services[name])
                for name in order
            )

    def stop(
        self,
        name: str,
        *,
        cascade: bool = True,
    ) -> tuple[ServiceStatus, ...]:
        """Stop a service, optionally stopping running dependents first."""

        normalized_name = self._validate_name(name)

        with self._lock:
            self._require_service_locked(normalized_name)

            dependent_names = self._dependent_closure_locked(
                normalized_name
            )

            running_dependents = tuple(
                dependent_name
                for dependent_name in dependent_names
                if self._services[dependent_name].state
                is ServiceState.RUNNING
            )

            if running_dependents and not cascade:
                raise ServiceDependencyError(
                    "running dependents block stop: "
                    + ", ".join(running_dependents)
                )

            impacted = set(dependent_names)
            impacted.add(normalized_name)

            order = self._resolve_order_locked(
                tuple(self._services)
            )

            stop_order = tuple(
                service_name
                for service_name in reversed(order)
                if service_name in impacted
            )

            self._stop_order_locked(stop_order)

            return tuple(
                self._snapshot(self._services[service_name])
                for service_name in stop_order
            )

    def stop_all(self) -> tuple[ServiceStatus, ...]:
        """Stop all running services in reverse dependency order."""

        with self._lock:
            order = self._resolve_order_locked(
                tuple(self._services)
            )
            stop_order = tuple(reversed(order))
            self._stop_order_locked(stop_order)

            return tuple(
                self._snapshot(self._services[name])
                for name in stop_order
            )

    def restart(self, name: str) -> ServiceStatus:
        """
        Restart a service and restore dependents that were running.

        Non-running dependents remain stopped.
        """

        normalized_name = self._validate_name(name)

        with self._lock:
            self._require_service_locked(normalized_name)

            impacted = set(
                self._dependent_closure_locked(normalized_name)
            )
            impacted.add(normalized_name)

            restore_names = {
                service_name
                for service_name in impacted
                if self._services[service_name].state
                is ServiceState.RUNNING
            }
            restore_names.add(normalized_name)

            self.stop(normalized_name, cascade=True)

            full_order = self._resolve_order_locked(
                tuple(self._services)
            )

            restart_order = tuple(
                service_name
                for service_name in full_order
                if service_name in restore_names
            )

            self._start_order_locked(restart_order)

            return self._snapshot(
                self._services[normalized_name]
            )

    def status(self, name: str) -> ServiceStatus:
        """Return a snapshot without running a health callback."""

        normalized_name = self._validate_name(name)

        with self._lock:
            return self._snapshot(
                self._require_service_locked(normalized_name)
            )

    def list_statuses(self) -> tuple[ServiceStatus, ...]:
        """Return service snapshots in registration order."""

        with self._lock:
            records = sorted(
                self._services.values(),
                key=lambda record: record.sequence,
            )

            return tuple(
                self._snapshot(record)
                for record in records
            )

    def check_health(self, name: str) -> bool:
        """Evaluate one service's health and publish the result."""

        normalized_name = self._validate_name(name)

        with self._lock:
            record = self._require_service_locked(normalized_name)

            if record.state is not ServiceState.RUNNING:
                healthy = False
            elif record.health_callback is None:
                healthy = True
            else:
                try:
                    healthy = bool(record.health_callback())
                except Exception as exc:
                    healthy = False
                    record.last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            payload = self._status_payload(
                self._snapshot(record)
            )
            payload["healthy"] = healthy

            self._publish(
                "ghostfire.service.health_checked",
                payload,
            )

            return healthy

    def is_running(self, name: str) -> bool:
        """Return whether a service is in the running state."""

        return self.status(name).state is ServiceState.RUNNING

    def _start_order_locked(
        self,
        order: tuple[str, ...],
    ) -> None:
        started_now: list[str] = []

        for service_name in order:
            record = self._services[service_name]

            if record.state is ServiceState.RUNNING:
                continue

            if record.state in {
                ServiceState.STARTING,
                ServiceState.STOPPING,
            }:
                raise ServiceManagerError(
                    f"service is busy: {service_name}"
                )

            record.state = ServiceState.STARTING
            record.last_error = None

            self._publish(
                "ghostfire.service.starting",
                self._status_payload(self._snapshot(record)),
            )

            try:
                record.start_callback()
            except Exception as exc:
                record.state = ServiceState.FAILED
                record.last_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                self._publish(
                    "ghostfire.service.start_failed",
                    self._status_payload(
                        self._snapshot(record)
                    ),
                )

                rollback_failures: list[
                    tuple[str, Exception]
                ] = []

                for started_name in reversed(started_now):
                    started_record = self._services[started_name]

                    try:
                        self._stop_record_locked(
                            started_record,
                            rollback=True,
                        )
                    except Exception as rollback_exc:
                        rollback_failures.append(
                            (started_name, rollback_exc)
                        )

                raise ServiceStartError(
                    service_name,
                    exc,
                    tuple(rollback_failures),
                ) from exc

            record.state = ServiceState.RUNNING
            record.start_count += 1
            record.last_error = None
            started_now.append(service_name)

            self._publish(
                "ghostfire.service.started",
                self._status_payload(self._snapshot(record)),
            )

    def _stop_order_locked(
        self,
        order: tuple[str, ...],
    ) -> None:
        failures: list[tuple[str, Exception]] = []

        for service_name in order:
            record = self._services[service_name]

            if record.state is not ServiceState.RUNNING:
                continue

            try:
                self._stop_record_locked(record)
            except Exception as exc:
                failures.append((service_name, exc))

        if failures:
            raise ServiceStopError(tuple(failures))

    def _stop_record_locked(
        self,
        record: _ServiceRecord,
        *,
        rollback: bool = False,
    ) -> bool:
        if record.state is not ServiceState.RUNNING:
            return False

        record.state = ServiceState.STOPPING

        payload = self._status_payload(self._snapshot(record))
        payload["rollback"] = rollback

        self._publish(
            "ghostfire.service.stopping",
            payload,
        )

        try:
            if record.stop_callback is not None:
                record.stop_callback()
        except Exception as exc:
            record.state = ServiceState.FAILED
            record.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            payload = self._status_payload(
                self._snapshot(record)
            )
            payload["rollback"] = rollback

            self._publish(
                "ghostfire.service.stop_failed",
                payload,
            )

            raise

        record.state = ServiceState.STOPPED
        record.stop_count += 1
        record.last_error = None

        payload = self._status_payload(self._snapshot(record))
        payload["rollback"] = rollback

        self._publish(
            "ghostfire.service.stopped",
            payload,
        )

        return True

    def _resolve_order_locked(
        self,
        names: Iterable[str],
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        visited: set[str] = set()
        active: list[str] = []

        def visit(service_name: str) -> None:
            if service_name in visited:
                return

            if service_name in active:
                cycle_start = active.index(service_name)
                cycle = active[cycle_start:] + [service_name]

                raise ServiceDependencyError(
                    "dependency cycle detected: "
                    + " -> ".join(cycle)
                )

            record = self._services.get(service_name)

            if record is None:
                raise ServiceDependencyError(
                    f"service dependency is not registered: {service_name}"
                )

            active.append(service_name)

            for dependency in record.dependencies:
                visit(dependency)

            active.pop()
            visited.add(service_name)
            ordered.append(service_name)

        for name in names:
            visit(name)

        return tuple(ordered)

    def _dependent_closure_locked(
        self,
        service_name: str,
    ) -> tuple[str, ...]:
        closure: set[str] = set()
        changed = True

        while changed:
            changed = False

            for candidate in self._services.values():
                if candidate.name in closure:
                    continue

                if (
                    service_name in candidate.dependencies
                    or any(
                        dependency in closure
                        for dependency in candidate.dependencies
                    )
                ):
                    closure.add(candidate.name)
                    changed = True

        order = self._resolve_order_locked(
            tuple(self._services)
        )

        return tuple(
            name
            for name in order
            if name in closure
        )

    def _direct_dependents_locked(
        self,
        service_name: str,
    ) -> tuple[str, ...]:
        return tuple(
            record.name
            for record in sorted(
                self._services.values(),
                key=lambda item: item.sequence,
            )
            if service_name in record.dependencies
        )

    def _require_service_locked(
        self,
        service_name: str,
    ) -> _ServiceRecord:
        record = self._services.get(service_name)

        if record is None:
            raise KeyError(f"service is not registered: {service_name}")

        return record

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
    def _snapshot(record: _ServiceRecord) -> ServiceStatus:
        return ServiceStatus(
            name=record.name,
            state=record.state,
            dependencies=record.dependencies,
            sequence=record.sequence,
            start_count=record.start_count,
            stop_count=record.stop_count,
            last_error=record.last_error,
        )

    @staticmethod
    def _status_payload(
        status: ServiceStatus,
    ) -> dict[str, Any]:
        return {
            "service_name": status.name,
            "state": status.state.value,
            "dependencies": list(status.dependencies),
            "sequence": status.sequence,
            "start_count": status.start_count,
            "stop_count": status.stop_count,
            "last_error": status.last_error,
        }

    @classmethod
    def _normalize_dependencies(
        cls,
        dependencies: Iterable[str],
    ) -> tuple[str, ...]:
        if isinstance(dependencies, str):
            raise TypeError(
                "dependencies must be an iterable of service names"
            )

        normalized: list[str] = []

        for dependency in dependencies:
            name = cls._validate_name(dependency)

            if name in normalized:
                raise ServiceRegistrationError(
                    f"duplicate dependency: {name}"
                )

            normalized.append(name)

        return tuple(normalized)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("name cannot be empty")

        return normalized_name
