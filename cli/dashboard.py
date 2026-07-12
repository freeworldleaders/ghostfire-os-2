"""Terminal-native operational dashboard for GhostFire OS."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, TextIO

from core.eventbus import EventBus
from core.scheduler import Scheduler
from core.service_manager import (
    ServiceManager,
    ServiceState,
)


Clock = Callable[[], datetime]


class DashboardState(str, Enum):
    """Aggregated runtime states shown by the terminal dashboard."""

    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class DashboardService:
    """Immutable service row rendered by the dashboard."""

    name: str
    state: str
    healthy: bool | None
    dependencies: tuple[str, ...]
    start_count: int
    stop_count: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Immutable point-in-time dashboard model."""

    app_name: str
    version: str
    overall_state: DashboardState
    configuration_revision: int
    configuration_sources: tuple[str, ...]
    scheduler_running: bool
    scheduled_task_count: int
    log_path: str | None
    services: tuple[DashboardService, ...]
    captured_at: datetime


class TerminalDashboard:
    """
    Render GhostFire runtime state without third-party dependencies.

    The dashboard is snapshot-based rather than an infinite interactive loop,
    which keeps it safe for startup scripts, redirected output, tests, and
    remote operator terminals.
    """

    MINIMUM_WIDTH = 72

    def __init__(
        self,
        *,
        app_name: str,
        version: str,
        configuration_revision: int,
        configuration_sources: tuple[str, ...] | list[str],
        service_manager: ServiceManager,
        scheduler: Scheduler,
        log_path: str | Path | None = None,
        event_bus: EventBus | None = None,
        stream: TextIO | None = None,
        width: int = 88,
        color: bool | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._app_name = self._validate_text(
            app_name,
            field_name="app_name",
        )
        self._version = self._validate_text(
            version,
            field_name="version",
        )

        if (
            isinstance(configuration_revision, bool)
            or not isinstance(configuration_revision, int)
        ):
            raise TypeError(
                "configuration_revision must be an integer"
            )

        if configuration_revision < 1:
            raise ValueError(
                "configuration_revision must be positive"
            )

        if not isinstance(service_manager, ServiceManager):
            raise TypeError(
                "service_manager must be a ServiceManager"
            )

        if not isinstance(scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError("width must be an integer")

        if width < self.MINIMUM_WIDTH:
            raise ValueError(
                f"width must be at least {self.MINIMUM_WIDTH}"
            )

        if color is not None and not isinstance(color, bool):
            raise TypeError("color must be True, False, or None")

        normalized_sources = tuple(
            self._validate_text(
                source,
                field_name="configuration source",
            )
            for source in configuration_sources
        )

        self._configuration_revision = configuration_revision
        self._configuration_sources = normalized_sources
        self._service_manager = service_manager
        self._scheduler = scheduler
        self._log_path = (
            str(Path(log_path).expanduser())
            if log_path is not None
            else None
        )
        self._event_bus = event_bus
        self._stream = stream if stream is not None else sys.stdout
        self._width = width
        self._color = color
        self._clock = clock or (
            lambda: datetime.now(timezone.utc)
        )
        self._lock = RLock()
        self._last_snapshot: DashboardSnapshot | None = None

    @property
    def last_snapshot(self) -> DashboardSnapshot | None:
        """Return the most recently captured snapshot."""

        with self._lock:
            return self._last_snapshot

    def capture(
        self,
        *,
        check_health: bool = True,
    ) -> DashboardSnapshot:
        """Capture current service, scheduler, and configuration state."""

        if not isinstance(check_health, bool):
            raise TypeError("check_health must be a boolean")

        service_rows: list[DashboardService] = []

        for status in self._service_manager.list_statuses():
            healthy: bool | None

            if check_health:
                healthy = self._service_manager.check_health(
                    status.name
                )
                status = self._service_manager.status(status.name)
            else:
                healthy = None

            service_rows.append(
                DashboardService(
                    name=status.name,
                    state=status.state.value,
                    healthy=healthy,
                    dependencies=status.dependencies,
                    start_count=status.start_count,
                    stop_count=status.stop_count,
                    last_error=status.last_error,
                )
            )

        services = tuple(service_rows)
        overall_state = self._calculate_overall_state(
            services,
            check_health=check_health,
        )

        captured_at = self._clock()

        if not isinstance(captured_at, datetime):
            raise TypeError("clock must return datetime")

        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)

        snapshot = DashboardSnapshot(
            app_name=self._app_name,
            version=self._version,
            overall_state=overall_state,
            configuration_revision=self._configuration_revision,
            configuration_sources=self._configuration_sources,
            scheduler_running=self._scheduler.is_running,
            scheduled_task_count=self._scheduler.task_count(),
            log_path=self._log_path,
            services=services,
            captured_at=captured_at,
        )

        with self._lock:
            self._last_snapshot = snapshot

        self._publish(
            "ghostfire.dashboard.captured",
            {
                "overall_state": snapshot.overall_state.value,
                "service_count": len(snapshot.services),
                "scheduler_running": snapshot.scheduler_running,
                "scheduled_task_count": (
                    snapshot.scheduled_task_count
                ),
            },
        )

        return snapshot

    def render(
        self,
        snapshot: DashboardSnapshot | None = None,
        *,
        color: bool | None = None,
    ) -> str:
        """Render a snapshot as a fixed-width terminal dashboard."""

        selected_snapshot = snapshot or self.capture()
        use_color = self._should_use_color(color)

        border = "+" + "-" * (self._width - 2) + "+"
        lines = [border]

        title = (
            f"{selected_snapshot.app_name} "
            f"{selected_snapshot.version}"
        )
        state = selected_snapshot.overall_state.value.upper()

        lines.append(
            self._row(f"{title} | STATE: {state}")
        )
        lines.append(border)

        captured = selected_snapshot.captured_at.astimezone(
            timezone.utc
        ).isoformat()

        lines.append(
            self._row(
                "Captured: "
                f"{captured} | "
                "Config revision: "
                f"{selected_snapshot.configuration_revision}"
            )
        )

        sources = ", ".join(
            selected_snapshot.configuration_sources
        ) or "none"

        lines.append(
            self._row(f"Config sources: {sources}")
        )

        scheduler_state = (
            "RUNNING"
            if selected_snapshot.scheduler_running
            else "STOPPED"
        )

        lines.append(
            self._row(
                "Scheduler: "
                f"{scheduler_state} | "
                "Queued tasks: "
                f"{selected_snapshot.scheduled_task_count}"
            )
        )

        lines.append(
            self._row(
                "Log: "
                f"{selected_snapshot.log_path or 'disabled'}"
            )
        )
        lines.append(border)
        lines.append(self._row("SERVICES"))
        lines.append(border)

        header = self._service_columns(
            "NAME",
            "STATE",
            "HEALTH",
            "DEPENDENCIES",
        )
        lines.append(self._row(header))

        for service in selected_snapshot.services:
            if service.healthy is True:
                health = "HEALTHY"
            elif service.healthy is False:
                health = "UNHEALTHY"
            else:
                health = "NOT CHECKED"

            dependencies = (
                ",".join(service.dependencies)
                if service.dependencies
                else "-"
            )

            line = self._service_columns(
                service.name,
                service.state.upper(),
                health,
                dependencies,
            )

            lines.append(self._row(line))

            if service.last_error:
                lines.append(
                    self._row(
                        f"  ERROR {service.name}: "
                        f"{service.last_error}"
                    )
                )

        if not selected_snapshot.services:
            lines.append(
                self._row("No services registered")
            )

        lines.append(border)

        rendered = "\n".join(lines)

        if use_color:
            rendered = self._apply_color(
                rendered,
                selected_snapshot.overall_state,
            )

        return rendered

    def display(
        self,
        *,
        check_health: bool = True,
        color: bool | None = None,
    ) -> DashboardSnapshot:
        """Capture, render, and write one dashboard frame."""

        snapshot = self.capture(check_health=check_health)
        rendered = self.render(snapshot, color=color)

        with self._lock:
            self._stream.write(rendered)
            self._stream.write("\n")

            flush = getattr(self._stream, "flush", None)

            if callable(flush):
                flush()

        self._publish(
            "ghostfire.dashboard.displayed",
            {
                "overall_state": snapshot.overall_state.value,
                "service_count": len(snapshot.services),
                "width": self._width,
            },
        )

        return snapshot

    def as_dict(
        self,
        snapshot: DashboardSnapshot | None = None,
    ) -> dict[str, Any]:
        """Convert a snapshot into a JSON-safe dictionary."""

        selected_snapshot = snapshot or self.capture()
        values = asdict(selected_snapshot)
        values["overall_state"] = (
            selected_snapshot.overall_state.value
        )
        values["captured_at"] = (
            selected_snapshot.captured_at.isoformat()
        )

        return values

    def to_json(
        self,
        snapshot: DashboardSnapshot | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize a snapshot for proof capture or transport."""

        if indent is not None and (
            isinstance(indent, bool)
            or not isinstance(indent, int)
        ):
            raise TypeError("indent must be an integer or None")

        return json.dumps(
            self.as_dict(snapshot),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            default=str,
        )

    def _row(self, content: str) -> str:
        available = self._width - 4
        fitted = self._fit(content, available)
        return f"| {fitted.ljust(available)} |"

    def _service_columns(
        self,
        name: str,
        state: str,
        health: str,
        dependencies: str,
    ) -> str:
        available = self._width - 4
        name_width = 18
        state_width = 10
        health_width = 12
        dependency_width = (
            available
            - name_width
            - state_width
            - health_width
            - 9
        )

        return (
            f"{self._fit(name, name_width).ljust(name_width)}"
            " | "
            f"{self._fit(state, state_width).ljust(state_width)}"
            " | "
            f"{self._fit(health, health_width).ljust(health_width)}"
            " | "
            f"{self._fit(dependencies, dependency_width)}"
        )

    def _should_use_color(
        self,
        override: bool | None,
    ) -> bool:
        if override is not None:
            return override

        if self._color is not None:
            return self._color

        if "NO_COLOR" in os.environ:
            return False

        isatty = getattr(self._stream, "isatty", None)

        return bool(callable(isatty) and isatty())

    @staticmethod
    def _apply_color(
        rendered: str,
        state: DashboardState,
    ) -> str:
        if state is DashboardState.ONLINE:
            code = "32"
        elif state is DashboardState.DEGRADED:
            code = "33"
        else:
            code = "31"

        token = f"STATE: {state.value.upper()}"

        return rendered.replace(
            token,
            f"\033[{code}m{token}\033[0m",
            1,
        )

    @staticmethod
    def _calculate_overall_state(
        services: tuple[DashboardService, ...],
        *,
        check_health: bool,
    ) -> DashboardState:
        if not services:
            return DashboardState.OFFLINE

        all_running = all(
            service.state == ServiceState.RUNNING.value
            for service in services
        )

        health_good = (
            all(service.healthy is True for service in services)
            if check_health
            else True
        )

        if all_running and health_good:
            return DashboardState.ONLINE

        if any(
            service.state == ServiceState.RUNNING.value
            for service in services
        ):
            return DashboardState.DEGRADED

        return DashboardState.OFFLINE

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
    def _fit(value: str, width: int) -> str:
        sanitized = " ".join(str(value).split())

        if len(sanitized) <= width:
            return sanitized

        if width <= 3:
            return sanitized[:width]

        return sanitized[: width - 3] + "..."

    @staticmethod
    def _validate_text(
        value: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty")

        return normalized
