import io
import json
import os
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from cli.dashboard import (
    DashboardState,
    TerminalDashboard,
)
from core.eventbus import EventBus
from core.scheduler import Scheduler
from core.service_manager import ServiceManager


FIXED_TIME = datetime(
    2026,
    7,
    12,
    16,
    30,
    tzinfo=timezone.utc,
)


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class TerminalDashboardTests(unittest.TestCase):
    def make_dashboard(
        self,
        *,
        stream: io.StringIO | None = None,
        event_bus: EventBus | None = None,
        color: bool | None = False,
        width: int = 88,
    ) -> tuple[
        TerminalDashboard,
        ServiceManager,
        Scheduler,
    ]:
        service_manager = ServiceManager(event_bus=event_bus)
        scheduler = Scheduler(event_bus=event_bus)

        dashboard = TerminalDashboard(
            app_name="Ghostfire OS",
            version="0.2.0",
            configuration_revision=1,
            configuration_sources=("defaults",),
            service_manager=service_manager,
            scheduler=scheduler,
            log_path="C:/Ghostfire/logs/runtime.jsonl",
            event_bus=event_bus,
            stream=stream or io.StringIO(),
            width=width,
            color=color,
            clock=lambda: FIXED_TIME,
        )

        return dashboard, service_manager, scheduler

    def test_capture_reports_online_services(self) -> None:
        dashboard, manager, scheduler = self.make_dashboard()

        manager.register("runtime", lambda: None)
        manager.register(
            "scheduler",
            lambda: None,
            dependencies=("runtime",),
            health=lambda: True,
        )
        manager.start_all()

        snapshot = dashboard.capture()

        self.assertIs(
            snapshot.overall_state,
            DashboardState.ONLINE,
        )
        self.assertEqual(len(snapshot.services), 2)
        self.assertFalse(snapshot.scheduler_running)
        self.assertEqual(snapshot.scheduled_task_count, 0)

    def test_unhealthy_service_reports_degraded(self) -> None:
        dashboard, manager, _ = self.make_dashboard()

        manager.register(
            "runtime",
            lambda: None,
            health=lambda: False,
        )
        manager.start_all()

        snapshot = dashboard.capture()

        self.assertIs(
            snapshot.overall_state,
            DashboardState.DEGRADED,
        )
        self.assertFalse(snapshot.services[0].healthy)

    def test_registered_services_report_offline(self) -> None:
        dashboard, manager, _ = self.make_dashboard()

        manager.register("runtime", lambda: None)

        snapshot = dashboard.capture()

        self.assertIs(
            snapshot.overall_state,
            DashboardState.OFFLINE,
        )

    def test_render_contains_runtime_sections(self) -> None:
        dashboard, manager, _ = self.make_dashboard()

        manager.register("runtime", lambda: None)
        manager.start_all()

        rendered = dashboard.render(dashboard.capture())

        self.assertIn("Ghostfire OS 0.2.0", rendered)
        self.assertIn("STATE: ONLINE", rendered)
        self.assertIn("Config revision: 1", rendered)
        self.assertIn("SERVICES", rendered)
        self.assertIn("runtime", rendered)
        self.assertIn("HEALTHY", rendered)

    def test_width_validation_rejects_small_dashboard(self) -> None:
        with self.assertRaises(ValueError):
            self.make_dashboard(width=60)

        with self.assertRaises(TypeError):
            self.make_dashboard(width=True)

    def test_rendered_lines_respect_configured_width(self) -> None:
        dashboard, manager, _ = self.make_dashboard(width=72)

        manager.register(
            "service-with-an-extremely-long-name",
            lambda: None,
            dependencies=(),
        )
        manager.start_all()

        rendered = dashboard.render(dashboard.capture())

        for line in rendered.splitlines():
            self.assertLessEqual(len(line), 72)

        self.assertIn("...", rendered)

    def test_snapshot_serializes_to_json(self) -> None:
        dashboard, manager, _ = self.make_dashboard()

        manager.register("runtime", lambda: None)
        manager.start_all()

        snapshot = dashboard.capture()
        payload = json.loads(
            dashboard.to_json(snapshot)
        )

        self.assertEqual(payload["app_name"], "Ghostfire OS")
        self.assertEqual(payload["overall_state"], "online")
        self.assertEqual(
            payload["captured_at"],
            FIXED_TIME.isoformat(),
        )
        self.assertEqual(
            payload["services"][0]["name"],
            "runtime",
        )

    def test_event_bus_receives_dashboard_telemetry(self) -> None:
        event_bus = EventBus()
        events: list[str] = []
        stream = io.StringIO()

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        dashboard, manager, _ = self.make_dashboard(
            stream=stream,
            event_bus=event_bus,
        )

        manager.register("runtime", lambda: None)
        manager.start_all()

        dashboard.display()

        self.assertIn(
            "ghostfire.dashboard.captured",
            events,
        )
        self.assertIn(
            "ghostfire.dashboard.displayed",
            events,
        )

    def test_color_can_be_explicitly_enabled(self) -> None:
        dashboard, manager, _ = self.make_dashboard(
            color=True,
        )

        manager.register("runtime", lambda: None)
        manager.start_all()

        rendered = dashboard.render(dashboard.capture())

        self.assertIn("\033[32mSTATE: ONLINE\033[0m", rendered)

    def test_auto_color_honors_no_color_environment(self) -> None:
        stream = TtyStringIO()
        dashboard, manager, _ = self.make_dashboard(
            stream=stream,
            color=None,
        )

        manager.register("runtime", lambda: None)
        manager.start_all()

        with patch.dict(
            os.environ,
            {"NO_COLOR": "1"},
            clear=False,
        ):
            rendered = dashboard.render(dashboard.capture())

        self.assertNotIn("\033[", rendered)

    def test_display_writes_frame_and_returns_snapshot(self) -> None:
        stream = io.StringIO()
        dashboard, manager, _ = self.make_dashboard(
            stream=stream,
        )

        manager.register("runtime", lambda: None)
        manager.start_all()

        snapshot = dashboard.display()

        self.assertIs(
            dashboard.last_snapshot,
            snapshot,
        )
        self.assertTrue(stream.getvalue().endswith("\n"))
        self.assertIn("Ghostfire OS", stream.getvalue())

    def test_concurrent_capture_is_safe(self) -> None:
        dashboard, manager, _ = self.make_dashboard()

        manager.register("runtime", lambda: None)
        manager.start_all()

        snapshots = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def capture() -> None:
            try:
                snapshot = dashboard.capture()

                with lock:
                    snapshots.append(snapshot)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=capture)
            for _ in range(6)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(snapshots), 6)
        self.assertTrue(
            all(
                snapshot.overall_state
                is DashboardState.ONLINE
                for snapshot in snapshots
            )
        )


if __name__ == "__main__":
    unittest.main()
