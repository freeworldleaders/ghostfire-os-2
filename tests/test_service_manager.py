import threading
import time
import unittest

from core.eventbus import EventBus
from core.service_manager import (
    ServiceDependencyError,
    ServiceManager,
    ServiceRegistrationError,
    ServiceStartError,
    ServiceState,
)


class ServiceManagerTests(unittest.TestCase):
    def test_register_and_list_status(self) -> None:
        manager = ServiceManager()

        status = manager.register(
            "runtime",
            lambda: None,
        )

        self.assertEqual(status.name, "runtime")
        self.assertIs(status.state, ServiceState.REGISTERED)
        self.assertEqual(status.dependencies, ())
        self.assertEqual(
            [item.name for item in manager.list_statuses()],
            ["runtime"],
        )

    def test_start_resolves_dependencies_in_order(self) -> None:
        manager = ServiceManager()
        calls: list[str] = []

        manager.register(
            "database",
            lambda: calls.append("database"),
        )
        manager.register(
            "api",
            lambda: calls.append("api"),
            dependencies=("database",),
        )
        manager.register(
            "web",
            lambda: calls.append("web"),
            dependencies=("api",),
        )

        status = manager.start("web")

        self.assertEqual(calls, ["database", "api", "web"])
        self.assertIs(status.state, ServiceState.RUNNING)
        self.assertTrue(manager.is_running("database"))
        self.assertTrue(manager.is_running("api"))

    def test_start_all_uses_registration_order(self) -> None:
        manager = ServiceManager()
        calls: list[str] = []

        manager.register(
            "alpha",
            lambda: calls.append("alpha"),
        )
        manager.register(
            "beta",
            lambda: calls.append("beta"),
        )
        manager.register(
            "gamma",
            lambda: calls.append("gamma"),
        )

        statuses = manager.start_all()

        self.assertEqual(
            [status.name for status in statuses],
            ["alpha", "beta", "gamma"],
        )
        self.assertEqual(calls, ["alpha", "beta", "gamma"])

    def test_duplicate_registration_is_rejected(self) -> None:
        manager = ServiceManager()
        manager.register("runtime", lambda: None)

        with self.assertRaises(ServiceRegistrationError):
            manager.register("runtime", lambda: None)

        with self.assertRaises(ServiceRegistrationError):
            manager.register(
                "self-dependent",
                lambda: None,
                dependencies=("self-dependent",),
            )

    def test_missing_dependency_is_rejected_at_start(self) -> None:
        manager = ServiceManager()

        manager.register(
            "api",
            lambda: None,
            dependencies=("database",),
        )

        with self.assertRaises(ServiceDependencyError):
            manager.start("api")

        self.assertIs(
            manager.status("api").state,
            ServiceState.REGISTERED,
        )

    def test_dependency_cycle_is_detected(self) -> None:
        manager = ServiceManager()

        manager.register(
            "alpha",
            lambda: None,
            dependencies=("beta",),
        )
        manager.register(
            "beta",
            lambda: None,
            dependencies=("alpha",),
        )

        with self.assertRaises(ServiceDependencyError):
            manager.start("alpha")

    def test_start_failure_rolls_back_started_dependencies(self) -> None:
        manager = ServiceManager()
        calls: list[str] = []

        manager.register(
            "database",
            lambda: calls.append("database.start"),
            stop=lambda: calls.append("database.stop"),
        )

        def fail_api() -> None:
            calls.append("api.start")
            raise RuntimeError("simulated startup failure")

        manager.register(
            "api",
            fail_api,
            dependencies=("database",),
        )

        with self.assertRaises(ServiceStartError):
            manager.start("api")

        self.assertEqual(
            calls,
            [
                "database.start",
                "api.start",
                "database.stop",
            ],
        )
        self.assertIs(
            manager.status("database").state,
            ServiceState.STOPPED,
        )
        self.assertIs(
            manager.status("api").state,
            ServiceState.FAILED,
        )

    def test_stop_all_runs_in_reverse_dependency_order(self) -> None:
        manager = ServiceManager()
        calls: list[str] = []

        manager.register(
            "database",
            lambda: calls.append("database.start"),
            stop=lambda: calls.append("database.stop"),
        )
        manager.register(
            "api",
            lambda: calls.append("api.start"),
            stop=lambda: calls.append("api.stop"),
            dependencies=("database",),
        )
        manager.register(
            "web",
            lambda: calls.append("web.start"),
            stop=lambda: calls.append("web.stop"),
            dependencies=("api",),
        )

        manager.start_all()
        manager.stop_all()

        self.assertEqual(
            calls[-3:],
            ["web.stop", "api.stop", "database.stop"],
        )

    def test_stop_cascades_through_running_dependents(self) -> None:
        manager = ServiceManager()
        stops: list[str] = []

        manager.register(
            "database",
            lambda: None,
            stop=lambda: stops.append("database"),
        )
        manager.register(
            "api",
            lambda: None,
            stop=lambda: stops.append("api"),
            dependencies=("database",),
        )
        manager.register(
            "web",
            lambda: None,
            stop=lambda: stops.append("web"),
            dependencies=("api",),
        )

        manager.start("web")

        with self.assertRaises(ServiceDependencyError):
            manager.stop("database", cascade=False)

        manager.stop("database")

        self.assertEqual(stops, ["web", "api", "database"])

    def test_restart_stops_and_starts_service(self) -> None:
        manager = ServiceManager()
        calls: list[str] = []

        manager.register(
            "worker",
            lambda: calls.append("start"),
            stop=lambda: calls.append("stop"),
        )

        manager.start("worker")
        status = manager.restart("worker")

        self.assertEqual(calls, ["start", "stop", "start"])
        self.assertEqual(status.start_count, 2)
        self.assertEqual(status.stop_count, 1)
        self.assertIs(status.state, ServiceState.RUNNING)

    def test_health_checks_publish_status(self) -> None:
        event_bus = EventBus()
        manager = ServiceManager(event_bus=event_bus)
        healthy = {"value": True}
        events: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        manager.register(
            "worker",
            lambda: None,
            health=lambda: healthy["value"],
        )
        manager.start("worker")

        self.assertTrue(manager.check_health("worker"))

        healthy["value"] = False

        self.assertFalse(manager.check_health("worker"))
        self.assertIn(
            "ghostfire.service.health_checked",
            events,
        )

    def test_event_bus_telemetry_captures_lifecycle(self) -> None:
        event_bus = EventBus()
        manager = ServiceManager(event_bus=event_bus)
        events: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        manager.register(
            "runtime",
            lambda: None,
            stop=lambda: None,
        )
        manager.start("runtime")
        manager.stop("runtime")

        self.assertIn(
            "ghostfire.service.registered",
            events,
        )
        self.assertIn(
            "ghostfire.service.starting",
            events,
        )
        self.assertIn(
            "ghostfire.service.started",
            events,
        )
        self.assertIn(
            "ghostfire.service.stopping",
            events,
        )
        self.assertIn(
            "ghostfire.service.stopped",
            events,
        )

    def test_concurrent_start_is_idempotent(self) -> None:
        manager = ServiceManager()
        start_count = 0
        count_lock = threading.Lock()
        errors: list[Exception] = []

        def start_worker() -> None:
            nonlocal start_count

            with count_lock:
                start_count += 1

            time.sleep(0.05)

        manager.register("worker", start_worker)

        def invoke_start() -> None:
            try:
                manager.start("worker")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=invoke_start)
            for _ in range(4)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(start_count, 1)
        self.assertEqual(
            manager.status("worker").start_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
