import json
import threading
import unittest

from agents.framework import AgentTask
from agents.orchestrator import (
    AgentTaskOrchestrator,
    OrchestratedTask,
    OrchestratedTaskState,
    OrchestratorCapacityError,
    OrchestratorPlanError,
    OrchestratorState,
    OrchestratorStateError,
)
from agents.registry import AgentRegistry
from core.eventbus import EventBus
from core.service_manager import ServiceManager


class AgentTaskOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.registry = AgentRegistry(event_bus=self.event_bus)
        self.registry.register(
            "Commander",
            role="orchestrator",
            capabilities=("plan", "build", "status"),
            handler=lambda task, context: {
                "task": task.identifier,
                "agent": context.agent_name,
                "payload": dict(task.payload),
            },
        )
        self.registry.register(
            "Guardian",
            role="safety",
            capabilities=("validate", "status"),
            handler=lambda task, context: {
                "validated": True,
                "agent": context.agent_name,
            },
        )
        self.registry.start_all()
        self.orchestrator = AgentTaskOrchestrator(
            self.registry,
            event_bus=self.event_bus,
            history_limit=3,
            max_tasks=20,
        )

    def tearDown(self) -> None:
        if self.orchestrator.state is OrchestratorState.RUNNING:
            self.orchestrator.stop()

        self.registry.stop_all()

    def test_task_definition_is_immutable(self) -> None:
        task = OrchestratedTask.create(
            "plan",
            {"scope": "alpha"},
            identifier="task-1",
        )

        with self.assertRaises(TypeError):
            task.payload["scope"] = "changed"

        self.assertEqual(task.capability, "plan")
        self.assertEqual(task.identifier, "task-1")

    def test_lifecycle_is_idempotent(self) -> None:
        self.assertTrue(self.orchestrator.start())
        self.assertFalse(self.orchestrator.start())
        self.assertTrue(self.orchestrator.health())
        self.assertTrue(self.orchestrator.stop())
        self.assertFalse(self.orchestrator.stop())

    def test_execution_requires_running_state(self) -> None:
        self.orchestrator.submit("plan")

        with self.assertRaises(OrchestratorStateError):
            self.orchestrator.execute_pending()

    def test_duplicate_task_ids_are_rejected(self) -> None:
        self.orchestrator.submit(
            "plan",
            identifier="duplicate",
        )

        with self.assertRaises(OrchestratorPlanError):
            self.orchestrator.submit(
                "plan",
                identifier="duplicate",
            )

    def test_missing_dependencies_are_rejected_atomically(self) -> None:
        task = OrchestratedTask.create(
            "build",
            identifier="child",
            dependencies=("missing",),
        )

        with self.assertRaises(OrchestratorPlanError):
            self.orchestrator.submit_plan((task,))

        self.assertEqual(self.orchestrator.task_count, 0)

    def test_dependency_cycles_are_rejected(self) -> None:
        first = OrchestratedTask.create(
            "plan",
            identifier="first",
            dependencies=("second",),
        )
        second = OrchestratedTask.create(
            "build",
            identifier="second",
            dependencies=("first",),
        )

        with self.assertRaises(OrchestratorPlanError):
            self.orchestrator.submit_plan((first, second))

    def test_dependency_order_is_deterministic(self) -> None:
        order: list[str] = []
        registry = AgentRegistry()

        registry.register(
            "Worker",
            capabilities=("step",),
            handler=lambda task, context: order.append(
                task.payload["name"]
            ) or task.payload["name"],
        )
        registry.start_all()

        orchestrator = AgentTaskOrchestrator(registry)
        orchestrator.start()
        orchestrator.submit(
            "step",
            {"name": "first"},
            identifier="first",
        )
        orchestrator.submit(
            "step",
            {"name": "second"},
            identifier="second",
            dependencies=("first",),
        )
        run = orchestrator.execute_pending()

        self.assertEqual(order, ["first", "second"])
        self.assertEqual(run.status, "completed")

        orchestrator.stop()
        registry.stop_all()

    def test_preferred_agent_is_respected(self) -> None:
        self.orchestrator.start()
        self.orchestrator.submit(
            "validate",
            identifier="guarded",
            preferred_agent="Guardian",
        )
        self.orchestrator.execute_pending()

        snapshot = self.orchestrator.get_task("guarded")

        self.assertEqual(snapshot.agent_name, "Guardian")
        self.assertEqual(
            snapshot.state,
            OrchestratedTaskState.COMPLETED,
        )

    def test_retry_can_fail_over_to_next_healthy_agent(self) -> None:
        registry = AgentRegistry()
        attempts: list[str] = []

        def fail(task: AgentTask, context):
            attempts.append(context.agent_name)
            raise RuntimeError("first worker failed")

        def succeed(task: AgentTask, context):
            attempts.append(context.agent_name)
            return "recovered"

        registry.register(
            "First",
            capabilities=("recover",),
            handler=fail,
        )
        registry.register(
            "Second",
            capabilities=("recover",),
            handler=succeed,
        )
        registry.start_all()

        orchestrator = AgentTaskOrchestrator(registry)
        orchestrator.start()
        orchestrator.submit(
            "recover",
            identifier="retry",
            max_attempts=2,
        )
        run = orchestrator.execute_pending()
        snapshot = orchestrator.get_task("retry")

        self.assertEqual(run.status, "completed")
        self.assertEqual(snapshot.attempts, 2)
        self.assertEqual(snapshot.agent_name, "Second")
        self.assertEqual(attempts, ["First", "Second"])

        orchestrator.stop()
        registry.stop_all()

    def test_failed_task_blocks_dependent(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "Failure",
            capabilities=("fail",),
            handler=lambda task, context: (_ for _ in ()).throw(
                RuntimeError("failure")
            ),
        )
        registry.register(
            "Builder",
            capabilities=("build",),
        )
        registry.start_all()

        orchestrator = AgentTaskOrchestrator(registry)
        orchestrator.start()
        orchestrator.submit(
            "fail",
            identifier="root",
        )
        orchestrator.submit(
            "build",
            identifier="dependent",
            dependencies=("root",),
        )
        run = orchestrator.execute_pending()

        self.assertEqual(run.status, "failed")
        self.assertEqual(
            orchestrator.get_task("root").state,
            OrchestratedTaskState.FAILED,
        )
        self.assertEqual(
            orchestrator.get_task("dependent").state,
            OrchestratedTaskState.BLOCKED,
        )

        orchestrator.stop()
        registry.stop_all()

    def test_independent_task_continues_after_failure(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "Failure",
            capabilities=("fail",),
            handler=lambda task, context: (_ for _ in ()).throw(
                RuntimeError("failure")
            ),
        )
        registry.register(
            "Builder",
            capabilities=("build",),
            handler=lambda task, context: "built",
        )
        registry.start_all()

        orchestrator = AgentTaskOrchestrator(registry)
        orchestrator.start()
        orchestrator.submit("fail", identifier="bad")
        orchestrator.submit("build", identifier="good")
        run = orchestrator.execute_pending()

        self.assertEqual(run.status, "partial")
        self.assertEqual(
            orchestrator.get_task("good").state,
            OrchestratedTaskState.COMPLETED,
        )

        orchestrator.stop()
        registry.stop_all()

    def test_cancel_pending_task(self) -> None:
        self.orchestrator.submit(
            "plan",
            identifier="cancel-me",
        )
        cancelled = self.orchestrator.cancel("cancel-me")

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(
            cancelled[0].state,
            OrchestratedTaskState.CANCELLED,
        )

    def test_cascade_cancel_includes_dependents(self) -> None:
        self.orchestrator.submit(
            "plan",
            identifier="root",
        )
        self.orchestrator.submit(
            "build",
            identifier="child",
            dependencies=("root",),
        )
        cancelled = self.orchestrator.cancel(
            "root",
            cascade=True,
        )

        self.assertEqual(
            [item.task.identifier for item in cancelled],
            ["root", "child"],
        )

    def test_capacity_limit_is_enforced(self) -> None:
        orchestrator = AgentTaskOrchestrator(
            self.registry,
            max_tasks=1,
        )
        orchestrator.submit("plan", identifier="one")

        with self.assertRaises(OrchestratorCapacityError):
            orchestrator.submit("plan", identifier="two")

    def test_snapshot_is_json_safe(self) -> None:
        self.orchestrator.start()
        self.orchestrator.submit(
            "plan",
            {"scope": "snapshot"},
            identifier="snapshot",
        )
        self.orchestrator.execute_pending()

        encoded = json.dumps(self.orchestrator.snapshot())

        self.assertIn('"state": "running"', encoded)
        self.assertIn('"snapshot"', encoded)

    def test_run_history_is_bounded(self) -> None:
        self.orchestrator.start()

        for index in range(5):
            self.orchestrator.submit(
                "plan",
                identifier=f"task-{index}",
            )
            self.orchestrator.execute_pending()

        self.assertEqual(self.orchestrator.run_count, 5)
        self.assertEqual(
            len(self.orchestrator.run_history()),
            3,
        )

    def test_event_bus_receives_orchestration_telemetry(self) -> None:
        events: list[str] = []

        self.event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        self.orchestrator.start()
        self.orchestrator.submit(
            "plan",
            identifier="telemetry",
        )
        self.orchestrator.execute_pending()
        self.orchestrator.stop()

        self.assertIn(
            "ghostfire.orchestrator.started",
            events,
        )
        self.assertIn(
            "ghostfire.orchestrator.task.submitted",
            events,
        )
        self.assertIn(
            "ghostfire.orchestrator.task.completed",
            events,
        )
        self.assertIn(
            "ghostfire.orchestrator.run.completed",
            events,
        )
        self.assertIn(
            "ghostfire.orchestrator.stopped",
            events,
        )

    def test_concurrent_execute_is_rejected(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        registry = AgentRegistry()

        def wait_handler(task, context):
            entered.set()
            release.wait(timeout=2)
            return "done"

        registry.register(
            "Worker",
            capabilities=("wait",),
            handler=wait_handler,
        )
        registry.start_all()

        orchestrator = AgentTaskOrchestrator(registry)
        orchestrator.start()
        orchestrator.submit("wait", identifier="wait")

        worker = threading.Thread(
            target=orchestrator.execute_pending,
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))

        with self.assertRaises(OrchestratorStateError):
            orchestrator.execute_pending()

        release.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())

        orchestrator.stop()
        registry.stop_all()

    def test_empty_run_completes_cleanly(self) -> None:
        self.orchestrator.start()
        run = self.orchestrator.execute_pending()

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.task_ids, ())
        self.assertEqual(
            run.counts["completed"],
            0,
        )

    def test_task_result_is_retained(self) -> None:
        self.orchestrator.start()
        self.orchestrator.submit(
            "plan",
            {"scope": "retained"},
            identifier="retained",
        )
        self.orchestrator.execute_pending()
        snapshot = self.orchestrator.get_task("retained")

        self.assertIsNotNone(snapshot.result)
        self.assertEqual(
            snapshot.result.output["payload"]["scope"],
            "retained",
        )

    def test_service_manager_controls_orchestrator(self) -> None:
        manager = ServiceManager(event_bus=self.event_bus)

        manager.register("runtime", lambda: None)
        manager.register(
            "agents",
            self.registry.start_all,
            stop=self.registry.stop_all,
            dependencies=("runtime",),
            health=self.registry.health,
        )
        manager.register(
            "agent_orchestrator",
            self.orchestrator.start,
            stop=self.orchestrator.stop,
            dependencies=("agents",),
            health=self.orchestrator.health,
        )

        manager.start_all()

        self.assertTrue(self.orchestrator.health())
        self.assertTrue(
            manager.check_health("agent_orchestrator")
        )

        manager.stop_all()

        self.assertFalse(self.orchestrator.health())


if __name__ == "__main__":
    unittest.main()
