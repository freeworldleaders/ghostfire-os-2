import concurrent.futures
import json
import unittest

from agents.base import Agent as LegacyAgent
from agents.framework import (
    Agent,
    AgentCapabilityError,
    AgentExecutionError,
    AgentRegistrationError,
    AgentState,
    AgentStateError,
    AgentTask,
)
from agents.registry import AgentRegistry
from core.eventbus import EventBus
from core.service_manager import ServiceManager


class AgentFrameworkTests(unittest.TestCase):
    def test_task_is_immutable_and_serializable(self) -> None:
        source = {"nested": {"value": 1}}
        task = AgentTask.create(
            "STATUS",
            source,
            metadata={"source": "test"},
            identifier="task-1",
        )
        source["nested"]["value"] = 99

        self.assertEqual(task.capability, "status")
        self.assertEqual(task.payload["nested"]["value"], 1)
        with self.assertRaises(TypeError):
            task.payload["new"] = True
        self.assertEqual(
            json.loads(json.dumps(task.as_dict()))["identifier"],
            "task-1",
        )

    def test_registration_validation(self) -> None:
        with self.assertRaises(ValueError):
            Agent("")
        with self.assertRaises(TypeError):
            Agent("A", capabilities="status")
        with self.assertRaises(ValueError):
            Agent("A", capabilities=())

        agent = Agent(
            "A",
            capabilities=("STATUS", "status", "COMMAND"),
        )
        self.assertEqual(
            agent.capabilities,
            ("status", "command"),
        )

    def test_start_stop_idempotency_and_health(self) -> None:
        agent = Agent("Commander")

        self.assertTrue(agent.start())
        self.assertFalse(agent.start())
        self.assertTrue(agent.health())
        self.assertEqual(agent.state, AgentState.ONLINE)
        self.assertTrue(agent.stop())
        self.assertFalse(agent.stop())
        self.assertFalse(agent.health())

    def test_execute_requires_online_state(self) -> None:
        agent = Agent("Commander")
        task = AgentTask.create("status")

        with self.assertRaises(AgentStateError):
            agent.execute(task)

    def test_unsupported_capability_is_rejected(self) -> None:
        agent = Agent(
            "Commander",
            capabilities=("status",),
        )
        agent.start()

        with self.assertRaises(AgentCapabilityError):
            agent.execute(AgentTask.create("command"))

    def test_successful_execution_returns_result_and_history(self) -> None:
        agent = Agent(
            "Commander",
            role="orchestrator",
            capabilities=("command",),
            handler=lambda task, context: {
                "value": task.payload["value"],
                "role": context.role,
            },
        )
        agent.start()
        result = agent.execute(
            AgentTask.create(
                "command",
                {"value": 7},
                identifier="task-success",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output["value"], 7)
        self.assertEqual(result.output["role"], "orchestrator")
        self.assertEqual(agent.execution_count, 1)
        self.assertEqual(agent.history(), (result,))
        self.assertEqual(agent.state, AgentState.ONLINE)

    def test_handler_failure_records_degraded_result(self) -> None:
        def fail(task, context):
            raise RuntimeError("boom")

        agent = Agent(
            "Guardian",
            capabilities=("validate",),
            handler=fail,
        )
        agent.start()

        with self.assertRaises(AgentExecutionError) as captured:
            agent.execute(AgentTask.create("validate"))

        self.assertEqual(
            captured.exception.result.status,
            "failed",
        )
        self.assertEqual(agent.failure_count, 1)
        self.assertEqual(agent.state, AgentState.DEGRADED)
        self.assertIn("RuntimeError: boom", agent.last_error)

    def test_restart_recovers_degraded_agent(self) -> None:
        agent = Agent(
            "Guardian",
            capabilities=("validate",),
            handler=lambda task, context: (_ for _ in ()).throw(
                ValueError("bad")
            ),
        )
        agent.start()

        with self.assertRaises(AgentExecutionError):
            agent.execute(AgentTask.create("validate"))

        self.assertTrue(agent.start())
        self.assertTrue(agent.health())
        self.assertIsNone(agent.last_error)

    def test_memory_is_bounded_and_managed_through_api(self) -> None:
        agent = Agent("Archivist", memory_limit=2)

        agent.set_memory("one", {"value": 1})
        agent.set_memory("two", {"value": 2})
        agent.set_memory("three", {"value": 3})

        self.assertIsNone(agent.get_memory("one"))
        self.assertEqual(
            agent.get_memory("two"),
            {"value": 2},
        )
        self.assertTrue(agent.delete_memory("two"))
        self.assertFalse(agent.delete_memory("missing"))
        self.assertEqual(agent.clear_memory(), 1)
        self.assertEqual(agent.memory_snapshot(), {})

    def test_history_limit_evicts_old_results(self) -> None:
        agent = Agent(
            "Worker",
            history_limit=2,
        )
        agent.start()

        for index in range(3):
            agent.execute(
                AgentTask.create(
                    "status",
                    {"index": index},
                    identifier=f"task-{index}",
                )
            )

        self.assertEqual(
            [result.task_id for result in agent.history()],
            ["task-1", "task-2"],
        )

    def test_event_bus_receives_agent_telemetry(self) -> None:
        event_bus = EventBus()
        events: list[str] = []
        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )
        agent = Agent(
            "Commander",
            event_bus=event_bus,
        )

        agent.start()
        agent.set_memory("mission", "active")
        agent.execute(AgentTask.create("status"))
        agent.stop()

        self.assertIn("ghostfire.agent.started", events)
        self.assertIn(
            "ghostfire.agent.memory.updated",
            events,
        )
        self.assertIn(
            "ghostfire.agent.task.started",
            events,
        )
        self.assertIn(
            "ghostfire.agent.task.completed",
            events,
        )
        self.assertIn("ghostfire.agent.stopped", events)

    def test_registry_rejects_duplicate_agents(self) -> None:
        registry = AgentRegistry()
        registry.register("Commander")

        with self.assertRaises(AgentRegistrationError):
            registry.register("Commander")

    def test_registry_start_stop_and_health(self) -> None:
        registry = AgentRegistry()
        registry.register("Commander")
        registry.register("Guardian")

        self.assertEqual(
            registry.start_all(),
            ("Commander", "Guardian"),
        )
        self.assertTrue(registry.health())
        self.assertEqual(
            registry.stop_all(),
            ("Guardian", "Commander"),
        )
        self.assertFalse(registry.health())

    def test_registry_dispatches_to_preferred_agent(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "Commander",
            capabilities=("status",),
        )
        registry.register(
            "Guardian",
            capabilities=("status",),
        )
        registry.start_all()

        result = registry.dispatch(
            "status",
            preferred_agent="Guardian",
        )

        self.assertEqual(result.agent_name, "Guardian")

    def test_registry_routes_to_first_capable_agent(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "Commander",
            capabilities=("command",),
        )
        registry.register(
            "Guardian",
            capabilities=("validate",),
        )
        registry.start_all()

        result = registry.dispatch("validate")

        self.assertEqual(result.agent_name, "Guardian")

    def test_registry_rejects_missing_capability(self) -> None:
        registry = AgentRegistry()
        registry.register("Commander")
        registry.start_all()

        with self.assertRaises(AgentCapabilityError):
            registry.dispatch("unknown")

    def test_registry_unregister_requires_stopped_agent(self) -> None:
        registry = AgentRegistry()
        agent = registry.register("Commander")
        agent.start()

        with self.assertRaises(AgentRegistrationError):
            registry.unregister("Commander")

        agent.stop()
        removed = registry.unregister("Commander")
        self.assertIs(removed, agent)
        self.assertEqual(registry.list_agents(), ())

    def test_registry_snapshot_is_json_serializable(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "Commander",
            role="orchestrator",
            capabilities=("command", "status"),
        )
        registry.start_all()

        payload = registry.snapshot()
        encoded = json.dumps(payload)

        self.assertIn('"Commander"', encoded)
        self.assertTrue(payload["healthy"])
        self.assertEqual(payload["count"], 1)

    def test_concurrent_dispatch_preserves_history(self) -> None:
        registry = AgentRegistry(history_limit=32)
        agent = registry.register(
            "Worker",
            capabilities=("work",),
            handler=lambda task, context: task.payload["index"],
        )
        registry.start_all()

        def dispatch(index: int) -> int:
            result = registry.dispatch(
                "work",
                {"index": index},
            )
            return result.output

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            outputs = list(executor.map(dispatch, range(16)))

        self.assertEqual(sorted(outputs), list(range(16)))
        self.assertEqual(len(agent.history()), 16)
        self.assertEqual(agent.execution_count, 16)
        self.assertTrue(agent.health())

    def test_service_manager_controls_agent_registry(self) -> None:
        event_bus = EventBus()
        registry = AgentRegistry(event_bus=event_bus)
        registry.register("Commander")
        registry.register("Guardian")

        manager = ServiceManager(event_bus=event_bus)
        manager.register("runtime", lambda: None)
        manager.register(
            "agents",
            registry.start_all,
            stop=registry.stop_all,
            dependencies=("runtime",),
            health=registry.health,
        )

        manager.start_all()

        self.assertTrue(manager.check_health("agents"))
        self.assertTrue(registry.health())

        manager.stop_all()

        self.assertFalse(registry.health())
        self.assertIs(LegacyAgent, Agent)


if __name__ == "__main__":
    unittest.main()
