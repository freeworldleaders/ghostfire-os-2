import concurrent.futures
import json
import threading
import unittest

from agents.framework import AgentTask
from agents.registry import AgentRegistry
from agents.tools import (
    AgentToolRegistry,
    ToolAuthorizationError,
    ToolExecutionError,
    ToolMode,
    ToolRegistrationError,
    ToolStateError,
    ToolValidationError,
)
from core.eventbus import EventBus
from core.service_manager import ServiceManager


class AgentToolRegistryTests(unittest.TestCase):
    def make_registry(
        self,
        *,
        history_limit: int = 100,
        allow_mutating: bool = False,
    ) -> AgentToolRegistry:
        return AgentToolRegistry(
            event_bus=EventBus(),
            history_limit=history_limit,
            allow_mutating=allow_mutating,
        )

    def test_definition_is_immutable_and_json_safe(self) -> None:
        registry = self.make_registry()
        definition = registry.register(
            "system.echo",
            lambda message: message,
            parameters={"message": str},
            required=("message",),
        )

        with self.assertRaises(TypeError):
            definition.parameters["message"] = (int,)

        encoded = json.dumps(definition.as_dict())

        self.assertIn('"system.echo"', encoded)
        self.assertIn('"required": true', encoded)

    def test_lifecycle_is_idempotent(self) -> None:
        registry = self.make_registry()

        self.assertTrue(registry.start())
        self.assertFalse(registry.start())
        self.assertTrue(registry.health())
        self.assertTrue(registry.stop())
        self.assertFalse(registry.stop())

    def test_invocation_requires_running_registry(self) -> None:
        registry = self.make_registry()
        registry.register("system.status", lambda: "ok")
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.status",),
        )

        with self.assertRaises(ToolStateError):
            client.invoke("system.status")

    def test_duplicate_registration_and_unregister(self) -> None:
        registry = self.make_registry()
        registry.register("system.status", lambda: "ok")

        with self.assertRaises(ToolRegistrationError):
            registry.register("system.status", lambda: "duplicate")

        removed = registry.unregister("system.status")

        self.assertEqual(removed.name, "system.status")
        self.assertEqual(registry.list_tools(), ())

    def test_missing_required_argument_is_rejected(self) -> None:
        registry = self.make_registry()
        registry.register(
            "system.echo",
            lambda message: message,
            parameters={"message": str},
            required=("message",),
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.echo",),
        )

        with self.assertRaises(ToolValidationError):
            client.invoke("system.echo")

        registry.stop()

    def test_unknown_argument_is_rejected(self) -> None:
        registry = self.make_registry()
        registry.register("system.status", lambda: "ok")
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.status",),
        )

        with self.assertRaises(ToolValidationError):
            client.invoke(
                "system.status",
                {"unexpected": True},
            )

        registry.stop()

    def test_argument_types_are_enforced(self) -> None:
        registry = self.make_registry()
        registry.register(
            "math.double",
            lambda value: value * 2,
            parameters={"value": int},
            required=("value",),
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("math.double",),
        )

        with self.assertRaises(ToolValidationError):
            client.invoke("math.double", {"value": "2"})

        with self.assertRaises(ToolValidationError):
            client.invoke("math.double", {"value": True})

        registry.stop()

    def test_read_only_invocation_returns_detached_result(self) -> None:
        registry = self.make_registry()
        original = {"items": ["alpha"]}

        registry.register(
            "system.copy",
            lambda: original,
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.copy",),
        )
        result = client.invoke("system.copy")
        original["items"].append("changed")

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            result.as_dict()["output"],
            {"items": ["alpha"]},
        )

        registry.stop()

    def test_agent_allowlist_is_enforced(self) -> None:
        registry = self.make_registry()
        registry.register(
            "system.private",
            lambda: "ok",
            allowed_agents=("Commander",),
        )
        registry.start()
        client = registry.client(
            agent_name="Guardian",
            agent_role="safety",
            allowed_tools=("system.private",),
        )

        with self.assertRaises(ToolAuthorizationError):
            client.invoke("system.private")

        registry.stop()

    def test_role_allowlist_is_enforced(self) -> None:
        registry = self.make_registry()
        registry.register(
            "system.plan",
            lambda: "ok",
            allowed_roles=("orchestrator",),
        )
        registry.start()
        client = registry.client(
            agent_name="Guardian",
            agent_role="safety",
            allowed_tools=("system.plan",),
        )

        with self.assertRaises(ToolAuthorizationError):
            client.invoke("system.plan")

        registry.stop()

    def test_client_allowlist_is_enforced(self) -> None:
        registry = self.make_registry()
        registry.register("system.status", lambda: "ok")
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=(),
        )

        with self.assertRaises(ToolAuthorizationError):
            client.invoke("system.status")

        self.assertEqual(client.list_tools(), ())
        registry.stop()

    def test_mutating_tool_is_disabled_by_default(self) -> None:
        registry = self.make_registry()
        registry.register(
            "system.write",
            lambda value: value,
            parameters={"value": str},
            required=("value",),
            mode=ToolMode.MUTATING,
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.write",),
        )

        with self.assertRaises(ToolAuthorizationError):
            client.invoke(
                "system.write",
                {"value": "blocked"},
            )

        registry.stop()

    def test_mutating_tool_can_be_owner_enabled(self) -> None:
        registry = self.make_registry(allow_mutating=True)
        registry.register(
            "system.write",
            lambda value: {"written": value},
            parameters={"value": str},
            required=("value",),
            mode=ToolMode.MUTATING,
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.write",),
        )

        result = client.invoke(
            "system.write",
            {"value": "approved"},
        )

        self.assertEqual(
            result.as_dict()["output"],
            {"written": "approved"},
        )
        registry.stop()

    def test_execution_failure_is_structured_and_retained(self) -> None:
        registry = self.make_registry()

        def fail() -> None:
            raise RuntimeError("tool failed")

        registry.register("system.fail", fail)
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.fail",),
        )

        with self.assertRaises(ToolExecutionError) as context:
            client.invoke("system.fail")

        self.assertEqual(context.exception.result.status, "failed")
        self.assertEqual(registry.failure_count, 1)
        self.assertEqual(len(registry.history()), 1)
        registry.stop()

    def test_history_is_bounded(self) -> None:
        registry = self.make_registry(history_limit=2)
        registry.register(
            "system.echo",
            lambda value: value,
            parameters={"value": int},
            required=("value",),
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.echo",),
        )

        for value in range(4):
            client.invoke("system.echo", {"value": value})

        self.assertEqual(registry.invocation_count, 4)
        self.assertEqual(len(registry.history()), 2)
        registry.stop()

    def test_telemetry_redacts_argument_values(self) -> None:
        event_bus = EventBus()
        events = []
        registry = AgentToolRegistry(event_bus=event_bus)

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event),
        )
        registry.register(
            "system.secret",
            lambda token: "done",
            parameters={"token": str},
            required=("token",),
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.secret",),
        )
        client.invoke(
            "system.secret",
            {"token": "DO_NOT_LOG_THIS"},
        )

        serialized = json.dumps(
            [
                event.payload
                for event in events
            ],
            default=str,
        )

        self.assertNotIn("DO_NOT_LOG_THIS", serialized)
        self.assertIn("token", serialized)
        registry.stop()

    def test_concurrent_invocations_are_thread_safe(self) -> None:
        registry = self.make_registry()
        registry.register(
            "math.square",
            lambda value: value * value,
            parameters={"value": int},
            required=("value",),
        )
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("math.square",),
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            results = list(
                executor.map(
                    lambda value: client.invoke(
                        "math.square",
                        {"value": value},
                    ).as_dict()["output"],
                    range(8),
                )
            )

        self.assertEqual(
            results,
            [value * value for value in range(8)],
        )
        self.assertEqual(registry.invocation_count, 8)
        self.assertEqual(registry.active_count, 0)
        registry.stop()

    def test_active_invocation_blocks_stop_and_unregister(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        registry = self.make_registry()

        def wait() -> str:
            entered.set()
            release.wait(timeout=2)
            return "done"

        registry.register("system.wait", wait)
        registry.start()
        client = registry.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.wait",),
        )
        worker = threading.Thread(
            target=lambda: client.invoke("system.wait"),
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))

        with self.assertRaises(ToolStateError):
            registry.stop()

        with self.assertRaises(ToolStateError):
            registry.unregister("system.wait")

        release.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        registry.stop()

    def test_agent_context_receives_bound_tool_client(self) -> None:
        tools = self.make_registry()
        tools.register(
            "system.echo",
            lambda message: {"message": message},
            parameters={"message": str},
            required=("message",),
            allowed_roles=("orchestrator",),
        )
        agents = AgentRegistry(tool_registry=tools)

        agents.register(
            "Commander",
            role="orchestrator",
            capabilities=("use_tool",),
            allowed_tools=("system.echo",),
            handler=lambda task, context: context.tools.invoke(
                "system.echo",
                {"message": task.payload["message"]},
            ).as_dict()["output"],
        )

        tools.start()
        agents.start_all()
        result = agents.dispatch(
            "use_tool",
            {"message": "GhostFire"},
        )

        self.assertEqual(
            result.output,
            {"message": "GhostFire"},
        )
        self.assertEqual(
            agents.get("Commander").snapshot()["tools"][
                "allowed_tools"
            ],
            ["system.echo"],
        )

        agents.stop_all()
        tools.stop()

    def test_service_manager_lifecycle_and_snapshot(self) -> None:
        registry = self.make_registry()
        registry.register("system.status", lambda: "ok")
        manager = ServiceManager()

        manager.register("runtime", lambda: None)
        manager.register(
            "agent_tools",
            registry.start,
            stop=registry.stop,
            dependencies=("runtime",),
            health=registry.health,
        )
        manager.start_all()

        self.assertTrue(manager.check_health("agent_tools"))
        encoded = json.dumps(registry.snapshot())
        self.assertIn('"tool_count": 1', encoded)

        manager.stop_all()
        self.assertFalse(registry.health())


if __name__ == "__main__":
    unittest.main()
