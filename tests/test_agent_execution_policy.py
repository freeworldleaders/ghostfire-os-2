import concurrent.futures
import json
import unittest

from agents.orchestrator import (
    AgentTaskOrchestrator,
    OrchestratedTaskState,
)
from agents.policy import (
    AgentExecutionPolicy,
    PolicyAction,
    PolicyApprovalRequiredError,
    PolicyDeniedError,
    PolicyEffect,
    PolicyRegistrationError,
    PolicyRequest,
    PolicyState,
    PolicyStateError,
)
from agents.registry import AgentRegistry
from agents.tools import AgentToolRegistry, ToolMode
from core.eventbus import EventBus
from core.service_manager import ServiceManager


class AgentExecutionPolicyTests(unittest.TestCase):
    def make_policy(
        self,
        *,
        history_limit: int = 100,
        default_effect: PolicyEffect | str = PolicyEffect.DENY,
        event_bus: EventBus | None = None,
    ) -> AgentExecutionPolicy:
        return AgentExecutionPolicy(
            event_bus=event_bus,
            history_limit=history_limit,
            default_effect=default_effect,
        )

    def allow_task_rule(
        self,
        policy: AgentExecutionPolicy,
        *,
        priority: int = 100,
    ) -> None:
        policy.register_rule(
            "allow-tasks",
            PolicyEffect.ALLOW,
            actions=(PolicyAction.AGENT_TASK,),
            roles=("orchestrator",),
            resources=("*",),
            modes=("execute",),
            priority=priority,
            reason="orchestrator task allowed",
        )

    def test_rule_and_request_are_immutable(self) -> None:
        policy = self.make_policy()
        rule = policy.register_rule(
            "allow-status",
            "allow",
            actions=("agent_task",),
            resources=("status",),
        )
        request = PolicyRequest.create(
            action="agent_task",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="status",
            mode="execute",
            attributes={"secret": "value"},
        )

        with self.assertRaises(AttributeError):
            rule.reason = "changed"

        with self.assertRaises(TypeError):
            request.attributes["secret"] = "changed"

        self.assertNotIn(
            "value",
            json.dumps(request.as_dict()),
        )

    def test_lifecycle_is_idempotent(self) -> None:
        policy = self.make_policy()

        self.assertTrue(policy.start())
        self.assertFalse(policy.start())
        self.assertTrue(policy.health())
        self.assertTrue(policy.stop())
        self.assertFalse(policy.stop())
        self.assertEqual(policy.state, PolicyState.STOPPED)

    def test_evaluation_requires_running_policy(self) -> None:
        policy = self.make_policy()
        request = PolicyRequest.create(
            action="agent_task",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="status",
            mode="execute",
        )

        with self.assertRaises(PolicyStateError):
            policy.evaluate(request)

    def test_default_deny_fails_closed(self) -> None:
        policy = self.make_policy()
        policy.start()

        with self.assertRaises(PolicyDeniedError) as context:
            policy.authorize(
                action="agent_task",
                agent_name="Commander",
                agent_role="orchestrator",
                resource="unknown",
                mode="execute",
            )

        self.assertIsNone(
            context.exception.decision.rule_name
        )
        policy.stop()

    def test_matching_allow_rule_authorizes(self) -> None:
        policy = self.make_policy()
        self.allow_task_rule(policy)
        policy.start()

        decision = policy.authorize(
            action="agent_task",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="status",
            mode="execute",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.rule_name, "allow-tasks")
        policy.stop()

    def test_higher_priority_deny_overrides_allow(self) -> None:
        policy = self.make_policy()
        self.allow_task_rule(policy, priority=10)
        policy.register_rule(
            "deny-command",
            "deny",
            actions=("agent_task",),
            agents=("Commander",),
            resources=("command",),
            priority=100,
            reason="command blocked",
        )
        policy.start()

        with self.assertRaises(PolicyDeniedError) as context:
            policy.authorize(
                action="agent_task",
                agent_name="Commander",
                agent_role="orchestrator",
                resource="command",
                mode="execute",
            )

        self.assertEqual(
            context.exception.decision.rule_name,
            "deny-command",
        )
        policy.stop()

    def test_require_approval_fails_closed(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "approve-mutations",
            "require_approval",
            actions=("tool_invocation",),
            resources=("*",),
            modes=("mutating",),
            priority=100,
            reason="owner approval required",
        )
        policy.start()

        with self.assertRaises(
            PolicyApprovalRequiredError
        ) as context:
            policy.authorize(
                action="tool_invocation",
                agent_name="Commander",
                agent_role="orchestrator",
                resource="system.write",
                mode="mutating",
            )

        self.assertEqual(
            context.exception.decision.effect,
            PolicyEffect.REQUIRE_APPROVAL,
        )
        policy.stop()

    def test_agent_filter_is_enforced(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "commander-only",
            "allow",
            actions=("agent_task",),
            agents=("Commander",),
            resources=("*",),
        )
        policy.start()

        with self.assertRaises(PolicyDeniedError):
            policy.authorize(
                action="agent_task",
                agent_name="Guardian",
                agent_role="safety",
                resource="status",
                mode="execute",
            )

        policy.stop()

    def test_role_filter_is_enforced(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "safety-only",
            "allow",
            actions=("agent_task",),
            roles=("safety",),
            resources=("*",),
        )
        policy.start()

        with self.assertRaises(PolicyDeniedError):
            policy.authorize(
                action="agent_task",
                agent_name="Commander",
                agent_role="orchestrator",
                resource="status",
                mode="execute",
            )

        policy.stop()

    def test_resource_wildcards_match_deterministically(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "status-family",
            "allow",
            actions=("tool_invocation",),
            resources=("ghostfire.*_status",),
            modes=("read_only",),
        )
        policy.start()

        decision = policy.authorize(
            action="tool_invocation",
            agent_name="Guardian",
            agent_role="safety",
            resource="ghostfire.agent_status",
            mode="read_only",
        )

        self.assertTrue(decision.allowed)
        policy.stop()

    def test_duplicate_rule_and_unregister(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "unique",
            "allow",
            actions=("agent_task",),
            resources=("*",),
        )

        with self.assertRaises(PolicyRegistrationError):
            policy.register_rule(
                "unique",
                "deny",
                actions=("agent_task",),
                resources=("*",),
            )

        removed = policy.unregister_rule("unique")

        self.assertEqual(removed.name, "unique")
        self.assertEqual(policy.list_rules(), ())

    def test_history_is_bounded(self) -> None:
        policy = self.make_policy(
            history_limit=2,
            default_effect="allow",
        )
        policy.start()

        for index in range(4):
            policy.authorize(
                action="agent_task",
                agent_name="Commander",
                agent_role="orchestrator",
                resource=f"task-{index}",
                mode="execute",
            )

        self.assertEqual(policy.evaluation_count, 4)
        self.assertEqual(len(policy.history()), 2)
        policy.stop()

    def test_telemetry_redacts_attribute_values(self) -> None:
        event_bus = EventBus()
        events = []
        policy = self.make_policy(
            event_bus=event_bus,
            default_effect="allow",
        )
        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event),
        )
        policy.start()
        policy.authorize(
            action="tool_invocation",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="system.echo",
            mode="read_only",
            attributes={"token": "DO_NOT_LOG_THIS"},
        )

        encoded = json.dumps(
            [event.payload for event in events],
            default=str,
        )

        self.assertNotIn("DO_NOT_LOG_THIS", encoded)
        self.assertIn("token", encoded)
        policy.stop()

    def test_concurrent_evaluations_are_thread_safe(self) -> None:
        policy = self.make_policy(default_effect="allow")
        policy.start()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            decisions = list(
                executor.map(
                    lambda index: policy.authorize(
                        action="agent_task",
                        agent_name="Commander",
                        agent_role="orchestrator",
                        resource=f"task-{index}",
                        mode="execute",
                    ).allowed,
                    range(12),
                )
            )

        self.assertTrue(all(decisions))
        self.assertEqual(policy.evaluation_count, 12)
        policy.stop()

    def test_agent_registry_integration_allows_task(self) -> None:
        policy = self.make_policy()
        self.allow_task_rule(policy)
        registry = AgentRegistry(
            execution_policy=policy,
        )
        registry.register(
            "Commander",
            role="orchestrator",
            capabilities=("status",),
        )

        policy.start()
        registry.start_all()
        result = registry.dispatch("status")

        self.assertEqual(result.status, "completed")
        self.assertTrue(
            registry.snapshot()["execution_policy_attached"]
        )

        registry.stop_all()
        policy.stop()

    def test_agent_registry_denies_before_handler(self) -> None:
        calls = []
        policy = self.make_policy()
        registry = AgentRegistry(
            execution_policy=policy,
        )
        registry.register(
            "Commander",
            role="orchestrator",
            capabilities=("command",),
            handler=lambda task, context: calls.append(
                task.identifier
            ),
        )

        policy.start()
        registry.start_all()

        with self.assertRaises(PolicyDeniedError):
            registry.dispatch("command")

        self.assertEqual(calls, [])
        registry.stop_all()
        policy.stop()

    def test_orchestrator_converts_policy_denial_to_failure(self) -> None:
        policy = self.make_policy()
        registry = AgentRegistry(
            execution_policy=policy,
        )
        registry.register(
            "Commander",
            role="orchestrator",
            capabilities=("status",),
        )
        orchestrator = AgentTaskOrchestrator(registry)

        policy.start()
        registry.start_all()
        orchestrator.start()
        orchestrator.submit(
            "status",
            identifier="denied-task",
        )
        run = orchestrator.execute_pending()

        self.assertEqual(run.status, "failed")
        self.assertEqual(
            orchestrator.get_task("denied-task").state,
            OrchestratedTaskState.FAILED,
        )

        orchestrator.stop()
        registry.stop_all()
        policy.stop()

    def test_tool_registry_policy_integration(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "allow-read",
            "allow",
            actions=("tool_invocation",),
            roles=("orchestrator",),
            resources=("system.status",),
            modes=("read_only",),
        )
        tools = AgentToolRegistry(
            execution_policy=policy,
        )
        tools.register(
            "system.status",
            lambda: "ok",
        )
        client = tools.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.status",),
        )

        policy.start()
        tools.start()
        result = client.invoke("system.status")

        self.assertEqual(result.as_dict()["output"], "ok")
        self.assertTrue(
            tools.snapshot()["execution_policy_attached"]
        )

        tools.stop()
        policy.stop()

    def test_mutating_tool_requires_approval(self) -> None:
        calls = []
        policy = self.make_policy()
        policy.register_rule(
            "approve-write",
            "require_approval",
            actions=("tool_invocation",),
            resources=("system.write",),
            modes=("mutating",),
            priority=100,
        )
        tools = AgentToolRegistry(
            allow_mutating=True,
            execution_policy=policy,
        )
        tools.register(
            "system.write",
            lambda value: calls.append(value),
            parameters={"value": str},
            required=("value",),
            mode=ToolMode.MUTATING,
        )
        client = tools.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.write",),
        )

        policy.start()
        tools.start()

        with self.assertRaises(
            PolicyApprovalRequiredError
        ):
            client.invoke(
                "system.write",
                {"value": "blocked"},
            )

        self.assertEqual(calls, [])
        tools.stop()
        policy.stop()

    def test_service_manager_lifecycle_and_snapshot(self) -> None:
        policy = self.make_policy()
        policy.register_rule(
            "allow-status",
            "allow",
            actions=("agent_task",),
            resources=("status",),
        )
        manager = ServiceManager()

        manager.register("runtime", lambda: None)
        manager.register(
            "execution_policy",
            policy.start,
            stop=policy.stop,
            dependencies=("runtime",),
            health=policy.health,
        )
        manager.start_all()

        self.assertTrue(
            manager.check_health("execution_policy")
        )
        encoded = json.dumps(policy.snapshot())
        self.assertIn('"rule_count": 1', encoded)

        manager.stop_all()
        self.assertFalse(policy.health())


if __name__ == "__main__":
    unittest.main()
