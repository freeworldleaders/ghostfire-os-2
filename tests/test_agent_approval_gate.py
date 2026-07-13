import concurrent.futures
import json
import unittest
from datetime import datetime, timedelta, timezone

from agents.approval import (
    AgentApprovalGate,
    ApprovalCapacityError,
    ApprovalGateState,
    ApprovalGateStateError,
    ApprovalIdentityError,
    ApprovalRequestError,
    ApprovalStatus,
)
from agents.policy import (
    AgentExecutionPolicy,
    PolicyAction,
    PolicyApprovalRequiredError,
    PolicyDeniedError,
    PolicyEffect,
)
from agents.tools import AgentToolRegistry, ToolMode
from core.eventbus import EventBus
from core.service_manager import ServiceManager


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(
            2026,
            7,
            12,
            tzinfo=timezone.utc,
        )

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class AgentApprovalGateTests(unittest.TestCase):
    def make_gate(
        self,
        *,
        history_limit: int = 100,
        max_pending: int = 100,
        ttl: float = 300.0,
        event_bus: EventBus | None = None,
        clock=None,
    ) -> AgentApprovalGate:
        return AgentApprovalGate(
            event_bus=event_bus,
            history_limit=history_limit,
            max_pending=max_pending,
            approval_ttl_seconds=ttl,
            owner_identity="owner",
            clock=clock,
        )

    def request(
        self,
        gate: AgentApprovalGate,
        *,
        value: str = "alpha",
    ):
        return gate.authorize_or_request(
            action="tool_invocation",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="system.write",
            mode="mutating",
            attributes={"value": value},
            policy_rule="approval-rule",
            policy_reason="owner approval required",
        )

    def make_policy(
        self,
        gate: AgentApprovalGate,
    ) -> AgentExecutionPolicy:
        policy = AgentExecutionPolicy(
            approval_gate=gate,
        )
        policy.register_rule(
            "approve-mutations",
            PolicyEffect.REQUIRE_APPROVAL,
            actions=(PolicyAction.TOOL_INVOCATION,),
            roles=("orchestrator",),
            resources=("system.write",),
            modes=("mutating",),
            priority=100,
            reason="owner approval required",
        )
        return policy

    def test_snapshot_is_immutable_and_redacted(self) -> None:
        gate = self.make_gate()
        gate.start()
        snapshot = self.request(gate)

        with self.assertRaises(AttributeError):
            snapshot.status = ApprovalStatus.APPROVED

        encoded = json.dumps(snapshot.as_dict())

        self.assertNotIn("alpha", encoded)
        self.assertIn("value", encoded)
        gate.stop()

    def test_lifecycle_is_idempotent(self) -> None:
        gate = self.make_gate()

        self.assertTrue(gate.start())
        self.assertFalse(gate.start())
        self.assertTrue(gate.health())
        self.assertTrue(gate.stop())
        self.assertFalse(gate.stop())
        self.assertEqual(
            gate.state,
            ApprovalGateState.STOPPED,
        )

    def test_request_requires_running_gate(self) -> None:
        gate = self.make_gate()

        with self.assertRaises(ApprovalGateStateError):
            self.request(gate)

    def test_exact_pending_request_is_deduplicated(self) -> None:
        gate = self.make_gate()
        gate.start()

        first = self.request(gate)
        second = self.request(gate)

        self.assertEqual(first.identifier, second.identifier)
        self.assertEqual(gate.pending_count, 1)
        gate.stop()

    def test_changed_attributes_require_new_approval(self) -> None:
        gate = self.make_gate()
        gate.start()

        first = self.request(gate, value="alpha")
        second = self.request(gate, value="beta")

        self.assertNotEqual(
            first.fingerprint,
            second.fingerprint,
        )
        self.assertEqual(gate.pending_count, 2)
        gate.stop()

    def test_only_configured_owner_can_approve(self) -> None:
        gate = self.make_gate()
        gate.start()
        pending = self.request(gate)

        with self.assertRaises(ApprovalIdentityError):
            gate.approve(
                pending.identifier,
                decided_by="intruder",
            )

        self.assertEqual(
            gate.get(pending.identifier).status,
            ApprovalStatus.PENDING,
        )
        gate.stop()

    def test_approved_request_is_consumed_once(self) -> None:
        gate = self.make_gate()
        gate.start()
        pending = self.request(gate)

        approved = gate.approve(
            pending.identifier,
            decided_by="owner",
            note="approved",
        )
        consumed = self.request(gate)

        self.assertEqual(
            approved.status,
            ApprovalStatus.APPROVED,
        )
        self.assertEqual(
            consumed.status,
            ApprovalStatus.CONSUMED,
        )
        self.assertEqual(gate.pending_count, 0)
        gate.stop()

    def test_consumed_request_requires_new_approval(self) -> None:
        gate = self.make_gate()
        gate.start()
        pending = self.request(gate)
        gate.approve(
            pending.identifier,
            decided_by="owner",
        )
        self.request(gate)

        next_request = self.request(gate)

        self.assertEqual(
            next_request.status,
            ApprovalStatus.PENDING,
        )
        self.assertNotEqual(
            next_request.identifier,
            pending.identifier,
        )
        gate.stop()

    def test_denial_is_sticky_until_cancelled(self) -> None:
        gate = self.make_gate()
        gate.start()
        pending = self.request(gate)
        denied = gate.deny(
            pending.identifier,
            decided_by="owner",
            note="not approved",
        )
        repeated = self.request(gate)

        self.assertEqual(
            denied.status,
            ApprovalStatus.DENIED,
        )
        self.assertEqual(
            repeated.identifier,
            pending.identifier,
        )

        gate.cancel(
            pending.identifier,
            decided_by="owner",
        )
        replacement = self.request(gate)

        self.assertNotEqual(
            replacement.identifier,
            pending.identifier,
        )
        gate.stop()

    def test_expired_request_is_replaced(self) -> None:
        clock = MutableClock()
        gate = self.make_gate(
            ttl=10,
            clock=clock,
        )
        gate.start()
        pending = self.request(gate)
        clock.advance(11)

        replacement = self.request(gate)

        self.assertEqual(
            gate.get(pending.identifier).status,
            ApprovalStatus.EXPIRED,
        )
        self.assertNotEqual(
            replacement.identifier,
            pending.identifier,
        )
        gate.stop()

    def test_capacity_gate_is_enforced(self) -> None:
        gate = self.make_gate(max_pending=1)
        gate.start()
        self.request(gate, value="one")

        with self.assertRaises(ApprovalCapacityError):
            self.request(gate, value="two")

        gate.stop()

    def test_invalid_state_transition_is_rejected(self) -> None:
        gate = self.make_gate()
        gate.start()
        pending = self.request(gate)
        gate.approve(
            pending.identifier,
            decided_by="owner",
        )

        with self.assertRaises(ApprovalRequestError):
            gate.deny(
                pending.identifier,
                decided_by="owner",
            )

        gate.stop()

    def test_history_is_bounded(self) -> None:
        gate = self.make_gate(history_limit=2)
        gate.start()

        for index in range(3):
            pending = self.request(
                gate,
                value=str(index),
            )
            gate.approve(
                pending.identifier,
                decided_by="owner",
            )
            self.request(gate, value=str(index))

        self.assertEqual(len(gate.history()), 2)
        gate.stop()

    def test_telemetry_redacts_attribute_values(self) -> None:
        event_bus = EventBus()
        events = []
        gate = self.make_gate(event_bus=event_bus)

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event),
        )
        gate.start()
        self.request(gate, value="DO_NOT_LOG_THIS")

        encoded = json.dumps(
            [event.payload for event in events],
            default=str,
        )

        self.assertNotIn("DO_NOT_LOG_THIS", encoded)
        self.assertIn("value", encoded)
        gate.stop()

    def test_concurrent_requests_deduplicate_safely(self) -> None:
        gate = self.make_gate()
        gate.start()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=6
        ) as executor:
            identifiers = list(
                executor.map(
                    lambda _: self.request(gate).identifier,
                    range(12),
                )
            )

        self.assertEqual(len(set(identifiers)), 1)
        self.assertEqual(gate.pending_count, 1)
        gate.stop()

    def test_policy_creates_pending_request(self) -> None:
        gate = self.make_gate()
        policy = self.make_policy(gate)
        gate.start()
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
                attributes={"value": "alpha"},
            )

        self.assertEqual(
            context.exception.approval.status,
            ApprovalStatus.PENDING,
        )
        policy.stop()
        gate.stop()

    def test_policy_replay_consumes_approval(self) -> None:
        gate = self.make_gate()
        policy = self.make_policy(gate)
        gate.start()
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
                attributes={"value": "alpha"},
            )

        gate.approve(
            context.exception.approval.identifier,
            decided_by="owner",
        )
        decision = policy.authorize(
            action="tool_invocation",
            agent_name="Commander",
            agent_role="orchestrator",
            resource="system.write",
            mode="mutating",
            attributes={"value": "alpha"},
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(
            decision.approval_id,
            context.exception.approval.identifier,
        )
        self.assertEqual(decision.approved_by, "owner")
        policy.stop()
        gate.stop()

    def test_policy_denial_becomes_policy_denied(self) -> None:
        gate = self.make_gate()
        policy = self.make_policy(gate)
        gate.start()
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
                attributes={"value": "alpha"},
            )

        gate.deny(
            context.exception.approval.identifier,
            decided_by="owner",
        )

        with self.assertRaises(PolicyDeniedError):
            policy.authorize(
                action="tool_invocation",
                agent_name="Commander",
                agent_role="orchestrator",
                resource="system.write",
                mode="mutating",
                attributes={"value": "alpha"},
            )

        policy.stop()
        gate.stop()

    def test_mutating_tool_runs_only_after_exact_approval(self) -> None:
        calls = []
        gate = self.make_gate()
        policy = self.make_policy(gate)
        tools = AgentToolRegistry(
            execution_policy=policy,
            allow_mutating=True,
        )
        tools.register(
            "system.write",
            lambda value: calls.append(value) or value,
            parameters={"value": str},
            required=("value",),
            mode=ToolMode.MUTATING,
            allowed_roles=("orchestrator",),
        )
        client = tools.client(
            agent_name="Commander",
            agent_role="orchestrator",
            allowed_tools=("system.write",),
        )

        gate.start()
        policy.start()
        tools.start()

        with self.assertRaises(
            PolicyApprovalRequiredError
        ) as context:
            client.invoke(
                "system.write",
                {"value": "approved-value"},
            )

        self.assertEqual(calls, [])

        gate.approve(
            context.exception.approval.identifier,
            decided_by="owner",
        )
        result = client.invoke(
            "system.write",
            {"value": "approved-value"},
        )

        self.assertEqual(
            result.as_dict()["output"],
            "approved-value",
        )
        self.assertEqual(calls, ["approved-value"])

        tools.stop()
        policy.stop()
        gate.stop()

    def test_service_manager_lifecycle_and_snapshot(self) -> None:
        gate = self.make_gate()
        manager = ServiceManager()

        manager.register("runtime", lambda: None)
        manager.register(
            "approval_gate",
            gate.start,
            stop=gate.stop,
            dependencies=("runtime",),
            health=gate.health,
        )
        manager.start_all()

        self.assertTrue(
            manager.check_health("approval_gate")
        )
        encoded = json.dumps(gate.snapshot())
        self.assertIn('"state": "running"', encoded)
        self.assertIn('"pending_count": 0', encoded)

        manager.stop_all()
        self.assertFalse(gate.health())


if __name__ == "__main__":
    unittest.main()
