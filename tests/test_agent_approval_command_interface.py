import concurrent.futures
import json
import unittest

from agents.approval import (
    AgentApprovalGate,
    ApprovalStatus,
)
from agents.approval_commands import (
    AgentApprovalCommandInterface,
    ApprovalCommandAuthenticationError,
    ApprovalCommandDisabledError,
    ApprovalCommandState,
    ApprovalCommandStateError,
    ApprovalCommandValidationError,
)
from api.websocket import WebSocketCommandServer
from core.eventbus import EventBus
from core.service_manager import ServiceManager
from tests.test_websocket_command_server import RawWebSocketClient


class AgentApprovalCommandInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates: list[AgentApprovalGate] = []
        self.interfaces: list[
            AgentApprovalCommandInterface
        ] = []
        self.servers: list[WebSocketCommandServer] = []
        self.clients: list[RawWebSocketClient] = []

    def tearDown(self) -> None:
        for client in reversed(self.clients):
            client.close()

        for server in reversed(self.servers):
            server.stop()

        for interface in reversed(self.interfaces):
            interface.stop()

        for gate in reversed(self.gates):
            gate.stop()

    def make_gate(self) -> AgentApprovalGate:
        gate = AgentApprovalGate(
            owner_identity="owner",
        )
        self.gates.append(gate)
        return gate

    def make_interface(
        self,
        gate: AgentApprovalGate,
        *,
        enabled: bool = True,
        owner_token: str | None = "owner-secret",
        history_limit: int = 100,
        max_note_length: int = 500,
        event_bus: EventBus | None = None,
    ) -> AgentApprovalCommandInterface:
        interface = AgentApprovalCommandInterface(
            gate,
            enabled=enabled,
            owner_token=owner_token,
            history_limit=history_limit,
            max_note_length=max_note_length,
            event_bus=event_bus,
        )
        self.interfaces.append(interface)
        return interface

    def create_pending(
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

    @staticmethod
    def message(
        action: str,
        **values,
    ) -> dict:
        return {
            "type": "approval",
            "action": action,
            "token": "owner-secret",
            **values,
        }

    def start_pair(
        self,
        gate: AgentApprovalGate,
        interface: AgentApprovalCommandInterface,
    ) -> None:
        gate.start()
        interface.start()

    def test_enabled_interface_requires_owner_token(self) -> None:
        gate = self.make_gate()

        with self.assertRaises(ValueError):
            AgentApprovalCommandInterface(
                gate,
                enabled=True,
                owner_token=None,
            )

    def test_lifecycle_is_idempotent(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)

        self.assertTrue(interface.start())
        self.assertFalse(interface.start())
        self.assertTrue(interface.health())
        self.assertTrue(interface.stop())
        self.assertFalse(interface.stop())
        self.assertEqual(
            interface.state,
            ApprovalCommandState.STOPPED,
        )

    def test_execute_requires_running_interface(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)

        with self.assertRaises(ApprovalCommandStateError):
            interface.execute(self.message("list"))

    def test_disabled_interface_fails_closed(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(
            gate,
            enabled=False,
            owner_token=None,
        )
        self.start_pair(gate, interface)

        with self.assertRaises(
            ApprovalCommandDisabledError
        ):
            interface.execute(self.message("list"))

        self.assertTrue(interface.snapshot()["safe_hold"])

    def test_wrong_token_is_rejected(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        message = self.message("list")
        message["token"] = "wrong"

        with self.assertRaises(
            ApprovalCommandAuthenticationError
        ):
            interface.execute(message)

        self.assertEqual(
            interface.snapshot()[
                "authentication_failure_count"
            ],
            1,
        )

    def test_snapshot_never_exposes_token(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)

        encoded = json.dumps(interface.snapshot())

        self.assertNotIn("owner-secret", encoded)
        self.assertTrue(
            interface.snapshot()["token_configured"]
        )

    def test_list_returns_redacted_pending_requests(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        response = interface.execute(
            self.message("list", status="pending")
        )

        self.assertEqual(response["status"], "ok")
        self.assertEqual(
            response["data"][0]["identifier"],
            pending.identifier,
        )
        self.assertNotIn(
            "alpha",
            json.dumps(response),
        )

    def test_get_returns_one_request(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        response = interface.execute(
            self.message(
                "get",
                approval_id=pending.identifier,
            )
        )

        self.assertEqual(
            response["data"]["status"],
            "pending",
        )

    def test_approve_updates_request(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        response = interface.execute(
            self.message(
                "approve",
                approval_id=pending.identifier,
                note="approved locally",
            )
        )

        self.assertEqual(
            response["data"]["status"],
            "approved",
        )
        self.assertEqual(
            gate.get(pending.identifier).decided_by,
            "owner",
        )

    def test_deny_updates_request(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        response = interface.execute(
            self.message(
                "deny",
                approval_id=pending.identifier,
            )
        )

        self.assertEqual(
            response["data"]["status"],
            "denied",
        )

    def test_cancel_clears_request(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        response = interface.execute(
            self.message(
                "cancel",
                approval_id=pending.identifier,
            )
        )

        self.assertEqual(
            response["data"]["status"],
            "cancelled",
        )

    def test_unknown_action_is_rejected(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)

        with self.assertRaises(
            ApprovalCommandValidationError
        ):
            interface.execute(
                self.message("destroy")
            )

    def test_unknown_fields_are_rejected(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)

        with self.assertRaises(
            ApprovalCommandValidationError
        ):
            interface.execute(
                self.message(
                    "list",
                    unexpected=True,
                )
            )

    def test_note_length_is_bounded(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(
            gate,
            max_note_length=4,
        )
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        with self.assertRaises(
            ApprovalCommandValidationError
        ):
            interface.execute(
                self.message(
                    "approve",
                    approval_id=pending.identifier,
                    note="too long",
                )
            )

    def test_history_is_bounded_and_redacted(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(
            gate,
            history_limit=2,
        )
        self.start_pair(gate, interface)

        for _ in range(3):
            interface.execute(self.message("list"))

        encoded = json.dumps(
            [
                record.as_dict()
                for record in interface.history()
            ]
        )

        self.assertEqual(len(interface.history()), 2)
        self.assertNotIn("owner-secret", encoded)

    def test_telemetry_redacts_token_and_note(self) -> None:
        event_bus = EventBus()
        events = []
        gate = self.make_gate()
        interface = self.make_interface(
            gate,
            event_bus=event_bus,
        )
        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event),
        )
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)

        interface.execute(
            self.message(
                "approve",
                approval_id=pending.identifier,
                note="DO_NOT_LOG_NOTE",
            )
        )

        encoded = json.dumps(
            [event.payload for event in events],
            default=str,
        )

        self.assertNotIn("owner-secret", encoded)
        self.assertNotIn("DO_NOT_LOG_NOTE", encoded)

    def test_concurrent_list_commands_are_thread_safe(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        self.create_pending(gate)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=6
        ) as executor:
            results = list(
                executor.map(
                    lambda _: interface.execute(
                        self.message("list")
                    )["status"],
                    range(12),
                )
            )

        self.assertEqual(results, ["ok"] * 12)
        self.assertEqual(interface.command_count, 12)

    def test_handle_maps_auth_failure_to_safe_error(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        message = self.message("list")
        message["token"] = "wrong"

        response = interface.handle(message)

        self.assertEqual(response["status"], "error")
        self.assertEqual(
            response["code"],
            "approval_unauthorized",
        )
        self.assertNotIn("wrong", json.dumps(response))

    def test_websocket_executes_approval_command(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        pending = self.create_pending(gate)
        server = WebSocketCommandServer(
            command_handler=lambda command: command,
            status_provider=lambda: {},
            approval_handler=interface.handle,
            host="127.0.0.1",
            port=0,
            allowed_commands=("BOOT",),
            idle_timeout=2,
        )
        self.servers.append(server)
        server.start()
        client = RawWebSocketClient(
            "127.0.0.1",
            server.bound_port,
        )
        self.clients.append(client)

        client.send_json(
            {
                "id": "approval-1",
                **self.message(
                    "approve",
                    approval_id=pending.identifier,
                ),
            }
        )
        response = client.receive_json()

        self.assertEqual(
            response["type"],
            "approval_result",
        )
        self.assertEqual(
            response["data"]["status"],
            "approved",
        )
        self.assertEqual(
            server.approval_command_count,
            1,
        )

    def test_websocket_rejects_bad_owner_token(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        self.start_pair(gate, interface)
        server = WebSocketCommandServer(
            command_handler=lambda command: command,
            status_provider=lambda: {},
            approval_handler=interface.handle,
            host="127.0.0.1",
            port=0,
            allowed_commands=("BOOT",),
            idle_timeout=2,
        )
        self.servers.append(server)
        server.start()
        client = RawWebSocketClient(
            "127.0.0.1",
            server.bound_port,
        )
        self.clients.append(client)

        client.send_json(
            {
                "id": "approval-2",
                "type": "approval",
                "action": "list",
                "token": "wrong",
            }
        )
        response = client.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(
            response["error"],
            "approval_unauthorized",
        )
        self.assertEqual(
            server.approval_command_count,
            0,
        )

    def test_service_manager_lifecycle(self) -> None:
        gate = self.make_gate()
        interface = self.make_interface(gate)
        manager = ServiceManager()

        manager.register("runtime", lambda: None)
        manager.register(
            "approval_gate",
            gate.start,
            stop=gate.stop,
            dependencies=("runtime",),
            health=gate.health,
        )
        manager.register(
            "approval_commands",
            interface.start,
            stop=interface.stop,
            dependencies=("approval_gate",),
            health=interface.health,
        )
        manager.start_all()

        self.assertTrue(
            manager.check_health("approval_commands")
        )
        manager.stop_all()
        self.assertFalse(interface.health())


if __name__ == "__main__":
    unittest.main()
