import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

from agents.approval import AgentApprovalGate
from agents.approval_commands import (
    AgentApprovalCommandInterface,
)
from agents.approval_owner import (
    AgentApprovalOwnerWorkflow,
    ApprovalOwnerConfigurationError,
    ApprovalOwnerConfirmationError,
    ApprovalOwnerResponseError,
)
from api.websocket import WebSocketCommandServer
from scripts import agent_approval_owner
from scripts.run_agent_approval_runtime import stop_runtime


OWNER_TOKEN = "owner-token-" + ("A" * 48)


class OwnerWorkflowIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = AgentApprovalGate()
        self.interface = AgentApprovalCommandInterface(
            self.gate,
            enabled=True,
            owner_token=OWNER_TOKEN,
        )
        self.server = WebSocketCommandServer(
            command_handler=lambda command: command,
            status_provider=lambda: {
                "agent_approval_commands": (
                    self.interface.snapshot()
                )
            },
            approval_handler=self.interface.handle,
            host="127.0.0.1",
            port=0,
            allowed_commands=("STATUS",),
            idle_timeout=2,
        )
        self.gate.start()
        self.interface.start()
        self.server.start()
        self.workflow = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=self.server.bound_port,
            path="/v1/commands",
            owner_token=OWNER_TOKEN,
            timeout=2,
        )

    def tearDown(self) -> None:
        self.workflow.close()
        self.server.stop()
        self.interface.stop()
        self.gate.stop()

    def test_status_reports_enabled_interface(self) -> None:
        response = self.workflow.status()
        snapshot = response["data"][
            "agent_approval_commands"
        ]

        self.assertTrue(snapshot["enabled"])
        self.assertFalse(snapshot["safe_hold"])

    def test_list_pending_returns_array(self) -> None:
        self.assertEqual(
            self.workflow.list(status="pending"),
            [],
        )

    def test_wrong_token_is_rejected(self) -> None:
        wrong = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=self.server.bound_port,
            path="/v1/commands",
            owner_token="wrong-token-" + ("B" * 48),
            timeout=2,
        )

        try:
            with self.assertRaises(
                ApprovalOwnerResponseError
            ) as caught:
                wrong.list()
        finally:
            wrong.close()

        self.assertEqual(
            caught.exception.code,
            "approval_unauthorized",
        )

    def test_snapshot_never_exposes_token(self) -> None:
        encoded = json.dumps(
            self.workflow.snapshot().as_dict()
        )

        self.assertNotIn(OWNER_TOKEN, encoded)
        self.assertFalse(
            self.workflow.snapshot().as_dict()[
                "secret_exposed"
            ]
        )

    def test_transport_bearer_authentication(self) -> None:
        protected_server = WebSocketCommandServer(
            command_handler=lambda command: command,
            status_provider=lambda: {"ok": True},
            approval_handler=self.interface.handle,
            host="127.0.0.1",
            port=0,
            auth_token="transport-secret-value",
            allowed_commands=("STATUS",),
            idle_timeout=2,
        )
        protected_server.start()
        workflow = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=protected_server.bound_port,
            path="/v1/commands",
            owner_token=OWNER_TOKEN,
            transport_token="transport-secret-value",
            timeout=2,
        )

        try:
            self.assertEqual(
                workflow.status()["status"],
                "ok",
            )
        finally:
            workflow.close()
            protected_server.stop()


class OwnerWorkflowValidationTests(unittest.TestCase):
    def test_non_loopback_host_is_rejected(self) -> None:
        with self.assertRaises(
            ApprovalOwnerConfigurationError
        ):
            AgentApprovalOwnerWorkflow(
                host="192.168.1.20",
                port=8103,
                path="/v1/commands",
                owner_token=OWNER_TOKEN,
            )

    def test_invalid_port_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgentApprovalOwnerWorkflow(
                host="127.0.0.1",
                port=0,
                path="/v1/commands",
                owner_token=OWNER_TOKEN,
            )

    def test_confirmation_phrase_is_deterministic(self) -> None:
        self.assertEqual(
            AgentApprovalOwnerWorkflow.confirmation_phrase(
                "approve",
                "approval-123",
            ),
            "APPROVE:approval-123",
        )

    def test_non_mutating_confirmation_phrase_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ApprovalOwnerConfirmationError
        ):
            AgentApprovalOwnerWorkflow.confirmation_phrase(
                "get",
                "approval-123",
            )

    def test_wrong_confirmation_blocks_before_network(self) -> None:
        workflow = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=65534,
            path="/v1/commands",
            owner_token=OWNER_TOKEN,
            timeout=0.05,
        )

        try:
            with self.assertRaises(
                ApprovalOwnerConfirmationError
            ):
                workflow.approve(
                    "approval-123",
                    confirmation="APPROVE:wrong",
                )
        finally:
            workflow.close()

    def test_close_clears_token_configuration(self) -> None:
        workflow = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=8103,
            path="/v1/commands",
            owner_token=OWNER_TOKEN,
        )
        workflow.close()

        self.assertFalse(
            workflow.snapshot().token_configured
        )

    def test_invalid_status_is_rejected(self) -> None:
        workflow = AgentApprovalOwnerWorkflow(
            host="127.0.0.1",
            port=8103,
            path="/v1/commands",
            owner_token=OWNER_TOKEN,
        )

        try:
            with self.assertRaises(ValueError):
                workflow.list(status="unknown")
        finally:
            workflow.close()

    def test_from_settings_requires_activation(self) -> None:
        settings = {
            "agent_approval_commands": {
                "enabled": False,
                "owner_token": OWNER_TOKEN,
                "owner_token_file": None,
            },
            "websocket_command_server": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8103,
                "path": "/v1/commands",
                "auth_token": None,
                "max_message_bytes": 65_536,
            },
        }

        with self.assertRaises(
            ApprovalOwnerConfigurationError
        ):
            AgentApprovalOwnerWorkflow.from_settings(
                settings
            )

    def test_from_settings_requires_websocket_server(self) -> None:
        settings = {
            "agent_approval_commands": {
                "enabled": True,
                "owner_token": OWNER_TOKEN,
                "owner_token_file": None,
            },
            "websocket_command_server": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 8103,
                "path": "/v1/commands",
                "auth_token": None,
                "max_message_bytes": 65_536,
            },
        }

        with self.assertRaises(
            ApprovalOwnerConfigurationError
        ):
            AgentApprovalOwnerWorkflow.from_settings(
                settings
            )


class OwnerWorkflowCliTests(unittest.TestCase):
    def test_phrase_command_does_not_load_configuration(
        self,
    ) -> None:
        output = io.StringIO()

        with mock.patch(
            "scripts.agent_approval_owner.load_settings"
        ) as load_settings:
            with redirect_stdout(output):
                code = agent_approval_owner.main(
                    [
                        "phrase",
                        "deny",
                        "approval-9",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertFalse(load_settings.called)
        self.assertEqual(
            json.loads(output.getvalue())["confirmation"],
            "DENY:approval-9",
        )

    def test_parser_requires_exact_confirmation_option(
        self,
    ) -> None:
        parser = agent_approval_owner.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["approve", "approval-1"]
            )

    def test_cli_has_no_owner_token_argument(self) -> None:
        parser = agent_approval_owner.build_parser()
        help_text = parser.format_help()

        self.assertNotIn("--owner-token", help_text)
        self.assertNotIn("--token", help_text)

    def test_cli_error_output_marks_secret_not_exposed(
        self,
    ) -> None:
        error = io.StringIO()

        with mock.patch(
            "scripts.agent_approval_owner.load_settings",
            side_effect=ApprovalOwnerConfigurationError(
                "not activated"
            ),
        ):
            with redirect_stderr(error):
                code = agent_approval_owner.main(
                    ["status"]
                )

        payload = json.loads(error.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(payload["secret_exposed"])

    def test_cli_success_closes_workflow(self) -> None:
        fake = mock.Mock()
        fake.status.return_value = {
            "type": "status",
            "status": "ok",
        }
        fake.snapshot.return_value.as_dict.return_value = {
            "secret_exposed": False,
        }
        output = io.StringIO()

        with mock.patch(
            "scripts.agent_approval_owner.load_settings",
            return_value={},
        ), mock.patch(
            "scripts.agent_approval_owner."
            "AgentApprovalOwnerWorkflow.from_settings",
            return_value=fake,
        ):
            with redirect_stdout(output):
                code = agent_approval_owner.main(
                    ["status"]
                )

        self.assertEqual(code, 0)
        fake.close.assert_called_once()
        self.assertFalse(
            json.loads(output.getvalue())[
                "secret_exposed"
            ]
        )


class RuntimeHostTests(unittest.TestCase):
    def test_stop_runtime_calls_service_manager(self) -> None:
        manager = mock.Mock()
        runtime_module = SimpleNamespace(
            service_manager=manager
        )

        stop_runtime(runtime_module)

        manager.stop_all.assert_called_once_with()

    def test_stop_runtime_tolerates_missing_manager(self) -> None:
        stop_runtime(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
