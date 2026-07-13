import argparse
import io
import json
import unittest
from unittest import mock

from agents.approval_operator_console import AgentApprovalOperatorConsole
from agents.approval_operator_ledger import (
    OwnerOperationLedgerQueryClient,
    OwnerOperationLedgerQueryError,
)
from agents.approval_owner import AgentApprovalOwnerWorkflow
from scripts import agent_approval_operator_console


class FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit):
        return self._raw[:limit]


class FakeWorkflow(AgentApprovalOwnerWorkflow):
    def __init__(self):
        super().__init__(
            host="127.0.0.1",
            port=8103,
            path="/v1/commands",
            owner_token="owner-token-" + ("A" * 48),
        )
        self.requests = [
            {
                "identifier": "approval-1",
                "status": "pending",
                "agent_name": "Commander",
                "action": "tool_invocation",
                "resource": "ghostfire.test",
                "attribute_names": ["case"],
            }
        ]

    def status(self):
        return {"status": "ok"}

    def list(self, *, status="pending"):
        return [
            dict(item)
            for item in self.requests
            if item["status"] == status
        ]

    def get(self, approval_id):
        return dict(self.requests[0])

    def snapshot(self):
        return mock.Mock(
            as_dict=mock.Mock(
                return_value={"status": "ok"}
            )
        )


class FakeLedgerClient(OwnerOperationLedgerQueryClient):
    def __init__(self):
        super().__init__("http://127.0.0.1:8791")
        self.calls = []

    def health(self):
        self.calls.append(("health", None))
        return {
            "status": "ok",
            "read_only": True,
            "loopback_only": True,
            "action_executed": False,
            "secret_exposed": False,
        }

    def list_operations(self):
        self.calls.append(("list", None))
        return {
            "status": "ok",
            "count": 1,
            "items": [{"request_id": "request-1"}],
            "action_executed": False,
            "secret_exposed": False,
        }

    def latest_operation(self):
        self.calls.append(("latest", None))
        return {
            "status": "ok",
            "item": {"request_id": "request-1"},
            "action_executed": False,
            "secret_exposed": False,
        }

    def get_operation(self, request_id):
        self.calls.append(("show", request_id))
        return {
            "status": "ok",
            "item": {"request_id": request_id},
            "action_executed": False,
            "secret_exposed": False,
        }

    def history(self):
        self.calls.append(("history", None))
        return {
            "status": "ok",
            "count": 1,
            "items": [{"request_id": "request-1"}],
            "action_executed": False,
            "secret_exposed": False,
        }

    def verify(self):
        self.calls.append(("verify", None))
        return {
            "status": "verified",
            "ledger_validated": True,
            "history_chain_validated": True,
            "latest_matches_canonical": True,
            "action_executed": False,
            "secret_exposed": False,
        }


class ClientConstructionTests(unittest.TestCase):
    def test_non_loopback_host_is_rejected(self):
        with self.assertRaises(ValueError):
            OwnerOperationLedgerQueryClient("http://192.168.1.5:8791")

    def test_https_is_rejected(self):
        with self.assertRaises(ValueError):
            OwnerOperationLedgerQueryClient("https://127.0.0.1:8791")

    def test_credentials_are_rejected(self):
        with self.assertRaises(ValueError):
            OwnerOperationLedgerQueryClient(
                "http://user:pass@127.0.0.1:8791"
            )

    def test_path_query_and_fragment_are_rejected(self):
        for value in (
            "http://127.0.0.1:8791/v1",
            "http://127.0.0.1:8791?x=1",
            "http://127.0.0.1:8791#x",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OwnerOperationLedgerQueryClient(value)

    def test_timeout_bounds_are_enforced(self):
        with self.assertRaises(ValueError):
            OwnerOperationLedgerQueryClient(timeout=0)


class ClientRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = OwnerOperationLedgerQueryClient(
            "http://127.0.0.1:8791"
        )

    def test_health_uses_get(self):
        payload = {
            "status": "ok",
            "read_only": True,
            "loopback_only": True,
            "action_executed": False,
            "secret_exposed": False,
        }

        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse(payload),
        ) as opener:
            result = self.client.health()

        request = opener.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "http://127.0.0.1:8791/health")
        self.assertEqual(result["status"], "ok")

    def test_read_only_endpoint_paths(self):
        payload = {
            "status": "ok",
            "action_executed": False,
            "secret_exposed": False,
        }

        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse(payload),
        ) as opener:
            self.client.list_operations()
            self.client.latest_operation()
            self.client.get_operation("request-1")
            self.client.history()
            self.client.verify()

        urls = [
            call.args[0].full_url
            for call in opener.call_args_list
        ]
        self.assertEqual(
            urls,
            [
                "http://127.0.0.1:8791/v1/owner-operations",
                "http://127.0.0.1:8791/v1/owner-operations/latest",
                "http://127.0.0.1:8791/v1/owner-operations/request-1",
                "http://127.0.0.1:8791/v1/owner-operations/history",
                "http://127.0.0.1:8791/v1/owner-operations/verify",
            ],
        )

    def test_invalid_request_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.client.get_operation("../secret")

    def test_secret_exposed_true_is_rejected(self):
        payload = {
            "status": "ok",
            "secret_exposed": True,
            "action_executed": False,
        }

        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse(payload),
        ), self.assertRaises(OwnerOperationLedgerQueryError):
            self.client.health()

    def test_action_executed_true_is_rejected(self):
        payload = {
            "status": "ok",
            "secret_exposed": False,
            "action_executed": True,
        }

        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse(payload),
        ), self.assertRaises(OwnerOperationLedgerQueryError):
            self.client.health()

    def test_forbidden_field_is_rejected(self):
        payload = {
            "status": "ok",
            "owner_token": "protected",
            "secret_exposed": False,
            "action_executed": False,
        }

        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse(payload),
        ), self.assertRaises(OwnerOperationLedgerQueryError):
            self.client.health()

    def test_non_object_response_is_rejected(self):
        with mock.patch(
            "agents.approval_operator_ledger.urlopen",
            return_value=FakeResponse([]),
        ), self.assertRaises(OwnerOperationLedgerQueryError):
            self.client.health()


class ConsoleLedgerIntegrationTests(unittest.TestCase):
    def make_console(self, ledger_client=None):
        output = io.StringIO()
        console = AgentApprovalOperatorConsole(
            FakeWorkflow(),
            input_stream=io.StringIO(),
            output_stream=output,
            ledger_client=ledger_client,
        )
        console.refresh()
        return console, output

    def test_snapshot_includes_safe_ledger_state(self):
        console, _ = self.make_console(FakeLedgerClient())
        snapshot = console.snapshot()

        self.assertEqual(
            snapshot["owner_operation_ledger"]["status"],
            "ok",
        )
        self.assertFalse(snapshot["action_executed"])
        self.assertFalse(snapshot["secret_exposed"])

    def test_ledger_commands_are_available(self):
        client = FakeLedgerClient()
        console, output = self.make_console(client)

        for command in (
            "ledger health",
            "ledger list",
            "ledger latest",
            "ledger show request-1",
            "ledger history",
            "ledger verify",
        ):
            self.assertTrue(console.execute_line(command))

        encoded = output.getvalue()
        self.assertIn('"request_id": "request-1"', encoded)
        self.assertIn('"status": "verified"', encoded)
        self.assertEqual(
            [name for name, _ in client.calls],
            ["health", "list", "latest", "show", "history", "verify"],
        )

    def test_ledger_command_is_blocked_when_disabled(self):
        console, output = self.make_console()
        console.execute_line("ledger health")

        self.assertIn(
            "owner-operation ledger API is not enabled",
            output.getvalue(),
        )

    def test_unsupported_ledger_command_is_rejected(self):
        console, output = self.make_console(FakeLedgerClient())
        console.execute_line("ledger approve")

        self.assertIn(
            "unsupported ledger command",
            output.getvalue(),
        )

    def test_help_documents_ledger_commands(self):
        console, _ = self.make_console(FakeLedgerClient())
        self.assertIn("ledger verify", console.help_text())


class ConsoleCliLedgerTests(unittest.TestCase):
    def test_parser_defaults_to_loopback_ledger_api(self):
        arguments = (
            agent_approval_operator_console
            .build_parser()
            .parse_args([])
        )

        self.assertEqual(
            arguments.ledger_api_url,
            "http://127.0.0.1:8791",
        )
        self.assertFalse(arguments.no_ledger_api)

    def test_snapshot_uses_injected_ledger_client(self):
        output = io.StringIO()
        arguments = argparse.Namespace(
            decision_mode=False,
            snapshot_json=True,
            width=100,
            timeout=3.0,
            ledger_api_url="http://127.0.0.1:8791",
            no_ledger_api=False,
        )

        code = agent_approval_operator_console.run_with_workflow(
            arguments,
            FakeWorkflow(),
            input_stream=io.StringIO(),
            output_stream=output,
            ledger_client=FakeLedgerClient(),
        )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(
            payload["owner_operation_ledger"]["status"],
            "ok",
        )
        self.assertFalse(payload["action_executed"])
        self.assertFalse(payload["secret_exposed"])


if __name__ == "__main__":
    unittest.main()
