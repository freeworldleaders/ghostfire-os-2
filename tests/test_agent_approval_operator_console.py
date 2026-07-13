import argparse
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from agents.approval_operator_console import (
    AgentApprovalOperatorConsole,
    OperatorConsoleDecisionBlocked,
    OperatorConsoleSelectionError,
    SAFETY_ACKNOWLEDGEMENT,
)
from agents.approval_owner import (
    AgentApprovalOwnerWorkflow,
)
from scripts import agent_approval_operator_console
from scripts import run_agent_approval_operator_console


TOKEN = "owner-token-" + ("A" * 48)


class FakeWorkflow(AgentApprovalOwnerWorkflow):
    def __init__(self) -> None:
        super().__init__(
            host="127.0.0.1",
            port=8103,
            path="/v1/commands",
            owner_token=TOKEN,
        )
        self.requests = [
            {
                "identifier": "abc123",
                "status": "pending",
                "agent_name": "Commander",
                "action": "tool_invocation",
                "resource": "ghostfire.test",
                "attribute_names": [
                    "case",
                    "protected_value",
                ],
            },
            {
                "identifier": "abc999",
                "status": "pending",
                "agent_name": "Guardian",
                "action": "tool_invocation",
                "resource": "ghostfire.guard",
                "attribute_names": ["case"],
            },
        ]
        self.operations = []

    def status(self):
        return {
            "id": "status",
            "type": "status",
            "status": "ok",
            "data": {
                "agent_approval_commands": {
                    "enabled": True,
                    "safe_hold": False,
                    "token_configured": True,
                }
            },
        }

    def list(self, *, status="pending"):
        return [
            dict(item)
            for item in self.requests
            if item["status"] == status
        ]

    def get(self, approval_id):
        for item in self.requests:
            if item["identifier"] == approval_id:
                return dict(item)

        raise RuntimeError("not found")

    def _decide(self, action, approval_id, note, confirmation):
        expected = self.confirmation_phrase(
            action,
            approval_id,
        )

        if confirmation != expected:
            raise RuntimeError("wrong confirmation")

        target_status = {
            "approve": "approved",
            "deny": "denied",
            "cancel": "cancelled",
        }[action]

        for item in self.requests:
            if item["identifier"] == approval_id:
                item["status"] = target_status
                self.operations.append(
                    {
                        "action": action,
                        "approval_id": approval_id,
                        "note": note,
                    }
                )
                return dict(item)

        raise RuntimeError("not found")

    def approve(
        self,
        approval_id,
        *,
        note="",
        confirmation,
    ):
        return self._decide(
            "approve",
            approval_id,
            note,
            confirmation,
        )

    def deny(
        self,
        approval_id,
        *,
        note="",
        confirmation,
    ):
        return self._decide(
            "deny",
            approval_id,
            note,
            confirmation,
        )

    def cancel(
        self,
        approval_id,
        *,
        note="",
        confirmation,
    ):
        return self._decide(
            "cancel",
            approval_id,
            note,
            confirmation,
        )


def make_console(
    *,
    decision_mode=False,
    input_text="",
    width=100,
):
    workflow = FakeWorkflow()
    output = io.StringIO()
    console = AgentApprovalOperatorConsole(
        workflow,
        input_stream=io.StringIO(input_text),
        output_stream=output,
        decision_mode=decision_mode,
        width=width,
    )
    console.refresh()
    return workflow, console, output


class ConsoleConstructionTests(unittest.TestCase):
    def test_console_defaults_to_review_only(self):
        _, console, _ = make_console()
        self.assertFalse(console.decision_mode)

    def test_width_below_minimum_is_rejected(self):
        workflow = FakeWorkflow()

        with self.assertRaises(ValueError):
            AgentApprovalOperatorConsole(
                workflow,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                width=71,
            )

    def test_invalid_workflow_type_is_rejected(self):
        with self.assertRaises(TypeError):
            AgentApprovalOperatorConsole(
                object(),
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
            )

    def test_history_limit_must_be_positive(self):
        workflow = FakeWorkflow()

        with self.assertRaises(ValueError):
            AgentApprovalOperatorConsole(
                workflow,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                history_limit=0,
            )


class ConsoleSnapshotTests(unittest.TestCase):
    def test_snapshot_is_secret_free(self):
        _, console, _ = make_console()
        encoded = json.dumps(console.snapshot())

        self.assertNotIn(TOKEN, encoded)
        self.assertFalse(
            console.snapshot()["secret_exposed"]
        )

    def test_snapshot_never_executes_action(self):
        _, console, _ = make_console()
        snapshot = console.snapshot()

        self.assertFalse(snapshot["action_executed"])
        self.assertEqual(
            snapshot["console"]["pending_count"],
            2,
        )

    def test_render_contains_review_only_banner(self):
        _, console, _ = make_console()
        rendered = console.render()

        self.assertIn("REVIEW-ONLY", rendered)
        self.assertIn("NO ACTION IS EXECUTED", rendered)

    def test_render_lines_respect_width(self):
        _, console, _ = make_console(width=80)

        self.assertTrue(
            all(
                len(line) <= 80
                for line in console.render().splitlines()
            )
        )


class ConsoleSelectionTests(unittest.TestCase):
    def test_select_by_row(self):
        _, console, _ = make_console()
        selected = console.select("1")

        self.assertEqual(selected["identifier"], "abc123")
        self.assertEqual(console.selected_id, "abc123")

    def test_select_by_exact_id(self):
        _, console, _ = make_console()
        selected = console.select("abc999")

        self.assertEqual(selected["agent_name"], "Guardian")

    def test_ambiguous_prefix_is_rejected(self):
        _, console, _ = make_console()

        with self.assertRaises(
            OperatorConsoleSelectionError
        ):
            console.select("abc")

    def test_out_of_range_row_is_rejected(self):
        _, console, _ = make_console()

        with self.assertRaises(
            OperatorConsoleSelectionError
        ):
            console.select("9")

    def test_selected_request_returns_redacted_data(self):
        _, console, _ = make_console()
        console.select("1")
        selected = console.selected_request()
        encoded = json.dumps(selected)

        self.assertIn("attribute_names", selected)
        self.assertNotIn("protected-secret", encoded)


class ConsoleDecisionTests(unittest.TestCase):
    def test_review_only_blocks_decision(self):
        _, console, _ = make_console()
        console.select("1")

        with self.assertRaises(
            OperatorConsoleDecisionBlocked
        ):
            console.record_decision(
                "approve",
                safety_acknowledgement=(
                    SAFETY_ACKNOWLEDGEMENT
                ),
                confirmation="APPROVE:abc123",
            )

    def test_wrong_safety_acknowledgement_is_blocked(self):
        _, console, _ = make_console(
            decision_mode=True
        )
        console.select("1")

        with self.assertRaises(
            OperatorConsoleDecisionBlocked
        ):
            console.record_decision(
                "approve",
                safety_acknowledgement="wrong",
                confirmation="APPROVE:abc123",
            )

    def test_wrong_action_confirmation_is_blocked(self):
        _, console, _ = make_console(
            decision_mode=True
        )
        console.select("1")

        with self.assertRaises(
            OperatorConsoleDecisionBlocked
        ):
            console.record_decision(
                "approve",
                safety_acknowledgement=(
                    SAFETY_ACKNOWLEDGEMENT
                ),
                confirmation="APPROVE:wrong",
            )

    def test_approve_records_decision_without_execution(self):
        workflow, console, _ = make_console(
            decision_mode=True
        )
        console.select("1")
        result = console.record_decision(
            "approve",
            safety_acknowledgement=(
                SAFETY_ACKNOWLEDGEMENT
            ),
            confirmation="APPROVE:abc123",
            note="approved by owner",
        )

        self.assertEqual(result["status"], "approved")
        self.assertEqual(len(workflow.operations), 1)
        self.assertEqual(
            console.history()[0].action,
            "approve",
        )

    def test_deny_records_decision(self):
        _, console, _ = make_console(
            decision_mode=True
        )
        console.select("2")
        result = console.record_decision(
            "deny",
            safety_acknowledgement=(
                SAFETY_ACKNOWLEDGEMENT
            ),
            confirmation="DENY:abc999",
        )

        self.assertEqual(result["status"], "denied")

    def test_cancel_records_decision(self):
        _, console, _ = make_console(
            decision_mode=True
        )
        console.select("1")
        result = console.record_decision(
            "cancel",
            safety_acknowledgement=(
                SAFETY_ACKNOWLEDGEMENT
            ),
            confirmation="CANCEL:abc123",
        )

        self.assertEqual(result["status"], "cancelled")

    def test_decision_requires_selection(self):
        _, console, _ = make_console(
            decision_mode=True
        )

        with self.assertRaises(
            OperatorConsoleSelectionError
        ):
            console.record_decision(
                "approve",
                safety_acknowledgement=(
                    SAFETY_ACKNOWLEDGEMENT
                ),
                confirmation="APPROVE:abc123",
            )

    def test_approve_requires_pending_request(self):
        workflow, console, _ = make_console(
            decision_mode=True
        )
        workflow.requests[0]["status"] = "denied"
        console.refresh(status="denied")
        console.select("1")

        with self.assertRaises(
            OperatorConsoleDecisionBlocked
        ):
            console.record_decision(
                "approve",
                safety_acknowledgement=(
                    SAFETY_ACKNOWLEDGEMENT
                ),
                confirmation="APPROVE:abc123",
            )

    def test_history_is_append_only_session_record(self):
        _, console, _ = make_console(
            decision_mode=True
        )
        console.select("1")
        console.record_decision(
            "approve",
            safety_acknowledgement=(
                SAFETY_ACKNOWLEDGEMENT
            ),
            confirmation="APPROVE:abc123",
        )
        records = console.history()

        self.assertIsInstance(records, tuple)
        self.assertEqual(records[0].status, "approved")


class ConsoleCommandTests(unittest.TestCase):
    def test_quit_command_stops_loop(self):
        _, console, _ = make_console()
        self.assertFalse(console.execute_line("quit"))

    def test_list_command_changes_filter(self):
        workflow, console, _ = make_console()
        workflow.requests[0]["status"] = "denied"

        self.assertTrue(console.execute_line("list denied"))
        self.assertEqual(console.list_status, "denied")

    def test_help_command_writes_safety_phrase(self):
        _, console, output = make_console()
        console.execute_line("help")

        self.assertIn(
            SAFETY_ACKNOWLEDGEMENT,
            output.getvalue(),
        )

    def test_eof_closes_console_safely(self):
        workflow = FakeWorkflow()
        output = io.StringIO()
        console = AgentApprovalOperatorConsole(
            workflow,
            input_stream=io.StringIO(""),
            output_stream=output,
        )

        self.assertEqual(console.run(), 0)
        self.assertIn(
            "No action was executed",
            output.getvalue(),
        )

    def test_interactive_decision_requires_two_phrases(self):
        workflow, console, output = make_console(
            decision_mode=True,
            input_text=(
                SAFETY_ACKNOWLEDGEMENT
                + "\nAPPROVE:abc123\nowner note\n"
            ),
        )
        console.select("1")
        console.execute_line("approve")

        self.assertEqual(
            workflow.operations[0]["action"],
            "approve",
        )
        self.assertIn(
            '"action_executed": false',
            output.getvalue(),
        )


class ConsoleCliTests(unittest.TestCase):
    def test_cli_has_no_token_argument(self):
        help_text = (
            agent_approval_operator_console
            .build_parser()
            .format_help()
        )

        self.assertNotIn("--owner-token", help_text)
        self.assertNotIn("--token", help_text)

    def test_cli_snapshot_json_is_read_only(self):
        workflow = FakeWorkflow()
        output = io.StringIO()
        arguments = argparse.Namespace(
            decision_mode=False,
            snapshot_json=True,
            width=100,
            timeout=3.0,
        )

        code = (
            agent_approval_operator_console
            .run_with_workflow(
                arguments,
                workflow,
                input_stream=io.StringIO(),
                output_stream=output,
            )
        )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertFalse(payload["action_executed"])
        self.assertFalse(payload["secret_exposed"])

    def test_cli_error_is_secret_free(self):
        error = io.StringIO()

        with mock.patch(
            "scripts.agent_approval_operator_console."
            "load_settings",
            side_effect=ValueError("bad config"),
        ):
            with mock.patch("sys.stderr", error):
                code = (
                    agent_approval_operator_console
                    .main(["--snapshot-json"])
                )

        payload = json.loads(error.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(payload["secret_exposed"])

    def test_launcher_overrides_configured_port_with_bound_port(
        self,
    ):
        fake_workflow = mock.Mock()
        fake_server = SimpleNamespace(bound_port=49152)
        runtime_module = SimpleNamespace(
            websocket_command_server=fake_server,
            approval_commands=SimpleNamespace(
                enabled=True
            ),
            settings={
                "websocket_command_server": {
                    "port": 8103,
                }
            },
        )

        with mock.patch.dict(
            "sys.modules",
            {"main": runtime_module},
        ), mock.patch(
            "scripts.run_agent_approval_operator_console."
            "AgentApprovalOwnerWorkflow.from_settings",
            return_value=fake_workflow,
        ) as from_settings, mock.patch(
            "scripts.run_agent_approval_operator_console."
            "run_with_workflow",
            return_value=0,
        ), mock.patch(
            "scripts.run_agent_approval_operator_console."
            "stop_runtime",
        ):
            code = (
                run_agent_approval_operator_console
                .main(["--snapshot-json"])
            )

        settings = from_settings.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(
            settings["websocket_command_server"]["port"],
            49152,
        )
        fake_workflow.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
