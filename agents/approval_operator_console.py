"""Terminal operator console for GhostFire owner approvals."""

from __future__ import annotations

import hmac
import json
import shlex
import textwrap
from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TextIO

from agents.approval_owner import AgentApprovalOwnerWorkflow


SAFETY_ACKNOWLEDGEMENT = (
    "Record decision only. Do not execute action."
)


class OperatorConsoleError(RuntimeError):
    """Base class for operator-console failures."""


class OperatorConsoleSelectionError(OperatorConsoleError):
    """Raised when a request selection is missing or ambiguous."""


class OperatorConsoleDecisionBlocked(OperatorConsoleError):
    """Raised when a decision fails console safety gates."""


@dataclass(frozen=True, slots=True)
class OperatorDecisionRecord:
    """Secret-free, append-only record of a console decision."""

    action: str
    approval_id: str
    status: str
    recorded_at: datetime

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision record."""

        return {
            "action": self.action,
            "approval_id": self.approval_id,
            "status": self.status,
            "recorded_at": self.recorded_at.isoformat(),
        }


class AgentApprovalOperatorConsole:
    """
    Local review console for owner approval decisions.

    The console defaults to review-only mode. Decision mode must be explicitly
    enabled, and every mutation requires both the fixed safety acknowledgement
    and the workflow's exact ACTION:APPROVAL_ID confirmation phrase.
    """

    _ACTIONS = frozenset({"approve", "deny", "cancel"})
    _STATUSES = (
        "pending",
        "approved",
        "denied",
        "consumed",
        "cancelled",
        "expired",
    )

    def __init__(
        self,
        workflow: AgentApprovalOwnerWorkflow,
        *,
        input_stream: TextIO,
        output_stream: TextIO,
        decision_mode: bool = False,
        width: int = 100,
        history_limit: int = 50,
    ) -> None:
        if not isinstance(
            workflow,
            AgentApprovalOwnerWorkflow,
        ):
            raise TypeError(
                "workflow must be an AgentApprovalOwnerWorkflow"
            )

        if not hasattr(input_stream, "readline"):
            raise TypeError("input_stream must provide readline")

        if not hasattr(output_stream, "write"):
            raise TypeError("output_stream must provide write")

        if not isinstance(decision_mode, bool):
            raise TypeError("decision_mode must be a boolean")

        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 72
            or width > 180
        ):
            raise ValueError(
                "width must be an integer between 72 and 180"
            )

        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError(
                "history_limit must be a positive integer"
            )

        self._workflow = workflow
        self._input = input_stream
        self._output = output_stream
        self._decision_mode = decision_mode
        self._width = width
        self._history: deque[OperatorDecisionRecord] = deque(
            maxlen=history_limit
        )
        self._requests: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._list_status = "pending"
        self._running = False

    @property
    def decision_mode(self) -> bool:
        return self._decision_mode

    @property
    def selected_id(self) -> str | None:
        return self._selected_id

    @property
    def list_status(self) -> str:
        return self._list_status

    def history(self) -> tuple[OperatorDecisionRecord, ...]:
        """Return append-only session decision history."""

        return tuple(self._history)

    def refresh(
        self,
        *,
        status: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Refresh the displayed request list."""

        if status is not None:
            normalized = status.strip().lower()

            if normalized not in self._STATUSES:
                raise ValueError(
                    "status must be pending, approved, denied, "
                    "consumed, cancelled, or expired"
                )

            self._list_status = normalized

        requests = self._workflow.list(
            status=self._list_status
        )
        self._requests = [
            deepcopy(dict(item))
            for item in requests
            if isinstance(item, Mapping)
        ]

        if (
            self._selected_id is not None
            and not any(
                request.get("identifier")
                == self._selected_id
                for request in self._requests
            )
        ):
            self._selected_id = None

        return tuple(
            deepcopy(request)
            for request in self._requests
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a secret-free console snapshot."""

        status_response = self._workflow.status()
        pending = self._workflow.list(status="pending")
        workflow_snapshot = (
            self._workflow.snapshot().as_dict()
        )

        return {
            "status": "ok",
            "console": {
                "mode": (
                    "decision"
                    if self._decision_mode
                    else "review-only"
                ),
                "safety_acknowledgement": (
                    SAFETY_ACKNOWLEDGEMENT
                ),
                "selected_id": self._selected_id,
                "pending_count": len(pending),
                "session_decision_count": len(self._history),
            },
            "runtime": deepcopy(status_response),
            "workflow": workflow_snapshot,
            "secret_exposed": False,
            "action_executed": False,
        }

    def select(self, selector: str) -> dict[str, Any]:
        """Select a request by one-based row or exact/unique ID prefix."""

        normalized = selector.strip()

        if not normalized:
            raise OperatorConsoleSelectionError(
                "selection cannot be empty"
            )

        selected: dict[str, Any] | None = None

        if normalized.isdigit():
            index = int(normalized)

            if not 1 <= index <= len(self._requests):
                raise OperatorConsoleSelectionError(
                    "selection row is out of range"
                )

            selected = self._requests[index - 1]
        else:
            matches = [
                request
                for request in self._requests
                if isinstance(
                    request.get("identifier"),
                    str,
                )
                and request["identifier"].startswith(
                    normalized
                )
            ]

            if not matches:
                raise OperatorConsoleSelectionError(
                    "approval request was not found"
                )

            if len(matches) > 1:
                raise OperatorConsoleSelectionError(
                    "approval request prefix is ambiguous"
                )

            selected = matches[0]

        identifier = selected.get("identifier")

        if not isinstance(identifier, str) or not identifier:
            raise OperatorConsoleSelectionError(
                "selected request has no valid identifier"
            )

        self._selected_id = identifier
        return deepcopy(selected)

    def selected_request(self) -> dict[str, Any]:
        """Fetch the currently selected redacted request."""

        if self._selected_id is None:
            raise OperatorConsoleSelectionError(
                "no approval request is selected"
            )

        return self._workflow.get(self._selected_id)

    def record_decision(
        self,
        action: str,
        *,
        safety_acknowledgement: str,
        confirmation: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Record one owner decision without executing the action."""

        normalized_action = action.strip().lower()

        if normalized_action not in self._ACTIONS:
            raise ValueError(
                "action must be approve, deny, or cancel"
            )

        if not self._decision_mode:
            raise OperatorConsoleDecisionBlocked(
                "console is in review-only mode"
            )

        if self._selected_id is None:
            raise OperatorConsoleSelectionError(
                "no approval request is selected"
            )

        if (
            not isinstance(safety_acknowledgement, str)
            or not hmac.compare_digest(
                safety_acknowledgement,
                SAFETY_ACKNOWLEDGEMENT,
            )
        ):
            raise OperatorConsoleDecisionBlocked(
                "exact safety acknowledgement is required"
            )

        expected_confirmation = (
            self._workflow.confirmation_phrase(
                normalized_action,
                self._selected_id,
            )
        )

        if (
            not isinstance(confirmation, str)
            or not hmac.compare_digest(
                confirmation,
                expected_confirmation,
            )
        ):
            raise OperatorConsoleDecisionBlocked(
                "exact action confirmation is required"
            )

        request = self._workflow.get(self._selected_id)
        request_status = request.get("status")

        if normalized_action in {"approve", "deny"}:
            if request_status != "pending":
                raise OperatorConsoleDecisionBlocked(
                    "approve and deny require a pending request"
                )
        elif request_status not in {
            "pending",
            "approved",
            "denied",
        }:
            raise OperatorConsoleDecisionBlocked(
                "cancel requires a pending, approved, or denied request"
            )

        operation = getattr(
            self._workflow,
            normalized_action,
        )
        result = operation(
            self._selected_id,
            note=note,
            confirmation=confirmation,
        )
        result_status = result.get("status")

        if not isinstance(result_status, str):
            raise OperatorConsoleError(
                "decision response did not include status"
            )

        self._history.append(
            OperatorDecisionRecord(
                action=normalized_action,
                approval_id=self._selected_id,
                status=result_status,
                recorded_at=datetime.now(timezone.utc),
            )
        )

        self.refresh(status=self._list_status)
        return deepcopy(result)

    def render(self) -> str:
        """Render the current fixed-width terminal view."""

        mode = (
            "DECISION MODE"
            if self._decision_mode
            else "REVIEW-ONLY"
        )
        lines = [
            self._line("="),
            self._fit("GHOSTFIRE OWNER APPROVAL CONSOLE"),
            self._fit(
                f"MODE: {mode} | FILTER: {self._list_status.upper()}"
            ),
            self._fit(
                "DECISIONS RECORD APPROVAL STATE ONLY; "
                "NO ACTION IS EXECUTED."
            ),
            self._line("-"),
        ]

        if not self._requests:
            lines.append(
                self._fit("No requests in the current filter.")
            )
        else:
            header = (
                " #  STATUS      AGENT          ACTION          "
                "RESOURCE"
            )
            lines.append(self._fit(header))
            lines.append(self._line("-"))

            for index, request in enumerate(
                self._requests,
                start=1,
            ):
                identifier = str(
                    request.get("identifier", "")
                )
                marker = (
                    "*"
                    if identifier == self._selected_id
                    else " "
                )
                row = (
                    f"{marker}{index:>2} "
                    f"{str(request.get('status', '')):<11}"
                    f"{str(request.get('agent_name', '')):<15}"
                    f"{str(request.get('action', '')):<16}"
                    f"{str(request.get('resource', ''))}"
                )
                lines.append(self._fit(row))

        lines.extend(
            [
                self._line("-"),
                self._fit(
                    "COMMANDS: refresh | list <status> | "
                    "select <row|id> | view"
                ),
                self._fit(
                    "          approve | deny | cancel | "
                    "history | help | quit"
                ),
                self._line("="),
            ]
        )
        return "\n".join(lines)

    def execute_line(self, line: str) -> bool:
        """Execute one console command; return False to stop."""

        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self._write_error(str(exc))
            return True

        if not parts:
            return True

        command = parts[0].lower()
        arguments = parts[1:]

        try:
            if command in {"quit", "exit", "q"}:
                return False

            if command in {"refresh", "r"}:
                self.refresh()
                self._write("Request list refreshed.")
                return True

            if command == "list":
                status = (
                    arguments[0]
                    if arguments
                    else self._list_status
                )
                self.refresh(status=status)
                self._write(
                    f"Loaded {len(self._requests)} "
                    f"{self._list_status} request(s)."
                )
                return True

            if command in {"select", "s"}:
                if len(arguments) != 1:
                    raise OperatorConsoleSelectionError(
                        "usage: select <row|approval-id>"
                    )

                selected = self.select(arguments[0])
                self._write(
                    "Selected "
                    + str(selected["identifier"])
                )
                return True

            if command == "view":
                self._write_json(self.selected_request())
                return True

            if command in self._ACTIONS:
                self._interactive_decision(command)
                return True

            if command == "history":
                self._write_json(
                    [
                        record.as_dict()
                        for record in self._history
                    ]
                )
                return True

            if command in {"help", "?"}:
                self._write(self.help_text())
                return True

            self._write_error(
                f"unsupported console command: {command}"
            )
            return True
        except (
            OperatorConsoleError,
            TypeError,
            ValueError,
        ) as exc:
            self._write_error(str(exc))
            return True

    def run(self) -> int:
        """Run the interactive console until quit, EOF, or Ctrl+C."""

        self._running = True

        try:
            self.refresh()

            while self._running:
                self._write(self.render())
                self._write_raw("ghostfire-owner> ")
                line = self._input.readline()

                if line == "":
                    break

                if not self.execute_line(line):
                    break
        except KeyboardInterrupt:
            self._write("")
            self._write("Operator console interrupted safely.")
        finally:
            self._running = False

        self._write(
            "Operator console closed. No action was executed."
        )
        return 0

    def help_text(self) -> str:
        """Return operator command help."""

        return textwrap.dedent(
            f"""
            refresh
              Refresh the current request filter.

            list <status>
              Load one of: {", ".join(self._STATUSES)}.

            select <row|approval-id>
              Select one displayed request.

            view
              Show the selected redacted request.

            approve | deny | cancel
              Record an owner decision. Decision mode, the exact
              safety acknowledgement, and ACTION:APPROVAL_ID are
              all required.

            history
              Show secret-free decisions from this console session.

            quit
              Close the console without executing any action.

            Safety acknowledgement:
              {SAFETY_ACKNOWLEDGEMENT}
            """
        ).strip()

    def _interactive_decision(self, action: str) -> None:
        if not self._decision_mode:
            raise OperatorConsoleDecisionBlocked(
                "console is in review-only mode; restart with "
                "--decision-mode to record owner decisions"
            )

        if self._selected_id is None:
            raise OperatorConsoleSelectionError(
                "select a request before recording a decision"
            )

        expected = self._workflow.confirmation_phrase(
            action,
            self._selected_id,
        )

        self._write(
            "This records a decision only. "
            "It does not execute the requested action."
        )
        acknowledgement = self._prompt(
            "Type the safety acknowledgement exactly: "
        )
        confirmation = self._prompt(
            f"Type {expected} exactly: "
        )
        note = self._prompt(
            "Decision note (optional): "
        )
        result = self.record_decision(
            action,
            safety_acknowledgement=acknowledgement,
            confirmation=confirmation,
            note=note,
        )
        self._write_json(
            {
                "status": "recorded",
                "action": action,
                "approval_id": result.get("identifier"),
                "approval_status": result.get("status"),
                "action_executed": False,
                "secret_exposed": False,
            }
        )

    def _prompt(self, prompt: str) -> str:
        self._write_raw(prompt)
        value = self._input.readline()

        if value == "":
            raise OperatorConsoleDecisionBlocked(
                "decision input ended before confirmation"
            )

        return value.rstrip("\r\n")

    def _line(self, character: str) -> str:
        return character * self._width

    def _fit(self, value: str) -> str:
        if len(value) <= self._width:
            return value

        return textwrap.shorten(
            value,
            width=self._width,
            placeholder="...",
        )

    def _write_json(self, value: Any) -> None:
        self._write(
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                default=str,
            )
        )

    def _write_error(self, message: str) -> None:
        self._write(f"ERROR: {message}")

    def _write(self, message: str) -> None:
        self._output.write(message + "\n")
        self._output.flush()

    def _write_raw(self, message: str) -> None:
        self._output.write(message)
        self._output.flush()
