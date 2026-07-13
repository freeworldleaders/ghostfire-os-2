"""Command-line owner workflow for GhostFire approval decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from agents.approval_owner import (
    AgentApprovalOwnerWorkflow,
    ApprovalOwnerError,
    ApprovalOwnerResponseError,
)
from config.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operate the local GhostFire approval command interface "
            "without exposing the DPAPI-protected owner token."
        )
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Local WebSocket request timeout in seconds.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "status",
        help="Read active GhostFire runtime status.",
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List approval requests.",
    )
    list_parser.add_argument(
        "--status",
        default="pending",
        choices=(
            "pending",
            "approved",
            "denied",
            "consumed",
            "cancelled",
            "expired",
        ),
    )

    get_parser = subparsers.add_parser(
        "get",
        help="Inspect one approval request.",
    )
    get_parser.add_argument("approval_id")

    phrase_parser = subparsers.add_parser(
        "phrase",
        help="Print the exact confirmation phrase for a mutation.",
    )
    phrase_parser.add_argument(
        "action",
        choices=("approve", "deny", "cancel"),
    )
    phrase_parser.add_argument("approval_id")

    for action in ("approve", "deny", "cancel"):
        action_parser = subparsers.add_parser(
            action,
            help=f"{action.title()} one approval request.",
        )
        action_parser.add_argument("approval_id")
        action_parser.add_argument(
            "--note",
            default="",
        )
        action_parser.add_argument(
            "--confirm",
            required=True,
            help=(
                f"Exact phrase {action.upper()}:APPROVAL_ID"
            ),
        )

    return parser


def execute(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Execute one parsed owner workflow command."""

    if arguments.command == "phrase":
        return {
            "status": "ok",
            "command": "phrase",
            "confirmation": (
                AgentApprovalOwnerWorkflow.confirmation_phrase(
                    arguments.action,
                    arguments.approval_id,
                )
            ),
            "secret_exposed": False,
        }

    settings = load_settings()
    workflow = AgentApprovalOwnerWorkflow.from_settings(
        settings,
        timeout=arguments.timeout,
    )

    try:
        if arguments.command == "status":
            data: Any = workflow.status()
        elif arguments.command == "list":
            data = workflow.list(status=arguments.status)
        elif arguments.command == "get":
            data = workflow.get(arguments.approval_id)
        elif arguments.command == "approve":
            data = workflow.approve(
                arguments.approval_id,
                note=arguments.note,
                confirmation=arguments.confirm,
            )
        elif arguments.command == "deny":
            data = workflow.deny(
                arguments.approval_id,
                note=arguments.note,
                confirmation=arguments.confirm,
            )
        elif arguments.command == "cancel":
            data = workflow.cancel(
                arguments.approval_id,
                note=arguments.note,
                confirmation=arguments.confirm,
            )
        else:
            raise RuntimeError("unsupported owner workflow command")

        return {
            "status": "ok",
            "command": arguments.command,
            "data": data,
            "workflow": workflow.snapshot().as_dict(),
            "secret_exposed": False,
        }
    finally:
        workflow.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        payload = execute(arguments)
    except ApprovalOwnerResponseError as exc:
        payload = {
            "status": "error",
            "error": exc.code,
            "message": exc.public_message,
            "secret_exposed": False,
        }
        print(
            json.dumps(payload, sort_keys=True),
            file=sys.stderr,
        )
        return 3
    except ApprovalOwnerError as exc:
        payload = {
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
            "secret_exposed": False,
        }
        print(
            json.dumps(payload, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except (TypeError, ValueError) as exc:
        payload = {
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
            "secret_exposed": False,
        }
        print(
            json.dumps(payload, sort_keys=True),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
