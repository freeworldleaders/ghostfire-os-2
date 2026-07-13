"""CLI entrypoint for the GhostFire owner approval console."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from agents.approval_operator_console import (
    AgentApprovalOperatorConsole,
)
from agents.approval_owner import (
    AgentApprovalOwnerWorkflow,
    ApprovalOwnerError,
)
from config.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open the local GhostFire owner approval console. "
            "The console defaults to review-only mode."
        )
    )
    parser.add_argument(
        "--decision-mode",
        action="store_true",
        help=(
            "Permit owner decisions after two exact confirmations. "
            "No requested action is executed by the console."
        ),
    )
    parser.add_argument(
        "--snapshot-json",
        action="store_true",
        help=(
            "Print one secret-free read-only console snapshot "
            "and exit."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Console width from 72 through 180.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Local WebSocket request timeout in seconds.",
    )
    return parser


def run_with_workflow(
    arguments: argparse.Namespace,
    workflow: AgentApprovalOwnerWorkflow,
    *,
    input_stream=None,
    output_stream=None,
) -> int:
    """Run one console session with an existing workflow."""

    input_stream = (
        sys.stdin
        if input_stream is None
        else input_stream
    )
    output_stream = (
        sys.stdout
        if output_stream is None
        else output_stream
    )
    console = AgentApprovalOperatorConsole(
        workflow,
        input_stream=input_stream,
        output_stream=output_stream,
        decision_mode=arguments.decision_mode,
        width=arguments.width,
    )

    if arguments.snapshot_json:
        payload = console.snapshot()
        output_stream.write(
            json.dumps(payload, sort_keys=True) + "\n"
        )
        output_stream.flush()
        return 0

    return console.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        settings = load_settings()
        workflow = AgentApprovalOwnerWorkflow.from_settings(
            settings,
            timeout=arguments.timeout,
        )
    except (ApprovalOwnerError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "secret_exposed": False,
                    "action_executed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        return run_with_workflow(
            arguments,
            workflow,
        )
    finally:
        workflow.close()


if __name__ == "__main__":
    raise SystemExit(main())
