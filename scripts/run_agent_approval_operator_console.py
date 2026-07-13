"""Start GhostFire and open the owner approval console."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from agents.approval_owner import (
    AgentApprovalOwnerWorkflow,
)
from scripts.agent_approval_operator_console import (
    build_parser,
    run_with_workflow,
)
from scripts.run_agent_approval_runtime import stop_runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    import main as runtime_module

    websocket_server = runtime_module.websocket_command_server
    approval_commands = runtime_module.approval_commands

    if websocket_server is None:
        stop_runtime(runtime_module)
        raise RuntimeError(
            "WebSocket command server is not enabled"
        )

    if not approval_commands.enabled:
        stop_runtime(runtime_module)
        raise RuntimeError(
            "agent approval command interface is not activated"
        )

    settings = deepcopy(runtime_module.settings)
    settings["websocket_command_server"]["port"] = (
        websocket_server.bound_port
    )
    workflow = AgentApprovalOwnerWorkflow.from_settings(
        settings,
        timeout=arguments.timeout,
    )

    try:
        return run_with_workflow(
            arguments,
            workflow,
        )
    finally:
        workflow.close()
        stop_runtime(runtime_module)


if __name__ == "__main__":
    raise SystemExit(main())
