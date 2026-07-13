"""Keep the GhostFire runtime alive for the local owner workflow."""

from __future__ import annotations

import json
import signal
import threading
from types import ModuleType
from typing import Any


def stop_runtime(runtime_module: ModuleType) -> None:
    """Stop all managed services in reverse dependency order."""

    service_manager = getattr(
        runtime_module,
        "service_manager",
        None,
    )

    if service_manager is not None:
        service_manager.stop_all()


def run() -> int:
    """Start the configured GhostFire runtime and block until shutdown."""

    import main as runtime_module

    websocket_server = getattr(
        runtime_module,
        "websocket_command_server",
        None,
    )
    approval_commands = getattr(
        runtime_module,
        "approval_commands",
        None,
    )

    if websocket_server is None:
        stop_runtime(runtime_module)
        raise RuntimeError(
            "WebSocket command server is not enabled"
        )

    if approval_commands is None or not approval_commands.enabled:
        stop_runtime(runtime_module)
        raise RuntimeError(
            "agent approval command interface is not activated"
        )

    stop_event = threading.Event()

    def request_stop(
        signum: int,
        frame: Any,
    ) -> None:
        _ = signum
        _ = frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    print(
        json.dumps(
            {
                "status": "ready",
                "websocket_url": websocket_server.base_url,
                "approval_commands_enabled": True,
                "secret_exposed": False,
                "stop_instruction": "Press Ctrl+C",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        stop_event.wait()
    finally:
        stop_runtime(runtime_module)

    print(
        json.dumps(
            {
                "status": "stopped",
                "secret_exposed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
