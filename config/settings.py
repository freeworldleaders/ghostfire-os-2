"""GhostFire OS configuration defaults and runtime loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from config.loader import (
    ConfigurationLoader,
    ConfigurationSnapshot,
)
from core.eventbus import EventBus


DEFAULT_SETTINGS: dict[str, Any] = {
    "app_name": "Ghostfire OS",
    "version": "0.2.0",
    "runtime": "online",
    "logging": {
        "root": None,
        "max_bytes": 5_000_000,
        "backup_count": 3,
    },
    "scheduler": {
        "poll_interval": 0.05,
    },
    "service_manager": {
        "scheduler_stop_timeout": 1.0,
    },
    "ai_agents": {
        "history_limit": 100,
        "memory_limit": 100,
    },
    "terminal_dashboard": {
        "enabled": True,
        "color": False,
        "width": 88,
        "show_health": True,
    },
    "rest_api": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8102,
        "auth_token": None,
        "request_timeout": 2.0,
    },
    "websocket_command_server": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8103,
        "auth_token": None,
        "allowed_commands": ["BOOT", "STATUS"],
        "path": "/v1/commands",
        "max_message_bytes": 65_536,
        "idle_timeout": 30.0,
    },
}

REQUIRED_SETTINGS = {
    "app_name": str,
    "version": str,
    "runtime": str,
    "logging.max_bytes": int,
    "logging.backup_count": int,
    "scheduler.poll_interval": (int, float),
    "service_manager.scheduler_stop_timeout": (int, float),
    "ai_agents.history_limit": int,
    "ai_agents.memory_limit": int,
    "terminal_dashboard.enabled": bool,
    "terminal_dashboard.color": bool,
    "terminal_dashboard.width": int,
    "terminal_dashboard.show_health": bool,
    "rest_api.enabled": bool,
    "rest_api.host": str,
    "rest_api.port": int,
    "rest_api.auth_token": (str, type(None)),
    "rest_api.request_timeout": (int, float),
    "websocket_command_server.enabled": bool,
    "websocket_command_server.host": str,
    "websocket_command_server.port": int,
    "websocket_command_server.auth_token": (str, type(None)),
    "websocket_command_server.allowed_commands": list,
    "websocket_command_server.path": str,
    "websocket_command_server.max_message_bytes": int,
    "websocket_command_server.idle_timeout": (int, float),
}


def load_configuration(
    *,
    event_bus: EventBus | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ConfigurationSnapshot:
    """Load the active GhostFire configuration snapshot."""

    loader = ConfigurationLoader(
        DEFAULT_SETTINGS,
        env_prefix="GHOSTFIRE_CONFIG__",
        event_bus=event_bus,
    )

    config_file = os.environ.get("GHOSTFIRE_CONFIG_FILE")

    return loader.load(
        path=config_file,
        overrides=overrides,
        required=REQUIRED_SETTINGS,
    )


def load_settings(
    *,
    event_bus: EventBus | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load active settings as a mutable dictionary."""

    return load_configuration(
        event_bus=event_bus,
        overrides=overrides,
    ).as_dict()


SETTINGS = deepcopy(DEFAULT_SETTINGS)
