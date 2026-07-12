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
    "terminal_dashboard": {
        "enabled": True,
        "color": False,
        "width": 88,
        "show_health": True,
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
    "terminal_dashboard.enabled": bool,
    "terminal_dashboard.color": bool,
    "terminal_dashboard.width": int,
    "terminal_dashboard.show_health": bool,
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
