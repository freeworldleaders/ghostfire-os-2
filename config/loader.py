"""Layered configuration loading for GhostFire OS."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

from core.eventbus import EventBus


ConfigValue = Any
RequiredSchema = Mapping[str, type | tuple[type, ...]]


class ConfigurationError(RuntimeError):
    """Base class for configuration failures."""


class ConfigurationFileError(ConfigurationError):
    """Raised when a configuration file cannot be loaded."""


class ConfigurationValidationError(ConfigurationError):
    """Raised when required configuration is missing or invalid."""


class ConfigurationNotLoadedError(ConfigurationError):
    """Raised when current configuration is requested before loading."""


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """Immutable configuration state produced by a loader."""

    revision: int
    values: Mapping[str, ConfigValue]
    sources: tuple[str, ...]
    loaded_at: datetime

    def get(
        self,
        path: str,
        default: ConfigValue = None,
    ) -> ConfigValue:
        """Read a dotted path, returning a default when absent."""

        try:
            return _read_path(self.values, path)
        except KeyError:
            return default

    def require(self, path: str) -> ConfigValue:
        """Read a dotted path and fail when absent."""

        try:
            return _read_path(self.values, path)
        except KeyError as exc:
            raise ConfigurationValidationError(
                f"required configuration is missing: {path}"
            ) from exc

    def as_dict(self) -> dict[str, ConfigValue]:
        """Return a mutable deep copy of this snapshot."""

        return _thaw(self.values)

    def redacted(
        self,
        *,
        sensitive_keys: tuple[str, ...] = (
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
        ),
        replacement: str = "***REDACTED***",
    ) -> dict[str, ConfigValue]:
        """Return a mutable copy with sensitive keys masked."""

        normalized_keys = {
            key.strip().lower()
            for key in sensitive_keys
            if key.strip()
        }

        return _redact_mapping(
            self.as_dict(),
            normalized_keys,
            replacement,
        )


class ConfigurationLoader:
    """
    Load configuration using deterministic precedence.

    Precedence is defaults < file < environment < explicit overrides.
    Environment variables use a configurable prefix and ``__`` for nesting.
    """

    def __init__(
        self,
        defaults: Mapping[str, ConfigValue],
        *,
        env_prefix: str = "GHOSTFIRE_CONFIG__",
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(defaults, Mapping):
            raise TypeError("defaults must be a mapping")

        if not isinstance(env_prefix, str):
            raise TypeError("env_prefix must be a string")

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        self._defaults = deepcopy(dict(defaults))
        self._env_prefix = env_prefix
        self._event_bus = event_bus
        self._lock = RLock()
        self._revision = 0
        self._current: ConfigurationSnapshot | None = None
        self._last_path: Path | None = None
        self._last_overrides: dict[str, ConfigValue] | None = None
        self._last_required: dict[
            str,
            type | tuple[type, ...],
        ] | None = None

    @property
    def current(self) -> ConfigurationSnapshot:
        """Return the most recently loaded snapshot."""

        with self._lock:
            if self._current is None:
                raise ConfigurationNotLoadedError(
                    "configuration has not been loaded"
                )

            return self._current

    def load(
        self,
        *,
        path: str | Path | None = None,
        overrides: Mapping[str, ConfigValue] | None = None,
        required: RequiredSchema | None = None,
    ) -> ConfigurationSnapshot:
        """Load and validate a new configuration snapshot."""

        resolved_path = (
            Path(path).expanduser()
            if path is not None
            else None
        )

        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping or None")

        if required is not None and not isinstance(required, Mapping):
            raise TypeError("required must be a mapping or None")

        try:
            merged = deepcopy(self._defaults)
            sources = ["defaults"]

            if resolved_path is not None:
                file_values = self._load_file(resolved_path)
                _deep_merge(merged, file_values)
                sources.append(str(resolved_path))

            environment_values = self._load_environment()

            if environment_values:
                _deep_merge(merged, environment_values)
                sources.append("environment")

            if overrides:
                _deep_merge(merged, dict(overrides))
                sources.append("overrides")

            self._validate_required(merged, required or {})

            with self._lock:
                self._revision += 1

                snapshot = ConfigurationSnapshot(
                    revision=self._revision,
                    values=_freeze(merged),
                    sources=tuple(sources),
                    loaded_at=datetime.now(timezone.utc),
                )

                self._current = snapshot
                self._last_path = resolved_path
                self._last_overrides = (
                    deepcopy(dict(overrides))
                    if overrides is not None
                    else None
                )
                self._last_required = (
                    dict(required)
                    if required is not None
                    else None
                )

            self._publish(
                "ghostfire.configuration.loaded",
                {
                    "revision": snapshot.revision,
                    "sources": list(snapshot.sources),
                },
            )

            return snapshot
        except Exception as exc:
            self._publish(
                "ghostfire.configuration.failed",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def reload(self) -> ConfigurationSnapshot:
        """Reload using the most recent load arguments."""

        with self._lock:
            path = self._last_path
            overrides = deepcopy(self._last_overrides)
            required = (
                dict(self._last_required)
                if self._last_required is not None
                else None
            )

        return self.load(
            path=path,
            overrides=overrides,
            required=required,
        )

    def _load_file(
        self,
        path: Path,
    ) -> dict[str, ConfigValue]:
        if not path.is_file():
            raise ConfigurationFileError(
                f"configuration file not found: {path}"
            )

        try:
            with path.open("rb") as handle:
                if path.suffix.lower() == ".toml":
                    loaded = tomllib.load(handle)
                elif path.suffix.lower() == ".json":
                    loaded = json.load(handle)
                else:
                    raise ConfigurationFileError(
                        "configuration file must use .toml or .json"
                    )
        except ConfigurationFileError:
            raise
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationFileError(
                f"failed to load configuration file {path}: {exc}"
            ) from exc

        if not isinstance(loaded, Mapping):
            raise ConfigurationFileError(
                "configuration file root must be an object/table"
            )

        return deepcopy(dict(loaded))

    def _load_environment(self) -> dict[str, ConfigValue]:
        values: dict[str, ConfigValue] = {}

        if not self._env_prefix:
            return values

        for variable_name, raw_value in os.environ.items():
            if not variable_name.startswith(self._env_prefix):
                continue

            suffix = variable_name[len(self._env_prefix):]

            if not suffix:
                continue

            path = tuple(
                part.strip().lower()
                for part in suffix.split("__")
                if part.strip()
            )

            if not path:
                continue

            _assign_path(
                values,
                path,
                _parse_environment_value(raw_value),
            )

        return values

    @staticmethod
    def _validate_required(
        values: Mapping[str, ConfigValue],
        required: RequiredSchema,
    ) -> None:
        for path, expected_type in required.items():
            if not isinstance(path, str) or not path.strip():
                raise TypeError(
                    "required configuration paths must be non-empty strings"
                )

            if not (
                isinstance(expected_type, type)
                or (
                    isinstance(expected_type, tuple)
                    and expected_type
                    and all(
                        isinstance(item, type)
                        for item in expected_type
                    )
                )
            ):
                raise TypeError(
                    f"invalid required type declaration for {path}"
                )

            try:
                value = _read_path(values, path)
            except KeyError as exc:
                raise ConfigurationValidationError(
                    f"required configuration is missing: {path}"
                ) from exc

            expected_types = (
                expected_type
                if isinstance(expected_type, tuple)
                else (expected_type,)
            )

            if (
                isinstance(value, bool)
                and bool not in expected_types
                and any(
                    item in {int, float, complex}
                    for item in expected_types
                )
            ):
                valid = False
            else:
                valid = isinstance(value, expected_type)

            if not valid:
                expected_name = _type_name(expected_type)
                raise ConfigurationValidationError(
                    f"configuration {path!r} must be {expected_name}; "
                    f"found {type(value).__name__}"
                )

    def _publish(
        self,
        event_name: str,
        payload: dict[str, ConfigValue],
    ) -> None:
        if self._event_bus is None:
            return

        self._event_bus.emit(
            event_name,
            payload,
            raise_exceptions=False,
        )


def _deep_merge(
    target: dict[str, ConfigValue],
    incoming: Mapping[str, ConfigValue],
) -> None:
    for key, value in incoming.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _assign_path(
    target: dict[str, ConfigValue],
    path: tuple[str, ...],
    value: ConfigValue,
) -> None:
    current = target

    for part in path[:-1]:
        existing = current.get(part)

        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing

        current = existing

    current[path[-1]] = value


def _read_path(
    values: Mapping[str, ConfigValue],
    path: str,
) -> ConfigValue:
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    parts = tuple(
        part.strip()
        for part in path.split(".")
        if part.strip()
    )

    if not parts:
        raise ValueError("path cannot be empty")

    current: ConfigValue = values

    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)

        current = current[part]

    return current


def _parse_environment_value(raw_value: str) -> ConfigValue:
    normalized = raw_value.strip()
    lowered = normalized.lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    if lowered in {"null", "none"}:
        return None

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return raw_value


def _freeze(value: ConfigValue) -> ConfigValue:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)

    return deepcopy(value)


def _thaw(value: ConfigValue) -> ConfigValue:
    if isinstance(value, Mapping):
        return {
            key: _thaw(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [_thaw(item) for item in value]

    return deepcopy(value)


def _redact_mapping(
    values: Mapping[str, ConfigValue],
    sensitive_keys: set[str],
    replacement: str,
) -> dict[str, ConfigValue]:
    redacted: dict[str, ConfigValue] = {}

    for key, value in values.items():
        if key.lower() in sensitive_keys:
            redacted[key] = replacement
        elif isinstance(value, Mapping):
            redacted[key] = _redact_mapping(
                value,
                sensitive_keys,
                replacement,
            )
        elif isinstance(value, list):
            redacted[key] = [
                _redact_mapping(
                    item,
                    sensitive_keys,
                    replacement,
                )
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        else:
            redacted[key] = value

    return redacted


def _type_name(
    expected_type: type | tuple[type, ...],
) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(item.__name__ for item in expected_type)

    return expected_type.__name__
