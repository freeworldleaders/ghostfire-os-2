"""Structured logging for GhostFire OS."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4

from core.eventbus import Event, EventBus, Subscription


JsonValue = Any


class JsonLogFormatter(logging.Formatter):
    """Serialize log records as one JSON object per line."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "ghostfire_fields", {})

        payload: dict[str, JsonValue] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
            "fields": fields,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class BoundLogger:
    """Logger view carrying immutable contextual fields."""

    def __init__(
        self,
        parent: GhostFireLogger,
        context: Mapping[str, JsonValue],
    ) -> None:
        self._parent = parent
        self._context = dict(context)

    def bind(self, **fields: JsonValue) -> BoundLogger:
        context = dict(self._context)
        context.update(fields)
        return BoundLogger(self._parent, context)

    def debug(self, message: str, **fields: JsonValue) -> None:
        self._write(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: JsonValue) -> None:
        self._write(logging.INFO, message, fields)

    def warning(self, message: str, **fields: JsonValue) -> None:
        self._write(logging.WARNING, message, fields)

    def error(self, message: str, **fields: JsonValue) -> None:
        self._write(logging.ERROR, message, fields)

    def critical(self, message: str, **fields: JsonValue) -> None:
        self._write(logging.CRITICAL, message, fields)

    def exception(self, message: str, **fields: JsonValue) -> None:
        context = dict(self._context)
        context.update(fields)
        self._parent.exception(message, **context)

    def _write(
        self,
        level: int,
        message: str,
        fields: Mapping[str, JsonValue],
    ) -> None:
        context = dict(self._context)
        context.update(fields)
        self._parent.log(level, message, **context)


class GhostFireLogger:
    """
    Structured JSON logger with file rotation and EventBus integration.

    Each instance owns its handlers, preventing duplicate global logging
    configuration across tests, plugins, and embedded runtimes.
    """

    def __init__(
        self,
        *,
        name: str = "ghostfire",
        level: int | str = logging.INFO,
        log_path: str | Path | None = None,
        stream: TextIO | None = None,
        max_bytes: int = 5_000_000,
        backup_count: int = 3,
        context: Mapping[str, JsonValue] | None = None,
    ) -> None:
        service_name = self._validate_name(name)
        resolved_level = self._resolve_level(level)

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")

        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        if isinstance(backup_count, bool) or not isinstance(backup_count, int):
            raise TypeError("backup_count must be an integer")

        if backup_count < 0:
            raise ValueError("backup_count cannot be negative")

        self._service_name = service_name
        self._context = dict(context or {})
        self._lock = RLock()
        self._handlers: list[logging.Handler] = []
        self._event_bus: EventBus | None = None
        self._event_subscription: Subscription | None = None
        self._closed = False

        internal_name = f"{service_name}.{uuid4().hex}"
        self._logger = logging.getLogger(internal_name)
        self._logger.setLevel(resolved_level)
        self._logger.propagate = False

        formatter = JsonLogFormatter(service_name)

        if log_path is not None:
            resolved_path = Path(log_path).expanduser()
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                resolved_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            self._handlers.append(file_handler)
            self._log_path: Path | None = resolved_path
        else:
            self._log_path = None

        if stream is not None:
            stream_handler = logging.StreamHandler(stream)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)
            self._handlers.append(stream_handler)

        if not self._handlers:
            null_handler = logging.NullHandler()
            self._logger.addHandler(null_handler)
            self._handlers.append(null_handler)

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def level(self) -> int:
        return self._logger.level

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def set_level(self, level: int | str) -> None:
        resolved_level = self._resolve_level(level)

        with self._lock:
            self._ensure_open()
            self._logger.setLevel(resolved_level)

    def bind(self, **fields: JsonValue) -> BoundLogger:
        context = dict(self._context)
        context.update(fields)
        return BoundLogger(self, context)

    def debug(self, message: str, **fields: JsonValue) -> None:
        self.log(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: JsonValue) -> None:
        self.log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: JsonValue) -> None:
        self.log(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: JsonValue) -> None:
        self.log(logging.ERROR, message, **fields)

    def critical(self, message: str, **fields: JsonValue) -> None:
        self.log(logging.CRITICAL, message, **fields)

    def exception(self, message: str, **fields: JsonValue) -> None:
        self.log(
            logging.ERROR,
            message,
            exc_info=True,
            **fields,
        )

    def log(
        self,
        level: int | str,
        message: str,
        *,
        exc_info: bool
        | tuple[
            type[BaseException],
            BaseException,
            TracebackType | None,
        ] = False,
        **fields: JsonValue,
    ) -> None:
        resolved_level = self._resolve_level(level)

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        with self._lock:
            self._ensure_open()
            merged_fields = dict(self._context)
            merged_fields.update(fields)

            self._logger.log(
                resolved_level,
                message,
                extra={"ghostfire_fields": merged_fields},
                exc_info=exc_info,
            )

    def attach_event_bus(self, event_bus: EventBus) -> Subscription:
        if not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus")

        with self._lock:
            self._ensure_open()

            if self._event_subscription is not None:
                raise RuntimeError("an EventBus is already attached")

            subscription = event_bus.subscribe(
                EventBus.WILDCARD,
                self._record_event,
            )

            self._event_bus = event_bus
            self._event_subscription = subscription

        self.info(
            "ghostfire.logging.eventbus_attached",
            subscription_id=subscription.identifier,
        )

        return subscription

    def detach_event_bus(self) -> bool:
        with self._lock:
            event_bus = self._event_bus
            subscription = self._event_subscription
            self._event_bus = None
            self._event_subscription = None

        if event_bus is None or subscription is None:
            return False

        removed = event_bus.unsubscribe(subscription)

        if not self.is_closed:
            self.info(
                "ghostfire.logging.eventbus_detached",
                subscription_id=subscription.identifier,
                removed=removed,
            )

        return removed

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()

            for handler in self._handlers:
                handler.flush()

    def shutdown(self) -> bool:
        with self._lock:
            if self._closed:
                return False

        self.detach_event_bus()

        with self._lock:
            if self._closed:
                return False

            for handler in self._handlers:
                handler.flush()
                handler.close()
                self._logger.removeHandler(handler)

            self._handlers.clear()
            self._closed = True

        return True

    def _record_event(self, event: Event) -> None:
        self.info(
            "ghostfire.eventbus.dispatch",
            event_name=event.name,
            event_sequence=event.sequence,
            event_created_at=event.created_at.isoformat(),
            event_payload=event.payload,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("name cannot be empty")

        return normalized_name

    @staticmethod
    def _resolve_level(level: int | str) -> int:
        if isinstance(level, bool):
            raise TypeError("level must be an integer or string")

        if isinstance(level, int):
            return level

        if isinstance(level, str):
            normalized_level = level.strip().upper()

            if normalized_level in logging._nameToLevel:
                return logging._nameToLevel[normalized_level]

            raise ValueError(f"unknown logging level: {level}")

        raise TypeError("level must be an integer or string")
