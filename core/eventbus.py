"""Thread-safe synchronous event dispatch for GhostFire OS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


EventCallback = Callable[["Event"], Any]


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event delivered to EventBus subscribers."""

    name: str
    payload: Any = None
    sequence: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class Subscription:
    """Opaque handle used to remove a registered listener."""

    event_name: str
    identifier: str


@dataclass(frozen=True, slots=True)
class _Listener:
    subscription: Subscription
    callback: EventCallback
    once: bool


class EventDispatchError(RuntimeError):
    """Raised after dispatch when one or more listeners failed."""

    def __init__(
        self,
        event: Event,
        failures: tuple[tuple[Subscription, Exception], ...],
    ) -> None:
        self.event = event
        self.failures = failures

        super().__init__(
            f"{len(failures)} listener(s) failed while dispatching "
            f"{event.name!r}"
        )


class EventBus:
    """
    Thread-safe synchronous event bus.

    Exact-name subscribers and ``*`` wildcard subscribers are supported.
    Listeners execute in subscription order. One failing listener does not
    prevent remaining listeners from receiving the event.
    """

    WILDCARD = "*"

    def __init__(self) -> None:
        self._lock = RLock()
        self._listeners: dict[str, dict[str, _Listener]] = {}
        self._sequence = 0

    def subscribe(
        self,
        event_name: str,
        callback: EventCallback,
        *,
        once: bool = False,
    ) -> Subscription:
        """Register a callback and return its subscription handle."""

        normalized_name = self._validate_event_name(event_name)

        if not callable(callback):
            raise TypeError("callback must be callable")

        subscription = Subscription(
            event_name=normalized_name,
            identifier=uuid4().hex,
        )

        listener = _Listener(
            subscription=subscription,
            callback=callback,
            once=bool(once),
        )

        with self._lock:
            bucket = self._listeners.setdefault(normalized_name, {})
            bucket[subscription.identifier] = listener

        return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription. Returns True when it existed."""

        if not isinstance(subscription, Subscription):
            raise TypeError("subscription must be a Subscription")

        with self._lock:
            return self._remove_listener_locked(subscription)

    def emit(
        self,
        event_name: str,
        payload: Any = None,
        *,
        raise_exceptions: bool = True,
    ) -> list[Any]:
        """Dispatch an event synchronously."""

        normalized_name = self._validate_event_name(event_name)

        with self._lock:
            self._sequence += 1

            event = Event(
                name=normalized_name,
                payload=payload,
                sequence=self._sequence,
            )

            listeners = list(
                self._listeners.get(normalized_name, {}).values()
            )

            if normalized_name != self.WILDCARD:
                listeners.extend(
                    self._listeners.get(self.WILDCARD, {}).values()
                )

            for listener in listeners:
                if listener.once:
                    self._remove_listener_locked(listener.subscription)

        results: list[Any] = []
        failures: list[tuple[Subscription, Exception]] = []

        for listener in listeners:
            try:
                results.append(listener.callback(event))
            except Exception as exc:
                failures.append((listener.subscription, exc))

        if failures and raise_exceptions:
            raise EventDispatchError(event, tuple(failures))

        return results

    def listener_count(self, event_name: str | None = None) -> int:
        """Return the number of active listeners."""

        with self._lock:
            if event_name is None:
                return sum(
                    len(bucket)
                    for bucket in self._listeners.values()
                )

            normalized_name = self._validate_event_name(event_name)
            return len(self._listeners.get(normalized_name, {}))

    def clear(self, event_name: str | None = None) -> int:
        """Remove listeners and return the number removed."""

        with self._lock:
            if event_name is None:
                removed_count = sum(
                    len(bucket)
                    for bucket in self._listeners.values()
                )

                self._listeners.clear()
                return removed_count

            normalized_name = self._validate_event_name(event_name)
            bucket = self._listeners.pop(normalized_name, {})
            return len(bucket)

    def _remove_listener_locked(
        self,
        subscription: Subscription,
    ) -> bool:
        bucket = self._listeners.get(subscription.event_name)

        if not bucket:
            return False

        removed = bucket.pop(subscription.identifier, None)

        if not bucket:
            self._listeners.pop(subscription.event_name, None)

        return removed is not None

    @staticmethod
    def _validate_event_name(event_name: str) -> str:
        if not isinstance(event_name, str):
            raise TypeError("event_name must be a string")

        normalized_name = event_name.strip()

        if not normalized_name:
            raise ValueError("event_name cannot be empty")

        return normalized_name
