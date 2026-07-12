import unittest

from core.eventbus import (
    Event,
    EventBus,
    EventDispatchError,
    Subscription,
)


class EventBusTests(unittest.TestCase):
    def test_subscribe_emit_and_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        subscription = bus.subscribe(
            "mission.created",
            received.append,
        )

        self.assertIsInstance(subscription, Subscription)
        self.assertEqual(bus.listener_count("mission.created"), 1)

        results = bus.emit(
            "mission.created",
            {"mission_id": "MISSION_0001"},
        )

        self.assertEqual(results, [None])
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].name, "mission.created")
        self.assertEqual(
            received[0].payload,
            {"mission_id": "MISSION_0001"},
        )
        self.assertEqual(received[0].sequence, 1)

        self.assertTrue(bus.unsubscribe(subscription))
        self.assertFalse(bus.unsubscribe(subscription))
        self.assertEqual(bus.listener_count("mission.created"), 0)

    def test_listener_order_is_preserved(self) -> None:
        bus = EventBus()
        calls: list[str] = []

        bus.subscribe(
            "ordered",
            lambda event: calls.append("first"),
        )
        bus.subscribe(
            "ordered",
            lambda event: calls.append("second"),
        )

        bus.emit("ordered")

        self.assertEqual(calls, ["first", "second"])

    def test_once_subscription_executes_once(self) -> None:
        bus = EventBus()
        calls: list[int] = []

        bus.subscribe(
            "runtime.ready",
            lambda event: calls.append(event.sequence),
            once=True,
        )

        bus.emit("runtime.ready")
        bus.emit("runtime.ready")

        self.assertEqual(calls, [1])
        self.assertEqual(bus.listener_count("runtime.ready"), 0)

    def test_wildcard_listener_receives_named_events(self) -> None:
        bus = EventBus()
        names: list[str] = []

        bus.subscribe(
            EventBus.WILDCARD,
            lambda event: names.append(event.name),
        )

        bus.emit("agent.online")
        bus.emit("plugin.loaded")

        self.assertEqual(
            names,
            ["agent.online", "plugin.loaded"],
        )

    def test_listener_failures_do_not_stop_dispatch(self) -> None:
        bus = EventBus()
        calls: list[str] = []

        def failing_listener(event: Event) -> None:
            calls.append("failing")
            raise RuntimeError("simulated listener failure")

        def successful_listener(event: Event) -> str:
            calls.append("successful")
            return "complete"

        bus.subscribe("proof.created", failing_listener)
        bus.subscribe("proof.created", successful_listener)

        with self.assertRaises(EventDispatchError) as context:
            bus.emit("proof.created", {"proof_id": "PROOF_0001"})

        self.assertEqual(calls, ["failing", "successful"])
        self.assertEqual(len(context.exception.failures), 1)
        self.assertEqual(
            context.exception.event.name,
            "proof.created",
        )

    def test_failures_can_be_suppressed(self) -> None:
        bus = EventBus()

        def failing_listener(event: Event) -> None:
            raise RuntimeError("expected")

        bus.subscribe("signal", failing_listener)

        results = bus.emit(
            "signal",
            raise_exceptions=False,
        )

        self.assertEqual(results, [])

    def test_clear_removes_listeners(self) -> None:
        bus = EventBus()

        bus.subscribe("event.one", lambda event: None)
        bus.subscribe("event.one", lambda event: None)
        bus.subscribe("event.two", lambda event: None)

        self.assertEqual(bus.clear("event.one"), 2)
        self.assertEqual(bus.listener_count(), 1)
        self.assertEqual(bus.clear(), 1)
        self.assertEqual(bus.listener_count(), 0)

    def test_invalid_event_names_are_rejected(self) -> None:
        bus = EventBus()

        with self.assertRaises(ValueError):
            bus.subscribe("   ", lambda event: None)

        with self.assertRaises(ValueError):
            bus.emit("")

        with self.assertRaises(TypeError):
            bus.subscribe("valid", object())


if __name__ == "__main__":
    unittest.main()
