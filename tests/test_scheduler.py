import threading
import unittest

from core.eventbus import EventBus
from core.scheduler import (
    Scheduler,
    SchedulerExecutionError,
    TaskHandle,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SchedulerTests(unittest.TestCase):
    def test_one_time_task_runs_only_when_due(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)
        calls: list[str] = []

        handle = scheduler.schedule_once(
            "mission.open",
            5,
            lambda: calls.append("executed"),
        )

        self.assertIsInstance(handle, TaskHandle)
        self.assertEqual(scheduler.task_count(), 1)
        self.assertEqual(scheduler.run_pending(), [])

        clock.advance(5)

        self.assertEqual(scheduler.run_pending(), [None])
        self.assertEqual(calls, ["executed"])
        self.assertEqual(scheduler.task_count(), 0)
        self.assertEqual(scheduler.run_pending(), [])

    def test_due_tasks_run_in_scheduled_order(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)
        calls: list[str] = []

        scheduler.schedule_once(
            "first",
            2,
            lambda: calls.append("first"),
        )
        scheduler.schedule_once(
            "second",
            2,
            lambda: calls.append("second"),
        )

        clock.advance(2)
        scheduler.run_pending()

        self.assertEqual(calls, ["first", "second"])

    def test_recurring_task_is_rescheduled(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)
        calls: list[float] = []

        scheduler.schedule_every(
            "heartbeat",
            10,
            lambda: calls.append(clock()),
        )

        clock.advance(10)
        scheduler.run_pending()

        self.assertEqual(calls, [10.0])
        self.assertEqual(scheduler.task_count(), 1)

        clock.advance(10)
        scheduler.run_pending()

        self.assertEqual(calls, [10.0, 20.0])

    def test_task_can_be_cancelled(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)

        handle = scheduler.schedule_once(
            "cancelled.task",
            1,
            lambda: self.fail("cancelled task executed"),
        )

        self.assertTrue(scheduler.cancel(handle))
        self.assertFalse(scheduler.cancel(handle))

        clock.advance(1)

        self.assertEqual(scheduler.run_pending(), [])
        self.assertEqual(scheduler.task_count(), 0)

    def test_task_failure_does_not_block_other_tasks(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)
        calls: list[str] = []

        def failing_task() -> None:
            calls.append("failing")
            raise RuntimeError("simulated failure")

        scheduler.schedule_once(
            "failing",
            0,
            failing_task,
        )

        scheduler.schedule_once(
            "successful",
            0,
            lambda: calls.append("successful"),
        )

        with self.assertRaises(
            SchedulerExecutionError
        ) as context:
            scheduler.run_pending()

        self.assertEqual(calls, ["failing", "successful"])
        self.assertEqual(len(context.exception.failures), 1)
        self.assertEqual(scheduler.task_count(), 0)

    def test_scheduler_publishes_event_bus_telemetry(self) -> None:
        clock = FakeClock()
        event_bus = EventBus()
        scheduler = Scheduler(
            clock=clock,
            event_bus=event_bus,
        )

        event_names: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: event_names.append(event.name),
        )

        scheduler.schedule_once(
            "proof.capture",
            0,
            lambda: "complete",
        )

        scheduler.run_pending()

        self.assertIn(
            "ghostfire.scheduler.task_scheduled",
            event_names,
        )
        self.assertIn(
            "ghostfire.scheduler.task_started",
            event_names,
        )
        self.assertIn(
            "ghostfire.scheduler.task_completed",
            event_names,
        )

    def test_clear_and_list_tasks(self) -> None:
        clock = FakeClock()
        scheduler = Scheduler(clock=clock)

        scheduler.schedule_once(
            "later",
            20,
            lambda: None,
        )
        scheduler.schedule_once(
            "sooner",
            10,
            lambda: None,
        )

        tasks = scheduler.list_tasks()

        self.assertEqual(
            [task.name for task in tasks],
            ["sooner", "later"],
        )
        self.assertEqual(scheduler.clear(), 2)
        self.assertEqual(scheduler.task_count(), 0)
        self.assertIsNone(scheduler.seconds_until_next())

    def test_background_worker_executes_and_stops(self) -> None:
        scheduler = Scheduler()
        completed = threading.Event()

        scheduler.schedule_once(
            "background.task",
            0,
            completed.set,
        )

        self.assertTrue(
            scheduler.start(poll_interval=0.01)
        )
        self.assertTrue(completed.wait(1.0))
        self.assertTrue(scheduler.is_running)
        self.assertTrue(scheduler.stop(timeout=1.0))
        self.assertFalse(scheduler.is_running)

    def test_invalid_schedule_arguments_are_rejected(self) -> None:
        scheduler = Scheduler()

        with self.assertRaises(ValueError):
            scheduler.schedule_once(
                "",
                0,
                lambda: None,
            )

        with self.assertRaises(ValueError):
            scheduler.schedule_once(
                "invalid.delay",
                -1,
                lambda: None,
            )

        with self.assertRaises(ValueError):
            scheduler.schedule_every(
                "invalid.interval",
                0,
                lambda: None,
            )

        with self.assertRaises(TypeError):
            scheduler.schedule_once(
                "invalid.callback",
                0,
                object(),
            )


if __name__ == "__main__":
    unittest.main()
