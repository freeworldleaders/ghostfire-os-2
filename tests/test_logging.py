import io
import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path

from core.eventbus import EventBus
from core.logging import GhostFireLogger


class LoggingSubsystemTests(unittest.TestCase):
    def make_file_logger(
        self,
        directory: str,
        **kwargs: object,
    ) -> tuple[GhostFireLogger, Path]:
        log_path = Path(directory) / "ghostfire.jsonl"

        logger = GhostFireLogger(
            name="ghostfire.test",
            log_path=log_path,
            **kwargs,
        )

        return logger, log_path

    def read_records(self, log_path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_file_logging_writes_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger, log_path = self.make_file_logger(directory)

            try:
                logger.info(
                    "mission.completed",
                    mission_id="MISSION_001",
                    outcome="PASS",
                )
                logger.flush()

                records = self.read_records(log_path)

                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["level"], "INFO")
                self.assertEqual(
                    records[0]["message"],
                    "mission.completed",
                )
                self.assertEqual(
                    records[0]["fields"]["mission_id"],
                    "MISSION_001",
                )
                self.assertIn("timestamp", records[0])
            finally:
                logger.shutdown()

    def test_bound_context_is_merged_with_call_fields(self) -> None:
        stream = io.StringIO()
        logger = GhostFireLogger(
            name="ghostfire.context",
            stream=stream,
            context={"realm": "KINGDOM"},
        )
        self.addCleanup(logger.shutdown)

        mission_logger = logger.bind(
            mission_id="MISSION_002",
        )

        mission_logger.info(
            "mission.started",
            operator="GhostFire",
        )
        logger.flush()

        record = json.loads(stream.getvalue().strip())

        self.assertEqual(record["fields"]["realm"], "KINGDOM")
        self.assertEqual(
            record["fields"]["mission_id"],
            "MISSION_002",
        )
        self.assertEqual(
            record["fields"]["operator"],
            "GhostFire",
        )

    def test_level_filtering_blocks_lower_levels(self) -> None:
        stream = io.StringIO()
        logger = GhostFireLogger(
            name="ghostfire.levels",
            level="WARNING",
            stream=stream,
        )
        self.addCleanup(logger.shutdown)

        logger.info("ignored")
        logger.warning("captured")
        logger.flush()

        records = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if line.strip()
        ]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["message"], "captured")

    def test_set_level_activates_debug_logging(self) -> None:
        stream = io.StringIO()
        logger = GhostFireLogger(
            name="ghostfire.dynamic-level",
            level="INFO",
            stream=stream,
        )
        self.addCleanup(logger.shutdown)

        logger.set_level("DEBUG")
        logger.debug("debug.active")
        logger.flush()

        record = json.loads(stream.getvalue().strip())

        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(record["level"], "DEBUG")

    def test_exception_logging_includes_traceback(self) -> None:
        stream = io.StringIO()
        logger = GhostFireLogger(
            name="ghostfire.exception",
            stream=stream,
        )
        self.addCleanup(logger.shutdown)

        try:
            raise RuntimeError("simulated failure")
        except RuntimeError:
            logger.exception(
                "mission.failed",
                mission_id="MISSION_003",
            )

        logger.flush()
        record = json.loads(stream.getvalue().strip())

        self.assertEqual(record["level"], "ERROR")
        self.assertIn("RuntimeError", record["exception"])
        self.assertIn(
            "simulated failure",
            record["exception"],
        )

    def test_event_bus_bridge_captures_events(self) -> None:
        stream = io.StringIO()
        event_bus = EventBus()
        logger = GhostFireLogger(
            name="ghostfire.eventbus",
            stream=stream,
        )
        self.addCleanup(logger.shutdown)

        logger.attach_event_bus(event_bus)

        event_bus.emit(
            "ghostfire.proof.created",
            {"proof_id": "PROOF_001"},
        )

        logger.flush()

        records = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if line.strip()
        ]

        event_records = [
            record
            for record in records
            if record["message"] == "ghostfire.eventbus.dispatch"
        ]

        self.assertEqual(len(event_records), 1)
        self.assertEqual(
            event_records[0]["fields"]["event_name"],
            "ghostfire.proof.created",
        )
        self.assertEqual(
            event_records[0]["fields"]["event_payload"]["proof_id"],
            "PROOF_001",
        )

    def test_event_bus_bridge_detaches_cleanly(self) -> None:
        stream = io.StringIO()
        event_bus = EventBus()
        logger = GhostFireLogger(
            name="ghostfire.detach",
            stream=stream,
        )
        self.addCleanup(logger.shutdown)

        logger.attach_event_bus(event_bus)
        self.assertTrue(logger.detach_event_bus())

        stream.seek(0)
        stream.truncate(0)

        event_bus.emit("ghostfire.after.detach")
        logger.flush()

        self.assertEqual(stream.getvalue(), "")

    def test_file_rotation_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger, log_path = self.make_file_logger(
                directory,
                max_bytes=250,
                backup_count=2,
            )

            try:
                for index in range(30):
                    logger.info(
                        "rotation.record",
                        index=index,
                        payload="X" * 80,
                    )

                logger.flush()

                self.assertTrue(log_path.exists())
                self.assertTrue(
                    Path(f"{log_path}.1").exists()
                )
            finally:
                logger.shutdown()

    def test_concurrent_logging_preserves_all_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger, log_path = self.make_file_logger(
                directory,
                max_bytes=1_000_000,
            )

            try:
                def worker(worker_id: int) -> None:
                    for sequence in range(20):
                        logger.info(
                            "concurrent.record",
                            worker_id=worker_id,
                            sequence=sequence,
                        )

                threads = [
                    threading.Thread(
                        target=worker,
                        args=(worker_id,),
                    )
                    for worker_id in range(4)
                ]

                for thread in threads:
                    thread.start()

                for thread in threads:
                    thread.join()

                logger.flush()

                records = self.read_records(log_path)

                self.assertEqual(len(records), 80)
                self.assertEqual(
                    {
                        record["fields"]["worker_id"]
                        for record in records
                    },
                    {0, 1, 2, 3},
                )
            finally:
                logger.shutdown()

    def test_validation_and_shutdown_are_safe(self) -> None:
        with self.assertRaises(ValueError):
            GhostFireLogger(name="")

        with self.assertRaises(ValueError):
            GhostFireLogger(
                name="ghostfire.invalid",
                max_bytes=0,
            )

        with self.assertRaises(ValueError):
            GhostFireLogger(
                name="ghostfire.invalid",
                backup_count=-1,
            )

        logger = GhostFireLogger(
            name="ghostfire.shutdown",
            stream=io.StringIO(),
        )

        self.assertTrue(logger.shutdown())
        self.assertFalse(logger.shutdown())
        self.assertTrue(logger.is_closed)

        with self.assertRaises(RuntimeError):
            logger.info("closed")


if __name__ == "__main__":
    unittest.main()
