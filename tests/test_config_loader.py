import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.loader import (
    ConfigurationFileError,
    ConfigurationLoader,
    ConfigurationNotLoadedError,
    ConfigurationValidationError,
)
from core.eventbus import EventBus


class ConfigurationLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.defaults = {
            "app_name": "Ghostfire OS",
            "version": "0.2.0",
            "runtime": "online",
            "scheduler": {
                "poll_interval": 0.05,
                "enabled": True,
            },
        }

    def test_defaults_create_immutable_snapshot(self) -> None:
        loader = ConfigurationLoader(self.defaults)
        snapshot = loader.load()

        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.get("app_name"), "Ghostfire OS")

        mutable = snapshot.as_dict()
        mutable["app_name"] = "Changed"

        self.assertEqual(snapshot.get("app_name"), "Ghostfire OS")

        with self.assertRaises(TypeError):
            snapshot.values["app_name"] = "Changed"

    def test_json_file_deep_merges_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghostfire.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "0.3.0",
                        "scheduler": {
                            "poll_interval": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            snapshot = ConfigurationLoader(
                self.defaults
            ).load(path=path)

            self.assertEqual(snapshot.get("version"), "0.3.0")
            self.assertEqual(
                snapshot.get("scheduler.poll_interval"),
                0.2,
            )
            self.assertTrue(
                snapshot.get("scheduler.enabled")
            )

    def test_toml_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghostfire.toml"
            path.write_text(
                (
                    'app_name = "Kingdom Runtime"\n'
                    '[scheduler]\n'
                    'enabled = false\n'
                ),
                encoding="utf-8",
            )

            snapshot = ConfigurationLoader(
                self.defaults
            ).load(path=path)

            self.assertEqual(
                snapshot.get("app_name"),
                "Kingdom Runtime",
            )
            self.assertFalse(
                snapshot.get("scheduler.enabled")
            )

    def test_environment_values_are_typed_and_nested(self) -> None:
        environment = {
            "GHOSTFIRE_CONFIG__APP_NAME": "Kingdom OS",
            "GHOSTFIRE_CONFIG__SCHEDULER__POLL_INTERVAL": "0.25",
            "GHOSTFIRE_CONFIG__SCHEDULER__ENABLED": "false",
            "GHOSTFIRE_CONFIG__AGENTS": '["Commander", "Guardian"]',
        }

        with patch.dict(os.environ, environment, clear=False):
            snapshot = ConfigurationLoader(
                self.defaults
            ).load()

        self.assertEqual(snapshot.get("app_name"), "Kingdom OS")
        self.assertEqual(
            snapshot.get("scheduler.poll_interval"),
            0.25,
        )
        self.assertFalse(snapshot.get("scheduler.enabled"))
        self.assertEqual(
            snapshot.get("agents"),
            ("Commander", "Guardian"),
        )

    def test_explicit_overrides_have_highest_precedence(self) -> None:
        environment = {
            "GHOSTFIRE_CONFIG__VERSION": "0.4.0",
        }

        with patch.dict(os.environ, environment, clear=False):
            snapshot = ConfigurationLoader(
                self.defaults
            ).load(
                overrides={"version": "9.9.9"}
            )

        self.assertEqual(snapshot.get("version"), "9.9.9")
        self.assertEqual(
            snapshot.sources,
            ("defaults", "environment", "overrides"),
        )

    def test_required_schema_detects_missing_values(self) -> None:
        loader = ConfigurationLoader(self.defaults)

        with self.assertRaises(
            ConfigurationValidationError
        ):
            loader.load(required={"database.url": str})

    def test_required_schema_detects_wrong_types(self) -> None:
        loader = ConfigurationLoader(self.defaults)

        with self.assertRaises(
            ConfigurationValidationError
        ):
            loader.load(required={"version": int})

    def test_get_require_and_default_behavior(self) -> None:
        snapshot = ConfigurationLoader(
            self.defaults
        ).load()

        self.assertEqual(
            snapshot.require("scheduler.poll_interval"),
            0.05,
        )
        self.assertEqual(
            snapshot.get("missing.path", "fallback"),
            "fallback",
        )

        with self.assertRaises(
            ConfigurationValidationError
        ):
            snapshot.require("missing.path")

    def test_reload_increments_revision(self) -> None:
        loader = ConfigurationLoader(self.defaults)

        first = loader.load(
            overrides={"version": "1.0.0"}
        )
        second = loader.reload()

        self.assertEqual(first.revision, 1)
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.get("version"), "1.0.0")
        self.assertIs(loader.current, second)

    def test_current_requires_initial_load(self) -> None:
        loader = ConfigurationLoader(self.defaults)

        with self.assertRaises(
            ConfigurationNotLoadedError
        ):
            _ = loader.current

    def test_event_bus_receives_load_and_failure_events(self) -> None:
        event_bus = EventBus()
        events: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        loader = ConfigurationLoader(
            self.defaults,
            event_bus=event_bus,
        )

        loader.load()

        with self.assertRaises(ConfigurationFileError):
            loader.load(path="missing.json")

        self.assertIn(
            "ghostfire.configuration.loaded",
            events,
        )
        self.assertIn(
            "ghostfire.configuration.failed",
            events,
        )

    def test_redaction_masks_nested_sensitive_values(self) -> None:
        snapshot = ConfigurationLoader(
            {
                "database": {
                    "password": "secret-value",
                    "user": "ghostfire",
                },
                "api_key": "key-value",
            }
        ).load()

        redacted = snapshot.redacted()

        self.assertEqual(
            redacted["database"]["password"],
            "***REDACTED***",
        )
        self.assertEqual(
            redacted["api_key"],
            "***REDACTED***",
        )
        self.assertEqual(
            redacted["database"]["user"],
            "ghostfire",
        )

    def test_unsupported_file_extension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghostfire.yaml"
            path.write_text("runtime: online", encoding="utf-8")

            with self.assertRaises(ConfigurationFileError):
                ConfigurationLoader(
                    self.defaults
                ).load(path=path)

    def test_configuration_file_root_must_be_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghostfire.json"
            path.write_text(
                '["not", "an", "object"]',
                encoding="utf-8",
            )

            with self.assertRaises(ConfigurationFileError):
                ConfigurationLoader(
                    self.defaults
                ).load(path=path)


if __name__ == "__main__":
    unittest.main()
