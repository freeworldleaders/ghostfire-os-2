import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.approval_commands import (
    AgentApprovalCommandInterface,
)
from agents.approval import AgentApprovalGate
from agents.approval_tokens import (
    AgentApprovalTokenStore,
    ApprovalTokenConfigurationError,
    ApprovalTokenExistsError,
    ApprovalTokenIntegrityError,
    ApprovalTokenStoreError,
    default_approval_token_path,
    resolve_approval_owner_token,
    token_fingerprint,
)
from core.eventbus import EventBus


class ReversibleProtector:
    scheme = "test-reversible-v1"

    def protect(self, plaintext: bytes) -> bytes:
        return b"TEST:" + plaintext[::-1]

    def unprotect(self, ciphertext: bytes) -> bytes:
        if not ciphertext.startswith(b"TEST:"):
            raise ValueError("invalid test ciphertext")
        return ciphertext[5:][::-1]


class AgentApprovalTokenProvisioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "owner-token.json"
        self.protector = ReversibleProtector()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_store(
        self,
        *,
        event_bus: EventBus | None = None,
    ) -> AgentApprovalTokenStore:
        return AgentApprovalTokenStore(
            self.path,
            protector=self.protector,
            event_bus=event_bus,
        )

    def test_provision_creates_protected_file(self) -> None:
        store = self.make_store()
        result = store.provision()

        self.assertTrue(result.created)
        self.assertFalse(result.rotated)
        self.assertTrue(self.path.is_file())
        self.assertEqual(len(result.metadata.fingerprint), 64)

    def test_provisioning_result_never_exposes_secret(self) -> None:
        result = self.make_store().provision()
        encoded = json.dumps(result.as_dict())

        self.assertFalse(result.as_dict()["secret_exposed"])
        self.assertNotIn("owner_token", encoded)
        self.assertNotIn("ciphertext", encoded)

    def test_load_round_trips_generated_token(self) -> None:
        store = self.make_store()
        result = store.provision()
        token = store.load()

        self.assertGreaterEqual(len(token), 32)
        self.assertEqual(
            token_fingerprint(token),
            result.metadata.fingerprint,
        )

    def test_existing_token_requires_explicit_reuse(self) -> None:
        store = self.make_store()
        store.provision()

        with self.assertRaises(ApprovalTokenExistsError):
            store.provision()

    def test_reuse_existing_preserves_fingerprint(self) -> None:
        store = self.make_store()
        first = store.provision()
        reused = store.provision(reuse_existing=True)

        self.assertFalse(reused.created)
        self.assertFalse(reused.rotated)
        self.assertEqual(
            reused.metadata.fingerprint,
            first.metadata.fingerprint,
        )

    def test_rotate_changes_fingerprint(self) -> None:
        store = self.make_store()
        first = store.provision()
        rotated = store.rotate()

        self.assertTrue(rotated.rotated)
        self.assertNotEqual(
            rotated.metadata.fingerprint,
            first.metadata.fingerprint,
        )
        self.assertIsNotNone(rotated.metadata.rotated_at)

    def test_fingerprint_tampering_is_detected(self) -> None:
        store = self.make_store()
        store.provision()
        envelope = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        envelope["fingerprint"] = "0" * 64
        self.path.write_text(
            json.dumps(envelope),
            encoding="utf-8",
        )

        with self.assertRaises(ApprovalTokenIntegrityError):
            store.load()

    def test_ciphertext_corruption_is_rejected(self) -> None:
        store = self.make_store()
        store.provision()
        envelope = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        envelope["ciphertext"] = "not-base64!"
        self.path.write_text(
            json.dumps(envelope),
            encoding="utf-8",
        )

        with self.assertRaises(ApprovalTokenStoreError):
            store.load()

    def test_unknown_schema_field_is_rejected(self) -> None:
        store = self.make_store()
        store.provision()
        envelope = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        envelope["secret"] = "not-allowed"
        self.path.write_text(
            json.dumps(envelope),
            encoding="utf-8",
        )

        with self.assertRaises(ApprovalTokenStoreError):
            store.load()

    def test_unsupported_version_is_rejected(self) -> None:
        store = self.make_store()
        store.provision()
        envelope = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        envelope["version"] = 999
        self.path.write_text(
            json.dumps(envelope),
            encoding="utf-8",
        )

        with self.assertRaises(ApprovalTokenStoreError):
            store.load()

    def test_missing_file_fails_closed(self) -> None:
        with self.assertRaises(ApprovalTokenStoreError):
            self.make_store().load()

    def test_verify_returns_secret_free_metadata(self) -> None:
        store = self.make_store()
        store.provision()
        metadata = store.verify()

        self.assertEqual(
            Path(metadata.path),
            self.path.resolve(strict=False),
        )
        self.assertEqual(metadata.scheme, self.protector.scheme)
        self.assertNotIn(
            store.load(),
            json.dumps(metadata.as_dict()),
        )

    def test_telemetry_does_not_expose_secret_or_path(self) -> None:
        event_bus = EventBus()
        events = []
        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event),
        )
        store = self.make_store(event_bus=event_bus)
        store.provision()
        token = store.load()
        store.verify()
        encoded = json.dumps(
            [event.payload for event in events],
            default=str,
        )

        self.assertNotIn(token, encoded)
        self.assertNotIn(str(self.path), encoded)

    def test_resolve_inline_token(self) -> None:
        token = "A" * 48

        self.assertEqual(
            resolve_approval_owner_token(
                inline_token=token,
                token_file=None,
            ),
            token,
        )

    def test_resolve_protected_file_token(self) -> None:
        store = self.make_store()
        store.provision()

        with mock.patch(
            "agents.approval_tokens.WindowsDpapiProtector",
            return_value=self.protector,
        ):
            token = resolve_approval_owner_token(
                inline_token=None,
                token_file=self.path,
            )

        self.assertEqual(
            token_fingerprint(token),
            store.metadata().fingerprint,
        )

    def test_ambiguous_sources_fail_closed(self) -> None:
        with self.assertRaises(
            ApprovalTokenConfigurationError
        ):
            resolve_approval_owner_token(
                inline_token="A" * 48,
                token_file=self.path,
            )

    def test_no_source_returns_none(self) -> None:
        self.assertIsNone(
            resolve_approval_owner_token(
                inline_token=None,
                token_file=None,
            )
        )

    def test_blank_environment_style_sources_are_unconfigured(
        self,
    ) -> None:
        self.assertIsNone(
            resolve_approval_owner_token(
                inline_token="",
                token_file="   ",
            )
        )

    def test_resolved_token_authenticates_interface(self) -> None:
        store = self.make_store()
        store.provision()
        token = store.load()
        gate = AgentApprovalGate()
        interface = AgentApprovalCommandInterface(
            gate,
            enabled=True,
            owner_token=token,
        )
        gate.start()
        interface.start()

        try:
            response = interface.execute(
                {
                    "type": "approval",
                    "action": "list",
                    "token": token,
                }
            )
        finally:
            interface.stop()
            gate.stop()

        self.assertEqual(response["status"], "ok")

    def test_default_path_prefers_local_app_data(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"LOCALAPPDATA": str(self.root)},
            clear=False,
        ):
            path = default_approval_token_path()

        self.assertEqual(
            path,
            self.root
            / "Ghostfire"
            / "secrets"
            / "agent-approval-owner-token.json",
        )

    def test_file_permissions_are_owner_only_when_supported(self) -> None:
        if os.name == "nt":
            self.skipTest(
                "Windows ACLs are enforced separately from POSIX mode bits"
            )

        store = self.make_store()
        store.provision()
        mode = self.path.stat().st_mode & 0o777

        self.assertEqual(mode & 0o077, 0)


if __name__ == "__main__":
    unittest.main()
