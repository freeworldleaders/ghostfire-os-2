"""Secure owner-token provisioning for GhostFire approval commands."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from core.eventbus import EventBus


_TOKEN_FILE_VERSION = 1
_TOKEN_BYTES = 48
_MAX_TOKEN_FILE_BYTES = 65_536
_DPAPI_DESCRIPTION = "GhostFire agent approval owner token"
_DPAPI_ENTROPY = b"ghostfire-agent-approval-owner-token-v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class ApprovalTokenError(RuntimeError):
    """Base class for approval-token failures."""


class ApprovalTokenProtectionError(ApprovalTokenError):
    """Raised when token encryption or decryption fails."""


class ApprovalTokenStoreError(ApprovalTokenError):
    """Raised when the token store is invalid or unavailable."""


class ApprovalTokenExistsError(ApprovalTokenStoreError):
    """Raised when provisioning would overwrite an existing token."""


class ApprovalTokenIntegrityError(ApprovalTokenStoreError):
    """Raised when protected token integrity validation fails."""


class ApprovalTokenConfigurationError(ApprovalTokenError):
    """Raised when runtime token-source configuration is ambiguous."""


class TokenProtector(Protocol):
    """Minimal token-protection contract."""

    @property
    def scheme(self) -> str:
        """Return the storage protection scheme name."""

    def protect(self, plaintext: bytes) -> bytes:
        """Protect plaintext bytes."""

    def unprotect(self, ciphertext: bytes) -> bytes:
        """Recover plaintext bytes."""


@dataclass(frozen=True, slots=True)
class ApprovalTokenMetadata:
    """Immutable, secret-free token metadata."""

    path: str
    fingerprint: str
    scheme: str
    version: int
    created_at: datetime
    rotated_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe secret-free representation."""

        return {
            "path": self.path,
            "fingerprint": self.fingerprint,
            "scheme": self.scheme,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "rotated_at": (
                self.rotated_at.isoformat()
                if self.rotated_at is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ApprovalTokenProvisioningResult:
    """Result of a provisioning or reuse operation."""

    metadata: ApprovalTokenMetadata
    created: bool
    rotated: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result without token material."""

        return {
            **self.metadata.as_dict(),
            "created": self.created,
            "rotated": self.rotated,
            "secret_exposed": False,
        }


class WindowsDpapiProtector:
    """Windows CurrentUser DPAPI protector."""

    @property
    def scheme(self) -> str:
        return "windows-dpapi-current-user"

    def protect(self, plaintext: bytes) -> bytes:
        """Encrypt bytes for the current Windows user."""

        return self._crypt(
            plaintext,
            operation="protect",
        )

    def unprotect(self, ciphertext: bytes) -> bytes:
        """Decrypt bytes for the current Windows user."""

        return self._crypt(
            ciphertext,
            operation="unprotect",
        )

    @staticmethod
    def _crypt(
        value: bytes,
        *,
        operation: str,
    ) -> bytes:
        if sys.platform != "win32":
            raise ApprovalTokenProtectionError(
                "Windows DPAPI is unavailable on this platform"
            )

        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")

        if not value:
            raise ApprovalTokenProtectionError(
                "DPAPI input cannot be empty"
            )

        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_byte)),
            ]

        def make_blob(data: bytes) -> tuple[DataBlob, Any]:
            buffer = ctypes.create_string_buffer(data)
            blob = DataBlob(
                len(data),
                ctypes.cast(
                    buffer,
                    ctypes.POINTER(ctypes.c_byte),
                ),
            )
            return blob, buffer

        input_blob, input_buffer = make_blob(value)
        entropy_blob, entropy_buffer = make_blob(
            _DPAPI_ENTROPY
        )
        output_blob = DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        if operation == "protect":
            success = crypt32.CryptProtectData(
                ctypes.byref(input_blob),
                _DPAPI_DESCRIPTION,
                ctypes.byref(entropy_blob),
                None,
                None,
                _CRYPTPROTECT_UI_FORBIDDEN,
                ctypes.byref(output_blob),
            )
        elif operation == "unprotect":
            success = crypt32.CryptUnprotectData(
                ctypes.byref(input_blob),
                None,
                ctypes.byref(entropy_blob),
                None,
                None,
                _CRYPTPROTECT_UI_FORBIDDEN,
                ctypes.byref(output_blob),
            )
        else:
            raise ValueError(
                f"unsupported DPAPI operation: {operation}"
            )

        _ = input_buffer
        _ = entropy_buffer

        if not success:
            error_code = ctypes.get_last_error()
            raise ApprovalTokenProtectionError(
                f"Windows DPAPI {operation} failed: "
                f"error {error_code}"
            )

        try:
            return ctypes.string_at(
                output_blob.pbData,
                output_blob.cbData,
            )
        finally:
            if output_blob.pbData:
                kernel32.LocalFree(output_blob.pbData)


class AgentApprovalTokenStore:
    """
    Atomic protected-file store for one approval-command owner token.

    Token material is never returned by provisioning methods, snapshots, or
    telemetry. The plaintext token is returned only by ``load`` for in-process
    authentication wiring.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        protector: TokenProtector | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        if event_bus is not None and not isinstance(
            event_bus,
            EventBus,
        ):
            raise TypeError("event_bus must be an EventBus or None")

        self._path = _normalize_path(path)
        self._protector = protector or WindowsDpapiProtector()
        self._event_bus = event_bus

        scheme = self._protector.scheme

        if not isinstance(scheme, str) or not scheme.strip():
            raise TypeError(
                "protector.scheme must be a non-empty string"
            )

        if not callable(getattr(self._protector, "protect", None)):
            raise TypeError("protector must define protect")
        if not callable(getattr(self._protector, "unprotect", None)):
            raise TypeError("protector must define unprotect")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.is_file()

    def provision(
        self,
        *,
        reuse_existing: bool = False,
    ) -> ApprovalTokenProvisioningResult:
        """
        Provision a new random token or safely reuse an existing valid token.

        Existing files are never overwritten by this method.
        """

        if self._path.exists():
            if not reuse_existing:
                raise ApprovalTokenExistsError(
                    f"approval token already exists: {self._path}"
                )

            metadata = self.metadata()
            result = ApprovalTokenProvisioningResult(
                metadata=metadata,
                created=False,
                rotated=False,
            )
            self._publish(
                "ghostfire.approval_token.reused",
                result.as_dict(),
            )
            return result

        token = secrets.token_urlsafe(_TOKEN_BYTES)
        envelope = self._build_envelope(
            token,
            created_at=datetime.now(timezone.utc),
            rotated_at=None,
        )
        self._write_envelope(envelope)
        metadata = self.metadata()
        result = ApprovalTokenProvisioningResult(
            metadata=metadata,
            created=True,
            rotated=False,
        )
        self._publish(
            "ghostfire.approval_token.provisioned",
            result.as_dict(),
        )
        return result

    def rotate(self) -> ApprovalTokenProvisioningResult:
        """Atomically replace an existing token with a new random token."""

        current = self._read_envelope()
        created_at = _parse_datetime(
            current["created_at"],
            field_name="created_at",
        )
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        envelope = self._build_envelope(
            token,
            created_at=created_at,
            rotated_at=datetime.now(timezone.utc),
        )
        self._write_envelope(
            envelope,
            overwrite=True,
        )
        metadata = self.metadata()
        result = ApprovalTokenProvisioningResult(
            metadata=metadata,
            created=False,
            rotated=True,
        )
        self._publish(
            "ghostfire.approval_token.rotated",
            result.as_dict(),
        )
        return result

    def load(self) -> str:
        """Decrypt and integrity-check the owner token."""

        envelope = self._read_envelope()
        ciphertext = _decode_ciphertext(
            envelope["ciphertext"]
        )

        try:
            plaintext = self._protector.unprotect(ciphertext)
        except ApprovalTokenError:
            raise
        except Exception as exc:
            raise ApprovalTokenProtectionError(
                "token decryption failed"
            ) from exc

        try:
            token = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApprovalTokenIntegrityError(
                "decrypted token is not valid UTF-8"
            ) from exc
        finally:
            plaintext = b""

        _validate_token(token)
        actual_fingerprint = token_fingerprint(token)
        expected_fingerprint = envelope["fingerprint"]

        if not hmac.compare_digest(
            actual_fingerprint,
            expected_fingerprint,
        ):
            raise ApprovalTokenIntegrityError(
                "approval token fingerprint mismatch"
            )

        return token

    def metadata(self) -> ApprovalTokenMetadata:
        """Return validated secret-free token metadata."""

        envelope = self._read_envelope()
        token = self.load()

        try:
            fingerprint = token_fingerprint(token)
        finally:
            token = ""

        return ApprovalTokenMetadata(
            path=str(self._path),
            fingerprint=fingerprint,
            scheme=envelope["scheme"],
            version=envelope["version"],
            created_at=_parse_datetime(
                envelope["created_at"],
                field_name="created_at",
            ),
            rotated_at=(
                _parse_datetime(
                    envelope["rotated_at"],
                    field_name="rotated_at",
                )
                if envelope["rotated_at"] is not None
                else None
            ),
        )

    def verify(self) -> ApprovalTokenMetadata:
        """Validate that the stored token decrypts and matches metadata."""

        metadata = self.metadata()
        self._publish(
            "ghostfire.approval_token.verified",
            metadata.as_dict(),
        )
        return metadata

    def _build_envelope(
        self,
        token: str,
        *,
        created_at: datetime,
        rotated_at: datetime | None,
    ) -> dict[str, Any]:
        _validate_token(token)

        plaintext = token.encode("utf-8")
        fingerprint = token_fingerprint_bytes(plaintext)

        try:
            protected = self._protector.protect(plaintext)
        except ApprovalTokenError:
            raise
        except Exception as exc:
            raise ApprovalTokenProtectionError(
                "token encryption failed"
            ) from exc
        finally:
            plaintext = b""
            token = ""

        if not isinstance(protected, bytes) or not protected:
            raise ApprovalTokenProtectionError(
                "protector returned invalid ciphertext"
            )

        return {
            "version": _TOKEN_FILE_VERSION,
            "scheme": self._protector.scheme,
            "fingerprint": fingerprint,
            "created_at": _normalize_datetime(
                created_at
            ).isoformat(),
            "rotated_at": (
                _normalize_datetime(rotated_at).isoformat()
                if rotated_at is not None
                else None
            ),
            "ciphertext": base64.b64encode(
                protected
            ).decode("ascii"),
        }

    def _read_envelope(self) -> dict[str, Any]:
        if self._path.is_symlink():
            raise ApprovalTokenStoreError(
                "approval token path cannot be a symbolic link"
            )

        if not self._path.is_file():
            raise ApprovalTokenStoreError(
                f"approval token file not found: {self._path}"
            )

        try:
            size = self._path.stat().st_size
        except OSError as exc:
            raise ApprovalTokenStoreError(
                "unable to inspect approval token file"
            ) from exc

        if size < 2 or size > _MAX_TOKEN_FILE_BYTES:
            raise ApprovalTokenStoreError(
                "approval token file size is invalid"
            )

        try:
            loaded = json.loads(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApprovalTokenStoreError(
                "approval token file is not valid JSON"
            ) from exc

        if not isinstance(loaded, Mapping):
            raise ApprovalTokenStoreError(
                "approval token file root must be an object"
            )

        envelope = dict(loaded)
        expected_fields = {
            "version",
            "scheme",
            "fingerprint",
            "created_at",
            "rotated_at",
            "ciphertext",
        }

        if set(envelope) != expected_fields:
            raise ApprovalTokenStoreError(
                "approval token file schema is invalid"
            )

        if envelope["version"] != _TOKEN_FILE_VERSION:
            raise ApprovalTokenStoreError(
                "unsupported approval token file version"
            )

        if envelope["scheme"] != self._protector.scheme:
            raise ApprovalTokenStoreError(
                "approval token protection scheme mismatch"
            )

        fingerprint = envelope["fingerprint"]

        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint
            )
        ):
            raise ApprovalTokenStoreError(
                "approval token fingerprint is invalid"
            )

        _parse_datetime(
            envelope["created_at"],
            field_name="created_at",
        )

        if envelope["rotated_at"] is not None:
            _parse_datetime(
                envelope["rotated_at"],
                field_name="rotated_at",
            )

        _decode_ciphertext(envelope["ciphertext"])
        return envelope

    def _write_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        if self._path.is_symlink():
            raise ApprovalTokenStoreError(
                "approval token path cannot be a symbolic link"
            )

        if self._path.exists() and not overwrite:
            raise ApprovalTokenExistsError(
                f"approval token already exists: {self._path}"
            )

        parent = self._path.parent
        parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        if parent.is_symlink():
            raise ApprovalTokenStoreError(
                "approval token directory cannot be a symbolic link"
            )

        payload = json.dumps(
            dict(envelope),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ) + "\n"
        temporary_path: Path | None = None

        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=str(parent),
            )
            temporary_path = Path(temporary_name)

            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self._path)
            os.chmod(self._path, 0o600)
        except OSError as exc:
            raise ApprovalTokenStoreError(
                "failed to write approval token file"
            ) from exc
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(missing_ok=True)

    def _publish(
        self,
        event_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self._event_bus is None:
            return

        safe_payload = deepcopy(dict(payload))
        safe_payload.pop("path", None)
        self._event_bus.emit(
            event_name,
            safe_payload,
            raise_exceptions=False,
        )


def resolve_approval_owner_token(
    *,
    inline_token: str | None,
    token_file: str | Path | None,
    event_bus: EventBus | None = None,
) -> str | None:
    """
    Resolve exactly one configured owner-token source.

    Inline and protected-file sources are mutually exclusive. A configured
    protected file must load successfully; failures are not silently ignored.
    """

    normalized_inline_token: str | None

    if inline_token is None:
        normalized_inline_token = None
    elif not isinstance(inline_token, str):
        raise TypeError("inline_token must be a string or None")
    elif not inline_token.strip():
        normalized_inline_token = None
    else:
        normalized_inline_token = _validate_token(
            inline_token
        )

    if token_file is None:
        normalized_path = None
    elif isinstance(token_file, str) and not token_file.strip():
        normalized_path = None
    else:
        normalized_path = _normalize_path(token_file)

    if (
        normalized_inline_token is not None
        and normalized_path is not None
    ):
        raise ApprovalTokenConfigurationError(
            "configure owner_token or owner_token_file, not both"
        )

    if normalized_inline_token is not None:
        return normalized_inline_token

    if normalized_path is None:
        return None

    return AgentApprovalTokenStore(
        normalized_path,
        event_bus=event_bus,
    ).load()


def default_approval_token_path() -> Path:
    """Return the per-user protected token path."""

    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data:
        root = Path(local_app_data) / "Ghostfire"
    else:
        root = Path.home() / ".ghostfire"

    return (
        root
        / "secrets"
        / "agent-approval-owner-token.json"
    )


def token_fingerprint(token: str) -> str:
    """Return a SHA-256 fingerprint without exposing the token."""

    _validate_token(token)
    return token_fingerprint_bytes(token.encode("utf-8"))


def token_fingerprint_bytes(value: bytes) -> str:
    """Return a SHA-256 fingerprint for token bytes."""

    if not isinstance(value, bytes) or not value:
        raise ValueError("token bytes cannot be empty")

    return hashlib.sha256(value).hexdigest()


def _decode_ciphertext(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApprovalTokenStoreError(
            "approval token ciphertext is invalid"
        )

    try:
        decoded = base64.b64decode(
            value,
            validate=True,
        )
    except ValueError as exc:
        raise ApprovalTokenStoreError(
            "approval token ciphertext is not valid base64"
        ) from exc

    if not decoded:
        raise ApprovalTokenStoreError(
            "approval token ciphertext is empty"
        )

    return decoded


def _normalize_path(value: str | Path) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("token path cannot be empty")
        path = Path(value.strip())
    else:
        raise TypeError("token path must be a string or Path")

    return path.expanduser().resolve(strict=False)


def _normalize_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime")

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _parse_datetime(
    value: Any,
    *,
    field_name: str,
) -> datetime:
    if not isinstance(value, str):
        raise ApprovalTokenStoreError(
            f"{field_name} must be an ISO-8601 string"
        )

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ApprovalTokenStoreError(
            f"{field_name} is not valid ISO-8601"
        ) from exc

    if parsed.tzinfo is None:
        raise ApprovalTokenStoreError(
            f"{field_name} must include a timezone"
        )

    return parsed.astimezone(timezone.utc)


def _validate_token(token: Any) -> str:
    if not isinstance(token, str):
        raise TypeError("owner token must be a string")

    if token != token.strip():
        raise ValueError(
            "owner token cannot include surrounding whitespace"
        )

    if len(token) < 32:
        raise ValueError(
            "owner token must contain at least 32 characters"
        )

    if any(character.isspace() for character in token):
        raise ValueError(
            "owner token cannot contain whitespace"
        )

    return token
