"""Secure local owner workflow for GhostFire approval commands."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any

from agents.approval_tokens import (
    resolve_approval_owner_token,
    token_fingerprint,
)


class ApprovalOwnerError(RuntimeError):
    """Base class for owner-workflow failures."""


class ApprovalOwnerConfigurationError(ApprovalOwnerError):
    """Raised when owner workflow configuration is unsafe or incomplete."""


class ApprovalOwnerConnectionError(ApprovalOwnerError):
    """Raised when the local WebSocket command server cannot be reached."""


class ApprovalOwnerProtocolError(ApprovalOwnerError):
    """Raised when the WebSocket peer violates the expected protocol."""


class ApprovalOwnerResponseError(ApprovalOwnerError):
    """Raised when the approval command server returns a public error."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = message
        super().__init__(f"{code}: {message}")


class ApprovalOwnerConfirmationError(ApprovalOwnerError):
    """Raised when a mutating owner command lacks exact confirmation."""


@dataclass(frozen=True, slots=True)
class ApprovalOwnerSnapshot:
    """Secret-free owner workflow status."""

    host: str
    port: int
    path: str
    enabled: bool
    token_configured: bool
    token_fingerprint: str | None
    transport_authenticated: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe secret-free snapshot."""

        return {
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "enabled": self.enabled,
            "token_configured": self.token_configured,
            "token_fingerprint": self.token_fingerprint,
            "transport_authenticated": self.transport_authenticated,
            "secret_exposed": False,
        }


class AgentApprovalOwnerWorkflow:
    """
    Local, token-authenticated owner workflow for approval decisions.

    Each operation opens one loopback WebSocket connection, sends one request,
    validates the correlated response, and closes the connection. Mutating
    actions require an exact action-and-request confirmation phrase.
    """

    _MUTATING_ACTIONS = frozenset({"approve", "deny", "cancel"})
    _ALLOWED_STATUSES = frozenset(
        {
            "pending",
            "approved",
            "denied",
            "consumed",
            "cancelled",
            "expired",
        }
    )

    def __init__(
        self,
        *,
        host: str,
        port: int,
        path: str,
        owner_token: str,
        transport_token: str | None = None,
        timeout: float = 3.0,
        max_response_bytes: int = 65_536,
        enabled: bool = True,
    ) -> None:
        self._host = _validate_loopback_host(host)
        self._port = _validate_port(port)
        self._path = _validate_path(path)
        self._owner_token = _validate_secret(
            owner_token,
            field_name="owner_token",
        )
        self._transport_token = (
            _validate_secret(
                transport_token,
                field_name="transport_token",
            )
            if transport_token is not None
            else None
        )
        self._timeout = _validate_positive_number(
            timeout,
            field_name="timeout",
        )
        self._max_response_bytes = _validate_positive_int(
            max_response_bytes,
            field_name="max_response_bytes",
        )

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        self._enabled = enabled
        self._fingerprint = token_fingerprint(self._owner_token)

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        *,
        timeout: float = 3.0,
    ) -> "AgentApprovalOwnerWorkflow":
        """Build the owner workflow from active GhostFire settings."""

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")

        command_settings = _require_mapping(
            settings,
            "agent_approval_commands",
        )
        websocket_settings = _require_mapping(
            settings,
            "websocket_command_server",
        )

        if command_settings.get("enabled") is not True:
            raise ApprovalOwnerConfigurationError(
                "agent approval commands are not activated"
            )

        if websocket_settings.get("enabled") is not True:
            raise ApprovalOwnerConfigurationError(
                "WebSocket command server is disabled"
            )

        owner_token = resolve_approval_owner_token(
            inline_token=command_settings.get("owner_token"),
            token_file=command_settings.get("owner_token_file"),
        )

        if owner_token is None:
            raise ApprovalOwnerConfigurationError(
                "approval owner token is not configured"
            )

        transport_token = websocket_settings.get("auth_token")

        if isinstance(transport_token, str) and not transport_token.strip():
            transport_token = None

        try:
            return cls(
                host=websocket_settings.get("host"),
                port=websocket_settings.get("port"),
                path=websocket_settings.get("path"),
                owner_token=owner_token,
                transport_token=transport_token,
                timeout=timeout,
                max_response_bytes=websocket_settings.get(
                    "max_message_bytes",
                    65_536,
                ),
                enabled=True,
            )
        finally:
            owner_token = ""

    @property
    def confirmation_format(self) -> str:
        return "ACTION:APPROVAL_ID"

    @staticmethod
    def confirmation_phrase(
        action: str,
        approval_id: str,
    ) -> str:
        """Return the exact confirmation phrase for a mutation."""

        normalized_action = _validate_action(action)
        normalized_id = _validate_text(
            approval_id,
            field_name="approval_id",
        )

        if normalized_action not in AgentApprovalOwnerWorkflow._MUTATING_ACTIONS:
            raise ApprovalOwnerConfirmationError(
                "confirmation phrases apply only to approve, deny, or cancel"
            )

        return f"{normalized_action.upper()}:{normalized_id}"

    def snapshot(self) -> ApprovalOwnerSnapshot:
        """Return secret-free connection and activation status."""

        return ApprovalOwnerSnapshot(
            host=self._host,
            port=self._port,
            path=self._path,
            enabled=self._enabled,
            token_configured=bool(self._owner_token),
            token_fingerprint=(
                self._fingerprint
                if self._owner_token
                else None
            ),
            transport_authenticated=(
                self._transport_token is not None
            ),
        )

    def close(self) -> None:
        """Clear in-process token references."""

        self._owner_token = ""
        self._transport_token = None
        self._fingerprint = ""

    def status(self) -> dict[str, Any]:
        """Read runtime status without sending the approval owner token."""

        return self._exchange(
            {
                "type": "status",
            },
            expected_type="status",
        )

    def list(
        self,
        *,
        status: str = "pending",
    ) -> list[dict[str, Any]]:
        """List approval requests by lifecycle status."""

        normalized_status = _validate_text(
            status,
            field_name="status",
        ).lower()

        if normalized_status not in self._ALLOWED_STATUSES:
            raise ValueError(
                "status must be pending, approved, denied, consumed, "
                "cancelled, or expired"
            )

        response = self._approval_exchange(
            "list",
            status=normalized_status,
        )
        data = response.get("data")

        if not isinstance(data, list):
            raise ApprovalOwnerProtocolError(
                "approval list response data must be an array"
            )

        return [
            deepcopy(dict(item))
            for item in data
            if isinstance(item, Mapping)
        ]

    def get(self, approval_id: str) -> dict[str, Any]:
        """Read one approval request."""

        response = self._approval_exchange(
            "get",
            approval_id=_validate_text(
                approval_id,
                field_name="approval_id",
            ),
        )
        return _require_response_object(response)

    def approve(
        self,
        approval_id: str,
        *,
        note: str = "",
        confirmation: str,
    ) -> dict[str, Any]:
        """Approve one request after exact owner confirmation."""

        return self._mutate(
            "approve",
            approval_id,
            note=note,
            confirmation=confirmation,
        )

    def deny(
        self,
        approval_id: str,
        *,
        note: str = "",
        confirmation: str,
    ) -> dict[str, Any]:
        """Deny one request after exact owner confirmation."""

        return self._mutate(
            "deny",
            approval_id,
            note=note,
            confirmation=confirmation,
        )

    def cancel(
        self,
        approval_id: str,
        *,
        note: str = "",
        confirmation: str,
    ) -> dict[str, Any]:
        """Cancel one request after exact owner confirmation."""

        return self._mutate(
            "cancel",
            approval_id,
            note=note,
            confirmation=confirmation,
        )

    def _mutate(
        self,
        action: str,
        approval_id: str,
        *,
        note: str,
        confirmation: str,
    ) -> dict[str, Any]:
        normalized_action = _validate_action(action)
        normalized_id = _validate_text(
            approval_id,
            field_name="approval_id",
        )
        normalized_note = _validate_note(note)
        expected = self.confirmation_phrase(
            normalized_action,
            normalized_id,
        )

        if (
            not isinstance(confirmation, str)
            or not hmac.compare_digest(
                confirmation,
                expected,
            )
        ):
            raise ApprovalOwnerConfirmationError(
                f"exact confirmation required: {expected}"
            )

        response = self._approval_exchange(
            normalized_action,
            approval_id=normalized_id,
            note=normalized_note,
        )
        return _require_response_object(response)

    def _approval_exchange(
        self,
        action: str,
        **fields: Any,
    ) -> dict[str, Any]:
        if not self._enabled:
            raise ApprovalOwnerConfigurationError(
                "agent approval commands are not activated"
            )

        if not self._owner_token:
            raise ApprovalOwnerConfigurationError(
                "approval owner token is unavailable"
            )

        message = {
            "type": "approval",
            "action": _validate_action(action),
            "token": self._owner_token,
            **fields,
        }
        return self._exchange(
            message,
            expected_type="approval_result",
        )

    def _exchange(
        self,
        message: Mapping[str, Any],
        *,
        expected_type: str,
    ) -> dict[str, Any]:
        request_id = secrets.token_hex(12)
        payload = {
            "id": request_id,
            **dict(message),
        }
        connection: socket.socket | None = None

        try:
            connection = socket.create_connection(
                (self._host, self._port),
                timeout=self._timeout,
            )
            connection.settimeout(self._timeout)
            self._handshake(connection)
            self._send_json(connection, payload)
            response = self._receive_json(connection)
        except ApprovalOwnerError:
            raise
        except (
            ConnectionError,
            OSError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise ApprovalOwnerConnectionError(
                "local GhostFire WebSocket command server is unavailable"
            ) from exc
        finally:
            if connection is not None:
                self._close_socket(connection)

        if response.get("id") != request_id:
            raise ApprovalOwnerProtocolError(
                "response request identifier did not match"
            )

        if response.get("status") == "error":
            code = response.get("error")
            message_value = response.get("message")

            if not isinstance(code, str) or not code:
                code = "approval_owner_request_failed"

            if (
                not isinstance(message_value, str)
                or not message_value
            ):
                message_value = "owner workflow request failed"

            raise ApprovalOwnerResponseError(
                code,
                message_value,
            )

        if response.get("status") != "ok":
            raise ApprovalOwnerProtocolError(
                "response status must be ok or error"
            )

        if response.get("type") != expected_type:
            raise ApprovalOwnerProtocolError(
                f"expected response type {expected_type}"
            )

        return deepcopy(response)

    def _handshake(self, connection: socket.socket) -> None:
        key = base64.b64encode(
            os.urandom(16)
        ).decode("ascii")
        lines = [
            f"GET {self._path} HTTP/1.1",
            f"Host: {self._host}:{self._port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]

        if self._transport_token is not None:
            lines.append(
                f"Authorization: Bearer {self._transport_token}"
            )

        request = "\r\n".join(lines) + "\r\n\r\n"
        connection.sendall(request.encode("ascii"))
        response = self._read_http_headers(connection)
        status_line, headers = _parse_http_headers(response)

        if status_line != "HTTP/1.1 101 Switching Protocols":
            raise ApprovalOwnerConnectionError(
                "WebSocket command server rejected the connection"
            )

        expected_accept = base64.b64encode(
            hashlib.sha1(
                (
                    key
                    + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                ).encode("ascii")
            ).digest()
        ).decode("ascii")

        if not hmac.compare_digest(
            headers.get("sec-websocket-accept", ""),
            expected_accept,
        ):
            raise ApprovalOwnerProtocolError(
                "WebSocket handshake accept value was invalid"
            )

    def _send_json(
        self,
        connection: socket.socket,
        payload: Mapping[str, Any],
    ) -> None:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        _send_masked_frame(connection, 0x1, encoded)

    def _receive_json(
        self,
        connection: socket.socket,
    ) -> dict[str, Any]:
        while True:
            opcode, payload = _receive_frame(
                connection,
                max_payload_bytes=self._max_response_bytes,
            )

            if opcode == 0x9:
                _send_masked_frame(connection, 0xA, payload)
                continue

            if opcode == 0xA:
                continue

            if opcode == 0x8:
                raise ApprovalOwnerConnectionError(
                    "WebSocket command server closed the connection"
                )

            if opcode != 0x1:
                raise ApprovalOwnerProtocolError(
                    "owner workflow requires a text response frame"
                )

            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ApprovalOwnerProtocolError(
                    "owner workflow response was not valid JSON"
                ) from exc

            if not isinstance(decoded, dict):
                raise ApprovalOwnerProtocolError(
                    "owner workflow response root must be an object"
                )

            return decoded

    def _read_http_headers(
        self,
        connection: socket.socket,
    ) -> bytes:
        data = bytearray()

        while b"\r\n\r\n" not in data:
            chunk = connection.recv(4096)

            if not chunk:
                break

            data.extend(chunk)

            if len(data) > 32_768:
                raise ApprovalOwnerProtocolError(
                    "WebSocket handshake headers were too large"
                )

        if b"\r\n\r\n" not in data:
            raise ApprovalOwnerProtocolError(
                "WebSocket handshake was incomplete"
            )

        return bytes(data)

    @staticmethod
    def _close_socket(connection: socket.socket) -> None:
        try:
            _send_masked_frame(
                connection,
                0x8,
                struct.pack("!H", 1000),
            )
        except OSError:
            pass

        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            connection.close()
        except OSError:
            pass


def _send_masked_frame(
    connection: socket.socket,
    opcode: int,
    payload: bytes,
) -> None:
    first = 0x80 | opcode
    length = len(payload)

    if length <= 125:
        header = bytes((first, 0x80 | length))
    elif length <= 65_535:
        header = (
            bytes((first, 0x80 | 126))
            + struct.pack("!H", length)
        )
    else:
        header = (
            bytes((first, 0x80 | 127))
            + struct.pack("!Q", length)
        )

    mask = os.urandom(4)
    masked = bytes(
        value ^ mask[index % 4]
        for index, value in enumerate(payload)
    )
    connection.sendall(header + mask + masked)


def _receive_frame(
    connection: socket.socket,
    *,
    max_payload_bytes: int,
) -> tuple[int, bytes]:
    first = _read_exact(connection, 2)
    first_byte, second_byte = first
    opcode = first_byte & 0x0F
    length = second_byte & 0x7F

    if not first_byte & 0x80:
        raise ApprovalOwnerProtocolError(
            "fragmented WebSocket responses are unsupported"
        )

    if second_byte & 0x80:
        raise ApprovalOwnerProtocolError(
            "server WebSocket responses must not be masked"
        )

    if length == 126:
        length = struct.unpack(
            "!H",
            _read_exact(connection, 2),
        )[0]
    elif length == 127:
        length = struct.unpack(
            "!Q",
            _read_exact(connection, 8),
        )[0]

    if length > max_payload_bytes:
        raise ApprovalOwnerProtocolError(
            "WebSocket response exceeded configured limit"
        )

    return opcode, _read_exact(connection, length)


def _read_exact(
    connection: socket.socket,
    length: int,
) -> bytes:
    data = bytearray()

    while len(data) < length:
        chunk = connection.recv(length - len(data))

        if not chunk:
            raise ConnectionError(
                "connection closed during WebSocket frame read"
            )

        data.extend(chunk)

    return bytes(data)


def _parse_http_headers(
    response: bytes,
) -> tuple[str, dict[str, str]]:
    try:
        head = response.split(b"\r\n\r\n", 1)[0].decode(
            "ascii"
        )
    except UnicodeDecodeError as exc:
        raise ApprovalOwnerProtocolError(
            "WebSocket handshake headers must be ASCII"
        ) from exc

    lines = head.split("\r\n")

    if not lines or not lines[0]:
        raise ApprovalOwnerProtocolError(
            "WebSocket handshake status line was missing"
        )

    headers: dict[str, str] = {}

    for line in lines[1:]:
        if ":" not in line:
            raise ApprovalOwnerProtocolError(
                "WebSocket handshake header was malformed"
            )

        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    return lines[0], headers


def _require_response_object(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    data = response.get("data")

    if not isinstance(data, Mapping):
        raise ApprovalOwnerProtocolError(
            "approval response data must be an object"
        )

    return deepcopy(dict(data))


def _require_mapping(
    settings: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = settings.get(key)

    if not isinstance(value, Mapping):
        raise ApprovalOwnerConfigurationError(
            f"configuration section is missing or invalid: {key}"
        )

    return value


def _validate_loopback_host(value: Any) -> str:
    host = _validate_text(value, field_name="host")

    if host.lower() == "localhost":
        return host

    try:
        loopback = ip_address(host).is_loopback
    except ValueError as exc:
        raise ApprovalOwnerConfigurationError(
            "owner workflow host must be a loopback address"
        ) from exc

    if not loopback:
        raise ApprovalOwnerConfigurationError(
            "owner workflow host must be a loopback address"
        )

    return host


def _validate_port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("port must be an integer")

    if not 1 <= value <= 65_535:
        raise ValueError("port must be between 1 and 65535")

    return value


def _validate_path(value: Any) -> str:
    path = _validate_text(value, field_name="path")

    if not path.startswith("/"):
        raise ValueError("path must begin with /")

    return path


def _validate_secret(
    value: Any,
    *,
    field_name: str,
) -> str:
    secret = _validate_text(value, field_name=field_name)

    if any(character.isspace() for character in secret):
        raise ValueError(f"{field_name} cannot contain whitespace")

    return secret


def _validate_action(value: Any) -> str:
    action = _validate_text(
        value,
        field_name="action",
    ).lower()

    if action not in {
        "list",
        "get",
        "approve",
        "deny",
        "cancel",
    }:
        raise ValueError(
            "action must be list, get, approve, deny, or cancel"
        )

    return action


def _validate_note(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("note must be a string")

    return value.strip()


def _validate_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return normalized


def _validate_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _validate_positive_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive number"
        )

    return float(value)
