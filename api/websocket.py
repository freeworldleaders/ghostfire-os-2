"""Authenticated WebSocket command server for GhostFire OS."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import socket
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from ipaddress import ip_address
from socketserver import (
    StreamRequestHandler,
    TCPServer,
    ThreadingMixIn,
)
from struct import pack, unpack
from threading import RLock, Thread
from typing import Any
from urllib.parse import urlsplit

from core.eventbus import EventBus


CommandHandler = Callable[[str], Any]
StatusProvider = Callable[[], Mapping[str, Any]]
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketCommandError(RuntimeError):
    """Base class for WebSocket command-server failures."""


class WebSocketSecurityError(WebSocketCommandError):
    """Raised when the server would be exposed without authentication."""


class WebSocketProtocolError(WebSocketCommandError):
    """Raised when a client violates the WebSocket protocol."""


class WebSocketMessageTooLarge(WebSocketProtocolError):
    """Raised when a client frame exceeds the configured message limit."""


class _ThreadingWebSocketServer(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


class WebSocketCommandServer:
    """
    Dependency-free RFC 6455 command channel for GhostFire OS.

    R1 supports complete, masked client text frames, ping/pong, close frames,
    allowlisted commands, status requests, authentication, and telemetry.
    Fragmented messages and binary application messages are rejected.
    """

    def __init__(
        self,
        *,
        command_handler: CommandHandler,
        status_provider: StatusProvider,
        host: str = "127.0.0.1",
        port: int = 8103,
        auth_token: str | None = None,
        allowed_commands: Iterable[str] = ("BOOT", "STATUS"),
        path: str = "/v1/commands",
        max_message_bytes: int = 65_536,
        idle_timeout: float = 30.0,
        event_bus: EventBus | None = None,
    ) -> None:
        if not callable(command_handler):
            raise TypeError("command_handler must be callable")

        if not callable(status_provider):
            raise TypeError("status_provider must be callable")

        self._host = self._validate_text(
            host,
            field_name="host",
        )

        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError("port must be an integer")

        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")

        if auth_token is not None:
            auth_token = self._validate_text(
                auth_token,
                field_name="auth_token",
            )

        if not self._is_loopback_host(self._host) and not auth_token:
            raise WebSocketSecurityError(
                "non-loopback WebSocket binding requires auth_token"
            )

        normalized_path = self._validate_text(
            path,
            field_name="path",
        )

        if not normalized_path.startswith("/"):
            raise ValueError("path must begin with /")

        if (
            isinstance(max_message_bytes, bool)
            or not isinstance(max_message_bytes, int)
        ):
            raise TypeError("max_message_bytes must be an integer")

        if max_message_bytes < 1:
            raise ValueError("max_message_bytes must be positive")

        if (
            isinstance(idle_timeout, bool)
            or not isinstance(idle_timeout, (int, float))
        ):
            raise TypeError("idle_timeout must be numeric")

        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        normalized_commands: list[str] = []

        if isinstance(allowed_commands, str):
            raise TypeError(
                "allowed_commands must be an iterable of command names"
            )

        for command in allowed_commands:
            normalized = self._normalize_command(command)

            if normalized in normalized_commands:
                raise ValueError(
                    f"duplicate allowed command: {normalized}"
                )

            normalized_commands.append(normalized)

        if not normalized_commands:
            raise ValueError("allowed_commands cannot be empty")

        self._command_handler = command_handler
        self._status_provider = status_provider
        self._configured_port = port
        self._auth_token = auth_token
        self._allowed_commands = tuple(normalized_commands)
        self._path = normalized_path
        self._max_message_bytes = max_message_bytes
        self._idle_timeout = float(idle_timeout)
        self._event_bus = event_bus
        self._lock = RLock()
        self._server: _ThreadingWebSocketServer | None = None
        self._thread: Thread | None = None
        self._connections: set[socket.socket] = set()
        self._total_connections = 0
        self._message_count = 0
        self._command_count = 0

    @property
    def bound_port(self) -> int:
        """Return the active bound port or configured port."""

        with self._lock:
            if self._server is None:
                return self._configured_port

            return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """Return the active WebSocket URL."""

        return f"ws://{self._host}:{self.bound_port}{self._path}"

    @property
    def active_connection_count(self) -> int:
        """Return the number of currently connected clients."""

        with self._lock:
            return len(self._connections)

    @property
    def total_connection_count(self) -> int:
        """Return the total number of successful handshakes."""

        with self._lock:
            return self._total_connections

    @property
    def message_count(self) -> int:
        """Return the number of application messages received."""

        with self._lock:
            return self._message_count

    @property
    def command_count(self) -> int:
        """Return the number of allowlisted commands executed."""

        with self._lock:
            return self._command_count

    def is_running(self) -> bool:
        """Return whether the listener worker is active."""

        with self._lock:
            return bool(
                self._server is not None
                and self._thread is not None
                and self._thread.is_alive()
            )

    def start(self) -> bool:
        """Start the command server in a daemon worker thread."""

        with self._lock:
            if self.is_running():
                return False

            handler_class = self._make_handler()
            server = _ThreadingWebSocketServer(
                (self._host, self._configured_port),
                handler_class,
            )

            thread = Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="ghostfire-websocket-command-server",
                daemon=True,
            )

            self._server = server
            self._thread = thread
            thread.start()

        self._publish(
            "ghostfire.websocket.started",
            {
                "host": self._host,
                "port": self.bound_port,
                "path": self._path,
                "authenticated": self._auth_token is not None,
                "allowed_commands": list(self._allowed_commands),
            },
        )

        return True

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop the listener and close active client sockets."""

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
        ):
            raise TypeError("timeout must be numeric")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        with self._lock:
            server = self._server
            thread = self._thread
            connections = tuple(self._connections)

            if server is None:
                return False

        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                connection.close()
            except OSError:
                pass

        server.shutdown()
        server.server_close()

        if thread is not None:
            thread.join(timeout=float(timeout))

        with self._lock:
            self._server = None
            self._thread = None
            self._connections.clear()

        self._publish(
            "ghostfire.websocket.stopped",
            {
                "host": self._host,
                "port": self._configured_port,
            },
        )

        return True

    def _make_handler(
        self,
    ) -> type[StreamRequestHandler]:
        owner = self

        class Handler(StreamRequestHandler):
            def handle(self) -> None:
                owner._handle_connection(
                    self.request,
                    self.rfile,
                    self.wfile,
                )

        return Handler

    def _handle_connection(
        self,
        connection: socket.socket,
        reader: Any,
        writer: Any,
    ) -> None:
        connection.settimeout(self._idle_timeout)
        connected = False
        connection_id = secrets.token_hex(8)

        try:
            request_line = reader.readline(8193)

            if not request_line:
                return

            if len(request_line) > 8192:
                self._send_http_error(
                    writer,
                    431,
                    "Request Header Fields Too Large",
                )
                return

            try:
                method, target, version = (
                    request_line.decode("ascii").strip().split(" ", 2)
                )
            except (UnicodeDecodeError, ValueError):
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            headers = self._read_headers(reader)
            path = urlsplit(target).path

            if method != "GET" or version != "HTTP/1.1":
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            if path != self._path:
                self._send_http_error(
                    writer,
                    404,
                    "Not Found",
                )
                return

            if not self._authorized(headers):
                self._send_http_error(
                    writer,
                    401,
                    "Unauthorized",
                    extra_headers={
                        "WWW-Authenticate": "Bearer"
                    },
                )
                return

            if headers.get("upgrade", "").lower() != "websocket":
                self._send_http_error(
                    writer,
                    426,
                    "Upgrade Required",
                    extra_headers={"Upgrade": "websocket"},
                )
                return

            connection_tokens = {
                token.strip().lower()
                for token in headers.get(
                    "connection",
                    "",
                ).split(",")
            }

            if "upgrade" not in connection_tokens:
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            if headers.get("sec-websocket-version") != "13":
                self._send_http_error(
                    writer,
                    426,
                    "Upgrade Required",
                    extra_headers={
                        "Sec-WebSocket-Version": "13"
                    },
                )
                return

            websocket_key = headers.get("sec-websocket-key")

            if not websocket_key:
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            try:
                decoded_key = base64.b64decode(
                    websocket_key,
                    validate=True,
                )
            except ValueError:
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            if len(decoded_key) != 16:
                self._send_http_error(
                    writer,
                    400,
                    "Bad Request",
                )
                return

            accept = base64.b64encode(
                hashlib.sha1(
                    (
                        websocket_key
                        + _WEBSOCKET_GUID
                    ).encode("ascii")
                ).digest()
            ).decode("ascii")

            writer.write(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "Cache-Control: no-store\r\n"
                    "\r\n"
                ).encode("ascii")
            )
            writer.flush()

            with self._lock:
                self._connections.add(connection)
                self._total_connections += 1

            connected = True

            self._publish(
                "ghostfire.websocket.connected",
                {
                    "connection_id": connection_id,
                    "active_connections": (
                        self.active_connection_count
                    ),
                },
            )

            while True:
                frame = self._read_frame(reader)

                if frame is None:
                    break

                opcode, payload = frame

                if opcode == 0x8:
                    self._send_frame(writer, 0x8, payload[:125])
                    break

                if opcode == 0x9:
                    self._send_frame(writer, 0xA, payload)
                    continue

                if opcode == 0xA:
                    continue

                if opcode == 0x2:
                    self._send_error(
                        writer,
                        request_id=None,
                        code="binary_not_supported",
                        message="binary messages are not supported",
                    )
                    self._send_close(
                        writer,
                        1003,
                        "binary messages are not supported",
                    )
                    break

                if opcode != 0x1:
                    self._send_close(
                        writer,
                        1002,
                        "unsupported opcode",
                    )
                    break

                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    self._send_close(
                        writer,
                        1007,
                        "invalid UTF-8",
                    )
                    break

                with self._lock:
                    self._message_count += 1

                self._handle_message(
                    writer,
                    text,
                    connection_id=connection_id,
                )
        except WebSocketMessageTooLarge:
            if connected:
                self._send_close(
                    writer,
                    1009,
                    "message too large",
                )
        except WebSocketProtocolError as exc:
            if connected:
                self._send_close(
                    writer,
                    1002,
                    str(exc),
                )
        except (
            ConnectionError,
            OSError,
            TimeoutError,
            socket.timeout,
        ):
            pass
        finally:
            if connected:
                with self._lock:
                    self._connections.discard(connection)

                self._publish(
                    "ghostfire.websocket.disconnected",
                    {
                        "connection_id": connection_id,
                        "active_connections": (
                            self.active_connection_count
                        ),
                    },
                )

    def _handle_message(
        self,
        writer: Any,
        text: str,
        *,
        connection_id: str,
    ) -> None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            self._send_error(
                writer,
                request_id=None,
                code="invalid_json",
                message="message must be valid JSON",
            )
            return

        if not isinstance(message, dict):
            self._send_error(
                writer,
                request_id=None,
                code="invalid_message",
                message="message root must be an object",
            )
            return

        request_id = message.get("id")

        if request_id is None:
            request_id = secrets.token_hex(8)

        message_type = message.get("type")

        if message_type == "ping":
            self._send_json(
                writer,
                {
                    "id": request_id,
                    "type": "pong",
                    "status": "ok",
                },
            )
            return

        if message_type == "status":
            try:
                status = deepcopy(
                    dict(self._status_provider())
                )
            except Exception:
                self._send_error(
                    writer,
                    request_id=request_id,
                    code="status_failed",
                    message="status could not be collected",
                )
                return

            self._send_json(
                writer,
                {
                    "id": request_id,
                    "type": "status",
                    "status": "ok",
                    "data": status,
                },
            )
            return

        if message_type != "command":
            self._send_error(
                writer,
                request_id=request_id,
                code="unsupported_type",
                message="type must be ping, status, or command",
            )
            return

        command_value = message.get("command")

        try:
            command = self._normalize_command(command_value)
        except (TypeError, ValueError):
            self._send_error(
                writer,
                request_id=request_id,
                code="invalid_command",
                message="command must be a non-empty string",
            )
            return

        if command not in self._allowed_commands:
            self._send_error(
                writer,
                request_id=request_id,
                code="command_not_allowed",
                message=f"command is not allowlisted: {command}",
            )
            return

        self._publish(
            "ghostfire.websocket.command.received",
            {
                "connection_id": connection_id,
                "request_id": request_id,
                "command": command,
            },
        )

        try:
            result = self._command_handler(command)
        except Exception as exc:
            self._publish(
                "ghostfire.websocket.command.failed",
                {
                    "connection_id": connection_id,
                    "request_id": request_id,
                    "command": command,
                    "error_type": type(exc).__name__,
                },
            )

            self._send_error(
                writer,
                request_id=request_id,
                code="command_failed",
                message="command execution failed",
            )
            return

        with self._lock:
            self._command_count += 1

        self._publish(
            "ghostfire.websocket.command.completed",
            {
                "connection_id": connection_id,
                "request_id": request_id,
                "command": command,
            },
        )

        self._send_json(
            writer,
            {
                "id": request_id,
                "type": "command_result",
                "status": "ok",
                "command": command,
                "data": result,
            },
        )

    def _read_frame(
        self,
        reader: Any,
    ) -> tuple[int, bytes] | None:
        first = reader.read(2)

        if not first:
            return None

        if len(first) != 2:
            raise WebSocketProtocolError(
                "incomplete frame header"
            )

        first_byte, second_byte = first
        final = bool(first_byte & 0x80)
        reserved = first_byte & 0x70
        opcode = first_byte & 0x0F
        masked = bool(second_byte & 0x80)
        payload_length = second_byte & 0x7F

        if reserved:
            raise WebSocketProtocolError(
                "reserved bits are not supported"
            )

        if not final:
            raise WebSocketProtocolError(
                "fragmented frames are not supported"
            )

        if not masked:
            raise WebSocketProtocolError(
                "client frames must be masked"
            )

        if payload_length == 126:
            payload_length = unpack(
                "!H",
                self._read_exact(reader, 2),
            )[0]
        elif payload_length == 127:
            extended = self._read_exact(reader, 8)

            if extended[0] & 0x80:
                raise WebSocketProtocolError(
                    "invalid 64-bit payload length"
                )

            payload_length = unpack("!Q", extended)[0]

        if opcode >= 0x8:
            if payload_length > 125:
                raise WebSocketProtocolError(
                    "control frame payload is too large"
                )
        elif payload_length > self._max_message_bytes:
            raise WebSocketMessageTooLarge(
                "message exceeds configured limit"
            )

        mask = self._read_exact(reader, 4)
        payload = self._read_exact(reader, payload_length)

        unmasked = bytes(
            value ^ mask[index % 4]
            for index, value in enumerate(payload)
        )

        return opcode, unmasked

    @staticmethod
    def _read_exact(
        reader: Any,
        length: int,
    ) -> bytes:
        data = reader.read(length)

        if len(data) != length:
            raise ConnectionError(
                "connection closed during frame read"
            )

        return data

    def _send_json(
        self,
        writer: Any,
        payload: Mapping[str, Any],
    ) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_default,
        ).encode("utf-8")

        self._send_frame(writer, 0x1, encoded)

    def _send_error(
        self,
        writer: Any,
        *,
        request_id: Any,
        code: str,
        message: str,
    ) -> None:
        self._send_json(
            writer,
            {
                "id": request_id,
                "type": "error",
                "status": "error",
                "error": code,
                "message": message,
            },
        )

    @staticmethod
    def _send_frame(
        writer: Any,
        opcode: int,
        payload: bytes,
    ) -> None:
        first_byte = 0x80 | opcode
        length = len(payload)

        if length <= 125:
            header = bytes((first_byte, length))
        elif length <= 65_535:
            header = bytes((first_byte, 126)) + pack(
                "!H",
                length,
            )
        else:
            header = bytes((first_byte, 127)) + pack(
                "!Q",
                length,
            )

        writer.write(header + payload)
        writer.flush()

    def _send_close(
        self,
        writer: Any,
        code: int,
        reason: str,
    ) -> None:
        reason_bytes = reason.encode("utf-8")[:123]
        payload = pack("!H", code) + reason_bytes

        try:
            self._send_frame(writer, 0x8, payload)
        except OSError:
            pass

    @staticmethod
    def _read_headers(reader: Any) -> dict[str, str]:
        headers: dict[str, str] = {}

        for _ in range(100):
            line = reader.readline(8193)

            if not line:
                raise ConnectionError(
                    "connection closed during handshake"
                )

            if len(line) > 8192:
                raise WebSocketProtocolError(
                    "header line is too large"
                )

            if line in {b"\r\n", b"\n"}:
                return headers

            try:
                decoded = line.decode("ascii")
            except UnicodeDecodeError as exc:
                raise WebSocketProtocolError(
                    "headers must be ASCII"
                ) from exc

            if ":" not in decoded:
                raise WebSocketProtocolError(
                    "malformed HTTP header"
                )

            name, value = decoded.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        raise WebSocketProtocolError(
            "too many HTTP headers"
        )

    @staticmethod
    def _send_http_error(
        writer: Any,
        status: int,
        reason: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            {
                "error": reason.lower().replace(" ", "_"),
                "status": status,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        lines = [
            f"HTTP/1.1 {status} {reason}",
            "Content-Type: application/json; charset=utf-8",
            f"Content-Length: {len(body)}",
            "Cache-Control: no-store",
            "Connection: close",
        ]

        if extra_headers is not None:
            lines.extend(
                f"{name}: {value}"
                for name, value in extra_headers.items()
            )

        response = (
            "\r\n".join(lines).encode("ascii")
            + b"\r\n\r\n"
            + body
        )

        writer.write(response)
        writer.flush()

    def _authorized(
        self,
        headers: Mapping[str, str],
    ) -> bool:
        if self._auth_token is None:
            return True

        authorization = headers.get("authorization", "")
        prefix = "Bearer "

        if not authorization.startswith(prefix):
            return False

        candidate = authorization[len(prefix):]

        return hmac.compare_digest(
            candidate,
            self._auth_token,
        )

    def _publish(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        if self._event_bus is None:
            return

        self._event_bus.emit(
            event_name,
            payload,
            raise_exceptions=False,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value

        return str(value)

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host.lower() == "localhost":
            return True

        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    @classmethod
    def _normalize_command(cls, command: Any) -> str:
        normalized = cls._validate_text(
            command,
            field_name="command",
        )
        return normalized.upper()

    @staticmethod
    def _validate_text(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        return normalized
