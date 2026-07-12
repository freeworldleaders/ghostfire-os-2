import base64
import concurrent.futures
import json
import os
import socket
import struct
import time
import unittest

from api.websocket import (
    WebSocketCommandServer,
    WebSocketSecurityError,
)
from core.eventbus import EventBus
from core.scheduler import Scheduler
from core.service_manager import ServiceManager


class RawWebSocketClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        path: str = "/v1/commands",
        token: str | None = None,
    ) -> None:
        self.socket = None

        for attempt in range(6):
            try:
                self.socket = socket.create_connection(
                    (host, port),
                    timeout=2,
                )
                break
            except OSError as exc:
                winerror = getattr(exc, "winerror", None)

                if winerror != 10048 or attempt == 5:
                    raise

                time.sleep(0.1 * (2 ** attempt))

        if self.socket is None:
            raise AssertionError(
                "WebSocket connection retry loop exhausted"
            )

        self.socket.settimeout(2)
        key = base64.b64encode(os.urandom(16)).decode("ascii")

        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]

        if token is not None:
            headers.append(f"Authorization: Bearer {token}")

        request = "\r\n".join(headers) + "\r\n\r\n"
        self.socket.sendall(request.encode("ascii"))
        self.handshake = self._read_headers()

    def close(self) -> None:
        try:
            self.send_frame(0x8, struct.pack("!H", 1000))
        except OSError:
            pass

        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            self.socket.close()
        except OSError:
            pass

    def send_json(self, payload: dict) -> None:
        self.send_frame(
            0x1,
            json.dumps(payload).encode("utf-8"),
        )

    def receive_json(self) -> dict:
        opcode, payload = self.receive_frame()
        if opcode != 0x1:
            raise AssertionError(
                f"expected text frame; received opcode {opcode}"
            )

        return json.loads(payload.decode("utf-8"))

    def send_frame(
        self,
        opcode: int,
        payload: bytes,
        *,
        masked: bool = True,
    ) -> None:
        first = 0x80 | opcode
        mask_bit = 0x80 if masked else 0
        length = len(payload)

        if length <= 125:
            header = bytes((first, mask_bit | length))
        elif length <= 65_535:
            header = (
                bytes((first, mask_bit | 126))
                + struct.pack("!H", length)
            )
        else:
            header = (
                bytes((first, mask_bit | 127))
                + struct.pack("!Q", length)
            )

        if masked:
            mask = b"\x01\x02\x03\x04"
            payload = bytes(
                value ^ mask[index % 4]
                for index, value in enumerate(payload)
            )
            frame = header + mask + payload
        else:
            frame = header + payload

        self.socket.sendall(frame)

    def receive_frame(self) -> tuple[int, bytes]:
        first = self._read_exact(2)
        first_byte, second_byte = first
        opcode = first_byte & 0x0F
        length = second_byte & 0x7F

        if length == 126:
            length = struct.unpack(
                "!H",
                self._read_exact(2),
            )[0]
        elif length == 127:
            length = struct.unpack(
                "!Q",
                self._read_exact(8),
            )[0]

        if second_byte & 0x80:
            mask = self._read_exact(4)
        else:
            mask = None

        payload = self._read_exact(length)

        if mask is not None:
            payload = bytes(
                value ^ mask[index % 4]
                for index, value in enumerate(payload)
            )

        return opcode, payload

    def _read_headers(self) -> bytes:
        data = bytearray()

        while b"\r\n\r\n" not in data:
            chunk = self.socket.recv(4096)

            if not chunk:
                break

            data.extend(chunk)

        return bytes(data)

    def _read_exact(self, length: int) -> bytes:
        data = bytearray()

        while len(data) < length:
            chunk = self.socket.recv(length - len(data))

            if not chunk:
                raise ConnectionError(
                    "connection closed before frame completed"
                )

            data.extend(chunk)

        return bytes(data)


class WebSocketCommandServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.servers: list[WebSocketCommandServer] = []
        self.clients: list[RawWebSocketClient] = []
        self.commands: list[str] = []

    def tearDown(self) -> None:
        for client in reversed(self.clients):
            client.close()

        for server in reversed(self.servers):
            server.stop()

    def make_server(
        self,
        *,
        auth_token: str | None = None,
        event_bus: EventBus | None = None,
        max_message_bytes: int = 65_536,
    ) -> WebSocketCommandServer:
        def command_handler(command: str):
            self.commands.append(command)
            return {"accepted": command}

        server = WebSocketCommandServer(
            command_handler=command_handler,
            status_provider=lambda: {
                "status": "online",
                "services": 2,
            },
            host="127.0.0.1",
            port=0,
            auth_token=auth_token,
            allowed_commands=("BOOT", "STATUS"),
            max_message_bytes=max_message_bytes,
            idle_timeout=2,
            event_bus=event_bus,
        )
        self.servers.append(server)
        return server

    def connect(
        self,
        server: WebSocketCommandServer,
        *,
        token: str | None = None,
        path: str = "/v1/commands",
    ) -> RawWebSocketClient:
        client = RawWebSocketClient(
            "127.0.0.1",
            server.bound_port,
            path=path,
            token=token,
        )
        self.clients.append(client)
        return client

    def test_non_loopback_requires_authentication(self) -> None:
        with self.assertRaises(WebSocketSecurityError):
            WebSocketCommandServer(
                command_handler=lambda command: None,
                status_provider=lambda: {},
                host="0.0.0.0",
                port=8103,
            )

    def test_start_and_stop_are_idempotent(self) -> None:
        server = self.make_server()

        self.assertTrue(server.start())
        self.assertFalse(server.start())
        self.assertTrue(server.is_running())
        self.assertTrue(server.stop())
        self.assertFalse(server.stop())
        self.assertFalse(server.is_running())

    def test_successful_handshake_returns_switching_protocols(
        self,
    ) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        self.assertIn(
            b"HTTP/1.1 101 Switching Protocols",
            client.handshake,
        )
        self.assertIn(
            b"Sec-WebSocket-Accept:",
            client.handshake,
        )

    def test_wrong_path_returns_404(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server, path="/wrong")

        self.assertIn(
            b"HTTP/1.1 404 Not Found",
            client.handshake,
        )

    def test_bearer_authentication_protects_handshake(self) -> None:
        server = self.make_server(
            auth_token="private-token"
        )
        server.start()

        denied = self.connect(server)
        allowed = self.connect(
            server,
            token="private-token",
        )

        self.assertIn(
            b"HTTP/1.1 401 Unauthorized",
            denied.handshake,
        )
        self.assertIn(
            b"HTTP/1.1 101 Switching Protocols",
            allowed.handshake,
        )

    def test_application_ping_returns_pong_message(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_json({"id": "p1", "type": "ping"})
        response = client.receive_json()

        self.assertEqual(response["id"], "p1")
        self.assertEqual(response["type"], "pong")
        self.assertEqual(response["status"], "ok")

    def test_status_message_uses_provider(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_json({"id": "s1", "type": "status"})
        response = client.receive_json()

        self.assertEqual(response["type"], "status")
        self.assertEqual(
            response["data"]["status"],
            "online",
        )
        self.assertEqual(
            response["data"]["services"],
            2,
        )

    def test_allowlisted_command_executes(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_json(
            {
                "id": "c1",
                "type": "command",
                "command": "boot",
            }
        )
        response = client.receive_json()

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["command"], "BOOT")
        self.assertEqual(self.commands, ["BOOT"])
        self.assertEqual(server.command_count, 1)

    def test_non_allowlisted_command_is_rejected(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_json(
            {
                "id": "c2",
                "type": "command",
                "command": "DELETE_ALL",
            }
        )
        response = client.receive_json()

        self.assertEqual(
            response["error"],
            "command_not_allowed",
        )
        self.assertEqual(self.commands, [])
        self.assertEqual(server.command_count, 0)

    def test_malformed_json_returns_structured_error(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_frame(0x1, b"{not-json")
        response = client.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(response["error"], "invalid_json")

    def test_binary_messages_are_rejected(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_frame(0x2, b"binary")
        response = client.receive_json()
        opcode, _ = client.receive_frame()

        self.assertEqual(
            response["error"],
            "binary_not_supported",
        )
        self.assertEqual(opcode, 0x8)

    def test_unmasked_client_frame_is_closed(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_frame(
            0x1,
            b'{"type":"ping"}',
            masked=False,
        )
        opcode, payload = client.receive_frame()

        self.assertEqual(opcode, 0x8)
        self.assertEqual(
            struct.unpack("!H", payload[:2])[0],
            1002,
        )

    def test_protocol_ping_receives_pong_frame(self) -> None:
        server = self.make_server()
        server.start()
        client = self.connect(server)

        client.send_frame(0x9, b"probe")
        opcode, payload = client.receive_frame()

        self.assertEqual(opcode, 0xA)
        self.assertEqual(payload, b"probe")

    def test_oversized_message_receives_close_1009(self) -> None:
        server = self.make_server(max_message_bytes=16)
        server.start()
        client = self.connect(server)

        client.send_frame(0x1, b"x" * 17)
        opcode, payload = client.receive_frame()

        self.assertEqual(opcode, 0x8)
        self.assertEqual(
            struct.unpack("!H", payload[:2])[0],
            1009,
        )

    def test_event_bus_receives_lifecycle_and_command_events(
        self,
    ) -> None:
        event_bus = EventBus()
        events: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        server = self.make_server(event_bus=event_bus)
        server.start()
        client = self.connect(server)

        client.send_json(
            {
                "id": "event-command",
                "type": "command",
                "command": "STATUS",
            }
        )
        client.receive_json()
        client.close()
        server.stop()

        self.assertIn("ghostfire.websocket.started", events)
        self.assertIn("ghostfire.websocket.connected", events)
        self.assertIn(
            "ghostfire.websocket.command.received",
            events,
        )
        self.assertIn(
            "ghostfire.websocket.command.completed",
            events,
        )
        self.assertIn("ghostfire.websocket.stopped", events)

    def test_concurrent_clients_are_isolated(self) -> None:
        server = self.make_server()
        server.start()

        def run_client(index: int) -> str:
            client = RawWebSocketClient(
                "127.0.0.1",
                server.bound_port,
            )

            try:
                client.send_json(
                    {
                        "id": str(index),
                        "type": "ping",
                    }
                )
                return client.receive_json()["id"]
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            responses = list(
                executor.map(run_client, range(8))
            )

        self.assertEqual(
            sorted(responses, key=int),
            [str(index) for index in range(8)],
        )
        self.assertEqual(server.message_count, 8)

    def test_service_manager_controls_server_lifecycle(self) -> None:
        event_bus = EventBus()
        manager = ServiceManager(event_bus=event_bus)
        scheduler = Scheduler(event_bus=event_bus)

        manager.register("runtime", lambda: None)
        manager.register(
            "scheduler",
            scheduler.start,
            stop=scheduler.stop,
            dependencies=("runtime",),
            health=lambda: scheduler.is_running,
        )

        server = WebSocketCommandServer(
            command_handler=lambda command: command,
            status_provider=lambda: {},
            port=0,
            event_bus=event_bus,
        )
        self.servers.append(server)

        manager.register(
            "websocket_command_server",
            server.start,
            stop=server.stop,
            dependencies=("runtime", "scheduler"),
            health=server.is_running,
        )

        manager.start_all()

        self.assertTrue(server.is_running())
        self.assertTrue(
            manager.check_health(
                "websocket_command_server"
            )
        )

        manager.stop_all()

        self.assertFalse(server.is_running())


if __name__ == "__main__":
    unittest.main()
