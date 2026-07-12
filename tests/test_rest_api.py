import concurrent.futures
import json
import time
import unittest
import urllib.error
import urllib.request

from api.rest import (
    RestApiSecurityError,
    RestApiServer,
)
from core.eventbus import EventBus
from core.scheduler import Scheduler
from core.service_manager import ServiceManager


class RestApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.servers: list[RestApiServer] = []

    def tearDown(self) -> None:
        for server in reversed(self.servers):
            server.stop()

    def make_server(
        self,
        *,
        auth_token: str | None = None,
        event_bus: EventBus | None = None,
        dashboard_provider=None,
    ) -> tuple[
        RestApiServer,
        ServiceManager,
        Scheduler,
    ]:
        manager = ServiceManager(event_bus=event_bus)
        scheduler = Scheduler(event_bus=event_bus)

        manager.register(
            "runtime",
            lambda: None,
            health=lambda: True,
        )
        manager.start_all()

        server = RestApiServer(
            app_name="Ghostfire OS",
            version="0.2.0",
            configuration_revision=3,
            configuration_sources=(
                "defaults",
                "environment",
            ),
            configuration={
                "app_name": "Ghostfire OS",
                "database": {
                    "password": "***REDACTED***",
                },
            },
            service_manager=manager,
            scheduler=scheduler,
            host="127.0.0.1",
            port=0,
            auth_token=auth_token,
            dashboard_provider=dashboard_provider,
            event_bus=event_bus,
            request_timeout=0.1,
        )
        self.servers.append(server)

        return server, manager, scheduler

    @staticmethod
    def request(
        server: RestApiServer,
        path: str,
        *,
        token: str | None = None,
        method: str = "GET",
    ) -> tuple[int, dict, dict]:
        headers = {"Connection": "close"}

        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(
            server.base_url + path,
            method=method,
            headers=headers,
        )

        for attempt in range(6):
            response = None

            try:
                try:
                    response = urllib.request.urlopen(
                        request,
                        timeout=2,
                    )
                except urllib.error.HTTPError as exc:
                    response = exc

                body = response.read()
                status = response.status
                response_headers = dict(
                    response.headers.items()
                )

                payload = (
                    json.loads(body.decode("utf-8"))
                    if body
                    else {}
                )

                return (
                    status,
                    payload,
                    response_headers,
                )
            except urllib.error.URLError as exc:
                winerror = getattr(
                    exc.reason,
                    "winerror",
                    None,
                )

                if winerror != 10048 or attempt == 5:
                    raise

                time.sleep(0.1 * (2 ** attempt))
            finally:
                if response is not None:
                    response.close()

        raise AssertionError("HTTP request retry loop exhausted")

    def test_non_loopback_requires_authentication(self) -> None:
        manager = ServiceManager()
        scheduler = Scheduler()

        with self.assertRaises(RestApiSecurityError):
            RestApiServer(
                app_name="Ghostfire OS",
                version="0.2.0",
                configuration_revision=1,
                configuration_sources=("defaults",),
                configuration={},
                service_manager=manager,
                scheduler=scheduler,
                host="0.0.0.0",
                port=8102,
            )

    def test_start_and_stop_are_idempotent(self) -> None:
        server, _, _ = self.make_server()

        self.assertTrue(server.start())
        self.assertFalse(server.start())
        self.assertTrue(server.is_running())
        self.assertTrue(server.stop())
        self.assertFalse(server.stop())
        self.assertFalse(server.is_running())

    def test_responses_close_connections(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, _, headers = self.request(
            server,
            "/health",
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            headers["Connection"].lower(),
            "close",
        )

    def test_health_endpoint_is_public(self) -> None:
        server, _, _ = self.make_server(
            auth_token="private-token"
        )
        server.start()

        status, payload, headers = self.request(
            server,
            "/health",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(
            payload["service"],
            "ghostfire-rest-api",
        )
        self.assertIn("X-Request-ID", headers)

    def test_status_endpoint_reports_runtime(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/status",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["app_name"], "Ghostfire OS")
        self.assertEqual(
            payload["configuration_revision"],
            3,
        )
        self.assertEqual(payload["services"]["total"], 1)
        self.assertEqual(
            payload["api"]["port"],
            server.bound_port,
        )

    def test_services_endpoint_reports_health(self) -> None:
        server, manager, _ = self.make_server()
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/services",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(
            payload["services"][0]["name"],
            "runtime",
        )
        self.assertTrue(
            payload["services"][0]["healthy"]
        )
        self.assertTrue(manager.is_running("runtime"))

    def test_configuration_endpoint_preserves_redaction(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/configuration",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["revision"], 3)
        self.assertEqual(
            payload["values"]["database"]["password"],
            "***REDACTED***",
        )

    def test_dashboard_endpoint_uses_provider(self) -> None:
        server, _, _ = self.make_server(
            dashboard_provider=lambda: {
                "overall_state": "online",
                "service_count": 1,
            }
        )
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/dashboard",
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["overall_state"],
            "online",
        )

    def test_dashboard_unavailable_returns_503(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/dashboard",
        )

        self.assertEqual(status, 503)
        self.assertEqual(
            payload["error"],
            "dashboard_unavailable",
        )

    def test_openapi_document_is_public(self) -> None:
        server, _, _ = self.make_server(
            auth_token="private-token"
        )
        server.start()

        status, payload, _ = self.request(
            server,
            "/v1/openapi.json",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["openapi"], "3.1.0")
        self.assertIn("/v1/status", payload["paths"])

    def test_unknown_route_returns_json_404(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, payload, _ = self.request(
            server,
            "/missing",
        )

        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")

    def test_bearer_token_protects_operational_routes(self) -> None:
        server, _, _ = self.make_server(
            auth_token="private-token"
        )
        server.start()

        denied, payload, headers = self.request(
            server,
            "/v1/status",
        )
        allowed, _, _ = self.request(
            server,
            "/v1/status",
            token="private-token",
        )

        self.assertEqual(denied, 401)
        self.assertEqual(payload["error"], "unauthorized")
        self.assertEqual(
            headers["WWW-Authenticate"],
            "Bearer",
        )
        self.assertEqual(allowed, 200)

    def test_mutating_methods_are_rejected(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        status, payload, headers = self.request(
            server,
            "/v1/status",
            method="POST",
        )

        self.assertEqual(status, 405)
        self.assertEqual(
            payload["error"],
            "method_not_allowed",
        )
        self.assertEqual(
            headers["Allow"],
            "GET, HEAD, OPTIONS",
        )

    def test_event_bus_receives_lifecycle_and_request_events(
        self,
    ) -> None:
        event_bus = EventBus()
        events: list[str] = []

        event_bus.subscribe(
            EventBus.WILDCARD,
            lambda event: events.append(event.name),
        )

        server, _, _ = self.make_server(
            event_bus=event_bus
        )
        server.start()
        self.request(server, "/health")
        server.stop()

        self.assertIn(
            "ghostfire.rest_api.started",
            events,
        )
        self.assertIn(
            "ghostfire.rest_api.request",
            events,
        )
        self.assertIn(
            "ghostfire.rest_api.stopped",
            events,
        )

    def test_concurrent_health_requests_are_safe(self) -> None:
        server, _, _ = self.make_server()
        server.start()

        def get_health(_: int) -> int:
            status, _, _ = self.request(
                server,
                "/health",
            )
            return status

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            statuses = list(
                executor.map(get_health, range(16))
            )

        self.assertEqual(statuses, [200] * 16)
        self.assertEqual(server.request_count, 16)

    def test_service_manager_can_manage_rest_api(self) -> None:
        event_bus = EventBus()
        manager = ServiceManager(event_bus=event_bus)
        scheduler = Scheduler(event_bus=event_bus)

        manager.register("runtime", lambda: None)

        server = RestApiServer(
            app_name="Ghostfire OS",
            version="0.2.0",
            configuration_revision=1,
            configuration_sources=("defaults",),
            configuration={},
            service_manager=manager,
            scheduler=scheduler,
            port=0,
            event_bus=event_bus,
        )
        self.servers.append(server)

        manager.register(
            "rest_api",
            server.start,
            stop=server.stop,
            dependencies=("runtime",),
            health=server.is_running,
        )

        manager.start_all()

        self.assertTrue(server.is_running())
        self.assertTrue(manager.check_health("rest_api"))

        manager.stop_all()

        self.assertFalse(server.is_running())


if __name__ == "__main__":
    unittest.main()
