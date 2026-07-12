"""Read-only REST API for GhostFire OS operational state."""

from __future__ import annotations

import hmac
import json
import secrets
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from threading import RLock, Thread
from typing import Any
from urllib.parse import urlsplit

from core.eventbus import EventBus
from core.scheduler import Scheduler
from core.service_manager import ServiceManager


DashboardProvider = Callable[[], Mapping[str, Any] | None]


class RestApiError(RuntimeError):
    """Base class for REST API subsystem failures."""


class RestApiSecurityError(RestApiError):
    """Raised when network exposure is unsafe."""


class _GhostFireHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RestApiServer:
    """
    Thread-safe, dependency-free HTTP interface for runtime observability.

    The R1 API is intentionally read-only. Command execution belongs to the
    dedicated command-server layer rather than this operational status API.
    """

    def __init__(
        self,
        *,
        app_name: str,
        version: str,
        configuration_revision: int,
        configuration_sources: tuple[str, ...] | list[str],
        configuration: Mapping[str, Any],
        service_manager: ServiceManager,
        scheduler: Scheduler,
        host: str = "127.0.0.1",
        port: int = 8102,
        auth_token: str | None = None,
        dashboard_provider: DashboardProvider | None = None,
        event_bus: EventBus | None = None,
        request_timeout: float = 2.0,
    ) -> None:
        self._app_name = self._validate_text(
            app_name,
            field_name="app_name",
        )
        self._version = self._validate_text(
            version,
            field_name="version",
        )
        self._host = self._validate_text(
            host,
            field_name="host",
        )

        if (
            isinstance(configuration_revision, bool)
            or not isinstance(configuration_revision, int)
        ):
            raise TypeError(
                "configuration_revision must be an integer"
            )

        if configuration_revision < 1:
            raise ValueError(
                "configuration_revision must be positive"
            )

        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError("port must be an integer")

        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")

        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
        ):
            raise TypeError(
                "request_timeout must be numeric"
            )

        if request_timeout <= 0:
            raise ValueError(
                "request_timeout must be positive"
            )

        if not isinstance(configuration, Mapping):
            raise TypeError("configuration must be a mapping")

        if not isinstance(service_manager, ServiceManager):
            raise TypeError(
                "service_manager must be a ServiceManager"
            )

        if not isinstance(scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")

        if dashboard_provider is not None and not callable(
            dashboard_provider
        ):
            raise TypeError(
                "dashboard_provider must be callable or None"
            )

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        if auth_token is not None:
            auth_token = self._validate_text(
                auth_token,
                field_name="auth_token",
            )

        if not self._is_loopback_host(self._host) and not auth_token:
            raise RestApiSecurityError(
                "non-loopback REST API binding requires auth_token"
            )

        self._configuration_revision = configuration_revision
        self._configuration_sources = tuple(
            self._validate_text(
                source,
                field_name="configuration source",
            )
            for source in configuration_sources
        )
        self._configuration = deepcopy(dict(configuration))
        self._service_manager = service_manager
        self._scheduler = scheduler
        self._configured_port = port
        self._auth_token = auth_token
        self._dashboard_provider = dashboard_provider
        self._event_bus = event_bus
        self._request_timeout = float(request_timeout)
        self._lock = RLock()
        self._server: _GhostFireHttpServer | None = None
        self._thread: Thread | None = None
        self._request_count = 0

    @property
    def request_count(self) -> int:
        """Return the number of requests received."""

        with self._lock:
            return self._request_count

    @property
    def bound_port(self) -> int:
        """Return the active bound port or configured port."""

        with self._lock:
            if self._server is None:
                return self._configured_port

            return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL."""

        return f"http://{self._host}:{self.bound_port}"

    def is_running(self) -> bool:
        """Return whether the server worker is active."""

        with self._lock:
            return bool(
                self._server is not None
                and self._thread is not None
                and self._thread.is_alive()
            )

    def start(self) -> bool:
        """Start the REST API in a daemon worker thread."""

        with self._lock:
            if self.is_running():
                return False

            handler_class = self._make_handler()
            server = _GhostFireHttpServer(
                (self._host, self._configured_port),
                handler_class,
            )
            server.timeout = self._request_timeout

            thread = Thread(
                target=server.serve_forever,
                kwargs={
                    "poll_interval": min(
                        0.5,
                        self._request_timeout,
                    )
                },
                name="ghostfire-rest-api",
                daemon=True,
            )

            self._server = server
            self._thread = thread
            thread.start()

        self._publish(
            "ghostfire.rest_api.started",
            {
                "host": self._host,
                "port": self.bound_port,
                "authenticated": self._auth_token is not None,
            },
        )

        return True

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop the server and release its listening socket."""

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

            if server is None:
                return False

        server.shutdown()
        server.server_close()

        if thread is not None:
            thread.join(timeout=float(timeout))

        with self._lock:
            self._server = None
            self._thread = None

        self._publish(
            "ghostfire.rest_api.stopped",
            {
                "host": self._host,
                "port": self._configured_port,
            },
        )

        return True

    def _make_handler(
        self,
    ) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "GhostFireREST/1"

            def do_GET(self) -> None:
                owner._handle_get(self)

            def do_HEAD(self) -> None:
                owner._handle_get(self, head_only=True)

            def do_OPTIONS(self) -> None:
                owner._handle_options(self)

            def do_POST(self) -> None:
                owner._send_json(
                    self,
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {
                        "error": "method_not_allowed",
                        "message": "R1 REST API is read-only",
                    },
                    extra_headers={"Allow": "GET, HEAD, OPTIONS"},
                )

            do_PUT = do_POST
            do_PATCH = do_POST
            do_DELETE = do_POST

            def log_message(
                self,
                format: str,
                *args: Any,
            ) -> None:
                return

        return Handler

    def _handle_get(
        self,
        handler: BaseHTTPRequestHandler,
        *,
        head_only: bool = False,
    ) -> None:
        request_id = secrets.token_hex(8)
        path = urlsplit(handler.path).path

        with self._lock:
            self._request_count += 1

        try:
            if (
                path not in {"/health", "/v1/openapi.json"}
                and not self._authorized(handler)
            ):
                self._send_json(
                    handler,
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "error": "unauthorized",
                        "message": "valid bearer token required",
                    },
                    request_id=request_id,
                    extra_headers={
                        "WWW-Authenticate": "Bearer"
                    },
                    head_only=head_only,
                )
                return

            status, payload = self._route(path)

            self._send_json(
                handler,
                status,
                payload,
                request_id=request_id,
                head_only=head_only,
            )
        except Exception:
            self._send_json(
                handler,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "internal_server_error",
                    "message": "request could not be completed",
                },
                request_id=request_id,
                head_only=head_only,
            )
            status = HTTPStatus.INTERNAL_SERVER_ERROR

        self._publish(
            "ghostfire.rest_api.request",
            {
                "method": handler.command,
                "path": path,
                "status": int(status),
                "request_id": request_id,
            },
        )

    def _handle_options(
        self,
        handler: BaseHTTPRequestHandler,
    ) -> None:
        handler.send_response(HTTPStatus.NO_CONTENT)
        handler.send_header("Allow", "GET, HEAD, OPTIONS")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _route(
        self,
        path: str,
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        if path == "/health":
            return (
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "ghostfire-rest-api",
                    "version": self._version,
                },
            )

        if path == "/v1/status":
            return HTTPStatus.OK, self._status_payload()

        if path == "/v1/services":
            return HTTPStatus.OK, self._services_payload()

        if path == "/v1/configuration":
            return (
                HTTPStatus.OK,
                {
                    "revision": self._configuration_revision,
                    "sources": list(
                        self._configuration_sources
                    ),
                    "values": deepcopy(
                        self._configuration
                    ),
                },
            )

        if path == "/v1/dashboard":
            if self._dashboard_provider is None:
                return (
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "dashboard_unavailable",
                    },
                )

            dashboard = self._dashboard_provider()

            if dashboard is None:
                return (
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "dashboard_unavailable",
                    },
                )

            return HTTPStatus.OK, deepcopy(dict(dashboard))

        if path == "/v1/openapi.json":
            return HTTPStatus.OK, self._openapi_payload()

        return (
            HTTPStatus.NOT_FOUND,
            {
                "error": "not_found",
                "path": path,
            },
        )

    def _status_payload(self) -> dict[str, Any]:
        statuses = self._service_manager.list_statuses()
        running_count = sum(
            status.state.value == "running"
            for status in statuses
        )

        return {
            "app_name": self._app_name,
            "version": self._version,
            "status": (
                "online"
                if statuses and running_count == len(statuses)
                else "degraded"
            ),
            "configuration_revision": (
                self._configuration_revision
            ),
            "scheduler": {
                "running": self._scheduler.is_running,
                "scheduled_task_count": (
                    self._scheduler.task_count()
                ),
            },
            "services": {
                "total": len(statuses),
                "running": running_count,
            },
            "api": {
                "host": self._host,
                "port": self.bound_port,
                "request_count": self.request_count,
            },
            "captured_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def _services_payload(self) -> dict[str, Any]:
        services = []

        for status in self._service_manager.list_statuses():
            healthy = self._service_manager.check_health(
                status.name
            )
            current = self._service_manager.status(
                status.name
            )

            services.append(
                {
                    "name": current.name,
                    "state": current.state.value,
                    "healthy": healthy,
                    "dependencies": list(
                        current.dependencies
                    ),
                    "start_count": current.start_count,
                    "stop_count": current.stop_count,
                    "last_error": current.last_error,
                }
            )

        return {
            "count": len(services),
            "services": services,
        }

    def _openapi_payload(self) -> dict[str, Any]:
        return {
            "openapi": "3.1.0",
            "info": {
                "title": "GhostFire OS REST API",
                "version": "1.0.0",
            },
            "paths": {
                "/health": {"get": {}},
                "/v1/status": {"get": {}},
                "/v1/services": {"get": {}},
                "/v1/configuration": {"get": {}},
                "/v1/dashboard": {"get": {}},
                "/v1/openapi.json": {"get": {}},
            },
        }

    def _authorized(
        self,
        handler: BaseHTTPRequestHandler,
    ) -> bool:
        if self._auth_token is None:
            return True

        header = handler.headers.get("Authorization", "")
        prefix = "Bearer "

        if not header.startswith(prefix):
            return False

        candidate = header[len(prefix):]

        return hmac.compare_digest(
            candidate,
            self._auth_token,
        )

    def _send_json(
        self,
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        payload: Mapping[str, Any],
        *,
        request_id: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        head_only: bool = False,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_default,
        ).encode("utf-8")

        handler.send_response(status)
        handler.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        handler.send_header(
            "Content-Length",
            str(len(body)),
        )
        handler.send_header("Cache-Control", "no-store")
        handler.send_header(
            "X-Content-Type-Options",
            "nosniff",
        )

        if request_id is not None:
            handler.send_header(
                "X-Request-ID",
                request_id,
            )

        if extra_headers is not None:
            for name, value in extra_headers.items():
                handler.send_header(name, value)

        handler.end_headers()

        if not head_only:
            handler.wfile.write(body)

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
        if isinstance(value, Path):
            return str(value)

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

    @staticmethod
    def _validate_text(
        value: str,
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
