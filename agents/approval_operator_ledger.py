"""Read-only loopback client for the owner-operation ledger query API."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class OwnerOperationLedgerQueryError(RuntimeError):
    """Raised when the local owner-operation ledger query API is unsafe or unavailable."""


class OwnerOperationLedgerQueryClient:
    """Strict GET-only client for the loopback owner-operation ledger API."""

    _REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
    _FORBIDDEN_KEYS = frozenset(
        {
            "owner_token",
            "protected_nonce",
            "protected_value",
            "password",
            "secret",
            "token",
            "path",
        }
    )
    _MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8791",
        *,
        timeout: float = 3.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")

        if timeout <= 0 or timeout > 30:
            raise ValueError("timeout must be greater than 0 and at most 30 seconds")

        parsed = urlsplit(base_url.strip())

        if parsed.scheme.lower() != "http":
            raise ValueError("ledger API must use http on loopback")

        if parsed.hostname != "127.0.0.1":
            raise ValueError("ledger API must use the exact host 127.0.0.1")

        if parsed.username is not None or parsed.password is not None:
            raise ValueError("ledger API URL must not contain credentials")

        if parsed.path not in {"", "/"}:
            raise ValueError("ledger API URL must not contain a path")

        if parsed.query or parsed.fragment:
            raise ValueError("ledger API URL must not contain query or fragment data")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("ledger API URL contains an invalid port") from exc

        if port is None:
            port = 80

        if port < 1024 or port > 65535:
            raise ValueError("ledger API port must be between 1024 and 65535")

        self._base_url = f"http://127.0.0.1:{port}"
        self._timeout = float(timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def read_only(self) -> bool:
        return True

    @property
    def loopback_only(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def list_operations(self) -> dict[str, Any]:
        return self._request("/v1/owner-operations")

    def latest_operation(self) -> dict[str, Any]:
        return self._request("/v1/owner-operations/latest")

    def get_operation(self, request_id: str) -> dict[str, Any]:
        normalized = self._validate_request_id(request_id)
        encoded = quote(normalized, safe="")
        return self._request(f"/v1/owner-operations/{encoded}")

    def history(self) -> dict[str, Any]:
        return self._request("/v1/owner-operations/history")

    def verify(self) -> dict[str, Any]:
        return self._request("/v1/owner-operations/verify")

    def snapshot(self) -> dict[str, Any]:
        health = self.health()
        verification = self.verify()

        return {
            "status": "ok",
            "service": deepcopy(health),
            "verification": deepcopy(verification),
            "read_only": True,
            "loopback_only": True,
            "action_executed": False,
            "secret_exposed": False,
        }

    def _validate_request_id(self, request_id: str) -> str:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")

        normalized = request_id.strip()

        if not self._REQUEST_ID.fullmatch(normalized):
            raise ValueError(
                "request_id must contain only letters, numbers, hyphen, or underscore"
            )

        return normalized

    def _request(self, path: str) -> dict[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise OwnerOperationLedgerQueryError("unsafe ledger API path")

        request = Request(
            self._base_url + path,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-store",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise OwnerOperationLedgerQueryError(
                f"ledger API returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OwnerOperationLedgerQueryError(
                "ledger API is unavailable"
            ) from exc

        if len(raw) > self._MAX_RESPONSE_BYTES:
            raise OwnerOperationLedgerQueryError(
                "ledger API response exceeded the safety limit"
            )

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OwnerOperationLedgerQueryError(
                "ledger API returned invalid JSON"
            ) from exc

        if not isinstance(payload, Mapping):
            raise OwnerOperationLedgerQueryError(
                "ledger API response must be a JSON object"
            )

        copied = deepcopy(dict(payload))
        self._validate_safe_payload(copied)
        return copied

    def _validate_safe_payload(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized_key = str(key).strip().lower()

                if normalized_key in self._FORBIDDEN_KEYS:
                    raise OwnerOperationLedgerQueryError(
                        "ledger API response contained a protected field"
                    )

                if normalized_key == "secret_exposed" and item is not False:
                    raise OwnerOperationLedgerQueryError(
                        "ledger API reported secret exposure"
                    )

                if normalized_key == "action_executed" and item is not False:
                    raise OwnerOperationLedgerQueryError(
                        "ledger API reported action execution"
                    )

                self._validate_safe_payload(item)

        elif isinstance(value, list):
            for item in value:
                self._validate_safe_payload(item)
