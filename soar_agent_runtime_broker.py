"""
soar_agent_runtime_broker.py
Client for the Tool Broker (RT-L01, L05, L06, L07; decision E10).

The runtime reads external data only through broker templates. This module
talks HTTP/JSON over mTLS to the broker and turns every outcome - success,
rejection, execution error, transport failure - into a typed result. It never
raises into the ReAct loop; the caller decides how a result becomes an
observation (2.3).

Wire names follow the broker specification (German); Python identifiers are
English (E7). The transport is injectable so everything is testable offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from soar_agent_runtime_utils import _scrub

# Rejection / error codes of the broker (TB-F16 plus AUDIT_UNAVAILABLE from the
# broker implementation) and the runtime's own transport codes.
BROKER_CODES = (
    "HALTED", "UNKNOWN_TEMPLATE", "NOT_ALLOWED", "CLIENT_NOT_ALLOWED", "INVALID_PARAM",
    "QUOTA_EXCEEDED", "TIMEOUT", "SOURCE_ERROR", "CATALOG_UNAVAILABLE", "AUDIT_UNAVAILABLE",
)
TRANSPORT_CODES = ("UNREACHABLE", "TIMEOUT", "BAD_RESPONSE")
ART_REJECTED = "abgelehnt"
ART_ERROR = "fehler"

# Transport: (method, url, params, json_body, timeout_s) -> (status_code, body)
# where body is the decoded JSON (dict/list) or raw text. Exceptions are
# mapped by the client; a TimeoutError means the deadline passed.
Transport = Callable[[str, str, dict | None, dict | None, float], tuple[int, Any]]


class BrokerConfigError(ValueError):
    """Invalid broker configuration; raised at construction, never during a run."""


@dataclass(frozen=True)
class BrokerConfig:
    url: str
    timeout_s: float = 30.0
    client_cert: str = ""
    client_key: str = ""
    ca: str = ""

    def validate(self) -> "BrokerConfig":
        url = (self.url or "").strip()
        parsed = urlparse(url)
        if not url or parsed.scheme.lower() != "https" or not parsed.hostname:
            raise BrokerConfigError(
                f"broker_url must be an https:// URL (mTLS, RT-L07), got {url!r}")
        if not self.client_cert or not self.client_key:
            raise BrokerConfigError(
                "broker_client_cert and broker_client_key are required (mTLS, RT-L07)")
        if not (float(self.timeout_s) > 0):
            raise BrokerConfigError("broker_timeout_s must be > 0")
        return self

    @property
    def base(self) -> str:
        return self.url.strip().rstrip("/")


@dataclass(frozen=True)
class ToolSpec:
    """One template as offered by GET /v1/tools (E10)."""
    vorlage_id: str
    version: int
    beschreibung: str
    parameter: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolsResponse:
    ok: bool
    katalog_version: str = ""
    angehalten: bool = False
    vorlagen: tuple[ToolSpec, ...] = ()
    code: str = ""
    detail: str = ""


@dataclass(frozen=True)
class InvokeResult:
    """Outcome of POST /v1/invoke. ok=True: data; otherwise art/code/detail."""
    ok: bool
    aufruf_id: str = ""
    vorlage_id: str = ""
    vorlage_version: int | None = None
    katalog_version: str = ""
    zeilen: int = 0
    gekuerzt: bool = False
    ergebnis_hash: str = ""
    daten: Any = None
    art: str = ""
    code: str = ""
    detail: str = ""

    @property
    def halted(self) -> bool:
        return self.code == "HALTED"

    @property
    def data_gap(self) -> bool:
        """RT-L06: broker unreachable or source failed -> data gap, never a fallback."""
        return (not self.ok) and self.code in ("UNREACHABLE", "TIMEOUT", "BAD_RESPONSE", "SOURCE_ERROR",
                                               "CATALOG_UNAVAILABLE", "AUDIT_UNAVAILABLE")


def requests_transport(config: BrokerConfig) -> Transport:
    """Default transport: requests with mTLS, no redirects, no environment proxies."""
    import requests  # bundled wheel; imported lazily so the module loads without it

    session = requests.Session()
    session.trust_env = False  # air-gap: never pick up proxy variables
    session.cert = (config.client_cert, config.client_key)
    session.verify = config.ca or True

    def _send(method: str, url: str, params: dict | None, body: dict | None, timeout_s: float) -> tuple[int, Any]:
        try:
            response = session.request(method, url, params=params, json=body, timeout=timeout_s,
                                       allow_redirects=False)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(str(e)) from e
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return response.status_code, payload

    return _send


class BrokerClient:
    """Thin, fail-closed client. Every method returns a result, never raises after construction."""

    def __init__(self, config: BrokerConfig, transport: Transport | None = None) -> None:
        self.config = config.validate()
        self.transport = transport or requests_transport(self.config)

    # -- low level ------------------------------------------------------------

    def _call(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> tuple[str, int, Any]:
        """Return (transport_code or "", status, payload). transport_code set = no usable answer."""
        url = f"{self.config.base}{path}"
        try:
            status, payload = self.transport(method, url, params, body, float(self.config.timeout_s))
        except TimeoutError as e:
            return "TIMEOUT", 0, _scrub(f"broker timeout after {self.config.timeout_s}s: {e}")
        except Exception as e:  # connection refused, TLS failure, DNS ...
            return "UNREACHABLE", 0, _scrub(f"broker unreachable: {type(e).__name__}: {e}")
        return "", int(status), payload

    @staticmethod
    def _rejection(payload: Any) -> tuple[str, str, str, str] | None:
        """Detect a rejection/error body in either documented shape. Returns (art, code, detail, aufruf_id)."""
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        if not code:
            return None
        art = payload.get("art") or (ART_REJECTED if payload.get("abgelehnt") else "")
        if not art:
            art = ART_ERROR if code in ("TIMEOUT", "SOURCE_ERROR") else ART_REJECTED
        return str(art), str(code), _scrub(str(payload.get("detail", ""))), str(payload.get("aufruf_id", ""))

    # -- API ----------------------------------------------------------------------

    def list_tools(self, agent_id: str) -> ToolsResponse:
        tcode, status, payload = self._call("GET", "/v1/tools", params={"agent_id": agent_id})
        if tcode:
            return ToolsResponse(ok=False, code=tcode, detail=str(payload))
        if rej := self._rejection(payload):
            art, code, detail, _ = rej
            return ToolsResponse(ok=False, code=code, detail=detail)
        if status >= 400 or not isinstance(payload, dict) or not isinstance(payload.get("vorlagen", []), list):
            return ToolsResponse(ok=False, code="BAD_RESPONSE",
                                 detail=_scrub(f"HTTP {status}: {_short(payload)}"))
        specs = []
        for raw in payload.get("vorlagen", []):
            if not isinstance(raw, dict) or not raw.get("vorlage_id"):
                continue
            params = raw.get("parameter") or {}
            specs.append(ToolSpec(
                vorlage_id=str(raw["vorlage_id"]),
                version=int(raw.get("version", 0) or 0),
                beschreibung=str(raw.get("beschreibung", "")),
                parameter={str(k): dict(v) for k, v in params.items() if isinstance(v, dict)},
            ))
        return ToolsResponse(ok=True, katalog_version=str(payload.get("katalog_version", "")),
                             angehalten=bool(payload.get("angehalten", False)), vorlagen=tuple(specs))

    def invoke(self, vorlage_id: str, parameter: dict, kontext: dict) -> InvokeResult:
        body = {"vorlage_id": vorlage_id, "parameter": dict(parameter or {}), "kontext": dict(kontext or {})}
        tcode, status, payload = self._call("POST", "/v1/invoke", body=body)
        if tcode:
            return InvokeResult(ok=False, vorlage_id=vorlage_id, art=ART_ERROR, code=tcode, detail=str(payload))
        if rej := self._rejection(payload):
            art, code, detail, aufruf_id = rej
            return InvokeResult(ok=False, vorlage_id=vorlage_id, aufruf_id=aufruf_id, art=art, code=code, detail=detail)
        if status >= 400 or not isinstance(payload, dict) or not payload.get("aufruf_id"):
            return InvokeResult(ok=False, vorlage_id=vorlage_id, art=ART_ERROR, code="BAD_RESPONSE",
                                detail=_scrub(f"HTTP {status}: {_short(payload)}"))
        version = payload.get("vorlage_version")
        return InvokeResult(
            ok=True,
            aufruf_id=str(payload["aufruf_id"]),
            vorlage_id=str(payload.get("vorlage_id", vorlage_id)),
            vorlage_version=int(version) if version is not None else None,
            katalog_version=str(payload.get("katalog_version", "")),
            zeilen=int(payload.get("zeilen", 0) or 0),
            gekuerzt=bool(payload.get("gekuerzt", False)),
            ergebnis_hash=str(payload.get("ergebnis_hash", "")),
            daten=payload.get("daten"),
        )

    def health(self) -> tuple[bool, str]:
        tcode, status, payload = self._call("GET", "/health")
        if tcode:
            return False, str(payload)
        if 200 <= status < 300:
            return True, f"broker healthy (HTTP {status})"
        return False, _scrub(f"broker unhealthy: HTTP {status}: {_short(payload)}")


def _short(payload: Any, limit: int = 200) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return text[:limit]
