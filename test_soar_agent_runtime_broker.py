"""BrokerClient tests (Umsetzungskonzept Runtime, sub-change 2.2; RT-L01, L05, L06, L07; E10)."""

from __future__ import annotations

import sys
import types

import pytest

from soar_agent_runtime_broker import (
    ART_ERROR,
    ART_REJECTED,
    BROKER_CODES,
    BrokerClient,
    BrokerConfig,
    BrokerConfigError,
    requests_transport,
)

CFG = BrokerConfig(url="https://broker.internal:8443/", timeout_s=7,
                   client_cert="/etc/rt.crt", client_key="/etc/rt.key", ca="/etc/ca.pem")
KONTEXT = {"agent_id": "triage", "lauf_id": "rt-x", "container_id": 4711, "schritt": 3, "schritt_id": "rt-x/3"}


class FakeTransport:
    def __init__(self, status=200, payload=None, raise_exc=None):
        self.status, self.payload, self.raise_exc = status, payload, raise_exc
        self.calls: list[tuple] = []

    def __call__(self, method, url, params, body, timeout_s):
        self.calls.append((method, url, params, body, timeout_s))
        if self.raise_exc:
            raise self.raise_exc
        return self.status, self.payload


def client(**kw):
    t = FakeTransport(**kw)
    return BrokerClient(CFG, transport=t), t


# --- config -----------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://broker.internal:8443", "", "broker.internal", "ftp://x"])
def test_config_requires_https(url):
    with pytest.raises(BrokerConfigError, match="https://"):
        BrokerClient(BrokerConfig(url=url, client_cert="c", client_key="k"), transport=FakeTransport())


@pytest.mark.parametrize("cert,key", [("", "k"), ("c", ""), ("", "")])
def test_config_requires_client_certificate_and_key(cert, key):
    with pytest.raises(BrokerConfigError, match="broker_client_cert and broker_client_key"):
        BrokerClient(BrokerConfig(url="https://b", client_cert=cert, client_key=key), transport=FakeTransport())


def test_config_allows_empty_ca_and_rejects_zero_timeout():
    BrokerClient(BrokerConfig(url="https://b", client_cert="c", client_key="k"), transport=FakeTransport())
    with pytest.raises(BrokerConfigError, match="timeout"):
        BrokerClient(BrokerConfig(url="https://b", timeout_s=0, client_cert="c", client_key="k"), transport=FakeTransport())


# --- list_tools (E10) -------------------------------------------------------

E10_TOOLS = {
    "katalog_version": "a41c9e2",
    "angehalten": False,
    "vorlagen": [
        {"vorlage_id": "cve_offen_auf_assets", "version": 3,
         "beschreibung": "Offene Assets zu einer CVE",
         "parameter": {"cve": {"typ": "string", "muster": "^CVE-\\d{4}-\\d{4,7}$", "pflicht": True},
                       "zeitfenster": {"typ": "relzeit", "standard": "-7d", "max": "-30d", "pflicht": False}},
         "irgendwas_neues": {"x": 1}},
        {"vorlage_id": "asset_kontext", "version": 1, "beschreibung": "Asset-Stammdaten"},
    ],
    "zukunft": "ignored",
}


def test_list_tools_parses_e10_and_ignores_unknown_fields():
    c, t = client(payload=E10_TOOLS)
    r = c.list_tools("triage")
    assert t.calls == [("GET", "https://broker.internal:8443/v1/tools", {"agent_id": "triage"}, None, 7.0)]
    assert r.ok and r.katalog_version == "a41c9e2" and r.angehalten is False
    assert [v.vorlage_id for v in r.vorlagen] == ["cve_offen_auf_assets", "asset_kontext"]
    cve = r.vorlagen[0]
    assert cve.version == 3 and cve.beschreibung == "Offene Assets zu einer CVE"
    assert cve.parameter["cve"]["muster"].startswith("^CVE") and cve.parameter["zeitfenster"]["standard"] == "-7d"
    assert r.vorlagen[1].parameter == {}


def test_list_tools_reports_halt_state():
    c, _ = client(payload={"katalog_version": "x", "angehalten": True, "vorlagen": []})
    r = c.list_tools("triage")
    assert r.ok and r.angehalten is True and r.vorlagen == ()


def test_list_tools_rejection_and_bad_response():
    c, _ = client(payload={"abgelehnt": True, "code": "CLIENT_NOT_ALLOWED", "detail": "cert unknown"})
    r = c.list_tools("triage")
    assert (r.ok, r.code, r.detail) == (False, "CLIENT_NOT_ALLOWED", "cert unknown")
    c, _ = client(status=500, payload="<html>oops</html>")
    r = c.list_tools("triage")
    assert (r.ok, r.code) == (False, "BAD_RESPONSE") and "HTTP 500" in r.detail


# --- invoke -----------------------------------------------------------------

OK_BODY = {"aufruf_id": "tb-000231", "vorlage_id": "cve_offen_auf_assets", "vorlage_version": 3,
           "katalog_version": "a41c9e2", "zeilen": 2, "gekuerzt": True, "ergebnis_hash": "sha256:abc",
           "daten": [{"asset": "srv-1"}, {"asset": "srv-2"}]}


def test_invoke_sends_context_and_parses_success():
    c, t = client(payload=OK_BODY)
    r = c.invoke("cve_offen_auf_assets", {"cve": "CVE-2026-1"}, KONTEXT)
    method, url, params, body, timeout = t.calls[0]
    assert (method, url, params, timeout) == ("POST", "https://broker.internal:8443/v1/invoke", None, 7.0)
    assert body == {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-1"}, "kontext": KONTEXT}
    assert r.ok and r.aufruf_id == "tb-000231" and r.vorlage_version == 3 and r.zeilen == 2 and r.gekuerzt is True
    assert r.ergebnis_hash == "sha256:abc" and r.daten == OK_BODY["daten"]
    assert not r.halted and not r.data_gap


def test_invoke_rejection_in_spec_shape():
    c, _ = client(payload={"aufruf_id": "tb-2", "abgelehnt": True, "code": "INVALID_PARAM", "detail": "cve pattern"})
    r = c.invoke("v", {}, KONTEXT)
    assert (r.ok, r.art, r.code, r.detail, r.aufruf_id) == (False, ART_REJECTED, "INVALID_PARAM", "cve pattern", "tb-2")


def test_invoke_error_in_broker_code_shape():
    c, _ = client(payload={"aufruf_id": "tb-3", "art": "fehler", "code": "SOURCE_ERROR", "detail": "splunk down"})
    r = c.invoke("v", {}, KONTEXT)
    assert (r.ok, r.art, r.code) == (False, ART_ERROR, "SOURCE_ERROR") and r.data_gap


@pytest.mark.parametrize("code", BROKER_CODES)
def test_every_broker_code_is_passed_through(code):
    c, _ = client(status=403, payload={"code": code, "detail": "d"})
    r = c.invoke("v", {}, KONTEXT)
    assert r.ok is False and r.code == code
    assert r.halted is (code == "HALTED")
    assert r.art in (ART_REJECTED, ART_ERROR)


def test_invoke_detail_is_scrubbed():
    c, _ = client(payload={"code": "NOT_ALLOWED", "detail": "token Bearer abcdefgh.ijklmnop leaked"})
    assert "abcdefgh" not in c.invoke("v", {}, KONTEXT).detail


@pytest.mark.parametrize("exc,code", [(ConnectionError("refused"), "UNREACHABLE"),
                                      (TimeoutError("deadline"), "TIMEOUT"),
                                      (OSError("tls handshake"), "UNREACHABLE")])
def test_transport_failures_become_results_not_exceptions(exc, code):
    c, _ = client(raise_exc=exc)
    r = c.invoke("v", {}, KONTEXT)
    assert (r.ok, r.art, r.code) == (False, ART_ERROR, code) and r.data_gap
    assert c.list_tools("a").code == code
    assert c.health()[0] is False


def test_invoke_bad_response_without_aufruf_id():
    c, _ = client(payload={"daten": []})
    r = c.invoke("v", {}, KONTEXT)
    assert (r.ok, r.code) == (False, "BAD_RESPONSE")


def test_health():
    c, t = client(status=200, payload={"status": "ok"})
    assert c.health() == (True, "broker healthy (HTTP 200)")
    assert t.calls[0][:2] == ("GET", "https://broker.internal:8443/health")
    c, _ = client(status=503, payload="down")
    ok, msg = c.health()
    assert ok is False and "HTTP 503" in msg


# --- default transport ------------------------------------------------------

def test_requests_transport_uses_mtls_no_proxy_no_redirects(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": 1}

        text = "{}"

    class FakeSession:
        def __init__(self):
            self.trust_env = True
            self.cert = None
            self.verify = True

        def request(self, method, url, **kw):
            calls.update(method=method, url=url, session=self, **kw)
            return FakeResponse()

    class Timeout(Exception):
        pass

    fake_requests = types.SimpleNamespace(Session=FakeSession, exceptions=types.SimpleNamespace(Timeout=Timeout))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    send = requests_transport(CFG)
    status, payload = send("POST", "https://broker.internal:8443/v1/invoke", None, {"a": 1}, 7.0)
    assert (status, payload) == (200, {"ok": 1})
    s = calls["session"]
    assert s.trust_env is False and s.cert == ("/etc/rt.crt", "/etc/rt.key") and s.verify == "/etc/ca.pem"
    assert calls["timeout"] == 7.0 and calls["allow_redirects"] is False and calls["json"] == {"a": 1}


def test_requests_transport_maps_timeout_and_non_json(monkeypatch):
    class Timeout(Exception):
        pass

    class FakeResponse:
        status_code = 502
        text = "bad gateway"

        def json(self):
            raise ValueError("no json")

    class FakeSession:
        trust_env = True

        def request(self, method, url, **kw):
            if url.endswith("/slow"):
                raise Timeout("read timed out")
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests",
                        types.SimpleNamespace(Session=FakeSession, exceptions=types.SimpleNamespace(Timeout=Timeout)))
    send = requests_transport(BrokerConfig(url="https://b", client_cert="c", client_key="k"))
    assert send("GET", "https://b/x", None, None, 1.0) == (502, "bad gateway")
    with pytest.raises(TimeoutError):
        send("GET", "https://b/slow", None, None, 1.0)


@pytest.mark.parametrize("payload,expected", [
    ({"aufruf_id": "tb-9", "datenlage": "vorhanden", "zeilen": 0, "daten": []}, "vorhanden"),
    ({"aufruf_id": "tb-9", "datenlage": "leer", "zeilen": 0, "daten": []}, "leer"),
    ({"aufruf_id": "tb-9", "zeilen": 0, "daten": []}, ""),          # broker before TB-F18
])
def test_invoke_passes_the_datenlage_through(payload, expected):
    c, _ = client(payload=payload)
    assert c.invoke("v", {}, KONTEXT).datenlage == expected
