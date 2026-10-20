"""Integration test: the real requests transport against an mTLS stub broker (sub-change 2.7).

Proves what the unit tests cannot: BrokerClient speaks HTTPS with a client
certificate to a server that REQUIRES one, verifies the server against the
configured CA, and parses the E10 / spec §6 bodies end to end. Certificates
are generated per test session with the openssl CLI into a temp directory;
without openssl the module is skipped with a reason, never silently green.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socket
import ssl
import subprocess
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from soar_agent_runtime_broker import BrokerClient, BrokerConfig, BrokerConfigError

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")

TOOLS = {"katalog_version": "stub-1", "angehalten": False, "vorlagen": [
    {"vorlage_id": "asset_kontext", "version": 1, "beschreibung": "Asset-Stammdaten", "parameter": {}}]}


def _openssl(*args, cwd):
    subprocess.run(["openssl", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """CA, server cert (SAN localhost), trusted client cert, and a rogue self-signed client cert."""
    d = tmp_path_factory.mktemp("pki")
    try:
        # OpenSSL >= 3 verifies strictly: leaf certs need SKI/AKI, the CA needs CA:TRUE.
        _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.crt",
                 "-subj", "/CN=test-ca", "-days", "2",
                 "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "subjectKeyIdentifier=hash",
                 "-addext", "keyUsage=critical,keyCertSign,cRLSign", cwd=d)
        leaf_ext = "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\nbasicConstraints=CA:FALSE\n"
        (d / "srv.ext").write_text(leaf_ext + "subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n")
        _openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "srv.key", "-out", "srv.csr",
                 "-subj", "/CN=localhost", cwd=d)
        _openssl("x509", "-req", "-in", "srv.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
                 "-out", "srv.crt", "-days", "2", "-extfile", "srv.ext", cwd=d)
        (d / "cli.ext").write_text(leaf_ext + "extendedKeyUsage=clientAuth\n")
        _openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "cli.key", "-out", "cli.csr",
                 "-subj", "/CN=soar_agent_runtime", cwd=d)
        _openssl("x509", "-req", "-in", "cli.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
                 "-out", "cli.crt", "-days", "2", "-extfile", "cli.ext", cwd=d)
        _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "rogue.key", "-out", "rogue.crt",
                 "-subj", "/CN=rogue", "-days", "2", cwd=d)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"openssl could not create test certificates: {e.stderr.decode(errors='replace')[:200]}")
    return d


class StubHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep pytest output clean
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send(200, {"status": "ok"})
        if url.path == "/v1/tools":
            agent = parse_qs(url.query).get("agent_id", [""])[0]
            self.server.seen.append(("GET", url.path, agent))
            return self._send(200, TOOLS)
        return self._send(404, {"code": "NOT_FOUND"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append(("POST", self.path, body))
        if self.path != "/v1/invoke":
            return self._send(404, {"code": "NOT_FOUND"})
        if body.get("vorlage_id") != "asset_kontext":
            return self._send(404, {"aufruf_id": "tb-stub-2", "abgelehnt": True, "code": "UNKNOWN_TEMPLATE",
                                    "detail": f"unknown template {body.get('vorlage_id')!r}"})
        return self._send(200, {"aufruf_id": "tb-stub-1", "vorlage_id": "asset_kontext", "vorlage_version": 1,
                                "katalog_version": "stub-1", "zeilen": 1, "gekuerzt": False,
                                "ergebnis_hash": "sha256:stub", "daten": [{"asset": "srv-1"}]})


@pytest.fixture(scope="module")
def stub(pki):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.seen = []
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(pki / "srv.crt"), str(pki / "srv.key"))
    ctx.load_verify_locations(str(pki / "ca.crt"))
    ctx.verify_mode = ssl.CERT_REQUIRED  # mTLS: no client certificate, no connection
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"https://localhost:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def config(pki, url, cert="cli", ca="ca.crt"):
    return BrokerConfig(url=url, timeout_s=5, client_cert=str(pki / f"{cert}.crt"),
                        client_key=str(pki / f"{cert}.key"), ca=str(pki / ca))


def test_mtls_roundtrip_health_tools_invoke(pki, stub):
    server, url = stub
    client = BrokerClient(config(pki, url))  # real requests transport
    ok, msg = client.health()
    assert ok, msg
    tools = client.list_tools("triage")
    assert tools.ok and tools.katalog_version == "stub-1" and [t.vorlage_id for t in tools.vorlagen] == ["asset_kontext"]
    kontext = {"agent_id": "triage", "lauf_id": "rt-x", "container_id": 1, "schritt": 1, "schritt_id": "rt-x/1"}
    r = client.invoke("asset_kontext", {}, kontext)
    assert r.ok and r.aufruf_id == "tb-stub-1" and r.daten == [{"asset": "srv-1"}]
    assert ("GET", "/v1/tools", "triage") in server.seen
    assert ("POST", "/v1/invoke", {"vorlage_id": "asset_kontext", "parameter": {}, "kontext": kontext}) in server.seen


def test_rejection_body_from_a_real_server_is_parsed(pki, stub):
    _, url = stub
    r = BrokerClient(config(pki, url)).invoke("nope", {}, {})
    assert (r.ok, r.code, r.aufruf_id) == (False, "UNKNOWN_TEMPLATE", "tb-stub-2") and "unknown template" in r.detail


def test_client_without_trusted_certificate_is_refused_by_the_server(pki, stub):
    _, url = stub
    r = BrokerClient(config(pki, url, cert="rogue")).invoke("asset_kontext", {}, {})
    assert r.ok is False and r.code == "UNREACHABLE"
    ok, _ = BrokerClient(config(pki, url, cert="rogue")).health()
    assert ok is False


def test_server_not_signed_by_configured_ca_is_refused_by_the_client(pki, stub):
    _, url = stub
    r = BrokerClient(config(pki, url, ca="rogue.crt")).list_tools("triage")
    assert r.ok is False and r.code == "UNREACHABLE"


def test_plain_http_url_never_reaches_the_network(pki, stub):
    _, url = stub
    with pytest.raises(BrokerConfigError):
        BrokerClient(config(pki, url.replace("https://", "http://")))


def test_closed_port_is_unreachable_not_an_exception(pki):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = BrokerClient(config(pki, f"https://localhost:{port}")).health()
    assert r[0] is False and "unreachable" in r[1]
