"""Integration tests for policy_mode 'service' against the real policy service from the bundled
wheel (sub-change 4.2b; RT-W01, RT-W04, RT-W09).

The service runs in-process over mTLS with a throwaway PKI made by the openssl CLI
(pattern: PolicyManager tests/service/certs.py). Without policy_service in the package
or without openssl the module is skipped; the stub-based tests cover the gate logic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time

import pytest

from soar_agent_runtime_policy import (
    ACTION_SEQUENCE,
    CODE_HALTED,
    GATE_APPROVE_2PERSON,
    PolicyClientAdapter,
    build_policy_client,
    sequence_step,
)
from test_soar_agent_runtime_acceptance import (  # noqa: F401  (pytest fixtures: platform, llm)
    RUN_ID,
    define_agent,
    effect_script,
    llm,
    make_connector,
    platform,
    run,
    step_log,
)
from test_soar_agent_runtime_policy_wheel import POLICY, request

pytest.importorskip("policy_service.server", reason="policy_manager wheel without policy_service")
OPENSSL = shutil.which("openssl")
if not OPENSSL:
    pytest.skip("openssl CLI not available; the mTLS integration tests need a throwaway PKI", allow_module_level=True)

from policy_service.app import ServiceConfig  # noqa: E402
from policy_service.config import ServerConfig  # noqa: E402
from policy_service.server import serve  # noqa: E402

PM_ID = re.compile(r"^pm-[0-9a-f]{16}$")
CLIENT_CN = "soar-agent-runtime"
_EC = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"]


# --- throwaway PKI (openssl CLI, EC keys, strict enough for Python 3.13) -----------------

def _openssl(*args: str, cwd: str) -> None:
    subprocess.run([OPENSSL, *args], cwd=cwd, check=True, capture_output=True)


def _make_ca(directory: str) -> tuple[str, str]:
    _openssl("req", "-x509", *_EC, "-keyout", "ca.key", "-out", "ca.crt", "-subj", "/CN=rt-test-ca", "-days", "1",
             "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
             cwd=directory)
    return os.path.join(directory, "ca.crt"), os.path.join(directory, "ca.key")


def _make_cert(directory: str, ca: tuple[str, str], cn: str, server: bool = False) -> tuple[str, str]:
    stem = f"{cn}.{'srv' if server else 'cli'}"
    _openssl("req", "-new", *_EC, "-keyout", f"{stem}.key", "-out", f"{stem}.csr", "-subj", f"/CN={cn}", cwd=directory)
    ext = os.path.join(directory, f"{stem}.ext")
    with open(ext, "w", encoding="utf-8") as fh:
        fh.write("basicConstraints=critical,CA:FALSE\nauthorityKeyIdentifier=keyid,issuer\nsubjectKeyIdentifier=hash\n")
        fh.write("subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n" if server
                 else "extendedKeyUsage=clientAuth\n")
    _openssl("x509", "-req", "-in", f"{stem}.csr", "-CA", ca[0], "-CAkey", ca[1], "-CAcreateserial",
             "-out", f"{stem}.crt", "-days", "1", "-extfile", ext, cwd=directory)
    return os.path.join(directory, f"{stem}.crt"), os.path.join(directory, f"{stem}.key")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("pki"))
    ca = _make_ca(d)
    return {"dir": d, "ca": ca[0],
            "server": _make_cert(d, ca, "127.0.0.1", server=True),
            "runtime": _make_cert(d, ca, CLIENT_CN),          # role decide
            "guest": _make_cert(d, ca, "gast")}                # valid certificate, no role


@pytest.fixture
def service(pki, tmp_path):
    """The real policy service from the wheel on 127.0.0.1 with the wheel-test policy."""
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    state = tmp_path / "state"
    state.mkdir()
    (state / "halt.json").write_text(json.dumps({"angehalten": False, "grund": ""}))
    cfg = ServerConfig(host="127.0.0.1", port=_free_port(), cert=pki["server"][0], key=pki["server"][1], ca=pki["ca"],
                       service=ServiceConfig(policy_path=str(policy), state_dir=str(state),
                                             clients={CLIENT_CN: frozenset({"decide"})}))
    server = serve(cfg)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"url": f"https://127.0.0.1:{cfg.port}", "state": state, "port": cfg.port}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def gate_for(service, pki, who="runtime", timeout_s=5.0):
    cert, key = pki[who]
    return build_policy_client("service", service_url=service["url"], service_timeout_s=timeout_s,
                               client_cert=cert, client_key=key, ca=pki["ca"])


# --- the gate over mTLS ------------------------------------------------------------------

def test_service_gate_is_healthy_decides_and_lists_the_catalogue(service, pki):
    gate = gate_for(service, pki)
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "service"
    assert gate.health() == (True, "policy service healthy, not halted")
    halt = gate.halt_status()
    assert halt.halted is False and halt.known is True                    # bekannt over the wire (API.md §5)
    d = gate.decide(request())
    assert d.allowed and PM_ID.match(d.decision_id) and "fall_aenderung" in d.reason_text()
    block, undo = gate.actions("triage")                                  # GET /v1/aktionen
    assert (block.id, block.min_stage, block.min_gate) == ("isolate_endpoint", 1, "APPROVE_2PERSON")
    assert undo.id == "unisolate_endpoint" and gate.actions("someone_else") == []
    seq = request(kind=ACTION_SEQUENCE, params={}, targets=("ws-0815",),
                  steps=(sequence_step("isolate_endpoint", ["ws-0815"], {"dauer_h": 2}),))
    d = gate.decide(seq)
    assert d.needs_approval and d.gate == GATE_APPROVE_2PERSON and d.approval_id
    assert (service["state"] / "audit.jsonl").is_file()                  # the service audits, not the runtime


def test_halt_on_the_server_side_is_visible_and_denies_agents(service, pki):
    gate = gate_for(service, pki)
    (service["state"] / "halt.json").write_text(json.dumps({"angehalten": True, "grund": "drill"}))
    halt = gate.halt_status()
    assert halt.halted is True and halt.reason == "drill" and halt.known is True
    assert gate.health() == (True, "policy service healthy, HALTED (drill)")
    d = gate.decide(request())
    assert d.denied and d.code == CODE_HALTED


def test_certificate_without_a_role_gets_nothing(service, pki):
    gate = gate_for(service, pki, who="guest")
    d = gate.decide(request())
    assert d.denied and not d.allowed and d.code == "UNAVAILABLE" and "HTTP 403" in d.reason_text()
    ok, detail = gate.health()
    assert ok is False and "halt state unknown (HTTP 403)" in detail
    assert gate.actions("triage") == []


def test_no_service_at_the_port_denies_without_fallback(pki):
    gate = build_policy_client("service", service_url=f"https://127.0.0.1:{_free_port()}", service_timeout_s=1.0,
                               client_cert=pki["runtime"][0], client_key=pki["runtime"][1], ca=pki["ca"])
    ok, detail = gate.health()
    assert ok is False and detail.startswith("policy service: halt state unknown (")
    d = gate.decide(request())
    assert d.denied and d.code == "UNAVAILABLE" and gate.actions("triage") == []


def test_silent_endpoint_hits_the_timeout_and_denies(pki):
    """A port that accepts but never speaks TLS: the timeout, not a hang, decides (RT-W04)."""
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(5)                                          # never accept()ed: handshake never answers
        port = silent.getsockname()[1]
        gate = build_policy_client("service", service_url=f"https://127.0.0.1:{port}", service_timeout_s=0.5,
                                   client_cert=pki["runtime"][0], client_key=pki["runtime"][1], ca=pki["ca"])
        started = time.monotonic()
        d = gate.decide(request())
        elapsed = time.monotonic() - started
    assert d.denied and d.code == "UNAVAILABLE" and elapsed < 3.0, (d, elapsed)


# --- through the connector: asset fields -> real service -> effecting tool ----------------

def _service_asset(service, pki, **extra):
    return {"airgap_mode": True, "policy_mode": "service", "policy_service_url": service["url"],
            "policy_client_cert": pki["runtime"][0], "policy_client_key": pki["runtime"][1], "policy_ca": pki["ca"],
            "policy_timeout_s": 2, **extra}


def test_airgap_run_executes_update_severity_through_the_real_service(platform, llm, monkeypatch, service, pki):  # noqa: F811
    define_agent(platform, provider="local")
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: RUN_ID)
    llm["responses"] = effect_script("update_severity")
    c = make_connector(**_service_asset(service, pki))
    status, result = run(c, stufe=2, domaene="cyber")
    assert status is True, result.message
    assert platform["severity"] == [(11, "high")]
    obs = step_log(result)[1]["observation"]
    assert re.search(r"\(entscheidung_id=pm-[0-9a-f]{16}\)$", obs), obs
    assert any(m.startswith("[policy] mode=service (ready)") for m in c.progress)
    prompt = llm["system_prompts"][0]
    assert "Catalogue building blocks (stufe 2):" in prompt and "- isolate_endpoint v2:" in prompt
    assert any("[policy] catalogue: 2 block(s) available at stufe 2" in m for m in c.progress)


def test_airgap_test_connectivity_reports_the_real_service(monkeypatch, service, pki):
    from test_soar_agent_runtime_connector import BROKER_ASSET, _health_transport, _record_connection_tests
    _record_connection_tests(monkeypatch)
    c = make_connector(**_service_asset(service, pki), **BROKER_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy service healthy, not halted"' in c.action_results[-1].message
    (service["state"] / "halt.json").write_text(json.dumps({"angehalten": True, "grund": "drill"}))
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy service healthy, HALTED (drill)"' in c.action_results[-1].message
    c = make_connector(**_service_asset(service, pki, policy_client_cert=pki["guest"][0], policy_client_key=pki["guest"][1]),
                       **BROKER_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).") and 'halt state unknown (HTTP 403)' in msg
