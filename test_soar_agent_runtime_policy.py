"""Policy gate interface tests (Umsetzungskonzept Runtime, sub-change 3.0; RT-W01, W03, W06, W08)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from soar_agent_runtime_policy import (
    ACTION_SEQUENCE,
    APPROVAL_GATES,
    CODE_HALTED,
    CODE_INVALID_RESPONSE,
    CODE_NO_POLICY_MODE,
    CODE_UNAVAILABLE,
    GATE_ALLOW,
    GATE_APPROVE_2PERSON,
    GATE_DENY,
    ApprovalState,
    CatalogBlock,
    ConsumeState,
    HaltState,
    NoPolicyClient,
    PolicyClientAdapter,
    PolicyDecision,
    PolicyRequest,
    StubPolicyClient,
    build_policy_client,
    parse_catalog,
    sequence_step,
)

REQ = PolicyRequest(kind="update_severity", params={"severity": "high"}, targets=("container:4711",),
                    initiator="triage", agent_id="triage", run_id="rt-x", step_id="rt-x/3",
                    evidence_ids=("tb-1", "rt-x/2"), judgement={"begruendung": "b", "gegenargumente": "g"},
                    stage=2, domain="soc", confidence=0.8)


# -- wire format (API.md) -----------------------------------------------------------

def test_request_wire_uses_spec_names_and_omits_reversibility():
    wire = REQ.to_wire()
    assert set(wire) == {"aktion", "kontext"}
    assert wire["aktion"] == {"art": "update_severity", "kategorie": "", "playbook": "",
                              "ziel": ["container:4711"], "parameter": {"severity": "high"}}
    k = wire["kontext"]
    assert k["principal_type"] == "agent" and k["initiator"] == "triage"
    assert k["lauf_id"] == "rt-x" and k["schritt_id"] == "rt-x/3"
    assert k["beleg_ids"] == ["tb-1", "rt-x/2"] and k["urteil"]["begruendung"] == "b"
    assert k["stufe"] == 2 and k["agent_konfidenz"] == 0.8 and k["domaene"] == "soc"
    assert "reversibilitaet" not in k and "reversibel" not in str(wire)


def test_request_wire_leaves_out_unknown_stage_and_confidence():
    k = PolicyRequest(kind="x").to_wire()["kontext"]
    assert "stufe" not in k and "agent_konfidenz" not in k


# -- decision parsing --------------------------------------------------------------

def test_allow_needs_decision_id():
    ok = PolicyDecision.from_wire({"gate": "ALLOW", "entscheidung_id": "e-1", "policy_version": "v3"})
    assert ok.allowed and ok.decision_id == "e-1" and ok.policy_version == "v3"
    bad = PolicyDecision.from_wire({"gate": "ALLOW"})
    assert bad.denied and bad.code == CODE_INVALID_RESPONSE


@pytest.mark.parametrize("gate", APPROVAL_GATES)
def test_approval_gate_needs_freigabe_id(gate):
    ok = PolicyDecision.from_wire({"gate": gate, "entscheidung_id": "e-2", "freigabe_id": "f-9",
                                   "freigaben_noetig": 2, "ablauf": "2026-10-20T10:00:00Z"})
    assert ok.needs_approval and not ok.allowed and ok.approval_id == "f-9" and ok.approvals_needed == 2
    bad = PolicyDecision.from_wire({"gate": gate, "entscheidung_id": "e-2"})
    assert bad.denied and bad.code == CODE_INVALID_RESPONSE and bad.decision_id == "e-2"


@pytest.mark.parametrize("gate", APPROVAL_GATES)
def test_approval_gate_without_freigabe_id_keeps_the_package_code(gate):
    held = PolicyDecision.from_wire({"gate": gate, "entscheidung_id": "e-3", "code": "NO_INITIATOR",
                                     "begruendung": [{"regel": "policy_layer", "grund": "base=APPROVE_2PERSON"}]})
    assert held.denied and not held.needs_approval and not held.allowed
    assert held.code == "NO_INITIATOR" and held.decision_id == "e-3" and held.reason_text() == "base=APPROVE_2PERSON"


@pytest.mark.parametrize("payload", [None, "ALLOW", {}, {"gate": "MAYBE"}, {"gate": 1}, []])
def test_malformed_decisions_are_deny(payload):
    d = PolicyDecision.from_wire(payload)
    assert d.gate == GATE_DENY and d.denied and d.code == CODE_INVALID_RESPONSE


def test_deny_keeps_code_and_reasons():
    d = PolicyDecision.from_wire({"gate": "DENY", "code": "NO_EVIDENCE",
                                  "begruendung": [{"regel": "R1", "grund": "no evidence"}, "junk"]})
    assert d.denied and d.code == "NO_EVIDENCE" and d.reason_text() == "no evidence"


def test_other_results_fail_closed():
    assert HaltState.from_wire({"angehalten": False}).halted is False
    assert HaltState.from_wire({"angehalten": "no"}).halted is True
    assert HaltState.from_wire(None).halted is True
    assert ConsumeState.from_wire({"gueltig": True}).ok is True
    assert ConsumeState.from_wire({"gueltig": 1}).ok is False
    assert ConsumeState.from_wire({"gueltig": False, "code": "EXPIRED"}).code == "EXPIRED"
    assert ApprovalState.from_wire("x", "f-1").status == "expired"
    assert ApprovalState.from_wire({"status": "pending", "erteilt": 1, "noetig": 2}, "f-1").have == 1


# -- NoPolicyClient (RT-W03) ------------------------------------------------------------

def test_no_policy_client_denies_everything_and_is_halted():
    gate = NoPolicyClient("policy_mode missing")
    d = gate.decide(REQ)
    assert d.denied and d.code == CODE_NO_POLICY_MODE and "policy_mode missing" in d.reason_text()
    assert gate.halt_status().halted is True
    assert gate.consume("f-1", REQ).ok is False
    assert gate.health() == (False, "policy_mode missing")


@pytest.mark.parametrize("mode", ["", None, "EMBEDDED ", "cloud", "service"])
def test_factory_without_usable_mode_returns_no_policy_client(mode):
    gate = build_policy_client(mode)
    assert isinstance(gate, NoPolicyClient)
    assert gate.decide(REQ).denied
    if mode == "service":
        assert gate.code == CODE_NO_POLICY_MODE and "policy_service_url" in gate.reason  # fields missing (4.1)
    elif mode == "EMBEDDED ":
        assert "policy_path" in gate.reason  # normalised mode, files missing (E11)


def test_factory_embedded_without_package_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "policy_client", None)   # import -> ImportError
    monkeypatch.setitem(sys.modules, "policy_client.base", None)
    monkeypatch.setitem(sys.modules, "policy_client.embedded", None)
    gate = build_policy_client("embedded", policy_path="/p.json", approvals_path="/a.json", halt_path="/h.json")
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_UNAVAILABLE
    assert gate.decide(REQ).code == CODE_UNAVAILABLE


# -- adapter around the real package (wire-only bridge) ---------------------------------

@dataclass
class _Wire:
    data: dict

    def to_wire(self):
        return self.data


class _FakeRequestType:
    seen: list = []

    @classmethod
    def from_wire(cls, data):
        cls.seen.append(data)
        return ("req", data)


class _FakePolicyClient:
    def __init__(self, decision=None, fail=False, halted=False):
        self.decision = decision or {"gate": "ALLOW", "entscheidung_id": "e-7"}
        self.fail = fail
        self.halted = halted
        self.calls = []

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("boom")

    def decide(self, request):
        self._maybe_fail()
        self.calls.append(("decide", request))
        return _Wire(self.decision)

    def approval_status(self, approval_id):
        self._maybe_fail()
        return _Wire({"freigabe_id": approval_id, "status": "pending", "erteilt": 0, "noetig": 1})

    def consume(self, approval_id, request):
        self._maybe_fail()
        self.calls.append(("consume", approval_id, request))
        return _Wire({"gueltig": True})

    def halt_status(self):
        self._maybe_fail()
        return _Wire({"angehalten": self.halted, "grund": "drill" if self.halted else ""})


def test_adapter_bridges_through_the_wire_format():
    inner = _FakePolicyClient()
    gate = PolicyClientAdapter(inner, _FakeRequestType, "embedded")
    d = gate.decide(REQ)
    assert d.allowed and d.decision_id == "e-7"
    assert inner.calls[0] == ("decide", ("req", REQ.to_wire()))
    assert gate.approval_status("f-1").status == "pending"
    assert gate.consume("f-1", REQ).ok is True
    assert gate.halt_status().halted is False
    assert gate.health() == (True, "policy embedded healthy, not halted")


def test_adapter_reports_halt_as_healthy_but_halted():
    gate = PolicyClientAdapter(_FakePolicyClient(halted=True), _FakeRequestType, "embedded")
    assert gate.halt_status() == HaltState(halted=True, reason="drill")
    ok, detail = gate.health()
    assert ok and "HALTED (drill)" in detail


def test_adapter_never_raises():
    gate = PolicyClientAdapter(_FakePolicyClient(fail=True), _FakeRequestType, "embedded")
    d = gate.decide(REQ)
    assert d.denied and d.code == CODE_UNAVAILABLE and "RuntimeError" in d.reason_text()
    assert gate.approval_status("f-1").status == "expired"
    assert gate.consume("f-1", REQ).ok is False
    assert gate.halt_status().halted is True
    ok, detail = gate.health()
    assert not ok and "RuntimeError" in detail


@pytest.mark.parametrize("content,problem", [
    (None, "policy file not readable (FileNotFoundError)"),
    ("{not json", "policy file is not valid JSON"),
    ("[1, 2]", "policy file is not a JSON object"),
])
def test_adapter_health_reports_an_unusable_policy_file(tmp_path, content, problem):
    path = tmp_path / "policy.json"
    if content is not None:
        path.write_text(content)
    gate = PolicyClientAdapter(_FakePolicyClient(), _FakeRequestType, "embedded", policy_path=str(path))
    ok, detail = gate.health()
    assert ok is False and detail == f"policy embedded: {problem}: {path}"
    assert gate.decide(REQ).allowed          # health is advisory; decisions still come from the package


def test_adapter_health_accepts_a_json_object_policy_file(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"default_gate": "DENY", "categories": {}}')
    gate = PolicyClientAdapter(_FakePolicyClient(), _FakeRequestType, "embedded", policy_path=str(path))
    assert gate.health() == (True, "policy embedded healthy, not halted")


def test_adapter_treats_malformed_package_answer_as_deny():
    gate = PolicyClientAdapter(_FakePolicyClient(decision={"gate": "ALLOW"}), _FakeRequestType, "embedded")
    assert gate.decide(REQ).code == CODE_INVALID_RESPONSE


def _install_fake_package(monkeypatch, embedded_cls=None, service_cls=None):
    """Fake policy_client package: P1 layout (no service module) unless service_cls is given."""
    pkg = types.ModuleType("policy_client")
    base = types.ModuleType("policy_client.base")
    base.DecisionRequest = _FakeRequestType
    emb = types.ModuleType("policy_client.embedded")
    emb.EmbeddedPolicyClient = embedded_cls or _FakePolicyClient
    modules = {"policy_client": pkg, "policy_client.base": base, "policy_client.embedded": emb,
               "policy_client.service": None}
    if service_cls is not None:
        svc = types.ModuleType("policy_client.service")
        svc.ServicePolicyClient = service_cls
        modules["policy_client.service"] = svc
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_factory_builds_adapter_from_p1_package_without_audit_kwargs(monkeypatch):
    built = {}

    class FakeEmbeddedP1(_FakePolicyClient):
        def __init__(self, policy_path, approvals_path, halt_path, ttl=600.0):
            super().__init__()
            built.update(policy_path=policy_path, approvals_path=approvals_path, halt_path=halt_path)

    _install_fake_package(monkeypatch, FakeEmbeddedP1)
    gate = build_policy_client("embedded", policy_path="/etc/pm/policy.json", approvals_path="/var/pm/approvals.json",
                               halt_path="/var/pm/halt.json", audit_path="/var/pm/audit.jsonl")
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "embedded"
    assert gate.policy_path == "/etc/pm/policy.json"
    assert built == {"policy_path": "/etc/pm/policy.json", "approvals_path": "/var/pm/approvals.json",
                     "halt_path": "/var/pm/halt.json"}
    assert gate.decide(REQ).allowed


def test_factory_passes_audit_kwargs_to_a_package_that_takes_them(monkeypatch):
    built = {}

    class FakeEmbeddedP3(_FakePolicyClient):
        def __init__(self, policy_path, approvals_path, halt_path, ttl=600.0, audit_path=None, audit_client=""):
            super().__init__()
            built.update(policy_path=policy_path, approvals_path=approvals_path, halt_path=halt_path,
                         audit_path=audit_path, audit_client=audit_client)

    _install_fake_package(monkeypatch, FakeEmbeddedP3)
    gate = build_policy_client("embedded", policy_path="/etc/pm/policy.json", approvals_path="/var/pm/approvals.json",
                               halt_path="/var/pm/halt.json", audit_path="/var/pm/audit.jsonl")
    assert isinstance(gate, PolicyClientAdapter)
    assert built == {"policy_path": "/etc/pm/policy.json", "approvals_path": "/var/pm/approvals.json",
                     "halt_path": "/var/pm/halt.json", "audit_path": "/var/pm/audit.jsonl",
                     "audit_client": "soar_agent_runtime"}
    assert build_policy_client("embedded", policy_path="/p", approvals_path="/a", halt_path="/h").decide(REQ).allowed
    assert built["audit_path"] is None   # empty audit_path is passed as None


def test_factory_without_embedded_client_in_package_is_unavailable(monkeypatch):
    pkg = types.ModuleType("policy_client")
    base = types.ModuleType("policy_client.base")
    base.DecisionRequest = _FakeRequestType
    monkeypatch.setitem(sys.modules, "policy_client", pkg)
    monkeypatch.setitem(sys.modules, "policy_client.base", base)
    monkeypatch.setitem(sys.modules, "policy_client.embedded", None)
    gate = build_policy_client("embedded", policy_path="/p", approvals_path="/a", halt_path="/h")
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_UNAVAILABLE


# -- factory: service mode (4.1, RT-W04) ------------------------------------------------

SERVICE = dict(service_url="https://pm.example:8443/", client_cert="/etc/pm/rt.crt",
               client_key="/etc/pm/rt.key", ca="/etc/pm/ca.pem")


class _FakeServiceClient(_FakePolicyClient):
    built: dict = {}

    def __init__(self, url, cert, key, ca, timeout_s=5.0):
        super().__init__()
        type(self).built = dict(url=url, cert=cert, key=key, ca=ca, timeout_s=timeout_s)


def test_factory_builds_service_adapter_from_the_package_client(monkeypatch):
    _install_fake_package(monkeypatch, service_cls=_FakeServiceClient)
    gate = build_policy_client("service", service_timeout_s=3, **SERVICE)
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "service" and gate.policy_path == ""
    assert _FakeServiceClient.built == {"url": "https://pm.example:8443/", "cert": "/etc/pm/rt.crt",
                                        "key": "/etc/pm/rt.key", "ca": "/etc/pm/ca.pem", "timeout_s": 3.0}
    assert gate.decide(REQ).allowed and gate.decide(REQ).decision_id == "e-7"
    assert gate.health() == (True, "policy service healthy, not halted")   # no policy file check in service mode


@pytest.mark.parametrize("timeout", [None, 0, -1])
def test_factory_service_timeout_defaults_to_five_seconds(monkeypatch, timeout):
    _install_fake_package(monkeypatch, service_cls=_FakeServiceClient)
    build_policy_client("service", service_timeout_s=timeout, **SERVICE)
    assert _FakeServiceClient.built["timeout_s"] == 5.0


@pytest.mark.parametrize("drop", ["service_url", "client_cert", "client_key", "ca"])
def test_factory_service_without_a_field_denies_and_names_it(monkeypatch, drop):
    _install_fake_package(monkeypatch, service_cls=_FakeServiceClient)
    gate = build_policy_client("service", **{**SERVICE, drop: " "})
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_NO_POLICY_MODE
    expected = {"service_url": "policy_service_url", "client_cert": "policy_client_cert",
                "client_key": "policy_client_key", "ca": "policy_ca"}[drop]
    assert expected in gate.reason and gate.decide(REQ).denied


def test_factory_service_requires_https(monkeypatch):
    _install_fake_package(monkeypatch, service_cls=_FakeServiceClient)
    gate = build_policy_client("service", **{**SERVICE, "service_url": "http://pm.example:8443"})
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_NO_POLICY_MODE and "https://" in gate.reason


def test_factory_service_without_service_module_is_unavailable(monkeypatch):
    _install_fake_package(monkeypatch)   # P1 wheel layout: embedded only
    gate = build_policy_client("service", **SERVICE)
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_UNAVAILABLE
    assert "policy_client.service" in gate.reason and "P3" in gate.reason
    assert gate.decide(REQ).code == CODE_UNAVAILABLE and gate.health()[0] is False


def test_factory_service_constructor_failure_is_unavailable(monkeypatch):
    class Broken(_FakeServiceClient):
        def __init__(self, *a, **k):
            raise OSError("no such cert")

    _install_fake_package(monkeypatch, service_cls=Broken)
    gate = build_policy_client("service", **SERVICE)
    assert isinstance(gate, NoPolicyClient) and gate.code == CODE_UNAVAILABLE and "OSError" in gate.reason


def test_service_adapter_fails_closed_without_fallback(monkeypatch):
    """RT-W04: a failing service client is DENY UNAVAILABLE and halted; nothing falls back to embedded."""
    class Unreachable(_FakeServiceClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fail = True

    _install_fake_package(monkeypatch, service_cls=Unreachable)
    gate = build_policy_client("service", **SERVICE)
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "service"
    assert gate.decide(REQ).code == CODE_UNAVAILABLE
    assert gate.halt_status().halted is True
    assert gate.consume("f-1", REQ).ok is False
    assert gate.health()[0] is False


# -- action catalogue and sequences (4.2-1, RT-W09, AC-09) ------------------------------

BLOCK_WIRE = {"id": "isolate_endpoint", "version": 2, "zweck": "Endpoint isolieren", "kategorie": "Isolation",
              "zieltypen": ["device"], "parameter": {"dauer_h": "Dauer in Stunden"}, "mindeststufe": 3,
              "umkehrbar": False, "ruecknahme": "unisolate_endpoint", "mindest_gate": "APPROVE_2PERSON"}


def test_sequence_request_puts_steps_into_aktion_schritte():
    req = PolicyRequest(kind=ACTION_SEQUENCE, initiator="rc", agent_id="rc", run_id="rt-1", step_id="rt-1/2",
                        targets=("srv-1", "srv-2"), evidence_ids=("tb-1",),
                        steps=(sequence_step("isolate_endpoint", ["srv-1"], {"dauer_h": 4}),
                               sequence_step("disable_account", ("u-7",))))
    wire = req.to_wire()
    assert wire["aktion"]["art"] == "folge" and wire["aktion"]["ziel"] == ["srv-1", "srv-2"]
    assert wire["aktion"]["schritte"] == [
        {"baustein": "isolate_endpoint", "ziel": ["srv-1"], "parameter": {"dauer_h": 4}},
        {"baustein": "disable_account", "ziel": ["u-7"], "parameter": {}}]
    assert "schritte" not in REQ.to_wire()["aktion"]   # single built-in action: no steps on the wire


def test_sequence_step_normalises_malformed_input():
    assert sequence_step(None, "srv-1", "x") == {"baustein": "", "ziel": [], "parameter": {}}   # str is not a list
    assert sequence_step(None, ["srv-1"], "x") == {"baustein": "", "ziel": ["srv-1"], "parameter": {}}
    assert sequence_step("b", None, None) == {"baustein": "b", "ziel": [], "parameter": {}}


def test_catalog_block_parses_the_aktionen_fields():
    b = CatalogBlock.from_wire(BLOCK_WIRE)
    assert b == CatalogBlock(id="isolate_endpoint", version=2, purpose="Endpoint isolieren", category="Isolation",
                             target_types=("device",), parameters={"dauer_h": "Dauer in Stunden"}, min_stage=3,
                             reversible=False, undo="unisolate_endpoint", min_gate="APPROVE_2PERSON")
    minimal = CatalogBlock.from_wire({"id": "x"})
    assert (minimal.min_stage, minimal.reversible, minimal.min_gate, minimal.parameters) == (0, False, "", {})


@pytest.mark.parametrize("bad", [
    "isolate_endpoint", {}, {"id": ""}, {"id": 7}, {"id": "x", "mindeststufe": 5}, {"id": "x", "mindeststufe": "hoch"},
    {"id": "x", "mindest_gate": "MAYBE"}, {"id": "x", "version": "v2"},
])
def test_catalog_block_drops_unusable_entries(bad):
    assert CatalogBlock.from_wire(bad) is None


def test_catalog_block_never_claims_reversibility_without_true():
    assert CatalogBlock.from_wire({"id": "x", "umkehrbar": "yes"}).reversible is False
    assert CatalogBlock.from_wire({"id": "x", "umkehrbar": 1}).reversible is False
    assert CatalogBlock.from_wire({"id": "x", "umkehrbar": True}).reversible is True


def test_parse_catalog_keeps_order_and_skips_bad_items():
    items = [BLOCK_WIRE, "junk", {"id": "disable_account", "zieltypen": "user"}, None]
    assert [b.id for b in parse_catalog(items)] == ["isolate_endpoint", "disable_account"]
    assert parse_catalog(items)[1].target_types == ()
    assert parse_catalog({"aktionen": [BLOCK_WIRE]}) == [] and parse_catalog(None) == []


def test_reason_text_names_the_step_of_sequence_reasons():
    d = PolicyDecision.from_wire({"gate": "DENY", "code": "BLOCK_NOT_ALLOWED", "begruendung": [
        {"regel": "katalog", "gate": "DENY", "grund": "x not in catalogue", "schritt": 1},
        {"regel": "halt", "gate": "ALLOW", "grund": "not halted"}]})
    assert d.reason_text() == "x not in catalogue (schritt 1); not halted"


def test_adapter_actions_parse_the_package_answer():
    class WithActions(_FakePolicyClient):
        def actions(self, agent_id):
            self.calls.append(("actions", agent_id))
            return [BLOCK_WIRE, {"id": ""}]

    client = WithActions()
    gate = PolicyClientAdapter(client, _FakeRequestType, "service")
    assert [b.id for b in gate.actions("rc")] == ["isolate_endpoint"] and ("actions", "rc") in client.calls


def test_adapter_actions_is_empty_for_a_p1_client_or_a_failure():
    assert PolicyClientAdapter(_FakePolicyClient(), _FakeRequestType, "embedded").actions("rc") == []   # no actions()
    class Failing(_FakePolicyClient):
        def actions(self, agent_id):
            raise RuntimeError("boom")
    assert PolicyClientAdapter(Failing(), _FakeRequestType, "embedded").actions("rc") == []


def test_no_policy_client_and_stub_serve_the_catalogue():
    assert NoPolicyClient("x").actions("rc") == []
    block = CatalogBlock.from_wire(BLOCK_WIRE)
    stub = StubPolicyClient(blocks=[block])
    assert stub.actions("rc") == [block] and stub.actions_asked == ["rc"]
    assert StubPolicyClient().actions("rc") == []


# -- stub used by 3.3-3.6 --------------------------------------------------------------

def test_stub_records_requests_and_honours_halt():
    stub = StubPolicyClient()
    assert stub.decide(REQ).allowed and stub.requests == [REQ]
    stub.halted = True
    assert stub.decide(REQ).code == CODE_HALTED and stub.halt_status().halted
    approve = StubPolicyClient(PolicyDecision(gate=GATE_APPROVE_2PERSON, decision_id="e", approval_id="f-2",
                                              approvals_needed=2))
    assert approve.decide(REQ).needs_approval
    assert approve.approval_status("f-2").status == "pending"
    assert approve.consume("f-2", REQ).ok and approve.consumed == ["f-2"]
    assert StubPolicyClient(consume_ok=False).consume("f", REQ).code == "NOT_APPROVED"
    assert PolicyDecision(gate=GATE_ALLOW, decision_id="").allowed is False
