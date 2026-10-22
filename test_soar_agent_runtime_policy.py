"""Policy gate interface tests (Umsetzungskonzept Runtime, sub-change 3.0; RT-W01, W03, W06, W08)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from soar_agent_runtime_policy import (
    APPROVAL_GATES,
    CODE_HALTED,
    CODE_INVALID_RESPONSE,
    CODE_NO_POLICY_MODE,
    CODE_UNAVAILABLE,
    GATE_ALLOW,
    GATE_APPROVE_2PERSON,
    GATE_DENY,
    ApprovalState,
    ConsumeState,
    HaltState,
    NoPolicyClient,
    PolicyClientAdapter,
    PolicyDecision,
    PolicyRequest,
    StubPolicyClient,
    build_policy_client,
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
        assert gate.code == CODE_UNAVAILABLE and "Phase 4" in gate.reason
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


def test_adapter_treats_malformed_package_answer_as_deny():
    gate = PolicyClientAdapter(_FakePolicyClient(decision={"gate": "ALLOW"}), _FakeRequestType, "embedded")
    assert gate.decide(REQ).code == CODE_INVALID_RESPONSE


def _install_fake_package(monkeypatch, embedded_cls):
    pkg = types.ModuleType("policy_client")
    base = types.ModuleType("policy_client.base")
    base.DecisionRequest = _FakeRequestType
    emb = types.ModuleType("policy_client.embedded")
    emb.EmbeddedPolicyClient = embedded_cls
    for name, mod in (("policy_client", pkg), ("policy_client.base", base), ("policy_client.embedded", emb)):
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
