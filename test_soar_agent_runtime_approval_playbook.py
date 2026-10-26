"""Tests for the approval prompt (PM-F42, Policy Manager Admin decision B, app 2.1.0).

After an APPROVE_* decision with a freigabe_id the runtime starts the input
playbook named in approval_playbook on the run's container. Only the playbook
input carries the freigabe_id; a failed start keeps the action held and names
only the freigabe_hash. Artifact, follow-up playbook and consume stay as before.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest

import soar_agent_runtime_utils as utils_module
from soar_agent_runtime_policy import GATE_ALLOW, GATE_APPROVE_2PERSON, PolicyDecision, StubPolicyClient
from soar_agent_runtime_utils import ToolRegistry

GOOD = {"severity": "high", "begruendung": "beacon confirmed", "gegenargumente": "could be a scanner",
        "unsicherheiten": "keine erkannt", "konfidenz": 0.8, "beleg_ids": ["rt-x/1"]}
APPROVE = PolicyDecision(
    gate=GATE_APPROVE_2PERSON, decision_id="e-2", approval_id="f-secret-2", approvals_needed=2,
    expires_at="2026-10-17T07:20:00+00:00", policy_version="p7",
    reasons=({"regel": "stufe_default", "grund": "stufe 1", "gate": "APPROVE_1CLICK"},
             {"regel": "severity_high", "grund": "high needs two people", "gate": GATE_APPROVE_2PERSON}))
FHASH = hashlib.sha256(b"f-secret-2").hexdigest()[:16]


@pytest.fixture
def soar(monkeypatch):
    """Fake phantom.rules (notes) and SOAR REST (playbook_run); records both."""
    state = {"notes": [], "runs": [], "answer": (200, {"playbook_run_id": 77})}
    rules = types.ModuleType("phantom.rules")
    rules.add_note = lambda **kw: state["notes"].append(kw)
    monkeypatch.setitem(sys.modules, "phantom.rules", rules)
    monkeypatch.setattr(sys.modules["phantom"], "rules", rules, raising=False)

    def rest(method, url, body, timeout_s):
        if url.endswith("/rest/playbook_run"):
            state["runs"].append((method, body))
            status, data = state["answer"]
            if isinstance(data, Exception):
                raise data
            return status, data
        raise AssertionError(f"unexpected REST call {method} {url}")

    monkeypatch.setattr(utils_module.SoarRest, "transport", staticmethod(rest))
    return state


def registry(approval_playbook="soar_pm_admin/pm_freigabe", decision=APPROVE):
    progress, artifacts = [], []

    def save_artifact(artifact):
        artifacts.append(artifact)
        return True, "ok", len(artifacts)

    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)),
                                      save_artifact=save_artifact)
    reg = ToolRegistry(connector, 4711, ["update_severity"], agent_id="triage", run_id="rt-x", airgap=True,
                       policy=StubPolicyClient(decision), approval_playbook=approval_playbook)
    reg._progress, reg._artifacts = progress, artifacts
    reg.set_step(1, "rt-x/1")
    reg.set_step(2, "rt-x/2")
    return reg


def test_approval_gate_starts_the_playbook_with_the_input_contract(soar):
    reg = registry()
    result = reg.execute("update_severity", GOOD)
    assert result.startswith("HELD: 'update_severity' requires APPROVE_2PERSON (2 approval(s))")
    assert result.endswith("Approval prompt started (playbook soar_pm_admin/pm_freigabe).")
    [(method, body)] = soar["runs"]
    assert method == "post"
    assert {k: v for k, v in body.items() if k != "inputs"} == {
        "container_id": 4711, "playbook_id": "soar_pm_admin/pm_freigabe", "scope": "new", "run": True}
    assert isinstance(body["inputs"], str)                       # 8.7 REST: JSON-encoded string
    assert json.loads(body["inputs"]) == {
        "freigabe_id": "f-secret-2", "noetig": "2", "initiator": "triage", "art": "update_severity",
        "ziel": "container:4711", "begruendung_kurz": "severity_high: high needs two people",
        "ablauf": "2026-10-17T07:20:00+00:00"}
    assert any("approval prompt started: playbook run 77" in m for m in reg._progress)


def test_artifact_and_follow_up_data_are_unchanged(soar):
    reg = registry()
    reg.execute("update_severity", GOOD)
    [proposal] = [a for a in reg._artifacts if a["label"] == "agent_action_proposal"]
    assert proposal["cef"]["freigabe_id"] == "f-secret-2"     # the follow-up playbook still consumes with it


def test_numeric_playbook_id_is_sent_as_number(soar):
    registry(approval_playbook="42").execute("update_severity", GOOD)
    assert soar["runs"][0][1]["playbook_id"] == 42


def test_begruendung_kurz_is_capped_and_never_carries_the_id(soar):
    long_reason = {"regel": "r", "grund": "f-secret-2 " + "x" * 600, "gate": GATE_APPROVE_2PERSON}
    decision = PolicyDecision(gate=GATE_APPROVE_2PERSON, decision_id="e-3", approval_id="f-secret-2",
                              approvals_needed=2, reasons=(long_reason,))
    registry(decision=decision).execute("update_severity", GOOD)
    short = json.loads(soar["runs"][0][1]["inputs"])["begruendung_kurz"]
    assert len(short) == 500 and "f-secret-2" not in short


def test_no_matching_reason_gives_an_empty_begruendung_kurz(soar):
    decision = PolicyDecision(gate=GATE_APPROVE_2PERSON, decision_id="e-4", approval_id="f-4", approvals_needed=2)
    registry(decision=decision).execute("update_severity", GOOD)
    assert json.loads(soar["runs"][0][1]["inputs"])["begruendung_kurz"] == ""


@pytest.mark.parametrize("answer", [(404, {"failed": True, "message": "playbook not found"}),
                                    (200, ConnectionError("refused"))])
def test_failed_start_keeps_the_action_held_and_names_only_the_hash(soar, answer):
    soar["answer"] = answer
    reg = registry()
    result = reg.execute("update_severity", GOOD)
    assert result.startswith("Failed - approval prompt not started (")
    assert f"stays HELD (freigabe_hash={FHASH}), nothing was executed" in result
    assert "f-secret-2" not in result
    [note] = soar["notes"][-1:]
    assert note["title"] == "Freigabe-Anfrage nicht gestartet: update_severity"
    assert "approval prompt not started" in note["content"] and FHASH in note["content"]
    assert "f-secret-2" not in note["content"]
    assert utils_module._audit_phase(result, False) == "fehler"


def test_without_approval_playbook_nothing_is_started(soar):
    result = registry(approval_playbook="").execute("update_severity", GOOD)
    assert result.startswith("HELD:") and "Approval prompt" not in result
    assert soar["runs"] == []


def test_allow_and_deny_never_start_the_playbook(soar, monkeypatch):
    monkeypatch.setattr(utils_module.SoarRest, "transport", staticmethod(
        lambda m, url, b, t: (200, {"success": True}) if "/rest/container/" in url else (_ for _ in ()).throw(
            AssertionError("playbook_run must not be called"))))
    allow = PolicyDecision(gate=GATE_ALLOW, decision_id="e-5")
    assert registry(decision=allow).execute("update_severity", GOOD).endswith("(entscheidung_id=e-5)")
    deny = PolicyDecision.deny("RULE", "no")
    assert registry(decision=deny).execute("update_severity", GOOD).startswith("POLICY DENIED")


def test_app_json_declares_approval_playbook_off_by_default():
    with open("soar_agent_runtime.json") as fh:
        field = json.load(fh)["configuration"]["approval_playbook"]
    assert (field["data_type"], field["required"], field["order"]) == ("string", False, 39)
    assert "default" not in field


def test_connector_hands_the_asset_field_to_every_agent_of_the_run():
    from soar_agent_runtime_connector import SoarAgentRuntimeConnector
    c = SoarAgentRuntimeConnector()
    c._fake_config = {"approval_playbook": " soar_pm_admin/pm_freigabe "}
    assert c.initialize() is True
    run = {"run_id": "rt-x", "container_id": 7, "broker": None, "policy": None, "audit": None,
           "stage": None, "domain": ""}
    loop, error = c._build_loop(run, "triage", {"provider": "local", "allowed_tools": ["add_note"]}, 3, depth=0)
    assert error is None and loop.tools.approval_playbook == "soar_pm_admin/pm_freigabe"
