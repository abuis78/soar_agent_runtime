"""Integration tests against the bundled policy_manager wheel (sub-change 3.1b; RT-W01, W03, W06).

Run with the wheel on the path, e.g.
  uv run --with pytest --with ./wheels/py3/policy_manager-0.1.0-py3-none-any.whl ... -m pytest
Without the package the module is skipped; the stub-based tests cover the gate logic.
"""

from __future__ import annotations

import json
import re

import pytest

from soar_agent_runtime_policy import (
    CODE_HALTED,
    GATE_APPROVE_2PERSON,
    PolicyClientAdapter,
    PolicyRequest,
    build_policy_client,
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

pytest.importorskip("policy_client.embedded", reason="policy_manager wheel not installed")

POLICY = {"default_gate": "DENY", "categories": {"Triage": "ALLOW", "Containment": "APPROVE_2PERSON"}}
PM_ID = re.compile(r"^pm-[0-9a-f]{16}$")


def request(category="Triage", initiator="triage", **extra) -> PolicyRequest:
    return PolicyRequest(kind="update_severity", category=category, params={"severity": "high"},
                         targets=("container:11",), initiator=initiator, agent_id="triage",
                         run_id="rt-wheel", step_id="rt-wheel/2", evidence_ids=("rt-wheel/1",),
                         judgement={"begruendung": "b", "gegenargumente": "g"}, **extra)


@pytest.fixture
def state(tmp_path):
    """Policy file plus state dir with an explicit 'not halted' file, as the operator would set up."""
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "halt.json").write_text(json.dumps({"angehalten": False, "grund": ""}))
    return {"policy_path": str(policy), "policy_state_dir": str(state_dir), "dir": state_dir}


def gate_for(state):
    return build_policy_client("embedded", policy_path=state["policy_path"],
                               approvals_path=str(state["dir"] / "approvals.json"),
                               halt_path=str(state["dir"] / "halt.json"),
                               audit_path=str(state["dir"] / "audit.jsonl"))


def test_factory_builds_the_real_embedded_gate(state):
    gate = gate_for(state)
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "embedded"
    assert gate.health() == (True, "policy embedded healthy, not halted")
    assert gate.halt_status().halted is False


def test_allow_for_a_configured_category_carries_a_pm_decision_id(state):
    d = gate_for(state).decide(request("Triage"))
    assert d.allowed and PM_ID.match(d.decision_id), d
    assert re.fullmatch(r"[0-9a-f]{12}", d.policy_version)      # sha256 prefix of the policy file
    assert "base=ALLOW" in d.reason_text()


def test_unknown_category_falls_to_the_default_gate_deny(state):
    d = gate_for(state).decide(request("Other"))
    assert d.denied and not d.allowed and "base=DENY" in d.reason_text()


def test_missing_policy_file_denies_everything(state):
    state["policy_path"] = str(state["dir"] / "nope.json")
    gate = gate_for(state)
    assert gate.decide(request("Triage")).denied
    assert gate.health()[0] is True        # health is the halt check; a missing policy shows up as DENY per action


def test_halt_file_halts_agent_requests_and_is_visible_in_health(state):
    (state["dir"] / "halt.json").write_text(json.dumps({"angehalten": True, "grund": "drill"}))
    gate = gate_for(state)
    d = gate.decide(request("Triage"))
    assert d.denied and d.code == CODE_HALTED
    halt = gate.halt_status()
    assert halt.halted and halt.reason == "drill"
    ok, detail = gate.health()
    assert ok and "HALTED (drill)" in detail


def test_missing_halt_file_means_halted(state):
    (state["dir"] / "halt.json").unlink()
    gate = gate_for(state)
    assert gate.decide(request("Triage")).code == CODE_HALTED
    assert gate.halt_status().reason == "halt file missing"


def test_approval_gate_opens_an_approval_and_consume_without_approvals_fails(state):
    gate = gate_for(state)
    d = gate.decide(request("Containment"))
    assert d.needs_approval and d.gate == GATE_APPROVE_2PERSON and d.approvals_needed == 2
    assert d.approval_id and PM_ID.match(d.decision_id) and d.expires_at.endswith("+00:00")
    status = gate.approval_status(d.approval_id)
    assert (status.status, status.have, status.need) == ("pending", 0, 2)
    consumed = gate.consume(d.approval_id, request("Containment"))
    assert consumed.ok is False and consumed.code == "NOT_APPROVED"
    assert (state["dir"] / "approvals.json").is_file()


def test_approval_gate_without_initiator_is_denied_with_the_package_code(state):
    # P1 answers with the approval gate, code NO_INITIATOR and no freigabe_id; the runtime
    # reports DENY with that code (nothing is pending, nothing runs).
    d = gate_for(state).decide(request("Containment", initiator=""))
    assert d.denied and not d.needs_approval and not d.allowed
    assert d.code == "NO_INITIATOR" and "base=APPROVE_2PERSON" in d.reason_text()


# --- through the connector: asset fields -> real gate -> effecting tool ---------------

def test_airgap_run_executes_update_severity_through_the_real_gate(platform, llm, monkeypatch, state):  # noqa: F811 (imported fixtures)
    define_agent(platform, provider="local")
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: RUN_ID)
    llm["responses"] = effect_script("update_severity")
    c = make_connector(airgap_mode=True, policy_mode="embedded", policy_path=state["policy_path"],
                       policy_state_dir=state["policy_state_dir"])
    status, result = run(c)
    assert status is True, result.message
    assert platform["severity"] == [(11, "high")]
    obs = step_log(result)[1]["observation"]
    assert re.search(r"\(entscheidung_id=pm-[0-9a-f]{16}\)$", obs), obs
    audit = [a["cef"] for a in c.artifacts if a["label"] == "agent_audit"
             and a["cef"]["phase"] == "erledigt" and a["cef"]["schritt_id"] == f"{RUN_ID}/2"]
    assert audit and PM_ID.match(audit[0]["entscheidung_id"])
    assert any(m.startswith("[policy] mode=embedded (ready)") for m in c.progress)


def test_airgap_test_connectivity_reports_the_real_gate(platform, llm, monkeypatch, state):  # noqa: F811 (imported fixtures)
    from test_soar_agent_runtime_connector import BROKER_ASSET, _health_transport, _record_connection_tests
    _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True, policy_mode="embedded", policy_path=state["policy_path"],
                       policy_state_dir=state["policy_state_dir"], **BROKER_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy embedded healthy, not halted"' in c.action_results[-1].message
    (state["dir"] / "halt.json").write_text(json.dumps({"angehalten": True, "grund": "drill"}))
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy embedded healthy, HALTED (drill)"' in c.action_results[-1].message
