"""Integration tests against the bundled policy_manager wheel (3.1b, 4.1b; RT-W01, W03, W06, W09).

Run with the wheel on the path, e.g.
  uv run --with pytest --with ./wheels/py3/policy_manager-0.5.2-py3-none-any.whl ... -m pytest
Without the package the module is skipped; the stub-based tests cover the gate logic.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from soar_agent_runtime_policy import (
    ACTION_SEQUENCE,
    CODE_HALTED,
    GATE_APPROVE_1CLICK,
    GATE_APPROVE_2PERSON,
    PolicyClientAdapter,
    PolicyRequest,
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

pytest.importorskip("policy_client.embedded", reason="policy_manager wheel not installed")

# Policy for policy_core 0.5.1: tool names map to action kinds (SPEC-MCP §3.4), the
# caller's category is ignored for everything but run_playbook; agents are gated by
# stufe, domaene and a complete judgement (W14); one released building block (RT-W09).
POLICY = {
    "version": 1,
    "default_gate": "DENY",
    "categories": {"fall_aenderung": "ALLOW", "fall_loeschung": "APPROVE_2PERSON", "Isolation": "APPROVE_2PERSON",
                   "Network Restore": "APPROVE_1CLICK"},
    "tool_action_kinds": {"update_severity": "fall_aenderung", "delete_container": "fall_loeschung"},
    "irreversible_categories": ["Isolation"],
    "stage_min_gate": {"0": "ALLOW", "1": "ALLOW", "2": "ALLOW", "3": "APPROVE_1CLICK", "4": "APPROVE_2PERSON"},
    "building_blocks": {
        "isolate_endpoint": {"version": 2, "playbook": "isolate_endpoint", "purpose": "Isolate a device",
                             "category": "Isolation", "domain": "cyber", "target_types": ["device"],
                             "parameters": {"dauer_h": {"typ": "int", "min": 1, "max": 72}}, "min_stage": 1,
                             "loud": True, "min_gate": "APPROVE_2PERSON", "reversible": True, "undo": "unisolate_endpoint",
                             "agents": ["triage"]},
        "unisolate_endpoint": {"version": 1, "playbook": "unisolate_endpoint", "purpose": "Lift the isolation",
                               "category": "Network Restore", "domain": "cyber", "target_types": ["device"],
                               "min_stage": 1, "loud": False, "reversible": True, "undo": "isolate_endpoint",
                               "agents": ["triage"]},
    },
}
PM_ID = re.compile(r"^pm-[0-9a-f]{16}$")
JUDGEMENT = {"empfehlung": "beacon confirmed", "gegenargumente": ["could be a scanner"],
             "unsicherheiten": ["keine erkannt"], "konfidenz": 0.8}


def request(kind="update_severity", initiator="triage", **extra) -> PolicyRequest:
    fields = dict(kind=kind, params={"severity": "high"}, targets=("container:11",), initiator=initiator,
                  agent_id="triage", run_id="rt-wheel", step_id="rt-wheel/2", stage=2, domain="cyber",
                  evidence_ids=("rt-wheel/1",), judgement=dict(JUDGEMENT))
    fields.update(extra)
    return PolicyRequest(**fields)


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
                               audit_path=str(state["dir"] / "audit.jsonl"),
                               sequences_path=str(state["dir"] / "sequences.json"))


def test_factory_builds_the_real_embedded_gate(state):
    gate = gate_for(state)
    assert isinstance(gate, PolicyClientAdapter) and gate.mode == "embedded"
    assert gate.health() == (True, "policy embedded healthy, not halted")
    assert gate.halt_status().halted is False and gate.halt_status().known is True   # bekannt (0.5.1)


def test_mapped_tool_is_allowed_with_a_pm_decision_id(state):
    d = gate_for(state).decide(request())
    assert d.allowed and PM_ID.match(d.decision_id), d
    assert re.fullmatch(r"[0-9a-f]{12}", d.policy_version)      # sha256 prefix of the policy file
    assert "fall_aenderung" in d.reason_text() and "complete" in d.reason_text()


def test_unmapped_tool_falls_to_the_default_gate_deny(state):
    d = gate_for(state).decide(request(kind="run_query"))
    assert d.denied and not d.allowed and "default_gate" in d.reason_text()


def test_old_judgement_form_is_incomplete_for_the_engine(state):
    """W14: the P1-era judgement (begruendung/gegenargumente text) is denied by 0.5.x."""
    d = gate_for(state).decide(request(judgement={"begruendung": "b", "gegenargumente": "g", "unsicherheiten": ""}))
    assert d.denied and d.code == "INCOMPLETE_JUDGEMENT"
    assert "incomplete: empfehlung, gegenargumente, unsicherheiten, konfidenz" in d.reason_text()


def test_missing_domain_raises_the_gate_to_one_click(state):
    d = gate_for(state).decide(request(domain=""))
    assert d.needs_approval and d.gate == GATE_APPROVE_1CLICK and "domaene=<missing>" in d.reason_text()


def test_stage_four_needs_two_persons_even_for_an_allowed_kind(state):
    d = gate_for(state).decide(request(stage=4))
    assert d.needs_approval and d.gate == GATE_APPROVE_2PERSON and "stufe=4" in d.reason_text()


def test_missing_policy_file_denies_everything_and_is_unhealthy(state):
    state["policy_path"] = str(state["dir"] / "nope.json")
    gate = gate_for(state)
    assert gate.decide(request()).denied
    ok, detail = gate.health()
    assert ok is False and detail.startswith("policy embedded: policy file not readable (FileNotFoundError)")


def test_halt_file_halts_agent_requests_and_is_visible_in_health(state):
    (state["dir"] / "halt.json").write_text(json.dumps({"angehalten": True, "grund": "drill"}))
    gate = gate_for(state)
    d = gate.decide(request())
    assert d.denied and d.code == CODE_HALTED
    halt = gate.halt_status()
    assert halt.halted and halt.reason == "drill" and halt.known is True
    ok, detail = gate.health()
    assert ok and "HALTED (drill)" in detail


def test_missing_halt_file_means_halted_and_unhealthy(state):
    (state["dir"] / "halt.json").unlink()
    gate = gate_for(state)
    assert gate.decide(request()).code == CODE_HALTED
    halt = gate.halt_status()
    assert halt.halted and halt.reason == "halt file missing" and halt.known is False      # bekannt=false
    assert gate.health() == (False, 'policy embedded: halt file missing (create halt.json {"angehalten": false} in policy_state_dir)')


def test_approval_gate_opens_an_approval_and_consume_without_approvals_fails(state):
    gate = gate_for(state)
    d = gate.decide(request(kind="delete_container"))
    assert d.needs_approval and d.gate == GATE_APPROVE_2PERSON and d.approvals_needed == 2
    assert d.approval_id and PM_ID.match(d.decision_id) and d.expires_at.endswith("+00:00")
    status = gate.approval_status(d.approval_id)
    assert (status.status, status.have, status.need) == ("pending", 0, 2)
    consumed = gate.consume(d.approval_id, request(kind="delete_container"))
    assert consumed.ok is False and consumed.code == "NOT_APPROVED"
    assert (state["dir"] / "approvals.json").is_file()


def test_approval_gate_without_initiator_is_denied_with_the_package_code(state):
    d = gate_for(state).decide(request(kind="delete_container", initiator=""))
    assert d.denied and not d.needs_approval and not d.allowed
    assert d.code == "NO_INITIATOR" and "APPROVE_2PERSON" in d.reason_text()


def test_audit_file_records_every_decision_with_the_request(state):
    gate = gate_for(state)
    gate.decide(request())
    events = [json.loads(line) for line in (state["dir"] / "audit.jsonl").read_text().splitlines()]
    assert events and events[-1]["anfrage"]["kontext"]["urteil"] == JUDGEMENT      # P5: full request in the audit
    assert events[-1]["lauf_id"] == "rt-wheel" and PM_ID.match(events[-1]["entscheidung_id"])


# --- action catalogue (RT-W09) -------------------------------------------------------

def test_actions_lists_the_blocks_released_for_the_agent(state):
    gate = gate_for(state)
    block, undo = gate.actions("triage")
    assert undo.id == "unisolate_endpoint" and undo.reversible is True and undo.min_gate == "ALLOW"   # block default; the category gate applies at decide()
    assert (block.id, block.version, block.purpose, block.category) == ("isolate_endpoint", 2, "Isolate a device", "Isolation")
    assert block.target_types == ("device",) and block.min_stage == 1 and block.min_gate == "APPROVE_2PERSON"
    assert block.reversible is False and block.undo == "unisolate_endpoint"   # irreversible category wins (H4)
    assert gate.actions("someone_else") == []


def test_sequence_is_decided_as_one_request_with_the_catalogue_gate(state):
    gate = gate_for(state)
    seq = request(kind=ACTION_SEQUENCE, params={}, targets=("ws-0815",),
                  steps=(sequence_step("isolate_endpoint", ["ws-0815"], {"dauer_h": 2}),))
    d = gate.decide(seq)
    assert d.needs_approval and d.gate == GATE_APPROVE_2PERSON and d.approval_id
    assert "isolate_endpoint v2 (schritt 0)" in d.reason_text() and "single step" in d.reason_text()
    two = request(kind=ACTION_SEQUENCE, params={}, targets=("ws-0815",),
                  steps=(sequence_step("isolate_endpoint", ["ws-0815"], {"dauer_h": 2}),
                         sequence_step("unisolate_endpoint", ["ws-0815"])))
    d = gate.decide(two)
    assert d.needs_approval and "sequence not in register" in d.reason_text()   # register (sequences_path) consulted, P4
    unknown = request(kind=ACTION_SEQUENCE, params={}, targets=("x",), steps=(sequence_step("nmap_scan", ["x"]),))
    d = gate.decide(unknown)
    assert d.denied and d.code == "BLOCK_NOT_ALLOWED"


# --- through the connector: asset fields -> real gate -> effecting tool ---------------

def test_airgap_run_executes_update_severity_through_the_real_gate(platform, llm, monkeypatch, state):  # noqa: F811 (imported fixtures)
    define_agent(platform, provider="local")
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: RUN_ID)
    llm["responses"] = effect_script("update_severity")
    c = make_connector(airgap_mode=True, policy_mode="embedded", policy_path=state["policy_path"],
                       policy_state_dir=state["policy_state_dir"])
    status, result = run(c, stufe=2, domaene="cyber")
    assert status is True, result.message
    assert platform["severity"] == [(11, "high")]
    obs = step_log(result)[1]["observation"]
    assert re.search(r"\(entscheidung_id=pm-[0-9a-f]{16}\)$", obs), obs
    audit = [a["cef"] for a in c.artifacts if a["label"] == "agent_audit"
             and a["cef"]["phase"] == "erledigt" and a["cef"]["schritt_id"] == f"{RUN_ID}/2"]
    assert audit and PM_ID.match(audit[0]["entscheidung_id"])
    assert any(m.startswith("[policy] mode=embedded (ready)") for m in c.progress)
    prompt = llm["system_prompts"][0]
    assert "Catalogue building blocks (stufe 2):" in prompt and "- isolate_endpoint v2:" in prompt and "- unisolate_endpoint v1:" in prompt


@pytest.mark.parametrize("param, expected, proposals", [
    ({}, "POLICY DENIED code=DENY (", 0),                                   # no stufe: unknown -> default_gate
    ({"stufe": 2}, "HELD: 'update_severity' requires APPROVE_1CLICK", 1),   # no domaene: at least one click
])
def test_airgap_run_without_stufe_or_domaene_executes_nothing(platform, llm, monkeypatch, state, param, expected, proposals):  # noqa: F811
    """0.5.x gates agents by stufe and domaene (RT-W08): the playbook must pass both for an ALLOW."""
    define_agent(platform, provider="local")
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: RUN_ID)
    llm["responses"] = effect_script("update_severity")
    c = make_connector(airgap_mode=True, policy_mode="embedded", policy_path=state["policy_path"],
                       policy_state_dir=state["policy_state_dir"])
    status, result = run(c, **param)
    assert status is True, result.message
    assert platform["severity"] == []
    obs = step_log(result)[1]["observation"]
    assert obs.startswith(expected), obs
    assert len([a for a in c.artifacts if a["label"] == "agent_action_proposal"]) == proposals


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
    (state["dir"] / "halt.json").unlink()
    assert c._handle_test_connectivity({}) is False
    assert '"policy": "ERROR: policy embedded: halt file missing (create halt.json' in c.action_results[-1].message
    (state["dir"] / "halt.json").write_text(json.dumps({"angehalten": False, "grund": ""}))
    pathlib.Path(state["policy_path"]).write_text("[]")
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).") and '"policy": "ERROR: policy embedded: policy file is not a JSON object' in msg
