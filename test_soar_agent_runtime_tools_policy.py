"""ToolRegistry effect gate (Umsetzungskonzept Runtime, sub-change 3.3; RT-W05, AC-08, E4).

An effecting tool needs begruendung, gegenargumente and beleg_ids; every
beleg_id must have originated in this run (broker aufruf_id or an earlier
schritt_id). The check runs locally (OQ-8) before anything else happens.
"""

from __future__ import annotations

import sys
import types

import pytest

from soar_agent_runtime_broker import InvokeResult
from soar_agent_runtime_policy import (
    CODE_HALTED,
    CODE_NO_POLICY_MODE,
    GATE_APPROVE_2PERSON,
    PolicyDecision,
    StubPolicyClient,
)
from soar_agent_runtime_utils import ReActLoop, ToolRegistry

OK = InvokeResult(ok=True, aufruf_id="tb-1", vorlage_id="asset_kontext", vorlage_version=1,
                  katalog_version="k", zeilen=1, daten=[{"asset": "srv-1"}])
GOOD = {"severity": "high", "begruendung": "beacon confirmed", "gegenargumente": "could be a scanner",
        "beleg_ids": ["rt-x/1"]}


class FakeBroker:
    def list_tools(self, agent_id):
        from soar_agent_runtime_broker import ToolSpec, ToolsResponse
        return ToolsResponse(ok=True, katalog_version="k", vorlagen=(ToolSpec("asset_kontext", 1, "Asset", {}),))

    def invoke(self, vorlage_id, parameter, kontext):
        return OK


@pytest.fixture
def rules(monkeypatch):
    """Fake phantom.rules recording update_container calls."""
    pkg = types.ModuleType("phantom")
    mod = types.ModuleType("phantom.rules")
    mod.calls = []
    mod.update_container = lambda **kw: mod.calls.append(kw)
    monkeypatch.setitem(sys.modules, "phantom", pkg)
    monkeypatch.setitem(sys.modules, "phantom.rules", mod)
    return mod


def registry(airgap=True, allowed=("update_severity", "add_note", "broker_invoke"), broker=None, steps=2,
             policy=None, artifact_ok=True, **extra):
    progress, artifacts = [], []

    def save_artifact(artifact):
        artifacts.append(artifact)
        return (True, "ok", len(artifacts)) if artifact_ok else (False, "database read-only", None)

    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)),
                                      save_artifact=save_artifact)
    if policy is None:
        policy = StubPolicyClient()   # ALLOW with pm-stub-1 unless a test says otherwise
    reg = ToolRegistry(connector, 4711, list(allowed), agent_id="triage", broker=broker, run_id="rt-x",
                       airgap=airgap, policy=policy, **extra)
    reg._progress, reg._artifacts, reg._policy = progress, artifacts, policy
    for n in range(1, steps + 1):          # the loop calls set_step before every tool call
        reg.set_step(n, f"rt-x/{n}")
    return reg


# --- evidence bookkeeping --------------------------------------------------------

def test_valid_evidence_is_earlier_steps_plus_broker_results():
    reg = registry(broker=FakeBroker(), steps=3)
    assert reg.valid_evidence() == {"rt-x/1", "rt-x/2"}      # rt-x/3 is the current step
    reg.load_broker_tools()
    assert "BROKER RESULT" in reg.execute("broker_invoke", {"vorlage_id": "asset_kontext"})
    assert reg.valid_evidence() == {"rt-x/1", "rt-x/2", "tb-1"}
    assert reg.evidence_ids() == {"tb-1"}                      # FINAL_ANSWER path unchanged


# --- RT-W05: mandatory fields -----------------------------------------------------

@pytest.mark.parametrize("params, missing", [
    ({"severity": "high"}, "'begruendung', 'gegenargumente', 'beleg_ids'"),
    ({"severity": "high", "begruendung": " ", "gegenargumente": "x", "beleg_ids": ["rt-x/1"]}, "'begruendung'"),
    ({"severity": "high", "begruendung": "b", "gegenargumente": "g", "beleg_ids": []}, "'beleg_ids'"),
    ({"severity": "high", "begruendung": "b", "gegenargumente": "g", "beleg_ids": "rt-x/1"}, "'beleg_ids'"),
])
def test_effecting_tool_without_mandatory_fields_is_refused_before_execution(rules, params, missing):
    reg = registry()
    result = reg.execute("update_severity", params)
    assert result.startswith("ERROR: Tool 'update_severity' has an effect and needs " + missing)
    assert "Nothing was executed" in result
    assert rules.calls == []


# --- AC-08: fabricated evidence ---------------------------------------------------

@pytest.mark.parametrize("beleg_ids", [["tb-999"], ["rt-x/1", "rt-x/7"], ["rt-x/2"], [42]])
def test_fabricated_or_current_step_evidence_is_refused(rules, beleg_ids):
    reg = registry()                                           # current step = rt-x/2
    result = reg.execute("update_severity", {**GOOD, "beleg_ids": beleg_ids})
    assert result.startswith("ERROR: beleg_id ")
    assert "did not originate in this run" in result and "no policy request was made" in result
    assert "Valid evidence: rt-x/1" in result
    assert rules.calls == []


def test_valid_evidence_lets_the_call_through(rules):
    reg = registry(broker=FakeBroker(), steps=3)
    reg.load_broker_tools()
    reg.execute("broker_invoke", {"vorlage_id": "asset_kontext"})
    reg.set_step(4, "rt-x/4")
    result = reg.execute("update_severity", {**GOOD, "beleg_ids": ["tb-1", "rt-x/3"]})
    assert result == "Container severity updated to: high (entscheidung_id=pm-stub-1)"
    assert rules.calls == [{"container": 4711, "severity": "high"}]


def test_read_tools_are_not_affected(rules):
    # add_note has wirkt=False: no evidence demanded, the handler is reached.
    rules.add_note = lambda **kw: rules.calls.append(("note", kw))
    reg = registry()
    assert ToolRegistry.has_effect("add_note") is False
    result = reg.execute("add_note", {"title": "t", "content": "c"})
    assert "has an effect" not in result and not result.startswith("ERROR")
    assert rules.calls and rules.calls[0][0] == "note"


# --- E4: outside airgap unchanged -----------------------------------------------------

def test_outside_airgap_update_severity_runs_without_evidence(rules):
    reg = registry(airgap=False)
    assert reg.execute("update_severity", {"severity": "low"}) == "Container severity updated to: low"
    assert rules.calls == [{"container": 4711, "severity": "low"}]


# --- prompt ---------------------------------------------------------------------------

def test_prompt_lists_effect_params_only_in_airgap():
    airgap = registry().get_tool_descriptions()
    line = next(ln for ln in airgap.splitlines() if ln.startswith("- update_severity("))
    assert "begruendung:" in line and "gegenargumente:" in line and "beleg_ids:" in line
    assert "aufruf_id" in line and "schritt_id" in line
    add_note = next(ln for ln in airgap.splitlines() if ln.startswith("- add_note("))
    assert "beleg_ids" not in add_note
    normal = registry(airgap=False).get_tool_descriptions()
    assert "beleg_ids" not in normal


# --- RT-W06/W07: the gate ---------------------------------------------------------------

APPROVE = PolicyDecision(gate=GATE_APPROVE_2PERSON, decision_id="e-2", approval_id="f-2", approvals_needed=2,
                         expires_at="2026-10-20T10:00:00Z", policy_version="v1")


def test_allow_executes_and_sends_the_request_without_meta_params(rules):
    reg = registry()
    result = reg.execute("update_severity", GOOD)
    assert result == "Container severity updated to: high (entscheidung_id=pm-stub-1)"
    assert reg.last_decision_id == "pm-stub-1" and rules.calls == [{"container": 4711, "severity": "high"}]
    req = reg._policy.requests[0]
    assert req.kind == "update_severity" and req.params == {"severity": "high"}
    assert req.targets == ("container:4711",) and req.initiator == "triage" and req.agent_id == "triage"
    assert req.run_id == "rt-x" and req.step_id == "rt-x/2" and req.evidence_ids == ("rt-x/1",)
    assert req.judgement == {"begruendung": "beacon confirmed", "gegenargumente": "could be a scanner",
                             "unsicherheiten": ""}
    assert req.category == "Triage"
    assert req.stage is None and req.domain == "" and req.confidence is None
    wire = req.to_wire()["kontext"]
    assert "stufe" not in wire and "agent_konfidenz" not in wire and wire["domaene"] == ""
    assert "[policy] decide update_severity (step 2)" in reg._progress


def test_no_policy_client_in_airgap_denies_every_effecting_action(rules):
    progress = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)))
    reg = ToolRegistry(connector, 4711, ["update_severity"], agent_id="triage", run_id="rt-x", airgap=True)
    reg.set_step(1, "rt-x/1")
    reg.set_step(2, "rt-x/2")
    result = reg.execute("update_severity", GOOD)
    assert result.startswith(f"POLICY DENIED code={CODE_NO_POLICY_MODE}")
    assert "policy_mode not configured" in result and "was not executed" in result
    assert rules.calls == []


def test_approval_gate_records_proposal_and_does_not_execute(rules):
    rules.notes = []
    rules.add_note = lambda **kw: rules.notes.append(kw)
    reg = registry(policy=StubPolicyClient(APPROVE))
    result = reg.execute("update_severity", GOOD)
    assert result.startswith("HELD: 'update_severity' requires APPROVE_2PERSON (2 approval(s))")
    assert "freigabe_id=f-2" in result and "entscheidung_id=e-2" in result and "not executed" in result
    assert "consume(freigabe_id)" in result and "Do not retry" in result
    assert rules.calls == []                                    # AC-07: nothing executed
    assert reg.last_decision_id == "e-2"
    [artifact] = reg._artifacts
    assert artifact["label"] == "agent_action_proposal" and artifact["container_id"] == 4711
    assert artifact["run_automation"] is False and artifact["name"] == "update_severity -> high"
    assert artifact["source_data_identifier"] == "f-2:rt-x/2"
    cef = artifact["cef"]
    assert cef["freigabe_id"] == "f-2" and cef["entscheidung_id"] == "e-2" and cef["gate"] == "APPROVE_2PERSON"
    assert cef["aktion"] == "update_severity" and cef["ziel"] == ["container:4711"]
    assert cef["parameter"] == '{"severity": "high"}' and cef["beleg_ids"] == ["rt-x/1"]
    assert cef["lauf_id"] == "rt-x" and cef["schritt_id"] == "rt-x/2" and cef["agent_id"] == "triage"
    assert cef["begruendung"] == "beacon confirmed" and cef["ablauf"] == "2026-10-20T10:00:00Z"
    assert cef["freigaben_noetig"] == 2 and cef["policy_version"] == "v1"
    [note] = rules.notes
    assert note["container"] == 4711 and note["title"] == "Freigabe ausstehend: update_severity -> high"
    assert "f-2" in note["content"] and "consume(freigabe_id)" in note["content"]
    assert reg._policy.consumed == []                            # the runtime never consumes


def test_failed_proposal_artifact_is_an_error_and_nothing_runs(rules):
    reg = registry(policy=StubPolicyClient(APPROVE), artifact_ok=False)
    result = reg.execute("update_severity", GOOD)
    assert result.startswith("ERROR: proposal not recorded (database read-only)")
    assert "no approval is pending" in result and rules.calls == []


def test_failed_note_does_not_undo_the_proposal(rules):
    def boom(**kw):
        raise RuntimeError("notes down")
    rules.add_note = boom
    reg = registry(policy=StubPolicyClient(APPROVE))
    assert reg.execute("update_severity", GOOD).startswith("HELD:")
    assert len(reg._artifacts) == 1 and any("proposal note failed" in m for m in reg._progress)


def test_deny_reports_code_and_reason(rules):
    deny = PolicyDecision.deny("NOT_ALLOWED_FOR_AGENT", "triage may not change severity", "e-9")
    reg = registry(policy=StubPolicyClient(deny))
    result = reg.execute("update_severity", GOOD)
    assert result == ("POLICY DENIED code=NOT_ALLOWED_FOR_AGENT (triage may not change severity); "
                      "'update_severity' was not executed.")
    assert rules.calls == [] and reg.halted is False and reg._artifacts == []


def test_halted_policy_manager_halts_the_registry(rules):
    reg = registry(policy=StubPolicyClient(halted=True))
    result = reg.execute("update_severity", GOOD)
    assert result.startswith(f"POLICY DENIED code={CODE_HALTED}")
    assert reg.halted is True and reg.halt_source == "Policy Manager" and rules.calls == []


def test_allow_with_failing_handler_reports_the_error_with_decision_id(rules):
    def boom(**kw):
        raise RuntimeError("platform down")
    rules.update_container = boom
    assert registry().execute("update_severity", GOOD) == \
        "Failed to update severity: RuntimeError: platform down (entscheidung_id=pm-stub-1)"


def test_outside_airgap_the_gate_is_not_consulted(rules):
    reg = registry(airgap=False, policy=StubPolicyClient(halted=True))
    assert reg.execute("update_severity", {"severity": "low"}) == "Container severity updated to: low"
    assert reg._policy.requests == [] and reg.halted is False


# --- RT-W08: full request content ----------------------------------------------------------

def test_request_carries_stage_domain_and_full_judgement(rules):
    reg = registry(stage=2, domain="soc")
    params = {**GOOD, "unsicherheiten": "only one sensor", "konfidenz": 0.85}
    assert reg.execute("update_severity", params).startswith("Container severity updated to: high")
    req = reg._policy.requests[0]
    assert req.stage == 2 and req.domain == "soc" and req.confidence == 0.85
    assert req.params == {"severity": "high"}                   # meta fields never reach the action
    assert req.judgement == {"begruendung": "beacon confirmed", "gegenargumente": "could be a scanner",
                             "unsicherheiten": "only one sensor", "konfidenz": 0.85}
    wire = req.to_wire()
    assert wire["kontext"]["stufe"] == 2 and wire["kontext"]["agent_konfidenz"] == 0.85
    assert wire["kontext"]["domaene"] == "soc" and wire["kontext"]["principal_type"] == "agent"
    assert wire["kontext"]["initiator"] == "triage" and "reversib" not in str(wire)
    assert rules.calls == [{"container": 4711, "severity": "high"}]


@pytest.mark.parametrize("konfidenz", ["high", 1.5, -0.1, True, None, "0.9x"])
def test_unusable_confidence_is_dropped_not_rejected(rules, konfidenz):
    reg = registry()
    assert reg.execute("update_severity", {**GOOD, "konfidenz": konfidenz}).startswith("Container severity")
    req = reg._policy.requests[0]
    assert req.confidence is None and "konfidenz" not in req.judgement


@pytest.mark.parametrize("stage", [True, "2", 3.5, None])
def test_stage_is_an_int_or_unknown(stage):
    assert registry(stage=stage).stage is None
    assert registry(stage=0).stage == 0 and registry(stage=4).stage == 4


def test_prompt_marks_unsicherheiten_and_konfidenz_optional():
    line = next(ln for ln in registry().get_tool_descriptions().splitlines() if ln.startswith("- update_severity("))
    assert "unsicherheiten: optional" in line and "konfidenz: optional" in line


# --- loop: audit carries entscheidung_id, policy HALTED ends the run ----------------------

class FakeLLM:
    provider, model = "local", "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def chat(self, system_prompt, messages):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


class FakeSink:
    fail_closed = False

    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)
        return True

    def flush(self):
        return True


def make_loop(responses, policy):
    reg = registry(policy=policy, steps=0)
    llm, sink = FakeLLM(responses), FakeSink()
    loop = ReActLoop(connector=reg.connector, llm=llm, tools=reg, agent_id="triage", system_prompt="x",
                     max_steps=5, container_id=4711, run_id="rt-x", audit=sink)
    return loop, llm, reg, sink


ACT = ('ACTION: update_severity\nPARAMS: {"severity": "high", "begruendung": "b", "gegenargumente": "g", '
       '"beleg_ids": ["rt-x/1"]}')


def test_loop_audit_event_carries_entscheidung_id(rules):
    loop, llm, reg, sink = make_loop(["ACTION: get_container_info", ACT, "FINAL_ANSWER: done"], StubPolicyClient())
    rules.get_container = lambda cid: {"id": cid}
    rules.get_artifacts = lambda container_id: []
    loop.run("t")
    done = [e for e in sink.events if e["werkzeug"] == "update_severity" and e["phase"] == "erledigt"]
    assert done and done[0]["entscheidung_id"] == "pm-stub-1"
    assert rules.calls == [{"container": 4711, "severity": "high"}]


def test_loop_ends_when_the_policy_manager_is_halted(rules):
    loop, llm, reg, sink = make_loop(["ACTION: get_container_info", ACT, "FINAL_ANSWER: never"],
                                     StubPolicyClient(halted=True))
    rules.get_container = lambda cid: {"id": cid}
    rules.get_artifacts = lambda container_id: []
    result = loop.run("t")
    assert result["final_answer"] == "ERROR: run halted by Policy Manager (HALTED)"
    assert result["steps_taken"] == 2 and len(llm.seen) == 2 and rules.calls == []
    assert sink.events[-1]["phase"] == "fehler" and sink.events[-1]["werkzeug"] == "update_severity"


def test_loop_continues_after_a_held_proposal(rules):
    rules.add_note = lambda **kw: None
    loop, llm, reg, sink = make_loop(["ACTION: get_container_info", ACT, "FINAL_ANSWER: proposed"],
                                     StubPolicyClient(APPROVE))
    rules.get_container = lambda cid: {"id": cid}
    rules.get_artifacts = lambda container_id: []
    result = loop.run("t")
    assert result["final_answer"] == "proposed" and result["steps_taken"] == 3   # AC-07: no waiting
    assert "HELD: 'update_severity'" in llm.seen[2][-1]["content"]
    assert rules.calls == [] and len(reg._artifacts) == 1
