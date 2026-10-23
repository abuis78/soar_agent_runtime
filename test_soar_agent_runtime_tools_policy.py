"""ToolRegistry effect gate (Umsetzungskonzept Runtime, sub-change 3.3; RT-W05, AC-08, E4).

An effecting tool needs begruendung, gegenargumente and beleg_ids; every
beleg_id must have originated in this run (broker aufruf_id or an earlier
schritt_id). The check runs locally (OQ-8) before anything else happens.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from soar_agent_runtime_broker import InvokeResult
from soar_agent_runtime_policy import (
    CODE_HALTED,
    CODE_NO_POLICY_MODE,
    GATE_APPROVE_2PERSON,
    CatalogBlock,
    NoPolicyClient,
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


# --- RT-W11: no path around the gate ------------------------------------------------------

def test_direct_handler_call_in_airgap_is_refused(rules):
    reg = registry()
    result = reg._tool_update_severity({"severity": "critical"})
    assert result == "ERROR: 'update_severity' runs only through the policy gate in airgap_mode (RT-W11); nothing was changed."
    assert rules.calls == [] and reg._active_decision == ""


def test_active_decision_exists_only_while_the_allowed_handler_runs(rules):
    seen = []
    reg = registry()
    rules.update_container = lambda **kw: seen.append(reg._active_decision)
    assert reg.execute("update_severity", GOOD).startswith("Container severity updated")
    assert seen == ["pm-stub-1"] and reg._active_decision == ""


def test_active_decision_is_cleared_after_a_failing_handler(rules):
    def boom(**kw):
        raise RuntimeError("down")
    rules.update_container = boom
    reg = registry()
    reg.execute("update_severity", GOOD)
    assert reg._active_decision == ""
    assert reg._tool_update_severity({"severity": "low"}).startswith("ERROR: 'update_severity' runs only")


def test_outside_airgap_direct_handler_still_works(rules):
    reg = registry(airgap=False)
    assert reg._tool_update_severity({"severity": "low"}) == "Container severity updated to: low"


def test_every_effecting_handler_is_guarded_and_update_container_has_one_call_site():
    import inspect
    import soar_agent_runtime_utils as u
    source = inspect.getsource(u)
    assert source.count("ph_rules.update_container(") == 1
    assert "ph_rules.update_container(" in inspect.getsource(ToolRegistry._tool_update_severity)
    for name, defn in ToolRegistry.TOOL_DEFINITIONS.items():
        if defn.get("wirkt", True) and (handler := getattr(ToolRegistry, f"_tool_{name}", None)):
            assert "_require_decision(" in inspect.getsource(handler), name


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


# --- action catalogue: catalog_invoke (4.2-2; RT-W09, AC-09, E12-E14) ---------------------

ISOLATE = CatalogBlock(id="isolate_endpoint", version=2, purpose="Isolate an endpoint", category="Isolation",
                       target_types=("device",), parameters={"dauer_h": "hours"}, min_stage=1,
                       reversible=False, undo="unisolate_endpoint", min_gate="APPROVE_2PERSON")
DISABLE = CatalogBlock(id="disable_account", version=1, purpose="Disable an account", category="Identity",
                       target_types=("user",), min_stage=0, reversible=True, min_gate="APPROVE_1CLICK")
WIPE = CatalogBlock(id="wipe_host", version=1, purpose="Wipe", category="Destruction", min_stage=4)
CATALOG = [ISOLATE, DISABLE, WIPE]
STEPS = [{"baustein": "isolate_endpoint", "ziel": ["srv-1"], "parameter": {"dauer_h": 4}},
         {"baustein": "disable_account", "ziel": ["u-7", "srv-1"]}]
SEQ = {"schritte": STEPS, "begruendung": "lateral movement", "gegenargumente": "admin session",
       "beleg_ids": ["rt-x/1"]}


def catalog_registry(policy=None, stage=2, **extra):
    reg = registry(policy=policy or StubPolicyClient(blocks=CATALOG), stage=stage, **extra)
    reg.load_catalog()
    return reg


def test_load_catalog_filters_by_stage_and_hides_the_rest():
    reg = catalog_registry(stage=2)
    assert list(reg.catalog) == ["isolate_endpoint", "disable_account"] and [b.id for b in reg.catalog_hidden] == ["wipe_host"]
    assert reg._policy.actions_asked == ["triage"]
    assert any("2 block(s) available at stufe 2, 1 hidden" in m for m in reg._progress)
    assert list(catalog_registry(stage=None).catalog) == ["disable_account"]        # unknown stufe counts as 0
    assert list(catalog_registry(stage=4).catalog) == ["isolate_endpoint", "disable_account", "wipe_host"]


def test_catalog_is_empty_without_a_gate_or_outside_airgap():
    assert registry(policy=StubPolicyClient()).load_catalog() is False
    reg = registry(airgap=False, policy=StubPolicyClient(blocks=CATALOG))
    reg.load_catalog()
    assert reg.catalog and "catalog_invoke" in reg.get_tool_descriptions()   # the gate decides visibility, not airgap
    plain = registry(airgap=False)
    plain.load_catalog()
    assert plain.catalog == {} and "catalog_invoke" not in plain.get_tool_descriptions()


def test_catalog_load_survives_a_raising_gate():
    class Raising(StubPolicyClient):
        def actions(self, agent_id):
            raise RuntimeError("boom")
    reg = registry(policy=Raising())
    assert reg.load_catalog() is False and any("catalogue unavailable" in m for m in reg._progress)


def test_prompt_lists_blocks_with_their_facts_and_hidden_ones():
    text = catalog_registry(stage=2).get_tool_descriptions()
    assert "- catalog_invoke(schritte: JSON list of steps" in text and "beleg_ids:" in text
    assert "Catalogue building blocks (stufe 2):" in text
    assert ("  - isolate_endpoint v2: Isolate an endpoint. kategorie Isolation, zieltypen device, "
            "mindest_gate APPROVE_2PERSON, umkehrbar nein, ruecknahme unisolate_endpoint. Parameters: dauer_h (hours)") in text
    assert "  - disable_account v1: Disable an account. kategorie Identity, zieltypen user, mindest_gate APPROVE_1CLICK, umkehrbar ja. Parameters: none" in text
    assert "Not available at this stufe: wipe_host (stufe>=4)" in text
    assert "catalog_invoke" not in registry().get_tool_descriptions()          # no catalogue, no tool


def test_catalog_invoke_is_implicitly_allowed_only_with_blocks(rules):
    reg = registry(policy=StubPolicyClient())
    assert reg.execute("catalog_invoke", SEQ).startswith("ERROR: Tool 'catalog_invoke' is not in the allowed tools list")
    assert catalog_registry()._is_allowed("catalog_invoke") is True


@pytest.mark.parametrize("schritte, problem", [
    (None, "needs 'schritte'"), ([], "needs 'schritte'"), ("isolate_endpoint", "needs 'schritte'"),
    (["isolate_endpoint"], "step 0 is not an object"),
    ([{"baustein": "nmap_scan", "ziel": ["x"]}], "baustein 'nmap_scan' is not in the catalogue"),
    ([{"baustein": "wipe_host", "ziel": ["x"]}], "baustein 'wipe_host' needs stufe>=4"),
    ([{"baustein": "disable_account"}], "ziel must be a non-empty list of strings"),
    ([{"baustein": "disable_account", "ziel": "u-7"}], "ziel must be a non-empty list of strings"),
    ([{"baustein": "disable_account", "ziel": ["u-7"], "parameter": "x"}], "parameter must be an object"),
])
def test_bad_steps_are_refused_before_any_policy_request(rules, schritte, problem):
    reg = catalog_registry()
    result = reg.execute("catalog_invoke", {**SEQ, "schritte": schritte})
    assert result.startswith("ERROR:") and problem in result and "no policy request" in result
    assert reg._policy.requests == [] and reg._artifacts == [] and rules.calls == []


def test_sequence_is_one_decide_request_with_schritte(rules):
    """AC-09: two steps, one request, art 'folge', ziel = union of the step targets."""
    reg = catalog_registry()
    reg.execute("catalog_invoke", SEQ)
    req, = reg._policy.requests
    assert req.kind == "folge" and req.category == "" and req.params == {} and req.targets == ("srv-1", "u-7")
    assert req.judgement["begruendung"] == "lateral movement" and req.evidence_ids == ("rt-x/1",)
    wire = req.to_wire()["aktion"]
    assert wire["art"] == "folge" and wire["schritte"] == [
        {"baustein": "isolate_endpoint", "ziel": ["srv-1"], "parameter": {"dauer_h": 4}},
        {"baustein": "disable_account", "ziel": ["u-7", "srv-1"], "parameter": {}}]


def test_allow_records_the_sequence_for_the_playbook_and_touches_nothing(rules):
    """E13: ALLOW is a proposal with gate ALLOW; the runtime never runs a building block."""
    rules.notes = []
    rules.add_note = lambda **kw: rules.notes.append(kw)
    reg = catalog_registry()
    result = reg.execute("catalog_invoke", SEQ)
    assert result == ("ALLOWED: 2 step(s) recorded for the follow-up playbook (isolate_endpoint, disable_account); "
                      "not executed by this run (entscheidung_id=pm-stub-1)")
    assert rules.calls == [] and reg._policy.consumed == []
    [artifact] = reg._artifacts
    cef = artifact["cef"]
    assert artifact["label"] == "agent_action_proposal" and artifact["name"] == "folge: isolate_endpoint, disable_account -> srv-1, u-7"
    assert artifact["source_data_identifier"] == "pm-stub-1:rt-x/2" and artifact["run_automation"] is False
    assert (cef["gate"], cef["freigabe_id"], cef["entscheidung_id"], cef["aktion"]) == ("ALLOW", "", "pm-stub-1", "catalog_invoke")
    assert json.loads(cef["schritte"]) == [
        {"baustein": "isolate_endpoint", "ziel": ["srv-1"], "parameter": {"dauer_h": 4}},
        {"baustein": "disable_account", "ziel": ["u-7", "srv-1"], "parameter": {}}]
    assert cef["ziel"] == ["srv-1", "u-7"] and cef["beleg_ids"] == ["rt-x/1"]
    [note] = rules.notes
    assert note["title"] == "Freigegeben (ALLOW): folge: isolate_endpoint, disable_account -> srv-1, u-7"
    assert "- Gate: ALLOW" in note["content"] and "kein consume noetig" in note["content"] and "freigabe_id" not in note["content"]
    assert reg._gated is None and reg._active_decision == ""


def test_allow_with_unrecordable_proposal_is_an_error(rules):
    reg = catalog_registry(artifact_ok=False)
    result = reg.execute("catalog_invoke", SEQ)
    assert result.startswith("ERROR: proposal not recorded (database read-only); nothing will be executed")


def test_approval_gate_holds_the_sequence_with_freigabe_id(rules):
    rules.notes = []
    rules.add_note = lambda **kw: rules.notes.append(kw)
    reg = catalog_registry(policy=StubPolicyClient(APPROVE, blocks=CATALOG))
    result = reg.execute("catalog_invoke", SEQ)
    assert result.startswith("HELD: 'catalog_invoke' requires APPROVE_2PERSON (2 approval(s))") and "freigabe_id=f-2" in result
    [artifact] = reg._artifacts
    assert artifact["cef"]["freigabe_id"] == "f-2" and json.loads(artifact["cef"]["schritte"])[0]["baustein"] == "isolate_endpoint"
    assert rules.notes[0]["title"].startswith("Freigabe ausstehend: folge: isolate_endpoint") and "consume(freigabe_id)" in rules.notes[0]["content"]
    assert rules.calls == []


def test_deny_from_the_catalogue_rules_names_the_step(rules):
    deny = PolicyDecision.from_wire({"gate": "DENY", "entscheidung_id": "pm-9", "code": "BLOCK_NOT_ALLOWED", "begruendung": [
        {"regel": "katalog", "gate": "DENY", "grund": "disable_account not released for agent_id=triage", "schritt": 1}]})
    reg = catalog_registry(policy=StubPolicyClient(deny, blocks=CATALOG))
    result = reg.execute("catalog_invoke", SEQ)
    assert result == ("POLICY DENIED code=BLOCK_NOT_ALLOWED (disable_account not released for agent_id=triage (schritt 1)); "
                      "'catalog_invoke' was not executed.")
    assert reg._artifacts == [] and rules.calls == []


def test_catalog_handler_outside_the_gate_is_refused(rules):
    reg = catalog_registry()
    assert reg._tool_catalog_invoke(SEQ).startswith("ERROR: 'catalog_invoke' runs only through the policy gate")
    assert reg._artifacts == []


# --- RT-W10 / E15: report_gap ---------------------------------------------------------

GAP = {"lage": "C2 beaconing from srv-1 to a fresh domain", "massnahme": "Block the C2 domain at the proxy",
       "ziele": ["evil.example", " srv-1 "], "warum_kein_baustein": "no block covers proxy rules; isolation is too broad"}


def test_report_gap_exists_only_in_airgap_with_a_configured_gate():
    reg = catalog_registry(stage=2)
    assert reg._is_allowed("report_gap") is True and ToolRegistry.has_effect("report_gap") is False
    assert ("- report_gap(lage: What the situation is (text), massnahme: Which measure is needed (text), "
            "ziele: JSON list of targets (hosts, users, ...), may be empty, warum_kein_baustein: Why none of the "
            "listed building blocks fits (text)): Record that a measure is needed for which no catalogue building "
            "block fits; an analyst acts manually, nothing is executed. Use it when a measure is needed and no "
            "catalogue building block fits, then finish with FINAL_ANSWER") in reg.get_tool_descriptions()
    assert registry(policy=StubPolicyClient())._is_allowed("report_gap") is True          # gate, but empty catalogue
    for reg in (registry(airgap=False, policy=StubPolicyClient(blocks=CATALOG)),         # outside airgap (AC-15)
                registry(policy=NoPolicyClient("policy_mode not configured")),           # airgap without a gate
                registry(airgap=False, allowed=("add_note", "report_gap"))):             # allowlist does not help
        assert reg._is_allowed("report_gap") is False and "report_gap" not in reg.get_tool_descriptions()
        assert reg.execute("report_gap", GAP) == "ERROR: Tool 'report_gap' is not in the allowed tools list."
        assert reg._artifacts == []


@pytest.mark.parametrize("params, missing", [
    ({}, "'lage', 'massnahme', 'warum_kein_baustein'"),
    ({**GAP, "lage": "  "}, "'lage'"),
    ({**GAP, "massnahme": 7}, "'massnahme'"),
    ({**GAP, "warum_kein_baustein": None}, "'warum_kein_baustein'"),
    ({**GAP, "ziele": "evil.example"}, "'ziele'"),
    ({**GAP, "ziele": [1]}, "'ziele'"),
])
def test_report_gap_refuses_incomplete_input_without_recording(rules, params, missing):
    reg = catalog_registry()
    result = reg.execute("report_gap", params)
    assert result.startswith(f"ERROR: 'report_gap' needs {missing}:") and result.endswith("Nothing was recorded.")
    assert reg._artifacts == [] and reg._policy.requests == []


def test_report_gap_records_artifact_and_note_without_touching_the_gate(rules):
    rules.notes = []
    rules.add_note = lambda **kw: rules.notes.append(kw)
    reg = catalog_registry(stage=2, domain="endpoint")
    result = reg.execute("report_gap", GAP)
    assert result == ("GAP recorded (artifact agent_action_gap, 2 visible block(s) did not fit); "
                      "an analyst acts manually. Continue with FINAL_ANSWER.")
    assert reg._policy.requests == [] and rules.calls == []           # no decide(), nothing executed
    [artifact] = reg._artifacts
    assert (artifact["label"], artifact["type"], artifact["run_automation"]) == ("agent_action_gap", "agent_action_gap", False)
    assert artifact["name"] == "report_gap: Block the C2 domain at the proxy -> evil.example, srv-1"
    assert artifact["source_data_identifier"] == "rt-x:rt-x/2" and artifact["container_id"] == 4711
    assert artifact["cef"] == {
        "lauf_id": "rt-x", "schritt_id": "rt-x/2", "agent_id": "triage", "stufe": 2, "domaene": "endpoint",
        "lage": "C2 beaconing from srv-1 to a fresh domain", "massnahme": "Block the C2 domain at the proxy",
        "ziele": ["evil.example", "srv-1"], "grund": "no block covers proxy rules; isolation is too broad",
        "katalog_sichtbar": ["disable_account", "isolate_endpoint"], "katalog_verborgen": ["wipe_host"]}
    [note] = rules.notes
    assert note["container"] == 4711 and note["title"] == "Manuelle Handlungsempfehlung: Block the C2 domain at the proxy"
    for line in ("Agent `triage` sieht Handlungsbedarf", "(Lauf `rt-x`, Schritt `rt-x/2`)",
                 "- Lage: C2 beaconing from srv-1 to a fresh domain", "- Ziele: evil.example, srv-1",
                 "- Warum kein Baustein: no block covers proxy rules; isolation is too broad",
                 "- Sichtbare Bausteine (stufe 2): disable_account, isolate_endpoint", "- Verborgene Bausteine: wipe_host",
                 "Nichts wurde ausgefuehrt; ein Analyst entscheidet.", "Artefakt `agent_action_gap`"):
        assert line in note["content"], line


def test_report_gap_without_targets_or_catalogue_is_still_recorded(rules):
    rules.notes = []
    rules.add_note = lambda **kw: rules.notes.append(kw)
    reg = registry(policy=StubPolicyClient(), stage=None)       # gate configured, catalogue empty, stufe unknown
    result = reg.execute("report_gap", {**GAP, "ziele": []})
    assert result.startswith("GAP recorded (artifact agent_action_gap, 0 visible block(s) did not fit)")
    [artifact] = reg._artifacts
    assert artifact["name"] == "report_gap: Block the C2 domain at the proxy"
    assert artifact["cef"]["ziele"] == [] and artifact["cef"]["stufe"] is None and artifact["cef"]["katalog_sichtbar"] == []
    assert "- Ziele: keine genannt" in rules.notes[0]["content"] and "(stufe unbekannt): keine" in rules.notes[0]["content"]


def test_report_gap_failed_artifact_is_an_error(rules):
    reg = catalog_registry(artifact_ok=False)
    result = reg.execute("report_gap", GAP)
    assert result == ("ERROR: gap not recorded (database read-only); state the needed measure in FINAL_ANSWER instead.")

    def boom(artifact):
        raise RuntimeError("socket closed")
    reg = catalog_registry()
    reg.connector.save_artifact = boom
    assert reg.execute("report_gap", GAP).startswith("ERROR: gap not recorded (RuntimeError: socket closed)")


def test_report_gap_note_failure_is_reported_not_fatal(rules):
    def failing_note(**kw):
        raise RuntimeError("notes disabled")
    rules.add_note = failing_note
    reg = catalog_registry()
    assert reg.execute("report_gap", GAP).startswith("GAP recorded")
    assert len(reg._artifacts) == 1 and any("gap note failed: RuntimeError: notes disabled" in m for m in reg._progress)
