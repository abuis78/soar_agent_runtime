"""Phase-1 acceptance tests (Umsetzungskonzept Runtime, sub-change 1.12).

One test per acceptance criterion of the specification (§6), all end-to-end
through SoarAgentRuntimeConnector._handle_run_agent on the fake platform
from conftest.py plus a stubbed phantom.rules (Custom List, notes, severity).

AC-01 airgap: no run with a cloud provider, no outbound connection
AC-10 FINAL_ANSWER without mandatory fields is rejected after one correction round
AC-11 invalid PARAMS JSON never calls a tool
AC-12 every step is audited with lauf_id/schritt_id; no effect when audit is not writable
AC-15 without airgap_mode the app behaves as before
"""

from __future__ import annotations

import json
import socket
import sys
import types

import pytest

import soar_agent_runtime_connector as connector_module
import soar_agent_runtime_utils as utils
from soar_agent_runtime_connector import SoarAgentRuntimeConnector
from soar_agent_runtime_policy import (GATE_APPROVE_2PERSON, GATE_DENY, CatalogBlock, PolicyClientAdapter,
                                       PolicyDecision, StubPolicyClient)

LOCAL_URL = "http://llm.internal:11434/v1"
RUN_ID_RE = r"^rt-\d{8}T\d{6}Z-[0-9a-f]{8}$"

VALID_VERDICT = json.dumps({
    "belege": [], "begruendung": "nothing suspicious", "gegenargumente": ["could be stealthy"],
    "unsicherheiten": ["no EDR data"], "konfidenz": 0.6, "datenlage": "teilweise",
})


# --- platform stubs ---------------------------------------------------------

@pytest.fixture
def platform(monkeypatch):
    """Stub phantom.rules: Custom List with agent definitions, notes, severity."""
    state = {"agents": {}, "notes": [], "severity": [], "queries": []}
    rules = types.ModuleType("phantom.rules")
    rules.get_list = lambda list_name: [[aid, json.dumps(cfg)] for aid, cfg in state["agents"].items()]
    rules.add_note = lambda **kw: state["notes"].append(kw)
    rules.update_container = lambda container=None, severity=None: state["severity"].append((container, severity))
    rules.get_container = lambda cid: {"id": cid, "name": "case"}
    rules.get_artifacts = lambda **kw: []
    rules.run_query = lambda **kw: state["queries"].append(kw) or []
    monkeypatch.setitem(sys.modules, "phantom.rules", rules)
    monkeypatch.setattr(sys.modules["phantom"], "rules", rules, raising=False)
    return state


@pytest.fixture
def llm(monkeypatch):
    """Fake OpenAI-compatible and Anthropic clients fed from a response script."""
    rec = {"responses": [], "openai_base_urls": [], "anthropic_clients": 0, "calls": 0, "system_prompts": []}

    def _next():
        rec["calls"] += 1
        return rec["responses"].pop(0)

    class _Completions:
        def create(self, **kw):
            for m in kw.get("messages", []):
                if m.get("role") == "system":
                    rec["system_prompts"].append(m["content"])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=_next()))])

    class FakeOpenAI:
        def __init__(self, **kw):
            rec["openai_base_urls"].append(kw.get("base_url"))
            self.chat = types.SimpleNamespace(completions=_Completions())

    class _Messages:
        def create(self, **kw):
            return types.SimpleNamespace(content=[types.SimpleNamespace(text=_next())])

    class FakeAnthropic:
        def __init__(self, **kw):
            rec["anthropic_clients"] += 1
            self.messages = _Messages()

    monkeypatch.setattr(utils, "OpenAI", FakeOpenAI, raising=False)
    monkeypatch.setattr(utils, "HAS_OPENAI", True)
    monkeypatch.setattr(utils, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic), raising=False)
    monkeypatch.setattr(utils, "HAS_ANTHROPIC", True)
    return rec


def make_connector(**config) -> SoarAgentRuntimeConnector:
    c = SoarAgentRuntimeConnector()
    c._fake_config = {"anthropic_api_key": "k", "local_llm_url": LOCAL_URL, **config}
    assert c.initialize() is True
    return c


def define_agent(platform, agent_id="triage", **cfg):
    platform["agents"][agent_id] = {"agent_id": agent_id, "allowed_tools": ["add_note", "update_severity", "get_container_info"],
                                    "max_steps": 6} | cfg


def run(c, agent_id="triage", container_id=11):
    status = c._handle_run_agent({"agent_id": agent_id, "task": "triage this case", "container_id": container_id})
    return status, c.action_results[-1]


def step_log(result):
    return json.loads(result.data[0]["step_log"])


@pytest.fixture
def no_network(monkeypatch):
    def refuse(self, *a, **k):
        raise AssertionError(f"outbound connection attempted: {a}")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# --- AC-01 ------------------------------------------------------------------

def test_ac01_airgap_rejects_cloud_provider_before_any_model_call(platform, llm, no_network):
    define_agent(platform, provider="anthropic")
    c = make_connector(airgap_mode=True)
    status, result = run(c)
    assert status is False
    assert result.message.startswith("airgap_mode: provider 'anthropic' is not allowed, only 'local'")
    assert llm["calls"] == 0 and llm["anthropic_clients"] == 0
    assert c.artifacts == []


def test_ac01_airgap_rejects_cloud_asset_default_when_definition_has_no_provider(platform, llm, no_network):
    define_agent(platform)  # no provider -> default_provider (anthropic)
    c = make_connector(airgap_mode=True, default_provider="anthropic")
    status, result = run(c)
    assert status is False and "airgap_mode" in result.message
    assert llm["calls"] == 0


def test_ac01_airgap_local_run_talks_only_to_the_local_endpoint(platform, llm, no_network):
    define_agent(platform, provider="local")
    llm["responses"] = ["THOUGHT: look\nACTION: get_container_info", f"FINAL_ANSWER: {VALID_VERDICT}"]
    c = make_connector(airgap_mode=True)
    status, result = run(c)
    assert status is True, result.message
    assert set(llm["openai_base_urls"]) == {LOCAL_URL}
    assert llm["anthropic_clients"] == 0


def test_ac01_airgap_test_connectivity_contacts_only_local_and_broker(llm, no_network):
    """Airgap test connectivity: local model + Tool Broker health (2.6) + Policy Manager health (3.7), nothing else."""
    llm["responses"] = ["OK"]
    broker_calls = []

    def broker_transport(method, url, params, body, timeout_s):
        broker_calls.append((method, url))
        return 200, {"status": "ok"}

    c = make_connector(airgap_mode=True, openai_api_key="k", gemini_api_key="k",
                       broker_url="https://broker.internal:8443", broker_client_cert="/etc/rt.crt",
                       broker_client_key="/etc/rt.key", broker_ca="/etc/ca.pem", policy_mode="embedded")
    c._broker_transport = broker_transport
    c._policy_client = StubPolicyClient()
    status = c._handle_test_connectivity({})
    assert status is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (airgap).")
    assert set(llm["openai_base_urls"]) == {LOCAL_URL} and llm["anthropic_clients"] == 0
    assert broker_calls == [("GET", "https://broker.internal:8443/health")]
    assert '"broker": "OK: broker healthy (HTTP 200)"' in msg
    assert '"policy": "OK: policy stub healthy"' in msg
    for p in ("anthropic", "openai", "gemini"):
        assert f'"{p}": "SKIPPED (airgap_mode)"' in msg


# --- AC-10 ------------------------------------------------------------------

def test_ac10_final_answer_without_mandatory_fields_is_rejected_after_one_correction(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = ["FINAL_ANSWER: looks fine to me", 'FINAL_ANSWER: {"verdict": "benign"}']
    c = make_connector(airgap_mode=True)
    status, result = run(c)
    assert status is False
    assert result.message.startswith("ERROR: FINAL_ANSWER invalid after correction round: missing field 'belege'")
    assert llm["calls"] == 2 and result.summary["steps_taken"] == 2
    log = step_log(result)
    assert log[0]["observation"].startswith("ERROR: FINAL_ANSWER invalid: FINAL_ANSWER must be a JSON object")


def test_ac10_corrected_final_answer_in_second_round_is_accepted(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = ["FINAL_ANSWER: looks fine to me", f"FINAL_ANSWER: {VALID_VERDICT}"]
    c = make_connector(airgap_mode=True)
    status, result = run(c)
    assert status is True
    assert result.data[0]["urteil"]["datenlage"] == "teilweise"
    assert result.summary["steps_taken"] == 2


# --- AC-11 ------------------------------------------------------------------

@pytest.mark.parametrize("airgap,final", [(True, f"FINAL_ANSWER: {VALID_VERDICT}"), (False, "FINAL_ANSWER: done")])
def test_ac11_invalid_params_json_never_calls_a_tool(platform, llm, airgap, final):
    define_agent(platform, provider="local")
    llm["responses"] = [
        'THOUGHT: x\nACTION: update_severity\nPARAMS: {"severity": "high"',      # broken JSON
        'THOUGHT: x\nACTION: add_note\nPARAMS: ["not", "an", "object"]',         # valid JSON, not an object
        'THOUGHT: x\nACTION: add_note\nPARAMS: {"title": "t", "content": "c"}',  # corrected
        final,
    ]
    c = make_connector(airgap_mode=airgap)
    status, result = run(c)
    assert status is True, result.message
    assert platform["severity"] == []
    assert [n["title"] for n in platform["notes"] if n["title"] == "t"] == ["t"]
    log = step_log(result)
    assert log[0]["observation"] == log[1]["observation"] == utils._PARAMS_PARSE_ERROR
    assert log[2]["observation"] == "Note added: 't'"


# --- AC-12 ------------------------------------------------------------------

def test_ac12_every_step_is_audited_with_lauf_id_and_schritt_id(platform, llm, monkeypatch):
    define_agent(platform, provider="local")
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: "rt-test")
    llm["responses"] = [
        "THOUGHT: a\nACTION: get_container_info",
        # RT-W05 (3.3): an effecting call cites evidence from this run - step 1 here.
        'THOUGHT: b\nACTION: update_severity\nPARAMS: {"severity": "high", "begruendung": "beacon confirmed", '
        '"gegenargumente": "could be a scanner", "beleg_ids": ["rt-test/1"]}',
        f"FINAL_ANSWER: {VALID_VERDICT}",
    ]
    c = make_connector(airgap_mode=True)
    c._policy_client = StubPolicyClient()                     # ALLOW (3.1a injection point)
    status, result = run(c, container_id=23)
    assert status is True, result.message
    lauf_id = result.summary["lauf_id"]
    cefs = [a["cef"] for a in c.artifacts]
    assert all(a["label"] == "agent_audit" and a["container_id"] == 23 and a["run_automation"] is False for a in c.artifacts)
    assert [(e["schritt_id"], e["werkzeug"], e["phase"]) for e in cefs] == [
        (f"{lauf_id}/1", "get_container_info", "erledigt"),
        (f"{lauf_id}/2", "update_severity", "beabsichtigt"),
        (f"{lauf_id}/2", "update_severity", "erledigt"),
        (f"{lauf_id}/3", "FINAL_ANSWER", "erledigt"),
    ]
    assert all(e["lauf_id"] == lauf_id for e in cefs)
    assert platform["severity"] == [(23, "high")]
    assert cefs[2]["entscheidung_id"] == "pm-stub-1"
    assert result.summary["audit_ereignisse"] == 4 and result.summary["audit_fehler"] == 0


def test_ac12_no_effect_when_audit_is_not_writable(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = ['THOUGHT: b\nACTION: update_severity\nPARAMS: {"severity": "high"}', "unused"]
    c = make_connector(airgap_mode=True)
    c.save_artifact = lambda artifact: (False, "database read-only", None)
    status, result = run(c)
    assert status is False
    assert result.message.startswith("ERROR: audit not writable at step ")
    assert "before 'update_severity'" in result.message
    assert platform["severity"] == []            # the effect did NOT happen
    assert llm["calls"] == 1                     # run stopped right there


def test_ac12_audit_is_mandatory_in_airgap_even_if_audit_enabled_is_false(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = [f"FINAL_ANSWER: {VALID_VERDICT}"]
    c = make_connector(airgap_mode=True, audit_enabled=False)
    status, result = run(c)
    assert status is True and len(c.artifacts) == 1


# --- AC-15 ------------------------------------------------------------------

def test_ac15_without_airgap_the_app_behaves_as_before(platform, llm):
    define_agent(platform)  # no provider -> asset default anthropic, like today
    llm["responses"] = [
        "THOUGHT: a\nACTION: get_container_info",
        'THOUGHT: b\nACTION: update_severity\nPARAMS: {"severity": "high"',   # broken -> error observation, run continues
        'THOUGHT: c\nACTION: update_severity\nPARAMS: {"severity": "high"}',
        "THOUGHT: d\nFINAL_ANSWER: The alert is a true positive. Severity raised.",
    ]
    c = make_connector()
    status, result = run(c, container_id=5)
    assert status is True
    assert result.message == "Agent completed in 4 steps."
    assert llm["anthropic_clients"] >= 1 and llm["openai_base_urls"] == []
    assert result.data[0]["final_answer"] == "The alert is a true positive. Severity raised."
    assert result.data[0]["urteil"] is None                   # no contract check outside airgap
    assert c.artifacts == []                                  # no audit artifacts by default
    assert platform["severity"] == [(5, "high")]
    assert [n["title"] for n in platform["notes"]][-1].startswith("[Agent: triage]")  # summary note as before
    assert {"agent_id", "steps_taken", "provider", "model"} <= set(result.summary)
    assert result.summary["provider"] == "anthropic"
    assert not any("[airgap]" in m for m in c.progress)


def test_ac15_normal_mode_prompt_has_no_mandatory_field_contract(platform, monkeypatch):
    define_agent(platform, provider="local")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            self.run_id = "rt-x"

        def run(self, task):
            return {"agent_id": "triage", "lauf_id": "rt-x", "final_answer": "ok", "steps_taken": 1, "provider": "local",
                    "model": "m", "audit_ereignisse": 0, "audit_fehler": 0, "urteil": None, "step_log": "[]"}

    monkeypatch.setattr(connector_module, "ReActLoop", Spy)
    c = make_connector()
    run(c)
    assert seen["strict_schema"] is False and seen["required_keys"] == []
    assert type(seen["audit"]).__name__ == "NullAuditSink"


def test_ac15_app_json_new_fields_are_optional_with_backward_compatible_defaults():
    with open("soar_agent_runtime.json") as f:
        cfg = json.load(f)["configuration"]
    expected = {"airgap_mode": False, "audit_enabled": False, "max_observation_chars": 8000,
                "llm_timeout_s": 120, "llm_temperature": 0, "llm_max_tokens": 4096,
                "broker_timeout_s": 30}
    for field in ("broker_url", "broker_client_cert", "broker_client_key", "broker_ca"):
        assert cfg[field]["required"] is False and "default" not in cfg[field], field
    for field, default in expected.items():
        assert cfg[field]["required"] is False, field
        assert cfg[field]["default"] == default, field
    assert cfg["llm_seed"]["required"] is False and "default" not in cfg["llm_seed"]
    assert cfg["default_provider"]["default"] == "anthropic"


# ============================================================================
# Phase 2 (Tool Broker): AC-02, AC-03, AC-04
# ============================================================================

BROKER_FIELDS = {"broker_url": "https://broker.internal:8443", "broker_client_cert": "/etc/rt.crt",
                 "broker_client_key": "/etc/rt.key", "broker_ca": "/etc/ca.pem"}
E10_TOOLS = {"katalog_version": "k7", "angehalten": False, "vorlagen": [
    {"vorlage_id": "cve_offen_auf_assets", "version": 3, "beschreibung": "Offene Assets zu einer CVE",
     "parameter": {"cve": {"typ": "string", "muster": "^CVE-\\d{4}-\\d{4,7}$", "pflicht": True}}},
]}
INVOKE_OK = {"aufruf_id": "tb-01HZZ", "vorlage_id": "cve_offen_auf_assets", "vorlage_version": 3,
             "katalog_version": "k7", "zeilen": 1, "gekuerzt": False, "ergebnis_hash": "sha256:1",
             "daten": [{"asset": "srv-fs-01", "status": "offen"}]}


class BrokerStub:
    """Scripted broker transport: records every call, answers per path."""

    def __init__(self, tools=E10_TOOLS, invoke=INVOKE_OK, raise_exc=None):
        self.tools, self.invoke, self.raise_exc = tools, invoke, raise_exc
        self.calls: list[tuple] = []

    def __call__(self, method, url, params, body, timeout_s):
        self.calls.append((method, url, params, body))
        if self.raise_exc:
            raise self.raise_exc
        if url.endswith("/v1/tools"):
            return 200, self.tools
        if url.endswith("/v1/invoke"):
            return 200, self.invoke
        return 200, {"status": "ok"}

    def invokes(self):
        return [c for c in self.calls if c[1].endswith("/v1/invoke")]


def verdict(belege, datenlage="teilweise"):
    return "FINAL_ANSWER: " + json.dumps({"belege": belege, "begruendung": "b", "gegenargumente": [],
                                          "unsicherheiten": [], "konfidenz": 0.5, "datenlage": datenlage})


def airgap_connector(stub):
    c = make_connector(airgap_mode=True, **BROKER_FIELDS)
    c._broker_transport = stub
    return c


# --- AC-02 ------------------------------------------------------------------

def test_ac02_airgap_has_no_free_spl_tool_and_prompt_tools_come_from_v1_tools(platform, llm, no_network):
    define_agent(platform, provider="local", allowed_tools=["splunk_search", "add_note"])
    stub = BrokerStub()
    llm["responses"] = ['THOUGHT: x\nACTION: splunk_search\nPARAMS: {"spl_query": "index=main"}', verdict([])]
    c = airgap_connector(stub)
    status, result = run(c)
    assert status is True, result.message
    prompt = llm["system_prompts"][0]
    assert "splunk_search" not in prompt
    assert "Broker templates (catalog k7)" in prompt and "cve_offen_auf_assets v3" in prompt
    assert platform["queries"] == []                          # no SPL reached the platform
    assert step_log(result)[0]["observation"].startswith("ERROR: Tool 'splunk_search' is disabled in airgap_mode")
    assert stub.calls[0][:3] == ("GET", "https://broker.internal:8443/v1/tools", {"agent_id": "triage"})
    assert any("[airgap] 'splunk_search' removed from allowed tools" in m for m in c.progress)


# --- AC-03 ------------------------------------------------------------------

def test_ac03_external_reads_go_through_broker_invoke_and_yield_an_aufruf_id(platform, llm, no_network):
    define_agent(platform, provider="local")
    stub = BrokerStub()
    llm["responses"] = [
        'THOUGHT: look up\nACTION: broker_invoke\nPARAMS: {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-12345"}}',
        'THOUGHT: other case?\nACTION: get_container_info\nPARAMS: {"container_id": 999}',
        verdict(["tb-01HZZ"]),
    ]
    c = airgap_connector(stub)
    status, result = run(c, container_id=11)
    assert status is True, result.message
    lauf_id = result.summary["lauf_id"]
    (_, _, _, body), = stub.invokes()
    assert body["vorlage_id"] == "cve_offen_auf_assets" and body["parameter"] == {"cve": "CVE-2026-12345"}
    assert body["kontext"] == {"agent_id": "triage", "lauf_id": lauf_id, "container_id": 11, "schritt": 1,
                               "schritt_id": f"{lauf_id}/1"}
    log = step_log(result)
    assert log[0]["observation"].startswith("BROKER RESULT aufruf_id=tb-01HZZ vorlage=cve_offen_auf_assets v3 katalog=k7")
    assert log[1]["observation"].startswith("ERROR: get_container_info reads only the container of this run (11)")
    assert result.data[0]["urteil"]["belege"] == ["tb-01HZZ"]           # aufruf_id accepted as evidence
    audit_step1 = [a["cef"] for a in c.artifacts if a["cef"]["schritt_id"] == f"{lauf_id}/1"]
    assert audit_step1 and audit_step1[0]["aufruf_id"] == "tb-01HZZ"
    assert platform["queries"] == []


def test_ac03_invented_aufruf_id_is_not_accepted(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = [verdict(["tb-forged"]), verdict([])]
    status, result = run(airgap_connector(BrokerStub()))
    assert status is True and result.summary["steps_taken"] == 2
    assert "did not originate in this run" in step_log(result)[0]["observation"]


# --- AC-04 ------------------------------------------------------------------

def test_ac04_unreachable_broker_means_data_gap_and_no_direct_access(platform, llm, no_network):
    define_agent(platform, provider="local", allowed_tools=["splunk_search", "add_note"])
    stub = BrokerStub(raise_exc=ConnectionError("connection refused"))
    llm["responses"] = [
        'ACTION: broker_invoke\nPARAMS: {"vorlage_id": "cve_offen_auf_assets", "parameter": {}}',
        'ACTION: splunk_search\nPARAMS: {"spl_query": "index=main"}',
        verdict([], datenlage="unbekannt"),
    ]
    c = airgap_connector(stub)
    status, result = run(c)
    assert status is True, result.message
    assert "Tool Broker: unavailable (UNREACHABLE)" in llm["system_prompts"][0]
    log = step_log(result)
    assert log[0]["observation"] == "ERROR: Tool 'broker_invoke' is not in the allowed tools list."
    assert log[1]["observation"].startswith("ERROR: Tool 'splunk_search' is disabled in airgap_mode")
    assert platform["queries"] == [] and stub.invokes() == []
    assert result.data[0]["urteil"]["datenlage"] == "unbekannt"


def test_ac04_source_error_from_broker_is_reported_as_gap_without_fallback(platform, llm, no_network):
    define_agent(platform, provider="local", allowed_tools=["splunk_search", "add_note"])
    stub = BrokerStub(invoke={"aufruf_id": "tb-err", "art": "fehler", "code": "SOURCE_ERROR", "detail": "splunk down"})
    llm["responses"] = [
        'ACTION: broker_invoke\nPARAMS: {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-1"}}',
        verdict([], datenlage="blinder_fleck"),
    ]
    c = airgap_connector(stub)
    status, result = run(c)
    assert status is True, result.message
    obs = step_log(result)[0]["observation"]
    assert obs.startswith("BROKER UNAVAILABLE code=SOURCE_ERROR detail=splunk down.") and "no other access path" in obs
    assert platform["queries"] == [] and len(stub.invokes()) == 1
    assert result.data[0]["urteil"]["belege"] == []                     # a failed call is no evidence
    assert result.data[0]["urteil"]["datenlage"] == "blinder_fleck"


# ============================================================================
# Phase 3 (Policy Manager): AC-05, AC-06, AC-07, AC-08
# ============================================================================

RUN_ID = "rt-ac3"
# One scripted call per effecting tool: params the agent sends, and the platform
# change it must (ALLOW) or must not (anything else) produce. A new `wirkt` tool
# fails test_ac05 until it gets a row here.
EFFECT_CALLS = {
    "update_severity": ({"severity": "high"}, lambda platform: platform["severity"], [(11, "high")]),
    # catalogue sequence (4.2): ALLOW records a proposal note for the follow-up playbook (E13),
    # the platform itself is never touched by the run.
    "catalog_invoke": ({"schritte": [{"baustein": "isolate_endpoint", "ziel": ["srv-1"], "parameter": {"dauer_h": 4}}]},
                       lambda platform: [n["title"] for n in platform["notes"] if n["title"].startswith("Freigegeben")],
                       ["Freigegeben (ALLOW): folge: isolate_endpoint -> srv-1"]),
}
EFFECTING_TOOLS = sorted(n for n, d in utils.ToolRegistry.TOOL_DEFINITIONS.items() if d.get("wirkt", True))
EXPECTED_KIND = {"catalog_invoke": "folge"}      # aktion.art on the wire (RT-W09)
CATALOG = [CatalogBlock(id="isolate_endpoint", version=2, purpose="Isolate an endpoint", category="Isolation",
                        target_types=("device",), parameters={"dauer_h": "hours"}, min_gate="APPROVE_2PERSON")]
JUDGEMENT = {"begruendung": "beacon confirmed by step 1", "gegenargumente": "could be a scanner",
             "beleg_ids": [f"{RUN_ID}/1"]}


def effect_script(tool, beleg_ids=None):
    params = EFFECT_CALLS[tool][0] | JUDGEMENT | ({"beleg_ids": beleg_ids} if beleg_ids is not None else {})
    return ["THOUGHT: look at the case\nACTION: get_container_info",
            f"THOUGHT: act\nACTION: {tool}\nPARAMS: {json.dumps(params)}",
            f"FINAL_ANSWER: {VALID_VERDICT}"]


def policy_connector(monkeypatch, policy=None, **config):
    """Airgap connector with a fixed lauf_id and an injected policy gate (None = asset config decides)."""
    monkeypatch.setattr("soar_agent_runtime_connector.new_run_id", lambda: RUN_ID)
    c = make_connector(airgap_mode=True, **config)
    c._policy_client = policy
    return c


def audit_events(c):
    return [a["cef"] for a in c.artifacts if a["label"] == "agent_audit"]


def proposals(c):
    return [a for a in c.artifacts if a["label"] == "agent_action_proposal"]


# --- AC-05 ------------------------------------------------------------------

def test_ac05_registry_has_an_acceptance_case_for_every_effecting_tool():
    assert EFFECTING_TOOLS == sorted(EFFECT_CALLS) and EFFECTING_TOOLS


@pytest.mark.parametrize("tool", EFFECTING_TOOLS)
def test_ac05_effecting_tool_does_nothing_without_an_allow_decision(platform, llm, monkeypatch, tool):
    define_agent(platform, provider="local", allowed_tools=[tool, "get_container_info"])
    llm["responses"] = effect_script(tool)
    stub = StubPolicyClient(PolicyDecision(gate=GATE_DENY, decision_id="pm-deny", code="RISK_TOO_HIGH",
                                           reasons=({"regel": "R7", "grund": "score 90 > 60"},)), blocks=CATALOG)
    c = policy_connector(monkeypatch, stub)
    status, result = run(c)
    assert status is True, result.message
    _, read, _ = EFFECT_CALLS[tool]
    assert read(platform) == [] and proposals(c) == []                # no platform change, no proposal
    assert len(stub.requests) == 1 and stub.requests[0].kind == EXPECTED_KIND.get(tool, tool)  # asked exactly once
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("POLICY DENIED code=RISK_TOO_HIGH (score 90 > 60)") and f"'{tool}' was not executed" in obs


@pytest.mark.parametrize("tool", EFFECTING_TOOLS)
def test_ac05_effecting_tool_runs_exactly_once_on_allow_and_the_audit_names_the_decision(platform, llm, monkeypatch, tool):
    define_agent(platform, provider="local", allowed_tools=[tool, "get_container_info"])
    llm["responses"] = effect_script(tool)
    stub = StubPolicyClient(blocks=CATALOG)                           # ALLOW, entscheidung_id pm-stub-1
    c = policy_connector(monkeypatch, stub)
    status, result = run(c)
    assert status is True, result.message
    _, read, expected = EFFECT_CALLS[tool]
    assert read(platform) == expected
    assert platform["severity"] == ([(11, "high")] if tool == "update_severity" else [])
    assert step_log(result)[1]["observation"].endswith("(entscheidung_id=pm-stub-1)")
    req, = stub.requests
    assert (req.kind, req.run_id, req.step_id, req.evidence_ids) == (EXPECTED_KIND.get(tool, tool), RUN_ID,
                                                                       f"{RUN_ID}/2", (f"{RUN_ID}/1",))
    assert req.judgement["begruendung"] == JUDGEMENT["begruendung"] and "beleg_ids" not in req.params
    done = [e for e in audit_events(c) if e["schritt_id"] == f"{RUN_ID}/2" and e["phase"] == "erledigt"]
    assert done and done[0]["entscheidung_id"] == "pm-stub-1"


# --- AC-06 ------------------------------------------------------------------

def test_ac06_missing_policy_mode_executes_nothing_and_names_the_reason(platform, llm, monkeypatch):
    define_agent(platform, provider="local")
    llm["responses"] = effect_script("update_severity")
    c = policy_connector(monkeypatch)                                 # no policy_mode in the asset
    status, result = run(c)
    assert status is True, result.message
    assert platform["severity"] == []
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("POLICY DENIED code=NO_POLICY_MODE (policy_mode missing or invalid")
    assert "'update_severity' was not executed" in obs
    assert any("policy_mode not set" in m for m in c.progress)


def test_ac06_missing_policy_package_executes_nothing_and_names_the_reason(platform, llm, monkeypatch):
    define_agent(platform, provider="local")
    llm["responses"] = effect_script("update_severity")
    monkeypatch.setitem(sys.modules, "policy_client", None)
    monkeypatch.setitem(sys.modules, "policy_client.embedded", None)
    c = policy_connector(monkeypatch, policy_mode="embedded", policy_path="/etc/pm/policy.json",
                         policy_state_dir="/var/pm")
    status, result = run(c)
    assert status is True, result.message
    assert platform["severity"] == []
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("POLICY DENIED code=UNAVAILABLE (policy_client package not available (ModuleNotFoundError)")
    assert any(m.startswith("[policy] mode=none (not available") for m in c.progress)


def test_ac06_unreachable_policy_manager_executes_nothing_and_names_the_reason(platform, llm, monkeypatch):
    class Unreachable:
        def decide(self, request):
            raise ConnectionError("policy manager refused")

    class Req:
        @staticmethod
        def from_wire(data):
            return data

    define_agent(platform, provider="local")
    llm["responses"] = effect_script("update_severity")
    c = policy_connector(monkeypatch, PolicyClientAdapter(Unreachable(), Req, "embedded"))
    status, result = run(c)
    assert status is True, result.message
    assert platform["severity"] == []
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("POLICY DENIED code=UNAVAILABLE (") and "ConnectionError" in obs
    assert "'update_severity' was not executed" in obs


# --- AC-07 ------------------------------------------------------------------

def test_ac07_approval_gate_records_a_proposal_with_freigabe_id_and_does_not_wait_or_execute(platform, llm, monkeypatch):
    define_agent(platform, provider="local")
    llm["responses"] = effect_script("update_severity")
    stub = StubPolicyClient(PolicyDecision(gate=GATE_APPROVE_2PERSON, decision_id="pm-7", approval_id="fg-7",
                                           approvals_needed=2, expires_at="2026-10-21T08:00:00Z", policy_version="v3"))
    c = policy_connector(monkeypatch, stub)
    status, result = run(c)
    assert status is True, result.message                             # the run continues to its verdict
    assert platform["severity"] == []                                 # nothing executed
    assert len(stub.requests) == 1 and stub.consumed == []            # asked once, never waited or consumed
    assert llm["calls"] == 3 and result.summary["steps_taken"] == 3
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("HELD: 'update_severity' requires APPROVE_2PERSON (2 approval(s)). "
                          "Proposal recorded with freigabe_id=fg-7 entscheidung_id=pm-7; not executed.")
    proposal, = proposals(c)
    assert proposal["container_id"] == 11 and proposal["run_automation"] is False
    assert proposal["source_data_identifier"] == f"fg-7:{RUN_ID}/2"
    cef = proposal["cef"]
    assert cef["freigabe_id"] == "fg-7" and cef["entscheidung_id"] == "pm-7" and cef["gate"] == "APPROVE_2PERSON"
    assert cef["aktion"] == "update_severity" and json.loads(cef["parameter"]) == {"severity": "high"}
    assert cef["lauf_id"] == RUN_ID and cef["schritt_id"] == f"{RUN_ID}/2" and cef["beleg_ids"] == [f"{RUN_ID}/1"]
    assert cef["ablauf"] == "2026-10-21T08:00:00Z" and cef["freigaben_noetig"] == 2 and cef["policy_version"] == "v3"
    assert any(n["title"].startswith("Freigabe ausstehend: update_severity") for n in platform["notes"])


# --- AC-08 ------------------------------------------------------------------

@pytest.mark.parametrize("beleg_ids", [["tb-forged"], [f"{RUN_ID}/2"], [f"{RUN_ID}/1", "made-up"]])
def test_ac08_invented_beleg_id_is_refused_before_the_policy_manager_is_asked(platform, llm, monkeypatch, beleg_ids):
    define_agent(platform, provider="local")
    llm["responses"] = effect_script("update_severity", beleg_ids=beleg_ids)
    stub = StubPolicyClient()                                         # would ALLOW if it were asked
    c = policy_connector(monkeypatch, stub)
    status, result = run(c)
    assert status is True, result.message
    assert stub.requests == []                                        # the Policy Manager was never asked
    assert platform["severity"] == []
    obs = step_log(result)[1]["observation"]
    assert obs.startswith("ERROR: beleg_id ") and "did not originate in this run" in obs
    assert "no policy request was made (RT-W05)" in obs and f"Valid evidence: {RUN_ID}/1." in obs
