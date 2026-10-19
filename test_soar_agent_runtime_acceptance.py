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
    state = {"agents": {}, "notes": [], "severity": []}
    rules = types.ModuleType("phantom.rules")
    rules.get_list = lambda list_name: [[aid, json.dumps(cfg)] for aid, cfg in state["agents"].items()]
    rules.add_note = lambda **kw: state["notes"].append(kw)
    rules.update_container = lambda container=None, severity=None: state["severity"].append((container, severity))
    rules.get_container = lambda cid: {"id": cid, "name": "case"}
    rules.get_artifacts = lambda **kw: []
    rules.run_query = lambda **kw: []
    monkeypatch.setitem(sys.modules, "phantom.rules", rules)
    monkeypatch.setattr(sys.modules["phantom"], "rules", rules, raising=False)
    return state


@pytest.fixture
def llm(monkeypatch):
    """Fake OpenAI-compatible and Anthropic clients fed from a response script."""
    rec = {"responses": [], "openai_base_urls": [], "anthropic_clients": 0, "calls": 0}

    def _next():
        rec["calls"] += 1
        return rec["responses"].pop(0)

    class _Completions:
        def create(self, **kw):
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
    """Airgap test connectivity: local model + Tool Broker health (2.6), nothing else."""
    llm["responses"] = ["OK"]
    broker_calls = []

    def broker_transport(method, url, params, body, timeout_s):
        broker_calls.append((method, url))
        return 200, {"status": "ok"}

    c = make_connector(airgap_mode=True, openai_api_key="k", gemini_api_key="k",
                       broker_url="https://broker.internal:8443", broker_client_cert="/etc/rt.crt",
                       broker_client_key="/etc/rt.key", broker_ca="/etc/ca.pem")
    c._broker_transport = broker_transport
    status = c._handle_test_connectivity({})
    assert status is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (airgap).")
    assert set(llm["openai_base_urls"]) == {LOCAL_URL} and llm["anthropic_clients"] == 0
    assert broker_calls == [("GET", "https://broker.internal:8443/health")]
    assert '"broker": "OK: broker healthy (HTTP 200)"' in msg
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

def test_ac12_every_step_is_audited_with_lauf_id_and_schritt_id(platform, llm):
    define_agent(platform, provider="local")
    llm["responses"] = [
        "THOUGHT: a\nACTION: get_container_info",
        'THOUGHT: b\nACTION: update_severity\nPARAMS: {"severity": "high"}',
        f"FINAL_ANSWER: {VALID_VERDICT}",
    ]
    c = make_connector(airgap_mode=True)
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
