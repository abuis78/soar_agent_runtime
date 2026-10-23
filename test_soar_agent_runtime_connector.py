"""Connector tests for RT-A01/RT-A02 (Umsetzungskonzept Runtime, sub-change 1.3).

Covers: Air-Gap profile rejects every cloud provider before any network
call (including the DEV fallback definition), 'local' keeps working, and the
llm_* asset fields reach LLMProvider. Runs off-platform via conftest.py.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

import soar_agent_runtime_connector as connector_module
from soar_agent_runtime_connector import SoarAgentRuntimeConnector
from soar_agent_runtime_policy import NoPolicyClient, StubPolicyClient

CLOUD_PROVIDERS = ["anthropic", "openai", "gemini"]
ALL_PROVIDERS = CLOUD_PROVIDERS + ["local"]


def make_connector(**config) -> SoarAgentRuntimeConnector:
    c = SoarAgentRuntimeConnector()
    c._fake_config = {
        "anthropic_api_key": "k",
        "openai_api_key": "k",
        "gemini_api_key": "k",
        **config,
    }
    assert c.initialize() is True
    return c


# --- normal mode: unchanged behaviour ---------------------------------------

@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_every_provider_builds_without_airgap(provider):
    c = make_connector()
    llm = c._build_llm_provider(provider, "")
    assert llm.provider == provider
    assert "[airgap]" not in " ".join(c.progress)


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_llm_bounds_from_asset_reach_every_provider(provider):
    c = make_connector(llm_timeout_s=30, llm_temperature=0.2, llm_max_tokens=512)
    llm = c._build_llm_provider(provider, "")
    assert (llm.timeout_s, llm.temperature, llm.max_tokens) == (30.0, 0.2, 512)


def test_llm_defaults_match_app_json():
    c = make_connector()
    llm = c._build_llm_provider("local", "")
    assert (llm.timeout_s, llm.temperature, llm.max_tokens, llm.seed) == (120.0, 0.0, 4096, None)


@pytest.mark.parametrize("raw,expected", [("", None), (None, None), ("7", 7), (7, 7)])
def test_llm_seed_empty_is_none_otherwise_int(raw, expected):
    c = make_connector(llm_seed=raw)
    assert c._build_llm_provider("local", "").seed == expected


def test_invalid_numeric_asset_field_falls_back_with_warning():
    c = make_connector(llm_timeout_s="abc")
    assert c._llm_timeout_s == 120.0
    assert any("llm_timeout_s" in m for m in c.progress)


def test_unknown_provider_still_rejected_without_airgap():
    c = make_connector()
    with pytest.raises(ValueError, match="Unknown provider"):
        c._build_llm_provider("foo", "")


# --- airgap mode ------------------------------------------------------------

def test_airgap_is_announced_in_progress():
    c = make_connector(airgap_mode=True)
    assert c._airgap_mode is True
    assert any(m.startswith("[airgap] enabled") for m in c.progress)


@pytest.mark.parametrize("provider", CLOUD_PROVIDERS)
def test_airgap_rejects_cloud_providers(provider):
    c = make_connector(airgap_mode=True)
    with pytest.raises(ValueError) as exc:
        c._build_llm_provider(provider, "")
    msg = str(exc.value)
    assert msg.startswith("airgap_mode:")
    assert f"provider '{provider}' is not allowed, only 'local'" in msg
    assert "default_provider=local" in msg


def test_airgap_allows_local():
    c = make_connector(airgap_mode=True, llm_timeout_s=15)
    llm = c._build_llm_provider("local", "")
    assert llm.provider == "local"
    assert llm.timeout_s == 15.0


def test_airgap_rejects_cloud_default_provider_when_definition_has_none():
    """Asset default 'anthropic' is not silently used in airgap (fail closed)."""
    c = make_connector(airgap_mode=True, default_provider="anthropic")
    with pytest.raises(ValueError, match="airgap_mode"):
        c._build_llm_provider(c._default_provider, "")


def test_run_agent_with_cloud_definition_in_airgap_stops_before_llm(monkeypatch):
    """Off-platform the store returns the DEV fallback (provider 'anthropic'),
    which in airgap must end as APP_ERROR before the ReAct loop starts."""
    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("ReActLoop must not be constructed in airgap with a cloud provider")

    monkeypatch.setattr(connector_module, "ReActLoop", Boom)
    c = make_connector(airgap_mode=True)
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 1})
    assert status is False  # phantom.APP_ERROR
    result = c.action_results[-1]
    assert result.message.startswith("airgap_mode: provider 'anthropic' is not allowed")


def test_run_agent_empty_provider_in_definition_uses_asset_default(monkeypatch):
    seen = {}

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def load(self, agent_id):
            return {"agent_id": agent_id, "provider": "", "allowed_tools": ["add_note"]}

    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("must fail before the loop")

    def fake_build(self, provider, model):
        seen["provider"] = provider
        raise ValueError("stop here")

    monkeypatch.setattr(connector_module, "AgentDefinitionStore", FakeStore)
    monkeypatch.setattr(connector_module, "ReActLoop", Boom)
    monkeypatch.setattr(SoarAgentRuntimeConnector, "_build_llm_provider", fake_build)
    c = make_connector(default_provider="local")
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 1})
    assert seen["provider"] == "local"


# --- test connectivity per mode (RT-A04, sub-change 1.4) ---------------------

def _record_connection_tests(monkeypatch, outcome=(True, "OK")):
    """Replace LLMProvider.test_connection; return the list of providers tested."""
    tested = []

    def fake_test(self):
        tested.append(self.provider)
        return outcome

    monkeypatch.setattr(connector_module.LLMProvider, "test_connection", fake_test)
    return tested


def test_connectivity_without_airgap_tests_every_configured_provider(monkeypatch):
    tested = _record_connection_tests(monkeypatch)
    c = make_connector()
    status = c._handle_test_connectivity({})
    assert status is True
    assert sorted(tested) == sorted(ALL_PROVIDERS)
    assert c.action_results[-1].message.startswith("Connectivity OK (normal).")


def test_connectivity_without_airgap_skips_unconfigured_cloud_providers(monkeypatch):
    tested = _record_connection_tests(monkeypatch)
    c = SoarAgentRuntimeConnector()
    c._fake_config = {}
    c.initialize()
    c._handle_test_connectivity({})
    assert tested == ["local"]
    assert "SKIPPED (no credentials configured)" in c.action_results[-1].message


def test_connectivity_in_airgap_only_contacts_local(monkeypatch):
    """Since 2.6/3.7 the airgap check also needs a healthy broker and policy gate; cloud stays untouched."""
    tested = _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True, broker_url="https://broker.internal:8443",
                       broker_client_cert="/etc/rt.crt", broker_client_key="/etc/rt.key", policy_mode="embedded")
    c._broker_transport = lambda method, url, params, body, timeout_s: (200, {"status": "ok"})
    c._policy_client = StubPolicyClient()
    status = c._handle_test_connectivity({})
    assert tested == ["local"]
    assert status is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (airgap).")
    for provider in CLOUD_PROVIDERS:
        assert f'"{provider}": "SKIPPED (airgap_mode)"' in msg
    assert "ERROR" not in msg


def test_connectivity_in_airgap_fails_when_local_is_down(monkeypatch):
    tested = _record_connection_tests(monkeypatch, outcome=(False, "connection refused"))
    c = make_connector(airgap_mode=True)
    status = c._handle_test_connectivity({})
    assert tested == ["local"]
    assert status is False
    assert c.action_results[-1].message.startswith("All providers failed (airgap).")


def test_connectivity_in_airgap_never_builds_a_cloud_client(monkeypatch):
    built = []
    original = SoarAgentRuntimeConnector._build_llm_provider

    def spy(self, provider, model):
        built.append(provider)
        return original(self, provider, model)

    monkeypatch.setattr(SoarAgentRuntimeConnector, "_build_llm_provider", spy)
    _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True)
    c._handle_test_connectivity({})
    assert built == ["local"]


# --- lauf_id in run agent result (RT-P01, sub-change 1.6) --------------------

def test_run_agent_result_and_summary_carry_lauf_id(monkeypatch):
    import re

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def load(self, agent_id):
            return {"agent_id": agent_id, "provider": "local", "allowed_tools": ["add_note"]}

    monkeypatch.setattr(connector_module, "AgentDefinitionStore", FakeStore)
    monkeypatch.setattr(connector_module.LLMProvider, "chat",
                        lambda self, system_prompt, messages: "FINAL_ANSWER: ok")
    c = make_connector()
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 1})
    assert status is True
    result = c.action_results[-1]
    lauf_id = result.summary["lauf_id"]
    assert re.match(r"^rt-\d{8}T\d{6}Z-[0-9a-f]{8}$", lauf_id)
    assert result.data[0]["lauf_id"] == lauf_id
    assert any(f"[lauf_id {lauf_id}]" in m for m in c.progress)


# --- audit sink per mode (RT-P02/P03, sub-change 1.8) -----------------------

from soar_agent_runtime_audit import ArtifactAuditSink, NullAuditSink  # noqa: E402


class _LocalStore:
    def __init__(self, *a, **kw):
        pass

    def load(self, agent_id):
        return {"agent_id": agent_id, "provider": "local", "allowed_tools": ["add_note"]}


def _capture_loop(monkeypatch):
    seen = {}

    class CapturingLoop:
        def __init__(self, **kw):
            seen.update(kw)
            self.run_id = "rt-x"

        def run(self, task):
            return {"agent_id": "triage", "lauf_id": "rt-x", "final_answer": "ok", "steps_taken": 1,
                    "provider": "local", "model": "m", "audit_ereignisse": 2, "audit_fehler": 0,
                    "step_log": "[]"}

    monkeypatch.setattr(connector_module, "ReActLoop", CapturingLoop)
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _LocalStore)
    return seen


@pytest.mark.parametrize("config,sink_type,fail_closed", [
    ({"airgap_mode": True}, ArtifactAuditSink, True),
    ({"airgap_mode": True, "audit_enabled": False}, ArtifactAuditSink, True),
    ({"audit_enabled": True}, ArtifactAuditSink, False),
    ({}, NullAuditSink, False),
])
def test_run_agent_picks_audit_sink_per_mode(monkeypatch, config, sink_type, fail_closed):
    seen = _capture_loop(monkeypatch)
    c = make_connector(**config)
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    sink = seen["audit"]
    assert isinstance(sink, sink_type)
    assert sink.fail_closed is fail_closed
    if sink_type is ArtifactAuditSink:
        assert sink.container_id == 7
    assert len(seen["def_version"]) == 64
    summary = c.action_results[-1].summary
    assert (summary["audit_ereignisse"], summary["audit_fehler"]) == (2, 0)


_VALID_VERDICT = "FINAL_ANSWER: " + json.dumps({
    "belege": [], "begruendung": "nothing suspicious", "gegenargumente": [],
    "unsicherheiten": ["no telemetry"], "konfidenz": 0.6, "datenlage": "teilweise",
})


def test_run_agent_in_airgap_writes_audit_artifacts_end_to_end(monkeypatch):
    """Airgap: strict FINAL_ANSWER contract (1.9) plus audit artifact per step (1.8)."""
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _LocalStore)
    monkeypatch.setattr(connector_module.LLMProvider, "chat",
                        lambda self, system_prompt, messages: _VALID_VERDICT)
    c = make_connector(airgap_mode=True)
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 9})
    assert status is True
    assert len(c.artifacts) == 1
    art = c.artifacts[0]
    lauf_id = c.action_results[-1].summary["lauf_id"]
    assert art["label"] == "agent_audit" and art["run_automation"] is False and art["container_id"] == 9
    assert art["cef"]["lauf_id"] == lauf_id and art["cef"]["schritt_id"] == f"{lauf_id}/1"
    assert c.action_results[-1].summary["audit_ereignisse"] == 1


def test_run_agent_in_airgap_fails_closed_when_artifacts_cannot_be_written(monkeypatch):
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _LocalStore)
    monkeypatch.setattr(connector_module.LLMProvider, "chat",
                        lambda self, system_prompt, messages: "FINAL_ANSWER: ok")
    c = make_connector(airgap_mode=True)
    c.save_artifact = lambda artifact: (False, "disk full", None)
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 9})
    assert status is False
    assert c.action_results[-1].message.startswith("ERROR: audit not writable")


# --- FINAL_ANSWER contract flags (RT-P04, sub-change 1.9) --------------------

class _SkillStoreWithSchemas:
    build_injection = staticmethod(lambda skills: "## skills")

    def __init__(self, *a, **kw):
        pass

    def load_many(self, ids):
        return [
            {"skill_id": "triage", "inject": "be brief",
             "output_schema": {"verdict": "TP | FP", "reason": "why"}},
            {"skill_id": "ioc", "output_schema": {"reason": "dup", "iocs": []}},
            {"skill_id": "plain", "inject": "no schema"},
        ]


class _SkilledStore(_LocalStore):
    def load(self, agent_id):
        return super().load(agent_id) | {"skills": "triage, ioc, plain"}


@pytest.mark.parametrize("config,strict", [({}, False), ({"airgap_mode": True}, True)])
def test_run_agent_passes_schema_flags_and_skill_keys(monkeypatch, config, strict):
    seen = _capture_loop(monkeypatch)
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _SkilledStore)
    monkeypatch.setattr(connector_module, "SkillStore", _SkillStoreWithSchemas)
    c = make_connector(**config)
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert seen["strict_schema"] is strict
    assert seen["required_keys"] == ["verdict", "reason", "iocs"]


def test_run_agent_without_skills_passes_no_required_keys(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector()
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert seen["required_keys"] == [] and seen["strict_schema"] is False


# --- max_observation_chars pass-through (RT-B04, sub-change 1.10) ------------

@pytest.mark.parametrize("config,expected", [({}, 8000), ({"max_observation_chars": 1234}, 1234),
                                              ({"max_observation_chars": "500"}, 500)])
def test_run_agent_passes_max_observation_chars(monkeypatch, config, expected):
    seen = _capture_loop(monkeypatch)
    c = make_connector(**config)
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert seen["max_observation_chars"] == expected


# --- Tool Broker asset fields (RT-L07, sub-change 2.1) -----------------------

def test_broker_fields_default_to_empty_and_30s():
    c = make_connector()
    assert (c._broker_url, c._broker_timeout_s, c._broker_client_cert, c._broker_client_key, c._broker_ca) == \
        ("", 30.0, "", "", "")
    assert not any("broker_url" in m for m in c.progress)


def test_broker_fields_are_read_from_the_asset():
    c = make_connector(broker_url=" https://broker.internal:8443 ", broker_timeout_s="12",
                       broker_client_cert="/opt/phantom/etc/rt.crt", broker_client_key="/opt/phantom/etc/rt.key",
                       broker_ca="/opt/phantom/etc/ca.pem")
    assert c._broker_url == "https://broker.internal:8443"
    assert c._broker_timeout_s == 12.0
    assert (c._broker_client_cert, c._broker_client_key, c._broker_ca) == \
        ("/opt/phantom/etc/rt.crt", "/opt/phantom/etc/rt.key", "/opt/phantom/etc/ca.pem")


def test_airgap_without_broker_url_warns_but_still_initializes():
    c = make_connector(airgap_mode=True)
    assert any(m == "[airgap] broker_url not set - read tools unavailable." for m in c.progress)


def test_airgap_with_broker_url_does_not_warn():
    c = make_connector(airgap_mode=True, broker_url="https://broker.internal:8443")
    assert not any("broker_url not set" in m for m in c.progress)


def test_app_json_declares_broker_fields_in_order():
    with open("soar_agent_runtime.json") as f:
        app = json.load(f)
    cfg = app["configuration"]
    assert app["app_version"] == "1.8.1"
    assert [cfg[k]["order"] for k in ("broker_url", "broker_timeout_s", "broker_client_cert", "broker_client_key", "broker_ca")] == [23, 24, 25, 26, 27]
    assert cfg["broker_client_key"]["data_type"] == "string"  # a path, never the key material
    assert len({f["order"] for f in cfg.values()}) == len(cfg)  # orders unique


# --- Tool Broker wiring (RT-L03, L06, L07, B01; sub-change 2.3b) --------------

BROKER_ASSET = {"broker_url": "https://broker.internal:8443", "broker_client_cert": "/etc/rt.crt",
                "broker_client_key": "/etc/rt.key", "broker_ca": "/etc/ca.pem"}
POLICY_ASSET = {"policy_mode": "embedded", "policy_path": "/opt/phantom/etc/policy.json",
                "policy_state_dir": "/opt/phantom/local/data/policy/"}
TOOLS_BODY = {"katalog_version": "k1", "angehalten": False,
              "vorlagen": [{"vorlage_id": "asset_kontext", "version": 1, "beschreibung": "Asset-Stammdaten"}]}


def _transport(status=200, body=None, exc=None):
    calls = []

    def send(method, url, params, json_body, timeout_s):
        calls.append((method, url, params, json_body, timeout_s))
        if exc:
            raise exc
        return status, body

    send.calls = calls
    return send


def test_airgap_with_broker_loads_templates_and_shares_run_id(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    c._broker_transport = _transport(body=TOOLS_BODY)
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    tools = seen["tools"]
    assert list(tools.broker_tools) == ["asset_kontext"] and tools.broker_status == "ok"
    assert tools.run_id == seen["run_id"] and tools.run_id.startswith("rt-")
    assert c._broker_transport.calls[0][:3] == ("GET", "https://broker.internal:8443/v1/tools", {"agent_id": "triage"})
    assert "asset_kontext v1" in tools.get_tool_descriptions()


def test_airgap_with_misconfigured_broker_fails_before_the_run(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True, **(BROKER_ASSET | {"broker_url": "http://broker.internal:8443"}))
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert status is False and seen == {}
    assert c.action_results[-1].message.startswith("broker misconfigured: broker_url must be an https:// URL")


def test_outside_airgap_misconfigured_broker_only_warns(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(**(BROKER_ASSET | {"broker_client_key": ""}))
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert status is True and seen["tools"].broker is None
    assert any(m.startswith("[broker] misconfigured, running without broker:") for m in c.progress)


def test_halted_broker_stops_before_the_first_model_call(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    c._broker_transport = _transport(body=TOOLS_BODY | {"angehalten": True})
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert status is False and seen == {}
    assert c.action_results[-1].message == "Tool Broker is HALTED - run not started (RT-B01)"
    assert c.artifacts == []


def test_unreachable_broker_starts_the_run_with_a_data_gap_note(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    c._broker_transport = _transport(exc=ConnectionError("refused"))
    status = c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert status is True
    tools = seen["tools"]
    assert tools.broker_status == "UNREACHABLE" and tools.broker_tools == {}
    assert "Tool Broker: unavailable (UNREACHABLE)" in tools.get_tool_descriptions()
    assert any("[broker] tools unavailable (UNREACHABLE)" in m for m in c.progress)


def test_without_broker_url_registry_has_no_broker(monkeypatch):
    seen = _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True)
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert seen["tools"].broker is None and seen["tools"].broker_status == ""


# --- create agent default tools (RT-L02, sub-change 2.4) ---------------------

def test_create_agent_defaults_to_add_note_only(monkeypatch):
    saved = {}

    class Store:
        def __init__(self, *a, **kw):
            pass

        def save(self, agent_id, config):
            saved[agent_id] = config
            return True

    monkeypatch.setattr(connector_module, "AgentDefinitionStore", Store)
    c = make_connector()
    assert c._handle_create_agent({"agent_id": "a1"}) is True
    assert saved["a1"]["allowed_tools"] == "add_note"
    c._handle_create_agent({"agent_id": "a2", "allowed_tools": ""})
    assert saved["a2"]["allowed_tools"] == "add_note"
    c._handle_create_agent({"agent_id": "a3", "allowed_tools": "splunk_search,add_note"})
    assert saved["a3"]["allowed_tools"] == "splunk_search,add_note"  # explicit choice is kept


def test_app_json_create_agent_default_has_no_spl():
    with open("soar_agent_runtime.json") as f:
        app = json.load(f)
    par = next(a for a in app["actions"] if a["action"] == "create agent")["parameters"]["allowed_tools"]
    assert par["default"] == "add_note"
    assert "deprecated" in par["description"] and "airgap_mode" in par["description"]


def test_run_agent_passes_airgap_flag_to_registry(monkeypatch):
    seen = _capture_loop(monkeypatch)
    make_connector(airgap_mode=True)._handle_run_agent({"agent_id": "t", "task": "x", "container_id": 1})
    assert seen["tools"].airgap is True
    seen.clear()
    make_connector()._handle_run_agent({"agent_id": "t", "task": "x", "container_id": 1})
    assert seen["tools"].airgap is False


# --- test connectivity: broker health (RT-A04 part 2, sub-change 2.6) --------

def _airgap_connector(policy=None, **overrides) -> SoarAgentRuntimeConnector:
    """Airgap asset with broker and policy configured; the policy gate is a stub (3.7)."""
    c = make_connector(**({"airgap_mode": True} | BROKER_ASSET | POLICY_ASSET | overrides))
    c._policy_client = policy or StubPolicyClient()
    return c


def _health_transport(status=200, exc=None):
    calls = []

    def send(method, url, params, body, timeout_s):
        calls.append((method, url))
        if exc:
            raise exc
        return status, {"status": "ok"} if status == 200 else "down"

    send.calls = calls
    return send


def test_connectivity_airgap_with_healthy_broker_succeeds(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector()
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (airgap).")
    assert '"broker": "OK: broker healthy (HTTP 200)"' in msg and '"anthropic": "SKIPPED (airgap_mode)"' in msg
    assert c._broker_transport.calls == [("GET", "https://broker.internal:8443/health")]
    assert "Testing broker..." in c.progress


def test_connectivity_airgap_with_unreachable_broker_fails_despite_healthy_local(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector()
    c._broker_transport = _health_transport(exc=ConnectionError("refused"))
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Broker check failed (airgap).") and '"broker": "ERROR: broker unreachable' in msg
    assert '"local": "OK"' in msg


def test_connectivity_airgap_without_broker_url_fails_as_not_configured(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True)
    assert c._handle_test_connectivity({}) is False
    assert '"broker": "NOT CONFIGURED (required in airgap_mode)"' in c.action_results[-1].message


def test_connectivity_airgap_with_misconfigured_broker_reports_it(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector(broker_url="http://broker.internal")
    assert c._handle_test_connectivity({}) is False
    assert '"broker": "ERROR: misconfigured: broker_url must be an https:// URL' in c.action_results[-1].message


def test_connectivity_airgap_local_failure_still_wins_over_broker(monkeypatch):
    _record_connection_tests(monkeypatch, outcome=(False, "connection refused"))
    c = _airgap_connector()
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    assert c.action_results[-1].message.startswith("All providers failed (airgap).")


def test_connectivity_normal_mode_without_broker_is_unchanged(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = make_connector()
    assert c._handle_test_connectivity({}) is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (normal).") and '"broker": "SKIPPED (no broker_url)"' in msg
    assert "Testing broker..." not in c.progress


def test_connectivity_normal_mode_unhealthy_broker_is_reported_but_not_fatal(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = make_connector(**BROKER_ASSET)
    c._broker_transport = _health_transport(status=503)
    assert c._handle_test_connectivity({}) is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (normal).") and '"broker": "ERROR: broker unhealthy: HTTP 503' in msg


# --- Policy Manager wiring (RT-W03, W08, E11; sub-change 3.1a) ----------------


def test_app_json_declares_policy_fields_and_run_agent_params():
    with open("soar_agent_runtime.json") as f:
        app = json.load(f)
    cfg = app["configuration"]
    assert [cfg[k]["order"] for k in ("policy_mode", "policy_path", "policy_state_dir")] == [28, 29, 30]
    assert cfg["policy_mode"]["value_list"] == ["", "embedded", "service"] and cfg["policy_mode"]["default"] == ""
    assert all(cfg[k]["required"] is False for k in ("policy_mode", "policy_path", "policy_state_dir"))
    assert len({f["order"] for f in cfg.values()}) == len(cfg)
    run = next(a for a in app["actions"] if a["action"] == "run agent")["parameters"]
    assert run["stufe"]["data_type"] == "numeric" and run["stufe"]["required"] is False
    assert run["domaene"]["data_type"] == "string" and run["domaene"]["required"] is False
    assert len({p["order"] for p in run.values()}) == len(run)


def test_policy_fields_are_read_and_normalised():
    c = make_connector(policy_mode=" Embedded ", policy_path=" /p.json ", policy_state_dir="/state/")
    assert (c._policy_mode, c._policy_path, c._policy_state_dir) == ("embedded", "/p.json", "/state")
    assert make_connector()._policy_mode == ""


# --- Policy Manager service fields (RT-W01, RT-W04; sub-change 4.1-2) ----------

SERVICE_ASSET = {"policy_mode": "service", "policy_service_url": "https://pm.internal:8443",
                 "policy_client_cert": "/etc/pm/rt.crt", "policy_client_key": "/etc/pm/rt.key",
                 "policy_ca": "/etc/pm/ca.pem"}
SERVICE_FIELDS = ("policy_service_url", "policy_timeout_s", "policy_client_cert", "policy_client_key", "policy_ca")


def test_app_json_declares_service_fields():
    with open("soar_agent_runtime.json") as f:
        cfg = json.load(f)["configuration"]
    assert [cfg[k]["order"] for k in SERVICE_FIELDS] == [31, 32, 33, 34, 35]
    assert all(cfg[k]["required"] is False and "service" in cfg[k]["description"] for k in SERVICE_FIELDS)
    assert cfg["policy_timeout_s"]["data_type"] == "numeric" and cfg["policy_timeout_s"]["default"] == 5
    assert len({f["order"] for f in cfg.values()}) == len(cfg)


def test_service_fields_are_read_and_normalised():
    c = make_connector(**{**SERVICE_ASSET, "policy_service_url": " https://pm.internal:8443 ",
                          "policy_timeout_s": "2.5", "policy_ca": " /etc/pm/ca.pem "})
    assert (c._policy_service_url, c._policy_timeout_s, c._policy_ca) == ("https://pm.internal:8443", 2.5, "/etc/pm/ca.pem")
    d = make_connector()
    assert (d._policy_service_url, d._policy_timeout_s, d._policy_client_cert, d._policy_client_key, d._policy_ca) == \
        ("", 5.0, "", "", "")
    assert make_connector(policy_timeout_s="abc")._policy_timeout_s == 5.0   # invalid = default + warning


def test_airgap_service_passes_the_fields_to_the_factory(monkeypatch):
    seen = {}
    monkeypatch.setattr(connector_module, "build_policy_client",
                        lambda mode, **kw: seen.update(mode=mode, **kw) or StubPolicyClient())
    make_connector(airgap_mode=True, policy_timeout_s=3, **SERVICE_ASSET)._build_policy_client()
    assert seen["mode"] == "service"
    assert {k: seen[k] for k in ("service_url", "service_timeout_s", "client_cert", "client_key", "ca")} == {
        "service_url": "https://pm.internal:8443", "service_timeout_s": 3.0, "client_cert": "/etc/pm/rt.crt",
        "client_key": "/etc/pm/rt.key", "ca": "/etc/pm/ca.pem"}
    assert seen["policy_path"] == "" and seen["approvals_path"] == ""   # embedded paths stay empty


def _install_fake_service_package(monkeypatch, halted=False, aktionen=()):
    """policy_client with a service module whose client answers from memory (P3/P4-style wheel)."""
    class Wire:
        def __init__(self, data):
            self.data = data

        def to_wire(self):
            return self.data

    class FakeServiceClient:
        def __init__(self, url, cert, key, ca, timeout_s=5.0):
            self.args = (url, cert, key, ca, timeout_s)

        def halt_status(self):
            return Wire({"angehalten": halted, "grund": "drill" if halted else ""})

        def decide(self, request):
            return Wire({"gate": "ALLOW", "entscheidung_id": "pm-svc"})

        def actions(self, agent_id):
            return list(aktionen)

    base = types.ModuleType("policy_client.base")
    base.DecisionRequest = types.SimpleNamespace(from_wire=lambda data: data)
    svc = types.ModuleType("policy_client.service")
    svc.ServicePolicyClient = FakeServiceClient
    for name, mod in (("policy_client", types.ModuleType("policy_client")), ("policy_client.base", base),
                      ("policy_client.service", svc)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_connectivity_airgap_service_reports_ok_through_the_package_client(monkeypatch):
    _record_connection_tests(monkeypatch)
    _install_fake_service_package(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET, **SERVICE_ASSET)   # no injected stub
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy service healthy, not halted"' in c.action_results[-1].message


def test_connectivity_airgap_service_halted_is_healthy_but_flagged(monkeypatch):
    _record_connection_tests(monkeypatch)
    _install_fake_service_package(monkeypatch, halted=True)
    c = make_connector(airgap_mode=True, **BROKER_ASSET, **SERVICE_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy service healthy, HALTED (drill)"' in c.action_results[-1].message


def test_connectivity_airgap_service_without_certificate_fields_fails_and_names_them(monkeypatch):
    _record_connection_tests(monkeypatch)
    _install_fake_service_package(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET, policy_mode="service",
                       policy_service_url="https://pm.internal:8443")
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).")
    assert '"policy": "ERROR: policy_mode \'service\' needs policy_client_cert, policy_client_key, policy_ca' in msg


def test_connectivity_airgap_service_with_p1_wheel_reports_unavailable(monkeypatch):
    _record_connection_tests(monkeypatch)
    monkeypatch.setitem(sys.modules, "policy_client.service", None)   # bundled P1 wheel: no service module
    c = make_connector(airgap_mode=True, **BROKER_ASSET, **SERVICE_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    assert '"policy": "ERROR: policy_client.service not in the bundled policy_manager wheel' in c.action_results[-1].message


AKTIONEN = [{"id": "disable_account", "version": 1, "zweck": "Disable", "kategorie": "Identity", "mindeststufe": 0},
            {"id": "wipe_host", "version": 1, "zweck": "Wipe", "kategorie": "Destruction", "mindeststufe": 4}]


@pytest.mark.parametrize("param, expected", [
    ({"stufe": 1}, "[policy] catalogue: 1 block(s) available at stufe 1, 1 hidden (higher stufe)."),
    ({}, "[policy] catalogue: 1 block(s) available at stufe 0, 1 hidden (higher stufe)."),
    ({"stufe": 4}, "[policy] catalogue: 2 block(s) available at stufe 4, 0 hidden (higher stufe)."),
])
def test_run_agent_service_mode_loads_the_catalogue_through_the_package_client(monkeypatch, param, expected):
    """RT-W09 / E12 over run_agent: GET /v1/aktionen via the package client, filtered by the run's stufe."""
    _install_fake_service_package(monkeypatch, aktionen=AKTIONEN)
    _capture_loop(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET, **SERVICE_ASSET)   # no injected stub
    c._broker_transport = _transport(body=TOOLS_BODY)
    assert c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7, **param}) is True
    assert any(m.startswith("[policy] mode=service (ready)") for m in c.progress)
    assert expected in c.progress


def test_airgap_without_policy_mode_warns():
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    assert any("policy_mode not set" in m and "denied" in m for m in c.progress)
    assert not any("policy_mode not set" in m for m in make_connector(airgap_mode=True, **BROKER_ASSET, **POLICY_ASSET).progress)


def test_outside_airgap_no_policy_client_is_built():
    assert make_connector(**POLICY_ASSET)._build_policy_client() is None


def test_airgap_without_policy_mode_builds_a_denying_client():
    gate = make_connector(airgap_mode=True)._build_policy_client()
    assert isinstance(gate, NoPolicyClient) and "policy_mode missing" in gate.reason


def test_airgap_embedded_without_package_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "policy_client", None)
    monkeypatch.setitem(sys.modules, "policy_client.embedded", None)
    gate = make_connector(airgap_mode=True, **POLICY_ASSET)._build_policy_client()
    assert isinstance(gate, NoPolicyClient) and gate.code == "UNAVAILABLE"


def test_airgap_embedded_derives_state_files_from_state_dir(monkeypatch):
    seen = {}
    monkeypatch.setattr(connector_module, "build_policy_client",
                        lambda mode, **kw: seen.update(mode=mode, **kw) or StubPolicyClient())
    make_connector(airgap_mode=True, **POLICY_ASSET)._build_policy_client()
    assert seen == {"mode": "embedded", "policy_path": "/opt/phantom/etc/policy.json",
                    "approvals_path": "/opt/phantom/local/data/policy/approvals.json",
                    "halt_path": "/opt/phantom/local/data/policy/halt.json",
                    "audit_path": "/opt/phantom/local/data/policy/audit.jsonl",
                    "sequences_path": "/opt/phantom/local/data/policy/sequences.json",
                    "service_url": "", "service_timeout_s": 5.0, "client_cert": "", "client_key": "", "ca": ""}


def _capture_registry(monkeypatch):
    seen = {}
    real = connector_module.ToolRegistry

    def factory(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(connector_module, "ToolRegistry", factory)
    return seen


@pytest.mark.parametrize("param,stage,domain", [
    ({}, None, ""), ({"stufe": 2, "domaene": " soc "}, 2, "soc"), ({"stufe": "3"}, 3, ""),
    ({"stufe": 7}, None, ""), ({"stufe": "x"}, None, ""), ({"stufe": ""}, None, ""),
])
def test_run_agent_passes_stage_domain_and_policy_to_the_registry(monkeypatch, param, stage, domain):
    _capture_loop(monkeypatch)
    seen = _capture_registry(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    c._broker_transport = _transport(body=TOOLS_BODY)
    stub = StubPolicyClient()
    c._policy_client = stub
    c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7, **param})
    assert seen["policy"] is stub and seen["stage"] == stage and seen["domain"] == domain
    assert any(m.startswith("[policy] mode=embedded (ready)") for m in c.progress)


def test_run_agent_outside_airgap_passes_no_policy(monkeypatch):
    _capture_loop(monkeypatch)
    seen = _capture_registry(monkeypatch)
    make_connector()._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7})
    assert seen["policy"] is None and seen["stage"] is None and seen["domain"] == ""


# --- test connectivity: Policy Manager health (RT-A04 part 2, sub-change 3.7) --

class _HealthGate(NoPolicyClient):
    """Gate whose health() is scripted: a (ok, message) tuple or an exception."""

    def __init__(self, outcome):
        super().__init__("scripted")
        self.outcome = outcome

    def health(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_connectivity_airgap_with_healthy_policy_reports_ok(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector()
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    assert '"policy": "OK: policy stub healthy"' in c.action_results[-1].message
    assert "Testing policy manager..." in c.progress


def test_connectivity_airgap_with_unhealthy_policy_fails_despite_healthy_local_and_broker(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector(policy=StubPolicyClient(healthy=False))
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).")
    assert '"policy": "ERROR: policy stub unhealthy"' in msg
    assert '"broker": "OK: broker healthy (HTTP 200)"' in msg and '"local": "OK"' in msg


def test_connectivity_airgap_without_policy_mode_fails_as_not_configured(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True, **BROKER_ASSET)
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).")
    assert '"policy": "NOT CONFIGURED (required in airgap_mode)"' in msg
    assert "Testing policy manager..." not in c.progress


def test_connectivity_airgap_without_policy_package_reports_unavailable(monkeypatch):
    _record_connection_tests(monkeypatch)
    monkeypatch.setitem(sys.modules, "policy_client", None)
    monkeypatch.setitem(sys.modules, "policy_client.embedded", None)
    c = make_connector(airgap_mode=True, **BROKER_ASSET, **POLICY_ASSET)   # no injected stub
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).")
    assert '"policy": "ERROR: policy_client package not available' in msg


def test_connectivity_airgap_halted_policy_manager_is_healthy_but_flagged(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector(policy=_HealthGate((True, "policy embedded healthy, HALTED (drill)")))
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (airgap).") and '"policy": "OK: policy embedded healthy, HALTED (drill)"' in msg


def test_connectivity_airgap_policy_health_exception_fails_closed(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector(policy=_HealthGate(OSError("policy.json unreadable")))
    c._broker_transport = _health_transport()
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Policy check failed (airgap).")
    assert '"policy": "ERROR: policy health check raised OSError: policy.json unreadable"' in msg


def test_connectivity_airgap_broker_failure_is_reported_before_policy(monkeypatch):
    _record_connection_tests(monkeypatch)
    c = _airgap_connector(policy=StubPolicyClient(healthy=False))
    c._broker_transport = _health_transport(status=503)
    assert c._handle_test_connectivity({}) is False
    msg = c.action_results[-1].message
    assert msg.startswith("Broker check failed (airgap).") and '"policy": "ERROR: policy stub unhealthy"' in msg


def test_connectivity_normal_mode_skips_policy_and_never_builds_a_gate(monkeypatch):
    _record_connection_tests(monkeypatch)
    monkeypatch.setattr(connector_module, "build_policy_client",
                        lambda *a, **k: pytest.fail("policy gate built outside airgap_mode"))
    c = make_connector(**POLICY_ASSET)
    assert c._handle_test_connectivity({}) is True
    msg = c.action_results[-1].message
    assert msg.startswith("Connectivity OK (normal).")
    assert '"policy": "SKIPPED (policy_mode \'embedded\' is only used in airgap_mode)"' in msg
    assert "Testing policy manager..." not in c.progress
    plain = make_connector()
    assert plain._handle_test_connectivity({}) is True
    assert '"policy": "SKIPPED (no policy_mode)"' in plain.action_results[-1].message
