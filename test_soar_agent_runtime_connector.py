"""Connector tests for RT-A01/RT-A02 (Umsetzungskonzept Runtime, sub-change 1.3).

Covers: Air-Gap profile rejects every cloud provider before any network
call (including the DEV fallback definition), 'local' keeps working, and the
llm_* asset fields reach LLMProvider. Runs off-platform via conftest.py.
"""

from __future__ import annotations

import pytest

import soar_agent_runtime_connector as connector_module
from soar_agent_runtime_connector import SoarAgentRuntimeConnector

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
    tested = _record_connection_tests(monkeypatch)
    c = make_connector(airgap_mode=True)
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


def test_run_agent_in_airgap_writes_audit_artifacts_end_to_end(monkeypatch):
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _LocalStore)
    monkeypatch.setattr(connector_module.LLMProvider, "chat",
                        lambda self, system_prompt, messages: "FINAL_ANSWER: ok")
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
