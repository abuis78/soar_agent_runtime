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
