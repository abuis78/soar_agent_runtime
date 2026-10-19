"""Telemetry tests for the SOAR Agent Runtime (AI-SOC Telemetry, handoff 3/4).

Covers the two guarantees from the handoff (§8):
  1. a decision-type agent action emits exactly one agent_action event
  2. a telemetry failure never blocks the agent action

plus the decisions this integration rests on: only decisions are counted, and
agent_role is a fixed enum.
"""

from __future__ import annotations

import sys
import types

import pytest

from soar_agent_runtime_utils import (
    AGENT_ROLES,
    DEFAULT_AGENT_ROLE,
    ToolRegistry,
    _outcome_from_result,
)

ALL_TOOLS = ["splunk_search", "add_note", "update_severity", "get_container_info"]


class FakeConnector:
    def __init__(self):
        self.progress = []

    def save_progress(self, msg):
        self.progress.append(msg)


class FakeTelemetry:
    """Stands in for TelemetryClient; records what would be shipped."""

    def __init__(self):
        self.sent = []

    def emit(self, event):
        self.sent.append(event)


class ExplodingTelemetry:
    def emit(self, event):
        raise RuntimeError("boom")


@pytest.fixture
def fake_phantom(monkeypatch):
    """Stub phantom.rules so tool handlers succeed off-platform."""
    calls = {"severity": [], "notes": []}

    rules = types.ModuleType("phantom.rules")
    rules.update_container = lambda container=None, severity=None: calls["severity"].append(
        (container, severity)
    )
    rules.add_note = lambda **kw: calls["notes"].append(kw)
    rules.run_query = lambda **kw: []
    rules.get_container = lambda cid: {"id": cid}
    rules.get_artifacts = lambda **kw: []

    pkg = types.ModuleType("phantom")
    pkg.rules = rules
    monkeypatch.setitem(sys.modules, "phantom", pkg)
    monkeypatch.setitem(sys.modules, "phantom.rules", rules)
    return calls


def build_registry(telemetry=None, allowed=None, role="triage"):
    return ToolRegistry(
        FakeConnector(),
        container_id=42,
        allowed_tools=allowed if allowed is not None else ALL_TOOLS,
        telemetry=telemetry,
        agent_id="triage-bot",
        agent_role=role,
    )


# --- §8.1: a decision action emits exactly one agent_action -----------------

def test_decision_action_emits_one_agent_action(fake_phantom):
    tel = FakeTelemetry()
    reg = build_registry(tel)

    result = reg.execute("update_severity", {"severity": "high"})

    assert result == "Container severity updated to: high"
    assert len(tel.sent) == 1
    event = tel.sent[0]
    assert event["event_type"] == "agent_action"
    assert event["actor"]["agent_role"] == "triage"
    assert event["actor"]["agent_id"] == "triage-bot"
    assert event["case_id"] == "42"
    assert event["action"]["tool"] == "update_severity"
    assert event["action"]["category"] == "Triage"


def test_event_carries_no_raw_text_only_ids(fake_phantom):
    """L8 / handoff §5: target must be an ID, never raw content."""
    tel = FakeTelemetry()
    build_registry(tel).execute("update_severity", {"severity": "critical"})
    assert tel.sent[0]["resource"]["target"] == "container:42"


# --- §8.2: telemetry must never block the action ---------------------------

def test_telemetry_failure_never_blocks_the_action(fake_phantom):
    reg = build_registry(ExplodingTelemetry())

    result = reg.execute("update_severity", {"severity": "high"})

    # The business action still went through, despite emit() raising.
    assert result == "Container severity updated to: high"
    assert fake_phantom["severity"] == [(42, "high")]


def test_missing_telemetry_client_is_safe(fake_phantom):
    reg = build_registry(telemetry=None)
    assert reg.execute("update_severity", {"severity": "low"}) == "Container severity updated to: low"


# --- decision #1: only real decisions are counted ---------------------------

@pytest.mark.parametrize(
    "tool, params",
    [
        ("splunk_search", {"spl_query": "index=main"}),
        ("add_note", {"title": "t", "content": "c"}),
        ("get_container_info", {}),
    ],
)
def test_enrichment_and_documentation_do_not_emit(fake_phantom, tool, params):
    """Only decisions belong in the Precision denominator."""
    tel = FakeTelemetry()
    build_registry(tel).execute(tool, params)
    assert tel.sent == []


# --- outcome mapping --------------------------------------------------------

def test_blocked_tool_emits_blocked_outcome(fake_phantom):
    tel = FakeTelemetry()
    reg = build_registry(tel, allowed=["add_note"])  # update_severity not allowed

    result = reg.execute("update_severity", {"severity": "high"})

    assert "not in the allowed tools list" in result
    assert len(tel.sent) == 1
    assert tel.sent[0]["outcome"] == "blocked"


def test_platform_failure_emits_failed_outcome(fake_phantom, monkeypatch):
    def blow_up(**kwargs):
        raise RuntimeError("SOAR down")

    monkeypatch.setattr(sys.modules["phantom.rules"], "update_container", blow_up)
    tel = FakeTelemetry()

    result = build_registry(tel).execute("update_severity", {"severity": "high"})

    assert result.startswith("Failed to update severity")
    assert tel.sent[0]["outcome"] == "failed"


def test_invalid_input_emits_failed_outcome(fake_phantom):
    tel = FakeTelemetry()
    result = build_registry(tel).execute("update_severity", {"severity": "banana"})
    assert result.startswith("ERROR: severity must be one of")
    assert tel.sent[0]["outcome"] == "failed"


@pytest.mark.parametrize(
    "result, expected",
    [
        ("Container severity updated to: high", "success"),
        ("ERROR: Tool 'x' is not in the allowed tools list.", "blocked"),
        ("ERROR: severity must be one of [...]", "failed"),
        ("Failed to update severity: boom", "failed"),
        ("ERROR executing update_severity: boom", "failed"),
    ],
)
def test_outcome_mapping(result, expected):
    assert _outcome_from_result(result) == expected


# --- decision #2: agent_role is a fixed enum --------------------------------

def test_unknown_agent_role_falls_back_to_default(fake_phantom):
    tel = FakeTelemetry()
    reg = build_registry(tel, role="wizard")  # not in the enum

    reg.execute("update_severity", {"severity": "high"})

    assert reg.agent_role == DEFAULT_AGENT_ROLE
    assert tel.sent[0]["actor"]["agent_role"] == DEFAULT_AGENT_ROLE


@pytest.mark.parametrize("role", AGENT_ROLES)
def test_every_declared_role_survives_to_the_event(fake_phantom, role):
    """The panel groups by agent_role; every enum value must pass through intact."""
    tel = FakeTelemetry()
    build_registry(tel, role=role).execute("update_severity", {"severity": "high"})
    assert tel.sent[0]["actor"]["agent_role"] == role


# --- §5.1: override is not knowable at emit time ----------------------------

def test_analyst_override_is_always_false_for_now(fake_phantom):
    """Documents handoff §5.1: Precision is NOT valid until the SPL-join lands."""
    tel = FakeTelemetry()
    build_registry(tel).execute("update_severity", {"severity": "high"})
    assert tel.sent[0]["analyst_override"] in (False, 0)
