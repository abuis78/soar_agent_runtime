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
from soar_agent_runtime_utils import ToolRegistry

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


def registry(airgap=True, allowed=("update_severity", "add_note", "broker_invoke"), broker=None, steps=2):
    progress = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)))
    reg = ToolRegistry(connector, 4711, list(allowed), agent_id="triage", broker=broker, run_id="rt-x", airgap=airgap)
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
    assert result == "Container severity updated to: high"    # gate itself arrives in 3.4
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
