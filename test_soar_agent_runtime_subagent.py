"""Tests for RT-B03 / AC-14 (Umsetzungskonzept 5.4, E6): run_sub_agent.

A sub-agent runs in the same run: same lauf_id, its own definition, allowlist
and broker context, step ids below the calling step, the caller's remaining
steps as budget, and a depth limit (max_subagent_depth). Runs off-platform
via conftest.py; agents come from TOML files (definitions_path) in tmp_path.
"""

from __future__ import annotations

import json
import re
import types

import pytest

import soar_agent_runtime_connector as connector_module
from soar_agent_runtime_connector import SoarAgentRuntimeConnector
from soar_agent_runtime_policy import StubPolicyClient
from soar_agent_runtime_utils import ReActLoop, ToolRegistry


# --- ToolRegistry._tool_run_sub_agent with a fake spawn ------------------------------

def registry(spawn=None, depth=0, max_depth=2, steps_left=5):
    progress = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)))
    reg = ToolRegistry(connector, 7, ["run_sub_agent"], agent_id="parent", run_id="rt-x",
                       spawn=spawn, depth=depth, max_depth=max_depth)
    reg.set_step(3, "rt-x/3")
    reg.steps_left = steps_left
    return reg


class Spawn:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"final_answer": "child done", "steps_taken": 2}

    def __call__(self, **kw):
        self.calls.append(kw)
        return dict(self.result)


def test_child_gets_depth_step_id_and_the_remaining_budget():
    spawn = Spawn()
    reg = registry(spawn, steps_left=4)
    result = reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert spawn.calls == [{"agent_id": "child", "task": "look", "depth": 1, "step_id": "rt-x/3", "max_steps": 4}]
    assert result.startswith("SUB-AGENT RESULT agent_id=child steps=2")
    assert result.endswith("child done")
    assert reg.sub_steps_used == 2


def test_steps_total_of_the_child_counts_its_own_sub_agents():
    reg = registry(Spawn({"final_answer": "ok", "steps_taken": 2, "steps_total": 5}))
    reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert reg.sub_steps_used == 5


def test_checked_verdict_is_returned_as_json():
    verdict = {"belege": ["rt-x/3/1"], "begruendung": "b", "konfidenz": 0.8}
    reg = registry(Spawn({"final_answer": json.dumps(verdict), "urteil": verdict, "steps_taken": 1}))
    result = reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert json.loads(result.split("\n", 1)[1]) == verdict


@pytest.mark.parametrize("depth,max_depth", [(2, 2), (0, 0), (5, 3)])
def test_ac14_sub_agent_beyond_the_depth_limit_is_refused(depth, max_depth):
    spawn = Spawn()
    reg = registry(spawn, depth=depth, max_depth=max_depth)
    result = reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert result.startswith(f"ERROR: sub-agent 'child' refused: depth {depth + 1} exceeds max_subagent_depth "
                             f"{max_depth}")
    assert spawn.calls == []


def test_no_steps_left_refuses_the_child():
    spawn = Spawn()
    result = registry(spawn, steps_left=0).execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert "no steps left" in result and spawn.calls == []


@pytest.mark.parametrize("params", [{}, {"agent_id": "child"}, {"task": "look"}, {"agent_id": " ", "task": "x"}])
def test_agent_id_and_task_are_required(params):
    spawn = Spawn()
    assert registry(spawn).execute("run_sub_agent", params) == "ERROR: agent_id and task are required."
    assert spawn.calls == []


def test_without_factory_nothing_runs():
    result = registry(None).execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert result.startswith("ERROR: run_sub_agent is not available")


def test_failed_child_is_reported_as_failure():
    reg = registry(Spawn({"final_answer": "ERROR: Agent 'x' not found", "steps_taken": 0}))
    result = reg.execute("run_sub_agent", {"agent_id": "x", "task": "look"})
    assert result == "Failed sub-agent 'x': ERROR: Agent 'x' not found"


def test_halt_in_the_child_halts_the_caller():
    reg = registry(Spawn({"final_answer": "ERROR: run halted by Tool Broker (HALTED)", "steps_taken": 1,
                          "halted": True, "halt_source": "Tool Broker"}))
    reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert reg.halted is True and reg.halt_source == "Tool Broker"


# --- ReActLoop: shared step budget ---------------------------------------------------

class FakeLLM:
    provider, model = "local", "fake"

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, system_prompt, messages):
        return self.responses.pop(0)


def test_steps_used_by_sub_agents_count_against_the_callers_max_steps():
    spawn = Spawn({"final_answer": "ok", "steps_taken": 3})
    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    tools = ToolRegistry(connector, 7, ["run_sub_agent"], agent_id="parent", run_id="rt-x", spawn=spawn)
    call = 'ACTION: run_sub_agent\nPARAMS: {"agent_id": "child", "task": "look"}'
    loop = ReActLoop(connector=connector, llm=FakeLLM([call, call, call]), tools=tools, agent_id="parent",
                     system_prompt="x", max_steps=6, container_id=7, run_id="rt-x")
    result = loop.run("t")
    # step 1 (+3 child) leaves 2, step 2 gets a budget of 2 and the child reports 3 -> no step 3
    assert [c["max_steps"] for c in spawn.calls] == [5, 1]
    assert result["steps_taken"] == 2 and result["steps_total"] == 8
    assert result["final_answer"].startswith("Max steps (6) reached")


# --- connector end to end with TOML definitions --------------------------------------

def agent_toml(agent_id, prompt, tools, max_steps=5, provider="local"):
    return (f'agent_id = "{agent_id}"\nversion = "1"\nprovider = "{provider}"\nmodel = "m"\n'
            f'max_steps = {max_steps}\nallowed_tools = {json.dumps(tools)}\nsystem_prompt = "{prompt}"\n')


def write_agents(root, agents):
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "skills").mkdir(exist_ok=True)
    for agent_id, text in agents.items():
        (root / "agents" / f"{agent_id}.toml").write_text(text)


def scripted_llm(monkeypatch, scripts):
    """scripts: marker in the system prompt -> list of responses; records each chat call."""
    calls = []

    def chat(self, system_prompt, messages):
        for marker, responses in scripts.items():
            if marker in system_prompt:
                calls.append((marker, messages[-1]["content"]))
                return responses.pop(0)
        raise AssertionError("no script for this agent")

    monkeypatch.setattr(connector_module.LLMProvider, "chat", chat)
    return calls


def make_connector(**config):
    c = SoarAgentRuntimeConnector()
    c._fake_config = config
    assert c.initialize() is True
    return c


def sub_call(agent_id, task="check it"):
    return f'ACTION: run_sub_agent\nPARAMS: {json.dumps({"agent_id": agent_id, "task": task})}'


def test_orchestrator_runs_a_specialist_in_the_same_run(monkeypatch, tmp_path):
    write_agents(tmp_path, {
        "orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent", "add_note"]),
        "specialist": agent_toml("specialist", "ROLE-SPEC", ["get_container_info"]),
    })
    calls = scripted_llm(monkeypatch, {
        "ROLE-ORCH": [sub_call("specialist"), "FINAL_ANSWER: specialist says benign"],
        "ROLE-SPEC": ["FINAL_ANSWER: benign"],
    })
    c = make_connector(definitions_path=str(tmp_path), audit_enabled=True)
    assert c._handle_run_agent({"agent_id": "orchestrator", "task": "triage", "container_id": 7}) is True
    result = c.action_results[-1].data[0]
    lauf_id = result["lauf_id"]
    assert result["final_answer"] == "specialist says benign"
    assert result["steps_taken"] == 2 and result["steps_total"] == 3
    assert calls[0] == ("ROLE-ORCH", "triage") and calls[1] == ("ROLE-SPEC", "check it")
    assert "SUB-AGENT RESULT agent_id=specialist" in calls[2][1] and "benign" in calls[2][1]
    # one lauf_id for the whole tree; the child's steps sit below the calling step
    cef = [a["cef"] for a in c.artifacts]
    assert {e["lauf_id"] for e in cef} == {lauf_id}
    assert [(e["agent_id"], e["schritt_id"]) for e in cef] == [
        ("specialist", f"{lauf_id}/1/1"), ("orchestrator", f"{lauf_id}/1"), ("orchestrator", f"{lauf_id}/2")]
    assert any("[sub-agent] orchestrator -> specialist (depth 1)" in m for m in c.progress)


def test_sub_agent_has_only_its_own_allowlist(monkeypatch, tmp_path):
    write_agents(tmp_path, {
        "orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent", "add_note"]),
        "specialist": agent_toml("specialist", "ROLE-SPEC", ["get_container_info"]),
    })
    note = 'ACTION: add_note\nPARAMS: {"title": "t", "content": "c"}'
    calls = scripted_llm(monkeypatch, {
        "ROLE-ORCH": [sub_call("specialist"), "FINAL_ANSWER: done"],
        "ROLE-SPEC": [note, "FINAL_ANSWER: could not note"],
    })
    c = make_connector(definitions_path=str(tmp_path))
    c._handle_run_agent({"agent_id": "orchestrator", "task": "triage", "container_id": 7})
    child_obs = [content for marker, content in calls if marker == "ROLE-SPEC"][1]
    assert "ERROR: Tool 'add_note' is not in the allowed tools list." in child_obs


def test_ac14_self_recursion_stops_at_the_depth_limit(monkeypatch, tmp_path):
    write_agents(tmp_path, {"loop": agent_toml("loop", "ROLE-LOOP", ["run_sub_agent"], max_steps=10)})
    calls = scripted_llm(monkeypatch, {"ROLE-LOOP": [sub_call("loop")] * 3 + ["FINAL_ANSWER: stop"] * 3})
    c = make_connector(definitions_path=str(tmp_path), max_subagent_depth=2)
    assert c._handle_run_agent({"agent_id": "loop", "task": "go", "container_id": 7}) is True
    refused = [content for _, content in calls if "exceeds max_subagent_depth 2" in content]
    assert len(refused) == 1 and "depth 3" in refused[0]


def test_max_subagent_depth_zero_disables_sub_agents(monkeypatch, tmp_path):
    write_agents(tmp_path, {
        "orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent"]),
        "specialist": agent_toml("specialist", "ROLE-SPEC", []),
    })
    calls = scripted_llm(monkeypatch, {"ROLE-ORCH": [sub_call("specialist"), "FINAL_ANSWER: alone"],
                                       "ROLE-SPEC": []})
    c = make_connector(definitions_path=str(tmp_path), max_subagent_depth=0)
    c._handle_run_agent({"agent_id": "orchestrator", "task": "triage", "container_id": 7})
    assert "depth 1 exceeds max_subagent_depth 0" in calls[1][1]


def test_unknown_sub_agent_comes_back_as_failure(monkeypatch, tmp_path):
    write_agents(tmp_path, {"orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent"])})
    calls = scripted_llm(monkeypatch, {"ROLE-ORCH": [sub_call("ghost"), "FINAL_ANSWER: no ghost"]})
    c = make_connector(definitions_path=str(tmp_path))
    assert c._handle_run_agent({"agent_id": "orchestrator", "task": "t", "container_id": 7}) is True
    assert calls[1][1].startswith("OBSERVATION:") and "Failed sub-agent 'ghost': ERROR: Agent 'ghost' not found" in calls[1][1]


def test_airgap_sub_agent_with_a_cloud_provider_is_refused_before_any_call(monkeypatch, tmp_path):
    write_agents(tmp_path, {
        "orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent"]),
        "cloudy": agent_toml("cloudy", "ROLE-CLOUD", [], provider="anthropic"),
    })
    verdict = {"belege": [], "begruendung": "b", "gegenargumente": ["g"], "unsicherheiten": ["u"],
               "konfidenz": 0.5, "datenlage": "unbekannt"}
    calls = scripted_llm(monkeypatch, {"ROLE-ORCH": [sub_call("cloudy"), "FINAL_ANSWER: " + json.dumps(verdict)],
                                       "ROLE-CLOUD": []})
    c = make_connector(definitions_path=str(tmp_path), airgap_mode=True, anthropic_api_key="k")
    c._policy_client = StubPolicyClient()
    assert c._handle_run_agent({"agent_id": "orchestrator", "task": "t", "container_id": 7}) is True
    obs = calls[1][1]
    assert "Failed sub-agent 'cloudy'" in obs and "airgap" in obs.lower()
    assert all(marker != "ROLE-CLOUD" for marker, _ in calls)


def test_app_json_declares_max_subagent_depth():
    with open("soar_agent_runtime.json") as fh:
        field = json.load(fh)["configuration"]["max_subagent_depth"]
    assert (field["data_type"], field["default"], field["order"], field["required"]) == ("numeric", 2, 38, False)


def test_max_subagent_depth_defaults_to_2_and_never_goes_negative():
    assert make_connector()._max_subagent_depth == 2
    assert make_connector(max_subagent_depth=-3)._max_subagent_depth == 0


# --- E-10 / E-11 (1.16.1): child evidence counts for the caller, audit totals ----------

def test_child_evidence_becomes_valid_evidence_of_the_caller_and_is_listed():
    reg = registry(Spawn({"final_answer": "ok", "steps_taken": 2, "evidence_ids": ["rt-x/3/1", "rt-x/3/2", "tb-9"],
                          "audit_ereignisse": 2, "audit_fehler": 1}))
    result = reg.execute("run_sub_agent", {"agent_id": "child", "task": "look"})
    assert "evidence ids you may cite: rt-x/3/1, rt-x/3/2, tb-9:" in result
    assert {"rt-x/3/1", "rt-x/3/2", "tb-9"} <= reg.valid_evidence()
    assert {"rt-x/3/1", "rt-x/3/2", "tb-9"} <= reg.evidence_ids()
    assert (reg.sub_audit_written, reg.sub_audit_failed) == (2, 1)


def test_long_evidence_lists_are_shortened_in_the_result_line():
    ids = [f"rt-x/3/{n}" for n in range(1, 26)]
    result = registry(Spawn({"final_answer": "ok", "steps_taken": 25, "evidence_ids": ids})).execute(
        "run_sub_agent", {"agent_id": "child", "task": "look"})
    assert "(+5 more)" in result


def test_e10_parent_may_cite_the_childs_steps_in_its_final_answer(monkeypatch, tmp_path):
    write_agents(tmp_path, {
        "orchestrator": agent_toml("orchestrator", "ROLE-ORCH", ["run_sub_agent"]),
        "specialist": agent_toml("specialist", "ROLE-SPEC", ["get_container_info"]),
    })

    def verdict(belege):
        return "FINAL_ANSWER: " + json.dumps({"belege": belege, "begruendung": "b", "gegenargumente": ["g"],
                                              "unsicherheiten": ["u"], "konfidenz": 0.6, "datenlage": "teilweise"})

    seen = []

    def chat(self, system_prompt, messages):
        last = messages[-1]["content"]
        seen.append(last)
        if "ROLE-SPEC" in system_prompt:
            return verdict([])
        if "SUB-AGENT RESULT" in last:
            child_step = re.search(r"evidence ids you may cite: ([^,:\s]+)", last).group(1)
            return verdict([child_step])
        return sub_call("specialist")

    monkeypatch.setattr(connector_module.LLMProvider, "chat", chat)
    c = make_connector(definitions_path=str(tmp_path), airgap_mode=True)
    c._policy_client = StubPolicyClient()
    assert c._handle_run_agent({"agent_id": "orchestrator", "task": "t", "container_id": 7}) is True
    result = c.action_results[-1].data[0]
    assert result["urteil"]["belege"] == [f"{result['lauf_id']}/1/1"]     # accepted, no correction round
    assert result["steps_taken"] == 2
    # E-11: the summary counts the child's audit events too
    assert result["audit_ereignisse"] == len(c.artifacts) == 3
    assert c.action_results[-1].summary["audit_ereignisse"] == 3
