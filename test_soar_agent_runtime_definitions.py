"""Tests for RT-B02 (Umsetzungskonzept 5.2): agent and skill definitions from Git.

GitDefinitionStore reads agents/<id>.toml and skills/<id>.toml below
definitions_path, validates each file and adds definition_version; the
connector uses it for run agent, list agents, list skills and test
connectivity when the asset field is set. Runs off-platform via conftest.py.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

import soar_agent_runtime_connector as connector_module
from soar_agent_runtime_connector import SoarAgentRuntimeConnector
from soar_agent_runtime_definitions import (
    AGENTS_DIR,
    SKILLS_DIR,
    DefinitionError,
    DefinitionNotFound,
    GitDefinitionStore,
)

AGENT = '''agent_id = "triage"
version = "1.2.0"
provider = "local"
model = "qwen2.5:7b"
agent_role = "triage"
max_steps = 6
allowed_tools = ["get_container_info", "add_note"]
skills = ["verdict"]
system_prompt = "You are a SOC analyst."
'''

SKILL = '''skill_id = "verdict"
version = "0.3.0"
description = "verdict"
inject = "Answer with a verdict."

[output_schema]
verdict = ""
reason = ""
'''


class _Connector:
    def __init__(self):
        self.progress = []

    def save_progress(self, message):
        self.progress.append(message)


def write_defs(root, agents=None, skills=None):
    for kind, files in ((AGENTS_DIR, agents or {}), (SKILLS_DIR, skills or {})):
        (root / kind).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / kind / f"{name}.toml").write_text(text)
    return root


def agents(root, known_tools=None):
    return GitDefinitionStore(_Connector(), str(root), AGENTS_DIR, known_tools=known_tools)


def skills(root):
    return GitDefinitionStore(_Connector(), str(root), SKILLS_DIR)


# --- store --------------------------------------------------------------------

def test_load_returns_the_definition_with_version_and_file_hash(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT})
    agent = agents(tmp_path).load("triage")
    digest = hashlib.sha256(AGENT.encode()).hexdigest()[:12]
    assert agent["definition_version"] == f"triage@1.2.0#{digest}"
    assert agent["allowed_tools"] == ["get_container_info", "add_note"]
    assert agent["max_steps"] == 6


def test_every_edit_changes_the_version_even_with_the_same_version_field(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT})
    before = agents(tmp_path).load("triage")["definition_version"]
    write_defs(tmp_path, agents={"triage": AGENT.replace("SOC analyst", "SOC lead")})
    after = agents(tmp_path).load("triage")["definition_version"]
    assert before.split("#")[0] == after.split("#")[0] and before != after


def test_missing_agent_raises_not_found(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT})
    with pytest.raises(DefinitionNotFound, match="Agent 'other' not found"):
        agents(tmp_path).load("other")


@pytest.mark.parametrize("bad_id", ["../triage", "a/b", "", "x" * 65])
def test_ids_that_could_leave_the_directory_are_not_found(tmp_path, bad_id):
    write_defs(tmp_path, agents={"triage": AGENT})
    with pytest.raises(DefinitionNotFound):
        agents(tmp_path).load(bad_id)


def test_missing_directory_names_the_path(tmp_path):
    with pytest.raises(DefinitionError, match="directory not found"):
        agents(tmp_path / "nowhere").load("triage")


@pytest.mark.parametrize("text,reason", [
    ("agent_id = ", "not valid TOML"),
    (AGENT.replace('agent_id = "triage"', 'agent_id = "other"'), "must be 'triage'"),
    (AGENT.replace('version = "1.2.0"\n', ""), "'version' missing"),
    (AGENT.replace('model = "qwen2.5:7b"', "model = 7"), "'model' missing or not a non-empty string"),
    (AGENT.replace('system_prompt = "You are a SOC analyst."', 'system_prompt = ""'), "'system_prompt'"),
    (AGENT.replace('allowed_tools = ["get_container_info", "add_note"]', 'allowed_tools = "add_note"'),
     "allowed_tools must be a list"),
    (AGENT.replace('skills = ["verdict"]', "skills = [1]"), "skills must be a list"),
    (AGENT.replace("max_steps = 6", "max_steps = 0"), "max_steps must be a positive integer"),
    (AGENT.replace("max_steps = 6", "max_steps = true"), "max_steps must be a positive integer"),
    (AGENT.replace('agent_role = "triage"', 'agent_role = "boss"'), "agent_role must be one of"),
])
def test_invalid_agent_files_name_file_and_reason(tmp_path, text, reason):
    write_defs(tmp_path, agents={"triage": text})
    with pytest.raises(DefinitionError) as exc:
        agents(tmp_path).load("triage")
    assert str(exc.value).startswith("agents/triage.toml: ") and reason in str(exc.value)


def test_unknown_tools_are_rejected_when_the_tool_list_is_known(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT.replace('"add_note"', '"add_notes"')})
    with pytest.raises(DefinitionError, match="unknown tool.*add_notes"):
        agents(tmp_path, known_tools=["get_container_info", "add_note"]).load("triage")


def test_oversized_file_is_rejected(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT + "# " + "x" * (256 * 1024) + "\n"})
    with pytest.raises(DefinitionError, match="larger than 256 KB"):
        agents(tmp_path).load("triage")


def test_list_all_keeps_valid_files_and_reports_the_rest(tmp_path):
    write_defs(tmp_path, agents={
        "triage": AGENT,
        "broken": "agent_id = ",
        "renamed": AGENT,                       # agent_id inside says "triage"
    })
    (tmp_path / AGENTS_DIR / "notes.md").write_text("ignored")
    store = agents(tmp_path)
    assert [a["agent_id"] for a in store.list_all()] == ["triage"]
    assert len(store.problems) == 2
    assert any(p.startswith("agents/broken.toml: not valid TOML") for p in store.problems)
    assert any(p.startswith("agents/renamed.toml: agent_id must be 'renamed'") for p in store.problems)


def test_skill_with_output_schema_table(tmp_path):
    write_defs(tmp_path, skills={"verdict": SKILL})
    skill = skills(tmp_path).load("verdict")
    assert skill["output_schema"] == {"verdict": "", "reason": ""}
    assert skill["definition_version"].startswith("verdict@0.3.0#")


def test_skill_output_schema_must_be_a_table(tmp_path):
    write_defs(tmp_path, skills={"verdict": 'skill_id = "verdict"\nversion = "1"\noutput_schema = "x"\n'})
    with pytest.raises(DefinitionError, match="output_schema must be a table"):
        skills(tmp_path).load("verdict")


def test_load_many_skips_missing_skills_with_a_warning(tmp_path):
    write_defs(tmp_path, skills={"verdict": SKILL})
    store = skills(tmp_path)
    assert [s["skill_id"] for s in store.load_many(["verdict", "gone"])] == ["verdict"]
    assert any("skill 'gone' not found" in m for m in store.connector.progress)


def test_load_many_fails_on_an_invalid_skill(tmp_path):
    write_defs(tmp_path, skills={"verdict": "skill_id = "})
    with pytest.raises(DefinitionError, match="skills/verdict.toml"):
        skills(tmp_path).load_many(["verdict"])


def test_save_is_refused_with_a_pointer_to_git(tmp_path):
    store = agents(tmp_path)
    assert store.save("triage", {}) is False
    assert "agents/triage.toml" in store.last_error


# --- connector ------------------------------------------------------------------

def make_connector(**config) -> SoarAgentRuntimeConnector:
    c = SoarAgentRuntimeConnector()
    c._fake_config = config
    assert c.initialize() is True
    return c


def _capture_loop(monkeypatch):
    seen = {}

    class CapturingLoop:
        def __init__(self, **kw):
            seen.update(kw)

        def run(self, task):
            return {"agent_id": "triage", "lauf_id": "rt-x", "final_answer": "ok", "steps_taken": 1,
                    "provider": "local", "model": "m", "audit_ereignisse": 0, "audit_fehler": 0,
                    "step_log": "[]"}

    monkeypatch.setattr(connector_module, "ReActLoop", CapturingLoop)
    return seen


def test_without_definitions_path_the_custom_list_stores_are_used():
    c = make_connector()
    assert type(c._agent_store()).__name__ == "AgentDefinitionStore"
    assert type(c._skill_store()).__name__ == "SkillStore"


def test_run_agent_reads_git_and_puts_agent_and_skill_versions_into_the_audit(monkeypatch, tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT}, skills={"verdict": SKILL})
    seen = _capture_loop(monkeypatch)
    c = make_connector(definitions_path=f"{tmp_path}/")
    assert c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7}) is True
    assert re.fullmatch(r"triage@1\.2\.0#[0-9a-f]{12}, verdict@0\.3\.0#[0-9a-f]{12}", seen["def_version"])
    assert seen["system_prompt"] == "You are a SOC analyst."
    assert seen["max_steps"] == 6
    assert seen["required_keys"] == ["verdict", "reason"]
    assert seen["tools"].allowed_tools == ["get_container_info", "add_note"]


def test_run_agent_with_an_invalid_git_definition_stops_with_the_reason(monkeypatch, tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT.replace("max_steps = 6", "max_steps = -1")})
    seen = _capture_loop(monkeypatch)
    c = make_connector(definitions_path=str(tmp_path))
    assert c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7}) is False
    assert "agents/triage.toml: max_steps must be a positive integer" in c.action_results[-1].message
    assert not seen


def test_run_agent_rejects_unknown_tools_from_git(monkeypatch, tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT.replace('"add_note"', '"delete_everything"')})
    _capture_loop(monkeypatch)
    c = make_connector(definitions_path=str(tmp_path))
    assert c._handle_run_agent({"agent_id": "triage", "task": "look", "container_id": 7}) is False
    assert "delete_everything" in c.action_results[-1].message


def test_list_agents_and_skills_read_git_and_name_rejected_files(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT, "broken": "x = "}, skills={"verdict": SKILL})
    c = make_connector(definitions_path=str(tmp_path))
    assert c._handle_list_agents({}) is True
    result = c.action_results[-1]
    assert [d["agent_id"] for d in result.data] == ["triage"]
    assert result.data[0]["allowed_tools"] == "get_container_info, add_note"
    assert "Rejected files: agents/broken.toml" in result.message
    assert c._handle_list_skills({}) is True
    assert [d["skill_id"] for d in c.action_results[-1].data] == ["verdict"]
    assert c.action_results[-1].data[0]["has_output_schema"] is True


def test_connectivity_reports_the_definitions(tmp_path):
    write_defs(tmp_path, agents={"triage": AGENT, "broken": "x = "}, skills={"verdict": SKILL})
    results = {}
    make_connector(definitions_path=str(tmp_path))._test_definitions(results)
    assert results["definitions"].startswith(f"OK: 1 agents, 1 skills from {tmp_path}; REJECTED: agents/broken.toml")


@pytest.mark.parametrize("config,expected", [
    ({}, "SKIPPED (no definitions_path"),
    ({"definitions_path": "/nowhere"}, "ERROR: definitions_path: directory not found"),
])
def test_connectivity_definitions_without_path_or_with_a_wrong_one(config, expected):
    results = {}
    make_connector(**config)._test_definitions(results)
    assert results["definitions"].startswith(expected)


def test_app_json_declares_definitions_path_last():
    with open("soar_agent_runtime.json") as fh:
        config = json.load(fh)["configuration"]
    field = config["definitions_path"]
    assert field["data_type"] == "string" and field["required"] is False
    assert field["order"] == max(v["order"] for v in config.values())


# --- create agent / create skill with definitions_path (E5, 5.3) ------------------

class _NoWriteStore:
    def __init__(self, *a, **kw):
        raise AssertionError("a Custom List store must not be built")


@pytest.mark.parametrize("handler,param,file", [
    ("_handle_create_agent", {"agent_id": "triage", "system_prompt": "p", "provider": "local", "model": "m"},
     "agents/triage.toml"),
    ("_handle_create_skill", {"skill_id": "verdict", "inject": "x"}, "skills/verdict.toml"),
])
def test_create_is_refused_and_writes_nothing_with_definitions_path(monkeypatch, tmp_path, handler, param, file):
    monkeypatch.setattr(connector_module, "AgentDefinitionStore", _NoWriteStore)
    monkeypatch.setattr(connector_module, "SkillStore", _NoWriteStore)
    c = make_connector(definitions_path=str(tmp_path))
    assert getattr(c, handler)(param) is False
    message = c.action_results[-1].message
    assert "Definitions come from Git" in message and "nothing was saved" in message and file in message
    assert not list(tmp_path.rglob("*.toml"))


def test_create_agent_without_definitions_path_still_uses_the_custom_list(monkeypatch):
    saved = []

    class Store:
        def __init__(self, *a, **kw):
            pass

        def save(self, agent_id, config):
            saved.append(agent_id)
            return True

    monkeypatch.setattr(connector_module, "AgentDefinitionStore", Store)
    c = make_connector()
    assert c._handle_create_agent({"agent_id": "triage", "system_prompt": "p", "provider": "local",
                                   "model": "m", "allowed_tools": "add_note"}) is True
    assert saved == ["triage"]
