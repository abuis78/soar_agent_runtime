"""Strict ReAct parser tests for RT-P05 (Umsetzungskonzept Runtime, sub-change 1.5).

Rule (concept §3.7): a missing PARAMS line means {} (parameterless tools keep
working); a present but invalid PARAMS line is an error observation and the
tool is NOT executed.
"""

from __future__ import annotations

import json
import re
import types

import pytest

from soar_agent_runtime_utils import ReActLoop, _PARAMS_PARSE_ERROR, new_run_id


# --- fakes ------------------------------------------------------------------

class FakeLLM:
    provider = "local"
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    def chat(self, system_prompt, messages):
        self.seen_messages.append([dict(m) for m in messages])
        return self.responses.pop(0)


class FakeTools:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_tool_descriptions(self):
        return "- get_container_info: no params\n- add_note: {\"title\": ..., \"content\": ...}"

    def execute(self, tool_name, params):
        self.calls.append((tool_name, params))
        return f"OK {tool_name}"

    @staticmethod
    def has_effect(tool_name):
        return tool_name == "update_severity"


def make_loop(responses, max_steps=5, audit=None):
    progress: list[str] = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)))
    llm = FakeLLM(responses)
    tools = FakeTools()
    loop = ReActLoop(
        connector=connector, llm=llm, tools=tools, agent_id="t",
        system_prompt="You are a test agent.", max_steps=max_steps,
        container_id=1, extra_context="", skill_injection="",
        audit=audit, def_version="defv1",
    )
    loop._progress = progress
    return loop, llm, tools


# --- _parse_response --------------------------------------------------------

def test_missing_params_line_is_empty_dict_without_error():
    loop, _, _ = make_loop([])
    tool, params, thought, final, err = loop._parse_response(
        "THOUGHT: check the case\nACTION: get_container_info")
    assert (tool, params, final, err) == ("get_container_info", {}, None, None)
    assert thought == "check the case"


def test_valid_multiline_params_are_parsed():
    loop, _, _ = make_loop([])
    text = 'THOUGHT: note it\nACTION: add_note\nPARAMS: {"title": "a",\n "content": "b"}'
    tool, params, _, _, err = loop._parse_response(text)
    assert (tool, params, err) == ("add_note", {"title": "a", "content": "b"}, None)


@pytest.mark.parametrize("bad", [
    '{"title": "a", "content": }',   # broken JSON
    "not json at all",
    '["a", "b"]',                    # valid JSON but not an object
    '"string"',
    "42",
])
def test_invalid_params_line_yields_parse_error_and_no_params(bad):
    loop, _, _ = make_loop([])
    tool, params, _, final, err = loop._parse_response(f"ACTION: add_note\nPARAMS: {bad}")
    assert tool == "add_note"
    assert params == {}
    assert final is None
    assert err == _PARAMS_PARSE_ERROR


def test_final_answer_still_parsed_unchanged():
    loop, _, _ = make_loop([])
    tool, params, _, final, err = loop._parse_response(
        "THOUGHT: done\nFINAL_ANSWER: {\"verdict\": \"benign\"}\nmore")
    assert tool is None and params == {} and err is None
    assert final == '{"verdict": "benign"}\nmore'


# --- run(): behaviour at the loop boundary ----------------------------------

def test_run_does_not_execute_tool_on_invalid_params_and_lets_model_retry():
    loop, llm, tools = make_loop([
        'THOUGHT: x\nACTION: add_note\nPARAMS: {"title": "a", "content": }',
        'THOUGHT: fixed\nACTION: add_note\nPARAMS: {"title": "a", "content": "b"}',
        "FINAL_ANSWER: done",
    ])
    result = loop.run("do it")
    assert tools.calls == [("add_note", {"title": "a", "content": "b"})]
    assert result["steps_taken"] == 3
    assert result["final_answer"] == "done"
    # the error went back to the model as the observation of step 1
    second_call = llm.seen_messages[1]
    assert second_call[-1] == {"role": "user", "content": f"OBSERVATION: {_PARAMS_PARSE_ERROR}"}
    assert loop.step_log[0]["action"] == "add_note"
    assert loop.step_log[0]["observation"] == _PARAMS_PARSE_ERROR
    assert loop.step_log[0]["params"] == {}


def test_run_executes_parameterless_tool_when_params_line_is_missing():
    loop, _, tools = make_loop([
        "THOUGHT: look\nACTION: get_container_info",
        "FINAL_ANSWER: ok",
    ])
    result = loop.run("do it")
    assert tools.calls == [("get_container_info", {})]
    assert result["final_answer"] == "ok"


def test_run_invalid_params_counts_against_max_steps():
    loop, _, tools = make_loop(
        ['ACTION: add_note\nPARAMS: {broken'] * 2, max_steps=2)
    result = loop.run("do it")
    assert tools.calls == []
    assert result["steps_taken"] == 2
    assert result["final_answer"].startswith("Max steps (2) reached")


# --- lauf_id / schritt_id (RT-P01, sub-change 1.6) ---------------------------

RUN_ID_RE = re.compile(r"^rt-\d{8}T\d{6}Z-[0-9a-f]{8}$")


def test_new_run_id_has_spec_format_and_is_unique():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(RUN_ID_RE.match(i) for i in ids)


def test_loop_generates_run_id_by_default_and_accepts_one():
    loop, _, _ = make_loop([])
    assert RUN_ID_RE.match(loop.run_id)
    assert loop.step_id_prefix == loop.run_id
    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    given = ReActLoop(connector=connector, llm=FakeLLM([]), tools=FakeTools(), agent_id="t",
                      system_prompt="x", max_steps=1, container_id=1,
                      run_id="rt-20260101T000000Z-deadbeef", step_id_prefix="rt-20260101T000000Z-deadbeef/3.1")
    assert given.run_id == "rt-20260101T000000Z-deadbeef"
    assert given.step_id_prefix == "rt-20260101T000000Z-deadbeef/3.1"


def test_every_step_carries_schritt_id_and_result_carries_lauf_id():
    loop, _, _ = make_loop([
        "ACTION: get_container_info",
        'ACTION: add_note\nPARAMS: {broken',
        "FINAL_ANSWER: ok",
    ])
    result = loop.run("do it")
    assert result["lauf_id"] == loop.run_id
    assert [e["schritt_id"] for e in loop.step_log] == [f"{loop.run_id}/{n}" for n in (1, 2, 3)]
    assert [e["schritt_id"] for e in json.loads(result["step_log"])] == [f"{loop.run_id}/{n}" for n in (1, 2, 3)]


def test_sub_agent_prefix_is_used_for_schritt_id():
    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    loop = ReActLoop(connector=connector, llm=FakeLLM(["FINAL_ANSWER: ok"]), tools=FakeTools(),
                     agent_id="child", system_prompt="x", max_steps=1, container_id=1,
                     run_id="rt-20260101T000000Z-deadbeef", step_id_prefix="rt-20260101T000000Z-deadbeef/4.1")
    loop.run("t")
    assert loop.step_log[0]["schritt_id"] == "rt-20260101T000000Z-deadbeef/4.1/1"


def test_error_result_also_carries_lauf_id():
    class ExplodingLLM(FakeLLM):
        def chat(self, system_prompt, messages):
            raise RuntimeError("down")

    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    loop = ReActLoop(connector=connector, llm=ExplodingLLM([]), tools=FakeTools(), agent_id="t",
                     system_prompt="x", max_steps=1, container_id=1)
    result = loop.run("t")
    assert result["final_answer"].startswith("ERROR: LLM call failed")
    assert result["lauf_id"] == loop.run_id


# --- audit per step, fail-closed (RT-P02/P03, sub-change 1.8) ---------------

class FakeSink:
    """Records events; ``fail_at`` = 1-based write numbers that fail."""

    def __init__(self, fail_closed=False, fail_at=()):
        self.fail_closed = fail_closed
        self.fail_at = set(fail_at)
        self.events: list[dict] = []
        self.attempts = 0

    def write(self, event):
        self.attempts += 1
        if self.attempts in self.fail_at or "all" in self.fail_at:
            return False
        self.events.append(event)
        return True

    def flush(self):
        return True


def _phases(sink):
    return [(e["schritt_id"].rsplit("/", 1)[1], e["werkzeug"], e["phase"]) for e in sink.events]


def test_no_audit_sink_keeps_counters_at_zero():
    loop, _, _ = make_loop(["FINAL_ANSWER: ok"])
    result = loop.run("t")
    assert (result["audit_ereignisse"], result["audit_fehler"]) == (0, 0)


def test_every_step_type_is_audited_and_effecting_tools_are_announced_first():
    sink = FakeSink()
    loop, _, tools = make_loop([
        "ACTION: get_container_info",
        'ACTION: update_severity\nPARAMS: {"severity": "high"}',
        "ACTION: add_note\nPARAMS: {broken",
        "just rambling",
        "FINAL_ANSWER: done",
    ], audit=sink)
    result = loop.run("t")
    assert _phases(sink) == [
        ("1", "get_container_info", "erledigt"),
        ("2", "update_severity", "beabsichtigt"),
        ("2", "update_severity", "erledigt"),
        ("3", "add_note", "fehler"),
        ("4", "FINAL_ANSWER", "fehler"),
        ("5", "FINAL_ANSWER", "erledigt"),
    ]
    assert all(e["lauf_id"] == loop.run_id and e["definition_version"] == "defv1" for e in sink.events)
    assert json.loads(sink.events[1]["parameter"]) == {"severity": "high"}
    assert sink.events[1]["ergebnis_hash"] == ""          # intended: nothing happened yet
    assert sink.events[2]["ergebnis_hash"] != ""          # done: hash of the observation
    assert (result["audit_ereignisse"], result["audit_fehler"]) == (6, 0)
    assert tools.calls == [("get_container_info", {}), ("update_severity", {"severity": "high"})]


def test_fail_closed_blocks_effecting_tool_when_intent_cannot_be_stored():
    sink = FakeSink(fail_closed=True, fail_at={"all"})
    loop, _, tools = make_loop(['ACTION: update_severity\nPARAMS: {"severity": "high"}'], audit=sink)
    result = loop.run("t")
    assert tools.calls == []
    assert result["final_answer"].startswith("ERROR: audit not writable at step ")
    assert "before 'update_severity'" in result["final_answer"]
    assert result["steps_taken"] == 1
    assert loop.step_log[0]["observation"].startswith("ERROR: audit not writable, 'update_severity' not executed")
    assert result["audit_fehler"] == 1


def test_fail_closed_non_effecting_tool_runs_then_stops_when_its_event_fails():
    sink = FakeSink(fail_closed=True, fail_at={1})
    loop, _, tools = make_loop(['ACTION: add_note\nPARAMS: {"title": "a", "content": "b"}', "FINAL_ANSWER: x"],
                               audit=sink)
    result = loop.run("t")
    assert tools.calls == [("add_note", {"title": "a", "content": "b"})]
    assert result["final_answer"].startswith("ERROR: audit not writable at step ")
    assert "after 'add_note'" in result["final_answer"]
    assert result["steps_taken"] == 1
    assert any("[audit] write failed" in m and "fail-closed" in m for m in loop._progress)


def test_fail_open_counts_failures_and_still_executes():
    sink = FakeSink(fail_closed=False, fail_at={"all"})
    loop, _, tools = make_loop(['ACTION: update_severity\nPARAMS: {"severity": "high"}', "FINAL_ANSWER: ok"],
                               audit=sink)
    result = loop.run("t")
    assert tools.calls == [("update_severity", {"severity": "high"})]
    assert result["final_answer"] == "ok"
    assert (result["audit_ereignisse"], result["audit_fehler"]) == (0, 3)
    assert any("fail-open" in m for m in loop._progress)


def test_llm_failure_is_audited_as_error():
    class ExplodingLLM(FakeLLM):
        def chat(self, system_prompt, messages):
            raise RuntimeError("down sk-abcdefghijklmnop")

    sink = FakeSink()
    progress = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(m))
    loop = ReActLoop(connector=connector, llm=ExplodingLLM([]), tools=FakeTools(), agent_id="t",
                     system_prompt="x", max_steps=1, container_id=1, audit=sink)
    result = loop.run("t")
    assert _phases(sink) == [("1", "LLM", "fehler")]
    assert "sk-abcdefghijklmnop" not in json.dumps(sink.events)
    assert result["audit_ereignisse"] == 1


def test_sink_exception_is_treated_as_failed_write():
    class Raising(FakeSink):
        def write(self, event):
            raise RuntimeError("boom")

    loop, _, _ = make_loop(["FINAL_ANSWER: ok"], audit=Raising(fail_closed=True))
    result = loop.run("t")
    assert result["final_answer"].startswith("ERROR: audit not writable")


def test_has_effect_flags():
    from soar_agent_runtime_utils import ToolRegistry
    assert ToolRegistry.has_effect("update_severity") is True
    for name in ("splunk_search", "add_note", "get_container_info", "run_sub_agent"):
        assert ToolRegistry.has_effect(name) is False, name
    assert ToolRegistry.has_effect("unknown_tool") is True


# --- RT-P05 with weak models: first ACTION runs, the rest is named (1.11.2) ------------
# Raw turn of qwen2.5:7b on SOAR 8.7 (run rt-20261025T053802Z-17af70c7). 1.11.0 dropped
# update_severity silently; 1.11.1 rejected the turn and the model repeated it 6 times.

TWO_ACTIONS = ('THOUGHT: Based on the current information, I will update the severity to high and add a note '
               'explaining the rationale.\n'
               'ACTION: update_severity\nPARAMS: {"severity": "high"}\n\n'
               'ACTION: add_note\nPARAMS: {"title": "Rationale for High Severity", "content": "malware URL"}')


def test_first_of_two_actions_is_parsed_with_its_own_params():
    loop, _, _ = make_loop([])
    tool, params, _thought, final, err = loop._parse_response(TWO_ACTIONS)
    assert (tool, params, final, err) == ("update_severity", {"severity": "high"}, None, None)
    assert loop._dropped_actions == ["add_note"] and loop._final_dropped is False


def test_action_with_final_answer_runs_the_action_and_drops_the_answer():
    loop, _, _ = make_loop([])
    tool, params, _t, final, err = loop._parse_response(
        'THOUGHT: done\nACTION: update_severity\nPARAMS: {"severity": "high"}\nFINAL_ANSWER: severity set')
    assert (tool, params, final, err) == ("update_severity", {"severity": "high"}, None, None)
    assert loop._final_dropped is True


def test_the_real_turn_sets_the_severity_and_tells_the_model_what_did_not_run():
    loop, llm, tools = make_loop([
        TWO_ACTIONS,
        'THOUGHT: now the note\nACTION: add_note\nPARAMS: {"title": "Why", "content": "malware URL"}',
        "THOUGHT: done\nFINAL_ANSWER: severity high, note added",
    ])
    result = loop.run("Call update_severity with severity high. Then add a note explaining why.")
    assert tools.calls == [("update_severity", {"severity": "high"}), ("add_note", {"title": "Why", "content": "malware URL"})]
    log = json.loads(result["step_log"])
    assert (log[0]["action"], log[0]["not_executed"]) == ("update_severity", ["add_note"])
    told = llm.seen_messages[1][-1]["content"]
    assert "NOTE: only the first ACTION of your response was executed. add_note was NOT executed" in told
    assert result["final_answer"] == "severity high, note added"


def test_invalid_params_of_the_first_action_still_block_it():
    loop, _, _ = make_loop([])
    _tool, _params, _t, _final, err = loop._parse_response(
        "ACTION: update_severity\nPARAMS: {not json}\nACTION: add_note\nPARAMS: {}")
    assert err == _PARAMS_PARSE_ERROR


def test_step_log_keeps_the_raw_model_turn_truncated():
    loop, _, _ = make_loop(["THOUGHT: x\nACTION: get_container_info", "THOUGHT: y\nFINAL_ANSWER: " + "z" * 5000])
    log = json.loads(loop.run("t")["step_log"])
    assert log[0]["raw_response"] == "THOUGHT: x\nACTION: get_container_info"
    assert len(log[1]["raw_response"]) == 2000
    assert "not_executed" not in log[0]


def test_a_single_action_still_parses_as_before():
    loop, _, _ = make_loop([])
    assert loop._parse_response('ACTION: add_note\nPARAMS: {"title": "t", "content": "c"}')[4] is None
    assert loop._dropped_actions == []



# --- E2E finding E-7: Markdown answers get a precise correction, the parser stays strict ------

MARKDOWN_TURN = (
    "### Action: update_severity\n\n```json\n{\n  \"container\": 212,\n  \"severity\": \"medium\"\n}\n```")


def test_markdown_turn_is_not_executed_and_the_model_is_told_the_exact_form():
    loop, llm, tools = make_loop([MARKDOWN_TURN, "THOUGHT: ok\nFINAL_ANSWER: done"])
    result = loop.run("t")
    assert tools.calls == []                                            # strict: nothing ran (RT-P05)
    told = llm.seen_messages[1][-1]["content"]
    assert told.startswith("OBSERVATION: Your response used Markdown (headings or code blocks); nothing was executed.")
    assert "ACTION: update_severity\nPARAMS: {...JSON object on one line...}" in told
    assert result["final_answer"] == "done"


def test_plain_garbage_gets_the_generic_reminder():
    loop, llm, _ = make_loop(["I think the severity should be high.", "THOUGHT: ok\nFINAL_ANSWER: done"])
    loop.run("t")
    told = llm.seen_messages[1][-1]["content"]
    assert told.startswith("OBSERVATION: No valid ACTION or FINAL_ANSWER; nothing was executed.")
    assert "ACTION: <tool_name>" in told
