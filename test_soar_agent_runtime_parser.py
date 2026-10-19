"""Strict ReAct parser tests for RT-P05 (Umsetzungskonzept Runtime, sub-change 1.5).

Rule (concept §3.7): a missing PARAMS line means {} (parameterless tools keep
working); a present but invalid PARAMS line is an error observation and the
tool is NOT executed.
"""

from __future__ import annotations

import types

import pytest

from soar_agent_runtime_utils import ReActLoop, _PARAMS_PARSE_ERROR


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


def make_loop(responses, max_steps=5):
    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    llm = FakeLLM(responses)
    tools = FakeTools()
    loop = ReActLoop(
        connector=connector, llm=llm, tools=tools, agent_id="t",
        system_prompt="You are a test agent.", max_steps=max_steps,
        container_id=1, extra_context="", skill_injection="",
    )
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
