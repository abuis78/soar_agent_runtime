"""FINAL_ANSWER contract tests for RT-P04/RT-P06 (Umsetzungskonzept Runtime, sub-change 1.9, E3)."""

from __future__ import annotations

import json
import types

import pytest

from soar_agent_runtime_utils import (
    DATENLAGE_VALUES,
    FINAL_ANSWER_CONTRACT,
    REQUIRED_ANSWER_FIELDS,
    ReActLoop,
    validate_final_answer,
)


# --- fakes ------------------------------------------------------------------

class FakeLLM:
    provider, model = "local", "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []
        self.system_prompts: list[str] = []

    def chat(self, system_prompt, messages):
        self.system_prompts.append(system_prompt)
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


class FakeTools:
    def __init__(self):
        self.calls = []

    def get_tool_descriptions(self):
        return "- get_container_info(): info"

    @staticmethod
    def has_effect(name):
        return False

    def execute(self, name, params):
        self.calls.append((name, params))
        return f"OK {name}"


def make_loop(responses, *, strict=False, required_keys=(), max_steps=6):
    connector = types.SimpleNamespace(save_progress=lambda *a, **k: None)
    llm = FakeLLM(responses)
    loop = ReActLoop(connector=connector, llm=llm, tools=FakeTools(), agent_id="t",
                     system_prompt="x", max_steps=max_steps, container_id=1,
                     strict_schema=strict, required_keys=required_keys)
    return loop, llm


def good(**over):
    base = {"belege": [], "begruendung": "because", "gegenargumente": ["maybe not"],
            "unsicherheiten": [], "konfidenz": 0.8, "datenlage": "teilweise"}
    base.update(over)
    return base


# --- validate_final_answer --------------------------------------------------

def test_valid_strict_answer_has_no_errors():
    data, errors = validate_final_answer(json.dumps(good()), strict=True)
    assert errors == [] and data["konfidenz"] == 0.8


@pytest.mark.parametrize("text", ["free text", "[1,2]", '"s"', ""])
def test_non_object_is_rejected_even_without_strict(text):
    data, errors = validate_final_answer(text, required_keys=["verdict"])
    assert data is None and errors == ["FINAL_ANSWER must be a JSON object"]


@pytest.mark.parametrize("field", REQUIRED_ANSWER_FIELDS)
def test_each_missing_mandatory_field_is_reported(field):
    d = good()
    d.pop(field)
    _, errors = validate_final_answer(json.dumps(d), strict=True)
    assert errors == [f"missing field '{field}'"]


@pytest.mark.parametrize("over,needle", [
    ({"belege": "x"}, "'belege' must be a list"),
    ({"belege": [""]}, "needs a string beleg_id"),
    ({"belege": [{"foo": 1}]}, "needs a string beleg_id"),
    ({"belege": ["rt-x/9"]}, "beleg_id 'rt-x/9' did not originate in this run"),
    ({"begruendung": ""}, "'begruendung' must be a non-empty string"),
    ({"begruendung": 3}, "'begruendung' must be a non-empty string"),
    ({"gegenargumente": "no"}, "'gegenargumente' must be a list"),
    ({"unsicherheiten": {}}, "'unsicherheiten' must be a list"),
    ({"konfidenz": 1.5}, "'konfidenz' must be a number between 0 and 1"),
    ({"konfidenz": "0.5"}, "'konfidenz' must be a number between 0 and 1"),
    ({"konfidenz": True}, "'konfidenz' must be a number between 0 and 1"),
    ({"datenlage": "complete"}, "'datenlage' must be one of"),
])
def test_type_and_range_violations(over, needle):
    _, errors = validate_final_answer(json.dumps(good(**over)), strict=True, valid_beleg_ids={"rt-x/1"})
    assert len(errors) == 1 and needle in errors[0]


def test_beleg_ids_from_this_run_are_accepted_as_string_or_object():
    d = good(belege=["rt-x/1", {"beleg_id": "rt-x/2", "note": "n"}])
    _, errors = validate_final_answer(json.dumps(d), strict=True, valid_beleg_ids={"rt-x/1", "rt-x/2"})
    assert errors == []


@pytest.mark.parametrize("value", DATENLAGE_VALUES)
def test_all_datenlage_values_are_accepted(value):
    _, errors = validate_final_answer(json.dumps(good(datenlage=value)), strict=True)
    assert errors == []


def test_skill_required_keys_without_strict():
    _, errors = validate_final_answer('{"verdict": "TP"}', required_keys=["verdict", "reason"])
    assert errors == ["missing skill field 'reason'"]
    _, errors = validate_final_answer('{"verdict": "TP", "reason": "r"}', required_keys=["verdict", "reason"])
    assert errors == []


def test_strict_and_skill_keys_combine():
    _, errors = validate_final_answer(json.dumps(good()), strict=True, required_keys=["verdict"])
    assert errors == ["missing skill field 'verdict'"]


# --- loop behaviour ---------------------------------------------------------

def test_without_strict_and_without_skill_keys_free_text_is_accepted():
    loop, llm = make_loop(["FINAL_ANSWER: looks benign"])
    result = loop.run("t")
    assert result["final_answer"] == "looks benign" and result["urteil"] is None
    assert "FINAL_ANSWER Contract" not in llm.system_prompts[0]


def test_strict_prompt_contains_contract_and_observations_carry_schritt_id():
    loop, llm = make_loop(["ACTION: get_container_info", "FINAL_ANSWER: " + json.dumps(good())], strict=True)
    loop.run("t")
    assert FINAL_ANSWER_CONTRACT.strip() in llm.system_prompts[0]
    assert llm.seen[1][-1]["content"].startswith(f"OBSERVATION [schritt_id {loop.run_id}/1]: OK get_container_info")


def test_strict_valid_answer_with_run_evidence_is_accepted_and_parsed():
    loop, _ = make_loop(["ACTION: get_container_info", None], strict=True)
    loop.llm.responses[1] = "FINAL_ANSWER: " + json.dumps(good(belege=[f"{loop.run_id}/1"]))
    result = loop.run("t")
    assert result["steps_taken"] == 2
    assert result["urteil"]["belege"] == [f"{loop.run_id}/1"]
    assert json.loads(result["final_answer"]) == result["urteil"]


def test_strict_invalid_answer_gets_exactly_one_correction_round_then_succeeds():
    loop, llm = make_loop(["FINAL_ANSWER: I think it is fine", "FINAL_ANSWER: " + json.dumps(good())], strict=True)
    result = loop.run("t")
    assert result["steps_taken"] == 2 and result["urteil"]["datenlage"] == "teilweise"
    feedback = llm.seen[1][-1]["content"]
    assert feedback.startswith("OBSERVATION: ERROR: FINAL_ANSWER invalid: FINAL_ANSWER must be a JSON object")
    assert feedback.endswith("Respond again with a corrected FINAL_ANSWER.")
    assert loop.step_log[0]["observation"].startswith("ERROR: FINAL_ANSWER invalid")


def test_strict_invalid_twice_aborts_with_error(ac10=True):
    loop, _ = make_loop(["FINAL_ANSWER: nope", "FINAL_ANSWER: " + json.dumps(good(konfidenz=7))], strict=True)
    result = loop.run("t")
    assert result["final_answer"].startswith("ERROR: FINAL_ANSWER invalid after correction round: 'konfidenz'")
    assert result["steps_taken"] == 2 and result["urteil"] is None


def test_invented_beleg_id_is_rejected_in_strict_mode():
    loop, llm = make_loop(["FINAL_ANSWER: " + json.dumps(good(belege=["rt-fake/1"])), "FINAL_ANSWER: " + json.dumps(good())],
                          strict=True)
    result = loop.run("t")
    assert "did not originate in this run" in llm.seen[1][-1]["content"]
    assert result["urteil"]["belege"] == []


def test_skill_keys_outside_airgap_get_one_correction_round():
    loop, llm = make_loop(['FINAL_ANSWER: {"verdict": "TP"}', 'FINAL_ANSWER: {"verdict": "TP", "reason": "r"}'],
                          required_keys=["verdict", "reason"])
    result = loop.run("t")
    assert result["urteil"] == {"verdict": "TP", "reason": "r"}
    assert "missing skill field 'reason'" in llm.seen[1][-1]["content"]
    assert "FINAL_ANSWER Contract" not in llm.system_prompts[0]


def test_correction_round_counts_against_max_steps():
    loop, _ = make_loop(["FINAL_ANSWER: nope"], strict=True, max_steps=1)
    result = loop.run("t")
    assert result["final_answer"].startswith("Max steps (1) reached")
