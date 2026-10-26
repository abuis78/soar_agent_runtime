"""ToolRegistry + ReActLoop with a Tool Broker (Umsetzungskonzept Runtime, sub-change 2.3).

RT-L01 broker_invoke with run context, RT-L03 prompt tools from /v1/tools,
RT-L04 aufruf_id as evidence, RT-L05 rejection codes / HALTED ends the run,
RT-L06 data gap instead of fallback, RT-P06 aufruf_id accepted as beleg.
"""

from __future__ import annotations

import json
import types

import pytest

from soar_agent_runtime_broker import InvokeResult, ToolSpec, ToolsResponse
from soar_agent_runtime_utils import ReActLoop, ToolRegistry

CVE = ToolSpec("cve_offen_auf_assets", 3, "Offene Assets zu einer CVE",
               {"cve": {"typ": "string", "muster": "^CVE-\\d{4}-\\d{4,7}$", "pflicht": True},
                "zeitfenster": {"typ": "relzeit", "standard": "-7d", "max": "-30d", "pflicht": False}})
ASSET = ToolSpec("asset_kontext", 1, "Asset-Stammdaten", {})
OK = InvokeResult(ok=True, aufruf_id="tb-1", vorlage_id="cve_offen_auf_assets", vorlage_version=3,
                  katalog_version="a41c9e2", zeilen=1, gekuerzt=False, ergebnis_hash="sha256:x",
                  daten=[{"asset": "srv-1", "status": "offen"}])


class FakeBroker:
    def __init__(self, tools=None, invoke=OK, raise_on_list=None):
        self.tools = tools if tools is not None else ToolsResponse(ok=True, katalog_version="a41c9e2", vorlagen=(CVE, ASSET))
        self.invoke_result = invoke
        self.raise_on_list = raise_on_list
        self.list_calls: list[str] = []
        self.invoke_calls: list[tuple] = []

    def list_tools(self, agent_id):
        self.list_calls.append(agent_id)
        if self.raise_on_list:
            raise self.raise_on_list
        return self.tools

    def invoke(self, vorlage_id, parameter, kontext):
        self.invoke_calls.append((vorlage_id, parameter, kontext))
        return self.invoke_result


def registry(broker=None, allowed=("add_note",), **kw):
    progress = []
    connector = types.SimpleNamespace(save_progress=lambda m, *a, **k: progress.append(str(m)))
    reg = ToolRegistry(connector, 4711, list(allowed), agent_id="triage", broker=broker, run_id="rt-x", **kw)
    reg._progress = progress
    return reg


# --- tool model -------------------------------------------------------------

def test_every_tool_declares_quelle_and_broker_invoke_is_a_read_tool():
    for name, defn in ToolRegistry.TOOL_DEFINITIONS.items():
        assert defn["quelle"] in ("builtin", "broker", "katalog"), name
    b = ToolRegistry.TOOL_DEFINITIONS["broker_invoke"]
    assert b["quelle"] == "broker" and b["wirkt"] is False
    assert ToolRegistry.has_effect("broker_invoke") is False


# --- without broker: unchanged ----------------------------------------------

def test_without_broker_nothing_changes():
    reg = registry()
    assert reg.load_broker_tools() is False
    assert reg.get_tool_descriptions() == "- add_note(title: Note title, content: Note content (markdown supported)): Add a note to the current SOAR container"
    assert reg.execute("broker_invoke", {"vorlage_id": "x"}) == "ERROR: Tool 'broker_invoke' is not in the allowed tools list."
    assert reg.evidence_ids() == set() and reg.halted is False


def test_broker_invoke_in_allowed_tools_without_broker_is_not_described_statically():
    reg = registry(allowed=("add_note", "broker_invoke"))
    assert "broker_invoke" not in reg.get_tool_descriptions()


# --- load_broker_tools / descriptions (RT-L03) ------------------------------

def test_templates_are_loaded_once_per_run_and_described_with_parameters():
    fb = FakeBroker()
    reg = registry(fb)
    assert reg.load_broker_tools() is True
    assert fb.list_calls == ["triage"]
    assert reg.broker_status == "ok" and reg.broker_katalog_version == "a41c9e2"
    desc = reg.get_tool_descriptions()
    assert desc.startswith("- add_note(")
    assert "- broker_invoke(vorlage_id: template id from the list below, parameter: JSON object with the template parameters)" in desc
    assert "Broker templates (catalog a41c9e2):" in desc
    assert "  - cve_offen_auf_assets v3: Offene Assets zu einer CVE. Parameters: cve (string, muster ^CVE-\\d{4}-\\d{4,7}$, pflicht); zeitfenster (relzeit, standard -7d, max -30d, optional)" in desc
    assert "  - asset_kontext v1: Asset-Stammdaten. Parameters: no parameters" in desc
    assert any("2 template(s) available" in m for m in reg._progress)


@pytest.mark.parametrize("answer,code", [
    (ToolsResponse(ok=False, code="UNREACHABLE", detail="refused"), "UNREACHABLE"),
    (ToolsResponse(ok=False, code="CLIENT_NOT_ALLOWED", detail="cert"), "CLIENT_NOT_ALLOWED"),
])
def test_unavailable_broker_puts_data_gap_note_in_prompt(answer, code):
    reg = registry(FakeBroker(tools=answer))
    assert reg.load_broker_tools() is False
    assert reg.broker_status == code and reg.broker_tools == {}
    desc = reg.get_tool_descriptions()
    assert f"- Tool Broker: unavailable ({code})" in desc and "datenlage 'unbekannt' or 'blinder_fleck'" in desc
    assert reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets"}).startswith("ERROR: Tool 'broker_invoke' is not in the allowed tools list.")


def test_list_tools_exception_is_contained():
    reg = registry(FakeBroker(raise_on_list=RuntimeError("boom sk-abcdefghijklmnop")))
    assert reg.load_broker_tools() is False
    assert reg.broker_status == "UNREACHABLE" and "sk-abcdefghijklmnop" not in reg.broker_detail


def test_halted_catalogue_marks_registry_halted():
    reg = registry(FakeBroker(tools=ToolsResponse(ok=True, katalog_version="v", angehalten=True)))
    assert reg.load_broker_tools() is False
    assert reg.halted is True and reg.broker_status == "halted"
    assert "- Tool Broker: HALTED." in reg.get_tool_descriptions()


# --- broker_invoke (RT-L01, L04, L05, L06) ----------------------------------

def test_unknown_template_is_rejected_locally_without_broker_call():
    fb = FakeBroker()
    reg = registry(fb)
    reg.load_broker_tools()
    obs = reg.execute("broker_invoke", {"vorlage_id": "nope", "parameter": {}})
    assert obs == "ERROR: unknown template 'nope'. Available templates: asset_kontext, cve_offen_auf_assets"
    assert fb.invoke_calls == []
    assert reg.execute("broker_invoke", {"vorlage_id": "asset_kontext", "parameter": "x"}) == "ERROR: 'parameter' must be a JSON object."
    assert fb.invoke_calls == []


def test_successful_invoke_sends_context_registers_evidence_and_formats_observation():
    fb = FakeBroker()
    reg = registry(fb)
    reg.load_broker_tools()
    reg.set_step(3, "rt-x/3")
    obs = reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-1"}})
    assert fb.invoke_calls == [("cve_offen_auf_assets", {"cve": "CVE-2026-1"},
                                {"agent_id": "triage", "lauf_id": "rt-x", "container_id": 4711, "schritt": 3, "schritt_id": "rt-x/3"})]
    head, hint, body = obs.split("\n", 2)
    assert head == ("BROKER RESULT aufruf_id=tb-1 vorlage=cve_offen_auf_assets v3 katalog=a41c9e2 zeilen=1 "
                    "gekuerzt=false datenlage=nicht gemeldet")
    assert hint == ToolRegistry.DATENLAGE_HINTS["unbekannt"]     # an older broker without datenlage (TB-F18)
    assert json.loads(body) == [{"asset": "srv-1", "status": "offen"}]
    assert reg.evidence_ids() == {"tb-1"} and reg.last_call_id == "tb-1"


def test_missing_parameter_defaults_to_empty_object():
    fb = FakeBroker()
    reg = registry(fb)
    reg.load_broker_tools()
    reg.execute("broker_invoke", {"vorlage_id": "asset_kontext"})
    assert fb.invoke_calls[0][1] == {}


def test_rejection_is_an_observation_with_code():
    rej = InvokeResult(ok=False, aufruf_id="tb-9", art="abgelehnt", code="INVALID_PARAM", detail="cve pattern")
    reg = registry(FakeBroker(invoke=rej))
    reg.load_broker_tools()
    obs = reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "x"}})
    assert obs == "BROKER REJECTED code=INVALID_PARAM aufruf_id=tb-9 detail=cve pattern"
    assert reg.evidence_ids() == set() and reg.last_call_id == "tb-9" and reg.halted is False


@pytest.mark.parametrize("code", ["UNREACHABLE", "TIMEOUT", "BAD_RESPONSE", "SOURCE_ERROR", "CATALOG_UNAVAILABLE", "AUDIT_UNAVAILABLE"])
def test_data_gap_codes_tell_the_model_there_is_no_other_path(code):
    reg = registry(FakeBroker(invoke=InvokeResult(ok=False, art="fehler", code=code, detail="d")))
    reg.load_broker_tools()
    obs = reg.execute("broker_invoke", {"vorlage_id": "asset_kontext"})
    assert obs.startswith(f"BROKER UNAVAILABLE code={code} detail=d.")
    assert "datenlage 'unbekannt' or 'blinder_fleck'" in obs and "no other access path" in obs


def test_halted_invoke_sets_halted_flag():
    reg = registry(FakeBroker(invoke=InvokeResult(ok=False, aufruf_id="tb-h", art="abgelehnt", code="HALTED", detail="stop")))
    reg.load_broker_tools()
    obs = reg.execute("broker_invoke", {"vorlage_id": "asset_kontext"})
    assert obs == "BROKER HALTED code=HALTED aufruf_id=tb-h detail=stop. Run stops."
    assert reg.halted is True


# --- loop integration -------------------------------------------------------

class FakeLLM:
    provider, model = "local", "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []
        self.system_prompts = []

    def chat(self, system_prompt, messages):
        self.system_prompts.append(system_prompt)
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


class FakeSink:
    fail_closed = False

    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)
        return True

    def flush(self):
        return True


def make_loop(responses, broker, strict=False):
    reg = registry(broker)
    reg.load_broker_tools()
    llm = FakeLLM(responses)
    sink = FakeSink()
    loop = ReActLoop(connector=reg.connector, llm=llm, tools=reg, agent_id="triage", system_prompt="x",
                     max_steps=5, container_id=4711, run_id="rt-x", audit=sink, strict_schema=strict)
    return loop, llm, reg, sink


def good(belege):
    return json.dumps({"belege": belege, "begruendung": "b", "gegenargumente": [], "unsicherheiten": [],
                       "konfidenz": 0.5, "datenlage": "teilweise"})


def test_loop_passes_step_context_and_accepts_aufruf_id_as_beleg():
    fb = FakeBroker()
    loop, llm, reg, sink = make_loop([
        'ACTION: broker_invoke\nPARAMS: {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-1"}}',
        "FINAL_ANSWER: " + good(["tb-1", "rt-x/1"]),
    ], fb, strict=True)
    result = loop.run("t")
    assert fb.invoke_calls[0][2]["schritt"] == 1 and fb.invoke_calls[0][2]["schritt_id"] == "rt-x/1"
    assert result["urteil"]["belege"] == ["tb-1", "rt-x/1"]
    assert "Broker templates (catalog a41c9e2)" in llm.system_prompts[0]
    assert sink.events[0]["aufruf_id"] == "tb-1" and sink.events[0]["phase"] == "erledigt"


def test_loop_rejects_invented_aufruf_id():
    loop, llm, _, _ = make_loop(["FINAL_ANSWER: " + good(["tb-forged"]), "FINAL_ANSWER: " + good([])], FakeBroker(), strict=True)
    result = loop.run("t")
    assert "beleg_id 'tb-forged' did not originate in this run" in llm.seen[1][-1]["content"]
    assert result["urteil"]["belege"] == []


def test_halted_broker_answer_ends_the_run_before_the_next_step():
    fb = FakeBroker(invoke=InvokeResult(ok=False, aufruf_id="tb-h", code="HALTED", art="abgelehnt", detail="stop"))
    loop, llm, reg, sink = make_loop([
        'ACTION: broker_invoke\nPARAMS: {"vorlage_id": "asset_kontext"}',
        "FINAL_ANSWER: never reached",
    ], fb)
    result = loop.run("t")
    assert result["final_answer"] == "ERROR: run halted by Tool Broker (HALTED)"
    assert result["steps_taken"] == 1 and len(llm.seen) == 1
    assert sink.events[0]["phase"] == "fehler" and sink.events[0]["aufruf_id"] == "tb-h"
    assert loop.step_log[0]["observation"].startswith("BROKER HALTED")


def test_data_gap_lets_the_run_continue_with_a_gap_observation():
    fb = FakeBroker(invoke=InvokeResult(ok=False, art="fehler", code="UNREACHABLE", detail="refused"))
    loop, llm, _, _ = make_loop([
        'ACTION: broker_invoke\nPARAMS: {"vorlage_id": "asset_kontext"}',
        "FINAL_ANSWER: " + good([]),
    ], fb, strict=True)
    result = loop.run("t")
    assert "BROKER UNAVAILABLE code=UNREACHABLE" in llm.seen[1][-1]["content"]
    assert result["urteil"]["datenlage"] == "teilweise" and result["steps_taken"] == 2


# --- splunk_search off in airgap, deprecated outside (RT-L02, sub-change 2.4) -

def test_airgap_removes_splunk_search_from_prompt_and_execution():
    reg = registry(allowed=("splunk_search", "add_note"), airgap=True)
    assert reg.allowed_tools == ["add_note"]
    assert "splunk_search" not in reg.get_tool_descriptions()
    obs = reg.execute("splunk_search", {"spl_query": "index=main"})
    assert obs == "ERROR: Tool 'splunk_search' is disabled in airgap_mode; read data via broker_invoke templates."
    assert any("[airgap] 'splunk_search' removed from allowed tools" in m for m in reg._progress)


def test_airgap_prompt_with_broker_has_templates_but_no_spl_tool():
    reg = registry(FakeBroker(), allowed=("splunk_search", "add_note"), airgap=True)
    reg.load_broker_tools()
    desc = reg.get_tool_descriptions()
    assert "splunk_search" not in desc and "broker_invoke" in desc and "cve_offen_auf_assets" in desc


def test_outside_airgap_splunk_search_says_it_is_unavailable(monkeypatch):
    # Option 1 (1.11.1): SOAR has no run_query for apps, so the tool never searches.
    import sys
    import types as _t
    rules = _t.ModuleType("phantom.rules")
    rules.run_query = lambda **kw: (_ for _ in ()).throw(AssertionError("no SPL may run"))
    monkeypatch.setitem(sys.modules, "phantom.rules", rules)
    reg = registry(allowed=("splunk_search", "add_note"))
    assert "(not available in SOAR apps: always returns an error; use broker_invoke templates)" in \
        reg.get_tool_descriptions()
    assert reg.execute("splunk_search", {"spl_query": "index=main"}) == (
        "ERROR: splunk_search is not available in SOAR apps (SOAR has no search API for apps); "
        "nothing was searched. Read data via broker_invoke templates.")


# --- get_container_info: own case only, size-capped (RT-L08, sub-change 2.5) --

@pytest.fixture
def case_rules(monkeypatch):
    """The case over /rest/container (1.10.1): 7 artifacts, the server returns a page of 5."""
    import soar_agent_runtime_utils as utils
    calls = []

    def fake(method, url, body, timeout_s):
        path = url.split("/rest/", 1)[1].split("?", 1)[0]
        cid = int(path.split("/")[1])
        if path.endswith("/artifacts"):
            return 200, {"count": 7, "data": [{"id": i, "cef": {"n": i}} for i in range(5)]}
        calls.append(cid)
        return 200, {"id": cid, "name": "case", "blob": "X" * 300}

    monkeypatch.setattr(utils.SoarRest, "transport", staticmethod(fake))
    return calls


@pytest.mark.parametrize("params", [{}, {"container_id": 4711}, {"container_id": "4711"}, {"case_id": 4711}])
def test_get_container_info_reads_own_case(case_rules, params):
    reg = registry(allowed=("get_container_info",))
    data = json.loads(reg.execute("get_container_info", params))
    assert case_rules == [4711]
    assert data["container"]["id"] == 4711
    assert (data["artifact_count"], data["artifacts_shown"], len(data["artifacts"])) == (5, 5, 5)
    assert data["artifacts_total"] == 7 and data["runtime_artifacts_hidden"] == 0


@pytest.mark.parametrize("params", [{"container_id": 4712}, {"container_id": "99"}, {"case_id": 1}])
def test_get_container_info_refuses_other_cases_without_platform_access(case_rules, params):
    reg = registry(allowed=("get_container_info",))
    obs = reg.execute("get_container_info", params)
    assert obs == ("ERROR: get_container_info reads only the container of this run (4711); "
                   "other cases are read via broker_invoke templates.")
    assert case_rules == []


def test_get_container_info_description_states_the_limits():
    desc = registry(allowed=("get_container_info",)).get_tool_descriptions()
    assert "this run's case only; max 5 artifacts, output is size-capped" in desc


def test_large_container_payload_is_truncated_at_the_loop_boundary(case_rules):
    reg = registry(allowed=("get_container_info",))
    llm = FakeLLM(["ACTION: get_container_info", "FINAL_ANSWER: ok"])
    loop = ReActLoop(connector=reg.connector, llm=llm, tools=reg, agent_id="t", system_prompt="x",
                     max_steps=3, container_id=4711, max_observation_chars=120)
    loop.run("t")
    obs = llm.seen[1][-1]["content"]
    assert "[...truncated " in obs and obs.endswith("<<<END DATA>>>")
    assert len(loop.step_log[0]["observation"]) > 120  # raw kept for forensics


# --- E-12 (1.16.2 / 2.0.0): the broker's datenlage reaches the model ------------------------

@pytest.mark.parametrize("datenlage,expected", [
    ("vorhanden", "0 rows here is a real negative finding"),
    ("leer", "datenlage 'blinder_fleck'"),
    ("unbekannt", "datenlage 'unbekannt'"),
])
def test_broker_datenlage_is_in_the_observation_with_its_meaning(datenlage, expected):
    ok = InvokeResult(ok=True, aufruf_id="tb-5", vorlage_id="cve_offen_auf_assets", vorlage_version=3,
                      katalog_version="k7", zeilen=0, daten=[], datenlage=datenlage)
    reg = registry(FakeBroker(invoke=ok))
    reg.load_broker_tools()
    obs = reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-5678"}})
    head, hint, _ = obs.split("\n", 2)
    assert head.endswith(f"zeilen=0 gekuerzt=false datenlage={datenlage}")
    assert expected in hint


# --- E-13 (1.16.3): reads without data are recorded as data gaps (D16) ----------------------

@pytest.mark.parametrize("datenlage,gap", [("vorhanden", None), ("leer", "cve_offen_auf_assets: datenlage=leer"),
                                           ("unbekannt", "cve_offen_auf_assets: datenlage=unbekannt"),
                                           ("", "cve_offen_auf_assets: datenlage=nicht gemeldet")])
def test_broker_reads_without_data_are_recorded_as_gaps(datenlage, gap):
    ok = InvokeResult(ok=True, aufruf_id="tb-6", vorlage_id="cve_offen_auf_assets", vorlage_version=3,
                      zeilen=0, daten=[], datenlage=datenlage)
    reg = registry(FakeBroker(invoke=ok))
    reg.load_broker_tools()
    reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-5678"}})
    assert reg.data_gaps == ([gap] if gap else [])


def test_unreachable_source_is_recorded_as_gap():
    gone = InvokeResult(ok=False, vorlage_id="cve_offen_auf_assets", art="fehler", code="SOURCE_ERROR", detail="down")
    reg = registry(FakeBroker(invoke=gone))
    reg.load_broker_tools()
    reg.execute("broker_invoke", {"vorlage_id": "cve_offen_auf_assets", "parameter": {"cve": "CVE-2026-1"}})
    assert reg.data_gaps == ["cve_offen_auf_assets: SOURCE_ERROR"]


def test_vorhanden_hint_names_when_vollstaendig_is_right():
    assert "'vollstaendig' only if every source" in ToolRegistry.DATENLAGE_HINTS["vorhanden"]
