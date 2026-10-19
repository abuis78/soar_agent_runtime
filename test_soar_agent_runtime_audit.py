"""Audit sink tests for RT-P02/RT-P03 (Umsetzungskonzept Runtime, sub-change 1.7)."""

from __future__ import annotations

import json

import pytest

from soar_agent_runtime_audit import (
    AUDIT_LABEL,
    PHASE_DONE,
    PHASE_INTENDED,
    ArtifactAuditSink,
    NullAuditSink,
    build_step_event,
    definition_version,
    mask_params,
    sha256_text,
)


class FakeConnector:
    def __init__(self, status=True, message="ok", raise_exc=None):
        self.status, self.message, self.raise_exc = status, message, raise_exc
        self.saved: list[dict] = []

    def save_artifact(self, artifact):
        if self.raise_exc:
            raise self.raise_exc
        self.saved.append(artifact)
        return self.status, self.message, 4711


# --- masking ----------------------------------------------------------------

def test_sensitive_keys_are_masked_whatever_the_value():
    out = mask_params({"api_key": "abc", "Password": 123, "Authorization": None, "title": "t"})
    assert out == {"api_key": "***", "Password": "***", "Authorization": "***", "title": "t"}


def test_embedded_secrets_in_values_are_scrubbed():
    out = mask_params({"query": "index=main sk-abcdefghijklmnop", "hdr": "Bearer abcdefgh.ijkl"})
    assert "sk-abcdefghijklmnop" not in out["query"]
    assert "[REDACTED]" in out["query"]
    assert "[REDACTED]" in out["hdr"]


def test_masking_is_recursive_and_keeps_scalars():
    out = mask_params({"a": [{"token": "x", "n": 1}, "plain"], "b": {"c": {"secret_key": "y"}}, "d": 2.5})
    assert out == {"a": [{"token": "***", "n": 1}, "plain"], "b": {"c": {"secret_key": "***"}}, "d": 2.5}


def test_masking_does_not_mutate_input():
    src = {"token": "x", "inner": {"password": "y"}}
    mask_params(src)
    assert src == {"token": "x", "inner": {"password": "y"}}


# --- event ------------------------------------------------------------------

def test_hashes_are_deterministic():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")
    assert sha256_text(None) == sha256_text("")
    assert definition_version({"b": 1, "a": 2}) == definition_version({"a": 2, "b": 1})


def test_build_step_event_uses_spec_names_and_masks_params():
    ev = build_step_event(
        run_id="rt-1", step_id="rt-1/3", agent_id="triage", phase=PHASE_DONE,
        thought="look sk-abcdefghijklmnop", tool="add_note",
        params={"title": "t", "api_key": "k"}, observation="OK", def_version="v",
    )
    assert set(ev) == {"lauf_id", "schritt_id", "agent_id", "phase", "gedanke", "werkzeug", "parameter",
                       "ergebnis_hash", "aufruf_id", "entscheidung_id", "definition_version", "zeit"}
    assert (ev["lauf_id"], ev["schritt_id"], ev["agent_id"], ev["phase"]) == ("rt-1", "rt-1/3", "triage", "erledigt")
    assert "sk-abcdefghijklmnop" not in ev["gedanke"]
    assert json.loads(ev["parameter"]) == {"api_key": "***", "title": "t"}
    assert ev["ergebnis_hash"] == sha256_text("OK")
    assert ev["definition_version"] == "v"
    assert ev["zeit"].endswith("Z")


def test_intended_event_has_no_result_hash():
    ev = build_step_event(run_id="r", step_id="r/1", agent_id="a", phase=PHASE_INTENDED, tool="update_severity")
    assert ev["ergebnis_hash"] == ""
    assert ev["aufruf_id"] == "" and ev["entscheidung_id"] == ""


# --- artifact sink ----------------------------------------------------------

def test_artifact_sink_writes_one_artifact_per_event_without_automation():
    c = FakeConnector()
    sink = ArtifactAuditSink(c, container_id="42", fail_closed=True)
    ev = build_step_event(run_id="r", step_id="r/1", agent_id="a", phase=PHASE_DONE, tool="add_note")
    assert sink.write(ev) is True
    assert len(c.saved) == 1
    art = c.saved[0]
    assert art["container_id"] == 42
    assert art["label"] == AUDIT_LABEL == "agent_audit"
    assert art["run_automation"] is False
    assert art["name"] == "r/1 add_note"
    assert art["source_data_identifier"] == "r/1:erledigt"
    assert art["cef"] == ev
    assert (sink.written, sink.failed, sink.fail_closed) == (1, 0, True)


def test_artifact_sink_reports_false_on_app_error():
    c = FakeConnector(status=False, message="db locked sk-abcdefghijklmnop")
    sink = ArtifactAuditSink(c, 1, fail_closed=False)
    assert sink.write({"schritt_id": "r/1", "phase": "x"}) is False
    assert sink.failed == 1
    assert "db locked" in sink.last_error and "sk-abcdefghijklmnop" not in sink.last_error


def test_artifact_sink_reports_false_on_exception_and_never_raises():
    c = FakeConnector(raise_exc=RuntimeError("boom"))
    sink = ArtifactAuditSink(c, 1, fail_closed=True)
    assert sink.write({"schritt_id": "r/1", "phase": "x"}) is False
    assert sink.last_error == "RuntimeError: boom"


def test_sinks_flush_synchronously():
    assert ArtifactAuditSink(FakeConnector(), 1, True).flush() is True
    assert NullAuditSink().flush() is True


def test_null_sink_writes_nothing_and_never_blocks():
    sink = NullAuditSink()
    assert sink.write({"x": 1}) is True
    assert sink.fail_closed is False


@pytest.mark.parametrize("fail_closed", [True, False])
def test_fail_closed_flag_is_carried(fail_closed):
    assert ArtifactAuditSink(FakeConnector(), 1, fail_closed).fail_closed is fail_closed
