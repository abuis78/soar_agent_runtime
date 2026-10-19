"""
soar_agent_runtime_audit.py
Audit sink for SOAR Agent Runtime (RT-P02, RT-P03).

One audit event per ReAct step. The primary sink writes a SOAR artifact
(label ``agent_audit``) to the case synchronously via
``BaseConnector.save_artifact`` - no queue, no daemon thread, so nothing can
be lost at process end (finding B12). ``write()`` reports success/failure so
the loop can act fail-closed for steps that have an effect.

Field names on the artifact follow the specification (German wire names,
decision E7); Python identifiers are English.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from soar_agent_runtime_utils import _scrub

AUDIT_LABEL = "agent_audit"
PHASE_INTENDED = "beabsichtigt"
PHASE_DONE = "erledigt"
PHASE_ERROR = "fehler"

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|token|secret|api_key|apikey|authorization|credential|cookie)",
    re.IGNORECASE,
)
_MASK = "***"


def mask_params(value):
    """Return a copy of ``value`` with sensitive keys masked and secrets scrubbed.

    Keys matching the sensitive pattern are replaced by ``***`` whatever their
    value; every remaining string runs through utils._scrub. Dicts and lists
    are handled recursively; other scalars pass through unchanged.
    """
    if isinstance(value, dict):
        return {
            k: (_MASK if _SENSITIVE_KEY_RE.search(str(k)) else mask_params(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask_params(v) for v in value]
    if isinstance(value, str):
        return _scrub(value)
    return value


def sha256_text(text: str | None) -> str:
    """Deterministic SHA-256 hex of a text (empty string for None)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def definition_version(agent_def: dict | None) -> str:
    """Version of the agent definition until Git provides one (5.x): hash of its JSON."""
    return sha256_text(json.dumps(agent_def or {}, sort_keys=True, default=str))


def build_step_event(
    *,
    run_id: str,
    step_id: str,
    agent_id: str,
    phase: str,
    thought: str = "",
    tool: str = "",
    params: dict | None = None,
    observation: str | None = None,
    call_id: str = "",
    decision_id: str = "",
    def_version: str = "",
) -> dict:
    """Build the CEF payload of one audit event (spec names, RT-P02)."""
    return {
        "lauf_id": run_id,
        "schritt_id": step_id,
        "agent_id": agent_id,
        "phase": phase,
        "gedanke": _scrub(thought or ""),
        "werkzeug": tool or "",
        "parameter": json.dumps(mask_params(params or {}), ensure_ascii=False, sort_keys=True),
        "ergebnis_hash": sha256_text(observation) if observation is not None else "",
        "aufruf_id": call_id,
        "entscheidung_id": decision_id,
        "definition_version": def_version,
        "zeit": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


class AuditSink:
    """Interface. ``write`` returns True only when the event is durably stored."""

    fail_closed: bool = False

    def write(self, event: dict) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def flush(self) -> bool:
        return True


class NullAuditSink(AuditSink):
    """Audit disabled (only allowed outside airgap_mode). Never blocks."""

    fail_closed = False

    def write(self, event: dict) -> bool:
        return True


class ArtifactAuditSink(AuditSink):
    """Writes one SOAR artifact per event to the case, synchronously."""

    def __init__(self, connector, container_id: int, fail_closed: bool) -> None:
        self.connector = connector
        self.container_id = int(container_id)
        self.fail_closed = bool(fail_closed)
        self.written = 0
        self.failed = 0
        self.last_error = ""

    def write(self, event: dict) -> bool:
        artifact = {
            "container_id": self.container_id,
            "label": AUDIT_LABEL,
            "name": f"{event.get('schritt_id', '')} {event.get('werkzeug') or event.get('phase', '')}".strip(),
            "type": "agent_audit",
            "source_data_identifier": f"{event.get('schritt_id', '')}:{event.get('phase', '')}",
            "cef": dict(event),
            "run_automation": False,
        }
        try:
            status, message, _artifact_id = self.connector.save_artifact(artifact)
        except Exception as e:  # never let the sink raise into the loop
            self.failed += 1
            self.last_error = _scrub(f"{type(e).__name__}: {e}")
            return False
        if status:
            self.written += 1
            return True
        self.failed += 1
        self.last_error = _scrub(str(message or "save_artifact returned APP_ERROR"))
        return False

    def flush(self) -> bool:
        # Synchronous sink: nothing is queued, so nothing can be lost (B12).
        return True
