"""
soar_agent_runtime_policy.py
Runtime-side interface to the Policy Manager (RT-W01, W03, W06, W08; sub-change 3.0).

Every tool that has an effect (``wirkt``) asks the Policy Manager before it
runs. This module defines what the runtime sends and what it gets back, and a
small gate interface that the real ``policy_client`` package (PolicyManager
repo) and the test stub both satisfy. The wire format mirrors PolicyManager
``docs/API.md`` (German field names, decision G1); Python names are English (E7).

Fail-closed by construction: nothing here raises into the ReAct loop. A
missing package, a missing ``policy_mode``, a malformed answer or any
exception becomes a ``DENY`` decision with a code (RT-W03), and an unknown
halt state counts as halted.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

GATE_ALLOW = "ALLOW"
GATE_APPROVE_1CLICK = "APPROVE_1CLICK"
GATE_APPROVE_2PERSON = "APPROVE_2PERSON"
GATE_DENY = "DENY"
GATES = (GATE_ALLOW, GATE_APPROVE_1CLICK, GATE_APPROVE_2PERSON, GATE_DENY)
APPROVAL_GATES = (GATE_APPROVE_1CLICK, GATE_APPROVE_2PERSON)

POLICY_MODE_EMBEDDED = "embedded"
POLICY_MODE_SERVICE = "service"
POLICY_MODES = (POLICY_MODE_EMBEDDED, POLICY_MODE_SERVICE)

# aktion.art of a sequence of catalogue building blocks (RT-W09, PM P4): a single
# block is a sequence with one step; the Policy Manager decides it as one request.
ACTION_SEQUENCE = "folge"
STAGES = range(0, 5)

# Runtime-side codes. Rule codes (NO_EVIDENCE, HALTED, ...) come from the
# Policy Manager and are passed through unchanged.
CODE_NO_POLICY_MODE = "NO_POLICY_MODE"
CODE_UNAVAILABLE = "UNAVAILABLE"
CODE_INVALID_RESPONSE = "INVALID_RESPONSE"
CODE_HALTED = "HALTED"

PRINCIPAL_AGENT = "agent"
AUDIT_CLIENT = "soar_agent_runtime"


def _opt_int(value: Any) -> Optional[int]:
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> Optional[float]:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v) for v in value if v is not None)
    return ()


# -- Request --------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRequest:
    """What the runtime sends to ``decide`` (RT-W08). Wire names in to_wire()."""
    kind: str                                   # aktion.art, e.g. update_severity
    category: str = ""                          # aktion.kategorie
    playbook: str = ""                          # aktion.playbook (catalogue blocks, Phase 4)
    targets: tuple[str, ...] = ()               # aktion.ziel
    params: dict = field(default_factory=dict)  # aktion.parameter
    principal_type: str = PRINCIPAL_AGENT       # kontext.principal_type
    initiator: str = ""                         # agent_id; the client identity is the mTLS CN
    agent_id: str = ""
    run_id: str = ""                            # lauf_id
    step_id: str = ""                           # schritt_id
    stage: Optional[int] = None                 # stufe, from the triggering playbook, else None
    domain: str = ""                            # domaene
    tipping_off: str = ""                       # laut | leise
    confidence: Optional[float] = None          # agent_konfidenz
    evidence_ids: tuple[str, ...] = ()          # beleg_ids (checked locally before decide, OQ-8)
    judgement: dict = field(default_factory=dict)  # urteil: begruendung, gegenargumente, ...
    artifact_values: tuple[str, ...] = ()       # artefakt_werte
    steps: tuple[dict, ...] = ()                # aktion.schritte, only for kind == ACTION_SEQUENCE

    def to_wire(self) -> dict:
        action = {"art": self.kind, "kategorie": self.category, "playbook": self.playbook,
                  "ziel": list(self.targets), "parameter": dict(self.params)}
        if self.steps:
            action["schritte"] = [sequence_step(**s) if "block" in s else dict(s) for s in self.steps]
        context: dict[str, Any] = {
            "principal_type": self.principal_type, "initiator": self.initiator,
            "agent_id": self.agent_id, "lauf_id": self.run_id, "schritt_id": self.step_id,
            "domaene": self.domain, "tipping_off": self.tipping_off,
            "beleg_ids": list(self.evidence_ids), "urteil": dict(self.judgement),
            "artefakt_werte": list(self.artifact_values),
        }
        if self.stage is not None:
            context["stufe"] = self.stage
        if self.confidence is not None:
            context["agent_konfidenz"] = self.confidence
        # Reversibility is deliberately not sent (RT-W08).
        return {"aktion": action, "kontext": context}


def sequence_step(block: str, targets: Any = (), params: Any = None) -> dict:
    """One step of a Maßnahmenfolge in wire form (aktion.schritte[], PM P4)."""
    return {"baustein": str(block or ""), "ziel": list(_str_tuple(targets)),
            "parameter": dict(params) if isinstance(params, dict) else {}}


# -- Action catalogue (RT-W09) --------------------------------------------------

@dataclass(frozen=True)
class CatalogBlock:
    """A building block released for this agent (GET /v1/aktionen, PM-F12).

    Field names follow the Policy Manager answer; ``from_wire`` drops an entry
    that has no id or an unusable stage or gate, so a malformed catalogue can
    only hide blocks, never show more.
    """
    id: str
    version: int = 0
    purpose: str = ""                  # zweck
    category: str = ""                 # kategorie
    target_types: tuple[str, ...] = ()  # zieltypen
    parameters: dict = field(default_factory=dict)  # parameter (name -> description or spec)
    min_stage: int = 0                 # mindeststufe
    reversible: bool = False           # umkehrbar (effective value, policy already applied)
    undo: str = ""                     # ruecknahme (block id)
    min_gate: str = ""                 # mindest_gate

    @classmethod
    def from_wire(cls, data: Any) -> Optional["CatalogBlock"]:
        if not isinstance(data, dict):
            return None
        block_id = data.get("id")
        if not isinstance(block_id, str) or not block_id.strip():
            return None
        stage = _opt_int(data.get("mindeststufe", 0))
        version = _opt_int(data.get("version", 0))
        gate = str(data.get("mindest_gate") or "")
        if stage is None or stage not in STAGES or version is None or (gate and gate not in GATES):
            return None
        params = data.get("parameter")
        return cls(
            id=block_id.strip(),
            version=version,
            purpose=str(data.get("zweck") or ""),
            category=str(data.get("kategorie") or ""),
            target_types=_str_tuple(data.get("zieltypen")),
            parameters=dict(params) if isinstance(params, dict) else {},
            min_stage=stage,
            reversible=data.get("umkehrbar") is True,
            undo=str(data.get("ruecknahme") or ""),
            min_gate=gate,
        )


def parse_catalog(items: Any) -> list[CatalogBlock]:
    """Blocks from a ``/v1/aktionen`` list; anything unusable is left out, order kept."""
    if not isinstance(items, (list, tuple)):
        return []
    blocks = (CatalogBlock.from_wire(item) for item in items)
    return [b for b in blocks if b is not None]


# -- Results --------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDecision:
    """Answer of ``decide`` (PM-F08). Only ``allowed`` permits execution."""
    gate: str
    decision_id: str = ""        # entscheidung_id
    approval_id: str = ""        # freigabe_id (approval gates)
    approvals_needed: int = 0    # freigaben_noetig
    expires_at: str = ""         # ablauf
    policy_version: str = ""
    reasons: tuple[dict, ...] = ()  # begruendung
    code: str = ""

    @property
    def allowed(self) -> bool:
        return self.gate == GATE_ALLOW and bool(self.decision_id)

    @property
    def needs_approval(self) -> bool:
        return self.gate in APPROVAL_GATES

    @property
    def denied(self) -> bool:
        return not (self.allowed or self.needs_approval)

    @classmethod
    def deny(cls, code: str, reason: str = "", decision_id: str = "") -> "PolicyDecision":
        reasons = ({"regel": "runtime", "grund": reason, "gate": GATE_DENY},) if reason else ()
        return cls(gate=GATE_DENY, decision_id=decision_id, reasons=reasons, code=code)

    @classmethod
    def from_wire(cls, data: Any) -> "PolicyDecision":
        """Parse a wire decision. Never raises: anything malformed is DENY."""
        if not isinstance(data, dict) or "gate" not in data:
            return cls.deny(CODE_INVALID_RESPONSE, "malformed decision")
        gate = str(data.get("gate"))
        if gate not in GATES:
            return cls.deny(CODE_INVALID_RESPONSE, f"unknown gate {gate!r}")
        decision_id = str(data.get("entscheidung_id") or "")
        approval_id = str(data.get("freigabe_id") or "")
        if gate == GATE_ALLOW and not decision_id:
            return cls.deny(CODE_INVALID_RESPONSE, "ALLOW without entscheidung_id")
        reasons = data.get("begruendung")
        reasons = tuple(r for r in reasons if isinstance(r, dict)) if isinstance(reasons, list) else ()
        if gate in APPROVAL_GATES and not approval_id:
            # No approval was opened, so nothing is pending and nothing may run (RT-W06/W07).
            # Keep the package's code and reasons (e.g. NO_INITIATOR); only a silent answer is malformed.
            code = str(data.get("code") or "")
            if code:
                return cls(gate=GATE_DENY, decision_id=decision_id, reasons=reasons, code=code)
            return cls.deny(CODE_INVALID_RESPONSE, f"{gate} without freigabe_id", decision_id)
        return cls(
            gate=gate,
            decision_id=decision_id,
            approval_id=approval_id,
            approvals_needed=_opt_int(data.get("freigaben_noetig")) or 0,
            expires_at=str(data.get("ablauf") or ""),
            policy_version=str(data.get("policy_version") or ""),
            reasons=reasons,
            code=str(data.get("code") or ""),
        )

    def reason_text(self) -> str:
        parts = []
        for r in self.reasons:
            text = str(r.get("grund") or r.get("regel") or "")
            step = _opt_int(r.get("schritt"))
            if text and step is not None:
                text = f"{text} (schritt {step})"   # sequence reasons name their step (PM P4)
            parts.append(text)
        return "; ".join(p for p in parts if p)


@dataclass(frozen=True)
class ApprovalState:
    """State of an approval (GET /v1/approvals/{id})."""
    approval_id: str
    status: str = "expired"   # pending | approved | rejected | expired | consumed
    have: int = 0
    need: int = 0
    code: str = ""

    @classmethod
    def from_wire(cls, data: Any, approval_id: str = "") -> "ApprovalState":
        if not isinstance(data, dict):
            return cls(approval_id=approval_id, status="expired", code=CODE_INVALID_RESPONSE)
        return cls(
            approval_id=str(data.get("freigabe_id") or approval_id),
            status=str(data.get("status") or "expired"),
            have=_opt_int(data.get("erteilt")) or 0,
            need=_opt_int(data.get("noetig")) or 0,
            code=str(data.get("code") or ""),
        )


@dataclass(frozen=True)
class ConsumeState:
    """Outcome of ``consume`` (PM-F23). Only ok=True permits execution."""
    ok: bool
    code: str = ""

    @classmethod
    def from_wire(cls, data: Any) -> "ConsumeState":
        if not isinstance(data, dict) or data.get("gueltig") is not True:
            code = str(data.get("code") or "") if isinstance(data, dict) else ""
            return cls(ok=False, code=code or CODE_INVALID_RESPONSE)
        return cls(ok=True, code=str(data.get("code") or ""))


@dataclass(frozen=True)
class HaltState:
    """Emergency stop (PM-F30/F31). Unknown state means halted."""
    halted: bool
    reason: str = ""

    @classmethod
    def from_wire(cls, data: Any) -> "HaltState":
        if not isinstance(data, dict) or data.get("angehalten") is not False:
            reason = str(data.get("grund") or "") if isinstance(data, dict) else ""
            return cls(halted=True, reason=reason or "unknown halt state")
        return cls(halted=False, reason=str(data.get("grund") or ""))


# -- Gate interface -------------------------------------------------------------

class PolicyGate(Protocol):
    """What ToolRegistry needs from a policy client. Implementations never raise."""
    mode: str

    def decide(self, request: PolicyRequest) -> PolicyDecision: ...
    def approval_status(self, approval_id: str) -> ApprovalState: ...
    def consume(self, approval_id: str, request: PolicyRequest) -> ConsumeState: ...
    def halt_status(self) -> HaltState: ...
    def health(self) -> tuple[bool, str]: ...
    def actions(self, agent_id: str) -> list[CatalogBlock]: ...


class NoPolicyClient:
    """No usable Policy Manager: every effecting action is denied (RT-W03).

    Used when policy_mode is missing or invalid, the package is not installed,
    or the embedded files are not configured. ``reason`` says which.
    """
    mode = ""

    def __init__(self, reason: str, code: str = CODE_NO_POLICY_MODE) -> None:
        self.reason = reason
        self.code = code

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        return PolicyDecision.deny(self.code, self.reason)

    def approval_status(self, approval_id: str) -> ApprovalState:
        return ApprovalState(approval_id=approval_id, status="expired", code=self.code)

    def consume(self, approval_id: str, request: PolicyRequest) -> ConsumeState:
        return ConsumeState(ok=False, code=self.code)

    def halt_status(self) -> HaltState:
        return HaltState(halted=True, reason=self.reason)

    def health(self) -> tuple[bool, str]:
        return False, self.reason

    def actions(self, agent_id: str) -> list[CatalogBlock]:
        return []


class PolicyClientAdapter:
    """Wraps a real ``policy_client.PolicyClient`` behind PolicyGate.

    The bridge is the wire format only (API.md, decision G1): requests go in
    via ``DecisionRequest.from_wire(PolicyRequest.to_wire())``, answers come
    back via ``to_wire()`` and the parsers above. Any exception inside the
    package is a DENY / halted / failed result, never a traceback in the loop.
    """

    def __init__(self, client: Any, request_type: Any, mode: str, policy_path: str = "") -> None:
        self._client = client
        self._request_type = request_type
        self.mode = mode
        self.policy_path = policy_path   # embedded: checked by health(), the package itself only logs

    def _request(self, request: PolicyRequest) -> Any:
        return self._request_type.from_wire(request.to_wire())

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        try:
            return PolicyDecision.from_wire(self._client.decide(self._request(request)).to_wire())
        except Exception as exc:  # fail-closed (RT-W03)
            return PolicyDecision.deny(CODE_UNAVAILABLE, f"policy client failed: {type(exc).__name__}")

    def approval_status(self, approval_id: str) -> ApprovalState:
        try:
            return ApprovalState.from_wire(self._client.approval_status(approval_id).to_wire(), approval_id)
        except Exception as exc:
            return ApprovalState(approval_id=approval_id, status="expired",
                                 code=f"{CODE_UNAVAILABLE}:{type(exc).__name__}")

    def consume(self, approval_id: str, request: PolicyRequest) -> ConsumeState:
        try:
            return ConsumeState.from_wire(self._client.consume(approval_id, self._request(request)).to_wire())
        except Exception as exc:
            return ConsumeState(ok=False, code=f"{CODE_UNAVAILABLE}:{type(exc).__name__}")

    def halt_status(self) -> HaltState:
        try:
            return HaltState.from_wire(self._client.halt_status().to_wire())
        except Exception as exc:
            return HaltState(halted=True, reason=f"policy client failed: {type(exc).__name__}")

    def actions(self, agent_id: str) -> list[CatalogBlock]:
        """Blocks released for the agent (RT-W09). A package without actions()
        (P1 wheel) or any failure means no catalogue, never a traceback."""
        fetch = getattr(self._client, "actions", None)
        if fetch is None:
            return []
        try:
            return parse_catalog(fetch(agent_id))
        except Exception:
            return []

    def health(self) -> tuple[bool, str]:
        problem = self._policy_file_problem()
        if problem:
            return False, f"policy {self.mode}: {problem}"
        halt = self.halt_status()
        if halt.reason.startswith("policy client failed") or halt.reason == "unknown halt state":
            return False, f"policy {self.mode}: {halt.reason}"
        state = f"HALTED ({halt.reason})" if halt.halted else "not halted"
        return True, f"policy {self.mode} healthy, {state}"

    def _policy_file_problem(self) -> str:
        """A policy file that is missing or not a JSON object makes the package deny every
        action via its default gate; say so in health() instead of per action."""
        if not self.policy_path:
            return ""
        try:
            with open(self.policy_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as exc:
            return f"policy file not readable ({type(exc).__name__}): {self.policy_path}"
        except ValueError:
            return f"policy file is not valid JSON: {self.policy_path}"
        if not isinstance(data, dict):
            return f"policy file is not a JSON object: {self.policy_path}"
        return ""


class StubPolicyClient:
    """In-memory PolicyGate for tests and for 3.3-3.6 before the wheel exists.

    ``decision`` is returned by decide(); ``halted`` by halt_status();
    ``raise_errors`` makes every call raise (adapter-style failure tests).
    Every request is recorded in ``requests``.
    """
    mode = POLICY_MODE_EMBEDDED

    def __init__(self, decision: PolicyDecision | None = None, *, halted: bool = False,
                 consume_ok: bool = True, healthy: bool = True,
                 blocks: list[CatalogBlock] | None = None) -> None:
        self.decision = decision or PolicyDecision(gate=GATE_ALLOW, decision_id="pm-stub-1")
        self.halted = halted
        self.consume_ok = consume_ok
        self.healthy = healthy
        self.blocks = list(blocks or [])      # catalogue returned by actions() (4.2)
        self.requests: list[PolicyRequest] = []
        self.consumed: list[str] = []
        self.actions_asked: list[str] = []

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        self.requests.append(request)
        if self.halted:
            return PolicyDecision.deny(CODE_HALTED, "policy manager halted")
        return self.decision

    def approval_status(self, approval_id: str) -> ApprovalState:
        pending = approval_id == self.decision.approval_id
        return ApprovalState(approval_id=approval_id, status="pending" if pending else "expired",
                             need=self.decision.approvals_needed)

    def consume(self, approval_id: str, request: PolicyRequest) -> ConsumeState:
        self.consumed.append(approval_id)
        return ConsumeState(ok=self.consume_ok, code="" if self.consume_ok else "NOT_APPROVED")

    def halt_status(self) -> HaltState:
        return HaltState(halted=self.halted, reason="stub halt" if self.halted else "")

    def health(self) -> tuple[bool, str]:
        return (True, "policy stub healthy") if self.healthy else (False, "policy stub unhealthy")

    def actions(self, agent_id: str) -> list[CatalogBlock]:
        self.actions_asked.append(agent_id)
        return list(self.blocks)


# -- Factory --------------------------------------------------------------------

SERVICE_TIMEOUT_DEFAULT_S = 5.0


def build_policy_client(mode: str, *, policy_path: str = "", approvals_path: str = "",
                        halt_path: str = "", audit_path: str = "",
                        audit_client: str = AUDIT_CLIENT, service_url: str = "",
                        service_timeout_s: float | None = None, client_cert: str = "",
                        client_key: str = "", ca: str = "") -> PolicyGate:
    """Return a PolicyGate for the asset configuration. Never raises.

    ``embedded`` needs the policy file plus approval and halt files (E11).
    ``service`` needs the https URL plus client certificate, key and CA for
    mTLS and uses the package's own client (RT-W01, RT-W04): unreachable or
    slow means DENY, never a fallback to embedded logic. Anything else is
    RT-W03: policy_mode missing or invalid, every effecting action denied.
    """
    mode = (mode or "").strip().lower()
    if mode not in POLICY_MODES:
        return NoPolicyClient(f"policy_mode missing or invalid ({mode!r}); set it to "
                              f"'{POLICY_MODE_EMBEDDED}' or '{POLICY_MODE_SERVICE}' - "
                              "every effecting action is denied (RT-W03)")
    if mode == POLICY_MODE_SERVICE:
        return _build_service_client(service_url, service_timeout_s, client_cert, client_key, ca)
    missing = [name for name, value in (("policy_path", policy_path), ("policy_approvals_path", approvals_path),
                                        ("policy_halt_path", halt_path)) if not (value or "").strip()]
    if missing:
        return NoPolicyClient(f"policy_mode 'embedded' needs {', '.join(missing)}; every effecting action is denied",
                              CODE_NO_POLICY_MODE)
    try:
        from policy_client.base import DecisionRequest  # bundled policy_manager wheel (3.1b)
        from policy_client.embedded import EmbeddedPolicyClient
    except Exception as exc:  # ImportError or a broken package: fail closed
        return NoPolicyClient(f"policy_client package not available ({type(exc).__name__}); "
                              "every effecting action is denied", CODE_UNAVAILABLE)
    kwargs: dict[str, Any] = {"policy_path": policy_path.strip(), "approvals_path": approvals_path.strip(),
                              "halt_path": halt_path.strip()}
    # The audit file arrives with Policy Manager P3; pass it only to a client that takes it,
    # so the same app works with the P1 wheel (no audit kwargs) and later ones.
    if _accepts(EmbeddedPolicyClient, "audit_path"):
        kwargs["audit_path"] = (audit_path or "").strip() or None
        kwargs["audit_client"] = audit_client
    try:
        client = EmbeddedPolicyClient(**kwargs)
    except Exception as exc:
        return NoPolicyClient(f"policy client could not start ({type(exc).__name__})", CODE_UNAVAILABLE)
    return PolicyClientAdapter(client, DecisionRequest, POLICY_MODE_EMBEDDED, policy_path=kwargs["policy_path"])


def _build_service_client(url: str, timeout_s: float | None, cert: str, key: str, ca: str) -> PolicyGate:
    """policy_mode 'service' (4.1): the package's ServicePolicyClient over mTLS."""
    fields = {"policy_service_url": url, "policy_client_cert": cert, "policy_client_key": key, "policy_ca": ca}
    missing = [name for name, value in fields.items() if not (value or "").strip()]
    if missing:
        return NoPolicyClient(f"policy_mode 'service' needs {', '.join(missing)}; every effecting action is denied",
                              CODE_NO_POLICY_MODE)
    url = url.strip()
    if not url.lower().startswith("https://"):
        return NoPolicyClient("policy_service_url must start with https:// (mTLS, RT-W01); "
                              "every effecting action is denied", CODE_NO_POLICY_MODE)
    try:
        from policy_client.base import DecisionRequest
        from policy_client.service import ServicePolicyClient   # Policy Manager P3 or later
    except Exception as exc:  # P1 wheel has no service module: fail closed, say why
        return NoPolicyClient("policy_client.service not in the bundled policy_manager wheel "
                              f"(needs Policy Manager P3, {type(exc).__name__}); every effecting action is denied",
                              CODE_UNAVAILABLE)
    timeout = SERVICE_TIMEOUT_DEFAULT_S if timeout_s is None or timeout_s <= 0 else float(timeout_s)
    try:
        client = ServicePolicyClient(url, cert.strip(), key.strip(), ca.strip(), timeout_s=timeout)
    except Exception as exc:
        return NoPolicyClient(f"policy service client could not start ({type(exc).__name__})", CODE_UNAVAILABLE)
    return PolicyClientAdapter(client, DecisionRequest, POLICY_MODE_SERVICE)


def _accepts(cls: Any, name: str) -> bool:
    """True when cls(...) takes keyword ``name``; False when unknown or not introspectable."""
    try:
        return name in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False
