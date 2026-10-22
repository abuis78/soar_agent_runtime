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

    def to_wire(self) -> dict:
        action = {"art": self.kind, "kategorie": self.category, "playbook": self.playbook,
                  "ziel": list(self.targets), "parameter": dict(self.params)}
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
        if gate in APPROVAL_GATES and not approval_id:
            # The proposal artifact needs the freigabe_id (RT-W06/W07).
            return cls.deny(CODE_INVALID_RESPONSE, f"{gate} without freigabe_id", decision_id)
        reasons = data.get("begruendung")
        return cls(
            gate=gate,
            decision_id=decision_id,
            approval_id=approval_id,
            approvals_needed=_opt_int(data.get("freigaben_noetig")) or 0,
            expires_at=str(data.get("ablauf") or ""),
            policy_version=str(data.get("policy_version") or ""),
            reasons=tuple(r for r in reasons if isinstance(r, dict)) if isinstance(reasons, list) else (),
            code=str(data.get("code") or ""),
        )

    def reason_text(self) -> str:
        parts = [str(r.get("grund") or r.get("regel") or "") for r in self.reasons]
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


class PolicyClientAdapter:
    """Wraps a real ``policy_client.PolicyClient`` behind PolicyGate.

    The bridge is the wire format only (API.md, decision G1): requests go in
    via ``DecisionRequest.from_wire(PolicyRequest.to_wire())``, answers come
    back via ``to_wire()`` and the parsers above. Any exception inside the
    package is a DENY / halted / failed result, never a traceback in the loop.
    """

    def __init__(self, client: Any, request_type: Any, mode: str) -> None:
        self._client = client
        self._request_type = request_type
        self.mode = mode

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

    def health(self) -> tuple[bool, str]:
        halt = self.halt_status()
        if halt.reason.startswith("policy client failed") or halt.reason == "unknown halt state":
            return False, f"policy {self.mode}: {halt.reason}"
        state = f"HALTED ({halt.reason})" if halt.halted else "not halted"
        return True, f"policy {self.mode} healthy, {state}"


class StubPolicyClient:
    """In-memory PolicyGate for tests and for 3.3-3.6 before the wheel exists.

    ``decision`` is returned by decide(); ``halted`` by halt_status();
    ``raise_errors`` makes every call raise (adapter-style failure tests).
    Every request is recorded in ``requests``.
    """
    mode = POLICY_MODE_EMBEDDED

    def __init__(self, decision: PolicyDecision | None = None, *, halted: bool = False,
                 consume_ok: bool = True, healthy: bool = True) -> None:
        self.decision = decision or PolicyDecision(gate=GATE_ALLOW, decision_id="pm-stub-1")
        self.halted = halted
        self.consume_ok = consume_ok
        self.healthy = healthy
        self.requests: list[PolicyRequest] = []
        self.consumed: list[str] = []

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


# -- Factory --------------------------------------------------------------------

def build_policy_client(mode: str, *, policy_path: str = "", approvals_path: str = "",
                        halt_path: str = "", audit_path: str = "",
                        audit_client: str = AUDIT_CLIENT) -> PolicyGate:
    """Return a PolicyGate for the asset configuration. Never raises.

    ``embedded`` needs the policy file plus approval and halt files (E11).
    ``service`` is Phase 4 (RT-W04) and is refused here, so nothing runs
    against a half-built path. Anything else is RT-W03: policy_mode missing
    or invalid, every effecting action denied.
    """
    mode = (mode or "").strip().lower()
    if mode not in POLICY_MODES:
        return NoPolicyClient(f"policy_mode missing or invalid ({mode!r}); set it to "
                              f"'{POLICY_MODE_EMBEDDED}' - every effecting action is denied (RT-W03)")
    if mode == POLICY_MODE_SERVICE:
        return NoPolicyClient("policy_mode 'service' is not supported by this app version yet "
                              "(Phase 4, RT-W04); use 'embedded'", CODE_UNAVAILABLE)
    missing = [name for name, value in (("policy_path", policy_path), ("policy_approvals_path", approvals_path),
                                        ("policy_halt_path", halt_path)) if not (value or "").strip()]
    if missing:
        return NoPolicyClient(f"policy_mode 'embedded' needs {', '.join(missing)}; every effecting action is denied",
                              CODE_NO_POLICY_MODE)
    try:
        from policy_client.embedded import EmbeddedPolicyClient  # bundled wheel (3.1b)
        from policy_core.model import DecisionRequest
    except Exception as exc:  # ImportError or a broken package: fail closed
        return NoPolicyClient(f"policy_client package not available ({type(exc).__name__}); "
                              "every effecting action is denied", CODE_UNAVAILABLE)
    try:
        client = EmbeddedPolicyClient(policy_path=policy_path.strip(), approvals_path=approvals_path.strip(),
                                      halt_path=halt_path.strip(), audit_path=(audit_path or "").strip() or None,
                                      audit_client=audit_client)
    except Exception as exc:
        return NoPolicyClient(f"policy client could not start ({type(exc).__name__})", CODE_UNAVAILABLE)
    return PolicyClientAdapter(client, DecisionRequest, POLICY_MODE_EMBEDDED)
