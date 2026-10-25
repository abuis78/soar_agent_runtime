"""
soar_agent_runtime_connector.py
Main BaseConnector subclass for SOAR Agent Runtime.
"""

import json
import re
import time
import phantom.app as phantom
from phantom.base_connector import BaseConnector
from phantom.action_result import ActionResult

from soar_agent_runtime_utils import (
    LLMProvider,
    ToolRegistry,
    AgentDefinitionStore,
    SkillStore,
    ReActLoop,
    AGENT_ROLES,
    DEFAULT_AGENT_ROLE,
    MAX_STEPS_HARD_LIMIT,
    new_run_id,
    SoarCustomList,
    compact_rows,
)
from soar_agent_runtime_audit import ArtifactAuditSink, NullAuditSink, definition_version
from soar_agent_runtime_broker import BrokerClient, BrokerConfig, BrokerConfigError
from soar_agent_runtime_policy import POLICY_MODES, build_policy_client

try:
    from ai_soc_telemetry import TelemetryClient
    HAS_TELEMETRY_CLIENT = True
except ImportError:
    HAS_TELEMETRY_CLIENT = False

# Open item #5: identifiers are used verbatim as SOAR Custom-List keys, so they
# must be restricted to a safe charset to prevent list-key injection.
_ID_RE = re.compile(r'^[A-Za-z0-9_.\-]{1,64}$')


# --- what really happened in a run (contract E, 1.10.3) ---------------------------------
# The model's FINAL_ANSWER is a claim. A 7B model once answered "severity updated to
# high" without ever calling update_severity (run rt-20261025T050736Z-cf477c13), so the
# summary note lists the executed actions from the step log next to the claim.

# Effecting tools: name -> (observation prefix that means it took effect, claim pattern).
EFFECT_CHECKS = {
    "update_severity": ("Container severity updated to:", r"\bseverit(y|ies)\b"),
    "catalog_invoke":  ("ALLOWED:", r"\b(propos(al|ed)|building block|baustein)"),
}


def executed_actions(step_log) -> list[dict]:
    """Tool calls of a run (FINAL_ANSWER excluded) with a one-line outcome each."""
    if isinstance(step_log, str):
        try:
            step_log = json.loads(step_log)
        except ValueError:
            return []
    actions = []
    for entry in step_log or []:
        tool = (entry or {}).get("action") or ""
        if not tool or tool == "FINAL_ANSWER":
            continue
        observation = str(entry.get("observation", "")).strip()
        first_line = observation.splitlines()[0] if observation else ""
        if observation.startswith("{"):
            first_line = "data returned"
        effect_prefix = EFFECT_CHECKS.get(tool, (None,))[0]
        actions.append({
            "step": entry.get("step"),
            "tool": tool,
            "outcome": first_line[:160],
            "took_effect": bool(effect_prefix and observation.startswith(effect_prefix)),
            "not_executed": list(entry.get("not_executed") or []),
        })
    return actions


def unbacked_claims(final_answer: str, actions: list[dict]) -> list[str]:
    """Effecting tools the answer seems to talk about although none of their calls took effect."""
    import re
    warnings = []
    for tool, (_prefix, pattern) in EFFECT_CHECKS.items():
        if re.search(pattern, final_answer or "", re.IGNORECASE) and not any(
                a["tool"] == tool and a["took_effect"] for a in actions):
            warnings.append(f"The answer mentions what {tool} does, but no {tool} call took effect in this run.")
    return warnings


def executed_actions_block(final_answer: str, actions: list[dict]) -> str:
    lines = ["**Executed actions** (from the step log, not from the model):"]
    lines += [f"- step {a['step']}: `{a['tool']}` - {a['outcome'] or 'no observation'}"
              + (f" (in the same response, not executed: {', '.join(a['not_executed'])})"
                 if a.get("not_executed") else "")
              for a in actions] or ["- none"]
    if warnings := unbacked_claims(final_answer, actions):
        lines += ["", *[f"**Warning:** {w}" for w in warnings]]
    return "\n".join(lines)


# --- tool catalogue for the user (contract H, 1.12.0) -------------------------------------

# Tools the runtime offers on its own when their source delivers them; listing them in
# allowed_tools is harmless but not needed.
IMPLICIT_TOOLS = ("broker_invoke", "catalog_invoke", "report_gap")


def known_tool_names() -> list[str]:
    return list(ToolRegistry.TOOL_DEFINITIONS)


def unknown_tool_names(allowed_tools) -> list[str]:
    names = allowed_tools.split(",") if isinstance(allowed_tools, str) else list(allowed_tools or [])
    known = set(known_tool_names())
    return [n.strip() for n in names if n.strip() and n.strip() not in known]


def tool_availability(name: str, airgap: bool, broker_url: str, policy_mode: str) -> str:
    """One sentence on whether and how a tool works with this asset's settings."""
    gate = bool(policy_mode)
    match name:
        case "splunk_search":
            return "not available: SOAR offers apps no search API; use broker_invoke templates"
        case "update_severity" if airgap:
            return ("available through the Policy Manager gate" if gate
                    else "denied: airgap_mode needs policy_mode for effecting tools")
        case "update_severity":
            return "available; outside airgap_mode no policy gate is consulted"
        case "broker_invoke":
            return ("offered when the Tool Broker delivers templates" if broker_url
                    else "not offered: set broker_url")
        case "catalog_invoke" | "report_gap":
            return ("offered in airgap_mode when the Policy Manager releases building blocks" if airgap and gate
                    else "not offered: needs airgap_mode and policy_mode")
        case _:
            return "available"


class SoarAgentRuntimeConnector(BaseConnector):

    def __init__(self):
        super().__init__()
        self._state = None
        self._config = {}
        self._default_max_steps = 25
        self._agent_list_name = "agent_definitions"
        self._skill_list_name = "agent_skills"
        self._default_provider = "anthropic"
        self._telemetry = None
        # RT-A01/RT-A03 (Umsetzungskonzept 1.3): Air-Gap profile and model bounds
        self._airgap_mode = False
        self._llm_timeout_s = 120.0
        self._llm_temperature = 0.0
        self._llm_max_tokens = 4096
        self._llm_seed = None
        self._max_observation_chars = 8000
        self._audit_enabled = False
        # RT-L07 (Umsetzungskonzept 2.1): Tool Broker connection, mTLS via file paths
        self._broker_url = ""
        self._broker_timeout_s = 30.0
        self._broker_client_cert = ""
        self._broker_client_key = ""
        self._broker_ca = ""
        self._broker_transport = None  # tests inject a fake; None = requests over mTLS
        # RT-W01/W03 (Umsetzungskonzept 3.1a, E11): Policy Manager, embedded mode
        self._policy_mode = ""
        self._policy_path = ""
        self._policy_state_dir = ""
        # RT-W01/W04 (4.1): service mode over mTLS, file paths like the broker
        self._policy_service_url = ""
        self._policy_timeout_s = 5.0
        self._policy_client_cert = ""
        self._policy_client_key = ""
        self._policy_ca = ""
        self._policy_client = None     # tests inject a PolicyGate; None = build from the asset

    def initialize(self):
        self._config = self.get_config()
        self._state = self.load_state()
        self._default_max_steps = int(self._config.get("default_max_steps", 25))
        self._agent_list_name = self._config.get("agent_list_name", "agent_definitions")
        self._skill_list_name = self._config.get("skill_list_name", "agent_skills")
        self._default_provider = self._config.get("default_provider", "anthropic")
        self._airgap_mode = bool(self._config.get("airgap_mode", False))
        self._llm_timeout_s = self._cfg_number("llm_timeout_s", 120.0, float)
        self._llm_temperature = self._cfg_number("llm_temperature", 0.0, float)
        self._llm_max_tokens = self._cfg_number("llm_max_tokens", 4096, int)
        self._llm_seed = self._cfg_number("llm_seed", None, int)
        self._max_observation_chars = self._cfg_number("max_observation_chars", 8000, int)
        self._audit_enabled = bool(self._config.get("audit_enabled", False))
        self._broker_url = (self._config.get("broker_url") or "").strip()
        self._broker_timeout_s = self._cfg_number("broker_timeout_s", 30.0, float)
        self._broker_client_cert = (self._config.get("broker_client_cert") or "").strip()
        self._broker_client_key = (self._config.get("broker_client_key") or "").strip()
        self._broker_ca = (self._config.get("broker_ca") or "").strip()
        self._policy_mode = (self._config.get("policy_mode") or "").strip().lower()
        self._policy_path = (self._config.get("policy_path") or "").strip()
        self._policy_state_dir = (self._config.get("policy_state_dir") or "").strip().rstrip("/")
        self._policy_service_url = (self._config.get("policy_service_url") or "").strip()
        self._policy_timeout_s = self._cfg_number("policy_timeout_s", 5.0, float)
        self._policy_client_cert = (self._config.get("policy_client_cert") or "").strip()
        self._policy_client_key = (self._config.get("policy_client_key") or "").strip()
        self._policy_ca = (self._config.get("policy_ca") or "").strip()
        if self._airgap_mode:
            self.save_progress("[airgap] enabled - only provider 'local' is allowed.")
            if not self._broker_url:
                self.save_progress("[airgap] broker_url not set - read tools unavailable.")
            if self._policy_mode not in POLICY_MODES:
                self.save_progress("[airgap] policy_mode not set - every effecting action is denied (RT-W03).")
        self._telemetry = self._build_telemetry()
        return phantom.APP_SUCCESS

    def finalize(self):
        self._flush_telemetry()
        self.save_state(self._state)
        return phantom.APP_SUCCESS

    # -----------------------------------------------------------------------
    # Telemetry (AI-SOC Telemetry)
    # -----------------------------------------------------------------------

    def _build_telemetry(self):
        """Build the telemetry client from the asset config.

        Fail-open: every problem disables telemetry instead of the app. Returns
        None when telemetry is off or unusable.
        """
        if not self._config.get("telemetry_enabled", False):
            return None
        if not HAS_TELEMETRY_CLIENT:
            self.save_progress("[telemetry] ai_soc_telemetry not installed - telemetry disabled.")
            return None
        hec_url = self._config.get("hec_url", "")
        hec_token = self._config.get("hec_token", "")
        if not hec_url or not hec_token:
            self.save_progress("[telemetry] hec_url/hec_token not configured - telemetry disabled.")
            return None
        try:
            client = TelemetryClient(
                hec_url=hec_url,
                hec_token=hec_token,
                index=self._config.get("telemetry_index", "ai_soc_audit"),
                verify_tls=self._config.get("telemetry_verify_tls", True),
            )
            self.save_progress("[telemetry] enabled.")
            return client
        except Exception as e:
            # Never leak the token into logs (L8) - report the type only.
            self.save_progress(f"[telemetry] init failed ({type(e).__name__}) - telemetry disabled.")
            return None

    def _flush_telemetry(self, timeout: float = 3.0) -> None:
        """Give the telemetry sender a bounded moment to ship queued events.

        The library ships events from a daemon thread and exposes no flush()/
        close(), so anything still queued is dropped when this short-lived action
        process exits - the last action of a run is the most likely to be lost.
        Polling the queue is a stopgap; the real fix is a flush() upstream.
        """
        q = getattr(self._telemetry, "_q", None)
        if q is None:
            return
        try:
            deadline = time.monotonic() + timeout
            while not q.empty() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.2)  # let a final in-flight POST complete
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Input helpers
    # -----------------------------------------------------------------------

    def _safe_int(self, value, default: int = 0, name: str = "parameter") -> int | None:
        """Issue #6: Safe integer conversion with error reporting."""
        try:
            result = int(value)
            if result < 0:
                self.save_progress(f"Warning: {name} must be non-negative, got {result!r}. Using {default}.")
                return default
            return result
        except (ValueError, TypeError):
            self.save_progress(f"Warning: {name} must be numeric, got {value!r}.")
            return None

    def _cfg_number(self, name: str, default, cast):
        """Read a numeric asset field; empty = default, invalid = default + warning."""
        raw = self._config.get(name)
        if raw is None or raw == "":
            return default
        try:
            return cast(raw)
        except (ValueError, TypeError):
            self.save_progress(f"Warning: asset field {name} must be numeric, got {raw!r}. Using {default}.")
            return default

    @staticmethod
    def _valid_id(value: str) -> bool:
        """Open item #5: validate identifiers used as Custom-List keys."""
        return bool(_ID_RE.match(value or ""))

    # -----------------------------------------------------------------------
    # Tool Broker client (RT-L07)
    # -----------------------------------------------------------------------

    def _build_broker_client(self):
        """BrokerClient from the asset fields, or None when no broker_url is set.

        Raises BrokerConfigError for an unusable configuration; the caller
        decides (fail-closed in airgap_mode, warning otherwise).
        """
        if not self._broker_url:
            return None
        config = BrokerConfig(
            url=self._broker_url,
            timeout_s=self._broker_timeout_s,
            client_cert=self._broker_client_cert,
            client_key=self._broker_client_key,
            ca=self._broker_ca,
        )
        return BrokerClient(config, transport=self._broker_transport)

    # -----------------------------------------------------------------------
    # LLM Provider Factory
    # -----------------------------------------------------------------------

    def _llm_kwargs(self) -> dict:
        """RT-A03: bounds from the asset, applied to every provider."""
        return {
            "timeout_s": self._llm_timeout_s,
            "temperature": self._llm_temperature,
            "max_tokens": self._llm_max_tokens,
            "seed": self._llm_seed,
        }

    def _build_llm_provider(self, provider: str, model: str) -> LLMProvider:
        provider = (provider or "").lower()
        # RT-A01/RT-A02: in Air-Gap mode no cloud provider may be constructed,
        # whatever the source of the value (agent definition, asset default,
        # DEV fallback definition). Fail closed before any network call.
        if self._airgap_mode and provider != "local":
            raise ValueError(
                f"airgap_mode: provider '{provider}' is not allowed, only 'local'; "
                "set provider=local in the agent definition or default_provider=local in the asset."
            )
        if provider == "anthropic":
            return LLMProvider(
                provider="anthropic",
                api_key=self._config.get("anthropic_api_key", ""),
                api_url=self._config.get("anthropic_api_url", "https://api.anthropic.com"),
                model=model or "claude-haiku-4-5-20251001",
                **self._llm_kwargs(),
            )
        elif provider == "openai":
            return LLMProvider(
                provider="openai",
                api_key=self._config.get("openai_api_key", ""),
                api_url=self._config.get("openai_api_url", "https://api.openai.com/v1"),
                model=model or "gpt-4o-mini",
                **self._llm_kwargs(),
            )
        elif provider == "gemini":
            return LLMProvider(
                provider="gemini",
                api_key=self._config.get("gemini_api_key", ""),
                api_url="",
                model=model or "gemini-2.0-flash",
                **self._llm_kwargs(),
            )
        elif provider == "local":
            return LLMProvider(
                provider="local",
                api_key=self._config.get("local_llm_api_key", "ollama"),
                api_url=self._config.get("local_llm_url", "http://localhost:11434/v1"),
                model=model or self._local_llm_model(),
                **self._llm_kwargs(),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: anthropic | openai | gemini | local")

    def _local_llm_model(self) -> str:
        """Model for provider 'local' when the agent definition names none; also the
        model `test connectivity` asks the local server for (asset field local_llm_model)."""
        return (self._config.get("local_llm_model") or "llama3").strip() or "llama3"

    # -----------------------------------------------------------------------
    # Action: test connectivity
    # -----------------------------------------------------------------------

    def _handle_test_connectivity(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        mode = "airgap" if self._airgap_mode else "normal"
        self.save_progress(f"Testing LLM provider connectivity ({mode} mode)...")
        results = {}
        providers = [
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("openai",    "gpt-4o-mini"),
            ("gemini",    "gemini-2.0-flash"),
            ("local",     self._local_llm_model())
        ]
        any_success = False
        for provider_name, default_model in providers:
            # RT-A04 / AC-01: in Air-Gap mode no cloud endpoint is contacted,
            # not even by the test action. Only 'local' is exercised.
            if self._airgap_mode and provider_name != "local":
                results[provider_name] = "SKIPPED (airgap_mode)"
                continue
            key_field = f"{provider_name}_api_key" if provider_name != "local" else "local_llm_api_key"
            url_field = f"{provider_name}_api_url" if provider_name != "local" else "local_llm_url"
            has_config = bool(self._config.get(key_field) or self._config.get(url_field))
            if not has_config and provider_name != "local":
                results[provider_name] = "SKIPPED (no credentials configured)"
                continue
            self.save_progress(f"Testing {provider_name}...")
            try:
                llm = self._build_llm_provider(provider_name, default_model)
                success, message = llm.test_connection()
                results[provider_name] = message
                if success:
                    any_success = True
            except Exception as e:
                results[provider_name] = f"ERROR: {str(e)}"

        # RT-A04 (part 2): Tool Broker and Policy Manager health. Both are
        # mandatory in airgap_mode; outside it the broker is informational and
        # the policy gate is not used at all (E4).
        broker_ok = self._test_broker(results)
        policy_ok = self._test_policy(results)
        self._test_custom_lists(results)  # informational only: never fails the test (contract B)

        if not any_success:
            return action_result.set_status(
                phantom.APP_ERROR, f"All providers failed ({mode}). Results: {json.dumps(results)}")
        if self._airgap_mode and not broker_ok:
            return action_result.set_status(
                phantom.APP_ERROR, f"Broker check failed ({mode}). Results: {json.dumps(results)}")
        if self._airgap_mode and not policy_ok:
            return action_result.set_status(
                phantom.APP_ERROR, f"Policy check failed ({mode}). Results: {json.dumps(results)}")
        return action_result.set_status(
            phantom.APP_SUCCESS, f"Connectivity OK ({mode}). Results: {json.dumps(results)}")

    def _build_policy_client(self):
        """PolicyGate for this asset (RT-W03, E11). Tests inject self._policy_client.
        Outside airgap_mode the gate is not used (E4), so None is returned and
        ToolRegistry keeps its fail-closed default for the airgap path."""
        if self._policy_client is not None:
            return self._policy_client
        if not self._airgap_mode:
            return None
        state = self._policy_state_dir
        return build_policy_client(
            self._policy_mode,
            policy_path=self._policy_path,
            approvals_path=f"{state}/approvals.json" if state else "",
            halt_path=f"{state}/halt.json" if state else "",
            audit_path=f"{state}/audit.jsonl" if state else "",
            sequences_path=f"{state}/sequences.json" if state else "",
            service_url=self._policy_service_url,
            service_timeout_s=self._policy_timeout_s,
            client_cert=self._policy_client_cert,
            client_key=self._policy_client_key,
            ca=self._policy_ca,
        )

    def _test_custom_lists(self, results: dict) -> None:
        """Report missing Custom Lists without writing anything; 'setup app' creates them."""
        missing = []
        for name in (self._agent_list_name, self._skill_list_name):
            lst = SoarCustomList(self, name)
            if not lst.available():
                return
            try:
                if not lst.exists():
                    missing.append(name)
            except ValueError as e:
                results["custom_lists"] = f"ERROR: {e}"
                return
        results["custom_lists"] = (f"MISSING: {', '.join(missing)} (run 'setup app')" if missing else "OK")

    def _test_broker(self, results: dict) -> bool:
        """Fill results["broker"]; True only when a configured broker answers /health."""
        if not self._broker_url:
            results["broker"] = ("NOT CONFIGURED (required in airgap_mode)" if self._airgap_mode
                                 else "SKIPPED (no broker_url)")
            return False
        self.save_progress("Testing broker...")
        try:
            client = self._build_broker_client()
        except BrokerConfigError as e:
            results["broker"] = f"ERROR: misconfigured: {e}"
            return False
        ok, message = client.health()
        results["broker"] = f"OK: {message}" if ok else f"ERROR: {message}"
        return ok

    def _test_policy(self, results: dict) -> bool:
        """Fill results["policy"]; True only when the Policy Manager gate reports healthy.

        A halted Policy Manager is reachable and therefore healthy; health()
        says "HALTED (<reason>)" so the operator sees it. Outside airgap_mode
        the gate is never consulted (E4), so nothing is built or checked.
        """
        if not self._policy_mode:
            results["policy"] = ("NOT CONFIGURED (required in airgap_mode)" if self._airgap_mode
                                 else "SKIPPED (no policy_mode)")
            return False
        if not self._airgap_mode:
            results["policy"] = f"SKIPPED (policy_mode '{self._policy_mode}' is only used in airgap_mode)"
            return False
        self.save_progress("Testing policy manager...")
        try:
            ok, message = self._build_policy_client().health()
        except Exception as e:  # fail closed, like every other gate call (RT-W03)
            ok, message = False, f"policy health check raised {type(e).__name__}: {e}"
        results["policy"] = f"OK: {message}" if ok else f"ERROR: {message}"
        return ok

    # -----------------------------------------------------------------------
    # Action: run agent
    # -----------------------------------------------------------------------

    def _handle_run_agent(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))

        agent_id = param.get("agent_id", "")
        task = param.get("task", "")
        extra_context_str = param.get("extra_context", "")

        if not agent_id:
            return action_result.set_status(phantom.APP_ERROR, "agent_id is required.")
        if not self._valid_id(agent_id):
            return action_result.set_status(
                phantom.APP_ERROR, "agent_id must match [A-Za-z0-9_.-] and be 1-64 chars.")
        if not task:
            return action_result.set_status(phantom.APP_ERROR, "task is required.")

        # Issue #6: safe integer conversion
        container_id = self._safe_int(param.get("container_id", 0), name="container_id")
        if not container_id:
            return action_result.set_status(phantom.APP_ERROR, "container_id must be a valid positive integer.")

        max_steps_override = self._safe_int(param.get("max_steps", 0), name="max_steps") or 0
        # RT-W08: stufe (0-4) from the triggering playbook, domaene of the run; empty = unknown
        stage_raw = param.get("stufe", "")
        stage = self._safe_int(stage_raw, name="stufe") if stage_raw not in (None, "") else None
        stage = stage if isinstance(stage, int) and 0 <= stage <= 4 else None
        domain = str(param.get("domaene") or "").strip()

        # Load agent definition
        self.save_progress(f"Loading agent: {agent_id}")
        store = AgentDefinitionStore(self, self._agent_list_name)
        try:
            agent_def = store.load(agent_id)
        except ValueError as e:
            return action_result.set_status(phantom.APP_ERROR, str(e))

        # Resolve max steps with hard cap (Issue #4: DoS prevention)
        max_steps = max_steps_override or agent_def.get("max_steps") or self._default_max_steps
        if max_steps <= 0:
            max_steps = self._default_max_steps
        if max_steps > MAX_STEPS_HARD_LIMIT:
            self.save_progress(f"Warning: max_steps capped at {MAX_STEPS_HARD_LIMIT} (requested: {max_steps}).")
            max_steps = MAX_STEPS_HARD_LIMIT

        # Build LLM provider
        provider = agent_def.get("provider") or self._default_provider
        model = agent_def.get("model", "")
        self.save_progress(f"Provider: {provider} / {model}")
        try:
            llm = self._build_llm_provider(provider, model)
        except ValueError as e:
            return action_result.set_status(phantom.APP_ERROR, str(e))

        # Tool Broker (RT-L07): misconfiguration is fatal in airgap_mode,
        # a warning otherwise. Unreachable brokers are handled per run (RT-L06).
        try:
            broker = self._build_broker_client()
        except BrokerConfigError as e:
            if self._airgap_mode:
                return action_result.set_status(phantom.APP_ERROR, f"broker misconfigured: {e}")
            self.save_progress(f"[broker] misconfigured, running without broker: {e}")
            broker = None

        policy = self._build_policy_client()
        if self._airgap_mode:
            self.save_progress(f"[policy] mode={getattr(policy, 'mode', '') or 'none'} "
                               f"({'ready' if policy and policy.health()[0] else 'not available - effecting actions denied'})")

        # One lauf_id for broker context, audit and result (RT-P01)
        run_id = new_run_id()

        # Build tool registry
        allowed_tools_raw = agent_def.get("allowed_tools", ["add_note"])
        if isinstance(allowed_tools_raw, str):
            allowed_tools = [t.strip() for t in allowed_tools_raw.split(",")]
        else:
            allowed_tools = allowed_tools_raw
        tools = ToolRegistry(
            self,
            container_id,
            allowed_tools,
            telemetry=self._telemetry,
            agent_id=agent_id,
            agent_role=agent_def.get("agent_role", DEFAULT_AGENT_ROLE),
            broker=broker,
            run_id=run_id,
            airgap=self._airgap_mode,
            policy=policy,
            stage=stage,
            domain=domain,
        )
        tools.load_broker_tools()  # RT-L03: templates once per run
        tools.load_catalog()       # RT-W09: building blocks once per run (empty outside airgap)
        if tools.halted:
            # RT-B01 / RT-L05: halt state known before the first step -> do not start.
            return action_result.set_status(
                phantom.APP_ERROR, "Tool Broker is HALTED - run not started (RT-B01)")

        # Load and inject skills
        skill_ids = agent_def.get("skills", [])
        if isinstance(skill_ids, str):
            skill_ids = [s.strip() for s in skill_ids.split(",") if s.strip()]
        skill_injection = ""
        required_keys: list[str] = []
        if skill_ids:
            self.save_progress(f"Loading skills: {skill_ids}")
            skill_store = SkillStore(self, self._skill_list_name)
            try:
                loaded_skills = skill_store.load_many(skill_ids)
            except ValueError as e:  # the agent asks for skills, so an unreadable list is fatal
                return action_result.set_status(phantom.APP_ERROR, str(e))
            skill_injection = SkillStore.build_injection(loaded_skills)
            # RT-P04 (E3): a skill's output_schema template is a required
            # top-level key set for FINAL_ANSWER.
            for skill in loaded_skills:
                schema = skill.get("output_schema")
                if isinstance(schema, dict):
                    required_keys += [k for k in schema if k not in required_keys]
            self.save_progress(f"Injected {len(loaded_skills)} skill(s).")

        # Audit sink (RT-P02/P03, E2): artifact per step; mandatory and
        # fail-closed in airgap_mode, optional and fail-open otherwise.
        if self._airgap_mode or self._audit_enabled:
            audit = ArtifactAuditSink(self, container_id, fail_closed=self._airgap_mode)
        else:
            audit = NullAuditSink()

        # Run ReAct loop
        system_prompt = agent_def.get("system_prompt", "You are a SOC analyst agent.")
        loop = ReActLoop(
            connector=self,
            llm=llm,
            tools=tools,
            run_id=run_id,
            agent_id=agent_id,
            system_prompt=system_prompt,
            max_steps=max_steps,
            container_id=container_id,
            extra_context=extra_context_str,
            skill_injection=skill_injection,
            audit=audit,
            def_version=definition_version(agent_def),
            strict_schema=self._airgap_mode,
            required_keys=required_keys,
            max_observation_chars=self._max_observation_chars,
        )
        result = loop.run(task)
        audit.flush()
        actions = executed_actions(result.get("step_log"))
        result["executed_actions"] = actions
        result["unbelegte_behauptungen"] = unbacked_claims(result["final_answer"], actions)
        action_result.add_data(result)

        # Add summary note
        try:
            import phantom.rules as ph_rules
            ph_rules.add_note(
                container=container_id,
                note_type="general",
                title=f"[Agent: {agent_id}] {task[:60]}",
                content=(
                    f"**Final Answer:**\n{result['final_answer']}\n\n"
                    f"{executed_actions_block(result['final_answer'], actions)}\n\n"
                    f"**Steps:** {result['steps_taken']} | **Provider:** {result['provider']} / {result['model']} "
                    f"| **lauf_id:** {result['lauf_id']}"
                )
            )
        except Exception:
            pass

        action_result.update_summary({
            "agent_id": agent_id,
            "lauf_id": result["lauf_id"],
            "steps_taken": result["steps_taken"],
            "audit_ereignisse": result["audit_ereignisse"],
            "audit_fehler": result["audit_fehler"],
            "provider": result["provider"],
            "model": result["model"],
            "unbelegte_behauptungen": len(result["unbelegte_behauptungen"]),
        })

        if result["final_answer"].startswith("ERROR"):
            return action_result.set_status(phantom.APP_ERROR, result["final_answer"])
        return action_result.set_status(phantom.APP_SUCCESS, f"Agent completed in {result['steps_taken']} steps.")

    # -----------------------------------------------------------------------
    # Action: create agent
    # -----------------------------------------------------------------------

    def _handle_create_agent(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        agent_id = param.get("agent_id", "")
        if not agent_id:
            return action_result.set_status(phantom.APP_ERROR, "agent_id is required.")
        if not self._valid_id(agent_id):
            return action_result.set_status(
                phantom.APP_ERROR, "agent_id must match [A-Za-z0-9_.-] and be 1-64 chars.")

        # Contract H: an unknown name used to be stored silently and the tool never offered.
        if unknown := unknown_tool_names(param.get("allowed_tools") or ""):
            return action_result.set_status(
                phantom.APP_ERROR,
                f"Unknown tool(s) in allowed_tools: {', '.join(unknown)}. "
                f"Valid tools: {', '.join(known_tool_names())} (see the action 'list tools').")

        skills_raw = param.get("skills", "")
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

        # agent_role groups the Agent Precision panel, so only fixed values are
        # accepted - a typo would silently fragment the metric.
        agent_role = (param.get("agent_role") or DEFAULT_AGENT_ROLE).strip().lower()
        if agent_role not in AGENT_ROLES:
            return action_result.set_status(
                phantom.APP_ERROR,
                f"agent_role must be one of {list(AGENT_ROLES)}, got '{agent_role}'."
            )

        config = {
            "agent_id": agent_id,
            "provider": param.get("provider", self._default_provider),
            "model": param.get("model", ""),
            "system_prompt": param.get("system_prompt", ""),
            # RT-L02 / B2: no free-form SPL by default; the same default lives in app.json
            "allowed_tools": param.get("allowed_tools") or "add_note",
            "max_steps": int(param.get("max_steps", 0)) or self._default_max_steps,
            "skills": skills,
            "agent_role": agent_role
        }

        store = AgentDefinitionStore(self, self._agent_list_name)
        if not store.save(agent_id, config):
            detail = getattr(store, "last_error", "")
            return action_result.set_status(
                phantom.APP_ERROR, f"Failed to save agent '{agent_id}'" + (f": {detail}" if detail else "."))

        action_result.add_data({
            "agent_id": agent_id, "status": "created", "skills": skills, "agent_role": agent_role
        })
        action_result.update_summary({"agent_id": agent_id})
        return action_result.set_status(phantom.APP_SUCCESS, f"Agent '{agent_id}' saved.")

    # -----------------------------------------------------------------------
    # Action: list agents
    # -----------------------------------------------------------------------

    def _handle_list_agents(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        store = AgentDefinitionStore(self, self._agent_list_name)
        try:
            agents = store.list_all()
        except ValueError as e:  # CustomListError: list missing or unreadable
            return action_result.set_status(phantom.APP_ERROR, str(e))
        if not agents:
            return action_result.set_status(phantom.APP_SUCCESS, "No agents found.")
        for agent in agents:
            action_result.add_data({
                "agent_id": agent.get("agent_id", ""),
                "provider": agent.get("provider", ""),
                "model": agent.get("model", ""),
                "allowed_tools": agent.get("allowed_tools", ""),
                "max_steps": agent.get("max_steps", self._default_max_steps),
                "skills": ", ".join(agent.get("skills", [])),
                "agent_role": agent.get("agent_role", DEFAULT_AGENT_ROLE)
            })
        action_result.update_summary({"total_agents": len(agents)})
        return action_result.set_status(phantom.APP_SUCCESS, f"Found {len(agents)} agent(s).")

    # -----------------------------------------------------------------------
    # Action: create skill
    # -----------------------------------------------------------------------

    def _handle_create_skill(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        skill_id = param.get("skill_id", "")
        if not skill_id:
            return action_result.set_status(phantom.APP_ERROR, "skill_id is required.")
        if not self._valid_id(skill_id):
            return action_result.set_status(
                phantom.APP_ERROR, "skill_id must match [A-Za-z0-9_.-] and be 1-64 chars.")

        output_schema_str = param.get("output_schema", "")
        output_schema = None
        if output_schema_str:
            try:
                output_schema = json.loads(output_schema_str)
            except Exception:
                return action_result.set_status(phantom.APP_ERROR, "output_schema must be valid JSON.")

        config = {
            "description": param.get("description", ""),
            "inject": param.get("inject", ""),
        }
        if output_schema:
            config["output_schema"] = output_schema

        skill_store = SkillStore(self, self._skill_list_name)
        if not skill_store.save(skill_id, config):
            detail = getattr(skill_store, "last_error", "")
            return action_result.set_status(
                phantom.APP_ERROR, f"Failed to save skill '{skill_id}'" + (f": {detail}" if detail else "."))

        action_result.add_data({"skill_id": skill_id, "status": "created"})
        action_result.update_summary({"skill_id": skill_id})
        return action_result.set_status(phantom.APP_SUCCESS, f"Skill '{skill_id}' saved.")

    # -----------------------------------------------------------------------
    # Action: list skills
    # -----------------------------------------------------------------------

    def _handle_list_skills(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        skill_store = SkillStore(self, self._skill_list_name)
        try:
            skills = skill_store.list_all()
        except ValueError as e:
            return action_result.set_status(phantom.APP_ERROR, str(e))
        if not skills:
            return action_result.set_status(phantom.APP_SUCCESS, "No skills found.")
        for skill in skills:
            action_result.add_data({
                "skill_id": skill.get("skill_id", ""),
                "description": skill.get("description", ""),
                "has_inject": bool(skill.get("inject", "")),
                "has_output_schema": bool(skill.get("output_schema"))
            })
        action_result.update_summary({"total_skills": len(skills)})
        return action_result.set_status(phantom.APP_SUCCESS, f"Found {len(skills)} skill(s).")

    # -----------------------------------------------------------------------
    # Action: list tools (contract H, 1.12.0)
    # -----------------------------------------------------------------------

    def _handle_list_tools(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        for name, defn in ToolRegistry.TOOL_DEFINITIONS.items():
            action_result.add_data({
                "tool": name,
                "description": defn.get("description", ""),
                "parameters": ", ".join(defn.get("parameters", {})),
                "effecting": bool(defn.get("wirkt", True)),
                "source": defn.get("quelle", ""),
                "in_allowed_tools": name not in IMPLICIT_TOOLS,
                "availability": tool_availability(name, self._airgap_mode, self._broker_url, self._policy_mode),
            })
        count = len(ToolRegistry.TOOL_DEFINITIONS)
        action_result.update_summary({"total_tools": count})
        return action_result.set_status(phantom.APP_SUCCESS, f"Found {count} tool(s).")

    # -----------------------------------------------------------------------
    # Action: test tools (contract I, 1.12.0)
    # -----------------------------------------------------------------------
    # Calls every built-in tool for real, without the model, against a test container,
    # because on SOAR 8.7 almost every tool was broken while all unit tests (which fake
    # SOAR) were green. Changes it makes: one note, and the severity briefly, restored.

    SELF_TEST_NOTE = "[tool self-test]"
    MODEL_CHECK_TASK = "Call update_severity with severity high. Then add a note explaining why."

    def _handle_test_tools(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        container_id = self._safe_int(param.get("container_id", 0), name="container_id")
        if not container_id:
            return action_result.set_status(phantom.APP_ERROR, "container_id of a test container is required.")
        model_check = str(param.get("model_check", False)).lower() in ("true", "1", "yes")
        rows: list[dict] = []

        def row(tool: str, status: str, detail: str) -> None:
            rows.append({"tool": tool, "status": status, "detail": str(detail)[:500]})

        tools = ToolRegistry(self, container_id, known_tool_names(), airgap=self._airgap_mode)
        original = self._self_test_read(tools, container_id, row)
        self._self_test_note(tools, row)
        self._self_test_severity(tools, container_id, original, row)
        spl = tools._tool_splunk_search({"spl_query": "index=_internal | head 1"})
        row("splunk_search", "OK" if spl == ToolRegistry.SPL_UNAVAILABLE else "FAIL",
            "reports 'not available' as designed" if spl == ToolRegistry.SPL_UNAVAILABLE else spl)
        row("run_sub_agent", "SKIPPED", "needs a second agent and the model; shown by run agent")
        self._self_test_services(row)
        if model_check:
            self._self_test_model(row)

        for r in rows:
            action_result.add_data(r)
        failed = [r for r in rows if r["status"] == "FAIL"]
        action_result.update_summary({"tools_ok": sum(r["status"] == "OK" for r in rows),
                                      "tools_failed": len(failed), "container_id": container_id})
        if failed:
            return action_result.set_status(
                phantom.APP_ERROR, f"test tools: {len(failed)} failed: "
                + "; ".join(f"{r['tool']}: {r['detail']}" for r in failed))
        return action_result.set_status(
            phantom.APP_SUCCESS, f"test tools: all {len(rows)} check(s) passed or skipped on container {container_id}.")

    def _self_test_read(self, tools, container_id: int, row) -> str:
        out = tools._tool_get_container_info({})
        try:
            data = json.loads(out)
            container = data["container"]
            if int(container.get("id", 0)) != container_id:
                raise ValueError(f"returned container {container.get('id')}")
        except (ValueError, KeyError, TypeError) as e:
            row("get_container_info", "FAIL", out if out.startswith("Failed") else f"unexpected output ({e})")
            return ""
        severity = str(container.get("severity", "")).lower()
        row("get_container_info", "OK", f"container {container_id} '{container.get('name', '')}', "
            f"severity {severity or 'unknown'}, {data.get('artifact_count', 0)} artifact(s)")
        return severity

    def _self_test_note(self, tools, row) -> None:
        out = tools._tool_add_note({"title": self.SELF_TEST_NOTE,
                                    "content": "Written by the action 'test tools' to prove add_note works. Safe to delete."})
        row("add_note", "OK" if out.startswith("Note added") else "FAIL", out)

    def _self_test_severity(self, tools, container_id: int, original: str, row) -> None:
        valid = ("low", "medium", "high", "critical")
        if self._airgap_mode:
            row("update_severity", "SKIPPED", "airgap_mode: effects run only through the policy gate; test it with run agent")
            return
        if original not in valid:
            row("update_severity", "SKIPPED", f"current severity '{original or 'unknown'}' cannot be restored safely")
            return
        probe = "low" if original != "low" else "medium"
        changed = tools._tool_update_severity({"severity": probe})
        now = self._self_test_severity_now(tools)
        restored = tools._tool_update_severity({"severity": original})
        after = self._self_test_severity_now(tools)
        if after != original:
            row("update_severity", "FAIL", f"container {container_id} left at '{after}' instead of '{original}' "
                f"- restore it by hand ({restored})")
        elif not changed.startswith("Container severity updated") or now != probe:
            row("update_severity", "FAIL", f"set to '{probe}' did not take effect (now '{now}'): {changed}")
        else:
            row("update_severity", "OK", f"set to '{probe}' and restored to '{original}'")

    @staticmethod
    def _self_test_severity_now(tools) -> str:
        try:
            return str(json.loads(tools._tool_get_container_info({}))["container"].get("severity", "")).lower()
        except (ValueError, KeyError, TypeError):
            return "unknown"

    def _self_test_services(self, row) -> None:
        results: dict = {}
        if self._broker_url:
            ok = self._test_broker(results)
            row("broker_invoke", "OK" if ok else "FAIL", results.get("broker", ""))
        else:
            row("broker_invoke", "SKIPPED", "no broker_url")
        if self._airgap_mode and self._policy_mode:
            ok = self._test_policy(results)
            row("catalog_invoke / report_gap", "OK" if ok else "FAIL", results.get("policy", ""))
        else:
            row("catalog_invoke / report_gap", "SKIPPED", "need airgap_mode and policy_mode")

    def _self_test_model(self, row) -> None:
        """Does the default model keep the ReAct format? Counts ACTION lines in one answer."""
        system = ("You are a SOC agent. Tools:\n- update_severity(severity): set the case severity\n"
                  "- add_note(title, content): add a note\nAnswer with THOUGHT, then exactly one ACTION and its "
                  "PARAMS (JSON), or FINAL_ANSWER.")
        try:
            model = self._local_llm_model() if self._default_provider == "local" else ""
            answer = self._build_llm_provider(self._default_provider, model).chat(
                system, [{"role": "user", "content": self.MODEL_CHECK_TASK}])
        except Exception as e:
            row("model format", "FAIL", f"{type(e).__name__}: {e}")
            return
        count = sum(1 for line in answer.splitlines() if line.strip().startswith("ACTION:"))
        if count == 1:
            row("model format", "OK", "one ACTION per response")
        elif count > 1:
            row("model format", "WARN", f"{count} ACTIONs in one response; the runtime runs the first and asks "
                "for the rest, so runs need more steps")
        else:
            row("model format", "FAIL", f"no ACTION line in the answer: {answer[:200]}")

    # -----------------------------------------------------------------------
    # Action: setup app (contract B, 1.11.0)
    # -----------------------------------------------------------------------
    # SOAR runs no app code at install time, so this action does the one-time setup
    # (Custom Lists) and checks the asset configuration in one place.

    def _handle_setup_app(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        dry_run = str(param.get("dry_run", False)).lower() in ("true", "1", "yes")
        checks: list[dict] = []

        def check(name: str, status: str, detail: str) -> None:
            checks.append({"check": name, "status": status, "detail": detail})

        self._setup_lists(check, dry_run)
        self._setup_llm(check)
        if self._airgap_mode:
            self._setup_airgap(check)
        else:
            check("airgap", "OK", "airgap_mode is off; broker and policy fields are not needed")

        for row in checks:
            action_result.add_data(row)
        problems = [c for c in checks if c["status"] in ("MISSING", "ERROR")]
        changed = [c for c in checks if c["status"] in ("CREATED", "FIXED")]
        action_result.update_summary({"checks": len(checks), "problems": len(problems),
                                      "changed": len(changed), "dry_run": dry_run})
        if problems:
            return action_result.set_status(
                phantom.APP_ERROR, f"Setup: {len(problems)} problem(s): "
                + "; ".join(f"{c['check']}: {c['detail']}" for c in problems))
        return action_result.set_status(
            phantom.APP_SUCCESS, f"Setup {'checked (dry run)' if dry_run else 'complete'}: "
            f"{len(checks)} check(s), {len(changed)} change(s).")

    def _setup_lists(self, check, dry_run: bool) -> None:
        for kind, name in (("agents", self._agent_list_name), ("skills", self._skill_list_name)):
            label = f"custom list {name}"
            lst = SoarCustomList(self, name)
            if not lst.available():
                check(label, "ERROR", "SOAR REST not available")
                continue
            try:
                if not lst.exists():
                    if dry_run:
                        check(label, "WOULD_CREATE", "list missing; run without dry_run to create it")
                    else:
                        lst.create([["", ""]])
                        check(label, "CREATED", f"empty {kind} list created")
                    continue
                compact, dropped = compact_rows(lst.rows())
                if kind == "agents":
                    self._setup_agent_tools(check, compact)
                if not dropped:
                    check(label, "OK", f"{sum(1 for r in compact if r[0])} {kind}, no duplicate rows")
                elif dry_run:
                    check(label, "WOULD_FIX", f"{dropped} older duplicate row(s) would be removed")
                else:
                    lst.replace(compact)
                    check(label, "FIXED", f"{dropped} older duplicate row(s) removed, latest row per id kept")
            except ValueError as e:  # CustomListError / SoarRestError
                check(label, "ERROR", str(e))

    @staticmethod
    def _setup_agent_tools(check, rows: list) -> None:
        """Contract H: stored agents may predate the name check in create agent."""
        bad = []
        for row in rows:
            if not row[0]:
                continue
            try:
                tools = json.loads(row[1]).get("allowed_tools", "")
            except (ValueError, AttributeError):
                continue
            if unknown := unknown_tool_names(tools):
                bad.append(f"{row[0]}: {', '.join(unknown)}")
        if bad:
            check("agent tool names", "WARN", "unknown tools are never offered - " + "; ".join(bad))

    def _setup_llm(self, check) -> None:
        provider = self._default_provider
        if self._airgap_mode and provider != "local":
            check("default_provider", "MISSING", f"'{provider}' is not allowed in airgap_mode; set it to local")
            return
        if provider == "local":
            missing = [f for f in ("local_llm_url", "local_llm_model") if not self._config.get(f)]
            if missing:
                check("local llm fields", "MISSING", f"set {', '.join(missing)}")
                return
        elif not self._config.get(f"{provider}_api_key"):
            check(f"{provider} key", "MISSING", f"set {provider}_api_key")
            return
        try:
            model = self._local_llm_model() if provider == "local" else ""
            ok, message = self._build_llm_provider(provider, model).test_connection()
        except Exception as e:
            ok, message = False, f"{type(e).__name__}: {e}"
        check(f"llm {provider}", "OK" if ok else "ERROR", message)

    def _setup_airgap(self, check) -> None:
        import os

        def readable_files(label: str, fields: tuple[str, ...]) -> None:
            values = {f: (self._config.get(f) or "").strip() for f in fields}
            if missing := [f for f, v in values.items() if not v]:
                check(label, "MISSING", f"set {', '.join(missing)}")
            elif bad := [f"{f}={v}" for f, v in values.items() if not os.access(v, os.R_OK)]:
                check(label, "MISSING", f"not readable by the app user: {', '.join(bad)}")
            else:
                check(label, "OK", "files readable")

        if not self._broker_url.startswith("https://"):
            check("broker_url", "MISSING", "set broker_url (https) - read tools come only from the broker")
        else:
            check("broker_url", "OK", self._broker_url)
            readable_files("broker certificates", ("broker_client_cert", "broker_client_key"))
        if self._policy_mode == "embedded":
            readable_files("policy_path", ("policy_path",))
            state = (self._policy_state_dir or "").strip()
            if not state or not os.access(state, os.W_OK):
                check("policy_state_dir", "MISSING", "set a directory the app user can write")
            elif not os.path.isfile(os.path.join(state, "halt.json")):
                check("halt.json", "MISSING",
                      f'create {state}/halt.json with {{"angehalten": false, "grund": ""}} (missing file = halted)')
            else:
                check("policy_state_dir", "OK", "writable, halt.json present")
        elif self._policy_mode == "service":
            check("policy_service_url", "OK" if self._policy_service_url.startswith("https://") else "MISSING",
                  self._policy_service_url or "set policy_service_url (https)")
            readable_files("policy certificates", ("policy_client_cert", "policy_client_key", "policy_ca"))
        else:
            check("policy_mode", "MISSING", "set policy_mode to embedded or service - otherwise every effect is denied")

    # -----------------------------------------------------------------------
    # Action dispatcher
    # -----------------------------------------------------------------------

    def handle_action(self, param):
        handlers = {
            "test_connectivity": self._handle_test_connectivity,
            "run_agent":         self._handle_run_agent,
            "create_agent":      self._handle_create_agent,
            "list_agents":       self._handle_list_agents,
            "create_skill":      self._handle_create_skill,
            "setup_app":         self._handle_setup_app,
            "list_tools":        self._handle_list_tools,
            "test_tools":        self._handle_test_tools,
            "list_skills":       self._handle_list_skills,
        }
        action_id = self.get_action_identifier()
        if action_id in handlers:
            return handlers[action_id](param)
        return phantom.APP_SUCCESS


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        in_json = json.loads(f.read())
    connector = SoarAgentRuntimeConnector()
    connector.print_progress_message = True
    ret_val = connector.handle_action(json.dumps(in_json), None)
    print(json.dumps(json.loads(connector.get_action_results()), indent=4))
    sys.exit(0 if phantom.APP_SUCCESS == ret_val else 1)
