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
            loaded_skills = skill_store.load_many(skill_ids)
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
            "model": result["model"]
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
        agents = store.list_all()
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
        skills = skill_store.list_all()
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
    # Action dispatcher
    # -----------------------------------------------------------------------

    def handle_action(self, param):
        handlers = {
            "test_connectivity": self._handle_test_connectivity,
            "run_agent":         self._handle_run_agent,
            "create_agent":      self._handle_create_agent,
            "list_agents":       self._handle_list_agents,
            "create_skill":      self._handle_create_skill,
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
