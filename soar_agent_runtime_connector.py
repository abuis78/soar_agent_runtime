"""
soar_agent_runtime_connector.py
Main BaseConnector subclass for SOAR Agent Runtime.
"""

import json
import phantom.app as phantom
from phantom.base_connector import BaseConnector
from phantom.action_result import ActionResult

from soar_agent_runtime_utils import (
    LLMProvider,
    ToolRegistry,
    AgentDefinitionStore,
    SkillStore,
    ReActLoop,
    MAX_STEPS_HARD_LIMIT,
)


class SoarAgentRuntimeConnector(BaseConnector):

    def __init__(self):
        super().__init__()
        self._state = None
        self._config = {}
        self._default_max_steps = 25
        self._agent_list_name = "agent_definitions"
        self._skill_list_name = "agent_skills"
        self._default_provider = "anthropic"

    def initialize(self):
        self._config = self.get_config()
        self._state = self.load_state()
        self._default_max_steps = int(self._config.get("default_max_steps", 25))
        self._agent_list_name = self._config.get("agent_list_name", "agent_definitions")
        self._skill_list_name = self._config.get("skill_list_name", "agent_skills")
        self._default_provider = self._config.get("default_provider", "anthropic")
        return phantom.APP_SUCCESS

    def finalize(self):
        self.save_state(self._state)
        return phantom.APP_SUCCESS

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

    # -----------------------------------------------------------------------
    # LLM Provider Factory
    # -----------------------------------------------------------------------

    def _build_llm_provider(self, provider: str, model: str) -> LLMProvider:
        provider = provider.lower()
        if provider == "anthropic":
            return LLMProvider(
                provider="anthropic",
                api_key=self._config.get("anthropic_api_key", ""),
                api_url=self._config.get("anthropic_api_url", "https://api.anthropic.com"),
                model=model or "claude-haiku-4-5-20251001"
            )
        elif provider == "openai":
            return LLMProvider(
                provider="openai",
                api_key=self._config.get("openai_api_key", ""),
                api_url=self._config.get("openai_api_url", "https://api.openai.com/v1"),
                model=model or "gpt-4o-mini"
            )
        elif provider == "gemini":
            return LLMProvider(
                provider="gemini",
                api_key=self._config.get("gemini_api_key", ""),
                api_url="",
                model=model or "gemini-2.0-flash"
            )
        elif provider == "local":
            return LLMProvider(
                provider="local",
                api_key=self._config.get("local_llm_api_key", "ollama"),
                api_url=self._config.get("local_llm_url", "http://localhost:11434/v1"),
                model=model or "llama3"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: anthropic | openai | gemini | local")

    # -----------------------------------------------------------------------
    # Action: test connectivity
    # -----------------------------------------------------------------------

    def _handle_test_connectivity(self, param):
        action_result = self.add_action_result(ActionResult(dict(param)))
        self.save_progress("Testing LLM provider connectivity...")
        results = {}
        providers = [
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("openai",    "gpt-4o-mini"),
            ("gemini",    "gemini-2.0-flash"),
            ("local",     "llama3")
        ]
        any_success = False
        for provider_name, default_model in providers:
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

        if any_success:
            return action_result.set_status(phantom.APP_SUCCESS, f"Connectivity OK. Results: {json.dumps(results)}")
        return action_result.set_status(phantom.APP_ERROR, f"All providers failed. Results: {json.dumps(results)}")

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
        if not task:
            return action_result.set_status(phantom.APP_ERROR, "task is required.")

        # Issue #6: safe integer conversion
        container_id = self._safe_int(param.get("container_id", 0), name="container_id")
        if not container_id:
            return action_result.set_status(phantom.APP_ERROR, "container_id must be a valid positive integer.")

        max_steps_override = self._safe_int(param.get("max_steps", 0), name="max_steps") or 0

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
        provider = agent_def.get("provider", self._default_provider)
        model = agent_def.get("model", "")
        self.save_progress(f"Provider: {provider} / {model}")
        try:
            llm = self._build_llm_provider(provider, model)
        except ValueError as e:
            return action_result.set_status(phantom.APP_ERROR, str(e))

        # Build tool registry
        allowed_tools_raw = agent_def.get("allowed_tools", ["add_note"])
        if isinstance(allowed_tools_raw, str):
            allowed_tools = [t.strip() for t in allowed_tools_raw.split(",")]
        else:
            allowed_tools = allowed_tools_raw
        tools = ToolRegistry(self, container_id, allowed_tools)

        # Load and inject skills
        skill_ids = agent_def.get("skills", [])
        if isinstance(skill_ids, str):
            skill_ids = [s.strip() for s in skill_ids.split(",") if s.strip()]
        skill_injection = ""
        if skill_ids:
            self.save_progress(f"Loading skills: {skill_ids}")
            skill_store = SkillStore(self, self._skill_list_name)
            loaded_skills = skill_store.load_many(skill_ids)
            skill_injection = SkillStore.build_injection(loaded_skills)
            self.save_progress(f"Injected {len(loaded_skills)} skill(s).")

        # Run ReAct loop
        system_prompt = agent_def.get("system_prompt", "You are a SOC analyst agent.")
        loop = ReActLoop(
            connector=self,
            llm=llm,
            tools=tools,
            agent_id=agent_id,
            system_prompt=system_prompt,
            max_steps=max_steps,
            container_id=container_id,
            extra_context=extra_context_str,
            skill_injection=skill_injection
        )
        result = loop.run(task)
        action_result.add_data(result)

        # Add summary note
        try:
            import phantom.rules as ph_rules
            ph_rules.add_note(
                container=container_id,
                note_type="general",
                title=f"[Agent: {agent_id}] {task[:60]}",
                content=f"**Final Answer:**\n{result['final_answer']}\n\n**Steps:** {result['steps_taken']} | **Provider:** {result['provider']} / {result['model']}"
            )
        except Exception:
            pass

        action_result.update_summary({
            "agent_id": agent_id,
            "steps_taken": result["steps_taken"],
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

        skills_raw = param.get("skills", "")
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

        config = {
            "agent_id": agent_id,
            "provider": param.get("provider", self._default_provider),
            "model": param.get("model", ""),
            "system_prompt": param.get("system_prompt", ""),
            "allowed_tools": param.get("allowed_tools", "splunk_search,add_note"),
            "max_steps": int(param.get("max_steps", 0)) or self._default_max_steps,
            "skills": skills
        }

        store = AgentDefinitionStore(self, self._agent_list_name)
        if not store.save(agent_id, config):
            return action_result.set_status(phantom.APP_ERROR, f"Failed to save agent '{agent_id}'.")

        action_result.add_data({"agent_id": agent_id, "status": "created", "skills": skills})
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
                "skills": ", ".join(agent.get("skills", []))
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
            return action_result.set_status(phantom.APP_ERROR, f"Failed to save skill '{skill_id}'.")

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
