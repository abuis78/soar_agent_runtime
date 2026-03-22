"""
soar_agent_runtime_utils.py
Utility classes for SOAR Agent Runtime:
  - LLMProvider abstraction (Anthropic, OpenAI, Gemini, Local)
  - ToolRegistry with built-in SOC tools
  - AgentDefinitionStore (Custom List wrapper)
  - ReActLoop engine
"""

import json
import time
import requests

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified interface for all LLM providers."""

    def __init__(self, provider: str, api_key: str, api_url: str, model: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    def chat(self, system_prompt: str, messages: list) -> str:
        """Send chat request and return assistant response text."""
        if self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, messages)
        elif self.provider == "openai":
            return self._chat_openai(system_prompt, messages)
        elif self.provider == "gemini":
            return self._chat_gemini(system_prompt, messages)
        elif self.provider == "local":
            return self._chat_local(system_prompt, messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_anthropic(self, system_prompt: str, messages: list) -> str:
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: phenv pip3 install anthropic")
        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.api_url)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    def _chat_openai(self, system_prompt: str, messages: list) -> str:
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: phenv pip3 install openai")
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def _chat_gemini(self, system_prompt: str, messages: list) -> str:
        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed. Run: phenv pip3 install google-generativeai")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})
        response = model.generate_content(gemini_messages)
        return response.text

    def _chat_local(self, system_prompt: str, messages: list) -> str:
        """OpenAI-compatible local LLM (Ollama, LM Studio, etc.)."""
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: phenv pip3 install openai")
        client = OpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_url
        )
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def test_connection(self) -> tuple:
        """Test provider connectivity. Returns (success: bool, message: str)."""
        try:
            result = self.chat(
                system_prompt="You are a test assistant.",
                messages=[{"role": "user", "content": "Reply with exactly: OK"}]
            )
            return True, f"{self.provider} ({self.model}): {result.strip()}"
        except Exception as e:
            return False, f"{self.provider} ({self.model}): {str(e)}"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of tools available to agents."""

    TOOL_DEFINITIONS = {
        "splunk_search": {
            "description": "Execute a Splunk SPL search query and return results",
            "parameters": {
                "spl_query": "The SPL search query to execute",
                "earliest": "Earliest time (default: -24h)",
                "latest": "Latest time (default: now)"
            }
        },
        "add_note": {
            "description": "Add a note to the current SOAR container",
            "parameters": {
                "title": "Note title",
                "content": "Note content"
            }
        },
        "update_severity": {
            "description": "Update the severity of the current container",
            "parameters": {
                "severity": "New severity: low | medium | high | critical"
            }
        },
        "get_container_info": {
            "description": "Get metadata and artifacts of the current container",
            "parameters": {}
        },
        "run_sub_agent": {
            "description": "Invoke another agent as a sub-task and return its result",
            "parameters": {
                "agent_id": "ID of the agent to invoke",
                "task": "Task description for the sub-agent"
            }
        }
    }

    def __init__(self, connector, container_id: int, allowed_tools: list):
        self.connector = connector
        self.container_id = container_id
        self.allowed_tools = allowed_tools

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions for the system prompt."""
        lines = []
        for tool_name in self.allowed_tools:
            if tool_name in self.TOOL_DEFINITIONS:
                t = self.TOOL_DEFINITIONS[tool_name]
                params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items())
                lines.append(f"- {tool_name}({params}): {t['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return its output as string."""
        if tool_name not in self.allowed_tools:
            return f"ERROR: Tool '{tool_name}' is not in the allowed tools list."
        handler = getattr(self, f"_tool_{tool_name}", None)
        if not handler:
            return f"ERROR: Tool '{tool_name}' is registered but not implemented."
        try:
            return handler(params)
        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}"

    def _tool_splunk_search(self, params: dict) -> str:
        spl = params.get("spl_query", "")
        earliest = params.get("earliest", "-24h")
        latest = params.get("latest", "now")
        if not spl:
            return "ERROR: spl_query parameter is required."
        self.connector.save_progress(f"Executing SPL: {spl[:80]}...")
        try:
            import phantom.rules as ph_rules
            results = ph_rules.run_query(
                query=spl,
                start_time=earliest,
                end_time=latest
            )
            if not results:
                return "SPL search returned 0 results."
            return f"SPL results ({len(results)} rows):\n{json.dumps(results[:20], indent=2)}"
        except Exception as e:
            return f"SPL search error: {str(e)}"

    def _tool_add_note(self, params: dict) -> str:
        title = params.get("title", "Agent Note")
        content = params.get("content", "")
        try:
            import phantom.rules as ph_rules
            ph_rules.add_note(
                container=self.container_id,
                note_type="general",
                title=title,
                content=content
            )
            return f"Note added: '{title}'"
        except Exception as e:
            return f"Failed to add note: {str(e)}"

    def _tool_update_severity(self, params: dict) -> str:
        severity = params.get("severity", "").lower()
        valid = ["low", "medium", "high", "critical"]
        if severity not in valid:
            return f"ERROR: severity must be one of {valid}"
        try:
            import phantom.rules as ph_rules
            ph_rules.update_container(
                container=self.container_id,
                severity=severity
            )
            return f"Container severity updated to: {severity}"
        except Exception as e:
            return f"Failed to update severity: {str(e)}"

    def _tool_get_container_info(self, params: dict) -> str:
        try:
            import phantom.rules as ph_rules
            container = ph_rules.get_container(self.container_id)
            artifacts = ph_rules.get_artifacts(container_id=self.container_id)
            info = {
                "container": container,
                "artifact_count": len(artifacts) if artifacts else 0,
                "artifacts": artifacts[:5] if artifacts else []
            }
            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            return f"Failed to get container info: {str(e)}"

    def _tool_run_sub_agent(self, params: dict) -> str:
        agent_id = params.get("agent_id", "")
        task = params.get("task", "")
        if not agent_id or not task:
            return "ERROR: agent_id and task are required."
        return f"[Sub-agent '{agent_id}' invocation queued for task: {task}] — implement via recursive run_agent call."


# ---------------------------------------------------------------------------
# Agent Definition Store
# ---------------------------------------------------------------------------

class AgentDefinitionStore:
    """Loads and saves agent definitions from SOAR Custom Lists."""

    def __init__(self, connector, list_name: str):
        self.connector = connector
        self.list_name = list_name

    def load(self, agent_id: str) -> dict:
        """Load agent definition from Custom List. Returns dict or raises."""
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name)
            if not rows:
                raise ValueError(f"Custom List '{self.list_name}' is empty or does not exist.")
            for row in rows:
                if isinstance(row, list) and len(row) >= 2:
                    if row[0] == agent_id:
                        return json.loads(row[1])
                elif isinstance(row, dict):
                    if row.get("agent_id") == agent_id:
                        return row
            raise ValueError(f"Agent '{agent_id}' not found in Custom List '{self.list_name}'.")
        except ImportError:
            return self._load_fallback(agent_id)

    def _load_fallback(self, agent_id: str) -> dict:
        """Fallback for local testing without phantom.rules."""
        self.connector.save_progress(f"[DEV] phantom.rules not available, using fallback for agent: {agent_id}")
        return {
            "agent_id": agent_id,
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "system_prompt": "You are a SOC analyst. Analyze the given task and respond with FINAL_ANSWER when done.",
            "allowed_tools": ["add_note", "get_container_info"],
            "max_steps": 5
        }

    def save(self, agent_id: str, config: dict) -> bool:
        """Save or update agent definition in Custom List."""
        try:
            import phantom.rules as ph_rules
            config["agent_id"] = agent_id
            ph_rules.add_to_list(
                list_name=self.list_name,
                values=[[agent_id, json.dumps(config)]]
            )
            return True
        except Exception as e:
            self.connector.save_progress(f"Failed to save agent definition: {str(e)}")
            return False

    def list_all(self) -> list:
        """Return all agent definitions as list of dicts."""
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name)
            agents = []
            if not rows:
                return agents
            for row in rows:
                if isinstance(row, list) and len(row) >= 2:
                    try:
                        config = json.loads(row[1])
                        config["agent_id"] = row[0]
                        agents.append(config)
                    except Exception:
                        pass
            return agents
        except ImportError:
            return []


# ---------------------------------------------------------------------------
# ReAct Loop Engine
# ---------------------------------------------------------------------------

REACT_SYSTEM_TEMPLATE = """You are {agent_id}, an autonomous SOC analyst agent.

{system_prompt}

## Available Tools
{tool_descriptions}

## ReAct Instructions
You operate in a Reason-Act-Observe loop. For EVERY response you MUST use one of these formats:

FORMAT A — Use a tool:
THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
PARAMS: <JSON object with tool parameters>

FORMAT B — Finish:
THOUGHT: <your final reasoning>
FINAL_ANSWER: <your complete answer or conclusion>

Rules:
- Always start with THOUGHT
- Use exactly one ACTION per response, OR use FINAL_ANSWER
- PARAMS must be valid JSON
- Be concise but thorough
- Max steps allowed: {max_steps}

## Context
Container ID: {container_id}
{extra_context}
"""


class ReActLoop:
    """Executes the ReAct (Reason-Act-Observe) loop for an agent."""

    def __init__(self, connector, llm: LLMProvider, tools: ToolRegistry,
                 agent_id: str, system_prompt: str, max_steps: int,
                 container_id: int, extra_context: str = ""):
        self.connector = connector
        self.llm = llm
        self.tools = tools
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.container_id = container_id
        self.extra_context = extra_context
        self.step_log = []

    def _build_system_prompt(self) -> str:
        return REACT_SYSTEM_TEMPLATE.format(
            agent_id=self.agent_id,
            system_prompt=self.system_prompt,
            tool_descriptions=self.tools.get_tool_descriptions(),
            max_steps=self.max_steps,
            container_id=self.container_id,
            extra_context=self.extra_context or "No extra context provided."
        )

    def _parse_response(self, text: str) -> tuple:
        """Parse LLM response. Returns (action_type, tool_name, params, thought, final_answer)."""
        thought = ""
        tool_name = None
        params = {}
        final_answer = None

        lines = text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.startswith("ACTION:"):
                tool_name = line[7:].strip()
            elif line.startswith("PARAMS:"):
                params_str = line[7:].strip()
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith(("THOUGHT:", "ACTION:", "FINAL_ANSWER:")):
                    params_str += "\n" + lines[j]
                    j += 1
                try:
                    params = json.loads(params_str)
                except Exception:
                    params = {}
            elif line.startswith("FINAL_ANSWER:"):
                final_answer = line[13:].strip()
                j = i + 1
                while j < len(lines):
                    final_answer += "\n" + lines[j]
                    j += 1
                final_answer = final_answer.strip()
            i += 1

        return tool_name, params, thought, final_answer

    def run(self, task: str) -> dict:
        """Execute the ReAct loop. Returns result dict."""
        self.connector.save_progress(f"Agent '{self.agent_id}' starting task: {task[:80]}...")
        system_prompt = self._build_system_prompt()
        messages = [{"role": "user", "content": task}]
        steps_taken = 0
        final_answer = None

        while steps_taken < self.max_steps:
            steps_taken += 1
            self.connector.save_progress(f"Step {steps_taken}/{self.max_steps}...")

            try:
                response_text = self.llm.chat(system_prompt, messages)
            except Exception as e:
                return self._error_result(f"LLM call failed at step {steps_taken}: {str(e)}", steps_taken)

            tool_name, params, thought, final_answer = self._parse_response(response_text)

            step_entry = {
                "step": steps_taken,
                "thought": thought,
                "action": tool_name or "FINAL_ANSWER",
                "params": params,
                "observation": None
            }

            if final_answer is not None:
                step_entry["observation"] = final_answer
                self.step_log.append(step_entry)
                self.connector.save_progress(f"Agent '{self.agent_id}' completed in {steps_taken} steps.")
                break

            if tool_name:
                observation = self.tools.execute(tool_name, params)
                step_entry["observation"] = observation
                self.step_log.append(step_entry)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            else:
                step_entry["observation"] = "ERROR: No ACTION or FINAL_ANSWER found in response."
                self.step_log.append(step_entry)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "OBSERVATION: Your response did not contain a valid ACTION or FINAL_ANSWER. Please follow the format exactly."})
        else:
            final_answer = f"Max steps ({self.max_steps}) reached without FINAL_ANSWER."
            self.connector.save_progress(f"Agent '{self.agent_id}' hit max steps limit.")

        return {
            "agent_id": self.agent_id,
            "final_answer": final_answer,
            "steps_taken": steps_taken,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "step_log": json.dumps(self.step_log, indent=2)
        }

    def _error_result(self, message: str, steps: int) -> dict:
        return {
            "agent_id": self.agent_id,
            "final_answer": f"ERROR: {message}",
            "steps_taken": steps,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "step_log": json.dumps(self.step_log, indent=2)
        }
