"""
soar_agent_runtime_utils.py
Utility classes for SOAR Agent Runtime.
Python 3.13 compatible.
"""

from __future__ import annotations
import json
import re
from urllib.parse import urlparse

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
# Security constants
# ---------------------------------------------------------------------------

# Issue #2 (Prompt Injection): ReAct control keywords must not appear in
# external data (artifacts, SPL results) fed back into the LLM conversation.
_REACT_KEYWORDS = ("THOUGHT:", "ACTION:", "PARAMS:", "FINAL_ANSWER:", "OBSERVATION:")

# Issue #5 (SPL Injection): Destructive/exfiltrating/command-executing SPL pipe
# commands. NOTE: a blocklist is defense-in-depth, not a guarantee — it can be
# evaded via macros, subsearches, or novel commands. The primary control is the
# per-agent `allowed_tools` allowlist that gates access to splunk_search at all.
_BLOCKED_SPL_RE = re.compile(
    r'\|\s*(delete|drop|truncate|map\s|outputlookup\b|outputcsv\b|rest\s+/services'
    r'|collect\b|tscollect\b|mcollect\b|sendemail\b|sendalert\b|runshellscript\b'
    r'|script\b|dump\b|crawl\b)',
    re.IGNORECASE
)

# Issue #3 (SSRF): Allowed URL schemes per provider.
_HTTPS_ONLY_PROVIDERS = {"anthropic", "openai", "gemini"}
_PRIVATE_IP_RE = re.compile(
    r'^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|169\.254\.|0\.0\.0\.0'
    r'|::1|::$|fc|fd)',
    re.IGNORECASE
)

# Issue #4 (DoS): Absolute maximum ReAct steps regardless of config.
MAX_STEPS_HARD_LIMIT = 50

# Length caps (open item #3: context stuffing). External/user text is truncated
# before it reaches the LLM so a single input cannot blow up the context window.
_MAX_OBSERVATION_LEN = 8000
_MAX_TASK_LEN = 4000
_MAX_CONTEXT_LEN = 4000

# Exception scrubbing (open item #2: secret leakage via tracebacks). Patterns
# that may carry credentials must never be echoed back into the LLM context or
# action results.
_SECRET_RE = re.compile(
    r'(sk-[A-Za-z0-9_\-]{8,}'                 # OpenAI / Anthropic style keys
    r'|Bearer\s+[A-Za-z0-9._\-]{8,}'          # bearer tokens
    r'|AIza[A-Za-z0-9_\-]{10,}'               # Google API keys
    r'|(?:api[_-]?key|token|authorization)\s*[=:]\s*\S+'  # key=value pairs
    r'|https?://[^\s/]*:[^\s/]*@\S+)',        # creds embedded in URLs
    re.IGNORECASE
)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit, appending a marker when cut."""
    if len(text) > limit:
        return text[:limit] + f"\n[...truncated {len(text) - limit} chars]"
    return text


def _scrub(text: str) -> str:
    """Redact anything resembling a secret before it is surfaced to the LLM/UI."""
    return _SECRET_RE.sub("[REDACTED]", text)


def _scrub_exc(exc: Exception) -> str:
    """Return a safe, secret-scrubbed one-line description of an exception."""
    return _scrub(f"{type(exc).__name__}: {exc}")


def _sanitize_observation(text: str) -> str:
    """Escape ReAct control keywords in external data before injecting into LLM."""
    for kw in _REACT_KEYWORDS:
        text = text.replace(kw, f"[{kw.rstrip(':')}]")
    return text


def _validate_api_url(url: str, provider: str) -> str:
    """Raise ValueError if the URL could be used for SSRF."""
    if not url:
        return url
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if provider in _HTTPS_ONLY_PROVIDERS and scheme != "https":
        raise ValueError(
            f"Provider '{provider}' requires HTTPS. Got scheme: '{scheme}'"
        )
    if provider == "local" and scheme not in ("http", "https"):
        raise ValueError(f"Local LLM URL must use http or https. Got: '{scheme}'")
    if _PRIVATE_IP_RE.match(hostname) and provider != "local":
        raise ValueError(
            f"API URL for '{provider}' must not point to a private/loopback address: {hostname}"
        )
    return url


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified interface for all LLM providers."""

    def __init__(self, provider: str, api_key: str, api_url: str, model: str) -> None:
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_url = _validate_api_url(api_url, self.provider)  # Issue #3: SSRF guard
        self.model = model

    def chat(self, system_prompt: str, messages: list[dict]) -> str:
        """Send chat request and return assistant response text."""
        match self.provider:
            case "anthropic":
                return self._chat_anthropic(system_prompt, messages)
            case "openai":
                return self._chat_openai(system_prompt, messages)
            case "gemini":
                return self._chat_gemini(system_prompt, messages)
            case "local":
                return self._chat_local(system_prompt, messages)
            case _:
                raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_anthropic(self, system_prompt: str, messages: list[dict]) -> str:
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic not installed. Run: phenv pip3 install anthropic")
        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.api_url)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    def _chat_openai(self, system_prompt: str, messages: list[dict]) -> str:
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: phenv pip3 install openai")
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def _chat_gemini(self, system_prompt: str, messages: list[dict]) -> str:
        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed. Run: phenv pip3 install google-generativeai")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        gemini_messages = [
            {"role": "user" if msg["role"] == "user" else "model",
             "parts": [msg["content"]]}
            for msg in messages
        ]
        response = model.generate_content(gemini_messages)
        return response.text

    def _chat_local(self, system_prompt: str, messages: list[dict]) -> str:
        """OpenAI-compatible local LLM (Ollama, LM Studio, etc.)."""
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: phenv pip3 install openai")
        client = OpenAI(api_key=self.api_key or "ollama", base_url=self.api_url)
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def test_connection(self) -> tuple[bool, str]:
        """Test provider connectivity. Returns (success, message)."""
        try:
            result = self.chat(
                system_prompt="You are a test assistant.",
                messages=[{"role": "user", "content": "Reply with exactly: OK"}]
            )
            return True, f"{self.provider} ({self.model}): {result.strip()}"
        except Exception as e:
            return False, f"{self.provider} ({self.model}): {e}"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of tools available to agents."""

    TOOL_DEFINITIONS: dict[str, dict] = {
        "splunk_search": {
            "description": "Execute a Splunk SPL search query and return results",
            "parameters": {
                "spl_query": "The SPL search query to execute",
                "earliest":  "Earliest time (default: -24h)",
                "latest":    "Latest time (default: now)"
            }
        },
        "add_note": {
            "description": "Add a note to the current SOAR container",
            "parameters": {
                "title":   "Note title",
                "content": "Note content (markdown supported)"
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
            "parameters":  {}
        },
        "run_sub_agent": {
            "description": "Invoke another agent as a sub-task and return its result",
            "parameters": {
                "agent_id": "ID of the agent to invoke",
                "task":     "Task description for the sub-agent"
            }
        }
    }

    def __init__(self, connector, container_id: int, allowed_tools: list[str]) -> None:
        self.connector = connector
        self.container_id = container_id
        self.allowed_tools = allowed_tools

    def get_tool_descriptions(self) -> str:
        lines = []
        for name in self.allowed_tools:
            if defn := self.TOOL_DEFINITIONS.get(name):
                params = ", ".join(f"{k}: {v}" for k, v in defn["parameters"].items())
                lines.append(f"- {name}({params}): {defn['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict) -> str:
        if tool_name not in self.allowed_tools:
            return f"ERROR: Tool '{tool_name}' is not in the allowed tools list."
        if not (handler := getattr(self, f"_tool_{tool_name}", None)):
            return f"ERROR: Tool '{tool_name}' is registered but not implemented."
        try:
            return handler(params)
        except Exception as e:
            return f"ERROR executing {tool_name}: {_scrub_exc(e)}"

    def _tool_splunk_search(self, params: dict) -> str:
        spl      = params.get("spl_query", "")
        earliest = params.get("earliest", "-24h")
        latest   = params.get("latest", "now")
        if not spl:
            return "ERROR: spl_query parameter is required."
        # Issue #5: Block destructive/exfiltrating/command-executing SPL commands
        if _BLOCKED_SPL_RE.search(spl):
            return ("ERROR: Query contains blocked SPL commands "
                    "(e.g. delete/drop/outputlookup/collect/sendemail/script/rest).")
        self.connector.save_progress(f"Executing SPL: {spl[:80]}...")
        try:
            import phantom.rules as ph_rules
            results = ph_rules.run_query(query=spl, start_time=earliest, end_time=latest)
            if not results:
                return "SPL search returned 0 results."
            # Issue #2 sanitization is applied centrally at the LLM boundary
            # (ReActLoop); returned raw here so step_log keeps forensic fidelity.
            return f"SPL results ({len(results)} rows):\n{json.dumps(results[:20], indent=2)}"
        except Exception as e:
            return f"SPL search error: {_scrub_exc(e)}"

    def _tool_add_note(self, params: dict) -> str:
        title   = params.get("title", "Agent Note")
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
            return f"Failed to add note: {_scrub_exc(e)}"

    def _tool_update_severity(self, params: dict) -> str:
        severity = params.get("severity", "").lower()
        valid = ["low", "medium", "high", "critical"]
        if severity not in valid:
            return f"ERROR: severity must be one of {valid}"
        try:
            import phantom.rules as ph_rules
            ph_rules.update_container(container=self.container_id, severity=severity)
            return f"Container severity updated to: {severity}"
        except Exception as e:
            return f"Failed to update severity: {_scrub_exc(e)}"

    def _tool_get_container_info(self, params: dict) -> str:
        try:
            import phantom.rules as ph_rules
            container = ph_rules.get_container(self.container_id)
            artifacts = ph_rules.get_artifacts(container_id=self.container_id)
            # Issue #2 sanitization is applied centrally at the LLM boundary.
            return json.dumps(
                {"container": container,
                 "artifact_count": len(artifacts) if artifacts else 0,
                 "artifacts": (artifacts or [])[:5]},
                indent=2, default=str
            )
        except Exception as e:
            return f"Failed to get container info: {_scrub_exc(e)}"

    def _tool_run_sub_agent(self, params: dict) -> str:
        agent_id = params.get("agent_id", "")
        task     = params.get("task", "")
        if not agent_id or not task:
            return "ERROR: agent_id and task are required."
        # Open item #6: recursive sub-agent execution is intentionally NOT
        # implemented. Wiring it to run_agent without a recursion-depth guard
        # would let an agent bypass its own max_steps budget and spawn unbounded
        # nested LLM loops (cost/DoS). Any future implementation must enforce a
        # hard depth limit and propagate the remaining step budget.
        return f"[Sub-agent '{agent_id}' is not executed: recursive invocation is disabled for safety.]"


# ---------------------------------------------------------------------------
# Agent Definition Store
# ---------------------------------------------------------------------------

class AgentDefinitionStore:
    """Loads and saves agent definitions from SOAR Custom List."""

    def __init__(self, connector, list_name: str) -> None:
        self.connector = connector
        self.list_name = list_name

    def load(self, agent_id: str) -> dict:
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name)
            if not rows:
                raise ValueError(f"Custom List '{self.list_name}' is empty or does not exist.")
            for row in rows:
                match row:
                    case [id_, config_str, *_] if id_ == agent_id:
                        return json.loads(config_str)
                    case {"agent_id": id_, **rest} if id_ == agent_id:
                        return {"agent_id": id_} | rest
            raise ValueError(f"Agent '{agent_id}' not found in '{self.list_name}'.")
        except ImportError:
            return self._load_fallback(agent_id)

    def _load_fallback(self, agent_id: str) -> dict:
        self.connector.save_progress(f"[DEV] phantom.rules not available, fallback for: {agent_id}")
        return {
            "agent_id":      agent_id,
            "provider":      "anthropic",
            "model":         "claude-haiku-4-5-20251001",
            "system_prompt": "You are a SOC analyst. Analyze the task and respond with FINAL_ANSWER when done.",
            "allowed_tools": ["add_note", "get_container_info"],
            "max_steps":     5,
            "skills":        []
        }

    def save(self, agent_id: str, config: dict) -> bool:
        try:
            import phantom.rules as ph_rules
            ph_rules.add_to_list(
                list_name=self.list_name,
                values=[[agent_id, json.dumps(config | {"agent_id": agent_id})]]
            )
            return True
        except Exception as e:
            self.connector.save_progress(f"Failed to save agent: {e}")
            return False

    def list_all(self) -> list[dict]:
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name) or []
            agents = []
            for row in rows:
                if isinstance(row, list) and len(row) >= 2:
                    try:
                        cfg = json.loads(row[1])
                        agents.append(cfg | {"agent_id": row[0]})
                    except Exception:
                        pass
            return agents
        except ImportError:
            return []


# ---------------------------------------------------------------------------
# Skill Store
# ---------------------------------------------------------------------------

class SkillStore:
    """
    Loads skill definitions from SOAR Custom List 'agent_skills'.

    Skill config fields:
        inject        (str)  — appended verbatim to system prompt
        output_schema (dict) — JSON schema enforced in FINAL_ANSWER
        description   (str)  — human-readable label (not injected)
    """

    DEFAULT_LIST_NAME = "agent_skills"

    def __init__(self, connector, list_name: str | None = None) -> None:
        self.connector = connector
        self.list_name = list_name or self.DEFAULT_LIST_NAME

    def load(self, skill_id: str) -> dict:
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name) or []
            for row in rows:
                if isinstance(row, list) and len(row) >= 2 and row[0] == skill_id:
                    try:
                        return json.loads(row[1])
                    except Exception:
                        return {}
            return {}
        except ImportError:
            self.connector.save_progress(f"[DEV] phantom.rules not available, skill '{skill_id}' skipped.")
            return {}

    def load_many(self, skill_ids: list[str]) -> list[dict]:
        skills = []
        for sid in skill_ids:
            if s := self.load(sid):
                skills.append(s | {"skill_id": sid})
            else:
                self.connector.save_progress(f"[Skills] Warning: skill '{sid}' not found — skipped.")
        return skills

    def save(self, skill_id: str, config: dict) -> bool:
        try:
            import phantom.rules as ph_rules
            ph_rules.add_to_list(
                list_name=self.list_name,
                values=[[skill_id, json.dumps(config)]]
            )
            return True
        except Exception as e:
            self.connector.save_progress(f"Failed to save skill '{skill_id}': {e}")
            return False

    def list_all(self) -> list[dict]:
        try:
            import phantom.rules as ph_rules
            rows = ph_rules.get_list(list_name=self.list_name) or []
            skills = []
            for row in rows:
                if isinstance(row, list) and len(row) >= 2:
                    try:
                        skills.append(json.loads(row[1]) | {"skill_id": row[0]})
                    except Exception:
                        pass
            return skills
        except ImportError:
            return []

    @staticmethod
    def build_injection(skills: list[dict]) -> str:
        """Build skill injection block appended to the system prompt."""
        if not skills:
            return ""
        parts = ["\n## Skills & Response Rules"]
        for skill in skills:
            sid    = skill.get("skill_id", "unknown")
            inject = skill.get("inject", "")
            schema = skill.get("output_schema")
            if inject:
                parts.append(f"\n### {sid}\n{inject.strip()}")
            if schema:
                parts.append(
                    f"\n### {sid} — Output Format\n"
                    f"Your FINAL_ANSWER MUST be valid JSON exactly matching this schema "
                    f"(no text outside the JSON):\n```json\n{json.dumps(schema, indent=2)}\n```"
                )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReAct Loop Engine
# ---------------------------------------------------------------------------

REACT_SYSTEM_TEMPLATE = """You are {agent_id}, an autonomous SOC analyst agent.

{system_prompt}
{skill_injection}

## Available Tools
{tool_descriptions}

## ReAct Instructions
You operate in a Reason-Act-Observe loop. For EVERY response use one of these formats:

FORMAT A — Use a tool:
THOUGHT: <your reasoning>
ACTION: <tool_name>
PARAMS: <JSON object with tool parameters>

FORMAT B — Finish:
THOUGHT: <your final reasoning>
FINAL_ANSWER: <your complete answer or conclusion>

Rules:
- Always start with THOUGHT
- Exactly one ACTION per response, OR FINAL_ANSWER
- PARAMS must be valid JSON
- Be concise but thorough
- Max steps: {max_steps}

## Context
Container ID: {container_id}
{extra_context}
"""


class ReActLoop:
    """Executes the ReAct loop for an agent, with optional skill injection."""

    def __init__(
        self,
        connector,
        llm: LLMProvider,
        tools: ToolRegistry,
        agent_id: str,
        system_prompt: str,
        max_steps: int,
        container_id: int,
        extra_context: str = "",
        skill_injection: str = ""
    ) -> None:
        self.connector       = connector
        self.llm             = llm
        self.tools           = tools
        self.agent_id        = agent_id
        self.system_prompt   = system_prompt
        self.max_steps       = max_steps
        self.container_id    = container_id
        self.extra_context   = extra_context
        self.skill_injection = skill_injection
        self.step_log: list[dict] = []

    def _build_system_prompt(self) -> str:
        return REACT_SYSTEM_TEMPLATE.format(
            agent_id        = self.agent_id,
            system_prompt   = self.system_prompt,
            skill_injection = self.skill_injection,
            tool_descriptions = self.tools.get_tool_descriptions(),
            max_steps       = self.max_steps,
            container_id    = self.container_id,
            extra_context   = _sanitize_observation(_truncate(self.extra_context, _MAX_CONTEXT_LEN))
                              or "No extra context provided."
        )

    def _parse_response(self, text: str) -> tuple[str | None, dict, str, str | None]:
        thought      = ""
        tool_name: str | None = None
        params: dict = {}
        final_answer: str | None = None

        lines = text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match line.split(":", 1):
                case ["THOUGHT", rest]:
                    thought = rest.strip()
                case ["ACTION", rest]:
                    tool_name = rest.strip()
                case ["PARAMS", rest]:
                    params_str = rest.strip()
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith(
                            ("THOUGHT:", "ACTION:", "FINAL_ANSWER:")):
                        params_str += "\n" + lines[j]
                        j += 1
                    try:
                        params = json.loads(params_str)
                    except Exception:
                        params = {}
                case ["FINAL_ANSWER", rest]:
                    tail = "\n".join(lines[i:])
                    final_answer = tail.removeprefix("FINAL_ANSWER:").strip()
                    break
            i += 1

        return tool_name, params, thought, final_answer

    def run(self, task: str) -> dict:
        self.connector.save_progress(f"Agent '{self.agent_id}' starting: {task[:80]}...")
        system_prompt = self._build_system_prompt()
        # Open item #2/#3: user-controlled task is length-capped and stripped of
        # ReAct control keywords before it can steer the loop.
        safe_task = _sanitize_observation(_truncate(task, _MAX_TASK_LEN))
        messages: list[dict] = [{"role": "user", "content": safe_task}]
        steps_taken  = 0
        final_answer: str | None = None

        while steps_taken < self.max_steps:
            steps_taken += 1
            self.connector.save_progress(f"Step {steps_taken}/{self.max_steps}...")

            try:
                response_text = self.llm.chat(system_prompt, messages)
            except Exception as e:
                return self._error_result(
                    f"LLM call failed at step {steps_taken}: {_scrub_exc(e)}", steps_taken)

            tool_name, params, thought, final_answer = self._parse_response(response_text)
            step_entry: dict = {
                "step":        steps_taken,
                "thought":     thought,
                "action":      tool_name or "FINAL_ANSWER",
                "params":      params,
                "observation": None
            }

            if final_answer is not None:
                step_entry["observation"] = final_answer
                self.step_log.append(step_entry)
                self.connector.save_progress(f"Agent '{self.agent_id}' done in {steps_taken} steps.")
                break

            if tool_name:
                observation = self.tools.execute(tool_name, params)
                step_entry["observation"] = observation  # raw kept for forensics
                self.step_log.append(step_entry)
                # Issue #2 (central boundary): every tool observation is sanitized
                # and length-capped before it re-enters the LLM conversation.
                safe_obs = _sanitize_observation(_truncate(observation, _MAX_OBSERVATION_LEN))
                messages += [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": f"OBSERVATION: {safe_obs}"}
                ]
            else:
                step_entry["observation"] = "ERROR: No ACTION or FINAL_ANSWER found."
                self.step_log.append(step_entry)
                messages += [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": "OBSERVATION: No valid ACTION or FINAL_ANSWER. Follow the format exactly."}
                ]
        else:
            final_answer = f"Max steps ({self.max_steps}) reached without FINAL_ANSWER."
            self.connector.save_progress(f"Agent '{self.agent_id}' hit max steps.")

        return {
            "agent_id":     self.agent_id,
            "final_answer": final_answer,
            "steps_taken":  steps_taken,
            "provider":     self.llm.provider,
            "model":        self.llm.model,
            "step_log":     json.dumps(self.step_log, indent=2)
        }

    def _error_result(self, message: str, steps: int) -> dict:
        return {
            "agent_id":     self.agent_id,
            "final_answer": f"ERROR: {message}",
            "steps_taken":  steps,
            "provider":     self.llm.provider,
            "model":        self.llm.model,
            "step_log":     json.dumps(self.step_log, indent=2)
        }
