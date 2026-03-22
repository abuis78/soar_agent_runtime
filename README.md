# SOAR Agent Runtime

## 1. Overview

SOAR Agent Runtime is a Splunk SOAR custom app that implements an autonomous **ReAct-Loop** (Reason → Act → Observe) agent framework directly within SOAR playbooks.

Instead of a fixed sequence of playbook actions, an agent receives a natural language task and autonomously decides which tools to call, in which order, until it reaches a conclusion (`FINAL_ANSWER`). This enables dynamic, context-aware SOC automation without hard-coded playbook logic.

**Use cases:**
- Autonomous triage of alerts (SPL queries → severity assessment → note)
- IOC enrichment chains driven by LLM reasoning
- Multi-step investigation workflows controlled by a single playbook action
- Multi-agent orchestration (agents calling sub-agents)

**Supported LLM providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini), Local (Ollama / OpenAI-compatible)

---

## 2. Prerequisites

### Python packages (auto-installed by SOAR)
| Package | Purpose |
|---|---|
| `requests` | HTTP calls |
| `anthropic` | Anthropic Claude API |
| `openai` | OpenAI + local LLM (OpenAI-compatible) |
| `google-generativeai` | Google Gemini API |

### System dependencies
None. All dependencies are pure-Python.

### Network requirements
- Outbound HTTPS to `api.anthropic.com` (Anthropic)
- Outbound HTTPS to `api.openai.com` (OpenAI)
- Outbound HTTPS to `generativelanguage.googleapis.com` (Gemini)
- Internal network access to local LLM endpoint (if using local provider)

---

## 3. App Installation

1. Navigate to **Apps → Install App**
2. Upload `soar_agent_runtime_v1.0.0.tar`
3. Click **Install**
4. Navigate to **Apps → SOAR Agent Runtime → Configure New Asset**
5. Fill in API keys for the providers you want to use (others can be left empty)
6. Click **Test Connectivity** to verify

---

## 4. Asset Configuration

| Field | Description | Required |
|---|---|---|
| `anthropic_api_key` | Anthropic API Key | Only if using Claude |
| `anthropic_api_url` | Anthropic API URL (default: `https://api.anthropic.com`) | No |
| `openai_api_key` | OpenAI API Key | Only if using GPT |
| `openai_api_url` | OpenAI API URL (default: `https://api.openai.com/v1`) | No |
| `gemini_api_key` | Google Gemini API Key | Only if using Gemini |
| `local_llm_url` | Local LLM base URL (e.g. `http://localhost:11434/v1`) | Only if using local |
| `local_llm_api_key` | Local LLM API key (use `ollama` for Ollama) | No |
| `default_provider` | Default provider: `anthropic`, `openai`, `gemini`, `local` | No (default: `anthropic`) |
| `default_max_steps` | Default max ReAct loop steps | No (default: `25`) |
| `agent_list_name` | Custom List name for agent definitions | No (default: `agent_definitions`) |

---

## 5. Test Connectivity

**Expected success output:**
```
Testing anthropic... OK
anthropic (claude-haiku-4-5-20251001): OK
Testing openai... OK
openai (gpt-4o-mini): OK
gemini: skipped (no credentials)
local: skipped (no credentials)
Connectivity test completed.
```

**Common errors:**
- `AuthenticationError` → Check API key in asset configuration
- `Connection refused` → Check local LLM URL and that Ollama is running
- `404` on local → Verify Ollama model is pulled: `ollama pull llama3`

---

## 6. Actions

### `run agent`

Executes a named agent using the ReAct loop.

**Parameters:**
| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_id` | string | Yes | Agent name as defined in `agent_definitions` Custom List |
| `task` | string | Yes | Natural language task description |
| `container_id` | numeric | Yes | SOAR Container ID the agent operates on |
| `max_steps` | numeric | No | Override max steps (0 = use app default) |
| `extra_context` | string | No | Additional JSON context string for the agent |

**Output data paths:**
| Path | Type | Description |
|---|---|---|
| `action_result.data.*.agent_id` | string | Agent identifier |
| `action_result.data.*.final_answer` | string | Agent's final conclusion |
| `action_result.data.*.steps_taken` | numeric | Number of ReAct loop steps |
| `action_result.data.*.provider` | string | LLM provider used |
| `action_result.data.*.model` | string | Model name used |
| `action_result.data.*.step_log` | string | JSON log of all steps |

---

### `create agent`

Creates or updates an agent definition in the `agent_definitions` Custom List.

**Parameters:**
| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_id` | string | Yes | Unique agent identifier |
| `system_prompt` | string | Yes | Agent's system prompt (role + instructions) |
| `provider` | string | Yes | `anthropic`, `openai`, `gemini`, or `local` |
| `model` | string | Yes | Model name |
| `allowed_tools` | string | No | Comma-separated tools (default: `splunk_search,add_note`) |
| `max_steps` | numeric | No | Max steps for this agent (0 = app default) |

**Available tools:**
| Tool | Description |
|---|---|
| `splunk_search` | Execute SPL query, return results |
| `add_note` | Add note to SOAR container |
| `update_severity` | Update container severity |
| `get_container_info` | Get container metadata and artifacts |
| `run_sub_agent` | Invoke another agent as sub-task |

---

### `list agents`

Lists all agent definitions from the Custom List.

**Output:** One row per agent with: `agent_id`, `provider`, `model`, `allowed_tools`, `max_steps`

---

## 7. Widget Display

The `run agent` action includes a custom Investigation panel widget showing:
- Agent ID, provider, model, step count
- Final Answer
- Full step-by-step log (Thought → Action → Observation)

**To activate:**
1. Open an Investigation
2. Click **Manage Widgets** (top right)
3. Find **SOAR Agent Runtime** → toggle **On**
4. Click **Save Layout**

---

## 8. Creating Your First Agent

**Step 1:** Create the `agent_definitions` Custom List in SOAR:
- Navigate to **Custom Lists → New List**
- Name: `agent_definitions`
- Columns: `agent_id`, `config`

**Step 2:** Create an agent via playbook action `create agent`:
```
agent_id:      triage_agent
provider:      anthropic
model:         claude-haiku-4-5-20251001
allowed_tools: splunk_search,add_note,update_severity,get_container_info
system_prompt: You are a SOC Tier-1 analyst. Analyze the given alert, search for related events in Splunk, assess the severity, add a summary note, and provide a FINAL_ANSWER with your triage decision.
```

**Step 3:** Run the agent from a playbook:
```
action:       run agent
agent_id:     triage_agent
task:         Triage this alert and determine if it is a true positive.
container_id: {{ container.id }}
```

---

## 9. Local Testing

Test JSON file `/tmp/test_run_agent.json`:
```json
{
  "identifier": "run_agent",
  "asset_id": "1",
  "parameters": [{
    "agent_id": "triage_agent",
    "task": "Analyze this alert and determine severity.",
    "container_id": 1,
    "max_steps": 3
  }],
  "config": {
    "anthropic_api_key": "YOUR_KEY_HERE",
    "default_provider": "anthropic",
    "default_max_steps": 25,
    "agent_list_name": "agent_definitions"
  }
}
```

Run on SOAR appliance:
```bash
cd /opt/phantom/apps/soar_agent_runtime_*/
phenv python3 soar_agent_runtime_connector.py /tmp/test_run_agent.json
```

---

## 10. Troubleshooting

| Error | Fix |
|---|---|
| `Agent 'X' not found in Custom List` | Create the Custom List `agent_definitions` and add agent via `create agent` action |
| `anthropic package not installed` | `phenv pip3 install anthropic` |
| `openai package not installed` | `phenv pip3 install openai` |
| `Connection refused` on local LLM | Verify Ollama is running: `systemctl status ollama` |
| `Max steps reached` | Increase `max_steps` in agent definition or asset config |
| Widget not showing | Activate via **Manage Widgets** in Investigation panel |
| `phantom.rules not available` | App is running outside SOAR — fallback mode active (dev only) |

---

## 11. Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-03-22 | Initial release. ReAct loop, 4 providers, 5 tools, 4 actions, custom widget |
