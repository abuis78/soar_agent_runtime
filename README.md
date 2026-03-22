# SOAR Agent Runtime

## 1. Overview

SOAR Agent Runtime is a Splunk SOAR custom app implementing an autonomous **ReAct-Loop** (Reason → Act → Observe) agent framework. Instead of a fixed playbook sequence, an agent receives a natural language task and autonomously decides which tools to call until it reaches a `FINAL_ANSWER`.

**Key features:**
- Multi-provider LLM support: Anthropic, OpenAI, Gemini, local (Ollama)
- Agent definitions stored in SOAR Custom Lists — no code deployment needed
- Skills system for reusable prompt injections and structured output schemas
- Full audit trail: every ReAct step logged as Container Note
- Agent chaining: agents can invoke sub-agents as tools

---

## 2. Prerequisites

### Python packages (auto-installed on import)
| Package | Purpose |
|---|---|
| `requests` | HTTP |
| `anthropic` | Claude API |
| `openai` | OpenAI + local LLM |
| `google-generativeai` | Gemini API |

### System dependencies
None.

### Network
- `api.anthropic.com` (Anthropic)
- `api.openai.com` (OpenAI)
- `generativelanguage.googleapis.com` (Gemini)
- Internal URL for local LLM (Ollama etc.)

---

## 3. App Installation

1. **Apps → Install App** → upload `soar_agent_runtime_v1.2.0.tar`
2. **Apps → SOAR Agent Runtime → Configure New Asset**
3. Fill in API keys for the providers you want to use
4. **Test Connectivity**

---

## 4. Asset Configuration

| Field | Description | Default |
|---|---|---|
| `anthropic_api_key` | Anthropic API Key | — |
| `anthropic_api_url` | Anthropic base URL | `https://api.anthropic.com` |
| `openai_api_key` | OpenAI API Key | — |
| `openai_api_url` | OpenAI base URL | `https://api.openai.com/v1` |
| `gemini_api_key` | Google Gemini API Key | — |
| `local_llm_url` | Local LLM URL | `http://localhost:11434/v1` |
| `local_llm_api_key` | Local LLM key | `ollama` |
| `default_provider` | Fallback provider | `anthropic` |
| `default_max_steps` | Max ReAct steps | `25` |
| `agent_list_name` | Custom List for agents | `agent_definitions` |
| `skill_list_name` | Custom List for skills | `agent_skills` |

---

## 5. Test Connectivity

Expected output:
```
anthropic (claude-haiku-4-5-20251001): OK
openai (gpt-4o-mini): SKIPPED (no credentials configured)
gemini: SKIPPED
local: SKIPPED
```

---

## 6. Actions

### `run agent`
Runs a named agent via the ReAct loop.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_id` | string | Yes | Agent name from `agent_definitions` |
| `task` | string | Yes | Natural language task |
| `container_id` | numeric | Yes | SOAR Container ID |
| `max_steps` | numeric | No | Override max steps (0 = use default) |
| `extra_context` | string | No | Additional JSON context |

Output paths:
| Path | Description |
|---|---|
| `action_result.data.*.agent_id` | Agent name |
| `action_result.data.*.final_answer` | Agent conclusion |
| `action_result.data.*.steps_taken` | Number of steps |
| `action_result.data.*.provider` | LLM provider |
| `action_result.data.*.model` | Model name |
| `action_result.data.*.step_log` | Full JSON step trace |

---

### `create agent`
Creates or updates an agent in the `agent_definitions` Custom List.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_id` | string | Yes | Unique name |
| `system_prompt` | string | Yes | Agent role and instructions |
| `provider` | string | Yes | `anthropic`, `openai`, `gemini`, `local` |
| `model` | string | Yes | Model name |
| `allowed_tools` | string | No | Comma-separated tools |
| `max_steps` | numeric | No | Max loop steps |
| `skills` | string | No | Comma-separated skill IDs |

Available tools: `splunk_search`, `add_note`, `update_severity`, `get_container_info`, `run_sub_agent`

---

### `create skill`
Creates or updates a skill in the `agent_skills` Custom List.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `skill_id` | string | Yes | Unique skill name |
| `description` | string | No | Human-readable label |
| `inject` | string | No | Text injected into system prompt |
| `output_schema` | string | No | JSON schema enforced in FINAL_ANSWER |

---

### `list agents` / `list skills`
Returns all entries from the respective Custom List. No parameters.

---

## 7. Skills System

Skills are reusable prompt components stored in the `agent_skills` Custom List.
One skill can be used by many agents. Agents reference skills by ID.

### How it works

When `run agent` executes, the Runtime:
1. Loads the agent definition
2. Reads the `skills` list from the agent config
3. Fetches each skill from `agent_skills` Custom List
4. Injects `inject` text and/or `output_schema` into the system prompt
5. The LLM receives the enriched prompt

### Skill types

**inject skill** — appends instructions to the system prompt:
```json
{
  "skill_id": "german_output",
  "description": "Forces German language output",
  "inject": "Always respond in German. All notes, summaries and FINAL_ANSWER must be written in German."
}
```

**output_schema skill** — forces structured JSON in FINAL_ANSWER:
```json
{
  "skill_id": "structured_triage",
  "description": "Enforces structured triage output",
  "output_schema": {
    "verdict": "TRUE_POSITIVE | FALSE_POSITIVE | NEEDS_ESCALATION",
    "confidence": "LOW | MEDIUM | HIGH",
    "evidence": ["string"],
    "recommended_action": "string",
    "mitre_techniques": ["T1234"]
  }
}
```

**Combined skill** — inject + schema:
```json
{
  "skill_id": "bsi_grundschutz",
  "description": "Maps findings to BSI IT-Grundschutz",
  "inject": "Map all findings to BSI IT-Grundschutz controls. Reference the relevant Baustein and Anforderung IDs.",
  "output_schema": {
    "bsi_controls": [{"baustein": "string", "anforderung": "string", "status": "string"}]
  }
}
```

### Predefined skills (create via `create skill` action)

| skill_id | Type | Description |
|---|---|---|
| `german_output` | inject | Forces German language |
| `mitre_mapping` | inject | MITRE ATT&CK references |
| `structured_triage` | schema | Triage JSON output |
| `structured_ioc` | schema | IOC enrichment JSON |
| `structured_ueba` | schema | UEBA risk assessment JSON |
| `bsi_grundschutz` | inject+schema | BSI IT-Grundschutz mapping |
| `ciso_report` | inject | Executive language, no jargon |

---

## 8. SOC Use Cases & Agent Prompts

### UC-1: Alert Triage (Tier-1)

**Recommended skills:** `structured_triage`, `mitre_mapping`

```
create agent:
  agent_id:      triage_agent
  provider:      anthropic
  model:         claude-haiku-4-5-20251001
  allowed_tools: get_container_info,splunk_search,add_note,update_severity
  max_steps:     15
  skills:        structured_triage,mitre_mapping
  system_prompt: |
    You are a SOC Tier-1 triage analyst. Your job is to assess whether an
    alert is a true positive, false positive, or requires escalation.
    For every alert:
    (1) Get container info and read all artifacts.
    (2) Search Splunk for related events (same source IP, user, host) in the last 24h.
    (3) Assess based on frequency, context, and known patterns.
    (4) Set severity accordingly.
    (5) Add a structured triage note.
    Be concise and factual. Never guess — only state what the data shows.

run agent:
  agent_id:     triage_agent
  container_id: {{ container.id }}
  task: |
    Triage this alert. Determine if it is a true positive or false positive.
    If true positive: set severity to high or critical, recommend escalation.
    If false positive: set severity to low, document why.
```

---

### UC-2: IOC Enrichment

**Recommended skills:** `structured_ioc`, `mitre_mapping`

```
create agent:
  agent_id:      ioc_enrichment_agent
  provider:      openai
  model:         gpt-4o-mini
  allowed_tools: get_container_info,splunk_search,add_note
  max_steps:     20
  skills:        structured_ioc,mitre_mapping
  system_prompt: |
    You are an IOC enrichment specialist. For each IOC (IP, domain, hash, URL)
    found in the container artifacts:
    (1) Search Splunk for historical occurrences across all indexes.
    (2) Check for related IOCs (same subnet, similar domains, related hashes).
    (3) Determine first seen / last seen.
    (4) Assess prevalence: isolated vs. widespread.
    (5) Add an enrichment note per IOC: Type, Value, First/Last Seen,
        Hit Count, Related IOCs, Risk Assessment.
    Format each note consistently for downstream automation.

run agent:
  agent_id:     ioc_enrichment_agent
  container_id: {{ container.id }}
  task: |
    Enrich all IOCs found in this container's artifacts.
    Search Splunk across all indexes for the last 30 days.
    Document findings as structured notes.
```

---

### UC-3: User Behaviour Investigation (UEBA)

**Recommended skills:** `structured_ueba`, `mitre_mapping`

```
create agent:
  agent_id:      ueba_agent
  provider:      anthropic
  model:         claude-sonnet-4-5
  allowed_tools: get_container_info,splunk_search,add_note,update_severity
  max_steps:     25
  skills:        structured_ueba,mitre_mapping
  system_prompt: |
    You are a user behaviour analyst. When given a username or user-related alert:
    (1) Get user activity from Splunk: logins, failed logins, VPN, email,
        file access, privileged commands in the last 7 days.
    (2) Identify anomalies: unusual hours, new locations, first-time access
        to sensitive systems, bulk data access, lateral movement indicators.
    (3) Build a timeline of significant events.
    (4) Score risk: Low / Medium / High / Critical with justification.
    (5) Add a UEBA summary note with timeline and risk score.
    Use MITRE ATT&CK technique references where applicable.

run agent:
  agent_id:     ueba_agent
  container_id: {{ container.id }}
  task: |
    Investigate user {{ artifact.cef.userName }} for suspicious behaviour.
    Build a 7-day activity timeline and assess insider threat risk.
```

---

### UC-4: Phishing Email Analysis

**Recommended skills:** `structured_triage`, `mitre_mapping`

```
create agent:
  agent_id:      phishing_agent
  provider:      anthropic
  model:         claude-sonnet-4-5
  allowed_tools: get_container_info,splunk_search,add_note,update_severity
  max_steps:     20
  skills:        structured_triage,mitre_mapping
  system_prompt: |
    You are a phishing analysis specialist. For a reported phishing email:
    (1) Extract all IOCs from artifacts: sender, reply-to, URLs, attachments, IPs.
    (2) Search Splunk for: other recipients of the same email, URL clicks,
        file executions from attachments, C2 connections from affected hosts.
    (3) Determine blast radius: how many users received / opened / clicked.
    (4) Classify: Credential Harvesting / Malware Delivery / BEC /
        Spear-Phishing / Mass Phishing.
    (5) Add analysis note with: Classification, IOCs, Blast Radius,
        Affected Users, Recommended Containment Actions.

run agent:
  agent_id:     phishing_agent
  container_id: {{ container.id }}
  task: |
    Analyze this phishing report. Determine the blast radius,
    classify the attack type, and recommend containment steps.
```

---

### UC-5: Lateral Movement Detection

**Recommended skills:** `mitre_mapping`

```
create agent:
  agent_id:      lateral_movement_agent
  provider:      openai
  model:         gpt-4o
  allowed_tools: get_container_info,splunk_search,add_note,update_severity
  max_steps:     30
  skills:        mitre_mapping
  system_prompt: |
    You are a threat hunter specializing in lateral movement detection.
    Given a potentially compromised host or user:
    (1) Search Splunk for authentication events from this source to other
        internal hosts (WMI, SMB, RDP, PSExec, SSH) in the last 48h.
    (2) Map the movement graph: Source → Target chains.
    (3) Check for credential dumping (lsass access, SAM database reads).
    (4) Check for new service installations, scheduled tasks, or registry
        persistence on target hosts.
    (5) Build an attack path narrative.
    (6) Set severity to critical if lateral movement confirmed.
    (7) Add note with: Attack Path, MITRE ATT&CK Techniques,
        Affected Hosts, Recommended Isolation Targets.

run agent:
  agent_id:     lateral_movement_agent
  container_id: {{ container.id }}
  task: |
    Investigate potential lateral movement from host {{ artifact.cef.sourceAddress }}.
    Map the full attack path and identify all affected systems.
```

---

### UC-6: Incident Summary (Management Report)

**Recommended skills:** `ciso_report`

```
create agent:
  agent_id:      report_agent
  provider:      anthropic
  model:         claude-sonnet-4-5
  allowed_tools: get_container_info,splunk_search,add_note
  max_steps:     15
  skills:        ciso_report
  system_prompt: |
    You are a senior SOC analyst writing executive incident summaries.
    Your output must be professional, concise, and free of technical jargon.
    For any incident:
    (1) Get all container notes and artifacts.
    (2) Synthesize the incident timeline.
    (3) Write a structured summary:
        - Executive Summary (3 sentences max)
        - Timeline
        - Impact Assessment
        - Root Cause (if known)
        - Actions Taken
        - Recommended Next Steps
    (4) Add note titled 'Incident Summary — [date]'.
    Write for a CISO audience: business impact first, technical detail second.

run agent:
  agent_id:     report_agent
  container_id: {{ container.id }}
  task: |
    Write a management-ready incident summary for this container.
    Synthesize all existing notes into a structured report.
    Include timeline, impact, and recommended next steps.
```

---

### UC-7: Multi-Agent Orchestration (Chaining)

```
create agent:
  agent_id:      orchestrator_agent
  provider:      anthropic
  model:         claude-sonnet-4-5
  allowed_tools: get_container_info,run_sub_agent,add_note,update_severity
  max_steps:     10
  system_prompt: |
    You are a SOC orchestrator. You do not investigate directly.
    Instead, you delegate tasks to specialist agents:
    - ioc_enrichment_agent  — for IOC lookups and enrichment
    - ueba_agent            — for user behaviour analysis
    - lateral_movement_agent — for host-to-host movement
    - report_agent          — for final summary generation
    Assess the alert type, invoke the appropriate specialist agents
    in the right order, then request a final report.

run agent:
  agent_id:     orchestrator_agent
  container_id: {{ container.id }}
  task: |
    Fully investigate this incident end-to-end.
    Delegate to the appropriate specialist agents based on alert type.
    Finish with a management report.
```

---

## 9. Widget Display

The `run agent` action includes a custom Investigation panel widget showing agent ID, provider, steps taken, final answer, and full step-by-step log.

**To activate:**
1. Open an Investigation
2. **Manage Widgets** → **SOAR Agent Runtime** → toggle **On**
3. **Save Layout**

---

## 10. Local Testing

```json
// /tmp/test_run_agent.json
{
  "identifier": "run_agent",
  "asset_id": "1",
  "parameters": [{
    "agent_id": "triage_agent",
    "task": "Triage this alert and determine severity.",
    "container_id": 1,
    "max_steps": 3
  }],
  "config": {
    "anthropic_api_key": "YOUR_KEY",
    "default_provider": "anthropic",
    "default_max_steps": 25,
    "agent_list_name": "agent_definitions",
    "skill_list_name": "agent_skills"
  }
}
```

```bash
cd /opt/phantom/apps/soar_agent_runtime_*/
phenv python3 soar_agent_runtime_connector.py /tmp/test_run_agent.json
```

---

## 11. Troubleshooting

| Error | Fix |
|---|---|
| `Agent 'X' not found` | Create Custom List `agent_definitions` + add agent via `create agent` |
| `Skill 'X' not found` | Create Custom List `agent_skills` + add skill via `create skill` |
| `Missing required JSON field(s): main_module` | Ensure `main_module` field exists in app JSON |
| `anthropic not installed` | `phenv pip3 install anthropic` |
| `openai not installed` | `phenv pip3 install openai` |
| `Connection refused` (local) | Check Ollama: `systemctl status ollama` and `ollama pull llama3` |
| `Max steps reached` | Increase `max_steps` in agent definition or asset config |
| Widget not showing | Activate via **Manage Widgets** in Investigation panel |

---

## 12. Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-03-22 | Initial release |
| 1.0.1 | 2026-03-22 | Fix: added `main_module` to app JSON |
| 1.2.0 | 2026-03-22 | Skills system (Option B): `SkillStore`, `create skill`, `list skills` actions; 7 SOC use case prompts in README |
| 1.3.0 | 2026-03-22 | Python 3.13 update: `match` statements, `X\|Y` type hints, walrus operator, list unpacking patterns; min_phantom_version 6.3.0 |
