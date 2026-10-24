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

0. Build the package from a checkout of `main`: `python3 tools/build_app.py` writes `soar_agent_runtime_v<app_version>.tar` next to app.json (`--dry-run` lists the files, `--out DIR` picks the directory). The tar carries all bundled wheels, so the SOAR host needs no internet access (SOAR ≥ 7.0.0 with Python 3.13 on Linux x86_64).
1. **Apps → Install App** → upload the tar from step 0
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

Policy Manager (effecting actions in `airgap_mode`, RT-W01/RT-W03; the air-gap and Tool Broker fields are documented with sub-change 5.5):

| Field | Description | Default |
|---|---|---|
| `policy_mode` | `embedded` (the bundled `policy_client` runs in-process) or `service` (Policy Manager service over mTLS). Empty = every effecting action is denied in `airgap_mode` | — |
| `policy_path` | Path on the SOAR host to the policy file (JSON/TOML from Git); `embedded` only | — |
| `policy_state_dir` | Writable directory for the embedded state: `approvals.json`, `halt.json`, `audit.jsonl`; `embedded` only | — |
| `policy_service_url` | Base URL of the Policy Manager service, `https://` only; `service` only | — |
| `policy_timeout_s` | Timeout per service call; on timeout the action is denied, there is no fallback to `embedded` (RT-W04) | `5` |
| `policy_client_cert` | Path to the client certificate (PEM); its CN is the runtime's identity at the Policy Manager | — |
| `policy_client_key` | Path to the private key (PEM); the key never leaves the host | — |
| `policy_ca` | Path to the CA bundle (PEM) that signed the service certificate | — |

The bundled wheel `wheels/py3/policy_manager-0.5.1-py3-none-any.whl` is Policy Manager 0.5.1, built from the PolicyManager tag `rp/p5-3` (commit `fbea672`, ahead of the PolicyManager pull requests by decision of 2026-10-23): `embedded`, `service` and the action catalogue work with it. Refresh it with `scripts/build_wheel.sh --copy-to wheels/py3` from the PolicyManager repo, then `python3 tools/update_wheels.py`.


### Operator checklist: Policy Manager

Every item below was learned against the real engine (0.5.1); missing any of them means denied actions, not errors.

1. **Halt file.** `policy_state_dir` must contain `halt.json` with `{"angehalten": false, "grund": ""}`, owned by the app user, mode `0600`. A missing, unreadable or world-writable file counts as halted; `test connectivity` then reports `halt file missing (create halt.json ...)`.
2. **Valid policy.** The engine validates the whole file on every decision; one error (for example a building block without `loud`, or an `undo` that names an unknown block) means `POLICY_INVALID` and every action is denied. Validate changes with the PolicyManager repo's `policy-check` before copying the file to the host.
3. **Tool name mapped.** The runtime sends `aktion.art = "update_severity"`. The policy needs `update_severity = "fall_aenderung"` under `[tool_action_kinds]` (PolicyManager example policy since 0.5.1); otherwise every severity change falls to `default_gate`.
4. **Stage and domain from the playbook.** Pass `stufe` (0-4) and `domaene` (for example `cyber`) to `run agent`. Without `stufe` the engine uses `default_gate`, without `domaene` at least `APPROVE_1CLICK` (RT-W08).
5. **Released building blocks.** A block is offered to an agent only when its `agents` list names the `agent_id`; `mindeststufe` above the run's `stufe` hides it. The runtime never executes a block; the follow-up playbook does (artifact `agent_action_proposal`).
6. **Service mode.** `policy_service_url` is `https://`, certificate, key and CA are files readable by the app user, the certificate's CN has the role `decide` at the service, `policy_timeout_s` fits the network. `test connectivity` must say `policy service healthy, not halted`; anything else denies every effecting action (no fallback, RT-W04).
7. **Sequence register.** `sequences.json` in `policy_state_dir` is written by the Policy Manager's register workflow; without it every multi-step sequence counts as new (at least `APPROVE_1CLICK`).

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
| `stufe` | numeric | No | Escalation stage 0-4 of the triggering playbook, sent with every policy request and used to filter the action catalogue (RT-W08, E12); empty = unknown = 0 |
| `domaene` | string | No | Domain of this run (e.g. `soc`, `it`, `ot`), sent with every policy request (RT-W08) |

Output paths:
| Path | Description |
|---|---|
| `action_result.data.*.agent_id` | Agent name |
| `action_result.data.*.final_answer` | Agent conclusion |
| `action_result.data.*.steps_taken` | Number of steps |
| `action_result.data.*.provider` | LLM provider |
| `action_result.data.*.model` | Model name |
| `action_result.data.*.step_log` | Full JSON step trace |

### Action catalogue and gap report (`airgap_mode`)

In `airgap_mode` every effecting tool call (`update_severity`, `catalog_invoke`) must carry a complete judgement
before the Policy Manager is asked: `begruendung` (text), `gegenargumente` and `unsicherheiten` (text or list of
strings, `"keine erkannt"` when there is nothing to say), `konfidenz` (0..1) and `beleg_ids` (ids that originated in
this run). Incomplete calls are refused locally and never reach the Policy Manager (RT-W05, AC-08).

**`catalog_invoke(schritte=[{baustein, ziel, parameter}, ...])`** proposes one or more building blocks from the
Policy Manager's action catalogue (`GET /v1/aktionen`, RT-W09). The tool appears as soon as the catalogue lists
blocks for the agent; the blocks are the allowlist, nothing is configured per agent (E14). Blocks whose
`mindeststufe` is above the run's `stufe` are shown as unavailable and refused locally (E12). A sequence of several
steps is decided as **one** request (`aktion.art = "folge"`, AC-09). Whatever the decision, the runtime never
executes a building block itself:

| Decision | What the run records | Who executes |
|---|---|---|
| `ALLOW` | artifact `agent_action_proposal` with `gate = ALLOW`, no `freigabe_id`, plus note "Freigegeben (ALLOW): ..." | the follow-up playbook, without `consume` (E13) |
| `APPROVE_*` | artifact `agent_action_proposal` with `freigabe_id`, plus note "Freigabe ausstehend: ..." | the follow-up playbook after `consume(freigabe_id)` (RT-W07) |
| `DENY` | nothing; the observation names the code and the step (`BLOCK_NOT_ALLOWED`, `INVALID_SEQUENCE`, ...) | nobody |

**`report_gap(lage, massnahme, ziele, warum_kein_baustein)`** is the way out when a measure is needed and no
released block fits (RT-W10). It exists only in `airgap_mode` with a configured `policy_mode`, has no effect and
makes no policy request: it writes the artifact `agent_action_gap` (CEF `lauf_id`, `schritt_id`, `agent_id`,
`stufe`, `domaene`, `lage`, `massnahme`, `ziele`, `grund`, `katalog_sichtbar`, `katalog_verborgen`) and the note
"Manuelle Handlungsempfehlung: ..." for an analyst; the catalogue curators read the artifacts (E15).

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

Unit and integration tests run with `uv` and the bundled wheels (see the command in `test_soar_agent_runtime_policy_wheel.py`). The mTLS integration tests in `test_soar_agent_runtime_policy_service.py` start the Policy Manager service from the bundled wheel in-process with a throwaway PKI made by the `openssl` CLI; without `openssl` they are skipped.

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
| 1.7.1 | 2026-10-22 | Policy Manager `embedded`: bundled `policy_manager` 0.1.0 wheel (`wheels/py3/`), so `policy_mode = embedded` decides effecting actions in `airgap_mode` (RT-W01, RT-W03) |
| 1.8.0 | 2026-10-23 | Phase 4: `policy_mode = service` over mTLS via the package's `ServicePolicyClient`, no fallback (RT-W01, RT-W04; asset fields `policy_service_url`, `policy_timeout_s`, `policy_client_cert`, `policy_client_key`, `policy_ca`); action catalogue with `catalog_invoke` (one decision per sequence, stage filter, proposal artifact also for `ALLOW`; RT-W09, AC-09, E12-E14); `report_gap` with artifact `agent_action_gap` (RT-W10, E15); judgement sent as `urteil = {empfehlung, gegenargumente[], unsicherheiten[], konfidenz}` and `unsicherheiten`/`konfidenz` mandatory for effecting calls (Policy Manager API.md §3, W14). Service mode and catalogue need a Policy Manager P3/P4 wheel (4.1b/4.2b) |
| 1.8.1 | 2026-10-23 | Fix: Policy Manager `health()` reads TOML policies (before: every `.toml` policy was reported as "not valid JSON" and `test connectivity` failed); an undeterminable halt state (service unreachable, halt file missing/unreadable) is reported as unhealthy with its cause and a hint to create `halt.json`, a deliberate halt stays healthy + HALTED. Found in the 4.1b spike against `policy_manager` 0.5.0 |
| 1.9.0 | 2026-10-23 | Bundled `policy_manager` 0.5.1 (built from PolicyManager tag `rp/p5-3`, commit `fbea672`; E16: tag instead of PolicyManager `main`): `policy_mode = service`, action catalogue, halt field `bekannt` and the sequence register work with the shipped package; `sequences_path` passed to the embedded client; `health()` decides "undetermined" from `bekannt` only; wheel integration tests follow the 0.5.x engine (tool-name mapping, `stufe`/`domaene`, judgement W14, catalogue, sequences, audit); README operator checklist for the Policy Manager (4.1b) |
