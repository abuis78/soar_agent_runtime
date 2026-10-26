# SOAR Agent Runtime

## 1. Overview

SOAR Agent Runtime is a Splunk SOAR custom app implementing an autonomous **ReAct-Loop** (Reason → Act → Observe) agent framework. Instead of a fixed playbook sequence, an agent receives a natural language task and decides which tools to call until it reaches a `FINAL_ANSWER`.

The app runs in two modes:

| | Normal mode | Air-gap mode (`airgap_mode`, chapter 5) |
|---|---|---|
| Models | Anthropic, OpenAI, Gemini or a local server (Ollama) | only the local server; cloud providers are refused before any call |
| Reading data | the run's own container (`get_container_info`) | the own container plus read-only Tool Broker templates (`broker_invoke`) |
| Taking effect | tools act directly (`update_severity`, `add_note`) | every effecting call needs a judgement with evidence and an `ALLOW` from the Policy Manager; catalogue measures are only proposed, a follow-up playbook executes them |
| Audit | optional (`audit_enabled`) | one artifact `agent_audit` per step, fail-closed |
| FINAL_ANSWER | free text, or the keys of a skill's `output_schema` | one JSON judgement with six mandatory fields, checked |

**Key features:**
- Agent and skill definitions in SOAR Custom Lists, or versioned as TOML files from Git (`definitions_path`, repository [soar-agent-definitions](https://github.com/abuis78/soar-agent-definitions)); the version of every definition goes into the audit
- Skills: reusable prompt rules and required output keys
- One `lauf_id` per run; every step has a `schritt_id` that later steps and the FINAL_ANSWER cite as evidence
- Audit artifact per step (`agent_audit`) and a summary note per run that lists the actions that really took effect
- Emergency stop: in air-gap mode the Policy Manager's halt state is checked before every step
- Sub-agents: `run_sub_agent` runs another agent in the same run, bounded by depth and the step budget
- Self-checks: `setup app`, `test connectivity`, `list tools`, `test tools`

---

## 2. Prerequisites

### Python packages (bundled)
The package carries every wheel for SOAR's Python 3.13 under `wheels/` and SOAR installs them offline. Besides the LLM SDKs (`anthropic`, `openai`, `google-generativeai`) and `requests`, two local wheels are bundled: `ai_soc_telemetry` (optional HEC telemetry) and `policy_manager` (the Policy Manager client, chapter 4).

### System dependencies
None on the SOAR host. For tests on a workstation: `uv` and, for the mTLS tests, the `openssl` CLI (chapter 11).

### Network
| Mode | Outgoing connections from the SOAR host |
|---|---|
| Normal | the configured cloud APIs (`api.anthropic.com`, `api.openai.com`, `generativelanguage.googleapis.com`) and/or the local LLM URL |
| Air-gap | only internal services: the local LLM (`local_llm_url`, e.g. Ollama on port 11434), the Tool Broker (`broker_url`, https with mTLS) and, with `policy_mode = service`, the Policy Manager (`policy_service_url`, https with mTLS). No internet. |

Every run also calls the SOAR REST API of its own instance (Custom Lists, container, artifacts), authenticated by the app framework as the user who runs the action.

---

## 3. App Installation

0. Build the package from a checkout of `main`: `python3 tools/build_app.py` writes `soar_agent_runtime_v<app_version>.tar` next to app.json (`--dry-run` lists the files, `--out DIR` picks the directory). The tar carries all bundled wheels, so the SOAR host needs no internet access (SOAR ≥ 7.0.0 with Python 3.13 on Linux x86_64).
1. **Apps → Install App** → upload the tar from step 0
2. **Apps → SOAR Agent Runtime → Configure New Asset**
3. Fill in the fields for the mode you want (chapter 4); for air-gap mode follow the checklist in chapter 5
4. **Test Connectivity**
5. Run the action **setup app** once: it creates the Custom Lists `agent_definitions` and `agent_skills` if they are missing and checks the asset (model, and in `airgap_mode` broker and policy fields, files and `halt.json`). `dry_run` only reports.
6. Run **test tools** on a container made for testing: it calls every built-in tool for real (one note, severity changed and restored). Set `model_check` to also see whether the model keeps one ACTION per response.
7. Before defining an agent, run **list tools** for the valid names in `allowed_tools`.

**Routine after every install, upgrade or asset change:** `setup app`, then `test tools` on the same test container. Both can run from a daily playbook so a SOAR update that changes its API shows up the next morning, not in an incident.

---

## 4. Asset Configuration

**LLM providers**

| Field | Description | Default |
|---|---|---|
| `anthropic_api_key` | Anthropic API Key | — |
| `anthropic_api_url` | Anthropic base URL | `https://api.anthropic.com` |
| `openai_api_key` | OpenAI API Key | — |
| `openai_api_url` | OpenAI base URL | `https://api.openai.com/v1` |
| `gemini_api_key` | Google Gemini API Key | — |
| `local_llm_url` | Local LLM base URL (Ollama: `http://<host>:11434/v1`) | `http://localhost:11434/v1` |
| `local_llm_api_key` | Local LLM key (`ollama` for Ollama, or empty) | — |
| `local_llm_model` | Local model when the agent definition names none; also used by `test connectivity` (Ollama tag, e.g. `qwen2.5:7b`) | `llama3` |
| `default_provider` | Provider when the agent definition names none (`anthropic`, `openai`, `gemini`, `local`) | `anthropic` |
| `llm_timeout_s` | Timeout in seconds for one model call (RT-A03) | `120` |
| `llm_temperature` | Sampling temperature; `0` = as deterministic as the model allows | `0` |
| `llm_max_tokens` | Maximum output tokens per model call | `4096` |
| `llm_seed` | Optional sampling seed, sent only when set and supported by the server | — |

**Agents and definitions**

| Field | Description | Default |
|---|---|---|
| `default_max_steps` | Step limit when the definition and the action name none (hard cap 50) | `25` |
| `agent_list_name` | Custom List for agents | `agent_definitions` |
| `skill_list_name` | Custom List for skills | `agent_skills` |
| `definitions_path` | Directory on the SOAR host with a checkout of [soar-agent-definitions](https://github.com/abuis78/soar-agent-definitions) (`agents/<agent_id>.toml`, `skills/<skill_id>.toml`). Set = `run agent`, `list agents` and `list skills` read the TOML files instead of the Custom Lists, and the audit field `definition_version` becomes `<id>@<version>#<sha256 prefix>` for the agent and each skill (RT-B02). `create agent` / `create skill` are refused while it is set (E5) | — |
| `max_subagent_depth` | Maximum nesting of `run_sub_agent`: the started agent has depth 0, its sub-agents 1, theirs 2; deeper is refused (RT-B03, AC-14). `0` = no sub-agents | `2` |
| `max_observation_chars` | Maximum characters of one tool observation, the `extra_context` block or a container payload passed to the model; longer input is truncated and marked (RT-B04) | `8000` |

**Air-gap mode and audit** (chapter 5)

| Field | Description | Default |
|---|---|---|
| `airgap_mode` | Only the `local` provider; cloud providers are refused at run start, `splunk_search` is removed, reading goes through the Tool Broker and every effect through the Policy Manager (RT-A01) | off |
| `audit_enabled` | One audit artifact `agent_audit` per step outside air-gap mode (RT-P02); always on and fail-closed in `airgap_mode` | off |

**Tool Broker** (read-only templates, RT-L07)

| Field | Description | Default |
|---|---|---|
| `broker_url` | Tool Broker base URL, `https://` only (mTLS); required in `airgap_mode` | — |
| `broker_timeout_s` | Timeout in seconds for every broker call | `30` |
| `broker_client_cert` | Path on the SOAR host to the client certificate (PEM) | — |
| `broker_client_key` | Path on the SOAR host to its private key (PEM); the key never leaves the host | — |
| `broker_ca` | Path on the SOAR host to the CA bundle (PEM) that signed the broker's server certificate | — |

**Policy Manager** (effecting actions in `airgap_mode`, RT-W01/RT-W03)

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

**Telemetry** (optional, AI-SOC Telemetry; never changes agent behaviour)

| Field | Description | Default |
|---|---|---|
| `telemetry_enabled` | Send `agent_action` events to Splunk HEC | off |
| `hec_url` | HEC event endpoint, e.g. `https://splunk.example.com:8088/services/collector/event` | — |
| `hec_token` | HEC token, scoped to the telemetry index | — |
| `telemetry_index` | Splunk index for the events | `ai_soc_audit` |
| `telemetry_verify_tls` | Verify the HEC certificate; disable only for a self-signed test instance | on |

The bundled wheel `wheels/py3/policy_manager-0.5.2-py3-none-any.whl` is Policy Manager 0.5.2, the release asset of PolicyManager tag `v0.5.2` (commit `c9e5982`, SHA-256 `b710eda193d9f2b6c58bc33e6eb267bfae0e05e2445f070f7a86327f2ab92f8e`): `embedded`, `service` and the action catalogue work with it. Breaking changes against 0.5.1 are listed in the PolicyManager [MIGRATION.md](https://github.com/abuis78/PolicyManager/blob/v0.5.2/docs/MIGRATION.md). Refresh it with `scripts/build_wheel.sh --copy-to wheels/py3` from the PolicyManager repo, then `python3 tools/update_wheels.py`.


### Operator checklist: Policy Manager

Every item below was learned against the real engine (0.5.1, checked against 0.5.2); missing any of them means denied actions, not errors.

0. **Emergency stop before every step.** In `airgap_mode` the runtime asks the Policy Manager for the halt state before every step, also before the first model call (RT-B01). Halted or undeterminable stops the run at once with `run halted by Policy Manager (<reason>)` and an audit event `angehalten`. Without a working gate (no `policy_mode`, unreachable service, missing files) the state is unknown, so airgap runs stop before step 1.
1. **Halt file.** `policy_state_dir` must contain `halt.json` with `{"angehalten": false, "grund": ""}`, owned by the app user (or root), mode `0600`; the directory likewise, not writable by group or other (0.5.2, F13). A missing or unreadable file, a symlink or FIFO, or group/other write permission on the file or its directory counts as halted; `test connectivity` then reports `halt file missing (create halt.json ...)`.
2. **Valid policy.** The engine validates the whole file on every decision; one error (for example a building block without `loud`, or an `undo` that names an unknown block) means `POLICY_INVALID` and every action is denied. Validate changes with the PolicyManager repo's `policy-check` before copying the file to the host.
3. **Tool name mapped.** The runtime sends `aktion.art = "update_severity"`. The policy needs `update_severity = "fall_aenderung"` under `[tool_action_kinds]` (PolicyManager example policy since 0.5.1); otherwise every severity change falls to `default_gate`.
4. **Stage and domain from the playbook.** Pass `stufe` (0-4) and `domaene` (for example `cyber`) to `run agent`. Without `stufe` the engine uses `default_gate`, without `domaene` at least `APPROVE_1CLICK` (RT-W08).
5. **Released building blocks.** A block is offered to an agent only when its `agents` list names the `agent_id`; `mindeststufe` above the run's `stufe` hides it. The runtime never executes a block; the follow-up playbook does (artifact `agent_action_proposal`).
6. **Agent playbooks (0.5.2, F2).** An agent `run_playbook` whose `playbook_id` is not listed under `[playbook_categories]` gets `default_gate`; the category in the request no longer counts for agents. Catalogue sequences take their category from the block and are not affected. `[category_min_stage]` (F10) can raise the minimum stage per category, also for every sequence step.
7. **Service mode.** `policy_service_url` is `https://`, certificate, key and CA are files readable by the app user, the certificate has exactly one CN, which is listed under `[principals]` (`"soar-agent-runtime" = ["agent"]`) and has the role `decide` at the service; TLS 1.2 needs ECDHE with AES-GCM (0.5.2), `policy_timeout_s` fits the network. `test connectivity` must say `policy service healthy, not halted`; anything else denies every effecting action (no fallback, RT-W04).
7. **Sequence register.** `sequences.json` in `policy_state_dir` is written by the Policy Manager's register workflow; without it every multi-step sequence counts as new (at least `APPROVE_1CLICK`).

---

## 5. Air-gap operation

With `airgap_mode` on, the runtime is one part of a closed system: the model runs on an internal server, the **Tool Broker** is the only way to read data beyond the case, and the **Policy Manager** decides every effect. Nothing leaves the network.

### How a run works

1. **Start.** The agent's provider must be `local`; any other is refused before a model call (`airgap_mode: provider '...' is not allowed, only 'local'`). The runtime loads the broker templates released for the `agent_id` (`GET /v1/tools`) and the catalogue building blocks for its `stufe` (`GET /v1/aktionen`); both appear in the prompt as tools. `splunk_search` is removed.
2. **Emergency stop before every step.** The halt state is asked before each step, also before the first model call. Halted or undeterminable ends the run with `run halted by Policy Manager before step N: <reason>` and an audit event `angehalten` (RT-B01, AC-13). A `HALTED` answer from the broker ends the run as well.
3. **Reading.** `get_container_info` reads only the run's own container and leaves out the runtime's own artifacts. `broker_invoke(vorlage_id, parameter)` runs a read-only template; the answer starts with `BROKER RESULT aufruf_id=... datenlage=...`, and that `aufruf_id` is valid evidence. `datenlage` tells whether the source had data at all: with `vorhanden`, 0 rows is a real negative finding; `leer` (no events, or no read right) and `unbekannt` are a blind spot or a gap, never proof of absence. An unreachable broker or source gives `BROKER UNAVAILABLE ...`: a data gap, never a fallback to direct access (RT-L06).
4. **Everything read is data.** Tool results, the case and `extra_context` reach the model inside `<<<DATA source=... untrusted>>>` blocks, truncated to `max_observation_chars`; the prompt tells the model never to follow instructions found there (RT-B04).
5. **Taking effect.** `update_severity` and `catalog_invoke` need a judgement in the call: `begruendung`, `gegenargumente`, `unsicherheiten`, `konfidenz` (0..1) and `beleg_ids` from this run. An incomplete call is refused locally and names what is missing, including misspelled keys (RT-W05). A complete call goes to the Policy Manager, and the run never waits for an approval (RT-W06): `ALLOW` runs `update_severity`; `APPROVE_*` records a proposal with a `freigabe_id` and answers `HELD: ...`, a follow-up playbook executes it after `consume(freigabe_id)`; `DENY` comes back as `POLICY DENIED code=...`. Catalogue measures are never executed by the run, only proposed (chapter 7, *Action catalogue and gap report*).
6. **Audit.** Every step writes an artifact `agent_audit` (severity `low`) with `lauf_id`, `schritt_id`, `agent_id`, the phase (`beabsichtigt`, `erledigt`, `abgelehnt`, `fehler`, `angehalten`), masked parameters, the hash of the result, `aufruf_id`, `entscheidung_id` and `definition_version`. An effecting step is announced (`beabsichtigt`) before it runs. If an artifact cannot be written, the effecting tool is not executed and the run stops (fail-closed, RT-P03).
7. **FINAL_ANSWER.** One JSON object with six fields: `belege` (`schritt_id`s or `aufruf_id`s of this run), `begruendung`, `gegenargumente`, `unsicherheiten`, `konfidenz` and `datenlage` (`vollstaendig`, `teilweise`, `unbekannt`, `blinder_fleck`). A skill's `output_schema` keys come on top, in the same object. An invalid answer gets exactly one correction round that names every required key; a second invalid answer ends the run with `FINAL_ANSWER invalid after correction round` (RT-P04, RT-P06).
8. **Summary note.** The note on the case lists the actions that really took effect and warns when the answer claims an effect that did not happen.

### Commissioning checklist

1. **Local model.** An inference server with an OpenAI-compatible API (e.g. Ollama) reachable from the SOAR host. Set `local_llm_url`, `local_llm_model` (exactly as `ollama list` shows it) and `default_provider = local`. Models with 7B parameters work but need the correction round more often; 14B is more reliable (E2E tests, 2026-10-25).
2. **Tool Broker.** `broker_url` (`https://`), `broker_client_cert`, `broker_client_key`, `broker_ca` as files readable by the user `phantom`. Release templates per `agent_id` in the broker; a sub-agent uses its own `agent_id`.
3. **Policy Manager.** `policy_mode` plus the fields of its mode, then work through the *Operator checklist: Policy Manager* in chapter 4 (halt file, tool mapping, stage and domain).
4. **Definitions** (recommended). Put the definitions repository on the host and set `definitions_path`; the audit then names the exact version of every agent and skill.
5. **Switch on** `airgap_mode`.
6. **Check.** `test connectivity` must report the local model, `broker` and `policy` as OK (and `definitions` when set). Then `setup app` and `test tools` on a test container; in air-gap mode `test tools` skips the severity effect, because that needs the policy gate.
7. **Trial run.** `run agent` on a test container with `stufe` and `domaene`. Check the `agent_audit` artifacts (one `lauf_id`, phases as expected) and the summary note.

---

## 6. Test Connectivity

Expected output:
```
anthropic (claude-haiku-4-5-20251001): OK
openai (gpt-4o-mini): SKIPPED (no credentials configured)
gemini: SKIPPED
local: SKIPPED
```

With `definitions_path` set, the result also has `definitions`: the number of agents and skills read and every rejected file with its reason (informational, never fails the test).

`test connectivity` always tries the `local` provider with `local_llm_model` (default `llama3`); set the field to a model the server actually serves (`ollama list`), otherwise the local check fails although the server is up. Cloud providers are tested only when a key is configured.

---

## 7. Actions

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
Creates or updates an agent in the `agent_definitions` Custom List. With `definitions_path` set, the action is refused and saves nothing: agents then come only from Git (decision E5).

**Prerequisite:** the Custom Lists `agent_definitions` and `agent_skills` (names from the asset fields `agent_list_name` and `skill_list_name`) must exist; create them once under **Home → Custom Lists**. The app reads and writes them over the SOAR REST API (`/rest/decided_list`) as the user who runs the action. Saving appends a row; for the same `agent_id` the latest row wins. If saving fails, the action message names the list and the cause.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_id` | string | Yes | Unique name |
| `system_prompt` | string | Yes | Agent role and instructions |
| `provider` | string | Yes | `anthropic`, `openai`, `gemini`, `local` |
| `model` | string | Yes | Model name |
| `allowed_tools` | string | No | Comma-separated tools |
| `max_steps` | numeric | No | Max loop steps |
| `skills` | string | No | Comma-separated skill IDs |
| `agent_role` | string | No | `triage`, `enrichment`, `response` or `generic` (default); groups the Agent Precision panel in AI-SOC Telemetry |

Available tools: `splunk_search`, `add_note`, `update_severity`, `get_container_info`, `run_sub_agent`

`run_sub_agent(agent_id, task)` runs another agent inside the same run (since 1.16.0, RT-B03): same `lauf_id`, the child's own definition, skills, `allowed_tools` and Tool Broker context, the same policy gate, `stufe`, `domaene` and audit. Its steps are numbered below the calling step (`<lauf_id>/3/1`, `<lauf_id>/3/2`, …) and use up the caller's remaining steps, so the whole run never makes more model calls than the started agent's `max_steps`. The child's checked FINAL_ANSWER comes back as the observation together with the evidence ids the caller may cite: the calling step's `schritt_id`, the child's own `schritt_id`s and its broker `aufruf_id`s (all from the same run). The run's `audit_ereignisse` counts the child's audit events too. A child deeper than `max_subagent_depth` is refused (AC-14); a halt in the child also stops the caller.

`splunk_search` is not available in SOAR apps (SOAR offers apps no search API); it answers with an error and runs no SPL. Read data via `broker_invoke` templates.

---

### `create skill`
Creates or updates a skill in the `agent_skills` Custom List. With `definitions_path` set, the action is refused and saves nothing (E5).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `skill_id` | string | Yes | Unique skill name |
| `description` | string | No | Human-readable label |
| `inject` | string | No | Text injected into system prompt |
| `output_schema` | string | No | JSON schema enforced in FINAL_ANSWER |

---

### `list tools`
Lists every tool: parameters, whether it takes effect, its source, whether it belongs in `allowed_tools` (broker, catalogue and gap tools are offered automatically) and whether it works with this asset's settings. `create agent` rejects names that are not in this list.

---

### `test tools`
Calls every built-in tool directly, without the model, against a test container. Rows: `OK`, `FAIL` (with the exact error), `WARN` or `SKIPPED`. In `airgap_mode` the severity effect is skipped; test it with `run agent` through the policy gate.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `container_id` | numeric | Yes | A container made for testing; it gets one note `[tool self-test]`, its severity is changed and restored |
| `model_check` | boolean | No | Ask the default model once whether it keeps one ACTION per response |

---

### `setup app`
Creates missing Custom Lists, removes older duplicate agent and skill rows (the latest row per id stays) and checks the asset configuration. Returns one row per check with the status `OK`, `CREATED`, `FIXED`, `MISSING`, `ERROR` or, with `dry_run`, `WOULD_CREATE` / `WOULD_FIX`. The action fails when any check is `MISSING` or `ERROR`; the message names each one.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `dry_run` | boolean | No | Only check and report; create and change nothing |

---

### `list agents` / `list skills`
Returns all entries from the respective Custom List, or from `definitions_path` when it is set; invalid files are named in the message. No parameters.

---

## 8. Skills System

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

## 9. SOC Use Cases & Agent Prompts

Every use case comes as a TOML file for the repository [soar-agent-definitions](https://github.com/abuis78/soar-agent-definitions) (`agents/<agent_id>.toml`, used with `definitions_path`). Without `definitions_path`, enter the same fields with `create agent` (lists become comma-separated strings).

**Rules that apply to every use case:**
- **Reading.** An agent reads its own case with `get_container_info`. Everything else (Splunk events, CMDB, identity data, other cases) comes from Tool Broker templates via `broker_invoke`. The broker releases templates per `agent_id`, and they appear in the prompt by themselves, so they are not listed in `allowed_tools`. Prompts describe *what* to look up, never template names: the runtime knows none (RT-L03). `splunk_search` does not work in SOAR apps and is left out.
- **Model.** The templates use the local model `qwen2.5:14b`. In normal mode any provider works, e.g. `provider = "anthropic"`; in air-gap mode only `local`.
- **Answer.** In air-gap mode the FINAL_ANSWER is the JSON judgement from chapter 5; a skill's keys come on top. Tell the model to cite the `schritt_id` / `aufruf_id` of what it read as `belege`.
- **Effects.** `update_severity` and catalogue measures go through the Policy Manager in air-gap mode. Pass `stufe` and `domaene` from the playbook, otherwise the policy falls back to its defaults.
- **Skills.** The skills named here are the examples from chapter 8; put them into `skills/` or the Custom List first.

### UC-1: Alert Triage (Tier-1)

```toml
agent_id = "triage_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
agent_role = "triage"
max_steps = 12
allowed_tools = ["get_container_info", "add_note", "update_severity"]
skills = ["structured_triage", "mitre_mapping"]
system_prompt = """
You are a SOC Tier-1 triage analyst. Decide whether the alert is a true positive,
a false positive or needs escalation.
(1) Read the case with get_container_info.
(2) Look up related events for the same source IP, user and host of the last 24 hours
    with the broker templates you have.
(3) Judge by frequency, context and known patterns. State only what the data shows.
(4) Set the severity when the evidence supports it.
(5) Write a short triage note.
"""
```

```
run agent:  agent_id triage_agent · container_id {{ container.id }} · stufe 1 · domaene soc
task:       Triage this alert. True positive: severity high, recommend escalation.
            False positive: severity low, document why.
```

### UC-2: IOC Enrichment

```toml
agent_id = "ioc_enrichment_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
agent_role = "enrichment"
max_steps = 15
allowed_tools = ["get_container_info", "add_note"]
skills = ["structured_ioc", "mitre_mapping"]
system_prompt = """
You are an IOC enrichment specialist. For each IOC (IP, domain, hash, URL) in the case artifacts:
(1) Look up historical occurrences with the broker templates you have.
(2) Note first seen, last seen and how often it occurs.
(3) Judge prevalence: isolated or widespread.
(4) Write one note per IOC: type, value, first/last seen, hit count, assessment.
If a template is missing or the broker is unavailable, say so as a data gap; never guess.
"""
```

```
run agent:  agent_id ioc_enrichment_agent · container_id {{ container.id }} · domaene soc
task:       Enrich every IOC of this case for the last 30 days and document each one as a note.
```

### UC-3: User Behaviour Investigation (UEBA)

```toml
agent_id = "ueba_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
agent_role = "enrichment"
max_steps = 15
allowed_tools = ["get_container_info", "add_note", "update_severity"]
skills = ["structured_ueba", "mitre_mapping"]
system_prompt = """
You are a user behaviour analyst. For the user named in the case:
(1) Look up logins, failed logins, VPN, e-mail and privileged activity of the last 7 days
    with the broker templates you have.
(2) Find anomalies: unusual hours, new locations, first access to sensitive systems, bulk access.
(3) Build a timeline of the significant events.
(4) Rate the risk (low, medium, high, critical) with reasons and set the severity if justified.
(5) Write a note with the timeline and the rating.
"""
```

```
run agent:  agent_id ueba_agent · container_id {{ container.id }} · stufe 1 · domaene soc
task:       Investigate user {{ artifact.cef.userName }}: 7-day timeline and insider-threat risk.
```

### UC-4: Phishing Email Analysis

```toml
agent_id = "phishing_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
agent_role = "triage"
max_steps = 15
allowed_tools = ["get_container_info", "add_note", "update_severity"]
skills = ["structured_triage", "mitre_mapping"]
system_prompt = """
You are a phishing analyst. For a reported e-mail:
(1) Take sender, reply-to, URLs, attachments and IPs from the case artifacts.
(2) Look up other recipients, clicks and executions with the broker templates you have.
(3) Determine the blast radius: received, opened, clicked.
(4) Classify: credential harvesting, malware delivery, BEC, spear phishing or mass phishing.
(5) Write a note with classification, IOCs, blast radius and the containment you recommend.
Containment itself is a catalogue measure: propose it with catalog_invoke if it is offered,
otherwise record it with report_gap.
"""
```

```
run agent:  agent_id phishing_agent · container_id {{ container.id }} · stufe 2 · domaene soc
task:       Analyse this phishing report: blast radius, attack type, recommended containment.
```

### UC-5: Lateral Movement Detection

```toml
agent_id = "lateral_movement_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
agent_role = "response"
max_steps = 20
allowed_tools = ["get_container_info", "add_note", "update_severity"]
skills = ["mitre_mapping"]
system_prompt = """
You are a threat hunter for lateral movement. For the host or user in the case:
(1) Look up authentications from this source to other internal hosts (SMB, RDP, WMI, SSH)
    of the last 48 hours with the broker templates you have.
(2) Map source -> target chains.
(3) Check for credential access and persistence on the targets.
(4) Set the severity to critical only when lateral movement is confirmed by the data.
(5) Write a note with the attack path, MITRE techniques and the hosts to isolate.
Isolation is a catalogue measure: propose it with catalog_invoke if offered, otherwise report_gap.
"""
```

```
run agent:  agent_id lateral_movement_agent · container_id {{ container.id }} · stufe 2 · domaene soc
task:       Check for lateral movement from host {{ artifact.cef.sourceAddress }} and map the path.
```

### UC-6: Incident Summary (Management Report)

```toml
agent_id = "report_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
max_steps = 6
allowed_tools = ["get_container_info", "add_note"]
skills = ["ciso_report"]
system_prompt = """
You are a senior SOC analyst writing executive summaries, without jargon.
(1) Read the case with get_container_info.
(2) Write a note titled 'Incident Summary': executive summary (3 sentences), timeline,
    impact, root cause if known, actions taken, recommended next steps.
Business impact first, technical detail second.
"""
```

```
run agent:  agent_id report_agent · container_id {{ container.id }}
task:       Write a management-ready summary of this case.
```

### UC-7: Multi-Agent Orchestration (sub-agents)

The orchestrator delegates with `run_sub_agent(agent_id, task)` (chapter 7). Each specialist runs in the same run with its **own** definition, `allowed_tools` and broker releases, so the orchestrator does not need the specialists' tools. Limits: `max_subagent_depth` (default 2) and the step budget. The orchestrator's `max_steps` covers the whole run including every specialist, so give it enough steps for all of them.

```toml
agent_id = "orchestrator_agent"
version = "1.0.0"
provider = "local"
model = "qwen2.5:14b"
max_steps = 40
allowed_tools = ["get_container_info", "run_sub_agent", "add_note"]
system_prompt = """
You are a SOC orchestrator. You do not investigate yourself; you delegate with run_sub_agent:
- triage_agent           first, for every alert
- ioc_enrichment_agent   when the case has IOCs
- ueba_agent             when a user is involved
- lateral_movement_agent when a host may be compromised
- report_agent           last, for the summary
Read the case once, call the specialists in that order, one per step, and pass each a precise
task. Base your FINAL_ANSWER on their results; cite the schritt_id of each run_sub_agent step.
"""
```

```
run agent:  agent_id orchestrator_agent · container_id {{ container.id }} · stufe 1 · domaene soc
task:       Investigate this incident end to end with the specialists and finish with a summary.
```

The specialists' audit events carry their own `agent_id` and `schritt_id`s below the calling step (`<lauf_id>/2/1`, …), all under the orchestrator's `lauf_id`.

---

## 10. Widget Display

The `run agent` action includes a custom Investigation panel widget showing agent ID, provider, steps taken, final answer, and full step-by-step log.

**To activate:**
1. Open an Investigation
2. **Manage Widgets** → **SOAR Agent Runtime** → toggle **On**
3. **Save Layout**

---

## 11. Local Testing

**On a workstation** (no SOAR needed): the test suite uses a fake of the SOAR modules (`conftest.py`) and the bundled wheels.

```bash
uv run --python 3.13 --with pytest --with requests \
  --with wheels/py3/policy_manager-0.5.2-py3-none-any.whl \
  --with wheels/py3/ai_soc_telemetry-0.1.0-py3-none-any.whl \
  --no-project python -m pytest -q
uvx ruff@0.15.21 check .
python3 tools/build_app.py --dry-run
```

Expected: `... passed`, `All checks passed!`, and a file list without `error:`. The mTLS integration tests in `test_soar_agent_runtime_policy_service.py` start the Policy Manager service from the bundled wheel in-process with a throwaway PKI made by the `openssl` CLI; without `openssl` they are skipped.

**On the SOAR host** the connector can run one action from a JSON file (the app directory depends on the installation, e.g. `/opt/apps/soaragentruntime_<uuid>`):

```json
{
  "identifier": "run_agent",
  "asset_id": "1",
  "parameters": [{"agent_id": "triage_agent", "task": "Triage this alert.", "container_id": 1, "max_steps": 3}],
  "config": {
    "default_provider": "local",
    "local_llm_url": "http://<ollama-host>:11434/v1",
    "local_llm_api_key": "ollama",
    "local_llm_model": "qwen2.5:14b"
  }
}
```

```bash
cd <app directory>
phenv python3 soar_agent_runtime_connector.py /tmp/test_run_agent.json
```

---

## 12. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `Missing app directory detected` on install | tar built without directory entries (before 1.9.1) | Build with `python3 tools/build_app.py` from a current `main` |
| `test connectivity`: local `Connection error` | wrong `local_llm_url`, or the model is not served | Check the URL (`http://<host>:11434/v1`) and network path; set `local_llm_model` to a tag from `ollama list` |
| `Custom List '...' is empty; create the agent with 'create agent'.` / list missing | lists not set up | Run `setup app` |
| `Agent 'X' not found in ...` | wrong id, or the agent is in the other source | `list agents`; with `definitions_path` set, agents come only from the TOML files |
| `Definitions come from Git (definitions_path ...); nothing was saved.` | `create agent` / `create skill` while `definitions_path` is set (E5) | Add the file in the definitions repository and update the checkout on the host |
| `agents/X.toml: ...` (e.g. `max_steps must be a positive integer`) | invalid definition file | Fix the named field; `test connectivity` lists every rejected file |
| `Unknown tool(s) in allowed_tools: ...` | typo or a tool that does not exist | Use the names from `list tools` |
| `ERROR: Tool 'X' is not in the allowed tools list.` | the agent called a tool it is not allowed to use | Add it to `allowed_tools`, or adjust the prompt |
| `ERROR: splunk_search is not available in SOAR apps ...` | SOAR has no search API for apps | Read via `broker_invoke` templates |
| `airgap_mode: provider '...' is not allowed, only 'local'` | cloud provider in air-gap mode, also for a sub-agent | `provider = "local"` in the definition or `default_provider = local` |
| `ERROR: Tool 'update_severity' has an effect and needs ...` | judgement fields missing or misspelled (the text names them, e.g. `unknown key 'begrün dung' - did you mean 'begruendung'?`) | Usually fixed by the model on the next step; repeated failures: a larger model |
| `beleg_id '...' did not originate in this run` | the model invented evidence | The message lists the valid ids; repeated failures: a larger model |
| `FINAL_ANSWER invalid after correction round: ...` | the answer lacks required keys twice | Make sure a skill's `inject` does not suggest its keys are the whole answer (judgement fields stay required); larger model |
| `POLICY DENIED code=...` | the Policy Manager refused | Read the code; check stage/domain, `[tool_action_kinds]`, `[playbook_categories]` (chapter 4 checklist) |
| `HELD: 'X' requires APPROVE_...` | the policy wants an approval | Approve in the Policy Manager; the follow-up playbook executes after `consume(freigabe_id)` |
| `run halted by Policy Manager before step N: ...` | emergency stop active or halt state unknown | Check `halt.json` / the service; `test connectivity` shows the halt reason |
| `policy embedded: halt file missing (create halt.json ...)` | no `halt.json` in `policy_state_dir` | Create it with `{"angehalten": false, "grund": ""}`, owner `phantom`, mode `0600`, directory `0700` |
| `Tool Broker is HALTED - run not started (RT-B01)` | the broker reports a halt | Resume in the Tool Broker |
| `BROKER UNAVAILABLE code=...` | broker or source not reachable | Network, certificates, `broker_timeout_s`; the run treats it as a data gap |
| `audit not writable ... (fail-closed, RT-P03)` | the audit artifact could not be saved | Check SOAR (database, permissions of the action user); in air-gap mode nothing takes effect without audit |
| `ERROR: sub-agent 'X' refused: depth N exceeds max_subagent_depth M` | sub-agent nesting too deep (AC-14) | Flatten the delegation or raise `max_subagent_depth` |
| `Max steps (N) reached without FINAL_ANSWER.` | step limit reached; sub-agents use up the same budget | Raise `max_steps` in the definition or the action |
| Severity jumps back up after `update_severity` | SOAR raises a container to its highest artifact severity | Fixed since 1.12.2 (runtime artifacts are `low`); other `medium`/`high` artifacts on the case still raise it |
| Widget not showing | widget not activated | **Manage Widgets** in the Investigation panel |

---

## 13. Changelog

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
| 1.9.1 | 2026-10-24 | Asset field `local_llm_model` (default `llama3`): the model `test connectivity` asks the local server for, and the fallback model for provider `local` when the agent definition names none; `tools/build_app.py` builds the installable tar with directory entries (B17) |
| 1.9.2 | 2026-10-25 | `create agent` and `create skill` put the exception from saving to the Custom List into the action message (SOAR keeps no progress lines); README names both Custom Lists as a prerequisite and documents `agent_role` |
| 1.10.0 | 2026-10-25 | Agent and skill definitions over the SOAR REST API (`/rest/decided_list`) instead of `phantom.rules`, which has no `add_to_list` and whose `get_list` returns a tuple: `create agent`, `create skill`, `list agents`, `list skills` and `run agent` work on SOAR 8.7; latest row per id wins, empty rows skipped; a missing list or a rejected call is named in the action message |
| 1.10.1 | 2026-10-25 | `get_container_info` reads the case over the SOAR REST API (`/rest/container/<id>` and its artifacts, at most 5) instead of `phantom.rules.get_container`/`get_artifacts`, which need a playbook context and failed on SOAR 8.7 |
| 1.10.2 | 2026-10-25 | `update_severity` sets the severity over the SOAR REST API (`POST /rest/container/<id>`) after an allowed policy decision instead of `phantom.rules.update_container`, which apps cannot call on SOAR 8.7; a rejected value is named in the observation |
| 1.10.3 | 2026-10-25 | `run agent` summary note lists the executed actions from the step log and warns when the answer claims an effect (`update_severity`, `catalog_invoke`) that did not take effect; result data `executed_actions` and `unbelegte_behauptungen` |
| 1.11.0 | 2026-10-25 | New action `setup app`: creates missing Custom Lists, removes older duplicate agent/skill rows, checks model and (in `airgap_mode`) broker, policy, files and `halt.json`; `dry_run`. `test connectivity` reports missing Custom Lists without writing |
| 1.11.1 | 2026-10-25 | Strict parser: a model turn with several `ACTION`s or `ACTION` plus `FINAL_ANSWER` runs nothing and asks the model to send one action at a time (a second ACTION used to replace the first silently); the step log keeps the raw model turn (2000 characters). `splunk_search` answers that SOAR apps cannot search (`phantom.rules.run_query` does not exist) |
| 1.11.2 | 2026-10-25 | A model turn with several `ACTION`s runs the first one; the others and a `FINAL_ANSWER` in the same turn are not executed and are named in the observation, the step log (`not_executed`) and the summary note (qwen2.5:7b sends two actions per turn and ignored the 1.11.1 rejection) |
| 1.12.0 | 2026-10-25 | New actions `list tools` (every tool with availability on this asset) and `test tools` (calls each built-in tool for real against a test container, restores the severity, optional model format check); `create agent` rejects unknown tool names; `setup app` warns about stored agents with unknown names |
| 1.12.1 | 2026-10-25 | Audit phase follows the outcome: refused calls (evidence check, allowlist, validation, policy DENY, broker rejection) are `abgelehnt`, failed execution, an unreachable source and halts are `fehler` (before: `erledigt`, E2E finding E-4); the message for missing `beleg_ids` lists the valid evidence ids of the run |
| 1.12.2 | 2026-10-25 | Runtime artifacts (`agent_audit`, `agent_action_proposal`, `agent_action_gap`) carry severity `low`: SOAR silently raises a container to a higher artifact severity, and the default `medium` undid `update_severity` to `low` (E2E finding E-6). A container at `informational` would still be raised to `low` |
| 1.12.3 | 2026-10-25 | `get_container_info` leaves out the runtime's own artifacts (`agent_audit`, proposals, gaps), which fed old thoughts and parameters back to the model (E2E finding E-7); new fields `runtime_artifacts_hidden`, `artifacts_total`. A turn in Markdown is still not executed, but the correction names it and shows the exact plain form |
| 1.13.0 | 2026-10-25 | Phase 5.1 (RT-B01, AC-13): in `airgap_mode` the halt state is asked before every step; halted or unknown ends the run before the model is called, with audit phase `angehalten`. Airgap without a working policy gate now stops before step 1 (before: only effecting calls were denied) |
| 1.14.0 | 2026-10-25 | Phase 5.2 (RT-B02): new asset field `definitions_path`; agents and skills come from TOML files in a checkout of the repository soar-agent-definitions, validated per file (id = file name, required fields, types, known tools). Audit `definition_version` = `<id>@<version>#<sha256 prefix>` for the agent and its skills. `test connectivity` reports the definitions. Empty field = Custom Lists as before |
| 1.15.0 | 2026-10-25 | Phase 5.3 (E5): with `definitions_path` set, `create agent` and `create skill` are refused with a pointer to the definitions repository and write nothing; `list agents` / `list skills` read from Git since 1.14.0. Empty field = unchanged |
| 1.15.1 | 2026-10-25 | Bundled `policy_manager` 0.5.2 (PolicyManager tag `v0.5.2`, P0 review fixes F1-F14). Operator checklist: halt file and directory permissions (F13), agent playbooks need `[playbook_categories]` (F2), `[category_min_stage]` (F10), service principals and TLS. Runtime code unchanged |
| 1.15.2 | 2026-10-25 | Error texts name misspelled keys with the right spelling (`unknown key 'begrün dung' - did you mean 'begruendung'?`) for effecting calls and FINAL_ANSWER; the key is still not accepted (E2E finding E-8). The FINAL_ANSWER correction round lists every required key, judgement fields and skill keys together (E-9) |
| 1.15.3 | 2026-10-25 | Cosmetic: the misspelled-key hint in the error of an effecting call no longer ends with `?.` |
| 1.16.0 | 2026-10-25 | Phase 5.4 (RT-B03, AC-14, E6): `run_sub_agent` really runs the named agent in the same run (was a placeholder): own definition, allowlist and broker context, same `lauf_id`, policy, `stufe`, `domaene` and audit; step ids below the calling step; shared step budget; new asset field `max_subagent_depth` (default 2) |
| 1.16.1 | 2026-10-26 | Sub-agents (E2E findings E-10, E-11): the caller may cite the child's `schritt_id`s and broker `aufruf_id`s as `belege` (they were refused although the result line announced them); the result line lists the real ids. `audit_ereignisse` / `audit_fehler` include the child's audit events |
| 1.16.2 | 2026-10-26 | `broker_invoke` passes the broker's `datenlage` (`vorhanden`, `leer`, `unbekannt`, TB-F18) to the model in the `BROKER RESULT` line, with one line on what it means for the judgement (`leer` → `blinder_fleck`, `unbekannt` → `unbekannt`); a broker without the field shows `nicht gemeldet` (E2E finding E-12) |
| 2.0.0 | 2026-10-26 | Conversion complete (E8): phases 1-5 done. Acceptance tests cover AC-01..AC-15 end to end; 5.6 adds AC-13 for a broker `HALTED` during the run and a halt inside a sub-agent (the whole run ends), AC-14 for self-recursion (refused at depth 3, audit `abgelehnt`), `max_subagent_depth = 0` and allowed nesting, plus a guard that every criterion has a test. No behaviour change against 1.16.2. Verified on SOAR-Prod: 1.13.0-1.15.2 ([setup issue #24](https://github.com/abuis78/augmenteic_soc_setup/issues/24)), 1.16.0-1.16.1 ([setup issue #29](https://github.com/abuis78/augmenteic_soc_setup/issues/29)) |
