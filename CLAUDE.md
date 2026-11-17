# CLAUDE.md — soar_agent_runtime

Supplements the global `~/.claude/CLAUDE.md`; only project-specific points here. No deviations from the global rules.

## Purpose
Splunk SOAR app (connector) that runs LLM agents in a ReAct loop for SOC automation. Every SOC action goes
through the Policy Manager (ALLOW / APPROVE / DENY); APPROVE creates a proposal that the playbook `pm_freigabe` approves.

## Stack
- Python 3.13 (SOAR ≥ 7.0.0, `python_version` in `soar_agent_runtime.json`), version in `app_version`.
- Bundled wheels, installed by SOAR with `pip --no-deps` (every transitive dependency must be there):
  `wheels/py313/` (PyPI, pinned in `requirements-py313.txt`), `wheels/py3/` (local: `policy_manager`, `ai_soc_telemetry`).
- Agent definitions (TOML) live in the separate repo `abuis78/soar-agent-definitions`.

## Commands
```bash
uv run --python 3.13 --with pytest --with requests \
  --with wheels/py3/policy_manager-0.6.0-py3-none-any.whl \
  --with wheels/py3/ai_soc_telemetry-0.1.0-py3-none-any.whl \
  --no-project python -m pytest -q > "$TMPDIR/t.txt"   # token: "N passed"
uvx ruff@0.15.21 check .                                 # token: "All checks passed!"
python3 tools/build_app.py --dry-run                     # token: "dry run, nothing written"
python3 tools/build_app.py                               # writes soar_agent_runtime_v<version>.tar (gitignored)
python3 tools/update_wheels.py [--dry-run]               # download pinned wheels, sync the JSON wheel list
```
- Release: bump `app_version` + the version assert in `test_soar_agent_runtime_connector.py`, add a README changelog entry
  (section 13), squash-merge, tag `v<semver>` on main.
- A new `policy_manager` wheel is built in the PolicyManager repo (`scripts/build_wheel.sh --copy-to <this repo>/wheels/py3`);
  update the file name in the test command, README §11, `tools/update_wheels.py` and the JSON.

## Domain rules / lessons learned
- Dependabot only bumps `requirements-py313.txt`, not the bundled wheel. A security bump = swap the wheel in `wheels/py313/`,
  check the sha256 against PyPI, update JSON + requirements + changelog, bump the patch version (see PR #55).
- PM-F42: the `freigabe_id` never appears in plain text in LLM context, notes, observations or logs; use
  `freigabe_hash()` (`soar_agent_runtime_utils.py`). Only `cef.freigabe_id` carries it.
- `approval_playbook` takes the numeric playbook ID or `<repo>/<name>` (e.g. `dev/pm_freigabe`); a bare name returns HTTP 400.
  `playbook_run` `inputs` must be a dict, not a JSON string (B3, verified on SOAR-Prod 2026-11-02).
- `add_to_list` does not exist in SOAR; lists are written via REST.
- Small local models (qwen 7B) send several ACTIONs per turn; the parser takes the first.
- AWS/SSM access to SOAR-Prod always with `--profile janus-env2`.
- An install/upgrade on SOAR-Prod is done by the setup session `augmenteic-soc-setup` and needs Andreas' go there.

## Status (2026-11-17)
- main = v2.2.2 (2.2.1 + prompt hint: cite the broker `aufruf_id` as beleg, PR #61); not installed on prod.
- SOAR-Prod: runtime 2.2.1 (app ID 371, asset 51 `ollama`, `policy_mode = service`), PM service 0.6.0 (rule `beleg_inhalt`
  is always active, no policy.toml entry), `approval_playbook = dev/pm_freigabe`. D22 acceptance passed on container 238
  ([setup PR #82](https://github.com/abuis78/augmenteic_soc_setup/pull/82)): schritt_id only → APPROVE_1CLICK (run 1009),
  broker aufruf_id → ALLOW (run 1011). Lesson: in service mode the PM service version decides, not the bundled wheel.
- Next: stage 2 four-eyes E2E, run by the setup session, Andreas votes as `max` and `moritz` (role `SOC-Freigabe`).
  Needs APPROVE_2PERSON (e.g. `tipping_off = laut`); approvals expire after 10 min; negative test: same person twice →
  DUPLICATE (SELF_APPROVAL never applies to agent runs). Afterwards deactivate `pm_approve`, then decide on 2.2.2 for prod.
- Before production: granular Asset Access Control for `pm_approve` (tested on non-prod first) and CRL (QS F11).
- Dependencies: PolicyManager (wheel, `pm_freigabe`), augmented-soc (concept doc), soar-agent-definitions, setup repo.

## Control
This session can be steered by an orchestrator in Quorda; questions then go to it via SendMessage.
