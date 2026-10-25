"""
soar_agent_runtime_definitions.py
Agent and skill definitions from a Git checkout (RT-B02, Umsetzungskonzept 5.2).

With the asset field ``definitions_path`` set, agents and skills come from TOML
files instead of the Custom Lists:

    <definitions_path>/agents/<agent_id>.toml
    <definitions_path>/skills/<skill_id>.toml

The format is documented in the repository soar-agent-definitions. The app
only reads the directory; getting the files onto the host (clone, bundle,
archive) is the operator's job, so no git binary and no network are needed.

Every definition carries ``version``; the store adds ``definition_version``
= ``<id>@<version>#<12 hex of the file's SHA-256>``, which goes into the audit.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import tomllib

from soar_agent_runtime_utils import AGENT_ROLES, SkillStore

AGENTS_DIR = "agents"
SKILLS_DIR = "skills"
MAX_FILE_BYTES = 256 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

_AGENT_REQUIRED = ("version", "provider", "model", "system_prompt")
_SKILL_REQUIRED = ("version",)


class DefinitionError(ValueError):
    """A definition file is missing, unreadable or invalid; the text names file and reason."""


class DefinitionNotFound(DefinitionError):
    """No file for the requested id."""


def _str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def _check_agent(data: dict, known_tools) -> str:
    if "allowed_tools" in data:
        if not _str_list(data["allowed_tools"]):
            return "allowed_tools must be a list of strings"
        if known_tools is not None:
            unknown = [t for t in data["allowed_tools"] if t not in known_tools]
            if unknown:
                return f"unknown tool(s) in allowed_tools: {', '.join(unknown)} (see the action 'list tools')"
    if "skills" in data and not _str_list(data["skills"]):
        return "skills must be a list of strings"
    if "max_steps" in data:
        steps = data["max_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            return "max_steps must be a positive integer"
    if "agent_role" in data and data["agent_role"] not in AGENT_ROLES:
        return f"agent_role must be one of {list(AGENT_ROLES)}"
    return ""


def _check_skill(data: dict, known_tools) -> str:
    for key in ("description", "inject"):
        if key in data and not isinstance(data[key], str):
            return f"{key} must be a string"
    if "output_schema" in data and not isinstance(data["output_schema"], dict):
        return "output_schema must be a table"
    return ""


class GitDefinitionStore:
    """Reads one kind of definition (agents or skills) from ``definitions_path``.

    Files are read on every call, so a run always sees the current checkout.
    ``problems`` lists every rejected file of the last listing as "file: reason".
    """

    def __init__(self, connector, path: str, kind: str, known_tools=None) -> None:
        self.connector = connector
        self.root = pathlib.Path(path)
        self.kind = kind
        self.id_key = "agent_id" if kind == AGENTS_DIR else "skill_id"
        self.required = _AGENT_REQUIRED if kind == AGENTS_DIR else _SKILL_REQUIRED
        self.check = _check_agent if kind == AGENTS_DIR else _check_skill
        self.known_tools = set(known_tools) if known_tools is not None else None
        self.problems: list[str] = []
        self.last_error = ""

    @property
    def directory(self) -> pathlib.Path:
        return self.root / self.kind

    def _require_directory(self) -> None:
        if not self.directory.is_dir():
            raise DefinitionError(f"definitions_path: directory not found: {self.directory}")

    def _read(self, file: pathlib.Path) -> dict:
        """Parse and validate one file; raises DefinitionError naming the file."""
        name = f"{self.kind}/{file.name}"
        try:
            raw = file.read_bytes()
        except OSError as e:
            raise DefinitionError(f"{name}: not readable ({type(e).__name__})") from None
        if len(raw) > MAX_FILE_BYTES:
            raise DefinitionError(f"{name}: larger than {MAX_FILE_BYTES // 1024} KB")
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            raise DefinitionError(f"{name}: not valid TOML ({e})") from None
        def_id = data.get(self.id_key)
        if def_id != file.stem:
            raise DefinitionError(f"{name}: {self.id_key} must be '{file.stem}' (the file name), got {def_id!r}")
        if not _ID_RE.match(file.stem):
            raise DefinitionError(f"{name}: {self.id_key} must match [A-Za-z0-9_.-] and be 1-64 chars")
        for key in self.required:
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise DefinitionError(f"{name}: required field '{key}' missing or not a non-empty string")
        if problem := self.check(data, self.known_tools):
            raise DefinitionError(f"{name}: {problem}")
        digest = hashlib.sha256(raw).hexdigest()[:12]
        return data | {"definition_version": f"{file.stem}@{data['version']}#{digest}"}

    def load(self, def_id: str) -> dict:
        """Definition of def_id; raises DefinitionError with a readable reason."""
        self._require_directory()
        file = self.directory / f"{def_id}.toml"
        if not _ID_RE.match(def_id or "") or not file.is_file():
            raise DefinitionNotFound(f"{self.id_key[:-3].capitalize()} '{def_id}' not found in {self.directory}.")
        return self._read(file)

    def list_all(self) -> list[dict]:
        """Every valid definition, sorted by id; rejected files go to ``problems``."""
        self._require_directory()
        found, self.problems = [], []
        for file in sorted(self.directory.glob("*.toml")):
            try:
                found.append(self._read(file))
            except DefinitionError as e:
                self.problems.append(str(e))
        return found

    def save(self, def_id: str, config: dict) -> bool:
        self.last_error = f"definitions come from Git (definitions_path {self.root}); edit {self.kind}/{def_id}.toml there"
        return False

    # -- SkillStore interface --------------------------------------------------

    def load_many(self, skill_ids: list[str]) -> list[dict]:
        """Like SkillStore.load_many: a missing skill is skipped with a warning,
        an invalid file is fatal (the agent asks for it, so it must not run without it)."""
        skills = []
        for sid in skill_ids:
            try:
                skills.append(self.load(sid))
            except DefinitionNotFound:
                self.connector.save_progress(f"[Skills] Warning: skill '{sid}' not found — skipped.")
        return skills

    build_injection = staticmethod(SkillStore.build_injection)
