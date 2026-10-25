#!/usr/bin/env python3
"""Download the pinned wheels for SOAR's Python 3.13 and sync app.json (B17).

Usage:
    python3 tools/update_wheels.py            # download + rewrite pip313_dependencies.wheel
    python3 tools/update_wheels.py --dry-run  # show what would change, touch nothing

SOAR installs bundled wheels with ``pip --no-deps``, so every transitive
dependency must be present. Pins live in requirements-py313.txt; wheels go to
wheels/py313/. Pure-Python local wheels (ai_soc_telemetry, policy_manager) stay under wheels/py3/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_JSON = ROOT / "soar_agent_runtime.json"
REQUIREMENTS = ROOT / "requirements-py313.txt"
WHEEL_DIR = ROOT / "wheels" / "py313"
# Local (non-PyPI) pure-Python wheels, bundled as-is. policy_manager is built from the
# PolicyManager repo: scripts/build_wheel.sh --copy-to <this app>/wheels/py3
EXTRA_WHEELS = [
    ROOT / "wheels" / "py3" / "ai_soc_telemetry-0.1.0-py3-none-any.whl",
    ROOT / "wheels" / "py3" / "policy_manager-0.5.2-py3-none-any.whl",
]
DEP_KEY = "pip313_dependencies"
PIP_ARGS = [
    "--only-binary=:all:",
    "--platform", "manylinux2014_x86_64",
    "--python-version", "3.13",
    "--implementation", "cp",
    "--abi", "cp313",
    "--abi", "abi3",
    "--abi", "none",
]


def module_name(wheel: pathlib.Path) -> str:
    """Distribution name from the wheel file name (PEP 427), normalised."""
    return re.split(r"-", wheel.name, maxsplit=1)[0].replace("_", "-").lower()


def download(dry_run: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-r", str(REQUIREMENTS),
           "-d", str(WHEEL_DIR), *PIP_ARGS]
    print("pip:", " ".join(cmd[cmd.index("download"):]))
    if dry_run:
        return
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def pinned_versions() -> dict[str, str]:
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name.replace("_", "-").lower()] = version
    return pins


def prune_stale(dry_run: bool) -> None:
    """Remove wheels whose distribution/version is no longer pinned (no duplicates)."""
    pins = pinned_versions()
    for wheel in sorted(WHEEL_DIR.glob("*.whl")):
        name = module_name(wheel)
        version = wheel.name.split("-")[1]
        if pins.get(name) != version:
            print("remove stale:", wheel.relative_to(ROOT))
            if not dry_run:
                wheel.unlink()


def wheel_entries() -> list[dict]:
    wheels = sorted(WHEEL_DIR.glob("*.whl")) + [w for w in EXTRA_WHEELS if w.exists()]
    return [{"module": module_name(w), "input_file": w.relative_to(ROOT).as_posix()} for w in wheels]


def sync_app_json(dry_run: bool) -> None:
    app = json.loads(APP_JSON.read_text())
    entries = wheel_entries()
    current = app.get(DEP_KEY, {}).get("wheel", [])
    if current == entries:
        print(f"app.json: {DEP_KEY}.wheel already up to date ({len(entries)} wheels)")
        return
    print(f"app.json: {DEP_KEY}.wheel {len(current)} -> {len(entries)} wheels")
    if dry_run:
        return
    app[DEP_KEY] = {"wheel": entries}
    APP_JSON.write_text(json.dumps(app, indent=4, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show actions, change nothing")
    args = parser.parse_args()
    download(args.dry_run)
    if WHEEL_DIR.exists():
        prune_stale(args.dry_run)
    sync_app_json(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
