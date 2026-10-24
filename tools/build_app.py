#!/usr/bin/env python3
"""Build the installable SOAR app package (B17, offline installation).

Usage:
    python3 tools/build_app.py              # write soar_agent_runtime_v<app_version>.tar to the repo root
    python3 tools/build_app.py --out DIR    # write it to DIR instead
    python3 tools/build_app.py --dry-run    # list the files that would be packed, write nothing

The tar carries one top-level directory ``soar_agent_runtime/`` with the app
modules, app.json, icons, the view files, the README and the bundled wheels
under ``wheels/``. SOAR installs the wheels with ``pip --no-deps``, so every
wheel that app.json references must be in the package; a wheel on disk that
app.json does not reference is a leftover and fails the build as well.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_JSON = ROOT / "soar_agent_runtime.json"
TOP_DIR = "soar_agent_runtime"
DEP_KEY = "pip313_dependencies"
WHEEL_DIRS = ("wheels/py3", "wheels/py313")
# Files at the repo root that belong in the app; tests, conftest.py, tools/ and
# requirements-py313.txt stay out.
STATIC_FILES = (
    "__init__.py",
    "README.md",
    "soar_agent_runtime.json",
    "soar_agent_runtime.svg",
    "soar_agent_runtime_dark.svg",
    "soar_agent_runtime_view.html",
)


def app_version() -> str:
    return json.loads(APP_JSON.read_text())["app_version"]


def referenced_wheels() -> list[str]:
    app = json.loads(APP_JSON.read_text())
    return sorted(entry["input_file"] for entry in app[DEP_KEY]["wheel"])


def wheels_on_disk() -> list[str]:
    found = []
    for wheel_dir in WHEEL_DIRS:
        found.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / wheel_dir).glob("*.whl"))
    return sorted(found)


def check_wheels() -> list[str]:
    """Return the wheels to pack, or exit when app.json and wheels/ disagree."""
    referenced, on_disk = referenced_wheels(), wheels_on_disk()
    missing = sorted(set(referenced) - set(on_disk))
    extra = sorted(set(on_disk) - set(referenced))
    if missing or extra:
        for path in missing:
            print(f"error: app.json references a wheel that is not on disk: {path}", file=sys.stderr)
        for path in extra:
            print(f"error: wheel on disk is not referenced by app.json: {path}", file=sys.stderr)
        print("hint: run python3 tools/update_wheels.py and commit the result", file=sys.stderr)
        sys.exit(1)
    return referenced


def module_files() -> list[str]:
    modules = sorted(p.name for p in ROOT.glob("soar_agent_runtime_*.py") if not p.name.startswith("test_"))
    if "soar_agent_runtime_connector.py" not in modules:
        print("error: soar_agent_runtime_connector.py not found next to app.json", file=sys.stderr)
        sys.exit(1)
    return modules


def package_files() -> list[str]:
    files = list(STATIC_FILES) + module_files() + check_wheels()
    for rel in files:
        if not (ROOT / rel).is_file():
            print(f"error: missing file: {rel}", file=sys.stderr)
            sys.exit(1)
    return files


def warn_if_dirty() -> None:
    try:
        status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    if status.strip():
        print("warning: the working tree has uncommitted changes; the package reflects the files on disk",
              file=sys.stderr)


def build(out_dir: pathlib.Path, dry_run: bool) -> pathlib.Path:
    files = package_files()
    target = out_dir / f"{TOP_DIR}_v{app_version()}.tar"
    print(f"package: {target}")
    for rel in files:
        print(f"  {TOP_DIR}/{rel}")
    print(f"{len(files)} files, {sum(1 for f in files if f.endswith('.whl'))} wheels")
    if dry_run:
        print("dry run, nothing written")
        return target
    warn_if_dirty()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w") as tar:
        for rel in files:
            tar.add(ROOT / rel, arcname=f"{TOP_DIR}/{rel}", recursive=False)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"written: {target} ({size_mb:.1f} MB)")
    print("next: SOAR → Apps → Install App → upload this file (no internet needed on the host)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT, help="directory for the tar (default: repo root)")
    parser.add_argument("--dry-run", action="store_true", help="list the files, write nothing")
    args = parser.parse_args()
    build(args.out.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
