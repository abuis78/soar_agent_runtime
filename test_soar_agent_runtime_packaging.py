"""Packaging tests for offline installation (finding B17, sub-change 1.11).

SOAR installs bundled wheels with ``pip --no-deps``: every pinned dependency
must ship as a wheel that matches SOAR's Python 3.13 on Linux x86_64, and
app.json must reference exactly those files.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
APP = json.loads((ROOT / "soar_agent_runtime.json").read_text())
WHEELS = APP["pip313_dependencies"]["wheel"]
# Pure-Python wheels that are not on PyPI (ai_soc_telemetry, policy_manager) live in wheels/py3/.
LOCAL_WHEELS = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "wheels" / "py3").glob("*.whl"))
WHEEL_RE = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)(-[^-]+)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def _pins() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements-py313.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version
    return pins


def test_app_targets_python_313_on_soar_7_or_newer():
    assert APP["python_version"] == "3.13"
    assert APP["min_phantom_version"] == "7.0.0"


def test_no_pypi_dependencies_remain_anywhere():
    for key in ("pip_dependencies", "pip3_dependencies", "pip39_dependencies"):
        assert key not in APP, key
    assert "pypi" not in APP["pip313_dependencies"]


def test_every_listed_wheel_exists_and_every_bundled_wheel_is_listed():
    listed = {w["input_file"] for w in WHEELS}
    for path in listed:
        assert (ROOT / path).is_file(), path
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "wheels" / "py313").glob("*.whl")}
    on_disk.update(LOCAL_WHEELS)
    assert on_disk == listed


def test_local_wheels_are_bundled():
    assert LOCAL_WHEELS == ["wheels/py3/ai_soc_telemetry-0.1.0-py3-none-any.whl",
                            "wheels/py3/policy_manager-0.5.1-py3-none-any.whl"]


def test_every_pin_is_bundled_with_the_pinned_version():
    bundled = {}
    for w in WHEELS:
        m = WHEEL_RE.match(pathlib.Path(w["input_file"]).name)
        assert m, w["input_file"]
        bundled[m["name"].lower().replace("_", "-")] = m["version"]
    for name, version in _pins().items():
        assert bundled.get(name) == version, (name, version, bundled.get(name))
    assert len(WHEELS) == len(_pins()) + len(LOCAL_WHEELS)


@pytest.mark.parametrize("entry", WHEELS, ids=lambda e: e["module"])
def test_wheel_tags_fit_soar_python_313_linux_x86_64(entry):
    m = WHEEL_RE.match(pathlib.Path(entry["input_file"]).name)
    assert m, entry["input_file"]
    py, abi, plat = m["py"], m["abi"], m["plat"]
    assert py in ("py3", "cp313") or py.startswith("py2.py3") or (py.startswith("cp3") and abi == "abi3"), entry
    assert abi in ("none", "cp313", "abi3"), entry
    assert plat == "any" or ("x86_64" in plat and "manylinux" in plat), entry
    assert "macosx" not in plat and "win" not in plat and "aarch64" not in plat, entry


def test_module_names_match_wheel_distribution_names():
    for w in WHEELS:
        dist = pathlib.Path(w["input_file"]).name.split("-")[0].lower().replace("_", "-")
        assert w["module"] == dist, w
