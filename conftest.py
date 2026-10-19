"""Pytest bootstrap: a minimal fake of the Splunk SOAR platform modules.

The connector imports ``phantom.app``, ``phantom.base_connector`` and
``phantom.action_result`` at module level. Off-platform these packages do
not exist, so a small stand-in is installed into ``sys.modules`` before
collection. It is only installed when the real platform is absent.

Note: ``phantom.rules`` is deliberately NOT provided here. Utils import it
lazily inside try/except ImportError, and tests that need it install their
own stub via monkeypatch (see test_soar_agent_runtime_telemetry.py).
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - real platform present
    import phantom.app  # noqa: F401
except ImportError:
    _app = types.ModuleType("phantom.app")
    _app.APP_SUCCESS = True
    _app.APP_ERROR = False

    class _ActionResult:
        def __init__(self, param=None):
            self.param = dict(param or {})
            self.status = None
            self.message = ""
            self.data: list = []
            self.summary: dict = {}

        def set_status(self, status, message=""):
            self.status = status
            self.message = message
            return status

        def add_data(self, item):
            self.data.append(item)

        def update_summary(self, summary):
            self.summary.update(summary)

        def get_status(self):
            return self.status

        def get_message(self):
            return self.message

    _action_result = types.ModuleType("phantom.action_result")
    _action_result.ActionResult = _ActionResult

    class _BaseConnector:
        """Just enough of BaseConnector for off-platform unit tests."""

        def __init__(self):
            self._fake_config: dict = {}
            self._fake_state: dict = {}
            self._fake_action_id = ""
            self.action_results: list = []
            self.progress: list = []

        def get_config(self):
            return self._fake_config

        def load_state(self):
            return self._fake_state

        def save_state(self, state):
            self._fake_state = state or {}

        def save_progress(self, message, *args, **kwargs):
            self.progress.append(str(message))

        def debug_print(self, *args, **kwargs):
            pass

        def add_action_result(self, action_result):
            self.action_results.append(action_result)
            return action_result

        def get_action_identifier(self):
            return self._fake_action_id

    _base_connector = types.ModuleType("phantom.base_connector")
    _base_connector.BaseConnector = _BaseConnector

    _pkg = types.ModuleType("phantom")
    _pkg.app = _app
    _pkg.action_result = _action_result
    _pkg.base_connector = _base_connector

    sys.modules.setdefault("phantom", _pkg)
    sys.modules.setdefault("phantom.app", _app)
    sys.modules.setdefault("phantom.action_result", _action_result)
    sys.modules.setdefault("phantom.base_connector", _base_connector)
