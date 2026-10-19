"""LLMProvider tests for RT-A03 (Umsetzungskonzept Runtime, sub-change 1.2).

Covers: timeout on the client, temperature and max_tokens in every call,
seed only for OpenAI-compatible servers and only when set, and backward
compatibility of the four positional constructor arguments.
"""

from __future__ import annotations

import types

import pytest

import soar_agent_runtime_utils as utils
from soar_agent_runtime_utils import LLMProvider


# --- fakes ------------------------------------------------------------------

class _Recorder:
    """Collects constructor and create() kwargs of a fake SDK client."""

    def __init__(self):
        self.init_kwargs: dict = {}
        self.create_kwargs: dict = {}


def _fake_openai(rec: _Recorder):
    class _Completions:
        def create(self, **kwargs):
            rec.create_kwargs = kwargs
            msg = types.SimpleNamespace(content="OK")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            rec.init_kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    return FakeOpenAI


def _fake_anthropic(rec: _Recorder):
    class _Messages:
        def create(self, **kwargs):
            rec.create_kwargs = kwargs
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="OK")])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            rec.init_kwargs = kwargs
            self.messages = _Messages()

    return types.SimpleNamespace(Anthropic=FakeAnthropic)


def _fake_genai(rec: _Recorder):
    class FakeModel:
        def __init__(self, **kwargs):
            rec.init_kwargs = kwargs

        def generate_content(self, contents, **kwargs):
            rec.create_kwargs = kwargs
            return types.SimpleNamespace(text="OK")

    return types.SimpleNamespace(configure=lambda **kw: None, GenerativeModel=FakeModel)


@pytest.fixture
def openai_rec(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(utils, "OpenAI", _fake_openai(rec), raising=False)
    monkeypatch.setattr(utils, "HAS_OPENAI", True)
    return rec


@pytest.fixture
def anthropic_rec(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(utils, "anthropic", _fake_anthropic(rec), raising=False)
    monkeypatch.setattr(utils, "HAS_ANTHROPIC", True)
    return rec


@pytest.fixture
def genai_rec(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(utils, "genai", _fake_genai(rec), raising=False)
    monkeypatch.setattr(utils, "HAS_GEMINI", True)
    return rec


MSGS = [{"role": "user", "content": "hi"}]


# --- defaults and backward compatibility --------------------------------------

def test_positional_constructor_keeps_working_with_defaults():
    llm = LLMProvider("local", "ollama", "http://localhost:11434/v1", "llama3")
    assert (llm.timeout_s, llm.temperature, llm.max_tokens, llm.seed) == (120.0, 0.0, 4096, None)


# --- OpenAI-compatible (local + openai) ---------------------------------------

@pytest.mark.parametrize("provider, url", [
    ("local", "http://localhost:11434/v1"),
    ("openai", "https://api.openai.com/v1"),
])
def test_openai_compatible_passes_timeout_temperature_max_tokens(openai_rec, provider, url):
    llm = LLMProvider(provider, "k", url, "m", timeout_s=7, temperature=0.2, max_tokens=512)

    assert llm.chat("sys", MSGS) == "OK"
    assert openai_rec.init_kwargs["timeout"] == 7.0
    assert openai_rec.create_kwargs["temperature"] == 0.2
    assert openai_rec.create_kwargs["max_tokens"] == 512
    assert openai_rec.create_kwargs["messages"][0] == {"role": "system", "content": "sys"}


def test_seed_is_omitted_when_not_set(openai_rec):
    LLMProvider("local", "k", "http://localhost:11434/v1", "m").chat("sys", MSGS)
    assert "seed" not in openai_rec.create_kwargs


def test_seed_is_sent_when_set(openai_rec):
    LLMProvider("local", "k", "http://localhost:11434/v1", "m", seed=42).chat("sys", MSGS)
    assert openai_rec.create_kwargs["seed"] == 42


def test_rejected_seed_surfaces_as_error_not_silent_fallback(monkeypatch, openai_rec):
    """RT-A03: no silent fallback when the inference server rejects a parameter."""
    def exploding_create(**kwargs):
        raise TypeError("unexpected parameter: seed")

    llm = LLMProvider("local", "k", "http://localhost:11434/v1", "m", seed=1)
    fake_client = utils.OpenAI()
    monkeypatch.setattr(fake_client.chat.completions, "create", exploding_create)
    monkeypatch.setattr(utils, "OpenAI", lambda **kw: fake_client)

    with pytest.raises(TypeError, match="seed"):
        llm.chat("sys", MSGS)


# --- Anthropic ----------------------------------------------------------------

def test_anthropic_passes_timeout_temperature_max_tokens(anthropic_rec):
    llm = LLMProvider("anthropic", "k", "https://api.anthropic.com", "m",
                      timeout_s=30, temperature=0.0, max_tokens=1024, seed=5)

    assert llm.chat("sys", MSGS) == "OK"
    assert anthropic_rec.init_kwargs["timeout"] == 30.0
    assert anthropic_rec.create_kwargs["temperature"] == 0.0
    assert anthropic_rec.create_kwargs["max_tokens"] == 1024
    assert anthropic_rec.create_kwargs["system"] == "sys"
    assert "seed" not in anthropic_rec.create_kwargs  # not supported there


# --- Gemini -------------------------------------------------------------------

def test_gemini_passes_generation_config_and_timeout(genai_rec):
    llm = LLMProvider("gemini", "k", "", "m", timeout_s=9, temperature=0.1, max_tokens=256)

    assert llm.chat("sys", MSGS) == "OK"
    assert genai_rec.init_kwargs["generation_config"] == {"temperature": 0.1, "max_output_tokens": 256}
    assert genai_rec.init_kwargs["system_instruction"] == "sys"
    assert genai_rec.create_kwargs["request_options"] == {"timeout": 9.0}
