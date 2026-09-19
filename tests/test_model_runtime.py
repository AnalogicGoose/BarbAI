import pytest

from barbai.core import model_runtime
from barbai.core.model_runtime import create_chat_completion


class FakeLlama:
    """Stands in for llama_cpp.Llama, exercising the same handler-resolution
    path create_chat_completion() uses, without needing a real GGUF loaded.
    """

    def __init__(self):
        self.chat_handler = None
        self.chat_format = "chat_template.default"
        self.last_call_kwargs = None

        def handler(**kwargs):
            self.last_call_kwargs = kwargs
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

        self._chat_handlers = {"chat_template.default": handler}


def test_invalid_thinking_mode_raises():
    with pytest.raises(ValueError):
        create_chat_completion(FakeLlama(), messages=[], thinking_mode="bogus")


def test_fast_mode_disables_thinking():
    llm = FakeLlama()
    create_chat_completion(llm, messages=[{"role": "user", "content": "hi"}], thinking_mode="fast")
    assert llm.last_call_kwargs["enable_thinking"] is False


def test_thinking_mode_enables_thinking():
    llm = FakeLlama()
    create_chat_completion(llm, messages=[{"role": "user", "content": "hi"}], thinking_mode="thinking")
    assert llm.last_call_kwargs["enable_thinking"] is True


def test_extended_mode_raises_max_tokens_floor():
    llm = FakeLlama()
    create_chat_completion(llm, messages=[], thinking_mode="extended", max_tokens=100)
    assert llm.last_call_kwargs["max_tokens"] == 1500


def test_extended_mode_respects_higher_explicit_max_tokens():
    llm = FakeLlama()
    create_chat_completion(llm, messages=[], thinking_mode="extended", max_tokens=3000)
    assert llm.last_call_kwargs["max_tokens"] == 3000


def test_no_resolvable_handler_raises_runtime_error():
    llm = FakeLlama()
    llm._chat_handlers = {}
    llm.chat_format = "missing"
    with pytest.raises(RuntimeError):
        create_chat_completion(llm, messages=[])

@pytest.fixture(autouse=True)
def _reset_model_state():
    """ensure_mode/current_mode share module-level state - reset it around
    every test in this file so tests don't leak into each other."""
    model_runtime.unload_model()
    yield
    model_runtime.unload_model()


def test_current_mode_none_when_nothing_loaded():
    assert model_runtime.current_mode() is None


def test_ensure_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        model_runtime.ensure_mode("bogus")


def test_ensure_mode_loads_when_nothing_loaded(monkeypatch):
    fake = FakeLlama()
    calls = []

    def fake_load(*, model_path=None, n_gpu_layers=-1, n_ctx=4096, mode="general"):
        calls.append(mode)
        model_runtime._model = fake
        model_runtime._current_mode = mode
        return fake

    monkeypatch.setattr(model_runtime, "load_model", fake_load)
    result = model_runtime.ensure_mode("coding")
    assert result is fake
    assert calls == ["coding"]
    assert model_runtime.current_mode() == "coding"


def test_ensure_mode_skips_reload_for_same_mode(monkeypatch):
    fake = FakeLlama()
    model_runtime._model = fake
    model_runtime._current_mode = "general"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_model should not be called when the mode is already active")

    monkeypatch.setattr(model_runtime, "load_model", fail_if_called)
    result = model_runtime.ensure_mode("general")
    assert result is fake


def test_ensure_mode_reloads_on_mode_switch(monkeypatch):
    old = FakeLlama()
    new = FakeLlama()
    model_runtime._model = old
    model_runtime._current_mode = "general"
    calls = []

    def fake_load(*, model_path=None, n_gpu_layers=-1, n_ctx=4096, mode="general"):
        calls.append(mode)
        model_runtime._model = new
        model_runtime._current_mode = mode
        return new

    monkeypatch.setattr(model_runtime, "load_model", fake_load)
    result = model_runtime.ensure_mode("coding")
    assert result is new
    assert calls == ["coding"]