import pytest

from barbai.core import model_runtime
from barbai.core.model_runtime import create_chat_completion


class FakeLlama:
    """Stands in for llama_cpp.Llama, exercising the same handler-resolution
    path create_chat_completion() uses, without needing a real GGUF loaded.
    Also stands in for fit_to_context()/count_tokens()'s needs: n_ctx()
    and a whitespace-count tokenize() close enough for a fixed-budget
    test, not meant to match a real tokenizer exactly.
    """

    def __init__(self, n_ctx: int = 4096):
        self.chat_handler = None
        self.chat_format = "chat_template.default"
        self.last_call_kwargs = None
        self._n_ctx = n_ctx

        def handler(**kwargs):
            self.last_call_kwargs = kwargs
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

        self._chat_handlers = {"chat_template.default": handler}

    def n_ctx(self) -> int:
        return self._n_ctx

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
        return list(range(len(data.split()) or 1))


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


def test_resolve_n_ctx_explicit_value_wins(monkeypatch):
    monkeypatch.setenv("BARBAI_N_CTX", "8192")
    assert model_runtime._resolve_n_ctx(2048) == 2048


def test_resolve_n_ctx_env_var_used_when_not_explicit(monkeypatch):
    monkeypatch.setenv("BARBAI_N_CTX", "16384")
    assert model_runtime._resolve_n_ctx(None) == 16384


def test_resolve_n_ctx_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("BARBAI_N_CTX", raising=False)
    assert model_runtime._resolve_n_ctx(None) == model_runtime.DEFAULT_N_CTX


def test_resolve_n_ctx_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("BARBAI_N_CTX", "not-a-number")
    with pytest.raises(ValueError):
        model_runtime._resolve_n_ctx(None)


def test_ensure_mode_is_locked_against_concurrent_loads(monkeypatch):
    """Regression test: two /agent/chat requests arriving close together,
    both seeing nothing loaded yet, must not both call load_model()
    concurrently - that's what let two simultaneous ~5GB CUDA
    allocations fail with 'out of memory' in practice even though a
    single load had plenty of free VRAM. Simulates the race with a
    barrier so both threads are genuinely inside ensure_mode at once,
    and a slow fake load so a real race would be caught if the lock
    weren't there.
    """
    import threading
    import time

    call_count = 0
    call_count_lock = threading.Lock()
    start_barrier = threading.Barrier(2)

    def fake_load(*, model_path=None, n_gpu_layers=-1, n_ctx=None, mode="general"):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.05)  # gives a real (unlocked) race a window to double-enter
        fake = FakeLlama()
        model_runtime._model = fake
        model_runtime._current_mode = mode
        return fake

    monkeypatch.setattr(model_runtime, "load_model", fake_load)

    results = []

    def worker():
        start_barrier.wait()
        results.append(model_runtime.ensure_mode("general"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1, "load_model() ran more than once for a concurrent request pair"
    assert results[0] is results[1]


def test_count_tokens_sums_content_plus_overhead():
    llm = FakeLlama()
    messages = [{"role": "user", "content": "one two three"}]  # 3 whitespace tokens
    assert model_runtime.count_tokens(llm, messages) == 3 + model_runtime._TOKENS_PER_MESSAGE_OVERHEAD + 1


def test_count_tokens_handles_missing_content():
    llm = FakeLlama()
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}]
    assert model_runtime.count_tokens(llm, messages) == model_runtime._TOKENS_PER_MESSAGE_OVERHEAD + 1


def test_fit_to_context_empty_messages():
    assert model_runtime.fit_to_context(FakeLlama(), []) == []


def test_fit_to_context_returns_unchanged_when_it_already_fits():
    llm = FakeLlama(n_ctx=100_000)
    messages = [
        {"role": "system", "content": "you are BarbAI"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you"},
    ]
    assert model_runtime.fit_to_context(llm, messages) == messages


def test_fit_to_context_drops_oldest_first():
    # Small n_ctx forces trimming; each message here is ~1 token of
    # content + the fixed per-message overhead (9 tokens), 6 messages =
    # 54 total - budget must be under that for trimming to actually kick in.
    llm = FakeLlama(n_ctx=50)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
        {"role": "user", "content": "newest"},
    ]
    trimmed = model_runtime.fit_to_context(llm, messages, reserved_for_response=10)
    assert trimmed[0] == messages[0]  # system message always kept
    assert trimmed[-1] == messages[-1]  # newest message always kept
    assert len(trimmed) < len(messages)  # something was actually dropped
    # whatever's left really does fit the budget
    assert model_runtime.count_tokens(llm, trimmed) <= llm.n_ctx() - 10


def test_fit_to_context_keeps_system_even_under_extreme_pressure():
    llm = FakeLlama(n_ctx=32)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a very long message " * 20},
    ]
    trimmed = model_runtime.fit_to_context(llm, messages, reserved_for_response=1)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1] == messages[-1]
    assert len(trimmed) == 2  # never drops below system + newest


def test_fit_to_context_no_system_message():
    # 5 messages * 9 tokens each = 45 total - budget must be under that.
    llm = FakeLlama(n_ctx=40)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
        {"role": "user", "content": "newest"},
    ]
    trimmed = model_runtime.fit_to_context(llm, messages, reserved_for_response=10)
    assert trimmed[-1] == messages[-1]
    assert len(trimmed) < len(messages)


def test_fit_to_context_budget_too_small_raises():
    llm = FakeLlama(n_ctx=100)
    with pytest.raises(ValueError):
        model_runtime.fit_to_context(llm, [{"role": "user", "content": "hi"}], reserved_for_response=200)