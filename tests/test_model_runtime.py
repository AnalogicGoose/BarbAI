import pytest

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
