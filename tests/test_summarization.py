from barbai.core.summarization import summarize_turns


class _FakeLlm:
    """Stands in for llama_cpp.Llama - just enough for
    model_runtime.fit_to_context() to work against it."""

    def __init__(self, n_ctx: int = 100_000):
        self._n_ctx = n_ctx

    def n_ctx(self) -> int:
        return self._n_ctx

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
        return list(range(len(data.split()) or 1))


def _raw(content, finish_reason="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]}


def test_summarize_turns_returns_model_reply(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.summarization.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw("A condensed summary."),
    )
    result = summarize_turns(_FakeLlm(), None, [{"role": "user", "content": "hi"}])
    assert result == "A condensed summary."


def test_summarize_turns_includes_prior_summary_in_the_prompt(monkeypatch):
    seen = {}

    def fake_create(llm, messages, **kwargs):
        seen["messages"] = messages
        return _raw("updated summary")

    monkeypatch.setattr("barbai.core.summarization.model_runtime.create_chat_completion", fake_create)
    summarize_turns(_FakeLlm(), "earlier summary text", [{"role": "user", "content": "new turn"}])

    prompt = seen["messages"][-1]["content"]
    assert "earlier summary text" in prompt
    assert "new turn" in prompt


def test_summarize_turns_uses_fast_thinking_mode(monkeypatch):
    seen = {}

    def fake_create(llm, messages, **kwargs):
        seen["kwargs"] = kwargs
        return _raw("summary")

    monkeypatch.setattr("barbai.core.summarization.model_runtime.create_chat_completion", fake_create)
    summarize_turns(_FakeLlm(), None, [{"role": "user", "content": "hi"}])

    assert seen["kwargs"].get("thinking_mode") == "fast"
