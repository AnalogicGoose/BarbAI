import json

import pytest

from barbai.core.tool_calls import (
    TagStreamParser,
    TruncatedToolCallError,
    UnrecognizedToolCallFormatError,
    generate_message,
    parse_tool_call_tags,
    to_openai_message,
)


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


def test_parse_tool_call_tags_basic():
    content = (
        "reasoning here</think>\n\n"
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>"
    )
    result = parse_tool_call_tags(content)
    assert result["reasoning_content"] == "reasoning here"
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_parse_tool_call_tags_no_tool_call():
    result = parse_tool_call_tags("thinking</think>\n\nplain answer")
    assert result["content"] == "plain answer"
    assert result["reasoning_content"] == "thinking"
    assert "tool_calls" not in result


def test_to_openai_message_native_tool_calls_passthrough():
    raw = {
        "role": "assistant",
        "content": "reasoning</think>\n\nHi",
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    }
    result = to_openai_message(raw)
    assert result["tool_calls"] == raw["tool_calls"]
    assert result["content"] == "Hi"
    assert result["reasoning_content"] == "reasoning"


def test_to_openai_message_tag_format():
    raw = {"role": "assistant", "content": "thinking</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>"}
    result = to_openai_message(raw)
    assert result["tool_calls"][0]["function"]["name"] == "f"


def test_to_openai_message_unrecognized_format_raises():
    raw = {"role": "assistant", "content": "</think>\n\n<tool_call>garbage not matching function tag"}
    with pytest.raises(UnrecognizedToolCallFormatError):
        to_openai_message(raw)


def test_to_openai_message_plain_reply():
    raw = {"role": "assistant", "content": "reasoning</think>\n\nHello there"}
    result = to_openai_message(raw)
    assert result["content"] == "Hello there"
    assert result["reasoning_content"] == "reasoning"


def test_to_openai_message_truncated_tool_call_raises():
    raw = {
        "role": "assistant",
        "content": "</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf a doc",
    }
    with pytest.raises(TruncatedToolCallError):
        to_openai_message(raw, finish_reason="length")


def test_to_openai_message_unclosed_tool_call_without_length_finish_reason_falls_through():
    """Only finish_reason == "length" is strong enough evidence of a
    truncated generation to raise on - an unclosed tag with no
    finish_reason info (or a non-length one) falls back to the old
    behavior (still not great, but not confidently "truncated" either)."""
    raw = {
        "role": "assistant",
        "content": "</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf a doc",
    }
    result = to_openai_message(raw)
    assert "tool_calls" not in result
    assert "<tool_call>" in result["content"]


def _merge(events):
    merged = []
    for e in events:
        if e["type"] in ("reasoning", "content") and merged and merged[-1]["type"] == e["type"]:
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + e["text"]}
        else:
            e = dict(e)
            if e.get("type") == "tool_call":
                e["id"] = None
            merged.append(e)
    return merged


def _run(chunks):
    parser = TagStreamParser()
    events = []
    for c in chunks:
        events.extend(parser.feed(c))
    events.extend(parser.finish())
    return _merge(events)


def test_stream_plain_word_chunked():
    result = _run(["reasoning", " done", "</think>", "\n\n", "Hello", " there"])
    assert result == [
        {"type": "reasoning", "text": "reasoning done"},
        {"type": "content", "text": "\n\nHello there"},
    ]


def test_stream_think_close_split_char_by_char():
    result = _run(["think", "ing", "<", "/", "t", "h", "i", "n", "k", ">", "answer"])
    assert result == [
        {"type": "reasoning", "text": "thinking"},
        {"type": "content", "text": "answer"},
    ]


def test_stream_tool_call_with_split_tags():
    chunks = [
        "deciding to call it",
        "</thi",
        "nk>\n\n<tool_",
        "call>\n<function=get_weather>\n<param",
        "eter=location>\nTokyo\n</parameter>\n</function>\n</tool_call",
        ">",
    ]
    result = _run(chunks)
    assert result == [
        {"type": "reasoning", "text": "deciding to call it"},
        {"type": "content", "text": "\n\n"},
        {"type": "tool_call", "id": None, "name": "get_weather", "arguments": {"location": "Tokyo"}},
    ]


def test_stream_truncated_mid_reasoning():
    result = _run(["still", " thinking", " when cut off"])
    assert result == [{"type": "reasoning", "text": "still thinking when cut off"}]


def test_stream_trailing_lone_angle_bracket():
    result = _run(["done</think>", "\n\nmath: 2 ", "<", " 3"])
    assert result == [
        {"type": "reasoning", "text": "done"},
        {"type": "content", "text": "\n\nmath: 2 < 3"},
    ]


def test_stream_close_tag_split_across_feed_boundary():
    """Regression test: checking only the newest chunk (not the full
    accumulated buffer) for </tool_call> misses it entirely when the tag
    is split right before the final '>' - a real bug caught and fixed.
    """
    chunks = [
        "</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call",
        ">",
    ]
    result = _run(chunks)
    tool_calls = [e for e in result if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[0]["arguments"] == {"location": "Tokyo"}


def test_stream_truncated_mid_tool_call_flushes_instead_of_dropping():
    """Regression test: a stream that ends mid <tool_call> (closing tags
    never arrived) used to vanish with no event at all - finish() only
    handled the reasoning/content states. Now the partial tag text is
    surfaced as content instead of silently discarded."""
    chunks = ["deciding", "</think>", "\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf a doc"]
    result = _run(chunks)
    assert result == [
        {"type": "reasoning", "text": "deciding"},
        {"type": "content", "text": "\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf a doc"},
    ]


def test_generate_message_returns_message_and_trimmed_messages(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.tool_calls.model_runtime.create_chat_completion",
        lambda llm, messages, **kwargs: _raw("Hello there"),
    )
    message, trimmed = generate_message(_FakeLlm(), [{"role": "user", "content": "hi"}])
    assert message["content"] == "Hello there"
    assert trimmed == [{"role": "user", "content": "hi"}]


def test_generate_message_retries_once_with_a_larger_reserve_on_truncation(monkeypatch):
    fit_calls = []

    def fake_fit(llm, messages, reserved_for_response=None):
        fit_calls.append(reserved_for_response)
        return messages

    completions = iter(
        [
            _raw("</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf", finish_reason="length"),
            _raw("all good now"),
        ]
    )

    monkeypatch.setattr("barbai.core.tool_calls.model_runtime.fit_to_context", fake_fit)
    monkeypatch.setattr(
        "barbai.core.tool_calls.model_runtime.create_chat_completion",
        lambda llm, messages, **kwargs: next(completions),
    )

    message, _ = generate_message(_FakeLlm(n_ctx=8192), [{"role": "user", "content": "hi"}])

    assert message["content"] == "all good now"
    assert len(fit_calls) == 2
    assert fit_calls[1] == fit_calls[0] * 2


def test_generate_message_gives_up_after_one_retry(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.tool_calls.model_runtime.fit_to_context",
        lambda llm, messages, reserved_for_response=None: messages,
    )
    monkeypatch.setattr(
        "barbai.core.tool_calls.model_runtime.create_chat_completion",
        lambda llm, messages, **kwargs: _raw(
            "</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\nhalf", finish_reason="length"
        ),
    )

    with pytest.raises(TruncatedToolCallError):
        generate_message(_FakeLlm(n_ctx=8192), [{"role": "user", "content": "hi"}])
