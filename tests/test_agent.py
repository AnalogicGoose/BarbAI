import json

import pytest

from barbai.core.agent import AgentError, run_agent
from barbai.core.tool_calls import UnrecognizedToolCallFormatError
from barbai.core.tools import ToolExecutionError


class _FakeLlm:
    """Stands in for llama_cpp.Llama - just enough for
    model_runtime.fit_to_context() to work against it. A large default
    n_ctx so trimming never kicks in for tests that aren't testing
    trimming itself; a simple whitespace-count tokenizer is close enough
    for a fixed-budget check, not meant to match a real tokenizer
    exactly.
    """

    def __init__(self, n_ctx: int = 100_000):
        self._n_ctx = n_ctx

    def n_ctx(self) -> int:
        return self._n_ctx

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
        return list(range(len(data.split()) or 1))


def _raw(content, tool_calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_no_tool_call_returns_immediately(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw("Hello there"),
    )
    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}])
    assert result["status"] == "final"
    assert result["message"]["content"] == "Hello there"


def test_multi_round_tool_calls_regression(monkeypatch):
    """Regression test for a real bug: the loop's "exceeded max iterations"
    raise was mis-indented one level too deep (inside the outer loop
    instead of after it), so it fired unconditionally after the *first*
    tool-call round, before a second round ever got a chance to run. This
    needs at least two sequential tool-call rounds to catch.
    """
    calls = [
        _raw(None, tool_calls=[_tool_call("call_1", "read_file", {"path": "pointer.txt"})], finish_reason="tool_calls"),
        _raw(None, tool_calls=[_tool_call("call_2", "read_file", {"path": "target.txt"})], finish_reason="tool_calls"),
        _raw("The answer is FINAL-42"),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "some tool result")

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=10)
    assert result["status"] == "final"
    assert result["message"]["content"] == "The answer is FINAL-42"


def test_exceeds_max_iterations_raises(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw(
            None, tool_calls=[_tool_call("call_x", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"
        ),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "result")

    with pytest.raises(AgentError):
        run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=2)


def test_tool_execution_error_fed_back_not_raised(monkeypatch):
    calls = [
        _raw(None, tool_calls=[_tool_call("call_1", "read_file", {"path": "bad.txt"})], finish_reason="tool_calls"),
        _raw("I couldn't read that file."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )

    def failing_execute(name, arguments):
        raise ToolExecutionError("no such file")

    monkeypatch.setattr("barbai.core.agent.execute_tool", failing_execute)

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}])
    assert result["status"] == "final"
    assert result["message"]["content"] == "I couldn't read that file."


def test_unrecognized_tool_format_raises_agent_error(monkeypatch):
    def raise_unrecognized(*args, **kwargs):
        raise UnrecognizedToolCallFormatError("bad format")

    monkeypatch.setattr("barbai.core.agent.to_openai_message", raise_unrecognized)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw("whatever"),
    )
    with pytest.raises(AgentError):
        run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}])


def test_gated_tool_call_pauses_for_approval(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw(
            None,
            tool_calls=[_tool_call("call_w1", "write_file", {"path": "out.txt", "content": "hi"})],
            finish_reason="tool_calls",
        ),
    )

    def unexpected_execute(name, arguments):
        raise AssertionError("execute_tool must not run before approval")

    monkeypatch.setattr("barbai.core.agent.execute_tool", unexpected_execute)

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "write me a file"}])
    assert result["status"] == "pending_approval"
    assert [tc["id"] for tc in result["pending"]] == ["call_w1"]
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["tool_calls"][0]["function"]["name"] == "write_file"


def test_resuming_with_approval_executes_the_tool(monkeypatch):
    calls = [
        _raw(
            None,
            tool_calls=[_tool_call("call_w1", "write_file", {"path": "out.txt", "content": "hi"})],
            finish_reason="tool_calls",
        ),
        _raw("Done, file written."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "wrote 2 characters")

    paused = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "write me a file"}], max_iterations=10)
    assert paused["status"] == "pending_approval"

    result = run_agent(
        llm=_FakeLlm(),
        messages=paused["messages"],
        approvals={"call_w1": True},
        max_iterations=10,
    )
    assert result["status"] == "final"
    assert result["message"]["content"] == "Done, file written."


def test_resuming_with_denial_skips_execution(monkeypatch):
    calls = [
        _raw(
            None,
            tool_calls=[_tool_call("call_w1", "write_file", {"path": "out.txt", "content": "hi"})],
            finish_reason="tool_calls",
        ),
        _raw("Okay, I won't write that file."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )

    def unexpected_execute(name, arguments):
        raise AssertionError("execute_tool must not run for a denied call")

    monkeypatch.setattr("barbai.core.agent.execute_tool", unexpected_execute)

    paused = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "write me a file"}], max_iterations=10)

    result = run_agent(
        llm=_FakeLlm(),
        messages=paused["messages"],
        approvals={"call_w1": False},
        max_iterations=10,
    )
    assert result["status"] == "final"
    assert result["message"]["content"] == "Okay, I won't write that file."


def test_read_file_not_gated_runs_without_approval(monkeypatch):
    calls = [
        _raw(None, tool_calls=[_tool_call("call_r1", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"),
        _raw("Read it, here's the answer."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "file contents")

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=10)
    assert result["status"] == "final"
    assert result["message"]["content"] == "Read it, here's the answer."


def test_stuck_detection_stops_on_repeated_identical_round(monkeypatch):
    calls = [
        _raw(None, tool_calls=[_tool_call("call_1", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"),
        _raw(None, tool_calls=[_tool_call("call_2", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"),
        _raw("should never be reached"),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "same result every time")

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=20)
    assert result["status"] == "stuck"
    assert "same" in result["message"]["content"].lower()


def test_stuck_detection_ignores_different_arguments(monkeypatch):
    calls = [
        _raw(None, tool_calls=[_tool_call("call_1", "read_file", {"path": "a.txt"})], finish_reason="tool_calls"),
        _raw(None, tool_calls=[_tool_call("call_2", "read_file", {"path": "b.txt"})], finish_reason="tool_calls"),
        _raw("Got both files."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "same content")

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=20)
    assert result["status"] == "final"
    assert result["message"]["content"] == "Got both files."


def test_stuck_detection_ignores_different_results(monkeypatch):
    calls = [
        _raw(None, tool_calls=[_tool_call("call_1", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"),
        _raw(None, tool_calls=[_tool_call("call_2", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"),
        _raw("Done."),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )
    results = iter(["first result", "second result"])
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: next(results))

    result = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}], max_iterations=20)
    assert result["status"] == "final"


def test_stuck_detection_across_resume_with_repeated_denial(monkeypatch):
    calls = [
        _raw(
            None,
            tool_calls=[_tool_call("call_w1", "write_file", {"path": "out.txt", "content": "hi"})],
            finish_reason="tool_calls",
        ),
        _raw(
            None,
            tool_calls=[_tool_call("call_w2", "write_file", {"path": "out.txt", "content": "hi"})],
            finish_reason="tool_calls",
        ),
        _raw("should never be reached"),
    ]
    call_iter = iter(calls)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: next(call_iter),
    )

    def unexpected_execute(name, arguments):
        raise AssertionError("a denied call must never execute")

    monkeypatch.setattr("barbai.core.agent.execute_tool", unexpected_execute)

    paused1 = run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "write a file"}], max_iterations=20)
    assert paused1["status"] == "pending_approval"

    resumed1 = run_agent(
        llm=_FakeLlm(), messages=paused1["messages"], approvals={"call_w1": False}, max_iterations=20
    )
    assert resumed1["status"] == "pending_approval"

    resumed2 = run_agent(
        llm=_FakeLlm(), messages=resumed1["messages"], approvals={"call_w2": False}, max_iterations=20
    )
    assert resumed2["status"] == "stuck"


def test_run_agent_trims_only_the_view_sent_to_the_model(monkeypatch):
    # run_agent uses fit_to_context's default 512-token reserve, so n_ctx
    # has to clear that plus enough room for system+newest (~20 tokens)
    # to avoid raising outright, while still being tight enough that the
    # 4 verbose "old" messages below don't all fit.
    llm = _FakeLlm(n_ctx=560)
    seen_messages = []

    def fake_create(llm, messages, **kwargs):
        seen_messages.append(list(messages))
        return _raw("done")

    monkeypatch.setattr("barbai.core.agent.model_runtime.create_chat_completion", fake_create)

    full_history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old message number one here"},
        {"role": "assistant", "content": "old reply number one here"},
        {"role": "user", "content": "old message number two here"},
        {"role": "assistant", "content": "old reply number two here"},
        {"role": "user", "content": "the newest question"},
    ]
    original_len = len(full_history)

    result = run_agent(llm=llm, messages=full_history)

    assert result["status"] == "final"
    assert len(seen_messages[0]) < original_len  # model saw a trimmed view
    assert seen_messages[0][0]["role"] == "system"  # but kept the system message
    assert seen_messages[0][-1] == full_history[-1]  # and the newest message
    assert len(full_history) == original_len  # caller's own list is untouched


def test_run_agent_converts_residual_context_overflow_to_agent_error(monkeypatch):
    def fake_create(llm, messages, **kwargs):
        raise ValueError("Requested tokens (9999) exceed context window of 4096")

    monkeypatch.setattr("barbai.core.agent.model_runtime.create_chat_completion", fake_create)

    with pytest.raises(AgentError, match="too long"):
        run_agent(llm=_FakeLlm(), messages=[{"role": "user", "content": "hi"}])
