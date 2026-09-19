import json

import pytest

from barbai.core.agent import AgentError, run_agent
from barbai.core.tool_calls import UnrecognizedToolCallFormatError
from barbai.core.tools import ToolExecutionError


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
    result = run_agent(llm=object(), messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "Hello there"


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

    result = run_agent(llm=object(), messages=[{"role": "user", "content": "hi"}], max_iterations=5)
    assert result["content"] == "The answer is FINAL-42"


def test_exceeds_max_iterations_raises(monkeypatch):
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw(
            None, tool_calls=[_tool_call("call_x", "read_file", {"path": "x.txt"})], finish_reason="tool_calls"
        ),
    )
    monkeypatch.setattr("barbai.core.agent.execute_tool", lambda name, arguments: "result")

    with pytest.raises(AgentError):
        run_agent(llm=object(), messages=[{"role": "user", "content": "hi"}], max_iterations=2)


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

    result = run_agent(llm=object(), messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "I couldn't read that file."


def test_unrecognized_tool_format_raises_agent_error(monkeypatch):
    def raise_unrecognized(*args, **kwargs):
        raise UnrecognizedToolCallFormatError("bad format")

    monkeypatch.setattr("barbai.core.agent.to_openai_message", raise_unrecognized)
    monkeypatch.setattr(
        "barbai.core.agent.model_runtime.create_chat_completion",
        lambda llm, **kwargs: _raw("whatever"),
    )
    with pytest.raises(AgentError):
        run_agent(llm=object(), messages=[{"role": "user", "content": "hi"}])
