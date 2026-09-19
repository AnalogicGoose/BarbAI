"""
Server-side agent loop: call the model, execute any tool it requests,
feed the result back, repeat until it gives a final answer.

This is what makes BarbAI itself agentic rather than a raw passthrough -
distinct from /v1/chat/completions, /chat, and /v1/messages (which hand a
tool_call back to whoever's calling and stop; correct behavior for
OpenAI/Anthropic compat, where the caller runs its own tools).
"""

from __future__ import annotations

import json

from barbai.core import model_runtime
from barbai.core.tool_calls import UnrecognizedToolCallFormatError, to_openai_message
from barbai.core.tools import ToolExecutionError, build_tool_defs, execute_tool

MAX_ITERATIONS = 5

class AgentError(RuntimeError):
    """The loop couldn't produce a final answer (model error or ran out of iterations)."""

def run_agent(
    llm, messages: list[dict], max_iterations: int = MAX_ITERATIONS, thinking_mode: str = "thinking"
) -> dict:
    """Run the loop, returning the final assistant message (no tool_calls)."""
    messages = list(messages)
    tool_defs = build_tool_defs()

    for _ in range(max_iterations):
        raw = model_runtime.create_chat_completion(
            llm, messages=messages, tools=tool_defs, thinking_mode=thinking_mode
        )
        try:
            message = to_openai_message(raw["choices"][0]["message"])
        except UnrecognizedToolCallFormatError as exc:
            raise AgentError(str(exc)) from exc

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message

        # arguments come back as a JSON string (OpenAI wire shape); feeding
        # that straight back into the next create_chat_completion call
        # breaks the Jinja template rendering, which expects a dict here
        # (same issue barbai.api.openai._prepare_messages works around).
        assistant_tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])},
            }
            for tc in tool_calls
        ]
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": assistant_tool_calls})

        for tc in tool_calls:
            name = tc["function"]["name"]
            arguments = json.loads(tc["function"]["arguments"])
            try:
                result = execute_tool(name, arguments)
            except ToolExecutionError as exc:
                result = f"Error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    raise AgentError(f"exceeded {max_iterations} tool-call iterations without a final answer")