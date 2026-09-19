"""
Server-side agent loop: call the model, execute any tool it requests,
feed the result back, repeat until it gives a final answer.

This is what makes BarbAI itself agentic rather than a raw passthrough -
distinct from /v1/chat/completions, /chat, and /v1/messages (which hand a
tool_call back to whoever's calling and stop; correct behavior for
OpenAI/Anthropic compat, where the caller runs its own tools).

Gated tools (currently just write_file) don't execute immediately - the
loop pauses and hands control back to the caller instead of running them,
per the roadmap's approval-gate design (human review for agent-authored
writes, not for routine reads). A paused call returns
{"status": "pending_approval", "pending": [...], "messages": [...]}; the
caller resumes by calling run_agent again with that same `messages` list
and an `approvals` dict mapping each pending tool_call_id to True/False.
There's no server-side session store - the caller holds the paused state
between requests.

Because pausing needs to happen *before* a gated tool executes, and
resuming must not re-ask the model for a new turn it already answered,
each iteration does at most one of "call the model" or "resolve the
pending tool call(s)" rather than both - so a full model-call+execute
round now costs two iterations instead of one. MAX_ITERATIONS is set
with that in mind.
"""

from __future__ import annotations

import json

from barbai.core import model_runtime
from barbai.core.tool_calls import UnrecognizedToolCallFormatError, to_openai_message
from barbai.core.tools import GATED_TOOLS, ToolExecutionError, build_tool_defs, execute_tool

MAX_ITERATIONS = 10

class AgentError(RuntimeError):
    """The loop couldn't produce a final answer (model error or ran out of iterations)."""

def run_agent(
    llm,
    messages: list[dict],
    max_iterations: int = MAX_ITERATIONS,
    thinking_mode: str = "thinking",
    approvals: dict[str, bool] | None = None,
) -> dict:
    """Run the loop. Returns {"status": "final", "message": {...}} or
    {"status": "pending_approval", "pending": [...], "messages": [...]}."""
    messages = list(messages)
    tool_defs = build_tool_defs()
    approvals = approvals or {}

    for _ in range(max_iterations):
        last = messages[-1] if messages else None
        if last is not None and last.get("role") == "assistant" and last.get("tool_calls"):
            tool_calls = last["tool_calls"]

            pending = [tc for tc in tool_calls if tc["function"]["name"] in GATED_TOOLS and tc["id"] not in approvals]
            if pending:
                return {"status": "pending_approval", "pending": pending, "messages": messages}

            for tc in tool_calls:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                call_id = tc["id"]
                if name in GATED_TOOLS and not approvals.get(call_id, False):
                    result = "Error: denied by user"
                else:
                    try:
                        result = execute_tool(name, arguments)
                    except ToolExecutionError as exc:
                        result = f"Error: {exc}"
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

            approvals = {}
            continue

        raw = model_runtime.create_chat_completion(
            llm, messages=messages, tools=tool_defs, thinking_mode=thinking_mode
        )
        try:
            message = to_openai_message(raw["choices"][0]["message"])
        except UnrecognizedToolCallFormatError as exc:
            raise AgentError(str(exc)) from exc

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return {"status": "final", "message": message}

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

    raise AgentError(f"exceeded {max_iterations} tool-call iterations without a final answer")
