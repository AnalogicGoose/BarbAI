"""
Server-side agent loop: call the model, execute any tool it requests,
feed the result back, repeat until it gives a final answer.

This is what makes BarbAI itself agentic rather than a raw passthrough -
distinct from /v1/chat/completions, /chat, and /v1/messages (which hand a
tool_call back to whoever's calling and stop; correct behavior for
OpenAI/Anthropic compat, where the caller runs its own tools).

Gated tools (write_file, patch_file, run_command, remember) don't execute
immediately - the loop pauses and hands control back to the caller
instead of running them, per the roadmap's approval-gate design (human
review for agent-authored writes, not for routine reads). A paused call
returns {"status": "pending_approval", "pending": [...], "messages": [...]};
the caller resumes by calling run_agent again with that same `messages`
list and an `approvals` dict mapping each pending tool_call_id to
True/False. There's no server-side session store - the caller holds the
paused state between requests.

Because pausing needs to happen *before* a gated tool executes, and
resuming must not re-ask the model for a new turn it already answered,
each iteration does at most one of "call the model" or "resolve the
pending tool call(s)" rather than both - so a full model-call+execute
round now costs two iterations instead of one. MAX_ITERATIONS is set
with that in mind, and bumped from the original 10 to 20 (Phase 3.2):
with seven tools now available and ungated read-investigation chains
(list_directory/search/read_file) realistically running longer than the
three-tool budget this was first tuned against, 10 was proving tight for
legitimate multi-step work, not just runaway loops.

Phase 3.2's other piece: a real "stuck" detector, not just the iteration
count. If the exact same tool call(s) with the exact same arguments
produce the exact same result(s) two rounds in a row, that's a strong
signal nothing is changing - stop and say so plainly
({"status": "stuck", ...}) rather than silently burning the rest of the
iteration budget repeating a dead end. This is a mechanical, cheap check
(compare round signatures), deliberately not an LLM-judged "is this
looping" call - simple and predictable beats clever here.

Explicitly NOT built here: forcing an automatic verification step after
every edit (e.g. "always run tests after patch_file"). There's no
reliable way to know what "verify" means for an arbitrary project
(pytest vs npm test vs cargo test vs ...) without per-project config that
doesn't exist yet, and it turned out not to be necessary - a live test
already showed the model chaining run_command -> read_file -> patch_file
-> run_command on its own, correctly, with the existing tools and no
special scaffolding (see docs/CODING_AGENT_ROADMAP.md Phase 3.1). The
loop already exhibits Observe-Think-Act-Verify emergently; what it
lacked was a way to recognize when that process stalls, which is what
this phase actually adds.

Every model call trims `messages` to fit the context window first
(model_runtime.fit_to_context) - a long session_id conversation or a
tool-heavy loop can grow past n_ctx, and without this the loop crashed
with a bare ValueError surfaced as a raw 500, a real crash a user hit
in practice, not a hypothetical. Only the copy sent to the model is
trimmed; `messages` itself keeps the full history for the return value
and session persistence.
"""

from __future__ import annotations

import json

from barbai.core.tool_calls import UnrecognizedToolCallFormatError, generate_message
from barbai.core.tools import GATED_TOOLS, ToolExecutionError, build_tool_defs, execute_tool

MAX_ITERATIONS = 20

class AgentError(RuntimeError):
    """The loop couldn't produce a final answer (model error or ran out of iterations)."""

def _round_signature(tool_calls: list[dict], results_by_id: dict[str, str]) -> tuple:
    """A comparable fingerprint for one resolved tool-call round: each
    call's name, its arguments, and the result it got back, order-
    independent. Two identical signatures in a row means the model took
    the exact same action and got the exact same outcome - no progress."""
    return tuple(sorted(
        (
            tc["function"]["name"],
            json.dumps(tc["function"]["arguments"], sort_keys=True),
            results_by_id.get(tc["id"], ""),
        )
        for tc in tool_calls
    ))

def _extract_round_signatures(messages: list[dict]) -> list[tuple]:
    """Walk the conversation and pull out every resolved tool-call round's
    signature, in order. A "round" is one assistant tool_calls message
    plus the tool-result messages immediately following it."""
    signatures = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            results_by_id: dict[str, str] = {}
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                results_by_id[messages[j]["tool_call_id"]] = messages[j].get("content", "")
                j += 1
            signatures.append(_round_signature(tool_calls, results_by_id))
            i = j
        else:
            i += 1
    return signatures

def run_agent(
    llm,
    messages: list[dict],
    max_iterations: int = MAX_ITERATIONS,
    thinking_mode: str = "thinking",
    approvals: dict[str, bool] | None = None,
) -> dict:
    """Run the loop. Returns one of:
    {"status": "final", "message": {...}}
    {"status": "pending_approval", "pending": [...], "messages": [...]}
    {"status": "stuck", "message": {...}, "messages": [...]}
    """
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

            signatures = _extract_round_signatures(messages)
            if len(signatures) >= 2 and signatures[-1] == signatures[-2]:
                note = (
                    "I ran the same action and got the same result two rounds in a row, "
                    "so I'm stopping here instead of repeating it again - this needs a "
                    "different approach or a closer look."
                )
                return {"status": "stuck", "message": {"role": "assistant", "content": note}, "messages": messages}

            continue

        # Trim only the copy sent to the model - messages itself keeps
        # accumulating in full and is what gets returned/persisted, so
        # what counts as "still relevant" is re-decided fresh next round
        # rather than permanently discarded from the record on disk.
        # fit_to_context itself can raise (context too small to leave any
        # room for a response at all) - same clean-error treatment as an
        # overflow surfacing from the actual model call below.
        try:
            message, _ = generate_message(llm, messages, thinking_mode=thinking_mode, tools=tool_defs)
        except ValueError as exc:
            raise AgentError(
                f"conversation is too long for the current context window even after trimming ({exc}) - "
                "start a new session or raise BARBAI_N_CTX"
            ) from exc
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
