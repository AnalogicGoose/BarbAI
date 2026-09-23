"""
BarbAI's own agentic endpoint - runs the tool-execution loop server-side.

Unlike /v1/chat/completions, /chat, and /v1/messages (which hand a
tool_call back to the caller and stop - correct behavior for OpenAI/
Anthropic compat, where the caller runs its own tools), this endpoint
executes tools itself and returns once the model has a final answer, a
gated tool call needs human approval, or (Phase 3.2) the loop detects
it's stuck - the same tool call producing the same result two rounds in
a row - and stops on its own rather than burning the rest of its
iteration budget repeating a dead end. See core.agent's module docstring
for the stuck-detection logic itself.

Phase 3.0: this is the one mode-aware endpoint. `mode` ("general" or
"coding", default "general") picks the model and persona via
model_runtime.ensure_mode()/core.persona. A mode switch is a real,
synchronous GGUF load that happens inline on this request if the
requested mode isn't already active - see model_runtime's module
docstring for the "this affects every other endpoint too" gotcha.

Four ways to call it:
  - Start a fresh, one-off conversation: send `messages` (role/content
    pairs), no `session_id`. The persona system prompt is built and
    prepended automatically. Nothing is remembered afterward.
  - Start or continue a remembered conversation: send `messages` (just
    the new turn(s)) plus a `session_id`. BarbAI loads that session's
    prior history (bounded by a rolling window - see core.memory),
    prepends it, and appends the new turn(s) plus its reply back to the
    session log once it has a final answer. Reuse the same `session_id`
    next time to continue it. For Coding-mode work, derive session_id
    from the project/working directory (e.g. a hash of its path) rather
    than a random id - that's what gives each project its own memory,
    the same way Claude Code scopes history per project, without needing
    any separate storage mechanism (see docs/CODING_AGENT_ROADMAP.md
    Phase 3.0).
  - Resume a paused one: send back the `conversation` this endpoint
    returned from a prior `"status": "pending_approval"` response,
    unchanged, plus `approvals` mapping each pending tool_call_id to
    true/false. This is a separate, shorter-lived mechanism from
    `session_id` - there's no server-side state for a pending approval
    either, the caller holds `conversation` between requests the same
    way.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from barbai.core import memory, model_runtime
from barbai.core.agent import AgentError, run_agent
from barbai.core.persona import build_system_prompt

router = APIRouter()

class AgentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AgentChatRequest(BaseModel):
    messages: list[AgentMessage] | None = None
    conversation: list[dict] | None = None
    approvals: dict[str, bool] | None = None
    system: str | None = None
    thinking: Literal["fast", "thinking", "extended"] = "thinking"
    session_id: str | None = None
    mode: Literal["general", "coding"] = "general"

@router.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    try:
        llm = model_runtime.ensure_mode(request.mode)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    new_messages: list[dict] = []
    if request.conversation is not None:
        messages = request.conversation
    elif request.messages is not None:
        new_messages = [m.model_dump() for m in request.messages]
        history = memory.session_replay(llm, request.session_id) if request.session_id else []
        messages = [
            {"role": "system", "content": build_system_prompt(request.system, request.thinking, request.mode)}
        ] + history + new_messages
    else:
        raise HTTPException(
            status_code=400,
            detail="provide either 'messages' (new conversation) or 'conversation' (resuming a pending_approval)",
        )

    try:
        result = run_agent(llm, messages, thinking_mode=request.thinking, approvals=request.approvals)
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if result["status"] == "pending_approval":
        return {
            "status": "pending_approval",
            "pending": [
                {"id": tc["id"], "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                for tc in result["pending"]
            ],
            "conversation": result["messages"],
        }

    message = result["message"]

    if request.session_id and request.conversation is None:
        to_persist = list(new_messages)
        if message.get("content") is not None:
            to_persist.append({"role": "assistant", "content": message["content"]})
        memory.append_to_session(request.session_id, to_persist)

    if result["status"] == "stuck":
        return {
            "status": "stuck",
            "reply": message.get("content"),
            "conversation": result["messages"],
        }

    return {
        "status": "final",
        "reply": message.get("content"),
        "reasoning": message.get("reasoning_content"),
    }