"""
BarbAI's own agentic endpoint - runs the tool-execution loop server-side.

Unlike /v1/chat/completions, /chat, and /v1/messages (which hand a
tool_call back to the caller and stop - correct behavior for OpenAI/
Anthropic compat, where the caller runs its own tools), this endpoint
executes tools itself and only returns once the model has a final answer.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.agent import AgentError, run_agent

router = APIRouter()

class AgentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AgentChatRequest(BaseModel):
    messages: list[AgentMessage]
    system: str | None = None

@router.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    messages: list[dict] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend(m.model_dump() for m in request.messages)

    try:
        final = run_agent(llm, messages)
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "reply": final.get("content"),
        "reasoning": final.get("reasoning_content"),
    }