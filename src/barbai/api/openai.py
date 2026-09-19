"""
OpenAI-compatible /v1/chat/completions surface.

Matches the OpenAI shape so any existing OpenAI-client tool can point at
this service for free.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.tool_calls import UnrecognizedToolCallFormatError, to_openai_message

router = APIRouter()

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict = {}

class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionDef

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDef] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False

def _prepare_messages(messages: list[ChatMessage]) -> list[dict]:
    """
    Convert request messages to llama-cpp-python's expected shape.

    The OpenAI wire format encodes tool_calls[].function.arguments as a
    JSON *string*; llama-cpp-python's Jinja chat-template rendering expects
    it as a dict (it runs `|items` on it directly) - without this, replaying
    a prior assistant tool call back in conversation history throws inside
    template rendering instead of a normal API error.
    """
    prepared = []
    for m in messages:
        d = m.model_dump(exclude_none=True)
        for tool_call in d.get("tool_calls") or []:
            args = tool_call.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tool_call["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
        prepared.append(d)
    return prepared

@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=501, detail="streaming not implemented yet")

    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    kwargs = {}
    if request.tools:
        kwargs["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice

    messages = _prepare_messages(request.messages)
    # llama-cpp-python's type stubs want its own narrow TypedDict union;
    # plain dicts are what it actually accepts and uses at runtime.
    raw = llm.create_chat_completion(messages=cast(Any, messages), **kwargs)

    try:
        message = to_openai_message(raw["choices"][0]["message"])
    except UnrecognizedToolCallFormatError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    finish_reason = "tool_calls" if message.get("tool_calls") else raw["choices"][0].get("finish_reason", "stop")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": raw.get("usage", {}),
    }