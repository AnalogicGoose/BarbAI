"""
BarbAI's own minimal, native endpoint - no vendor wire-format ceremony.

For projects that don't care about OpenAI/Anthropic compatibility and just
want the shortest path to "send messages, get a reply or a tool call back",
without the id/object/created/choices/usage envelope those formats require.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.tool_calls import UnrecognizedToolCallFormatError, to_openai_message

router = APIRouter()

class NativeToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = {}


class NativeMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[NativeToolCall] | None = None
    tool_call_id: str | None = None

class NativeTool(BaseModel):
    name: str
    description: str | None = None
    parameters: dict = {}


class NativeChatRequest(BaseModel):
    messages: list[NativeMessage]
    system: str | None = None
    tools: list[NativeTool] | None = None

def _to_llama_messages(request: NativeChatRequest) -> list[dict]:
    messages: list[dict] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for m in request.messages:
        d: dict = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        messages.append(d)
    return messages

def _to_llama_tools(tools: list[NativeTool] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]

@router.post("/chat")
def chat(request: NativeChatRequest):
    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    kwargs = {}
    tools = _to_llama_tools(request.tools)
    if tools:
        kwargs["tools"] = tools

    messages = _to_llama_messages(request)
    raw = llm.create_chat_completion(messages=cast(Any, messages), **kwargs)

    try:
        message = to_openai_message(raw["choices"][0]["message"])
    except UnrecognizedToolCallFormatError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    tool_calls = None
    if message.get("tool_calls"):
        tool_calls = [
            {
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"]),
            }
            for tc in message["tool_calls"]
        ]

    return {
        "reply": message.get("content"),
        "reasoning": message.get("reasoning_content"),
        "tool_calls": tool_calls,
    }