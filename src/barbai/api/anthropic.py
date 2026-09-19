"""
Anthropic-compatible /v1/messages surface.

Translates Anthropic's Messages API shape - content blocks instead of a
flat string, tool_use/tool_result blocks instead of a separate tool_calls
field, a top-level system prompt, input_schema-named tools, stop_reason
instead of finish_reason - to and from the internal message shape the
model runtime understands (the same shape barbai.api.openai uses).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.tool_calls import UnrecognizedToolCallFormatError, to_openai_message

router = APIRouter()

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict = {}

class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"] = "auto"
    name: str | None = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | None = None
    stream: bool = False
    temperature: float | None = None

def _block_text(block: dict) -> str:
    return block.get("text", "")

def _to_llama_messages(request: MessagesRequest) -> list[dict]:
    messages: list[dict] = []

    if request.system:
        system_text = (
            request.system
            if isinstance(request.system, str)
            else "\n".join(b.get("text", "") for b in request.system if b.get("type") == "text")
        )
        messages.append({"role": "system", "content": system_text})

    for m in request.messages:
        if isinstance(m.content, str):
            messages.append({"role": m.role, "content": m.content})
            continue

        text_parts = []
        tool_calls = []
        tool_result_entries = []

        for block in m.content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": block.get("input", {})},
                    }
                )
            elif btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(_block_text(b) for b in content if b.get("type") == "text")
                tool_result_entries.append((block["tool_use_id"], content))

        if tool_result_entries:
            # Anthropic sends tool results as content blocks inside a "user"
            # message; the model runtime expects one "tool" role message per
            # result instead.
            for tool_use_id, content in tool_result_entries:
                messages.append({"role": "tool", "tool_call_id": tool_use_id, "content": content})
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        entry: dict = {"role": m.role}
        if text_parts:
            entry["content"] = "\n".join(text_parts)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        messages.append(entry)

    return messages

def _to_llama_tools(tools: list[AnthropicTool] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]

def _to_llama_tool_choice(tool_choice: ToolChoice | None) -> Any:
    if tool_choice is None:
        return None
    if tool_choice.type == "tool" and tool_choice.name:
        return {"type": "function", "function": {"name": tool_choice.name}}
    # "any" (must call some tool) has no direct llama-cpp-python equivalent
    # verified to work - falling back to "auto" rather than risk an
    # unsupported value at request time.
    return "auto"

@router.post("/v1/messages")
def messages(request: MessagesRequest):
    if request.stream:
        raise HTTPException(status_code=501, detail="streaming not implemented yet")

    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    kwargs: dict = {}
    tools = _to_llama_tools(request.tools)
    if tools:
        kwargs["tools"] = tools
    tool_choice = _to_llama_tool_choice(request.tool_choice)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    llama_messages = _to_llama_messages(request)
    raw = llm.create_chat_completion(
        messages=cast(Any, llama_messages),
        max_tokens=request.max_tokens,
        **kwargs,
    )

    try:
        message = to_openai_message(raw["choices"][0]["message"])
    except UnrecognizedToolCallFormatError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    content: list[dict] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        content.append(
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            }
        )

    stop_reason = "tool_use" if message.get("tool_calls") else "end_turn"
    usage = raw.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }