"""Anthropic-compatible /v1/messages surface.

Translates Anthropic's Messages API shape - content blocks instead of a
flat string, tool_use/tool_result blocks instead of a separate tool_calls
field, a top-level system prompt, input_schema-named tools, stop_reason
instead of finish_reason - to and from the internal message shape the
model runtime understands (the same shape barbai.api.openai uses).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.tool_calls import TagStreamParser, UnrecognizedToolCallFormatError, to_openai_message

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


_STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens"}


def _stop_reason(llama_finish_reason: str | None, any_tool_call: bool) -> str:
    if any_tool_call:
        return "tool_use"
    return _STOP_REASON_MAP.get(llama_finish_reason or "stop", "end_turn")


def _to_llama_tool_choice(tool_choice: ToolChoice | None) -> Any:
    if tool_choice is None:
        return None
    if tool_choice.type == "tool" and tool_choice.name:
        return {"type": "function", "function": {"name": tool_choice.name}}
    # "any" (must call some tool) has no direct llama-cpp-python equivalent
    # verified to work - falling back to "auto" rather than risk an
    # unsupported value at request time.
    return "auto"


def _stream_messages(llm, llama_messages: list[dict], model_name: str, max_tokens: int, **kwargs) -> Iterator[str]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    parser = TagStreamParser()

    def sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **data})}\n\n"

    yield sse(
        "message_start",
        {
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        },
    )

    block_index = -1
    block_kind: str | None = None

    def open_block(kind: str, content_block: dict) -> str:
        nonlocal block_index, block_kind
        block_index += 1
        block_kind = kind
        return sse("content_block_start", {"index": block_index, "content_block": content_block})

    def close_block() -> str:
        nonlocal block_kind
        ev = sse("content_block_stop", {"index": block_index})
        block_kind = None
        return ev

    def emit_reasoning(text: str) -> Iterator[str]:
        nonlocal block_kind
        if block_kind != "thinking":
            if block_kind is not None:
                yield close_block()
            yield open_block("thinking", {"type": "thinking", "thinking": ""})
        yield sse("content_block_delta", {"index": block_index, "delta": {"type": "thinking_delta", "thinking": text}})

    def emit_content(text: str) -> Iterator[str]:
        nonlocal block_kind
        if block_kind != "text":
            if block_kind is not None:
                yield close_block()
            yield open_block("text", {"type": "text", "text": ""})
        yield sse("content_block_delta", {"index": block_index, "delta": {"type": "text_delta", "text": text}})

    def emit_tool_call(tool_id: str, name: str, arguments: dict | str) -> Iterator[str]:
        nonlocal block_kind
        if block_kind is not None:
            yield close_block()
        yield open_block("tool_use", {"type": "tool_use", "id": tool_id, "name": name, "input": {}})
        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        yield sse("content_block_delta", {"index": block_index, "delta": {"type": "input_json_delta", "partial_json": args_str}})
        yield close_block()

    raw_stream = llm.create_chat_completion(
        messages=cast(Any, llama_messages), max_tokens=max_tokens, stream=True, **kwargs
    )
    any_tool_call = False
    llama_finish_reason = "stop"

    for chunk in raw_stream:
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        if choice.get("finish_reason"):
            llama_finish_reason = choice["finish_reason"]

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                yield from emit_tool_call(tc["id"], tc["function"]["name"], tc["function"]["arguments"])
                any_tool_call = True
            continue

        content = delta.get("content")
        if content:
            for event in parser.feed(content):
                if event["type"] == "reasoning":
                    yield from emit_reasoning(event["text"])
                elif event["type"] == "content":
                    yield from emit_content(event["text"])
                elif event["type"] == "tool_call":
                    yield from emit_tool_call(event["id"], event["name"], event["arguments"])
                    any_tool_call = True

    for event in parser.finish():
        if event["type"] == "reasoning":
            yield from emit_reasoning(event["text"])
        elif event["type"] == "content":
            yield from emit_content(event["text"])

    if block_kind is not None:
        yield close_block()

    stop_reason = _stop_reason(llama_finish_reason, any_tool_call)
    # llama-cpp-python's streaming chunks don't carry per-chunk token usage,
    # so output_tokens is left at 0 rather than fabricated.
    yield sse("message_delta", {"delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 0}})
    yield sse("message_stop", {})


@router.post("/v1/messages")
def messages(request: MessagesRequest):
    if request.stream:
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
        return StreamingResponse(
            _stream_messages(llm, llama_messages, request.model, request.max_tokens, **kwargs),
            media_type="text/event-stream",
        )

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

    stop_reason = _stop_reason(raw["choices"][0].get("finish_reason"), bool(message.get("tool_calls")))
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