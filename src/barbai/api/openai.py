"""
OpenAI-compatible /v1/chat/completions surface.

Matches the OpenAI shape so any existing OpenAI-client tool can point at
this service for free.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from barbai.core import model_runtime
from barbai.core.persona import build_system_prompt
from barbai.core.tool_calls import TagStreamParser, UnrecognizedToolCallFormatError, to_openai_message

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
    thinking: Literal["fast", "thinking", "extended"] = "thinking"


def _prepare_messages(messages: list[ChatMessage], thinking_mode: str) -> list[dict]:
    """
    Convert request messages to llama-cpp-python's expected shape.

    The OpenAI wire format encodes tool_calls[].function.arguments as a
    JSON *string*; llama-cpp-python's Jinja chat-template rendering expects
    it as a dict (it runs `|items` on it directly) - without this, replaying
    a prior assistant tool call back in conversation history throws inside
    template rendering instead of a normal API error.
    """
    prepared = []
    system_found = False
    for m in messages:
        d = m.model_dump(exclude_none=True)
        if d.get("role") == "system" and not system_found:
            d["content"] = build_system_prompt(d.get("content"), thinking_mode)
            system_found = True
        for tool_call in d.get("tool_calls") or []:
            args = tool_call.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tool_call["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
        prepared.append(d)
    if not system_found:
        prepared.insert(0, {"role": "system", "content": build_system_prompt(None, thinking_mode)})
    return prepared


def _stream_chat_completions(llm, messages, model_name: str, **kwargs) -> Iterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    parser = TagStreamParser()
    tool_call_index = 0
    emitted_tool_call = False

    def sse(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield sse({"role": "assistant"})

    raw_stream = model_runtime.create_chat_completion(llm, messages=cast(Any, messages), stream=True, **kwargs)
    finish_reason = "stop"
    for chunk in raw_stream:
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})

        if delta.get("tool_calls"):
            # native path: the model/parser already gave structured calls,
            # bypass the tag state machine entirely for this chunk.
            for tc in delta["tool_calls"]:
                yield sse({"tool_calls": [{"index": tool_call_index, **tc}]})
                tool_call_index += 1
                emitted_tool_call = True
            continue

        content = delta.get("content")
        if content:
            for event in parser.feed(content):
                if event["type"] == "reasoning":
                    yield sse({"reasoning_content": event["text"]})
                elif event["type"] == "content":
                    yield sse({"content": event["text"]})
                elif event["type"] == "tool_call":
                    yield sse(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": event["id"],
                                    "type": "function",
                                    "function": {
                                        "name": event["name"],
                                        "arguments": json.dumps(event["arguments"]),
                                    },
                                }
                            ]
                        }
                    )
                    tool_call_index += 1
                    emitted_tool_call = True

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    for event in parser.finish():
        if event["type"] == "reasoning":
            yield sse({"reasoning_content": event["text"]})
        elif event["type"] == "content":
            yield sse({"content": event["text"]})

    if emitted_tool_call:
        finish_reason = "tool_calls"
    yield sse({}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    kwargs = {"thinking_mode": request.thinking}
    if request.tools:
        kwargs["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice

    messages = _prepare_messages(request.messages, request.thinking)

    if request.stream:
        return StreamingResponse(
            _stream_chat_completions(llm, messages, request.model, **kwargs),
            media_type="text/event-stream",
        )

    # llama-cpp-python's type stubs want its own narrow TypedDict union;
    # plain dicts are what it actually accepts and uses at runtime.
    raw = model_runtime.create_chat_completion(llm, messages=cast(Any, messages), **kwargs)

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