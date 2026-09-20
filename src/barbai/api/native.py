"""
BarbAI's own minimal, native endpoint - no vendor wire-format ceremony.

For projects that don't care about OpenAI/Anthropic compatibility and just
want the shortest path to "send messages, get a reply or a tool call back",
without the id/object/created/choices/usage envelope those formats require.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from barbai.core import memory, model_runtime
from barbai.core.persona import build_system_prompt
from barbai.core.tool_calls import TagStreamParser, UnrecognizedToolCallFormatError, to_openai_message

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
    stream: bool = False
    thinking: Literal["fast", "thinking", "extended"] = "thinking"
    session_id: str | None = None


def _to_llama_messages(messages: list[NativeMessage]) -> list[dict]:
    converted: list[dict] = []
    for m in messages:
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
        converted.append(d)
    return converted


def _to_llama_tools(tools: list[NativeTool] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


def _stream_chat(llm, messages: list[dict], **kwargs) -> Iterator[str]:
    parser = TagStreamParser()

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    raw_stream = model_runtime.create_chat_completion(llm, messages=cast(Any, messages), stream=True, **kwargs)
    for chunk in raw_stream:
        delta = chunk["choices"][0].get("delta", {})

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                yield sse(
                    {
                        "type": "tool_call",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    }
                )
            continue

        content = delta.get("content")
        if content:
            for event in parser.feed(content):
                if event["type"] in ("reasoning", "content"):
                    yield sse({"type": event["type"], "text": event["text"]})
                elif event["type"] == "tool_call":
                    yield sse(
                        {"type": "tool_call", "id": event["id"], "name": event["name"], "arguments": event["arguments"]}
                    )

    for event in parser.finish():
        yield sse({"type": event["type"], "text": event["text"]})

    yield sse({"type": "done"})


@router.post("/chat")
def chat(request: NativeChatRequest):
    try:
        llm = model_runtime.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.session_id and request.stream:
        raise HTTPException(status_code=400, detail="session_id isn't supported with stream=true yet")

    kwargs = {"thinking_mode": request.thinking}
    tools = _to_llama_tools(request.tools)
    if tools:
        kwargs["tools"] = tools

    new_messages = _to_llama_messages(request.messages)
    history = memory.rolling_window(memory.load_session(request.session_id)) if request.session_id else []
    messages = [
        {"role": "system", "content": build_system_prompt(request.system, request.thinking)}
    ] + history + new_messages
    # A long session_id conversation can grow past the context window on
    # its own (see core.model_runtime.fit_to_context's docstring for why
    # this matters - a real crash, not a hypothetical). The streaming
    # path can still fail mid-stream if this estimate runs short, since
    # headers are already sent by the time llama.cpp would raise - a much
    # rarer residual case now than an unguarded call, not eliminated.
    messages = model_runtime.fit_to_context(llm, messages)

    if request.stream:
        return StreamingResponse(_stream_chat(llm, messages, **kwargs), media_type="text/event-stream")

    try:
        raw = model_runtime.create_chat_completion(llm, messages=cast(Any, messages), **kwargs)
    except ValueError as exc:
        raise HTTPException(
            status_code=413,
            detail=(
                f"conversation is too long for the current context window even after trimming ({exc}) - "
                "start a new session or raise BARBAI_N_CTX"
            ),
        ) from exc

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

    if request.session_id:
        to_persist = list(new_messages)
        if message.get("content") is not None:
            to_persist.append({"role": "assistant", "content": message["content"]})
        memory.append_to_session(request.session_id, to_persist)

    return {
        "reply": message.get("content"),
        "reasoning": message.get("reasoning_content"),
        "tool_calls": tool_calls,
    }