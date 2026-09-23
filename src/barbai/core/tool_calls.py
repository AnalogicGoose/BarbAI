"""
Parses model tool-call output into a provider-agnostic normalized shape.

Different models emit tool calls in different formats depending on how
they were fine-tuned, independent of size/tier - there's no way to know a
new model's format without testing it (see docs/BARBAI_ROADMAP.md section
2). `to_openai_message` tries llama-cpp-python's own parser first (covers
any model using Hermes-style JSON tool calls natively), falls back to the
tag parser below for Qwen3.5-style output, and raises loudly if neither
recognizes the content - so a future model's unfamiliar format surfaces as
an error instead of silently coming back as unparsed prose.

Also normalizes Qwen3.5's <think>...</think> reasoning block out of
`content` into `reasoning_content` on every path (native tool call, tag-
parsed tool call, or plain reply) - not just when a tool call is present.

TagStreamParser does the same tag detection incrementally, for streaming:
fed raw text deltas as they arrive (which can split a tag like </tool_call>
across arbitrary chunk boundaries - llama-cpp-python streams word/sub-word
fragments, not whole tags), it yields reasoning/content/tool_call events
without waiting for the full response.
"""

from __future__ import annotations

import json
import re
import uuid

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>]+)>\s*(?P<body>.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)

_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


class UnrecognizedToolCallFormatError(RuntimeError):
    """Content looks like a tool call but matches no known format."""


class TruncatedToolCallError(UnrecognizedToolCallFormatError):
    """Generation hit the token/context limit before the tool call's
    closing tags arrived - this isn't a real, executable tool call, and
    must never be silently treated as one or shown to the user as if it
    were a normal reply. Subclasses UnrecognizedToolCallFormatError so
    every existing call site that already catches that error handles this
    the same way with no extra except clause needed."""


def _split_reasoning(content: str) -> tuple[str, str | None]:
    if "</think>" not in content:
        return content, None
    reasoning, _, rest = content.partition("</think>")
    return rest, reasoning.replace("<think>", "").strip()


def _parse_function_body(body: str) -> dict | None:
    function_match = _FUNCTION_RE.search(body)
    if not function_match:
        return None
    arguments = {}
    for param_match in _PARAMETER_RE.finditer(function_match.group("body")):
        raw_value = param_match.group("value")
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            value = raw_value
        arguments[param_match.group("name")] = value
    return {"name": function_match.group("name"), "arguments": arguments}


def parse_tool_call_tags(content: str) -> dict:
    """
    Convert Qwen3.5's <tool_call><function=...><parameter=...> tags into
    an OpenAI-compatible chat message dict.
    """
    content, reasoning_content = _split_reasoning(content)

    tool_call_matches = list(_TOOL_CALL_RE.finditer(content))
    if not tool_call_matches:
        return {"role": "assistant", "content": content.strip() or None, "reasoning_content": reasoning_content}

    visible_text = content[: tool_call_matches[0].start()].strip()

    tool_calls = []
    for match in tool_call_matches:
        parsed = _parse_function_body(match.group(1))
        if not parsed:
            continue
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": parsed["name"], "arguments": json.dumps(parsed["arguments"])},
            }
        )

    return {
        "role": "assistant",
        "content": visible_text or None,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


def to_openai_message(raw_message: dict, finish_reason: str | None = None) -> dict:
    """Normalize a llama-cpp-python chat completion message to OpenAI shape.

    finish_reason, when passed, lets this tell a genuinely truncated tool
    call ("length" - the model ran out of room, closing tags never came)
    apart from one whose format is simply unrecognized."""
    content = raw_message.get("content") or ""

    if raw_message.get("tool_calls"):
        content, reasoning_content = _split_reasoning(content)
        return {**raw_message, "content": content.strip() or None, "reasoning_content": reasoning_content}

    if _TOOL_CALL_OPEN in content and _TOOL_CALL_CLOSE not in content and finish_reason == "length":
        raise TruncatedToolCallError(
            "generation hit the token/context limit before the tool call finished - "
            f"reserved_for_response was too small for this response: {content[:200]!r}"
        )

    if "<tool_call>" in content and "<function=" in content:
        return parse_tool_call_tags(content)

    if "<tool_call>" in content:
        raise UnrecognizedToolCallFormatError(
            f"content contains <tool_call> but matches no known format: {content[:200]!r}"
        )

    content, reasoning_content = _split_reasoning(content)
    return {**raw_message, "content": content.strip() or None, "reasoning_content": reasoning_content}


def _safe_split(buffer: str, watch_tags: list[str]) -> tuple[str, str]:
    """
    Split buffer into (flushable, held_back), where held_back is the
    longest suffix of buffer that could still grow into one of watch_tags
    as more text arrives - so a tag never leaks out half-formed.
    """
    max_hold = 0
    for tag in watch_tags:
        for hold_len in range(min(len(tag), len(buffer)), 0, -1):
            if buffer.endswith(tag[:hold_len]):
                max_hold = max(max_hold, hold_len)
                break
    if max_hold == 0:
        return buffer, ""
    return buffer[:-max_hold], buffer[-max_hold:]


class TagStreamParser:
    """
    Incrementally parses Qwen3.5's <think>/<tool_call> tags out of a
    stream of raw text deltas, without waiting for the full response.

    Usage: call feed(chunk) for each raw text delta, then finish() once the
    stream ends to flush anything still buffered.
    """

    def __init__(self) -> None:
        self._state = "reasoning"
        self._buf = ""

    def feed(self, chunk: str) -> list[dict]:
        self._buf += chunk
        events: list[dict] = []
        while True:
            if self._state == "reasoning":
                if _THINK_CLOSE in self._buf:
                    before, _, after = self._buf.partition(_THINK_CLOSE)
                    before = before.replace("<think>", "")
                    self._buf = after
                    self._state = "content"
                    if before:
                        events.append({"type": "reasoning", "text": before})
                    continue
                flush, self._buf = _safe_split(self._buf, [_THINK_CLOSE])
                flush = flush.replace("<think>", "")
                if flush:
                    events.append({"type": "reasoning", "text": flush})
                break

            if self._state == "content":
                if _TOOL_CALL_OPEN in self._buf:
                    before, _, after = self._buf.partition(_TOOL_CALL_OPEN)
                    self._buf = after
                    self._state = "tool_call"
                    if before:
                        events.append({"type": "content", "text": before})
                    continue
                flush, self._buf = _safe_split(self._buf, [_TOOL_CALL_OPEN])
                if flush:
                    events.append({"type": "content", "text": flush})
                break

            if self._state == "tool_call":
                # No partial-tag holdback needed here (unlike the other two
                # states): self._buf just keeps growing across feed() calls
                # until the full close tag shows up in it, so a split close
                # tag is always checked against the complete accumulated
                # text, never just the newest fragment.
                if _TOOL_CALL_CLOSE in self._buf:
                    before, _, after = self._buf.partition(_TOOL_CALL_CLOSE)
                    self._buf = after
                    self._state = "content"
                    parsed = _parse_function_body(before)
                    if parsed:
                        events.append(
                            {
                                "type": "tool_call",
                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                "name": parsed["name"],
                                "arguments": parsed["arguments"],
                            }
                        )
                    continue
                break

        return events

    def finish(self) -> list[dict]:
        events: list[dict] = []
        if self._state == "reasoning" and self._buf:
            events.append({"type": "reasoning", "text": self._buf.replace("<think>", "")})
        elif self._state == "content" and self._buf:
            events.append({"type": "content", "text": self._buf})
        elif self._state == "tool_call" and self._buf:
            # Stream ended mid tool-call - closing tags never arrived.
            # Surface the raw partial text instead of silently dropping
            # it, so at least the caller sees something happened.
            events.append({"type": "content", "text": _TOOL_CALL_OPEN + self._buf})
        self._buf = ""
        return events