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
"""

from __future__ import annotations

import json
import re
import uuid

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>]+)>\s*(?P<body>.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)

class UnrecognizedToolCallFormatError(RuntimeError):
    """Content looks like a tool call but matches no known format."""


def _split_reasoning(content: str) -> tuple[str, str | None]:
    if "</think>" not in content:
        return content, None
    reasoning, _, rest = content.partition("</think>")
    return rest, reasoning.replace("<think>", "").strip()

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
        function_match = _FUNCTION_RE.search(match.group(1))
        if not function_match:
            continue
        arguments = {}
        for param_match in _PARAMETER_RE.finditer(function_match.group("body")):
            raw_value = param_match.group("value")
            try:
                value = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                value = raw_value
            arguments[param_match.group("name")] = value
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": function_match.group("name"),
                    "arguments": json.dumps(arguments),
                },
            }
        )

    return {
        "role": "assistant",
        "content": visible_text or None,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }

def to_openai_message(raw_message: dict) -> dict:
    """Normalize a llama-cpp-python chat completion message to OpenAI shape."""
    content = raw_message.get("content") or ""

    if raw_message.get("tool_calls"):
        content, reasoning_content = _split_reasoning(content)
        return {**raw_message, "content": content.strip() or None, "reasoning_content": reasoning_content}

    if "<tool_call>" in content and "<function=" in content:
        return parse_tool_call_tags(content)

    if "<tool_call>" in content:
        raise UnrecognizedToolCallFormatError(
            f"content contains <tool_call> but matches no known format: {content[:200]!r}"
        )

    content, reasoning_content = _split_reasoning(content)
    return {**raw_message, "content": content.strip() or None, "reasoning_content": reasoning_content}