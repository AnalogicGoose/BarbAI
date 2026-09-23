"""
Memory V2 (roadmap Phase 2.5): folds session turns that are about to fall
out of the replay window into a running summary, instead of
core.memory.rolling_window() just dropping them with no trace (v1).

Deliberately its own small model call, not reusing the live conversation -
summarizing is a distinct task from answering, benefits from its own
focused prompt, and doesn't need the reasoning pass a live reply might
(thinking_mode="fast"). Routed through the same
fit_to_context/create_chat_completion/to_openai_message path as every
other generation, so a summarization call gets the same context-fit and
truncation-detection guarantees (see Phase 2.3's truncation fix) - a
summary is itself a generation that can run long and hit the same limits.
"""

from __future__ import annotations

from barbai.core import model_runtime
from barbai.core.tool_calls import to_openai_message

_SUMMARY_SYSTEM_PROMPT = (
    "You are condensing an earlier part of a conversation into a short, "
    "dense summary for your own later reference - not a reply to the "
    "user. Preserve names, decisions, facts, and anything the user asked "
    "you to remember or do. Drop pleasantries and restating the obvious. "
    "Write plain prose, a few sentences to a short paragraph - not a "
    "list of every message."
)


def _render_prompt(prior_summary: str | None, new_messages: list[dict]) -> str:
    turns = "\n".join(f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in new_messages)
    if prior_summary:
        return (
            f"Existing summary of the conversation so far:\n{prior_summary}\n\n"
            f"New turns to fold in:\n{turns}\n\n"
            "Write the updated summary, incorporating the new turns."
        )
    return f"Turns to summarize:\n{turns}\n\nWrite the summary."


def summarize_turns(llm, prior_summary: str | None, new_messages: list[dict]) -> str:
    """Produce an updated running summary covering `prior_summary` (if
    any) plus `new_messages`. Raises like any other model call - the
    caller (core.memory.session_replay) decides how to degrade if this
    fails, this function doesn't hide errors itself."""
    prompt_messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _render_prompt(prior_summary, new_messages)},
    ]
    trimmed = model_runtime.fit_to_context(llm, prompt_messages)
    raw = model_runtime.create_chat_completion(llm, messages=trimmed, thinking_mode="fast")
    message = to_openai_message(raw["choices"][0]["message"], finish_reason=raw["choices"][0].get("finish_reason"))
    return (message.get("content") or "").strip()
