"""
BarbAI's default identity/persona.

Injected as a system prompt whenever the caller doesn't supply their own -
edit DEFAULT_SYSTEM_PROMPT directly to change how BarbAI introduces itself
or reshape its personality.

Keep instructions here simple and direct. Tested against Qwen3.5-9B: a
nested "don't mention X unless Y, except when Z" conditional reliably sent
it into an unproductive reasoning loop (repeatedly re-litigating the same
constraint instead of ever answering) - a known failure mode for mid-size
reasoning models given conflicting/nuanced constraints. Straightforward,
unconditional instructions don't trigger this.

Phase 3.0 adds a second, Coding-mode identity (CODING_SYSTEM_PROMPT) -
same BarbAI persona underneath, layered with the product differentiator
docs/BARBAI_ROADMAP.md names for Coding mode: explain the fix afterward,
don't just dump a diff. build_system_prompt's `mode` param picks which
base identity to use; the thinking nudge, remembered facts, and any
custom prompt layer on top exactly the same way regardless of mode.
"""

from barbai.core import global_memory

DEFAULT_SYSTEM_PROMPT = (
    "You are BarbAI, a local-first AI assistant that runs entirely on the "
    "user's own hardware. Speak in first person as BarbAI, not as any "
    "other AI brand. Be direct, a little informal, and genuinely helpful. "
    "If asked what model or technology powers you under the hood, mention "
    "briefly that you run on a local open-weight model, then move on."
)

CODING_SYSTEM_PROMPT = (
    "You are BarbAI, a local-first AI assistant that runs entirely on the "
    "user's own hardware, working right now in Coding mode. Speak in "
    "first person as BarbAI, not as any other AI brand. You're not a "
    "snippet generator: read the relevant code before changing it, make "
    "the precise edit, verify it (run the tests/build when you can), and "
    "explain what you changed and why afterward - not just the diff. If "
    "asked what model or technology powers you under the hood, mention "
    "briefly that you run on a local open-weight model, then move on."
)

EXTENDED_THINKING_NUDGE = (
    "For this response, think through the problem thoroughly and from "
    "multiple angles before answering."
)

def build_system_prompt(custom: str | None, thinking_mode: str = "thinking", mode: str = "general") -> str:
    """Combine the base identity (general or coding), the thinking-mode
    nudge, remembered global-memory facts, and a caller-supplied system
    prompt, in that order.

    The base identity always applies. The extended-thinking nudge (only
    added when thinking_mode == "extended") comes next since it shapes how
    the model approaches the whole response, not a specific instruction.
    Remembered facts (core.global_memory - only ever written via an
    explicit user request, see that module) come next, so they're
    available as context but still below identity. Any custom prompt is
    layered on top rather than replacing anything, with an explicit note
    that it takes priority on conflict - verified against Qwen3.5-9B to
    correctly resolve a deliberately conflicting instruction (identity vs.
    forced brevity) in the custom prompt's favor while still keeping the
    BarbAI identity.
    """
    if mode not in ("general", "coding"):
        raise ValueError(f"unknown mode {mode!r}, expected 'general' or 'coding'")

    base = CODING_SYSTEM_PROMPT if mode == "coding" else DEFAULT_SYSTEM_PROMPT
    parts = [base]
    if thinking_mode == "extended":
        parts.append(EXTENDED_THINKING_NUDGE)
    memory_block = global_memory.render_for_prompt()
    if memory_block:
        parts.append(memory_block)
    if custom:
        parts.append(
            "Additional instructions for this conversation, these take "
            f"priority over anything above if they conflict: {custom}"
        )
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(parts)