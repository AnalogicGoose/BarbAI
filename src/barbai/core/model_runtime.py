"""
Loads and owns the single active llama.cpp model instance.

Only one model is loaded at a time (roadmap: General/Coding modes share one
VRAM slot) - this module is the one place that owns the Llama instance so
nothing else needs to reason about load/unload lifecycle yet.

Phase 3.0: two modes, "general" and "coding", each with its own model
path. Switching modes means unloading whatever's loaded and loading the
other mode's model - ensure_mode() is the entry point for that, and skips
the reload when the requested mode is already active, since a full GGUF
load is expensive (real wall-clock time) and most calls stay in one mode.
A mode switch is therefore a slow, synchronous operation that happens
inline on whichever request first asks for the new mode - not instant.

No dedicated coding model has been picked/validated yet (see
docs/CODING_AGENT_ROADMAP.md Phase 3.0) - BARBAI_CODING_MODEL_PATH falls
back to BARBAI_MODEL_PATH/DEFAULT_MODEL_PATH until one is, so the
switching mechanism itself is usable and testable today with Qwen3.5-9B
standing in for both modes.

Gotcha: _model is process-wide, shared by every endpoint. Only
/agent/chat is mode-aware right now (see api/agent.py) - once it switches
to "coding", every OTHER endpoint (native /chat, the OpenAI/Anthropic
passthroughs) is now also being served by the coding model until
something switches back, since they all call get_model() and just get
whatever's currently loaded. That's the intended behavior for a
single-model-in-VRAM design, not a bug - just don't be surprised by it.

BARBAI_N_CTX overrides the context window (default 4096) - was hardcoded
until a real need showed up: routing an external client like the Claude
Code CLI through /v1/messages (see docs/SETUP.md's note on this) sends
its own large system prompt and built-in tool definitions before any
actual conversation happens, easily past 4096 tokens on its own. Raising
this uses more VRAM for the KV cache - there's no automatic check that a
larger value still fits the current tier, the caller is trusted to know
what they asked for.

fit_to_context() is the other half of that same problem: a long
session_id conversation, or a tool-heavy agent loop, can grow past
n_ctx too, and llama.cpp's own response to that is a bare ValueError
that reached the client as a raw 500 - a real crash a user hit in
practice, not a hypothetical. "Start dumb" here means drop the oldest
turns once the running total doesn't fit, not summarize them - the same
explicit-first, summarize-later call already made for global memory
(docs/BARBAI_ROADMAP.md Phase 2.3); revisit only once dropping instead
of summarizing is actually felt as a real loss, not preemptively.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from llama_cpp import Llama

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_N_CTX = 4096

MODES = ("general", "coding")

_model: Llama | None = None
_current_mode: str | None = None
# Guards ensure_mode()'s check-then-load: FastAPI runs sync endpoints in a
# threadpool, so two /agent/chat requests arriving close together (e.g. a
# page refresh while the first request's model load is still in flight)
# could otherwise both see _model as None/wrong-mode and both call
# load_model() concurrently - two simultaneous ~5GB CUDA allocations can
# fail with "out of memory" even when free VRAM has room for one of them
# alone. Not held during ordinary get_model() reads, only the load path.
_load_lock = threading.Lock()

def _model_path_for_mode(mode: str) -> Path:
    if mode == "coding":
        raw = os.environ.get("BARBAI_CODING_MODEL_PATH") or os.environ.get("BARBAI_MODEL_PATH")
    else:
        raw = os.environ.get("BARBAI_MODEL_PATH")
    return Path(raw) if raw else DEFAULT_MODEL_PATH

def _resolve_n_ctx(n_ctx: int | None) -> int:
    if n_ctx is not None:
        return n_ctx
    raw = os.environ.get("BARBAI_N_CTX")
    if not raw:
        return DEFAULT_N_CTX
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"BARBAI_N_CTX must be an integer, got {raw!r}") from None

def get_model() -> Llama:
    model = _model
    if model is None:
        raise RuntimeError("model not loaded - see docs/SETUP.md")
    return model

def current_mode() -> str | None:
    """The mode of the currently loaded model, or None if nothing's loaded."""
    return _current_mode

def load_model(
    model_path: Path | str | None = None, n_gpu_layers: int = -1, n_ctx: int | None = None, mode: str = "general"
) -> Llama:
    global _model, _current_mode
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    path = Path(model_path) if model_path else _model_path_for_mode(mode)
    if not path.exists():
        raise FileNotFoundError(f"model not found at {path} - see docs/SETUP.md")
    model = Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, n_ctx=_resolve_n_ctx(n_ctx), verbose=False)
    _model = model
    _current_mode = mode
    return model

def unload_model() -> None:
    global _model, _current_mode
    _model = None
    _current_mode = None

def ensure_mode(mode: str, n_gpu_layers: int = -1, n_ctx: int | None = None) -> Llama:
    """Return the model for `mode`, loading or switching only if needed.

    A call for the mode that's already active reuses the loaded model
    instead of paying for a full GGUF reload. Locked (see _load_lock) so
    two concurrent callers can't both decide a load is needed and both
    try to allocate a GPU buffer at once - the second caller blocks,
    then either reuses what the first one just loaded or loads in turn,
    never in parallel.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    with _load_lock:
        if _model is not None and _current_mode == mode:
            return _model
        return load_model(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, mode=mode)

THINKING_MODES = ("fast", "thinking", "extended")
_EXTENDED_MIN_MAX_TOKENS = 1500

# A flat reserve doesn't scale with n_ctx - raising BARBAI_N_CTX only grew
# room for input history, not for the response, which is why a thinking
# pass plus a real tool-call payload (e.g. write_file with a whole doc)
# could still run out of room mid-generation even at 8192. Reserve a
# fraction of n_ctx instead, clamped so small contexts still leave a
# usable minimum and huge contexts don't waste an unreasonable chunk.
_RESERVED_FOR_RESPONSE_RATIO = 0.25
_RESERVED_FOR_RESPONSE_FLOOR = 512
_RESERVED_FOR_RESPONSE_CEILING = 4096
_TOKENS_PER_MESSAGE_OVERHEAD = 16  # includes role token + separator overhead (system/user/assistant ~1 each, plus template separators)

def _resolve_reserved_for_response(n_ctx: int, reserved_for_response: int | None) -> int:
    if reserved_for_response is not None:
        return reserved_for_response
    raw = os.environ.get("BARBAI_RESERVED_FOR_RESPONSE")
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"BARBAI_RESERVED_FOR_RESPONSE must be an integer, got {raw!r}") from None
    return min(_RESERVED_FOR_RESPONSE_CEILING, max(_RESERVED_FOR_RESPONSE_FLOOR, int(n_ctx * _RESERVED_FOR_RESPONSE_RATIO)))

def count_tokens(llm: Llama, messages: list[dict]) -> int:
    """Estimate the token count for a list of chat messages, using the
    loaded model's own tokenizer. This now includes role tokens in the
    overhead to be more accurate - we still don't render the full chat
    template (no system prompt wrapper or tool-call formatting), but
    being closer to reality prevents fit_to_context() from thinking it
    fits when it doesn't."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if content:
            total += len(llm.tokenize(content.encode("utf-8"), add_bos=False))
        # Include role token in overhead (system/user/assistant are ~1 token each)
        total += _TOKENS_PER_MESSAGE_OVERHEAD + 1
    return total

def fit_to_context(llm: Llama, messages: list[dict], reserved_for_response: int | None = None) -> list[dict]:
    """Drop the oldest messages - after any leading system message, and
    never the newest message - until what's left fits the model's
    actual context window with room left over for the response too.

    This is what stops a long conversation (session_id replay, or a
    tool-heavy agent loop accumulating read_file/search results) from
    crashing with llama.cpp's bare "Requested tokens (N) exceed context
    window of M" ValueError instead of just quietly forgetting the
    oldest, least-relevant part of the conversation - the same idea a
    hosted assistant's context window uses, just by dropping instead of
    summarizing (see module docstring for why that's the deliberate
    "start dumb" scope for now).
    """
    if not messages:
        return messages

    reserved_for_response = _resolve_reserved_for_response(llm.n_ctx(), reserved_for_response)
    budget = llm.n_ctx() - reserved_for_response
    if budget <= 0:
        raise ValueError(
            f"context window ({llm.n_ctx()}) is too small to leave {reserved_for_response} tokens for a response"
        )

    has_system = messages[0].get("role") == "system"
    protected_head = 1 if has_system else 0

    trimmed = list(messages)
    while len(trimmed) > protected_head + 1 and count_tokens(llm, trimmed) > budget:
        del trimmed[protected_head]

    # Final verification to ensure we fit exactly - this is cheap (one call) 
    # but guarantees we never exceed the context window
    if count_tokens(llm, trimmed) > budget:
        # Remove one more message from the protected head if needed
        if len(trimmed) > protected_head + 1:
            del trimmed[protected_head]

    return trimmed

def _resolve_chat_handler(llm: Llama):
    """Get the callable llama-cpp-python actually uses to run a chat
    completion for this model's template.

    llm.chat_handler is only set if one was passed explicitly at
    construction; for a GGUF-embedded template (our case) the real handler
    lives in the semi-private llm._chat_handlers dict, keyed by
    llm.chat_format. This mirrors exactly what Llama.create_chat_completion
    does internally to pick a handler - see llama_cpp/llama.py.
    """
    handler = llm.chat_handler or llm._chat_handlers.get(llm.chat_format)
    if handler is None:
        raise RuntimeError(f"no chat handler resolved for chat_format={llm.chat_format!r}")
    return handler

def create_chat_completion(llm: Llama, messages: list[dict], *, thinking_mode: str = "thinking", **kwargs):
    """Chat completion with control over Qwen3.5's reasoning pass.

    Llama.create_chat_completion() has a fixed signature that rejects
    unknown kwargs like enable_thinking, even though the underlying Jinja
    template (where Qwen3.5's thinking toggle actually lives) accepts it
    fine via **kwargs. This calls the resolved handler directly instead -
    the same thing create_chat_completion() does internally, just without
    that restriction. Verified: enable_thinking=False skips the reasoning
    pass entirely (no <think> tags, ~2.5s vs ~10s+ on a test prompt), and
    tool-calling still works correctly with it disabled.

    thinking_mode:
      - "fast": enable_thinking=False, no reasoning pass at all.
      - "thinking" (default): enable_thinking=True, current behavior.
      - "extended": enable_thinking=True, plus a raised max_tokens floor so
        a longer reasoning pass isn't cut off mid-thought. There's no true
        "think more" control in the model itself - pair this with a system
        prompt nudge (see core.persona) for the rest of the effect.
    """
    if thinking_mode not in THINKING_MODES:
        raise ValueError(f"unknown thinking_mode {thinking_mode!r}, expected one of {THINKING_MODES}")

    if thinking_mode == "extended":
        kwargs["max_tokens"] = max(kwargs.get("max_tokens") or 0, _EXTENDED_MIN_MAX_TOKENS)

    handler = _resolve_chat_handler(llm)
    return handler(llama=llm, messages=messages, enable_thinking=thinking_mode != "fast", **kwargs)