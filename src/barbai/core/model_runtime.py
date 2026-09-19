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
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_cpp import Llama

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_N_CTX = 4096

MODES = ("general", "coding")

_model: Llama | None = None
_current_mode: str | None = None

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
    instead of paying for a full GGUF reload.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if _model is not None and _current_mode == mode:
        return _model
    return load_model(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, mode=mode)

THINKING_MODES = ("fast", "thinking", "extended")
_EXTENDED_MIN_MAX_TOKENS = 1500

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