"""
Loads and owns the single active llama.cpp model instance.

Only one model is loaded at a time (roadmap: General/Coding modes share one
VRAM slot) - this module is the one place that owns the Llama instance so
nothing else needs to reason about load/unload lifecycle yet.
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_cpp import Llama

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"

_model: Llama | None = None

def get_model() -> Llama:
    model = _model
    if model is None:
        raise RuntimeError("model not loaded - see docs/SETUP.md")
    return model

def load_model(model_path: Path | str | None = None, n_gpu_layers: int = -1, n_ctx: int = 4096) -> Llama:
    global _model
    path = Path(model_path) if model_path else Path(os.environ.get("BARBAI_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(f"model not found at {path} - see docs/SETUP.md")
    model = Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    _model = model
    return model

def unload_model() -> None:
    global _model
    _model = None


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