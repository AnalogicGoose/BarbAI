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