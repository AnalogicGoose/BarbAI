"""
Shared FastAPI app: lifecycle (hardware detection + model load) and health.

Provider-specific surfaces (OpenAI, Anthropic, native, ...) are routers
mounted onto this one app, so they all share one loaded model and one
lifespan instead of each managing their own.

GET / serves a single static HTML page (static/chat.html) - a throwaway
chat UI, not the real bundled frontend docs/PACKAGING_ROADMAP.md Phase
4.0 describes. Served from this same app specifically so it's same-origin
with /agent/chat: no CORS middleware needed, no separate process to run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from barbai.api.native import router as native_router
from barbai.api.openai import router as openai_router
from barbai.api.anthropic import router as anthropic_router
from barbai.api.agent import router as agent_router
from barbai.core import model_runtime
from barbai.core.hardware import InsufficientVramError, NoGpuDetectedError, Tier, detect_tier

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_hardware_tier: Tier | None = None
_hardware_error: str | None = None
_model_error: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hardware_tier, _hardware_error, _model_error
    try:
        _hardware_tier = detect_tier()
    except (NoGpuDetectedError, InsufficientVramError) as exc:
        _hardware_error = str(exc)

    try:
        model_runtime.load_model()
    except FileNotFoundError as exc:
        _model_error = str(exc)

    yield

app = FastAPI(title="BarbAI", version="0.1.0", lifespan=lifespan)
app.include_router(native_router)
app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(agent_router)

@app.get("/")
def chat_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "chat.html")

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "hardware_tier": _hardware_tier.name if _hardware_tier else None,
        "hardware_error": _hardware_error,
        "model_loaded": _model_error is None,
        "model_error": _model_error,
    }