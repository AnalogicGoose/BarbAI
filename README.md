# BarbAI

A local-first AI assistant that runs entirely on your own hardware via
`llama-cpp-python` — a fallback for when cloud tokens/credits run out or
you're offline. Speaks as **BarbAI** (its own persona, not the underlying
model's identity) and supports a `fast` / `thinking` / `extended`
reasoning-effort switch per request.

Full design rationale, hardware-tiering strategy, and phased build plan live
in [`docs/BARBAI_ROADMAP.md`](docs/BARBAI_ROADMAP.md).

## Status

Phase 2.0 (core loop) and most of Phase 2.2 (tool calling) are done — see
the roadmap doc for the detailed breakdown. In short: model loading,
hardware-tier detection, streaming, tool calling, and a server-side agent
loop with a read-only file tool are all built and tested. Phase 2.1's other
two target laptops (3050, 5070 Ti) still need real hardware to verify
against; Coding mode and write/shell tools haven't been started.

## API

Four endpoints, all backed by the same loaded model:

| Endpoint | Shape | Behavior |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible | Passthrough — hands a `tool_call` back to the caller to execute |
| `POST /v1/messages` | Anthropic-compatible | Passthrough, same as above |
| `POST /chat` | BarbAI's own minimal shape | Passthrough, no vendor envelope |
| `POST /agent/chat` | BarbAI's own minimal shape | **Agentic** — executes tool calls itself server-side and loops to a final answer |

All support `"stream": true` (SSE) except `/agent/chat`. All accept an
optional `"thinking": "fast" | "thinking" | "extended"` field (default
`"thinking"`).

Easiest way to try any of them: start the server, then open
`http://127.0.0.1:8000/docs` for FastAPI's interactive Swagger UI.

## Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/) — this project uses uv exclusively for
  dependency and environment management. No `pip`.
- A GPU for accelerated inference (optional but recommended) — NVIDIA
  (Linux/Windows, CUDA), Apple Silicon (macOS, Metal). Works CPU-only
  without one, just slower.
- A GGUF model file (see below).

## Setup

```
uv sync
```

This works on Linux, macOS, and Windows out of the box (CPU-only). For GPU
acceleration, see the platform-specific steps in
[`docs/SETUP.md`](docs/SETUP.md).

## Model

BarbAI doesn't bundle a model — point it at a GGUF file yourself. The build
so far has been developed and tested against
[`bartowski/Qwen_Qwen3.5-9B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF)
(`Q4_K_M` quant, ~5.8GB), which fits an 8GB card with headroom:

```
uvx --from huggingface_hub hf download bartowski/Qwen_Qwen3.5-9B-GGUF --include "Qwen_Qwen3.5-9B-Q4_K_M.gguf" --local-dir models/
```

By default BarbAI looks for `models/Qwen_Qwen3.5-9B-Q4_K_M.gguf` relative to
the project root. Point it elsewhere with `BARBAI_MODEL_PATH`.

A different model will very likely need its own tool-call parsing — see
`core/tool_calls.py`'s module docstring and `docs/BARBAI_ROADMAP.md` section
2 for why, and how to tell if a new model needs the same treatment.

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BARBAI_MODEL_PATH` | `models/Qwen_Qwen3.5-9B-Q4_K_M.gguf` | Path to the GGUF file to load |
| `BARBAI_TOOLS_ROOTS` | current working directory | Comma-separated allowlist for the `read_file` tool — mix whole directories (everything inside readable) and individual files (only that exact file readable) |

## Running

```
uv run barbai
```

## Development

```
uv run pytest
```
