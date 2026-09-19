# BarbAI

A local-first AI assistant that runs entirely on your own hardware via
`llama-cpp-python` — a fallback for when cloud tokens/credits run out or
you're offline. Speaks as **BarbAI** (its own persona, not the underlying
model's identity) and supports a `fast` / `thinking` / `extended`
reasoning-effort switch per request.

Full design rationale, hardware-tiering strategy, and phased build plan live
in [`docs/BARBAI_ROADMAP.md`](docs/BARBAI_ROADMAP.md). The sub-phase plan for
Coding mode specifically lives in
[`docs/CODING_AGENT_ROADMAP.md`](docs/CODING_AGENT_ROADMAP.md).

## Status

Phase 2.0 (core loop), most of Phase 2.2 (tool calling), and Phase 2.3
(memory) are done — see the roadmap doc for the detailed breakdown. In
short: model loading, hardware-tier detection, streaming, tool calling, a
server-side agent loop with read/write file tools plus a `remember` tool
(all gated behind a human approval step), opt-in per-conversation session
memory, and an explicit-only global memory that persists across every
conversation are all built and tested. Phase 2.1's other two target
laptops (3050, 5070 Ti) still need real hardware to verify against.
Coding mode is planned in sub-phases (see
`docs/CODING_AGENT_ROADMAP.md`) but not started — no shell/exec tool, no
second model, no mode switching yet.

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

`/agent/chat` can pause mid-loop: if the model requests a gated tool call
(currently `write_file` or `remember`), the response comes back as
`{"status": "pending_approval", "pending": [...], "conversation": [...]}`
instead of a final reply. Resume by calling `/agent/chat` again with that
same `conversation` plus an `approvals` object mapping each pending call's
`id` to `true`/`false` — there's no server-side session, so the caller
holds the paused state between requests.

`/chat` and `/agent/chat` also accept an optional `"session_id"`. Give one
and BarbAI remembers that conversation itself — send only the new
message(s) each turn instead of the full history, same idea as OpenAI's
Responses API `previous_response_id` chaining. Omit it and both endpoints
behave exactly as the stateless passthrough described above. Not
supported yet together with `"stream": true` on `/chat`.

Separately, BarbAI also has **global memory**: a `remember` tool the
model can call (only when you explicitly ask it to, e.g. "remember
that...") to save a short fact that's then shown to it in *every*
conversation afterward, regardless of `session_id`. It's off by default
for auto-writing — nothing is ever remembered unless you ask — and the
whole feature can be switched off with `BARBAI_GLOBAL_MEMORY=off`. Like
`write_file`, `remember` is gated: it pauses for approval before the fact
is actually stored.

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
| `BARBAI_TOOLS_ROOTS` | current working directory | Comma-separated allowlist for the `read_file`/`write_file` tools — mix whole directories (everything inside readable/writable) and individual files (only that exact file readable, not writable as a new-file target) |
| `BARBAI_SESSIONS_DIR` | `./sessions` | Where session-memory JSONL logs are written, one file per `session_id` |
| `BARBAI_GLOBAL_MEMORY` | `on` | Set to `off` to disable the `remember` tool and stop injecting remembered facts into the system prompt |
| `BARBAI_GLOBAL_MEMORY_PATH` | `./global_memory.json` | Where remembered facts are stored |

## Running

```
uv run barbai
```

## Development

```
uv run pytest
```
