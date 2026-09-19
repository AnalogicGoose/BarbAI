# BarbAI

A local-first AI assistant that runs entirely on your own hardware via
`llama-cpp-python` — a fallback for when cloud tokens/credits run out or
you're offline. Speaks as **BarbAI** (its own persona, not the underlying
model's identity) and supports a `fast` / `thinking` / `extended`
reasoning-effort switch per request.

Full design rationale, hardware-tiering strategy, and phased build plan live
in [`docs/BARBAI_ROADMAP.md`](docs/BARBAI_ROADMAP.md). The sub-phase plan for
Coding mode specifically lives in
[`docs/CODING_AGENT_ROADMAP.md`](docs/CODING_AGENT_ROADMAP.md). Everything
below describes today's **dev-mode** setup (source checkout, `uv`, env
vars) — the plan for turning this into an installable app with a bundled
web UI lives in [`docs/PACKAGING_ROADMAP.md`](docs/PACKAGING_ROADMAP.md).

## Status

Phase 2.0 (core loop), most of Phase 2.2 (tool calling), and Phase 2.3
(memory) are done — see the roadmap doc for the detailed breakdown. In
short: model loading, hardware-tier detection, streaming, tool calling, a
server-side agent loop with six tools — `read_file` (whole-file or a
line range), `list_directory`, and `search` (all three ungated, routine
reads), plus `write_file`, `patch_file` (exact-string search/replace,
cross-platform line-ending-safe), and `remember` (all three gated behind
a human approval step) — opt-in per-conversation session memory, and an explicit-only
global memory that persists across every conversation are all built and
tested. Phase 2.1's other two target laptops (3050, 5070 Ti) still need
real hardware to verify against. Coding mode is planned in sub-phases
(see `docs/CODING_AGENT_ROADMAP.md`, Phase 3.1 partly done — the
read-only tool half); the mode-switching mechanism itself
(`/agent/chat`'s `mode` field, a Coding-specific persona) is built and
tested. No dedicated coding model — decided, not just unfinished: the
obvious pick (Qwen2.5-Coder-7B) is the model this project already found
broken for structured tool calls, so `mode: "coding"` deliberately
reuses Qwen3.5-9B with a different system prompt until real usage shows
that's actually insufficient. `fast`-tier model (Qwen3.5-4B, both modes)
is tested and working — validated on this dev machine by artificially
constraining free VRAM into the `fast` tier's range (see
`scripts/hog_vram.py`), not real 3050 hardware. The Qwen3.5-2B fallback
is also tested and works, but is noticeably lower quality (tool calls
stay reliable, but replies sometimes leak planning-talk instead of
answering directly) — use 4B whenever it fits. There's still no
shell/exec tool.

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

`/agent/chat` additionally accepts `"mode": "general" | "coding"`
(default `"general"`). Switching mode reloads whichever model that mode
points at (`BARBAI_MODEL_PATH` for general, `BARBAI_CODING_MODEL_PATH`
for coding — falls back to the general model until a dedicated coding
model is picked) and swaps in a Coding-specific persona. This reload is
synchronous and happens inline on the first request in a new mode, and
it affects every endpoint, not just `/agent/chat` — the server only ever
has one model loaded at a time. For Coding-mode work, derive
`session_id` from the project/working directory instead of a random ID;
that gives each project its own memory the same way Claude Code scopes
history per project.

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
- [`ripgrep`](https://github.com/BurntSushi/ripgrep) (the `rg` binary) —
  required for the `search` tool. Not a Python dependency, so `uv sync`
  won't install it: `sudo dnf install ripgrep` / `sudo apt install
  ripgrep` / `brew install ripgrep`, or grab a static binary from the
  [releases page](https://github.com/BurntSushi/ripgrep/releases) if you
  don't have root. Everything else works fine without it — only `search`
  itself fails, with a clear error, until it's installed.

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

For tighter-VRAM hardware (the `fast` tier — see
`docs/BARBAI_ROADMAP.md` Phase 2.1), Qwen3.5-4B is tested and
recommended; Qwen3.5-2B works as a fallback if even that's too tight, but
is noticeably lower quality (see that same section):

```
uvx --from huggingface_hub hf download unsloth/Qwen3.5-4B-GGUF --include "Qwen3.5-4B-Q4_K_M.gguf" --local-dir models/
uvx --from huggingface_hub hf download unsloth/Qwen3.5-2B-GGUF --include "Qwen3.5-2B-Q4_K_M.gguf" --local-dir models/
```

A different model will very likely need its own tool-call parsing — see
`core/tool_calls.py`'s module docstring and `docs/BARBAI_ROADMAP.md` section
2 for why, and how to tell if a new model needs the same treatment.

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BARBAI_MODEL_PATH` | `models/Qwen_Qwen3.5-9B-Q4_K_M.gguf` | Path to the GGUF file to load for General mode |
| `BARBAI_CODING_MODEL_PATH` | falls back to `BARBAI_MODEL_PATH` | Path to the GGUF file to load for Coding mode (`/agent/chat`'s `"mode": "coding"`) |
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
