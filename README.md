# BarbAI

A local-first AI assistant with two modes sharing one model slot: **General**
(conversation, everyday tasks, tool use) and **Coding** (a real agent loop —
read, edit, run, verify). Runs entirely on local hardware via
`llama-cpp-python`, behind an OpenAI-compatible FastAPI service, as a
fallback for when cloud tokens/credits run out or you're offline.

Full design rationale, hardware-tiering strategy, and phased build plan live
in [`docs/BARBAI_ROADMAP.md`](docs/BARBAI_ROADMAP.md).

## Status

Early Phase 2.0 (core loop) — FastAPI skeleton and hardware-detection module
are being wired up. No model is loaded/served yet.

## Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/) — this project uses uv exclusively for
  dependency and environment management. No `pip`.
- A GPU for accelerated inference (optional but recommended) — NVIDIA
  (Linux/Windows, CUDA), Apple Silicon (macOS, Metal). Works CPU-only
  without one, just slower.

## Setup

```
uv sync
```

This works on Linux, macOS, and Windows out of the box (CPU-only). For GPU
acceleration, see the platform-specific steps in
[`docs/SETUP.md`](docs/SETUP.md).

## Running

```
uv run barbai
```

## Development

```
uv run pytest
```
