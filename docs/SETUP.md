# Setup

BarbAI runs anywhere `uv` and Python >= 3.14 do. GPU acceleration is
optional but strongly recommended — `llama-cpp-python` falls back to
CPU-only if it can't build against a GPU toolchain, which works but is much
slower.

## 1. Base install (all platforms)

```
uv sync
```

This gets you a working CPU-only build. Verify with:

```
uv run python -c "import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

If that prints `False`, follow the section for your OS/GPU below, then
re-run the same check — it should flip to `True`.

## 2. GPU acceleration

### Linux + NVIDIA (CUDA)

1. Install the NVIDIA proprietary driver (RPM Fusion on Fedora, or your
   distro's equivalent) — this alone is **not** enough, it only gives you
   the runtime driver, not the compiler.
2. Install the full CUDA Toolkit from NVIDIA directly (distro package
   managers generally don't ship it):
   - https://developer.nvidia.com/cuda-downloads — pick your distro, use
     the "network" (repo) install method, e.g. on Fedora:
     ```
     sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/<fedoraNN>/x86_64/cuda-<fedoraNN>.repo
     sudo dnf install -y cuda-toolkit
     ```
     (substitute the fedora version NVIDIA currently supports; if the exact
     release isn't published yet, the most recent available one usually
     still works)
3. Make sure `nvcc` is findable. `/usr/local/cuda/bin` is the usual
   location; add it to `PATH` in your shell's rc file, or just point CMake
   at it directly for the build (works regardless of shell):
   ```
   CMAKE_ARGS="-DGGML_CUDA=on" CUDACXX=/usr/local/cuda/bin/nvcc \
     uv add llama-cpp-python --reinstall-package llama-cpp-python --no-cache
   ```

AMD GPUs (ROCm/`GGML_HIP`) aren't covered here — not tested against this
project yet.

### Windows + NVIDIA (CUDA)

1. Install **Visual Studio Build Tools** (or full Visual Studio) with the
   "Desktop development with C++" workload — this provides the MSVC
   compiler and CMake needs it to configure the build.
2. Install the CUDA Toolkit for Windows:
   https://developer.nvidia.com/cuda-downloads
3. Build from a shell that has the MSVC compiler on `PATH` — the
   "**x64 Native Tools Command Prompt for VS**" (or `Developer PowerShell
   for VS`) from the Start menu, not a plain terminal. From there:
   ```powershell
   $env:CMAKE_ARGS = "-DGGML_CUDA=on"
   uv add llama-cpp-python --reinstall-package llama-cpp-python --no-cache
   ```

### macOS (Apple Silicon)

Metal is llama.cpp's GPU backend here, and it's on by default — a plain
`uv sync` should already build with Metal support with no extra flags. If
you need to force a rebuild after a `uv` cache issue:

```
xcode-select --install   # if you don't already have the CLI tools
CMAKE_ARGS="-DGGML_METAL=on" uv add llama-cpp-python --reinstall-package llama-cpp-python --no-cache
```

Intel Macs have no Metal GPU offload path in this stack — CPU-only.

## 3. Known gap: hardware-tier auto-detection is NVIDIA-only right now

`barbai.hardware` (see `docs/BARBAI_ROADMAP.md`) detects VRAM via
`nvidia-smi` and is only exercised against the project's 3 target NVIDIA
laptops. On a Mac or an AMD GPU, `nvidia-smi` won't exist, so
`detect_tier()` raises `NoGpuDetectedError` and the API just reports
`hardware_tier: null` — inference itself still works fine (llama.cpp
handles Metal/CPU regardless), you just don't get the automatic model-class
recommendation. Extending detection to Metal/ROCm is unscoped for now.

## 4. Pointing an external Anthropic-compatible client at BarbAI

`POST /v1/messages` (`src/barbai/api/anthropic.py`) speaks the real
Anthropic Messages wire format, so any client built against it — the
Claude Code CLI included — can be redirected to talk to BarbAI instead,
via two environment variables the CLI reads once at startup:

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=unused   # BarbAI doesn't check this, but the CLI wants something set
claude
```

(`ANTHROPIC_BASE_URL` is a value the CLI reads once at process start —
set it before launching, restarting an already-running session won't
pick it up.)

What actually happens: BarbAI's own persona is always injected as the
system prompt (`core/persona.py`), layered underneath whatever system
prompt the CLI sends — so it answers as **BarbAI**, on a completely
different (and much smaller) model, not as Claude. It isn't a way to
"continue" a Claude conversation; it's a fresh conversation with a
different assistant, backed by this project's own docs/memory for
orientation instead of the prior chat's context.

**The real blocker, not a maybe:** the Claude Code CLI's own system
prompt plus its full built-in tool-definition set is large enough that
it's very likely already past BarbAI's default 4096-token context window
before any actual conversation happens. Raise it with `BARBAI_N_CTX`
(e.g. `export BARBAI_N_CTX=32768`, within what Qwen3.5's own native
context supports) — but a larger context means a larger KV cache, which
needs more VRAM on top of the model weights, and it adds up fast:
measured live on this project's own 4070 (8GB), Qwen3.5-9B at the
default 4096 leaves comfortable headroom, but `BARBAI_N_CTX=16384` alone
drops free VRAM from ~7780MiB to **~1408MiB** — right at
`core/hardware.py`'s `RESERVED_FLOOR_MB` safety floor. There's no
automatic check that a given value still fits your card; watch
`nvidia-smi` or `/health` after raising it, and expect to need a smaller
model (the `fast`-tier picks, Qwen3.5-4B/2B - see
`docs/BARBAI_ROADMAP.md` Phase 2.1) if you want real headroom at a large
context on an 8GB-class card. Beyond context size, Qwen3.5-9B/4B's
tool-calling reliability has only been verified against BarbAI's own
small native tool set (Phase 2.2/3.1's testing) — not against the CLI's
larger, more complex built-in tools, which is genuinely untested
territory.
