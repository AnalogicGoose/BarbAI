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
context supports) — see section 5 below for what that actually costs in
VRAM before picking a number. Beyond context size, Qwen3.5-9B/4B's
tool-calling reliability has only been verified against BarbAI's own
small native tool set (Phase 2.2/3.1's testing) — not against the CLI's
larger, more complex built-in tools, which is genuinely untested
territory.

## 5. Choosing `BARBAI_N_CTX`

Not just a Claude Code CLI concern — since Phase 2.3's context-overflow
fix (`core/model_runtime.py`'s `fit_to_context()`, see
`docs/BARBAI_ROADMAP.md`), a conversation that outgrows `n_ctx` no
longer crashes, it just starts quietly forgetting its oldest turns. The
default 4096 makes that kick in fast - real multi-turn conversations hit
it within a handful of exchanges. Raising it buys more conversation
before that starts, at a real, measurable VRAM cost.

Measured live on this project's own dev machine (RTX 4070 Laptop, 8GB,
with a normal desktop session - browser, Discord, etc. - already running
in the background, since that's the realistic case, not an idle
benchmark rig) - loading Qwen3.5-9B Q4_K_M at each context size:

| `BARBAI_N_CTX` | Free VRAM after load | Headroom above the 1536MiB safety floor |
|---|---|---|
| 4096 (default) | ~1650MiB | comfortable |
| 8192 | ~1522MiB | thin - right at the floor |
| 12288 | ~1394MiB | **below the floor** |
| 32768 | ~122MiB | dangerously tight, don't |

The jump from 4096 to 8192 only costs ~128MiB - Qwen3.5's architecture
mixes attention layers with SSM (Mamba-style) layers that don't scale
with context the way a pure-attention KV cache does, which is why this
is far gentler than a naive per-token estimate would suggest. But it's
not free, and it's not perfectly linear either - the cost accelerates
at higher context sizes (12288→32768, a 2.7x jump, cost roughly 10x the
VRAM that 4096→8192 did).

**Recommendation for an 8GB-class card:** `BARBAI_N_CTX=8192` is a
reasonable ceiling *if nothing else is contending for VRAM at the
moment* - it's right at this project's own safety floor with a normal
background app load, not comfortably above it. Going to 12288 or higher
means closing other GPU-using apps first, not just setting the variable
and hoping. Stay at the 4096 default if you want real margin for
whatever else is running. Either way: this is what *this* machine showed
with *this* background load - your own headroom depends on what else is
using your GPU, the same "measure it, don't assume it" lesson
`core/hardware.py`'s own tiering is built around. Check `nvidia-smi` or
`/health` after changing it, on your own machine, before trusting a
number from here.

Setting it: `export BARBAI_N_CTX=8192` before starting the server (`uv
run barbai`) - it's read once at model-load time, so an already-running
server needs a restart to pick up a new value.
