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
