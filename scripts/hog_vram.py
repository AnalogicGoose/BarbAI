#!/usr/bin/env python3
"""
Occupies a chunk of GPU VRAM so BarbAI's free-VRAM tier detection
(core/hardware.py) reports a lower tier than this card's actual headroom
- lets the `fast`-tier model pick (Qwen3.5-4B, see
docs/BARBAI_ROADMAP.md Phase 2.1) be validated on hardware that has more
VRAM than a real 3050, without needing to own one. Doesn't replace real
hardware validation, but it beats leaving the pick untested indefinitely.

Usage: run this FIRST, in its own terminal, and leave it running. Then
start BarbAI (`uv run barbai`) in a second terminal - it'll see the
reduced free VRAM and pick its tier accordingly. Ctrl+C this script when
you're done to release the memory.

    python scripts/hog_vram.py --free-mb 6000   # simulates a 6GB-class card
    python scripts/hog_vram.py --free-mb 4000   # simulates a tighter 4GB-class card
    python scripts/hog_vram.py --free-mb 1200   # below RESERVED_FLOOR_MB - tests the refusal path

Pick --free-mb based on what you want BarbAI to see as "free":
  - under 7168 lands in the `fast` tier
  - 7168-10240 lands in `default` (where this card normally sits)
  - at or under 1536 (RESERVED_FLOOR_MB) tests InsufficientVramError
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import subprocess
import time


def free_vram_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def _load_cudart() -> ctypes.CDLL:
    lib_name = ctypes.util.find_library("cudart")
    candidates = [lib_name] if lib_name else []
    candidates += ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise SystemExit("couldn't find libcudart.so - is the NVIDIA CUDA runtime installed?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--free-mb", type=int, required=True, help="target free VRAM to leave, in MiB")
    args = parser.parse_args()

    current_free = free_vram_mb()
    to_grab_mb = current_free - args.free_mb
    if to_grab_mb <= 0:
        raise SystemExit(f"already at or below {args.free_mb}MiB free ({current_free}MiB) - nothing to grab")

    cudart = _load_cudart()
    ptr = ctypes.c_void_p()
    size = to_grab_mb * 1024 * 1024
    result = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if result != 0:
        raise SystemExit(f"cudaMalloc failed (CUDA error code {result}) - couldn't grab {to_grab_mb}MiB")

    print(f"grabbed {to_grab_mb}MiB - free VRAM now ~{free_vram_mb()}MiB (target was {args.free_mb}MiB)")
    print("leave this running, start BarbAI in another terminal. Ctrl+C here to release the memory.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("releasing VRAM...")
        cudart.cudaFree(ptr)


if __name__ == "__main__":
    main()
