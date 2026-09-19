"""
GPU VRAM detection and hardware-tier selection.

Mirrors Mana's tiering (node-bot/model-management.js): pick a model class
by *free* VRAM at load time, not the card's nominal/total spec, and refuse
to load if that would leave less than a safety floor free.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

class NoGpuDetectedError(RuntimeError):
    """No NVIDIA GPU / nvidia-smi found."""


class InsufficientVramError(RuntimeError):
    """Free VRAM is below the safety floor; refuse to load."""

RESERVED_FLOOR_MB = 1536

_TIER_BOUNDARIES_MB = (
    (7168, "fast"),
    (10240, "default"),
)
_TOP_TIER = "quality"

@dataclass(frozen=True)
class Tier:
    name: str
    model_class: str

TIERS = {
    "fast": Tier("fast", "1.5B-4B class (e.g. Qwen3-1.7B/4B)"),
    "default": Tier("default", "4B class (e.g. Qwen3-4B)"),
    "quality": Tier("quality", "8-14B class (e.g. Qwen3-8B/14B)"),
}

def get_free_vram_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise NoGpuDetectedError("nvidia-smi unavailable or no NVIDIA GPU present") from exc

    first_line = result.stdout.strip().splitlines()[0]
    return int(first_line)

def select_tier(free_vram_mb: int) -> Tier:
    if free_vram_mb < RESERVED_FLOOR_MB:
        raise InsufficientVramError(
            f"only {free_vram_mb}MiB free, below the {RESERVED_FLOOR_MB}MiB safety floor"
        )
    for boundary_mb, tier_name in _TIER_BOUNDARIES_MB:
        if free_vram_mb < boundary_mb:
            return TIERS[tier_name]
    return TIERS[_TOP_TIER]


def detect_tier() -> Tier:
    return select_tier(get_free_vram_mb())