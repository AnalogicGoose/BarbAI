import pytest

from barbai.core.hardware import TIERS, InsufficientVramError, select_tier


def test_below_floor_raises():
    with pytest.raises(InsufficientVramError):
        select_tier(1000)


def test_fast_tier():
    assert select_tier(4000) is TIERS["fast"]
    assert select_tier(7167) is TIERS["fast"]


def test_default_tier():
    assert select_tier(7168) is TIERS["default"]
    assert select_tier(10239) is TIERS["default"]


def test_quality_tier():
    assert select_tier(10240) is TIERS["quality"]
    assert select_tier(20000) is TIERS["quality"]
