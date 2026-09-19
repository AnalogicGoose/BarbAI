import pytest

from barbai.core.global_memory import add_fact
from barbai.core.persona import DEFAULT_SYSTEM_PROMPT, EXTENDED_THINKING_NUDGE, build_system_prompt


@pytest.fixture(autouse=True)
def _isolated_global_memory(tmp_path, monkeypatch):
    """Every test in this file gets its own empty global-memory store, so a
    stray global_memory.json on disk (or a fact left by another test) can
    never change these exact-match assertions."""
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY_PATH", str(tmp_path / "global_memory.json"))


def test_no_custom_thinking_mode_default():
    assert build_system_prompt(None, "thinking") == DEFAULT_SYSTEM_PROMPT


def test_no_custom_fast_mode():
    assert build_system_prompt(None, "fast") == DEFAULT_SYSTEM_PROMPT


def test_no_custom_extended_mode_adds_nudge():
    result = build_system_prompt(None, "extended")
    assert DEFAULT_SYSTEM_PROMPT in result
    assert EXTENDED_THINKING_NUDGE in result


def test_custom_layers_on_top_with_priority_note():
    result = build_system_prompt("be terse", "thinking")
    assert DEFAULT_SYSTEM_PROMPT in result
    assert "be terse" in result
    assert "take priority" in result


def test_extended_plus_custom_order():
    result = build_system_prompt("be terse", "extended")
    assert result.index(DEFAULT_SYSTEM_PROMPT) < result.index(EXTENDED_THINKING_NUDGE) < result.index("be terse")


def test_remembered_facts_are_included():
    add_fact("likes tea")
    result = build_system_prompt(None, "thinking")
    assert "likes tea" in result


def test_remembered_facts_come_before_custom():
    add_fact("likes tea")
    result = build_system_prompt("be terse", "thinking")
    assert result.index("likes tea") < result.index("be terse")


def test_no_facts_stored_matches_default_exactly():
    assert build_system_prompt(None, "thinking") == DEFAULT_SYSTEM_PROMPT


def test_disabled_global_memory_omits_facts(monkeypatch):
    add_fact("likes tea")
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY", "off")
    result = build_system_prompt(None, "thinking")
    assert result == DEFAULT_SYSTEM_PROMPT
