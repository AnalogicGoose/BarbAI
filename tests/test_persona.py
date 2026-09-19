from barbai.core.persona import DEFAULT_SYSTEM_PROMPT, EXTENDED_THINKING_NUDGE, build_system_prompt


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
