import pytest

from barbai.core.global_memory import add_fact, is_enabled, load_facts, remove_fact, render_for_prompt


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY_PATH", str(tmp_path / "global_memory.json"))
    return tmp_path / "global_memory.json"


def test_load_empty_store_returns_empty_list(store):
    assert load_facts() == []


def test_add_then_load_round_trips(store):
    fact = add_fact("prefers concise answers")
    facts = load_facts()
    assert len(facts) == 1
    assert facts[0] == fact
    assert fact["text"] == "prefers concise answers"
    assert "id" in fact and "created_at" in fact


def test_add_multiple_facts_preserves_order(store):
    add_fact("first fact")
    add_fact("second fact")
    facts = load_facts()
    assert [f["text"] for f in facts] == ["first fact", "second fact"]


def test_remove_fact_by_id(store):
    fact = add_fact("temporary fact")
    assert remove_fact(fact["id"]) is True
    assert load_facts() == []


def test_remove_unknown_id_returns_false(store):
    add_fact("keep me")
    assert remove_fact("does-not-exist") is False
    assert len(load_facts()) == 1


def test_is_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("BARBAI_GLOBAL_MEMORY", raising=False)
    assert is_enabled() is True


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "False"])
def test_is_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY", value)
    assert is_enabled() is False


def test_render_for_prompt_empty_when_no_facts(store):
    assert render_for_prompt() == ""


def test_render_for_prompt_includes_facts(store):
    add_fact("likes tea")
    add_fact("works in Go")
    rendered = render_for_prompt()
    assert "likes tea" in rendered
    assert "works in Go" in rendered


def test_render_for_prompt_empty_when_disabled(store, monkeypatch):
    add_fact("should not be shown")
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY", "off")
    assert render_for_prompt() == ""
