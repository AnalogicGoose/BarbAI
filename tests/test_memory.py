import pytest

from barbai.core import memory
from barbai.core.memory import (
    InvalidSessionIdError,
    append_to_session,
    load_session,
    load_summary,
    rolling_window,
    session_replay,
)


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BARBAI_SESSIONS_DIR", str(tmp_path / "sessions"))
    return tmp_path / "sessions"


def test_load_nonexistent_session_returns_empty(sessions_dir):
    assert load_session("brand-new") == []


def test_append_then_load_round_trips(sessions_dir):
    append_to_session("s1", [{"role": "user", "content": "hi"}])
    append_to_session("s1", [{"role": "assistant", "content": "hello"}])

    assert load_session("s1") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_append_creates_sessions_dir(sessions_dir):
    assert not sessions_dir.exists()
    append_to_session("s1", [{"role": "user", "content": "hi"}])
    assert sessions_dir.exists()


def test_append_empty_list_is_a_no_op(sessions_dir):
    append_to_session("s1", [])
    assert not sessions_dir.exists()


def test_sessions_are_independent(sessions_dir):
    append_to_session("s1", [{"role": "user", "content": "session one"}])
    append_to_session("s2", [{"role": "user", "content": "session two"}])

    assert load_session("s1") == [{"role": "user", "content": "session one"}]
    assert load_session("s2") == [{"role": "user", "content": "session two"}]


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", "x" * 129, "with space"])
def test_invalid_session_id_rejected(sessions_dir, bad_id):
    with pytest.raises(InvalidSessionIdError):
        load_session(bad_id)
    with pytest.raises(InvalidSessionIdError):
        append_to_session(bad_id, [{"role": "user", "content": "x"}])


def test_rolling_window_keeps_last_n():
    messages = [{"role": "user", "content": str(i)} for i in range(30)]
    windowed = rolling_window(messages, limit=5)
    assert windowed == messages[-5:]


def test_rolling_window_shorter_than_limit_returns_all():
    messages = [{"role": "user", "content": "only one"}]
    assert rolling_window(messages, limit=20) == messages


def _seed(session_id: str, count: int) -> None:
    for i in range(count):
        append_to_session(session_id, [{"role": "user", "content": str(i)}])


def test_session_replay_under_limit_skips_summarization_entirely(sessions_dir, monkeypatch):
    monkeypatch.setattr(memory.summarization, "summarize_turns", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    _seed("s1", 3)
    assert session_replay(llm=object(), session_id="s1", limit=5) == load_session("s1")


def test_session_replay_folds_dropped_messages_into_a_summary(sessions_dir, monkeypatch):
    monkeypatch.setattr(memory.summarization, "summarize_turns", lambda llm, prior, new: "SUMMARY")
    _seed("s1", 25)

    replayed = session_replay(llm=object(), session_id="s1", limit=5)

    assert replayed[0] == {"role": "system", "content": "[Summary of earlier conversation]\nSUMMARY"}
    assert replayed[1:] == load_session("s1")[-5:]
    assert load_summary("s1") == {"text": "SUMMARY", "summarized_through": 20}


def test_session_replay_only_summarizes_newly_dropped_messages(sessions_dir, monkeypatch):
    calls = []

    def fake_summarize(llm, prior, new):
        calls.append((prior, list(new)))
        return f"SUMMARY-{len(calls)}"

    monkeypatch.setattr(memory.summarization, "summarize_turns", fake_summarize)

    _seed("s1", 25)
    session_replay(llm=object(), session_id="s1", limit=5)
    _seed("s1", 5)  # session now has 30 messages
    session_replay(llm=object(), session_id="s1", limit=5)

    assert len(calls) == 2
    first_prior, first_new = calls[0]
    second_prior, second_new = calls[1]
    assert first_prior is None
    assert [m["content"] for m in first_new] == [str(i) for i in range(20)]
    assert second_prior == "SUMMARY-1"
    assert [m["content"] for m in second_new] == [str(i) for i in range(20, 25)]


def test_session_replay_disabled_falls_back_to_plain_drop(sessions_dir, monkeypatch):
    monkeypatch.setenv("BARBAI_SESSION_SUMMARIZATION", "off")
    monkeypatch.setattr(memory.summarization, "summarize_turns", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    _seed("s1", 25)

    assert session_replay(llm=object(), session_id="s1", limit=5) == load_session("s1")[-5:]


def test_session_replay_survives_summarization_failure(sessions_dir, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(memory.summarization, "summarize_turns", boom)
    _seed("s1", 25)

    replayed = session_replay(llm=object(), session_id="s1", limit=5)

    assert replayed == load_session("s1")[-5:]
    assert load_summary("s1") == {"text": "", "summarized_through": 0}
