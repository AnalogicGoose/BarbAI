import pytest

from barbai.core.memory import (
    InvalidSessionIdError,
    append_to_session,
    load_session,
    rolling_window,
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
