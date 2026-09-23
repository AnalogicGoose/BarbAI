"""
Server-side conversation memory (roadmap Phase 2.3) - "start dumb": a
JSONL session log on disk plus a rolling window bounding what's replayed
back to the model each turn.

Modeled on OpenAI's Responses API (`previous_response_id`-style chaining)
rather than the stateless Chat Completions/Anthropic Messages shape -
/v1/chat/completions, /v1/messages, and (implicitly) unbounded /chat calls
still hand the full history back to the caller to resend every time, and
that passthrough contract stays untouched on purpose (external
OpenAI/Anthropic-compatible clients expect exactly that). Native /chat and
/agent/chat additionally accept a `session_id`: give one and BarbAI
remembers the conversation itself, so the caller only needs to send the
new message(s) each turn.

Sessions are opt-in - omit session_id and behavior is exactly the
stateless passthrough it always was. Only role/content turns are
persisted (no raw tool_calls/tool_call_id structure) - a session-memory
turn's tool round-trips are already resolved into a final assistant reply
within the request that produced them, so there's nothing structurally
valid to replay next time regardless. This is not a vector store or
RAG - that's deferred (see roadmap) until the lack of retrieval is
actually felt.

Memory V2 (Phase 2.5): rolling_window() is still here and still the raw
v1 behavior (silently drop anything past the window - used directly by
tests and as the no-summary fallback below). session_replay() is what
the API layer actually calls now: same window, but instead of just
dropping older turns it folds them into a running summary
(core.summarization) stored alongside the session log
(`<session_id>.summary.json`), so a long conversation loses detail
gradually rather than losing whole turns outright. Toggle via
BARBAI_SESSION_SUMMARIZATION=off (default on) - disabled, or on a
summarization failure, this degrades to the plain v1 drop rather than
taking the conversation down.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from barbai.core import summarization

DEFAULT_SESSIONS_DIR = Path("sessions")
ROLLING_WINDOW_MESSAGES = 20  # last N stored messages replayed to the model

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

class InvalidSessionIdError(ValueError):
    """session_id isn't a safe filename component."""


def _sessions_dir() -> Path:
    raw = os.environ.get("BARBAI_SESSIONS_DIR")
    return Path(raw).resolve() if raw else DEFAULT_SESSIONS_DIR.resolve()

def _session_path(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise InvalidSessionIdError(f"invalid session_id: {session_id!r}")
    return _sessions_dir() / f"{session_id}.jsonl"

def _summary_path(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise InvalidSessionIdError(f"invalid session_id: {session_id!r}")
    return _sessions_dir() / f"{session_id}.summary.json"

def is_summarization_enabled() -> bool:
    return os.environ.get("BARBAI_SESSION_SUMMARIZATION", "on").strip().lower() not in ("off", "0", "false")

def load_session(session_id: str) -> list[dict]:
    """Full history for a session, oldest first. Empty list for a new one."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    messages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages

def append_to_session(session_id: str, messages: list[dict]) -> None:
    """Append new turns to the durable log (the full record, unbounded)."""
    if not messages:
        return
    path = _session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for message in messages:
            f.write(json.dumps(message) + "\n")

def rolling_window(messages: list[dict], limit: int = ROLLING_WINDOW_MESSAGES) -> list[dict]:
    """Bound what actually gets replayed to the model - the log itself
    stays complete on disk regardless of this. v1 behavior: anything past
    the window is just dropped, no trace. See session_replay() for the
    v2 caller actually uses now."""
    return messages[-limit:] if len(messages) > limit else messages

def load_summary(session_id: str) -> dict:
    """{"text": str, "summarized_through": int} - summarized_through is
    how many messages, from the start of the full on-disk log, are
    already folded into `text`. {"text": "", "summarized_through": 0} for
    a session with no summary yet."""
    path = _summary_path(session_id)
    if not path.exists():
        return {"text": "", "summarized_through": 0}
    return json.loads(path.read_text(encoding="utf-8"))

def _save_summary(session_id: str, text: str, summarized_through: int) -> None:
    path = _summary_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text, "summarized_through": summarized_through}), encoding="utf-8")

def session_replay(llm, session_id: str, limit: int = ROLLING_WINDOW_MESSAGES) -> list[dict]:
    """History to replay for a session: the most recent `limit` messages
    verbatim, plus - Memory V2 - anything older folded into a running
    summary instead of silently dropped. Falls back to rolling_window()'s
    plain v1 drop when summarization is disabled or fails - a degraded
    memory beats a broken conversation.
    """
    full = load_session(session_id)
    if len(full) <= limit:
        return full

    kept = full[-limit:]
    if not is_summarization_enabled():
        return kept

    state = load_summary(session_id)
    cutoff = len(full) - limit  # messages before this index are falling out of `kept`
    newly_dropped = full[state["summarized_through"] : cutoff]

    text = state["text"]
    if newly_dropped:
        try:
            text = summarization.summarize_turns(llm, state["text"] or None, newly_dropped)
        except Exception:
            # Best-effort: a broken summarization pass shouldn't take the
            # whole conversation down - keep whatever summary already
            # existed and just retry the fold-in next turn.
            text = state["text"]
        else:
            _save_summary(session_id, text, cutoff)

    if not text:
        return kept

    return [{"role": "system", "content": f"[Summary of earlier conversation]\n{text}"}] + kept
