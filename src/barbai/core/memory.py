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
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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
    stays complete on disk regardless of this."""
    return messages[-limit:] if len(messages) > limit else messages
