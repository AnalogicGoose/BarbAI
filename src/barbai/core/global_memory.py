"""
BarbAI's global, cross-conversation memory - a small store of facts the
user has *explicitly* asked BarbAI to remember, rendered into every
system prompt (see core.persona.build_system_prompt) when enabled.

Design choice, from a discussion comparing how ChatGPT and Claude do this:
ChatGPT auto-extracts facts silently from anything you say, one global
store; Claude auto-summarizes whole conversations, scoped per-project,
visible/editable. BarbAI does neither yet - this is explicit-only, the
"start dumb" v1: a fact is written only when the `remember` tool
(core/tools.py) is called, and that tool's own description instructs the
model to only call it on an explicit user request ("remember that...",
"don't forget..."), never inferred on its own. `remember` is also in
GATED_TOOLS, so a human approves the exact text before it's stored
permanently - the same safety net write_file already gets, since a
misphrased "remembered" fact would otherwise quietly shape every future
conversation. Claude-style automatic end-of-session summarization (reusing
core.memory's session log) is the natural v2 once this is proven useful -
not built yet.

Toggle: BARBAI_GLOBAL_MEMORY=off disables both reading and writing facts
entirely (checked per call, not just at startup). Default is on.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STORE_PATH = Path("global_memory.json")


def is_enabled() -> bool:
    return os.environ.get("BARBAI_GLOBAL_MEMORY", "on").strip().lower() not in ("off", "0", "false")

def _store_path() -> Path:
    raw = os.environ.get("BARBAI_GLOBAL_MEMORY_PATH")
    return Path(raw).resolve() if raw else DEFAULT_STORE_PATH.resolve()

def load_facts() -> list[dict]:
    """All stored facts, oldest first. Empty list if none stored yet."""
    path = _store_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

def add_fact(text: str) -> dict:
    """Store a new fact. Returns the stored record (id, text, created_at)."""
    facts = load_facts()
    fact = {
        "id": uuid.uuid4().hex,
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    facts.append(fact)
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return fact

def remove_fact(fact_id: str) -> bool:
    """Delete a stored fact by id. Returns whether anything was removed."""
    facts = load_facts()
    remaining = [f for f in facts if f["id"] != fact_id]
    if len(remaining) == len(facts):
        return False
    _store_path().write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    return True

def render_for_prompt() -> str:
    """Formatted block for the system prompt, or "" if disabled/empty."""
    if not is_enabled():
        return ""
    facts = load_facts()
    if not facts:
        return ""
    lines = "\n".join(f"- {f['text']}" for f in facts)
    return "Things the user has explicitly asked you to remember, from past conversations:\n" + lines
