"""
Built-in tools the agent runtime can execute server-side.

Three tools so far: a read-only file read and a file write (roadmap
Phase 2.2), both scoped to an allowlist, and `remember` (Phase 2.3 v1),
which writes a single fact to BarbAI's global cross-conversation memory
(core/global_memory.py) - see that module's docstring for the
explicit-only design rationale. Hold off on shell tools until this is
proven solid.

The allowlist (BARBAI_TOOLS_ROOTS, comma-separated) can mix two kinds of
entries, matching how Claude Code/Codex-style tools scope file access:
  - a directory: everything inside it (recursively) is readable/writable
  - a single file: only that exact file is readable, not its siblings
    (and can't be a write target for a *new* file, since a relative path
    can't target a file-root directly - see _resolve_path)
Multiple directories can be attached at once (Codex-style multi-root).

GATED_TOOLS marks tools that write/change state - the agent loop
(core/agent.py) pauses for human approval before running one of these,
rather than executing it immediately like a routine read. `remember` is
gated for the same reason write_file is: a misphrased "remembered" fact
would otherwise quietly shape every future conversation with no review.
"""

from __future__ import annotations

import os
from pathlib import Path

from barbai.core import global_memory

MAX_FILE_BYTES = 100_000
MAX_WRITE_BYTES = 1_000_000
MAX_REMEMBER_CHARS = 500

GATED_TOOLS = {"write_file", "remember"}

class ToolExecutionError(RuntimeError):
    """
    A tool ran but failed in an expected way (bad path, too large, ...).

    The message is written to be safe to feed back to the model as the
    tool result, not just logged - the model should be able to react to it
    (e.g. try a different path) rather than treat it as a hard crash.
    """


def _workspace_roots() -> list[Path]:
    raw = os.environ.get("BARBAI_TOOLS_ROOTS")
    if not raw:
        return [Path.cwd().resolve()]
    return [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]

def _is_allowed(resolved: Path, roots: list[Path]) -> bool:
    for root in roots:
        if root.is_file():
            if resolved == root:
                return True
        else:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
    return False

def _resolve_path(path: str, roots: list[Path], *, must_exist: bool = True) -> Path | None:
    requested = Path(path)

    if requested.is_absolute():
        resolved = requested.resolve()
        return resolved if _is_allowed(resolved, roots) else None

    # Relative path: try each directory root in the configured order. The
    # first root where the file actually exists wins (must_exist=True,
    # used for reads and for overwriting an existing file). With
    # must_exist=False (used as a write fallback for brand-new files),
    # the first directory root is the target - there's nothing to
    # disambiguate by yet. (A relative path can't target a file-root
    # directly - point at it with its full path.)
    for root in roots:
        if not root.is_dir():
            continue
        candidate = (root / requested).resolve()
        if not _is_allowed(candidate, roots):
            continue
        if must_exist and not candidate.exists():
            continue
        return candidate

    return None

def _describe_roots(roots: list[Path]) -> str:
    parts = []
    for root in roots:
        kind = "file" if root.is_file() else "directory"
        parts.append(f"{root} ({kind})")
    return "; ".join(parts)

def read_file(path: str) -> str:
    roots = _workspace_roots()
    candidate = _resolve_path(path, roots)
    if candidate is None:
        raise ToolExecutionError(f"path {path!r} is outside every allowed location")

    if not candidate.exists():
        raise ToolExecutionError(f"no such file: {path!r}")
    if not candidate.is_file():
        raise ToolExecutionError(f"not a file: {path!r}")

    size = candidate.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ToolExecutionError(f"file too large ({size} bytes, limit {MAX_FILE_BYTES})")

    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolExecutionError(f"{path!r} is not a text file") from None

def write_file(path: str, content: str, create_dirs: bool = False) -> str:
    roots = _workspace_roots()
    # Prefer overwriting a file that already exists somewhere in the
    # allowlist (same root-priority as read_file); only fall back to
    # "first directory root" when nothing matches yet, i.e. a new file.
    candidate = _resolve_path(path, roots) or _resolve_path(path, roots, must_exist=False)
    if candidate is None:
        raise ToolExecutionError(f"path {path!r} is outside every allowed location")

    if candidate.exists() and candidate.is_dir():
        raise ToolExecutionError(f"not a file: {path!r}")

    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ToolExecutionError(f"content too large ({size} bytes, limit {MAX_WRITE_BYTES})")

    if not candidate.parent.exists():
        if not create_dirs:
            raise ToolExecutionError(
                f"parent directory does not exist: {candidate.parent} (retry with create_dirs=true)"
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)

    candidate.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"

def remember(text: str) -> str:
    if not global_memory.is_enabled():
        raise ToolExecutionError("global memory is disabled (BARBAI_GLOBAL_MEMORY=off)")

    text = text.strip()
    if not text:
        raise ToolExecutionError("nothing to remember - text was empty")
    if len(text) > MAX_REMEMBER_CHARS:
        raise ToolExecutionError(f"fact too long ({len(text)} chars, limit {MAX_REMEMBER_CHARS})")

    fact = global_memory.add_fact(text)
    return f"remembered: {fact['text']!r}"

def read_file_tool_def() -> dict:
    roots = _workspace_roots()
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file. Allowed locations: "
                f"{_describe_roots(roots)}. Inside a directory entry, any "
                "file within it (including subdirectories) is readable; a "
                "file entry only allows that exact file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to one of the allowed directories.",
                    }
                },
                "required": ["path"],
            },
        },
    }

def write_file_tool_def() -> dict:
    roots = _workspace_roots()
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file, creating it if it doesn't "
                "exist or overwriting it if it does. Allowed locations: "
                f"{_describe_roots(roots)}. Requires human approval before "
                "it actually runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to one of the allowed directories.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write.",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create missing parent directories. Default false.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    }

def remember_tool_def() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a short fact to BarbAI's persistent, cross-conversation "
                "memory - it will be shown to you at the start of every future "
                "conversation. Only call this when the user explicitly asks "
                "you to remember something (e.g. says \"remember that...\" or "
                "\"don't forget...\"). Never call this on your own initiative "
                "for something you merely noticed or inferred."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The fact to remember, written concisely in third "
                            "person (e.g. \"prefers concise answers\")."
                        ),
                    }
                },
                "required": ["text"],
            },
        },
    }

TOOLS = {"read_file": read_file, "write_file": write_file, "remember": remember}

def build_tool_defs() -> list[dict]:
    defs = [read_file_tool_def(), write_file_tool_def()]
    if global_memory.is_enabled():
        defs.append(remember_tool_def())
    return defs

def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolExecutionError(f"unknown tool: {name!r}")
    return tool(**arguments)
