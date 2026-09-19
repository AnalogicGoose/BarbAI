"""
Built-in tools the agent runtime can execute server-side.

Ship exactly one tool first (roadmap Phase 2.2): a read-only file read,
scoped to an allowlist. Hold off on write/shell tools until this loop is
proven solid.

The allowlist (BARBAI_TOOLS_ROOTS, comma-separated) can mix two kinds of
entries, matching how Claude Code/Codex-style tools scope file access:
  - a directory: everything inside it (recursively) is readable
  - a single file: only that exact file is readable, not its siblings
Multiple directories can be attached at once (Codex-style multi-root).
"""

from __future__ import annotations

import os
from pathlib import Path

MAX_FILE_BYTES = 100_000

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

def _resolve_path(path: str, roots: list[Path]) -> Path | None:
    requested = Path(path)

    if requested.is_absolute():
        resolved = requested.resolve()
        return resolved if _is_allowed(resolved, roots) else None

    # Relative path: try each directory root in the configured order: the
    # first one where the file actually exists wins. (A relative path
    # can't target a file-root directly - point at it with its full path.)
    for root in roots:
        if not root.is_dir():
            continue
        candidate = (root / requested).resolve()
        if _is_allowed(candidate, roots) and candidate.exists():
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

TOOLS = {"read_file": read_file}

def build_tool_defs() -> list[dict]:
    return [read_file_tool_def()]

def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolExecutionError(f"unknown tool: {name!r}")
    return tool(**arguments)