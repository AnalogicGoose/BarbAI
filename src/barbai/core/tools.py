"""
Built-in tools the agent runtime can execute server-side.

Read-only tools (Phase 2.2's read_file, Phase 3.1's list_directory,
search, and read_file's line-range mode) are ungated - routine reads
don't need human approval, matching Mana's own model (see
docs/BARBAI_ROADMAP.md section 3). write_file and remember mutate state
and are gated: the agent loop (core/agent.py) pauses for human approval
before running one of these, rather than executing it immediately.

The allowlist (BARBAI_TOOLS_ROOTS, comma-separated) can mix two kinds of
entries, matching how Claude Code/Codex-style tools scope file access:
  - a directory: everything inside it (recursively) is readable/writable
  - a single file: only that exact file is readable, not its siblings
    (and can't be a write target for a *new* file, since a relative path
    can't target a file-root directly - see _resolve_path)
Multiple directories can be attached at once (Codex-style multi-root).
Every tool below reuses the same _resolve_path/_is_allowed boundary
check - there's exactly one place that decides what's in scope.

search shells out to ripgrep (`rg`) rather than reimplementing it -
correct multiline/binary handling and .gitignore-awareness aren't worth
rebuilding, and this project already leans on external binaries the same
way for hardware detection (nvidia-smi). Always invoked as an argument
list (never shell=True), with `--` before the user-supplied pattern, so
neither the pattern nor a resolved path can be interpreted as a flag or
reach a shell.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
from pathlib import Path

from barbai.core import global_memory

MAX_FILE_BYTES = 100_000
MAX_WRITE_BYTES = 1_000_000
MAX_REMEMBER_CHARS = 500
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200

GATED_TOOLS = {"write_file", "remember"}

_NOISE_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", "dist", "build", ".idea", ".codegraph",
}

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

def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    roots = _workspace_roots()
    candidate = _resolve_path(path, roots)
    if candidate is None:
        raise ToolExecutionError(f"path {path!r} is outside every allowed location")

    if not candidate.exists():
        raise ToolExecutionError(f"no such file: {path!r}")
    if not candidate.is_file():
        raise ToolExecutionError(f"not a file: {path!r}")

    if start_line is None and end_line is None:
        size = candidate.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolExecutionError(
                f"file too large ({size} bytes, limit {MAX_FILE_BYTES}) - retry with "
                "start_line/end_line to read a slice instead of the whole file"
            )
        try:
            return candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolExecutionError(f"{path!r} is not a text file") from None

    start = start_line if start_line is not None else 1
    if start < 1:
        raise ToolExecutionError(f"start_line must be >= 1, got {start_line}")
    if end_line is not None and end_line < start:
        raise ToolExecutionError(f"end_line ({end_line}) is before start_line ({start})")

    # Streams rather than reading the whole file up front - keeps memory
    # bounded even when the file itself is far past MAX_FILE_BYTES, which
    # is the whole point of a range read.
    try:
        with candidate.open("r", encoding="utf-8") as f:
            selected = list(itertools.islice(f, start - 1, end_line))
    except UnicodeDecodeError:
        raise ToolExecutionError(f"{path!r} is not a text file") from None

    if not selected:
        raise ToolExecutionError(f"start_line {start} is past the end of {path!r}")

    result = "".join(selected)
    result_size = len(result.encode("utf-8"))
    if result_size > MAX_FILE_BYTES:
        raise ToolExecutionError(
            f"requested range too large ({result_size} bytes, limit {MAX_FILE_BYTES}) - narrow the range"
        )
    return result

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

def list_directory(path: str, recursive: bool = False) -> str:
    roots = _workspace_roots()
    candidate = _resolve_path(path, roots)
    if candidate is None:
        raise ToolExecutionError(f"path {path!r} is outside every allowed location")

    if not candidate.exists():
        raise ToolExecutionError(f"no such directory: {path!r}")
    if not candidate.is_dir():
        raise ToolExecutionError(f"not a directory: {path!r}")

    if recursive:
        entries = [
            e for e in candidate.rglob("*")
            if not _NOISE_DIR_NAMES & set(e.relative_to(candidate).parts)
        ]
        names = [
            f"{e.relative_to(candidate)}/" if e.is_dir() else str(e.relative_to(candidate))
            for e in sorted(entries)
        ]
    else:
        names = [f"{e.name}/" if e.is_dir() else e.name for e in sorted(candidate.iterdir())]

    if not names:
        return f"{path} is empty"

    truncated = len(names) > MAX_LIST_ENTRIES
    listing = "\n".join(names[:MAX_LIST_ENTRIES])
    if truncated:
        listing += f"\n... truncated at {MAX_LIST_ENTRIES} of {len(names)} entries - narrow the path"
    return listing

def search(pattern: str, path: str | None = None, ignore_case: bool = False) -> str:
    roots = _workspace_roots()

    if path is not None:
        candidate = _resolve_path(path, roots)
        if candidate is None:
            raise ToolExecutionError(f"path {path!r} is outside every allowed location")
        if not candidate.exists():
            raise ToolExecutionError(f"no such path: {path!r}")
        search_targets = [candidate]
    else:
        search_targets = roots

    rg_path = shutil.which("rg")
    if rg_path is None:
        raise ToolExecutionError("ripgrep ('rg') isn't installed or isn't on PATH")

    base_cmd = [rg_path, "--line-number", "--with-filename", "--no-heading", "--color=never"]
    if ignore_case:
        base_cmd.append("--ignore-case")

    lines: list[str] = []
    for target in search_targets:
        proc = subprocess.run(
            [*base_cmd, "--", pattern, str(target)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # rg exits 1 for "no matches" (not an error), 2 for a real error.
        if proc.returncode == 2:
            raise ToolExecutionError(f"search failed: {proc.stderr.strip()}")
        if proc.stdout:
            lines.extend(proc.stdout.splitlines())
        if len(lines) >= MAX_SEARCH_RESULTS:
            break

    if not lines:
        return "no matches"

    truncated = len(lines) > MAX_SEARCH_RESULTS
    result = "\n".join(lines[:MAX_SEARCH_RESULTS])
    if truncated:
        result += f"\n... truncated at {MAX_SEARCH_RESULTS} results - narrow the pattern or path"
    return result

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
                "file entry only allows that exact file. Whole-file reads "
                f"are capped at {MAX_FILE_BYTES} bytes - for a larger file, "
                "pass start_line/end_line (1-indexed, inclusive) to read a "
                "slice instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to one of the allowed directories.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read, 1-indexed. Omit to read from the start of the file.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read, inclusive. Omit to read to the end of the file.",
                    },
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

def list_directory_tool_def() -> dict:
    roots = _workspace_roots()
    return {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List the contents of a directory. Allowed locations: "
                f"{_describe_roots(roots)}. Non-recursive by default (like "
                "`ls`); set recursive=true for a full tree (like `tree`), "
                "which skips common noise directories (.git, node_modules, "
                "__pycache__, .venv, build artifacts, ...) and is capped at "
                f"{MAX_LIST_ENTRIES} entries. Directory entries end with '/'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to one of the allowed directories.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List the full subtree instead of just the immediate contents. Default false.",
                    },
                },
                "required": ["path"],
            },
        },
    }

def search_tool_def() -> dict:
    roots = _workspace_roots()
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search file contents for a regex pattern (ripgrep/Rust "
                "regex syntax), returning matches as "
                "'path:line:matched_text'. Allowed locations: "
                f"{_describe_roots(roots)}. Omit path to search everywhere "
                "allowed; give a directory or file to narrow it. Respects "
                f".gitignore automatically, capped at {MAX_SEARCH_RESULTS} "
                "results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (ripgrep/Rust regex syntax).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search within. Omit to search every allowed location.",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search. Default false.",
                    },
                },
                "required": ["pattern"],
            },
        },
    }

TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "remember": remember,
    "list_directory": list_directory,
    "search": search,
}

def build_tool_defs() -> list[dict]:
    defs = [read_file_tool_def(), write_file_tool_def(), list_directory_tool_def(), search_tool_def()]
    if global_memory.is_enabled():
        defs.append(remember_tool_def())
    return defs

def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolExecutionError(f"unknown tool: {name!r}")
    return tool(**arguments)
