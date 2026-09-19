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

Cross-platform (Linux/macOS/Windows is a hard requirement, not aspirational):
write_file and patch_file both detect and preserve a file's *existing*
line-ending convention (\r\n on Windows, \n on Unix/macOS) rather than
letting Python's default text-mode write silently normalize it to
whatever the host OS prefers - that would otherwise mean editing a single
line of a Windows-authored file on a Linux server rewrites every line
ending in it. A brand-new file just gets \n, the portable default every
OS's tooling handles fine. See _detect_newline.

run_command's design was checked against how OpenAI Codex, GitHub
Copilot, and Claude Code's own Bash tool handle this before building it.
Codex and Copilot both do real OS-level sandboxing (macOS seatbelt,
Windows native sandbox, deny-by-default network) - a genuine engineering
undertaking per platform, and explicitly Phase 3.3 territory here
(command sanitization is already deferred there), not this first pass.
Claude Code's own Bash tool is the closer model for where this project
is right now: no kernel sandbox, relies on approval + sane limits -
its numbers (2min default timeout, 10min cap, 30,000-char output cap
with *middle* truncation - keep the start and end, cut the middle, since
errors are usually at the end and setup context at the start) are
adopted directly here as proven defaults. Its persistent shell session
and background-execution support are deliberately NOT adopted yet -
both add real state across calls that doesn't fit this project's
stateless-per-call tool model without its own design pass; each
run_command call is a fresh subprocess, cwd passed explicitly every time.
"""

from __future__ import annotations

import itertools
import os
import platform
import shutil
import subprocess
from pathlib import Path

from barbai.core import global_memory

MAX_FILE_BYTES = 100_000
MAX_WRITE_BYTES = 1_000_000
MAX_REMEMBER_CHARS = 500
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200
DEFAULT_COMMAND_TIMEOUT = 120  # seconds - same default Claude Code's own Bash tool uses
MAX_COMMAND_TIMEOUT = 600  # seconds - same cap Claude Code's own Bash tool uses
MAX_COMMAND_OUTPUT_CHARS = 30_000  # same limit Claude Code's own Bash tool uses

GATED_TOOLS = {"write_file", "remember", "patch_file", "run_command"}

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

def _detect_newline(raw: bytes) -> str:
    """Sniff a file's existing line-ending convention from its raw bytes -
    \r\n if any CRLF is present, else \n. Used to preserve that convention
    on write instead of letting the host OS's text-mode default decide,
    which would otherwise silently rewrite every line ending in a file
    whenever any tool touches it (see module docstring)."""
    return "\r\n" if b"\r\n" in raw else "\n"

def _describe_roots(roots: list[Path]) -> str:
    parts = []
    for root in roots:
        kind = "file" if root.is_file() else "directory"
        parts.append(f"{root} ({kind})")
    return "; ".join(parts)

def _shell_invocation(command: str) -> list[str]:
    """Wrap a command string for the platform's real shell. On Windows,
    cmd.exe (always present, no WSL/PowerShell assumption). On Unix,
    the user's actual $SHELL (bash/zsh, not just /bin/sh) so bashisms in
    a command behave the way they would in the user's own terminal -
    same reasoning Claude Code's own Bash tool sources ~/.bashrc for."""
    if platform.system() == "Windows":
        return ["cmd", "/c", command]
    return [os.environ.get("SHELL", "/bin/sh"), "-c", command]

def _truncate_output(output: str, limit: int = MAX_COMMAND_OUTPUT_CHARS) -> str:
    """Middle-truncate rather than cut the tail: command errors are
    usually at the end of the output, setup/context at the start -
    losing the middle is the least harmful place to cut."""
    if len(output) <= limit:
        return output
    marker = "\n... [truncated] ...\n"
    half = (limit - len(marker)) // 2
    return output[:half] + marker + output[-half:]

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

    # Preserve an existing file's line-ending convention rather than the
    # host OS's default (see module docstring); a brand-new file gets the
    # portable \n default. Normalize any stray \r first so this can't
    # double up an already-CRLF fragment into \r\r\n.
    newline = _detect_newline(candidate.read_bytes()) if candidate.exists() else "\n"
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    on_disk = normalized.replace("\n", newline) if newline != "\n" else normalized

    size = len(on_disk.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ToolExecutionError(f"content too large ({size} bytes, limit {MAX_WRITE_BYTES})")

    if not candidate.parent.exists():
        if not create_dirs:
            raise ToolExecutionError(
                f"parent directory does not exist: {candidate.parent} (retry with create_dirs=true)"
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)

    with candidate.open("w", encoding="utf-8", newline="") as f:
        f.write(on_disk)
    return f"wrote {len(normalized)} characters to {path}"

def patch_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Surgical edit: replace exact text rather than rewriting the whole
    file, the same old_string/new_string shape as Claude Code's own Edit
    tool - old_string must match exactly once (or replace_all=true to
    change every match), which is far more forgiving for a small local
    model to produce correctly than a unified diff with line numbers."""
    roots = _workspace_roots()
    candidate = _resolve_path(path, roots)
    if candidate is None:
        raise ToolExecutionError(f"path {path!r} is outside every allowed location")

    if not candidate.exists():
        raise ToolExecutionError(
            f"no such file: {path!r} - patch_file only edits existing files, use write_file to create one"
        )
    if not candidate.is_file():
        raise ToolExecutionError(f"not a file: {path!r}")

    raw = candidate.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ToolExecutionError(
            f"file too large ({len(raw)} bytes, limit {MAX_FILE_BYTES}) - patch_file reads the whole file to apply the edit"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolExecutionError(f"{path!r} is not a text file") from None

    if not old_string:
        raise ToolExecutionError("old_string cannot be empty")
    if old_string == new_string:
        raise ToolExecutionError("old_string and new_string are identical - nothing to change")

    # Match against \n-normalized content - the same convention read_file
    # already hands the model, so old_string/new_string are written
    # against that, never the file's raw on-disk bytes.
    newline = _detect_newline(raw)
    normalized = text.replace("\r\n", "\n")

    count = normalized.count(old_string)
    if count == 0:
        raise ToolExecutionError(f"old_string not found in {path!r}")
    if count > 1 and not replace_all:
        raise ToolExecutionError(
            f"old_string matches {count} times in {path!r} - make it more specific and unique, "
            "or pass replace_all=true to replace every match"
        )

    replaced = count if replace_all else 1
    updated = normalized.replace(old_string, new_string, -1 if replace_all else 1)

    on_disk = updated.replace("\n", newline) if newline != "\n" else updated
    size = len(on_disk.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ToolExecutionError(f"result too large ({size} bytes, limit {MAX_WRITE_BYTES})")

    # Preserve the file's original line-ending convention on write (see
    # module docstring) - never let this or the host OS silently convert
    # a Windows CRLF file to LF (or vice versa) as a side effect.
    with candidate.open("w", encoding="utf-8", newline="") as f:
        f.write(on_disk)

    return f"replaced {replaced} occurrence{'s' if replaced != 1 else ''} in {path}"

def run_command(command: str, cwd: str | None = None, timeout_seconds: int | None = None) -> str:
    """Run a shell command - the tool that actually lets the agent verify
    its own work (run tests/build), not just read and edit code. No
    sandbox and no command sanitization yet (both explicitly deferred to
    Phase 3.3, see module docstring) - the approval gate, the working-dir
    allowlist, the timeout, and the output cap are the whole safety
    boundary for this first pass. A nonzero exit code is normal, useful
    information (a failing test), not a tool failure - it's returned,
    never raised; only things that mean the command genuinely couldn't
    be evaluated (bad cwd, timeout, no such shell) raise."""
    roots = _workspace_roots()

    if cwd is not None:
        candidate = _resolve_path(cwd, roots)
        if candidate is None:
            raise ToolExecutionError(f"path {cwd!r} is outside every allowed location")
        if not candidate.is_dir():
            raise ToolExecutionError(f"not a directory: {cwd!r}")
    else:
        directory_roots = [r for r in roots if r.is_dir()]
        if not directory_roots:
            raise ToolExecutionError("no directory in the allowlist to run a command from - pass cwd explicitly")
        candidate = directory_roots[0]

    if not command.strip():
        raise ToolExecutionError("command cannot be empty")

    timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_COMMAND_TIMEOUT
    if timeout <= 0:
        raise ToolExecutionError(f"timeout_seconds must be positive, got {timeout_seconds}")
    timeout = min(timeout, MAX_COMMAND_TIMEOUT)

    try:
        proc = subprocess.run(
            _shell_invocation(command),
            cwd=str(candidate),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        raise ToolExecutionError(
            f"command timed out after {timeout}s in {candidate}. Partial output:\n{_truncate_output(partial)}"
        ) from None
    except OSError as exc:
        raise ToolExecutionError(f"couldn't run command: {exc}") from None

    return f"exit code: {proc.returncode}\n\n{_truncate_output(proc.stdout or '')}"

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

def patch_file_tool_def() -> dict:
    roots = _workspace_roots()
    return {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Replace an exact piece of text in an existing file, "
                "instead of rewriting the whole thing - the precise way "
                "to make a small edit. Allowed locations: "
                f"{_describe_roots(roots)}. old_string must match the "
                "file's current content exactly (read the file first) "
                "and must be unique in the file unless replace_all is "
                "set - include enough surrounding context in old_string "
                "to make it unique rather than guessing. Fails clearly if "
                "old_string isn't found or matches more than once. "
                "Requires human approval before it actually runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to one of the allowed directories. Must already exist.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace, including any surrounding context needed to make it unique.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string instead of requiring exactly one match. Default false.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    }

def run_command_tool_def() -> dict:
    roots = _workspace_roots()
    directory_roots = [r for r in roots if r.is_dir()]
    default_cwd = str(directory_roots[0]) if directory_roots else "none configured"
    return {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command (e.g. running tests, a build, a "
                "linter) and get back its exit code and output - this is "
                "how you verify a change actually works, not just that it "
                "looks right. Allowed working directories: "
                f"{_describe_roots(roots)}. Defaults to {default_cwd} if "
                "cwd is omitted. Output over "
                f"{MAX_COMMAND_OUTPUT_CHARS} characters is truncated in "
                "the middle (start and end are kept). Requires human "
                "approval before it actually runs. A nonzero exit code "
                "just means the command failed (e.g. a test failed) - "
                "that's useful information, not an error to give up on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run, exactly as you'd type it in a terminal.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory to run the command from. Must be one of the allowed locations. Omit to use the default.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": (
                            f"Max time to let the command run, in seconds. Default {DEFAULT_COMMAND_TIMEOUT}, "
                            f"capped at {MAX_COMMAND_TIMEOUT}."
                        ),
                    },
                },
                "required": ["command"],
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
    "patch_file": patch_file,
    "run_command": run_command,
    "remember": remember,
    "list_directory": list_directory,
    "search": search,
}

def build_tool_defs() -> list[dict]:
    defs = [
        read_file_tool_def(),
        write_file_tool_def(),
        patch_file_tool_def(),
        run_command_tool_def(),
        list_directory_tool_def(),
        search_tool_def(),
    ]
    if global_memory.is_enabled():
        defs.append(remember_tool_def())
    return defs

def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolExecutionError(f"unknown tool: {name!r}")
    return tool(**arguments)
