# Coding Agent Roadmap — Phase 3

`docs/BARBAI_ROADMAP.md` names Coding mode as one of BarbAI's two product
shapes but never breaks it down — it's a single unstarted line item there.
This doc is that breakdown: Coding mode aims for Claude Code/Codex-level
capability (read the project, find the problem, edit files, run
tests/commands, verify the fix, explain what/why/result afterward — not a
snippet generator), including the ability to delegate scoped sub-tasks to
its own sub-agents. It's split into sub-phases (3.0–3.5) so it stays
buildable in the same incremental, verify-as-you-go style as Phase 2.

## Where this comes from

The sub-phase shape below was checked against a 6-pillar architecture
brief (Observe-Think-Act-Verify loop, local tool suite, context
extension/semantic search, multi-layer safety, plus this project's own
addition of multi-agent delegation) that the user got from Gemini,
prompted as a "Principal AI Systems Architect" persona. Treated as
reference material, not a spec to import wholesale — two of its pieces
(AST mapping via Tree-sitter, hybrid vector+BM25 retrieval) pull in real
dependencies and complexity that this project has consistently deferred
elsewhere (see `docs/BARBAI_ROADMAP.md` Phase 2.3: "add a vector store
only once the lack of retrieval is actually felt"). Same bias applies
here — they're Phase 3.4, explicitly not started, not scheduled ahead of
being needed.

## Baseline already built (don't re-derive this)

Everything below already exists and is tested (`uv run pytest`) as of
this doc:
- `core/tools.py`: `read_file`, `write_file`, `remember` — all scoped to
  the `BARBAI_TOOLS_ROOTS` allowlist, `write_file`/`remember` in
  `GATED_TOOLS`.
- `core/agent.py` + `api/agent.py`: a server-side tool-execution loop
  (`run_agent`) with a `pending_approval`/resume flow for gated tools —
  this **is** the human-in-the-loop mechanism pillar 4 asks for; it
  already exists, Coding mode extends it rather than building a new one.
- `core/memory.py` (per-session) and `core/global_memory.py`
  (cross-conversation, explicit-only) — conversation memory, separate
  concern from code/repo context (Phase 3.4 below).
- `core/hardware.py`: tier detection or a coding-model tier table.
- `core/persona.py`: identity/system-prompt composition, currently one
  persona (BarbAI general-purpose).

## Sub-phases

### Phase 3.0 — Foundation: a second model, and a way to switch to it — mostly done
Coding mode is a *different model*, loaded instead of General's (roadmap:
only one model in VRAM at a time). Nothing else in this doc works without
this piece landing first.
- **Done.** `core/model_runtime.py`: a mode concept (`MODES = ("general",
  "coding")`) with `ensure_mode(mode)` — loads/switches only when the
  requested mode isn't already active, so staying in one mode never pays
  for a reload. `BARBAI_CODING_MODEL_PATH` selects the Coding-mode GGUF,
  falling back to `BARBAI_MODEL_PATH`/the General default until a real
  coding model is picked (next bullet) — the switching mechanism is fully
  usable and tested today with Qwen3.5-9B standing in for both modes.
- **Done.** A Coding-specific persona (`CODING_SYSTEM_PROMPT` in
  `core/persona.py`) — instructs the model to explain the fix afterward,
  not just dump a diff (the actual product differentiator vs. "snippet
  generator" per the original roadmap's framing). `build_system_prompt`
  takes a `mode` param that picks the base identity; everything else
  (thinking nudge, remembered facts, custom prompt) layers on top the
  same way regardless of mode.
- **Done.** `/agent/chat` is mode-aware: a `"mode": "general" | "coding"`
  field (default `"general"`) picks the model via `ensure_mode` and the
  persona via `build_system_prompt`. `/chat` and the OpenAI/Anthropic
  passthroughs are still General-only by design (see that endpoint's own
  docstring) — a mode switch via `/agent/chat` affects what they serve
  too, since there's only ever one model loaded process-wide, but they
  have no `mode` field of their own.
- **Decided, `default` tier: no dedicated coding model — reuse
  Qwen3.5-9B.** Full reasoning in `docs/BARBAI_ROADMAP.md` Phase 2.1: the
  obvious pick (Qwen2.5-Coder-7B) is the exact model this project already
  found broken for structured tool calls, no official small Qwen3.5-Coder
  exists yet, and the one community coder fine-tune of Qwen3.5-9B is
  unvetted. So `mode: "coding"` on the `default` tier is, for now,
  deliberately just "General's model with a different system prompt" -
  not a placeholder waiting to be filled in, an actual decision to defer
  a separate model until Phase 3.1+ usage shows Qwen3.5-9B is actually
  insufficient for coding tasks.
- **`fast` tier (3050), both modes: Qwen3.5-4B, tested on simulated
  hardware (fallback Qwen3.5-2B, also tested).** Full results in
  `docs/BARBAI_ROADMAP.md` Phase 2.1 - validated via `scripts/hog_vram.py`
  forcing this dev machine's free VRAM down into the `fast` tier's range,
  not real 3050 hardware. Qwen3.5-4B: clean load, correct tier detection,
  3/3 reliable tool calls, approval gate confirmed working. Qwen3.5-2B
  (deliberately tested under a tighter simulated constraint, the actual
  scenario it exists for): tool calls stayed 6/6 structurally reliable,
  but about half its replies leaked planning-talk into the final answer
  instead of a clean direct response - a real quality gap, not just
  "smaller and a bit worse." Use 4B whenever it fits; 2B is a fallback
  for VRAM too tight for 4B, not an equivalent option. Real 3050
  confirmation is still the open item.
- **Not started:** enforcing Coding-only tools by mode (the `mode` field
  currently only affects model/persona, not tool exposure) — moot until
  Phase 3.1 actually adds Coding-only tools like `patch_file`/
  `run_command`; nothing to restrict yet.
- **Resolved — memory/context scoping.** Checked against how Claude Code
  and Codex actually do this: neither splits by a flat "chatbot vs.
  coding agent" switch — Claude Code scopes history **per project**
  (`~/.claude/projects/<repo-path>/`, a separate transcript for every
  working directory), completely disconnected from claude.ai's own chat
  memory, which is a different product and mechanism entirely.
  - **Session memory (`core/memory.py`) splits by project, via
    `session_id` convention, not new storage code.** `session_id` is
    already just an opaque caller-supplied string — a Coding-mode client
    derives it from the working directory (e.g. a hash of the project
    path) instead of a random id, and General vs. Coding sessions are
    already fully isolated with the mechanism that exists today. One
    small addition still worth making: mode-scoped storage roots so the
    files are organized on disk (`sessions/general/` vs
    `sessions/coding/<project>/`, not one flat directory) — an
    organizational nicety, not a functional requirement. The `/agent/chat`
    `mode` field itself already exists (Phase 3.0 above); using it to
    enforce Coding-only tools is Phase 3.1's job, once Coding-only tools
    (`patch_file`, `run_command`) actually exist to restrict.
  - **Global memory (`core/global_memory.py`) stays ONE shared store**
    across both modes — decided explicitly, not defaulted. A remembered
    fact is about the *user* ("prefers concise answers"), not the task,
    so it's equally relevant whether you're chatting or coding. This is
    the one place BarbAI's design deliberately diverges from the
    Claude Code precedent (which shares nothing with claude.ai's memory)
    — the divergence is intentional, not an oversight.

### Phase 3.1 — Expanded local tool suite — done
`read_file`/`write_file` are whole-file only and capped at 100KB/1MB —
fine for a general assistant, not enough for real repo work. New tools,
all scoped to the existing `BARBAI_TOOLS_ROOTS` allowlist pattern:
- **Done.** `list_directory` (`core/tools.py`) — read-only, ungated,
  `ls`-style by default, `recursive=true` for a `tree`-style full
  listing that skips common noise directories (`.git`, `node_modules`,
  `__pycache__`, `.venv`, build artifacts, ...) and is capped at 500
  entries.
- **Done.** `search` (`core/tools.py`) — read-only, ungated, shells out
  to ripgrep (`rg`) rather than reimplementing regex/gitignore/binary
  handling, returning `path:line:matched_text` results, capped at 200.
  Requires the `rg` binary on `PATH` - not a Python dependency, so it's
  not in `pyproject.toml`; install it separately (see README). Verified
  live against a running server: correctly chains with `list_directory`
  and `read_file` to answer real "where is X defined" questions.
- **Done.** Line-range reads — extended `read_file` with optional
  `start_line`/`end_line` (1-indexed, inclusive) rather than adding a
  separate tool. Streams the file instead of reading it whole first, so
  a narrow range on a file far past `MAX_FILE_BYTES` still works; the
  *returned slice* is still capped at `MAX_FILE_BYTES`. Verified
  byte-exact against the real file on disk in a live test.
- **Observed here, acted on in Phase 3.2:** a live two-tool-call request
  (search, then a line-range read) occasionally hit `MAX_ITERATIONS`
  (10) even though both tools work correctly in isolation and the same
  combined request succeeds on a retry - sampling variance on a
  genuinely multi-step task, not a tool bug. Phase 3.2 raised
  `MAX_ITERATIONS` to 20 to give this real headroom.
- **Done.** `patch_file` (`core/tools.py`) — exact-string search/replace
  (`old_string`/`new_string`, `replace_all` to change every match instead
  of requiring exactly one), the same shape as Claude Code's own Edit
  tool rather than a unified diff - far more forgiving for a small local
  model to produce correctly than line-numbered diff hunks. **Gated**,
  same reasoning as `write_file`. Only edits existing files (use
  `write_file` to create one).
  - **Cross-platform correctness (Linux/macOS/Windows is a hard
    requirement - see project memory):** detects and preserves a file's
    *existing* line-ending convention (`\r\n` vs `\n`) rather than
    letting Python's default text-mode write silently normalize it to
    whatever the host OS prefers - editing one line of a Windows-authored
    CRLF file from a Linux server no longer rewrites every line ending in
    it. `write_file` got the same fix for the same reason (see
    `core/tools.py::_detect_newline`). Verified live, not just unit
    tested: asked the running model to edit a genuine CRLF file via
    `patch_file`, approved the gated call, and confirmed byte-for-byte
    that every remaining line ending stayed `\r\n` - only the intended
    line changed.
- **Done.** `run_command` (`core/tools.py`) — a real shell tool (`pytest`,
  `npm run build`, ...). **Gated.** Working directory must resolve
  within the same `BARBAI_TOOLS_ROOTS` allowlist file tools use (no
  separate allowlist). Design checked against OpenAI Codex, GitHub
  Copilot, and Claude Code's own Bash tool before building (see
  `core/tools.py` module docstring for the full comparison): Codex and
  Copilot both do real OS-level sandboxing (macOS seatbelt, Windows
  native sandbox) - explicitly Phase 3.3 territory, not this pass, since
  command sanitization is already deferred there. Claude Code's own Bash
  tool is the closer model for where this project is now - its proven
  numbers were adopted directly: 120s default timeout (600s cap, same as
  Claude Code's 2min/10min), and a 30,000-character output cap with
  *middle* truncation (keep the start and end, cut the middle - errors
  are usually at the end, setup context at the start). Its persistent
  shell session and background-execution support were deliberately
  **not** adopted - both add real state across calls that doesn't fit
  this project's stateless-per-call tool model without their own design
  pass; every `run_command` call is a fresh subprocess, `cwd` passed
  explicitly each time. A nonzero exit code is normal output (a failing
  test), never raised as an error - only things that mean the command
  genuinely couldn't be evaluated (bad `cwd`, timeout, no such shell) do.
  - **Verified live, full loop, not just unit tests:** gave the running
    model a file with a deliberately broken function and a failing test,
    and one instruction ("run the tests, fix what's broken, verify it
    passes"). Unprompted structurally, it ran `run_command` (test
    failed) → `read_file` → `patch_file` (fixed the actual bug) →
    `run_command` again (confirmed the fix), all through the approval
    gate, and correctly summarized what was wrong. The file on disk was
    genuinely fixed. This is Phase 3.2's Observe-Think-Act-Verify loop
    already happening with the current tool set, even before that phase
    formally builds the scaffolding to make it reliable/automatic.

### Phase 3.2 — The Observe-Think-Act-Verify loop — done
Turned out to need less new scaffolding than expected, because the
first half was already happening. A live Phase 3.1 test gave the model
a broken function and one instruction ("run the tests, fix it, verify")
and it chained `run_command` → `read_file` → `patch_file` →
`run_command` correctly on its own, no special-casing required - the
existing tools plus the existing loop already exhibit
Observe-Think-Act-Verify emergently. Building a forced "always run tests
after an edit" step on top of that was considered and rejected: there's
no reliable way to know what "verify" means for an arbitrary project
(pytest vs npm test vs cargo test vs ...) without per-project config
that doesn't exist yet, and it wasn't needed anyway.
- **Done.** What the loop actually lacked was a way to recognize when
  that process *stalls* - the real gap, and what this phase built.
  `core/agent.py` now detects when the exact same tool call (name +
  arguments) produces the exact same result two rounds in a row and
  stops with a clear explanation (`{"status": "stuck", ...}`) instead of
  silently burning the rest of the iteration budget repeating a dead
  end. Deliberately mechanical (compare round signatures), not an
  LLM-judged "is this looping" call - simple and predictable beats
  clever here. Works across an approval-gate pause/resume too, not just
  within one call, since the check reads from the full conversation
  history either way. Verified two ways: 5 unit tests covering the
  trigger case and three ways it must *not* false-positive (different
  arguments, different results, only one round so far), plus a live
  request against the real `/agent/chat` API confirming the new `stuck`
  status and its `conversation` field come back correctly end-to-end.
- **Done.** `MAX_ITERATIONS` raised from 10 to 20. Not about the
  gated write/verify loop (each approval pause naturally resets the
  budget on resume) - it's ungated read-investigation chains
  (`list_directory`/`search`/`read_file`) that can now legitimately run
  longer with seven tools available than the three-tool budget this was
  first tuned against, per the sampling-variance iteration failure noted
  in Phase 3.1.

### Phase 3.3 — Safety hardening
`run_command` and `patch_file` are real risk surface — harden once
they're working, not before (a restricted, gated tool that does nothing
dangerous yet is safe to iterate on).
- Command sanitization/denylist for `run_command`: regex-screen for
  destructive patterns (`rm -rf /`, `sudo`, `curl|sh`/`wget|sh` piping,
  writes outside the working-directory allowlist, force-pushes,
  credential-shaped strings) — this is the Mana tripwire model
  `docs/BARBAI_ROADMAP.md` section 3 already cites as a pattern worth
  reusing, applied concretely for the first time here.
- `patch_file` and `run_command` both confirmed in `GATED_TOOLS`.
- A cost/time guard beyond `MAX_ITERATIONS`: a wall-clock budget for the
  whole loop, not just a per-command timeout.
- Depends on 3.1, 3.2.

### Phase 3.4 — Context extension (deferred until the lack is felt)
**Not started, not scheduled ahead of need** — same "don't build
speculatively" call already made for memory's vector store
(`docs/BARBAI_ROADMAP.md` Phase 2.3). `search` + line-range reads (3.1)
cover real usage for a while; only reach for this once a real repo
actually exceeds what those two can find.
- AST-based code mapping (Tree-sitter) for a function/class/import graph
  across files, so the agent can navigate by structure, not just text
  search.
- Hybrid retrieval (a local vector store + BM25 keyword matching) to pull
  relevant snippets into context instead of relying on the model's raw
  context window.

### Phase 3.5 — Multi-agent delegation
The user's explicit requirement beyond the Gemini brief: the coding
agent should be able to delegate scoped sub-tasks instead of doing
everything serially in one loop, the way Claude Code's own subagent tool
works.
- A `dispatch_agent`-style tool: spawns a new, scoped `run_agent` loop
  with its own tool subset/allowlist and a focused task description,
  returns its result to the parent loop.
- **Open design question — isolation:** does a sub-agent share the
  parent's `session_id`/global memory, or start fresh with only the task
  description it's given? Recommend fresh/scoped-only for v1 — avoids
  context bleed between parent and sub-agent, matches how a fresh
  (non-forked) subagent works in Claude Code itself.
- **Open design question — concurrency:** sequential-only dispatch for
  v1 (simplest, avoids concurrent file writes/tool execution racing each
  other); parallel dispatch is a later increment once sequential is
  solid.
- Depends on 3.1–3.3 being solid — a shaky core loop compounds badly once
  multiplied across sub-agents.

## Pillar-to-phase map

| Gemini brief pillar | Landed in |
|---|---|
| 1. Observe-Think-Act-Verify loop | Phase 3.2 |
| 2. Local tool suite (fs nav, patching, PTY) | Phase 3.1 |
| 3. Context extension / semantic search | Phase 3.4 (deferred) |
| 4. Multi-layer safety (HITL, sanitization, loop guard) | HITL already built (`core/agent.py`'s approval gate); sanitization + loop guard in Phase 3.3 |
| (project addition) Multi-agent delegation | Phase 3.5 |
