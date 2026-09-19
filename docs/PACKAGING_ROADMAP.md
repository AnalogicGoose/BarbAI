# Packaging & Distribution Roadmap — Phase 4

Everything built so far (Phases 1–3) is **dev mode**: a Python project you
run from a source checkout with `uv run barbai`, configure with shell
environment variables, and talk to with `curl` or FastAPI's `/docs` page.
That's fine for building BarbAI; it's not something a regular user could
install and use. This doc is the register the user asked for: everything
that has to exist before "install it, use it" is true for someone who
has never heard of `uv`, GGUF files, or environment variables.

## The shape

Three install units, matching what the user described:

- **Core (required)** — the local server, exactly what exists today,
  packaged into something installable rather than a source checkout.
  Ships with a **bundled local web UI** out of the box - installing the
  core alone is a complete, usable product (a website you talk to
  BarbAI through), not just a headless API.
- **Desktop app (optional)** — a thin native wrapper that loads the
  core's own web UI in a window, plus OS integration (tray icon,
  autostart). Requires the core to already be installed and running -
  it's a client, not a second copy of the whole stack.
- **CLI (optional)** — a terminal chat client against the core's native
  API, for people who'd rather never open a browser. Also requires the
  core to be running; ships and installs independently of the desktop
  app.

Nothing here changes what Phase 2/3 build - this is entirely about
*how a non-technical person ends up with a working install*, layered on
top of the API that already exists.

## Current state, honestly

What's true today, so nothing below is solved twice or assumed already
done:
- **No web frontend exists at all.** `/docs` is FastAPI's own
  interactive Swagger page (a developer tool), not a chat UI. Phase 4.0
  starts from zero.
- **Hardware detection and model selection are disconnected.**
  `core/hardware.py`'s `detect_tier()` is purely informational right now
  - it only feeds `/health`'s `hardware_tier` field. Which model file
  actually loads is 100% manual, via `BARBAI_MODEL_PATH`/
  `BARBAI_CODING_MODEL_PATH`. There is no auto-download of anything.
- **All persistent state is CWD-relative**, which only makes sense from
  a dev checkout: `DEFAULT_MODEL_PATH` resolves relative to the
  installed package location, and `core/memory.py`'s
  `DEFAULT_SESSIONS_DIR` and `core/global_memory.py`'s
  `DEFAULT_STORE_PATH` both default to plain relative paths
  (`Path("sessions")`, `Path("global_memory.json")`) resolved against
  whatever the current working directory happens to be when the process
  starts. Launched from a desktop icon or a background service, "current
  working directory" is meaningless or arbitrary.
- **`BARBAI_TOOLS_ROOTS` defaults to the current working directory too**
  (`core/tools.py::_workspace_roots`) - fine when you `cd` into a repo
  and run `uv run barbai` there, meaningless/risky for a GUI-launched app
  with no natural "current directory."
- **One genuinely good default already in place:** the server binds to
  `127.0.0.1` only (`src/barbai/__init__.py::main`), not `0.0.0.0` -
  correct for a local-first tool and should stay the default through all
  of this, not get "fixed" into something more open.
- **No packaging, no installer, no service management, no
  code-signing** - `uv run barbai` in a terminal, full stop.

## Sub-phases

### Phase 4.0 — A bundled local web UI
The foundational piece - nothing else in "install the core, get a usable
app" is true until this exists.
- A minimal chat UI: message history, a thinking-mode toggle, a
  session picker (wired to the existing `session_id` mechanism).
  Talks to `/agent/chat` specifically, not the OpenAI/Anthropic
  passthrough endpoints - it needs the tool-execution loop and the mode
  field, and those endpoints exist for external clients, not this one.
- **The approval-pending UI is the safety-critical part, not an
  afterthought.** `write_file`/`remember` are gated specifically so a
  human reviews them before they run (`core/agent.py`'s
  `pending_approval` flow) - that only means anything if the UI shows
  the exact tool name and arguments clearly and makes approve/deny an
  explicit action, not a rubber stamp.
- Build approach: a static single-page app, built once and served by
  FastAPI itself via a `StaticFiles` mount - keeps "the core ships with
  the UI" trivially true (one process, one port, no separate web
  server to run or fail independently).
- Mode switching (General/Coding) exposed once there's a real reason to
  use it from the UI - today `mode: "coding"` is functionally identical
  to General with a different prompt (see
  `docs/CODING_AGENT_ROADMAP.md` Phase 3.0), so this can lag slightly
  behind the rest of 4.0 without blocking it.

### Phase 4.1 — Zero-config first run
- **Wire hardware detection to actual model selection.** Reuse the
  tier→model picks already validated in `docs/BARBAI_ROADMAP.md` Phase
  2.1 (Qwen3.5-9B for `default`, Qwen3.5-4B for `fast`, `quality` tier
  still unresearched) as a real download step: detect tier, auto-fetch
  the matching GGUF on first run if none is configured, with progress
  shown in the UI - not a manual `hf download` command a user has to
  find and run themselves.
- **Per-OS user-data directories**, replacing the CWD-relative defaults
  called out above. Standard locations (e.g. via a library like
  `platformdirs` rather than hand-rolling per-OS path logic): sessions,
  global memory, and downloaded models all move under something like
  `~/.local/share/BarbAI/` (Linux), `~/Library/Application
  Support/BarbAI/` (macOS), `%LOCALAPPDATA%\BarbAI\` (Windows). The
  existing env vars (`BARBAI_SESSIONS_DIR`, `BARBAI_GLOBAL_MEMORY_PATH`,
  `BARBAI_MODEL_PATH`) stay as overrides for power users - just not the
  defaults a fresh install relies on.
- **A real default workspace folder** for `BARBAI_TOOLS_ROOTS`, instead
  of "wherever the process happened to start" - something explicit and
  visible, e.g. a `BarbAI Workspace/` folder created under the user's
  home directory on first run, that the user can see and understand
  (this is the folder BarbAI can read/write in), not an invisible
  default tied to launch mechanics.
- A first-run flow tying it together: detect hardware → show the
  recommended tier/model → download with progress → confirm the
  workspace folder → ready. Lives in the Phase 4.0 UI, not a separate
  installer step, so it works the same whether launched from the
  installer's first run or a manual restart.

### Phase 4.2 — Packaging the core into an installable binary
Turning "a source checkout plus `uv run`" into "download one file,
install, done." **Real open decision, not made here:**
- **Bundle a Python runtime + deps into a single executable**
  (PyInstaller, Nuitka, or similar) - feels like a normal app, but
  `llama-cpp-python`'s compiled CUDA/Metal extension makes correct
  per-platform bundling meaningfully harder than a typical pure-Python
  PyInstaller build.
- **Ship a native installer that installs `uv` + this project under the
  hood** - much simpler to build and maintain, still a double-click
  installer from the user's perspective, at the cost of a `uv`-managed
  environment living on disk that the user never sees directly (fine, as
  long as it's genuinely invisible to them).

  Either approach needs separate build artifacts per OS/arch, and
  probably per GPU vendor - bundling every possible CUDA/Metal/CPU-only
  build into one installer is likely too large. **Also undecided:**
  whether the default download is CPU-only (smaller, always works, slow)
  with GPU acceleration as an explicit later step, or whether install-time
  GPU detection picks the right build automatically. Given install size
  and reliability matter more than raw capability for a first-run
  experience, leaning CPU-first is worth real consideration - but that's
  a call to make deliberately, not bury in an installer script.

### Phase 4.3 — Background/service management
Today the server has to run in a foreground terminal. Needed for
anything resembling a normal app:
- A tray/menu-bar icon: start, stop, open the web UI, quit.
- Optional, user-toggleable autostart on login - OS-native mechanisms
  (a systemd user unit on Linux, a LaunchAgent on macOS, a Startup-folder
  entry or scheduled task on Windows), not a custom cross-platform
  reimplementation.
- Logging to a file under the Phase 4.1 data directory, with rotation,
  instead of only stdout.
- Single-instance handling, so a user can't accidentally end up with two
  servers fighting over the same port.

### Phase 4.4 — Desktop app (optional install)
A thin wrapper, not a second implementation of anything:
- Loads the core's own web UI (`http://127.0.0.1:<port>`) in a native
  window. Tauri is the likely pick over Electron given llama.cpp already
  has a real memory/CPU footprint and Tauri's bundle/runtime overhead is
  much smaller - not finalized here, flagged as the open call it is.
- Provides the Phase 4.3 tray icon and autostart toggle as native OS
  integration, if that ends up living in the desktop app rather than the
  core's own background process (another open call - depends on whether
  a tray icon needs a GUI toolkit the headless core shouldn't carry as a
  dependency).
- Explicitly requires the core to already be installed and reachable -
  this app doesn't bundle a second copy of the model runtime or start
  its own server.
- Separate installer per OS (dmg/pkg, msi/exe, AppImage/deb/rpm).

### Phase 4.5 — CLI (optional install)
A terminal chat client against the already-running core, independent of
both the web UI and the desktop app:
- Interactive chat loop against `/chat` or `/agent/chat`, with a terminal
  equivalent of the approval UI for gated tools (show the pending tool
  call, y/n) - the same safety requirement as Phase 4.0's web version,
  just rendered differently.
- Basic admin commands worth having early: `status` (is the core
  running, which model/tier), `models list`/`models download`.
- **Naming/packaging is unresolved:** the core's own entry point is
  already named `barbai` (`pyproject.toml`'s `[project.scripts]`). Since
  the CLI installs independently *from* the core per the shape above, it
  likely needs its own distinct package and command name (e.g.
  `barbai-chat` as a separate PyPI/binary distribution) rather than
  colliding with or being folded into the core's command - not decided
  here.

### Phase 4.6 — Release engineering
- **Code signing / notarization is a real cost item, not just
  engineering work:** macOS Gatekeeper requires an Apple Developer
  Program membership (paid, annual) plus a notarization step; Windows
  SmartScreen reputation effectively requires a code-signing certificate
  (also a recurring cost). Flagging this explicitly so it's a conscious
  budget decision later, not a surprise when the first unsigned installer
  gets flagged as untrusted by both OSes.
- CI matrix builds (e.g. GitHub Actions) producing installers for
  Linux/macOS/Windows per release, instead of hand-building artifacts.
- At minimum, an update-check notice ("a newer version is available") in
  the UI/tray. Full silent auto-update is explicitly a stretch goal, not
  assumed - see "deferred" below.
- An actual versioning/release process - `pyproject.toml` still says
  `0.1.0` with no tagging convention established yet.

## Explicitly deferred - not needed to call this "shippable"

Matching this project's bias everywhere else (Phase 2.3's vector-store
deferral, Phase 3.4's AST/retrieval deferral): don't build these until
the lack is actually felt.
- **Full silent auto-update** - a manual "check for updates" notice
  (Phase 4.6) is enough for v1.
- **Telemetry or crash reporting** - actively counter to a local-first
  tool's whole premise; if this is ever added, it must be opt-in and
  clearly disclosed, never a default.
- **Remote/LAN access** - the `127.0.0.1`-only default (already correct
  today) should stay the default through all of Phase 4. Exposing BarbAI
  over a network is a real security-review-worthy decision on its own,
  not something to bundle into a packaging pass.
- **Mobile apps.**

## Open decisions this doc deliberately doesn't resolve

Called out inline above, collected here so they don't get silently
decided by default while implementing:
1. Core packaging approach (bundled interpreter vs. installer-that-installs-uv) - Phase 4.2.
2. CPU-first-by-default vs. install-time GPU detection - Phase 4.2.
3. Desktop app framework (Tauri vs. Electron vs. something else) - Phase 4.4.
4. Where the tray icon/autostart mechanism actually lives (core background process vs. desktop app only) - Phase 4.3/4.4.
5. CLI package/command naming, given it must install independently of the core - Phase 4.5.
6. Code-signing budget and timeline - Phase 4.6.
