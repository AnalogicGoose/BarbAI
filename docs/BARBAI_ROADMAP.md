# BarbAI Roadmap — Phase 2 (Production Assistant)

Phase 1 (`src/llmtest/`, this repo's NumPy engine) is done as a learning
exercise: custom autograd `Tensor`, `Linear`/`LayerNorm`/`GELU`/`Embedding`/
`CausalSelfAttention`/`TransformerBlock`, char-level training in `train.py`.
It taught the transformer mechanics; it does not and cannot produce
coherent answers at any scale reachable on pure NumPy/CPU training. It is
not the starting point for BarbAI's actual brain — Phase 2 below is a
separate build around a real pretrained model.

## Product shape: General vs. Coding modes

BarbAI is not trying to beat GPT/Claude/Codex — it's a **local fallback**
for when we're out of tokens/credits or offline. Two modes, one model
loaded at a time:

- **General** — conversation, everyday knowledge, basic programming/math,
  explanations, tool use (web, PC/app control, reminders).
- **Coding** — an actual agent loop (read the project, find the problem,
  edit files, run tests/commands, verify the fix, explain what/why/result
  afterward), not a snippet generator. This depends on the agent runtime
  and tool access at least as much as on the model itself. Broken down
  into buildable sub-phases in
  [`docs/CODING_AGENT_ROADMAP.md`](CODING_AGENT_ROADMAP.md) — this doc
  just names it as a destination, that one is the actual plan.

Only one mode's model occupies VRAM at a time: switching modes unloads the
current model and loads the other. Never run both simultaneously — on an
8GB card in particular, the OS/desktop/other apps still need headroom, so
the goal is a well-quantized model (Q4-class) that leaves room, not the
largest model that technically fits.

**Headroom must be measured, not assumed.** This is a personal laptop, not
a dedicated inference box — VRAM can already be spoken for before BarbAI
loads anything. All the "fits in N GB" guidance in this doc (Mana's
thresholds and the model shortlist below) comes from sources that size a
model against a card's *total* VRAM, as if the whole card is dedicated to
it — none of them model concurrent GPU usage on the same machine.

What actually competes for that VRAM depends on the laptop's graphics
mode, which gaming laptops (3050/4070/5070 Ti are all "gaming" tier cards)
routinely switch between:

- **Hybrid/Optimus (the common default)** — the integrated GPU drives the
  desktop, browser, and most everyday apps; the discrete NVIDIA GPU sits
  mostly idle until something explicitly launches on it. On this setup the
  dominant real risk isn't desktop/browser overhead, it's **a game (or
  another dGPU-bound app) already running** when BarbAI tries to load a
  model — that can eat most or all of the card's VRAM.
- **Discrete-only / MUX switched** (common when a user flips to max
  gaming performance via vendor GPU-switch software) — the dGPU then
  drives the whole desktop like a normal single-GPU machine, so it *does*
  carry real desktop/browser overhead on top of whatever game is running.

Either way, and even if the mode changes mid-session (MUX flip, a game
launching after BarbAI already picked a tier), a fixed assumption is
wrong. The hardware-detection module (Phase 2.0) must check **free** VRAM
at the moment of loading (`nvidia-smi --query-gpu=memory.free`, not
`memory.total`), reserve an explicit floor (e.g. always keep 1-1.5GB
free), and fall back to a smaller quant/model — or refuse and warn — if
free VRAM is under that floor, rather than trusting the static table
blindly. This single mechanism covers both graphics modes correctly
without needing separate hybrid-vs-discrete logic.

### Layered architecture

```text
Interfaces (Desktop app / CLI / Web / API clients)
        ↓
      API (FastAPI)
        ↓
  Agent Runtime  ←→  Tools (filesystem, terminal, git, web, reminders, ...)
        ↓
  Model Runtime (llama-cpp-python, one model loaded at a time)
        ↓
    Local LLM (GGUF, per active mode + hardware profile)
```

The agent loop itself: `LLM → tool call → tool result → LLM → ... → final
response` — same shape as section 2's tool-calling test below, and the
same shape Mana already runs. Interfaces stay decoupled from the model:
any client (desktop app, CLI, another application) talks to the same API,
the same way Mana's OpenAI-compatible endpoints let external tools plug in
for free (section 4 below).

## Reference project

A friend shared **[Mana](https://github.com/Yuuzulight/Mana)** — a mature,
local-first Windows AI companion (Electron + Node.js backend,
`llama.cpp`/`whisper.cpp`, OpenAI-compatible API, plugin system, memory
graph, tool-calling, approval gates). It's a different stack (Node, not
Python/FastAPI) and a different target (Windows desktop companion with
avatar/voice, not a headless API service), so it isn't something to build
on top of directly — but several of its design decisions are already
tested against real hardware and real models, and are worth carrying over
as patterns. Extracted below, each tagged with where it came from in that
repo.

## Extracted patterns worth reusing

### 1. Hardware-aware model tiering (`node-bot/model-management.js`)

Mana detects actual VRAM via `nvidia-smi` (falls back to system RAM if no
NVIDIA GPU) rather than trusting a GPU's nominal spec — a real 16GB card
reports ~16000-16300MB after driver/OS reservation, so their cutoff is
15360MB, not 16384. Their thresholds:

| Detected VRAM | Profile | Model class |
|---|---|---|
| < 8GB | `fast` | 1.5B-4B class (Qwen3-1.7B/4B) |
| 8-15GB | `default` | 4B class (Qwen3-4B) |
| 15GB+ | `quality` | 8-14B class (Qwen3-8B/14B) |

**Applied to our 3 laptops** — detect at runtime, don't hardcode by GPU
name (a "RTX 3050 Laptop" ships as either 4GB or 6GB depending on OEM SKU,
so the name alone isn't enough):

| GPU (laptop) | Typical VRAM (verify via `nvidia-smi`, don't trust the spec sheet) | Tier | Suggested model class |
|---|---|---|---|
| RTX 3050 | 4-6GB | fast | 1.5B-4B, Q4 |
| RTX 4070 | 8GB (right at the tier boundary) | default | 4B comfortably, 7-8B Q4 tight |
| RTX 5070 Ti | ~12GB | default/quality | 8B comfortably, 14B Q4 possible |

Implementation note: store this table as data a detection module reads,
not as per-GPU branches in code — a 4th machine later is a new row, not
new logic.

**Note on manual vs. auto profiles:** a ChatGPT-drafted spec for this
project argued for fully manual, predefined profiles with no GPU
auto-detection at all. Decision: **keep Mana's auto-detection** — when a
tested, running reference implementation and a ChatGPT brainstorm disagree,
the reference implementation wins (see project memory). This isn't actually
in tension with the "predefined profiles, no silent auto-download" spirit
of that spec, though: detection only *selects a row* in the fixed table
above at startup — it never searches the internet or downloads a model on
its own. Each user still installs only the GGUF file(s) their own row
calls for.

### 2. Tool-calling reliability is model-specific, not infra-specific (`docs/roadmap/issue-51-tool-calling.md`)

Tested directly against `llama-server` with a real OpenAI-style `tools`
array:

- **Qwen3-4B** — reliable, 3/3 requests returned a correctly structured
  `tool_calls` array.
- **Qwen2.5-Coder-7B-Instruct** — produces the *correct* tool call JSON but
  wraps it in a markdown code fence instead of the `<tool_call>` tags its
  own chat template asks for, so the structured-output parser never
  recognizes it. `--tools`/`--jinja` being enabled in the llama.cpp build
  does not guarantee any given model actually complies with its own
  template.

**Lesson:** test tool-calling reliability against the *specific* model
we ship before building a tool loop around it — "instruct model" is not
sufficient evidence it emits parseable tool calls.

### 3. Approval gate scoped to agent-authored content, not every tool call (`node-bot/approval-gate.js`)

Not a blanket "confirm every action" — Mana explicitly does not gate
routine reads (web search, memory reads). It gates *content the agent
authored itself* before it's trusted (a generated script, a proposed file
edit), plus lightweight regex tripwires (`child_process`/`exec`,
filesystem writes, `curl|wget ... | sh`, credential-shaped strings) that
flag a pending request for human review — never auto-deny, always a human
call.

### 4. OpenAI-compatible surface (`node-bot/server.js`, `server-routes.js`)

`/v1/chat/completions`, `/v1/embeddings`, `/v1/models` — lets any existing
OpenAI-client tool talk to the local backend for free. Directly maps onto
the "API-type AI" goal; a FastAPI service can mirror this exact shape.

### 5. Plugin architecture (`plugins/README.md`)

Each optional feature is its own package — own dependencies, own tests,
own README, an explicit `defaultEnabled` flag — loaded at startup instead
of hardcoded into the core server. Relevant once there are two people
adding capabilities: independent plugins avoid both of us editing the same
core file.

### 6. Memory beyond flat RAG (`node-bot/acp-memory-store.js`)

Session memory plus idle-triggered consolidation (deciding what's actually
worth remembering, not embedding everything indiscriminately), entity
tagging, and cross-session linking — richer than "embed every message and
cosine-search it," worth keeping in mind once memory becomes a real need.

### Explicitly not carried over (yet)

Live2D/VRM avatar, Telegram/Discord/Matrix remote bridges, FFXIV/stock
market plugins, mobile companion app — all real Mana features, all
orthogonal to "can it reliably answer things," and all scope creep for a
v1. Revisit only after the core loop below is solid.

## Roadmap

Phases 1–3 (here and in `docs/CODING_AGENT_ROADMAP.md`) are all dev-mode
work: a source checkout run with `uv`. Turning that into something a
regular user can install - a bundled web UI, an actual installer, no env
vars to hand-configure - is its own phase, broken down separately in
[`docs/PACKAGING_ROADMAP.md`](PACKAGING_ROADMAP.md).

### Phase 2.0 — Core loop (single machine, text only) — done, and then some
- FastAPI service — shipped with *three* wire-compatible surfaces instead
  of just OpenAI: `POST /v1/chat/completions` (OpenAI-compatible),
  `POST /v1/messages` (Anthropic-compatible), and `POST /chat` (BarbAI's
  own minimal native shape, no vendor envelope). All three stream via SSE
  (`stream: true`), including correct mid-stream tool-call detection.
- `llama-cpp-python` loading a GGUF model, CUDA-accelerated. Replaces the
  NumPy engine as the actual brain — Phase 1's engine has served its
  (educational) purpose.
- Hardware-detection module (`src/barbai/core/hardware.py`, `nvidia-smi`
  via subprocess) picks a model tier per the table in section 1.
- Not originally scoped here, added along the way: a default BarbAI
  identity/persona (`src/barbai/core/persona.py`, always injected unless
  the caller overrides it, with custom prompts layered on top rather than
  replacing it) and a `fast` / `thinking` / `extended` reasoning-effort
  switch per request (`src/barbai/core/model_runtime.py`) — Qwen3.5 is a
  reasoning model that otherwise always pays for a "thinking" pass, even
  on trivial questions.

### Phase 2.1 — Model tiers for the 3 target machines — partially done
- Verified on real hardware: **RTX 4070 only** (this dev machine). Idle
  free VRAM measures ~7780MiB out of 8188MiB total — confirms free VRAM
  runs meaningfully below total even at idle, which is why tiering is
  based on free VRAM, not the spec-sheet number.
- 3050 and 5070 Ti: **still not verified** — nobody's had that hardware in
  hand yet. The 5070 Ti's spec used here was also corrected mid-project
  from an earlier (wrong) "16GB" assumption to the real ~12GB.
- Current tier boundaries (`core/hardware.py`), recalibrated from Mana's
  original numbers (which assumed a 16GB reference card) against what's
  actually been measured plus the corrected 5070 Ti spec:

  | Free VRAM | Tier | Machine (assumed, only 4070 confirmed) |
  |---|---|---|
  | < 7168MiB | `fast` | 3050 |
  | 7168–10240MiB | `default` | 4070 ✅ confirmed |
  | ≥ 10240MiB | `quality` | 5070 Ti (estimated, not measured) |

  The `default`/`quality` boundary (10240) is an estimate assuming the
  5070 Ti loses a similar ~400MiB to idle desktop/driver overhead as the
  4070 does — verify for real once someone has that laptop.
- General model **picked and fully validated** for the `default` tier:
  **Qwen3.5-9B, Q4_K_M** (`bartowski/Qwen_Qwen3.5-9B-GGUF`). CUDA-loads
  fully offloaded, tool-calling verified reliable (see Phase 2.2 note
  below), persona/thinking-mode features all tested against it.
- **Coding-mode model, `default` tier — decided: reuse Qwen3.5-9B, no
  separate coding model for now.** Researched the obvious alternative
  first: Qwen2.5-Coder-7B tops most "best local coding LLM" rankings, but
  it's the exact model this project already tested and found broken for
  structured tool calls (Phase 2.2 note below) — and a fresh check
  confirms that's not a fluke, it's a widely-documented, well-understood
  failure (vLLM issues #10952/#32926, multiple client-side workarounds):
  Qwen2.5-Coder emits the tool-call JSON inside a markdown code fence
  instead of the tagged format its own template calls for, and gets
  *worse* at it as the prompt grows (5/5 fenced at ~8k tokens vs. 0/5 at
  ~200) — the opposite of what a coding agent, which needs to hand it
  real file contents, can tolerate. No official small Qwen3.5-Coder
  variant exists yet to fall back to (Qwen's only 3.5-generation coder
  release so far, Qwen3-Coder-Next, is 80B - far outside this tier); a
  community fine-tune (`Jackrong/Qwopus3.5-9B-Coder`) exists but is an
  unvetted single-contributor release with no confirmation it preserves
  Qwen3.5's tool-call template. Given all that, reusing Qwen3.5-9B costs
  nothing (Phase 3.0's `BARBAI_CODING_MODEL_PATH` already falls back to
  it) and inherits tool-calling reliability that's actually verified,
  rather than adding an unvalidated model for a marginal, unconfirmed
  coding-quality gain. Revisit only once real Coding-mode usage (Phase
  3.1+) shows Qwen3.5-9B is actually insufficient for coding tasks - not
  before, matching this project's "don't build until the lack is felt"
  bias elsewhere (Phase 2.3's vector-store deferral, Phase 3.4's
  AST/retrieval deferral).
- **`fast`-tier model, both modes — Qwen3.5-4B, validated on simulated
  hardware.** Same in-family reasoning as the `default` tier pick above:
  Qwen's official 0.8B/2B/4B/9B GGUF variants (released March 2026) very
  likely share Qwen3.5-9B's tag-based tool-call format, already handled
  by `core/tool_calls.py`, versus an unrelated small model needing its
  own from-scratch validation. **Tested** on this project's own 4070
  with a script (`scripts/hog_vram.py`) that grabs GPU memory via a raw
  `cudaMalloc` call to force `core/hardware.py`'s free-VRAM detection
  down into the `fast` tier's range (<7168MiB) without needing to own a
  3050 - not a substitute for real 3050 hardware, but real validation
  rather than a paper pick. Result at ~5.8GB free: loaded cleanly,
  `/health` correctly reported `fast`, and 3/3 `/agent/chat` requests
  requiring a `read_file` tool call returned correct, well-formed
  results - the same reliability bar Qwen3.5-9B was validated against
  (Phase 2.2 below). The `write_file` approval-gate pause was also
  confirmed working correctly at this model size.
- **Fallback, tighter VRAM — Qwen3.5-2B, tested, real quality caveat
  found.** Tested the same way at a deliberately tighter ~3.9GB free
  (simulating a worse-case 4GB 3050 SKU, the actual scenario this
  fallback exists for): loaded cleanly, tier detection correct, and tool
  calls were still **6/6 structurally well-formed** - the narrow
  "does it call tools correctly" bar Phase 2.2's method checks for. But
  a distinct, reproducible quality problem showed up in the same test
  run: in roughly half the replies, the model's final answer (the
  `reply` field, not `reasoning`) was self-narration ("The user asked me
  to... I should respond with this information in a friendly way as
  BarbAI would.") instead of a clean direct answer - the underlying
  facts were usually still in there, just wrapped in leaked
  planning-talk rather than stated directly. This is real evidence
  Qwen3.5-2B is meaningfully worse than 4B for actual use, not just a
  smaller/slower version of the same thing - reach for 4B whenever it
  fits, and treat 2B as a last resort for VRAM too tight for 4B, not an
  equivalent alternative.
- Both are models Qwen released and GGUF-quantized directly, both are
  downloaded and sitting in `models/` on this dev machine already
  (`Qwen3.5-4B-Q4_K_M.gguf`, `Qwen3.5-2B-Q4_K_M.gguf`) - a real 3050
  owner can confirm these results without re-downloading anything.
- `quality`-tier (5070 Ti) candidates: still not researched - out of
  scope for this pass, revisit when that hardware question comes up.

### Phase 2.2 — Tool calling — read + write tools done, gated by an approval flow; shell tool not started
- Tool-calling reliability tested directly against Qwen3.5-9B (per section
  2's method): the model reliably emits well-formed tool calls, but in its
  *own* tag format (`<tool_call><function=...><parameter=...>`), not the
  Hermes-style JSON `llama-cpp-python`'s built-in parser expects — same
  class of mismatch section 2 flagged for Qwen2.5-Coder-7B. Handled with a
  custom parser (`core/tool_calls.py`, `parse_tool_call_tags` /
  `TagStreamParser` for the streaming case) that falls back only when the
  native parser comes up empty.
- Two tools now, both in `core/tools.py`, scoped to the same allowlist
  (`BARBAI_TOOLS_ROOTS`, comma-separated) that can mix whole directories
  and individual files, and supports multiple attached roots at once
  (Codex-style) — rejects path traversal and absolute-path escapes in
  either case:
  - `read_file` — read-only, unchanged since it first shipped.
  - `write_file` — creates or overwrites a text file (`create_dirs` to
    make missing parent directories); resolves a relative path against
    an existing match first (same priority as `read_file`), falling back
    to the first configured directory root for a brand-new file.
- Wired into the server-side agent loop (`core/agent.py`, `POST
  /agent/chat`) that executes tools itself and loops until the model
  gives a final answer, unlike the three passthrough endpoints, which
  correctly hand a `tool_call` back to the caller instead (that's the
  right behavior for OpenAI/Anthropic-client compatibility).
- Shell tool: **not started**, per the "hold off" guidance above.
- Approval gate (section 3): **built**. `GATED_TOOLS` in `core/tools.py`
  marks which tools need a human decision before they run (currently just
  `write_file` — `read_file` stays ungated, matching section 3's own
  model, Mana, which doesn't gate routine reads). When the agent loop
  hits a gated call it pauses instead of executing it and returns
  `{"status": "pending_approval", "pending": [...], "messages": [...]}`;
  the caller resumes by calling `run_agent` again with that same
  `messages` list and an `approvals` dict mapping each pending
  `tool_call_id` to `True`/`False`. `POST /agent/chat` mirrors this at
  the API layer via `conversation` (the opaque paused state) and
  `approvals` — there's no server-side session store, the caller holds
  the paused conversation between requests, consistent with this
  project's "start dumb" bias (Phase 2.3 below). A denied call feeds
  `"Error: denied by user"` back to the model as the tool result rather
  than raising, so the model can react and keep going.

### Phase 2.3 — Memory — done, minimal
- Shipped as planned, "start dumb": `core/memory.py` is a JSONL session
  log on disk (`sessions/<id>.jsonl`, root configurable via
  `BARBAI_SESSIONS_DIR`) plus a rolling window (last 20 messages) bounding
  what's actually replayed to the model each turn — the log itself stays
  complete regardless of the window.
- Modeled on OpenAI's Responses API `previous_response_id` chaining
  (send just the new turn + an id, the server remembers) rather than the
  stateless Chat Completions/Anthropic Messages shape — deliberately
  *not* applied to `/v1/chat/completions` or `/v1/messages`, since an
  external OpenAI/Anthropic-compatible client expects pure passthrough by
  contract. Opt-in via a new `session_id` field on native `/chat` and
  `/agent/chat` only: omit it and behavior is exactly the stateless
  passthrough both endpoints always had.
- Known simplification: only role/content turns are persisted, not raw
  `tool_calls`/`tool_call_id` structure — a turn's tool round-trips are
  already resolved into a final assistant reply within the request that
  produced them, so there's nothing structurally valid to replay next
  time regardless. `session_id` + `stream: true` together isn't supported
  yet on `/chat` (rejected with a 400) — streaming would need to
  accumulate the full reply before it could be persisted, deferred until
  it's actually needed.
- **Context-window overflow — real bug, found in practice, fixed —
  drop-based v1, summarization is the deferred v2.** The rolling window
  above bounds session history by *message count*, not token count, so a
  handful of verbose exchanges could still exceed `n_ctx` and crash with
  a bare `ValueError` surfaced as a raw 500 - hit by an actual user, not
  a hypothetical. Fixed with `core/model_runtime.py`'s `count_tokens()`/
  `fit_to_context()`: drops the oldest messages (after any system
  message, never the newest one) until what's left fits the model's real
  context window, using the model's own tokenizer rather than trusting
  the message-count window alone. Wired into both `core/agent.py`'s
  `run_agent` (every model call) and `api/native.py`'s `/chat`. This is
  the same explicit-first-summarize-later split already made for global
  memory above, applied to a different thing: **v1 (done) drops the
  oldest turns with no trace**; **v2 (deferred, not started) would
  summarize them before dropping** - the same gap between ChatGPT/Claude
  auto-extracting facts (v1-shaped, cheap) versus Claude's own
  conversation-level auto-summarization (v2-shaped, preserves
  continuity). Revisit only once "conversations quietly losing their
  early context" is actually felt as a real problem in practice, not
  preemptively - dropping already stops the crash, which was the
  actually-reported bug.
- **Global memory (cross-conversation, not per-session) — done, explicit-only.**
  Session memory above is per-`session_id` and scoped to one conversation;
  this is the separate "BarbAI remembers things about you across every
  conversation" feature, decided after comparing how ChatGPT and Claude do
  it: ChatGPT auto-extracts facts silently, one global store; Claude
  auto-summarizes whole conversations, scoped per-project, visible and
  editable. BarbAI does neither yet — shipped the cheapest, safest of the
  three options discussed: **explicit-only**, nothing is written unless
  the user directly asks ("remember that...").
  - `core/global_memory.py`: a small JSON fact store
    (`global_memory.json`, path configurable via
    `BARBAI_GLOBAL_MEMORY_PATH`), each fact `{id, text, created_at}`.
    Toggle via `BARBAI_GLOBAL_MEMORY=off` (default on), checked per call,
    not just at startup — matches the "can be active or not" requirement.
  - A new `remember` tool (`core/tools.py`), alongside `read_file` /
    `write_file`. Its description instructs the model to only call it on
    an explicit user request, never inferred on its own — enforced at the
    prompting level, since a tool call can't be structurally prevented.
    It's excluded from `build_tool_defs()` entirely when global memory is
    disabled.
  - `remember` is in `GATED_TOOLS` — same human-approval pause as
    `write_file` (see Phase 2.2 above), since a misphrased "remembered"
    fact would otherwise quietly shape every future conversation with no
    review.
  - Stored facts are rendered into every system prompt
    (`core/persona.build_system_prompt`), positioned after the base
    identity and thinking-mode nudge but before any caller-supplied
    custom prompt (which still wins on conflict, unchanged from before).
  - Deferred: Claude-style automatic end-of-session summarization (reusing
    the Phase 2.3 session log above) is the natural v2 once explicit-only
    is proven useful in practice — not built yet.
- Add a vector store (Chroma or LanceDB) only once the lack of retrieval is
  actually felt — don't build it speculatively.

### Phase 2.4 — Collaboration setup — mostly done
- Git repo initialized, pushed to a private GitHub repo
  (`AnalogicGoose/BarbAI`). The `core/` (provider-agnostic model logic) +
  `api/` (one module per wire format) split from the Mana-style layout
  suggestion is already in place, though there's no separate `plugins/`
  package yet — not needed until a second contributor actually shows up
  and needs to work on an independent piece without colliding.
- Not done: no automated tests exist anywhere in the project yet.

### Deferred
Voice (whisper.cpp STT + a TTS provider), avatar, remote messaging
bridges. Real, validated Mana features — bolt on after Phase 2.0-2.2 are
solid, not before.

## Sources

- [Best Local LLM for Coding in 2026 (Self-Hosted) — Tembo.io](https://www.tembo.io/blog/best-local-llm-for-coding)
- [The Best Local Coding Models for Any Setup — Kilo blog](https://blog.kilo.ai/p/the-best-local-coding-models-for)
- [Run AI Locally: Best LLMs for 8GB/16GB/32GB — Micro Center](https://www.microcenter.com/site/mc-news/article/best-local-llms-8gb-16gb-32gb-memory-guide.aspx)
- [Best Local LLMs for 4GB, 6GB, and 8GB VRAM in 2026 — Mayhem Code](https://www.mayhemcode.com/2026/06/best-local-llms-for-4gb-6gb-and-8gb.html)
- [Best Local LLM Models 2026 — SitePoint](https://www.sitepoint.com/best-local-llm-models-2026/)
- [7 Best Small Language Models Under 10B Parameters in 2026 — Labellerr](https://www.labellerr.com/blog/best-small-language-models-under-10b-parameters/)
- [Small LLMs in 2026: When 7B Beats Last Year's 70B — Amir Teymoori](https://amirteymoori.com/small-llms-7b-on-device-qwen-gemma-efficiency-2026/)
