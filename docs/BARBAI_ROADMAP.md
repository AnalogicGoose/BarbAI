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
  and tool access at least as much as on the model itself.

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

### Phase 2.0 — Core loop (single machine, text only)
- FastAPI service exposing one endpoint shape: OpenAI-compatible
  `/v1/chat/completions`, streamed via SSE.
- `llama-cpp-python` loading a GGUF model. This replaces the NumPy engine
  as the actual brain — Phase 1's engine has served its (educational)
  purpose.
- Hardware-detection module (`nvidia-smi` via subprocess, or `pynvml`) that
  picks a model tier per the table in section 1, so the same codebase
  behaves correctly on all three laptops without per-GPU branches.

### Phase 2.1 — Model tiers for the 3 target machines
- Verify actual `nvidia-smi`-reported VRAM on each of the 3050/4070/5070 Ti
  laptops (not the spec-sheet number) and pin down real tier boundaries
  for our own hardware, mirroring section 1's table.
- Pick one General model + one Coding model per profile row, Q4-class
  quantized, sized to leave VRAM headroom for the OS/desktop (per the
  Product Shape section above) — not "the largest that technically fits."

**RTX 4070 (8GB) candidates — current as of a Sept 2026 web pass, not yet
verified against our own hardware or benchmarked head-to-head, so treat as
a shortlist to test rather than a final pick. These sizes assume a mostly-
idle desktop, not concurrent gaming/heavy GPU use — see the free-VRAM-check
note above:**

| Role | Candidates | Notes |
|---|---|---|
| Coding | Qwen2.5-Coder 7B; CodeGemma 7B (backup) | Qwen2.5-Coder was the code-specialized pick in Mana's own `coding` profile too — but section 2 above found *this exact model* unreliable at tool-calling (wraps tool JSON in a markdown fence instead of its own template's tags). Verify tool-calling before committing. |
| General | Qwen3.5 9B (~6.6GB Q4, leaves headroom on 8GB); Llama 3.3 8B; IBM Granite 4.1 8B (strong tool-calling); Mistral Small 3 7B (fastest tokens/sec) | Phi-4-mini also came up specifically for *reliable structured-output/tool-calling at small size*, worth a look if the 7-9B options are unreliable callers. |

3050 (4-6GB) and 5070 Ti (~12GB) candidates: not yet researched — repeat
this pass for those tiers once the 4070 picks are validated end-to-end.

### Phase 2.2 — Tool calling
- Test the chosen model's tool-calling reliability directly (per section
  2's method) before building a loop around it.
- Ship exactly one read-only tool first (file read, scoped to an
  allowlisted root, reject path traversal / absolute paths outside root).
  Hold off on write/shell tools until that loop is proven solid.
- Approval gate modeled on section 3: gate agent-authored artifacts before
  they're trusted, not every routine call.

### Phase 2.3 — Memory
- Start dumb: rolling conversation window + a session log.
- Add a vector store (Chroma or LanceDB) only once the lack of retrieval is
  actually felt — don't build it speculatively.

### Phase 2.4 — Collaboration setup
- This repo isn't a git repository yet. Before a friend joins:
  `git init`, push to a private GitHub repo, and consider a Mana-style
  layout (`core/` API service + `plugins/` for independent features) so
  two people can work on separate pieces without colliding on one file.

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
