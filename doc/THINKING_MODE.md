# Thinking / Reasoning Mode — the inference server's role

Reasoning models ("thinking" models) emit an internal chain-of-thought
*before* the user-facing answer. The model weights produce the tokens; the
**inference server** decides whether thinking happens, where the thought ends
and the answer begins, and how each is surfaced to the client. This doc maps
that responsibility, how the ecosystem implements it, what flambeau does today,
and the gaps.

The model is not "in thinking mode" as a weight-level state — thinking is a
**prompt + parsing protocol** the server drives. Two independent knobs:

1. **Whether to think** — controlled by the rendered chat template (a flag),
   not a sampling parameter.
2. **What to show** — the server splits the chain-of-thought out of the
   answer and returns it in a separate field, so a chat UI shows a collapsible
   "thinking…" panel and the actual reply stays clean.

## The server's four jobs

1. **Gate thinking at template-render time.** The chat template carries an
   `enable_thinking` (Jinja) variable. `true` lets the model think; `false`
   typically injects an *empty* `<think></think>` pair into the prompt so the
   model skips straight to the answer (Qwen3's mechanism). Some families also
   honor an in-band **soft switch** — Qwen3 reads `/think` / `/no_think` in the
   latest user/system message and follows the most recent one per turn.

2. **Decode without prematurely stopping inside the thought.** The thought can
   be long. The server must NOT treat the thinking-open marker as a stop, must
   keep the turn-end marker (e.g. `<turn|>`, `<|im_end|>`, `<end_of_turn>`) as
   the real stop, and — when thinking is OFF — must cut cleanly if the model
   *leaks* a reasoning marker anyway.

3. **Parse the reasoning channel out of the output.** This is the
   "reasoning parser." Each family delimits reasoning differently:
   - **DeepSeek-R1 / QwQ / Qwen3-Thinking** — `<think> … </think>` tags.
   - **OpenAI gpt-oss (harmony)** — typed *channels*: `analysis` (raw CoT),
     `commentary` (tool preambles), `final` (the answer). Reasoning is a
     channel, not a tag pair.
   - **gemma (this family in the repo)** — a thought channel delimited by
     `<|channel|>thought … <|eot_thought|>` / `</channel>` (NOT `<think>`).
   The server strips the reasoning span from `content` and returns it
   separately.

4. **Surface it on the API + stream it separately.** The de-facto standard
   (DeepSeek, vLLM, SGLang) is an extra `reasoning_content` field on the chat
   message, alongside `content`. When streaming, reasoning deltas and answer
   deltas are emitted as distinct events so a UI can render the thinking panel
   live, then switch to the answer.

## How the ecosystem implements it

| Engine / API | Gate thinking | Parse reasoning | Surface |
|---|---|---|---|
| **vLLM** | template `enable_thinking` | `--reasoning-parser deepseek_r1\|qwen3\|…` (pluggable `ReasoningParser`) | `message.reasoning_content` (chat endpoint only) |
| **SGLang** | template flag | `--reasoning-parser deepseek-r1\|qwen3\|deepseek-v3`; `/separate_reasoning` post-hoc endpoint | `reasoning_content`, streamed separately |
| **Qwen3** | `enable_thinking=` kwarg + `/think` `/no_think` soft switch; `False` injects empty `<think></think>` | `<think>…</think>` | (server-dependent) |
| **OpenAI gpt-oss / harmony** | `reasoning_effort: low\|medium\|high` | `analysis` / `commentary` / `final` channels | channel-routed; raw CoT exposed for analysis |
| **Anthropic (Claude)** | `thinking:{type:enabled, budget_tokens:N}` (min 1024) | `thinking` content blocks; `redacted_thinking` (encrypted `data`) | `thinking_delta` SSE; `signature` (encrypted, replayed on multi-turn for verification/continuity) |

Cross-cutting lessons the mature engines encode:
- **Reasoning is per-family, so the parser is a plugin**, selected by a flag —
  not hardcoded to one tag pair.
- **`reasoning_content` is the portable surface** (OpenAI-superset that
  DeepSeek popularized; vLLM/SGLang both adopted it).
- **Budget / effort is a first-class control** — Anthropic's `budget_tokens`,
  OpenAI's `reasoning_effort` — because unbounded CoT is the main latency cost.
- **Multi-turn continuity matters** — Anthropic replays signed thinking blocks;
  most OSS stacks instead *drop* prior-turn reasoning from the context.

## Flambeau's current support

Flambeau already implements the core of jobs 1–4 for the **Qwen/DeepSeek
`<think>` convention**:

- **`enable_thinking` request flag** (`api.rs`) — opt-in per request; default
  `None` ⇒ thinking suppressed (renders the template with
  `enable_thinking=false`). `routes.rs` feeds it to the Jinja template.
- **Capability advertised** — `supports_thinking` is detected at boot via
  `tpl_src.contains("enable_thinking")` (`serve_common.rs`) and surfaced on
  `/v1/models` so clients know to expose the flag.
- **Reasoning split** (`routes/finalise.rs`) — when thinking is on, splits
  `<think> … </think>` into `reasoning_content` + the trailing answer; when
  off, truncates at any *leaked* `</think>` / `<end_thought>` / `<end_think>` /
  `</thought>` marker (plus the arch's `chat_stop_markers`).
- **Reasoning-marker stop logic** (`tokenizer.rs`, `decode_loop.rs`) —
  `<think>` / `</think>` are registered as `always_stop_ids` that bypass the
  `MIN_RESPONSE_TOKENS` early-window mask, so a confused-mode `</think>` leak
  cuts the response cleanly.

## Gaps (flambeau)

1. **gemma4 channel reasoning — FIXED.** `finalise` now detects the
   harmony-style channel from the output (`<|channel>thought {cot} <channel|>
   {answer}`, content-based not arch-string) and splits the CoT into
   `reasoning_content` with a clean `content`; the `<think>…</think>` path is
   unchanged for qwen/deepseek (commit 97c32fd). The companion short-answer
   degeneration ("Canberra. thought Canberra…" / off-topic drift) was the
   `MIN_RESPONSE_TOKENS=24` + `STOP_BIAS=3.0` force-window — a Qwen3.6
   immediate-EOS guard that gemma4 doesn't need; both are now arch-aware
   (`min_response_tokens_for` / `stop_bias_for`, gemma → 2 / 0.0; commit
   158cc70). gemma-4-12b chat battery: clean + 8/8. STILL TODO: the harmony
   `analysis`/`final`/`commentary` channel set for gpt-oss; and lift this from
   output-format detection to a proper per-arch `reasoning_markers()` table
   (rule-13) once a third reasoning format lands.
2. **No streaming separation of `reasoning_content`.** The split happens in
   `finalise` (non-stream). Streaming should emit reasoning deltas distinctly.
3. **No thinking budget / effort control.** No `budget_tokens` / max-thinking
   cap or `reasoning_effort` mapping — unbounded CoT is the latency cost.
4. **No soft switch** (`/think` `/no_think`) parsing for Qwen3.

The immediate, in-scope fix is gap 1 for gemma — it both unblocks gemma4
reasoning and stops the short-answer degeneration. It belongs in the server +
quant layers (chat rendering + finalise + stop set), consistent with project
rule 11 (the server renders the template, decodes, emits — it does not run
agentic loops), and rule 13 (per-arch behavior behind a trait/table, not
`if arch == "gemma4"` scattered through shared files).

## Sources

- [vLLM — Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- [SGLang — Reasoning Parser](https://docs.sglang.io/docs/advanced_features/separate_reasoning)
- [Qwen3 chat template / `enable_thinking`](https://huggingface.co/Qwen/Qwen3-8B) · [deep dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [OpenAI Harmony response format (gpt-oss channels)](https://cookbook.openai.com/articles/openai-harmony) · [raw CoT handling](https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot)
- [Anthropic — Building with extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking) · [streaming](https://docs.anthropic.com/en/api/messages-streaming)
