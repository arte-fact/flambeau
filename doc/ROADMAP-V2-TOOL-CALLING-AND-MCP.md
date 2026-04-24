# Roadmap — Tool Calling & MCP Support (V2 track)

Sibling of `ROADMAP-V1-QWEN36-GFX906.md`. V1 shipped a single-model, single-arch, max-perf Qwen3.6-35B-A3B slice with an OpenAI-compatible server. V2's tool-calling + MCP track makes that server a first-class agentic endpoint: tools rendered into the Qwen Jinja template, Hermes-style tool calls parsed out (streaming + non-streaming), MCP in both directions.

Two axes of MCP deliberately kept separate:

- **Axis A — MCP client embedded in `flambeau serve`** so the running model can drive external MCP tool servers. Neither `llama.cpp` nor `vLLM` ships this today; it is a flambeau differentiator.
- **Axis B — MCP server exposing flambeau's dev surface** (sweep / pmc / dispatch / inspect / cert-diff / tune-dry). The M-track scaffold at `crates/mcp-server` currently holds only a 10-line doc-comment and `cli mcp` is a `todo!()`.

Intended end state: one flambeau instance (A) can drive another instance (B) via MCP to run kernel-engineering workflows — sweeps, PMC probes, dispatch-row A/B, cert diffing — and the tool results round-trip back to A as committable artefacts (cert JSON, dispatch-row TOML fragments, PMC snapshots). That is the "instance-manages-instance via MCP" loop.

## Current-state inventory (2026-04-24, HEAD `38daa3c`)

**Wire format — `crates/server/src/api.rs`**
- `ChatCompletionRequest`: no `tools`, `tool_choice`, `parallel_tool_calls`.
- `ChatMessage`: role/content strings only; no `tool_call_id`, `tool_calls[]`, no `role="tool"` handling.
- Response: no `tool_calls[]`, no `finish_reason="tool_calls"`.
- SSE delta: `delta.content` only — no `delta.tool_calls[].function.{name,arguments}`.

**Chat template — `crates/quant/src/chat_template.rs:111`**
- `render()` hardcodes `tools => MjValue::from_serialize(Vec::<String>::new())`. The Qwen3.6 Jinja template (7816 B, embedded in GGUF) already knows how to render `tools` — the server just never passes them.

**Decoder / stop-token policy — `routes.rs:379–428`**
- First 24 tokens force stop-token logits to `−∞`, then `−3.0`-nat penalty thereafter. For tool-call turns (which can legitimately be very short JSON) this injects noise before the terminator. Bug for tool calling independent of the wire-format work.
- No grammar / GBNF / JSON-schema-guided sampling. `crates/runtime/src/sampling.rs:74–250` supports only greedy / temperature / top-p.

**Tokenizer — `crates/quant/src/tokenizer.rs:152–199`**
- Auto-registers `<|...|>` bracket-form specials from the vocab scan. Qwen's `<tool_call>` / `</tool_call>` are textual (BPE-decoded) not single tokens — confirm during T1 and add explicit tests.
- Stop IDs: `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`, EOS. No tool-call-aware stop logic.

**MCP scaffold — `crates/mcp-server/src/lib.rs`**
- 10-line doc-comment listing the seven intended tools (`flambeau_sweep`, `flambeau_matrix`, `flambeau_profile`, `flambeau_dispatch_ab`, `flambeau_inspect`, `flambeau_cert_diff`, `flambeau_tune_dry`). Zero implementation.
- CLI: `flambeau mcp --port 9090` dispatches to `todo!()` at `crates/cli/src/main.rs:128`.

**Dev-surface CLI commands the MCP server would wrap**
- Live today: `sweep`, `cert-check`, `pmc-probe`, `pmc-refresh`, `inspect-gguf`.
- Stubbed / not yet real: `matrix`, `tune` (T-track), `dispatch_ab`, `inspect-ptx/hsaco`, `cert_diff`.

**Autotuner (`cli tune`)** — T-track, unstarted (`crates/autotune/src/lib.rs` is an 8-line doc-comment).

## External landscape (validated 2026-04)

- **Qwen3 / Qwen3.5 / Qwen3.6 tool-call wire format** is Hermes-style: `<tool_call>\n{"name": "...", "arguments": "{...json-string...}"}\n</tool_call>`. Qwen3.6 is `qwen35moe` architecture — the V1 target. Qwen3-Coder is a *different* XML format (`<function=name><parameter=k>v</parameter></function>`) with a different parser surface; see "Qwen family coverage" below.
- **Streaming tool-call parsing is a class-wide hard problem in 2026, not a llama.cpp idiosyncrasy**:
  - llama.cpp: #12601 (stream+tools banned), #20198 (`arguments` as object broke openai-python; fixed on master per current `function-calling.md` showing string args), #20809 (b8429 mis-routes non-reasoning Qwen3 as thinking, tool_calls land in `reasoning_content`), #21118 (`</think>` swallowed before `<tool_call>`), #20837 (tool calls emitted inside `<think>`, early stop), #20164 (long-context + many-optional-parameters → tool-call looping).
  - vLLM: #31871 (Feb 2026 — `stream=True --tool-call-parser hermes` returns raw text not parsed `tool_calls`), #19056 (Hermes stream output error on Qwen3 specific tokens), #21544 ("Error trying to handle streaming tool call"), #16880 (xgrammar rejects `minItems` → agent tool-calling non-functional).
  - SGLang: migrating to XGrammar **structural-tag** format (#13026 / #13032) precisely because the string-matching parsers drift. Named parsers: `qwen25` (Hermes-shape) for Qwen3, `qwen3_coder` for Qwen3-Coder.
  - **Implication**: a post-hoc text parser is how everyone shipped v1 and how everyone broke it on streaming. The architecturally correct primitive is **structural-tag constrained decoding** — "any text until `<trigger>`, then JSON-schema body to `</trigger>`, then any text" — where the parser state IS the decoder state. T2 must be factored so it can later become the mask-generator for T5 without re-architecture.
- **llama.cpp's parser shape** (`common/chat.cpp`): per-family named handlers (Hermes/Qwen2.5, Qwen2.5-Coder, Functionary v3.1/v3.2, Mistral Nemo, FireFunction v2, Command R7B, DeepSeek R1) + Generic fallback. Activated by `--jinja [-fa] --chat-template-file <tpl.jinja>`. `arguments` field is a JSON string on current master. **No `--mcp` flag** in llama.cpp; vLLM also doesn't ship an MCP client built into `serve`.
- **Rust MCP SDK**: `rmcp` 0.16.0 (modelcontextprotocol/rust-sdk), macro-based, Tokio-first, MCP protocol 2025-11-25. This is what `crates/mcp-server` should build on.
- **Rust constrained-decoding options for T5**:
  - **llguidance** (Microsoft / guidance-ai) — **Rust-native crate** (`docs.rs/llguidance`), derivre-based regex lexer + Earley parser for CFG. ~50 µs/token on 128K vocab, negligible startup cost. Measured invalid-JSON rate 0.12% vs xgrammar's 2.21% on benchmarks. OpenAI credited it for their structured-outputs work (May 2025). **First-choice for flambeau** because it's native Rust; no C++/Python glue.
  - **xgrammar** — C++ with Rust bindings (`docs.rs/xgrammar`); strong where grammars are reused (caching wins). Supports structural-tag natively, which matches Hermes tool-calls. Known JSON-Schema gaps (e.g. `minItems` — see vLLM #16880).
  - **outlines** — Python-centric; not relevant.

## Qwen family coverage

| Family | Format | Parser name (vLLM / SGLang) | llama.cpp | flambeau-V2 scope |
|---|---|---|---|---|
| Qwen3 / Qwen3.5 / Qwen3.6 | Hermes-JSON `<tool_call>{json}</tool_call>` | `hermes` / `qwen25` | Hermes/Qwen2.5 parser, working post Feb-19 fixes | **In scope** (T1–T5) |
| Qwen3-Coder / Qwen3-Coder-Next | XML `<tool_call><function=f><parameter=k>v</parameter>…</function></tool_call>` | `qwen3_coder` / `qwen3_xml` | **Missing** — issue #15012 open | Trait surface only; concrete parser is V2.x once the Coder family is loaded |

T2's parser is factored as a trait with `HermesJsonParser` as the V2.13 impl. The Coder XML parser drops in as a sibling file when a Coder-family GGUF becomes a supported architecture.

## Practitioner-voice findings (r/LocalLLaMA + community, 2026-02..04)

- **Canonical streaming failure mode**: users report llama.cpp "sends raw XML-formatted tool calls at the end of the stream" — the server buffers the parse and dumps it as a trailing text chunk, which the openai-python SDK can't interpret as `tool_calls`. Design rule for T3: **never buffer to end-of-stream**; the SSE producer must switch to `delta.tool_calls` chunks the moment the parser enters the tool-call body state.
- **`presence_penalty` + `repetition_penalty` are load-bearing for Qwen3.5/3.6 agent loops**. Ollama dropping them was *the* cited cause of "excessively long CoT loops, broken tool call formatting, garbage output" on Qwen3.5 (r/LocalLLaMA 2026-03-03). Flambeau's `sampling.rs` currently ships only greedy / temperature / top-p. Shipping tool calling without them will reproduce the Ollama symptom.
- **Long-context + many-optional-parameters → tool-call looping** (llama.cpp #20164): removing optional parameters unsticks it. Not a bug flambeau can fix — import it as a fixture and document the mitigation in server docs.
- **`enable_thinking=false` is the production-stable agentic mode** per community guides. `chat_template.rs:109` already hardcodes that default — keep it; expose an opt-in in T1 but don't flip the default.
- **Parser choice matters per model family**: SGLang `qwen25` for Qwen3 (Hermes-shape), `qwen3_coder` for Qwen3-Coder. vLLM `hermes` for Qwen3, `qwen3_coder` for Coder. Don't cross-wire them.
- **MoE routing is cited as unpredictable in extended agent sessions**. Not fixable server-side on Qwen3.6-35B-A3B (which IS MoE); surface via telemetry in M2, document as a known property.
- **MCP practitioner norm**: "3–5 MCP servers max; every added server increases context-window use and agent confusion." T1's template rendering should measure tools-overhead token count and warn (not error) past ~10% of context budget.

## Scope decisions

- **V2.x tool calling**: `tools[]`, `tool_choice` (`"auto" | "none" | {"type":"function","function":{"name":...}}`), `parallel_tool_calls`, `finish_reason="tool_calls"`. Qwen3.6 is well-tuned enough that a defensive parser beats shipping a grammar engine on day one.
- **Streaming tool calls are in scope** — this is where llama.cpp and vLLM both shipped broken implementations; the SSE state machine is the work.
- **Both MCP axes in scope, sequenced**: M-track server (B) first (bounded, scaffold exists); MCP client in `serve` (A) second (unlocks "instance-manages-instance").
- **Out of scope**: vision/audio tool results, logprobs, free-form structured-output mode (`response_format={"type":"json_schema",...}`), and the **Qwen3-Coder XML parser implementation itself** (different family, different arch). The *trait surface* for it IS in scope — T2 is factored so the Coder parser drops in later as one file alongside the Hermes impl. Grammar-constrained decoding is **conditionally in scope** as T5: design T2 so llguidance drops in without rewrite, ship T5 only if parity testing forces it.
- **Never load-bearing rule**: `flambeau serve` must not depend on `mcp-server`; `mcp-server` is a dev surface.

## Phasing

### T1 — OpenAI tool-call wire format + template rendering  (V2.12)

- Extend `crates/server/src/api.rs`:
  - `ChatCompletionRequest`: `tools: Option<Vec<ToolDef>>`, `tool_choice: Option<ToolChoice>`, `parallel_tool_calls: Option<bool>`.
  - `ChatMessage`: add `tool_call_id: Option<String>`, `tool_calls: Option<Vec<ToolCall>>`. Accept `role="tool"`.
  - Response: `finish_reason` enum extended with `"tool_calls"`; response `message.tool_calls: Option<Vec<ToolCall>>`.
  - `ToolCall.function.arguments` is `String` (JSON-encoded) on the wire, **not** an object — sidesteps llama.cpp #20198 by construction.
- `crates/quant/src/chat_template.rs::render()`: thread `tools` through to the Jinja context and drop the hardcoded empty vec at line 111. Keep `enable_thinking` as a separate knob, default false.
- Add fixture-based parity cert `certs/chat_template/qwen35moe_tools.json` diffing flambeau's rendered prompt byte-for-byte against `llama.cpp --jinja` on N (messages, tools, tool_choice) fixtures.

Critical files: `crates/server/src/api.rs`, `crates/quant/src/chat_template.rs:111`, `crates/server/src/routes.rs:80+`.

### T2 — Streaming-capable tool-call parser (trait-based, multi-format)  (V2.13)

- New module tree `crates/server/src/tool_call_parser/` with a **trait**, not a single struct:
  ```rust
  trait ToolCallParser {
      fn push(&mut self, chunk: &str) -> Vec<ParserEvent>;
      fn finish(&mut self) -> Vec<ParserEvent>;
  }
  enum ParserEvent {
      TextDelta(String),
      ThinkDelta(String),               // discarded today; reasoning_content in V3
      ToolCallOpen { index: u32, name: String },
      ToolCallArgumentsDelta { index: u32, arguments: String },
      ToolCallClose { index: u32 },
  }
  ```
- V2.13 ships **only `HermesJsonParser`** (Qwen3.5 / Qwen3.6) — the only format flambeau V1 needed.
- States of `HermesJsonParser`: `{Text, MaybeThinkOpen, InThink, MaybeToolOpen, InToolBody, MaybeToolClose}`. Deliberately mirrors an XGrammar-style structural tag (`Outside | Inside{schema}`) so T5 can swap in llguidance-driven token masking without rewriting the parser layer.
- JSON body parsed with `serde_json` at `ToolCallClose`; `arguments` re-serialised as `String` for the wire (guards llama.cpp #20198).
- **Buffer-before-emit discipline**: when `Text` state sees `<`, hold subsequent characters until tool-call-open vs think-open vs literal is decided. Never emit an ambiguous prefix as `TextDelta` and then retract. This is the bug class that produces "raw XML at end of stream" in practitioner reports.
- **Parser selection** at request time: request field `tool_call_format: Option<"hermes" | "qwen3_coder" | "auto">`. `"auto"` inspects GGUF metadata architecture / template at model-load time; default for `qwen35moe` is `hermes`. The field is future-proofing so the Coder parser drops in later.
- Non-streaming path first: accumulate events, emit `tool_calls[]` + `finish_reason="tool_calls"` in the JSON response when ≥1 `ToolCallClose` seen.

Critical files: new `crates/server/src/tool_call_parser/mod.rs` (trait + enum + dispatcher), `crates/server/src/tool_call_parser/hermes.rs` (V2.13 impl), `crates/server/src/routes.rs`. `qwen3_coder.rs` sibling stubbed as `unimplemented!()` with a tracking reference.

### T2.b — Shared failure-fixture corpus  (V2.13)

- Committed set of model-output fixtures at `crates/server/tests/tool_call_fixtures/*.txt` plus expected `(events[], tool_calls[])` JSON:
  - `llamacpp_21118_think_swallow.txt` — `…</think><tool_call>…` with no newline.
  - `llamacpp_20837_tool_inside_think.txt` — tool call *inside* a `<think>` block; surface as text, not tool call, no early-stop.
  - `llamacpp_20164_looping_optional.txt` — multi-turn looping with many optional params (negative test; each attempt parses cleanly, loop detection is outside the parser).
  - `llamacpp_20198_args_object.txt` — the erroneous object-not-string variant; assert we never produce it on the wire.
  - `vllm_31871_stream_raw.txt` — "raw text at end of stream" degenerate case.
  - `reddit_xml_end_of_stream.txt` — community-reported variant of the above.
  - `qwen_parallel_tool_calls.txt` — two `<tool_call>…</tool_call>` blocks in one assistant turn.
  - `qwen_bare_brace_not_tool.txt` — `{` in free text must never be promoted to a tool call.
- One `tool_call_parser_fixtures` test walks the directory; new bugs land as new fixture files, not new assertion code. This is the regression surface each of llama.cpp / vLLM / SGLang rediscovered one at a time.

### T3 — Streaming tool-call SSE deltas  (V2.14)

- Extend the SSE producer (`routes.rs:454–623`) to consume parser events:
  - `TextDelta` → existing `delta.content`.
  - `ToolCallOpen{i}` → `delta.tool_calls = [{index: i, id, type:"function", function:{name}}]`.
  - `ToolCallArgumentsDelta` → `delta.tool_calls = [{index: i, function:{arguments: chunk}}]` (OpenAI contract: arguments is a growing string across chunks).
  - `ToolCallClose{i}` → no-op on wire; bookkeeping only.
  - Final chunk: `finish_reason="tool_calls"` if ≥1 emitted else `"stop"`.
- Handle `parallel_tool_calls=true` with disjoint `index` values.
- Gate: openai-python SDK with `stream=True, tools=[...]`; assert identical tool-call reconstruction vs the non-streaming response on the same seed/prompt.

### T4 — Stop-token & tool-call-aware decoder hygiene  (V2.15)

- Remove the first-24-token stop-token mask when `tools` is non-empty (`routes.rs:379–428`). Keep the `−3.0`-nat soft penalty off too for tool turns — Qwen emits valid JSON + `</tool_call>` + `<|im_end|>` reliably; don't tilt it.
- Do *not* try to terminate on `</tool_call>` text in the decoder — let EOS drive the loop; parser decides when the response is tool-call-shaped.
- Fix the `</think>`-before-`<tool_call>` case at the parser level (T2), not by decoder hacks — llama.cpp's #21118 is exactly the decoder-hack approach that drifts.

### T4.b — Sampler gap: `presence_penalty` + `repetition_penalty`  (ships with T1)

- Extend `crates/runtime/src/sampling.rs` with `presence_penalty`, `repetition_penalty`, `top_k`, `min_p` — all present in the OpenAI `ChatCompletionRequest` and in Qwen's `generation_config.json`.
- Load-bearing for Qwen3.5/3.6 agent loops. Omitting them is cited as the root cause of "long CoT loops, broken tool call formatting, garbage output" that broke Ollama's Qwen3.5 support. Ship T4.b *with* T1, not as an afterthought — tool calling without these will technically work on one-shot prompts and visibly break on multi-turn agentic workloads.
- Wire the request fields through `api.rs::ChatCompletionRequest` → `state::SamplingParams` → the sampler. Defaults per Qwen3.6 model card (`presence_penalty=1.5` for Qwen3.5 per Qwen docs; read per-model `generation_config.json` embedded in GGUF where available).

### T5 — llguidance-backed structural-tag constrained decoding  (conditional)

- First-choice backend: **llguidance** (Rust-native, `llguidance = "0.7+"`). Fallback: `xgrammar` Rust bindings if llguidance Qwen3.6-vocab integration hits a wall.
- Mask generator around the T2 state machine: while `Outside`, no masking; on `MaybeToolOpen` confirmed, switch to llguidance with the JSON-schema derived from the active `tool` definition(s); on `ToolCallClose`, back to free decoding.
- **Decision gate**: if T1–T3 parity cert shows invalid-JSON rate > 1% on realistic tool args, T5 moves to V2; otherwise stays V3+. Industry data says llguidance gives 0.12% invalid-JSON vs xgrammar's 2.21% — upper-bound expectation.
- Even deferred, T2 must be *shaped* like a structural-tag dispatcher so T5 is additive, not a rewrite.

### M1 — M-track MCP server (dev surface)  (V2.16)

- `crates/mcp-server` takes `rmcp = "0.16"` as a workspace dep. Stdio transport first (Claude Code default); HTTP transport second (port 9090, for cross-instance management in M3).
- Tools (each a thin wrapper over an existing `flambeau` CLI subcommand, returning the JSON artefact that subcommand already produces):
  - `flambeau_sweep {arch, op?, dtype?, impl_id?}` → wraps `sweep`, returns cert-JSON diff.
  - `flambeau_cert_check {arch, backend}` → wraps `cert-check`, returns `{rows, failures[]}`.
  - `flambeau_pmc_probe {kernel, m, k, n}` → wraps `pmc-probe`.
  - `flambeau_inspect_gguf {path}` → wraps `inspect-gguf`.
  - `flambeau_dispatch_read {backend, arch}` → reads TOML, returns rows as JSON.
  - `flambeau_dispatch_ab {impl_a, impl_b, shapes}` → needs new `bench ab`; scope as M1.b if it looks bigger than it sounds.
  - `flambeau_cert_diff {commit_a, commit_b}` → `git show`-driven JSON diff.
  - `flambeau_tune_dry` → stubbed with a clear "T-track not shipped" error until T-track lands; shipping the tool surface early lets callers code against it.
  - `flambeau_matrix` → same treatment; real once `matrix` subcommand is real.
- CLI: `flambeau mcp --port 9090 [--stdio]` replaces the `todo!()` at `crates/cli/src/main.rs:128`.
- **CLAUDE.md rule carried**: every tool returns a JSON artefact that is committable (cert row, dispatch-row TOML fragment, PMC snapshot). MCP responses embed the artefact *and* the canonical path they'd be written to.

Critical files: `crates/mcp-server/src/lib.rs` (grow from stub), new `crates/mcp-server/src/tools/*.rs`, `crates/cli/src/main.rs:128`, workspace `Cargo.toml`.

### M2 — MCP client inside `flambeau serve` (agent loop)  (V2.17)

- `flambeau serve --mcp <url>...` connects to 0-N external MCP servers at startup, enumerates their tools, exposes them to the model via the T1 `tools[]` path.
- When the model emits a tool call, the server bridges it to MCP `tools/call` JSON-RPC on the right server, appends the response as `role="tool"`, and re-prompts.
- Default behaviour: agent loop bounded by `max_tool_iterations` (10). `stream=true` surfaces intermediate tool calls; non-stream returns only the final answer unless `expose_tool_calls=true`.
- **Context-budgeting** (per "3–5 MCP servers max" norm): at startup, compute token cost of the rendered `tools[]` list. Warn (not error) if tools overhead > 10% of model context, naming the largest schemas.
- **Agent-loop telemetry** (per MoE-instability risk): tracing spans per iteration capturing `{iteration_index, tool_name, latency_ms, retry_count, cumulative_tokens}`; read-only `/v1/agent/stats` introspection route. Lets users notice when Qwen3.6's MoE routing starts oscillating in a long session.
- This is the "instance-A drives instance-B" primitive: A runs `serve --mcp http://B:9090`, prompt A "sweep my Q6_K indexed-MoE tile8 kernel and tell me if the cert is green," A's model calls `flambeau_sweep` on B via MCP.
- **Differentiator note**: the same scenario run against llama.cpp's server or vLLM has no path — neither ships an MCP client in their `serve`. Worth calling out in release notes; worth also not over-complicating M2's loop so the differentiator isn't "yes but only with these 4 caveats".

Critical files: new `crates/server/src/mcp_client.rs`, `crates/server/src/routes.rs`.

### M3 — Inter-instance cert/PMC round-trip  (V2.18)

- MCP tool responses carry `{artefact, canonical_path, suggested_commit_message}`.
- Agent on instance-A can then: sweep on B → receive cert JSON → propose a dispatch row → write it to instance-A's working tree → open a PR. Closes the loop CLAUDE.md implies with "every finding round-trips into a committable artefact".
- Mostly glue plus stricter contracts on M1 tool return types; no new kernels or wire format.

## Verification

End-to-end smoke on Qwen3.6-35B-A3B, 4×MI50 rig:

1. `curl -s -X POST http://localhost:8080/v1/chat/completions -d '{..., "tools":[...], "tool_choice":"auto"}'` → response has `tool_calls[]`, `arguments` is a JSON *string*, `finish_reason="tool_calls"`. (Guards llama.cpp #20198.)
2. openai-python `client.chat.completions.create(..., stream=True, tools=[...])` → `tool_calls` reconstruct identically to non-stream. (Guards llama.cpp #12601 / vLLM #31871.)
3. Multi-turn with `<think>` + tool call in the same assistant turn → `</think>` preserved in surface text, `tool_calls[]` still extracted cleanly. (Guards llama.cpp #21118 / #20837.)
4. `certs/chat_template/qwen35moe_tools.json` — byte-exact prompt match to `llama.cpp --jinja` on 10 fixtures.
5. `flambeau mcp --port 9090` + `mcp-cli ls-tools` → 9 tools listed; `mcp-cli call flambeau_sweep --arch gfx906 --impl mmq_q4_1_dp4a_ds4` returns the same cert JSON as the CLI produces.
6. **Inter-instance**: instance B at `--port 9090 --mcp-server`, instance A `serve --mcp http://B:9090`, prompt A: "sweep my Q6_K indexed-MoE tile8 kernel and tell me if the cert is green." Expect A's model to call `flambeau_sweep` on B, receive the cert, summarise.
7. **Parity**: same agent task run against `llama.cpp --jinja` or `vllm serve` — no MCP-in-serve path exists; documents the differentiator.

## Critical files (quick reference)

- `crates/server/src/api.rs` — wire types (T1).
- `crates/server/src/routes.rs:80+` — chat handler (T1–T3).
- `crates/server/src/routes.rs:379–428` — stop-token mask to soften for tool turns (T4).
- `crates/server/src/routes.rs:454–623` — SSE streaming path (T3).
- `crates/quant/src/chat_template.rs:111` — hardcoded empty `tools` to remove (T1).
- `crates/quant/src/tokenizer.rs:152–199` — verify `<tool_call>` handling; likely no change.
- new `crates/server/src/tool_call_parser/{mod,hermes,qwen3_coder}.rs` (T2).
- new `crates/server/src/mcp_client.rs` (M2).
- `crates/mcp-server/src/lib.rs` — grow from 10-line stub (M1).
- `crates/cli/src/main.rs:128` — `todo!()` to replace (M1).
- `crates/runtime/src/sampling.rs:74–250` — presence/repetition penalties + top-k + min-p (T4.b).
- `Cargo.toml` workspace — add `rmcp = "0.16"` (M1), optionally `llguidance = "0.7+"` (T5).

## Sequencing summary

| Phase | Version | Ships with | Description |
|---|---|---|---|
| T1 | V2.12 | T4.b | OpenAI wire fields + `tools` rendered into Jinja |
| T2 | V2.13 | T2.b | Trait-based parser, `HermesJsonParser` impl, fixture corpus |
| T3 | V2.14 | — | Streaming SSE deltas for `tool_calls` |
| T4 | V2.15 | — | Stop-token mask relaxed for tool turns |
| T4.b | V2.12 | (with T1) | Sampler fills (`presence_penalty`, etc.) |
| T5 | V3+ conditional | — | llguidance mask generator; gated on T1–T3 invalid-JSON rate |
| M1 | V2.16 | — | `crates/mcp-server` on `rmcp`, 9 tools, `cli mcp` |
| M2 | V2.17 | — | `serve --mcp <url>` agent loop |
| M3 | V2.18 | — | Inter-instance artefact round-trip |

## Risks / non-obvious tradeoffs

- **Parser brittleness on Qwen3.6 is class-wide, not a llama.cpp idiosyncrasy** — vLLM (#31871, #19056, #21544) and llama.cpp (#20198, #21118, #20837, #20164) each rediscovered the streaming-parser failure modes. Import their failing-case fixtures into T2.b from day one. Null-parse → surface as regular text (warn), never drop the turn.
- **Streaming + tools** is specifically where llama.cpp and vLLM both shipped v1 broken. The SSE state machine in T3 is the hard part. The specific anti-pattern to avoid is "buffer tool-call text, dump at end of stream" — that is what produces the "raw XML at end of stream" user complaints.
- **Sampler gap is silent until agent loops start** — without T4.b's `presence_penalty`/`repetition_penalty`, tool calling will technically work on one-shot prompts and visibly break on multi-turn agentic workloads. Ship T4.b with T1.
- **MoE routing instability on long agent sessions** (community-reported) is outside our fix surface on Qwen3.6-35B-A3B; surface via M2 telemetry, document as a known property.
- **`rmcp` version churn** — SDK 0.16, MCP protocol 2025-11-25; pin exactly, rebuild on each bump; don't let server code depend on MCP internals beyond rmcp's abstractions.
- **Axis A vs Axis B confusion** — keep the two MCP directions as two crates / two subcommand paths. `flambeau mcp` = Axis B server; `flambeau serve --mcp <url>` = Axis A client. Never one flag doing both.
- **Grammar deferral is a measured bet** — Qwen3.6 is well-behaved enough that a defensive parser beats shipping a grammar engine on day one; llguidance's 0.12% invalid-JSON is strong but not free in complexity. Gate T5 on measured parity numbers, don't ship speculatively. Keep T2 shaped like a structural-tag dispatcher so T5 is additive.
- **MCP-in-server is an industry differentiator** — neither llama.cpp nor vLLM ship it. Worth emphasising in release notes; worth also not over-complicating M2's loop so the differentiator isn't "yes but only with these 4 caveats".
