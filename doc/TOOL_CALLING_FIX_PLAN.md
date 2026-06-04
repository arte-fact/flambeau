# Tool-Calling Fix Plan — gemma4 + qwen3.6

Live-tested on `main @ 3dcae50` (post-Q4c). Both gemma4 and qwen3.6
families produce broken tool-call output today. This plan captures
what upstream stacks have already learned, what the current Flambeau
parser surface looks like, and the slice order that should land the
fix without rewriting kernels.

## Scope

Two model arches, server-side only — no kernel work.
1. **gemma4** — `gemma-4-26B-A4B-it` MoE + `gemma-4-31B-it` dense.
2. **qwen35moe / qwen35** — Qwen3.6-27B dense + Qwen3.6-35B-A3B MoE.

Out of scope for this branch:
- New kernels.
- Streaming tool-call delta plumbing (separate slice, file under
  S-stream if needed).
- Anthropic `/v1/messages` tool-result round-trip beyond what
  `/v1/chat/completions` already does.

## What we observed in the live test

From `bash /tmp/long_test.sh ... --ctx-cap 4096`:
- **qwen3.6-27B-Q4_0** (dense): coherent prose, but no tool calls
  emitted during agentic flow (gap visible only with a real
  tools array; the plain-prose live test didn't exercise it).
- **gemma-4-26B-A4B-Q8_0**: `<channel|>thought\n` marker bleeds into
  visible output. This is the **reasoning-channel leak** that asf0's
  template fix targets (see *Upstream findings* below).
- **gemma-4-31B-it-Q4_0**: same channel-leak shape (`<channel|>thought`).

These match the known upstream issues exactly — the bug isn't in our
inference path, it's in the chat-template + parser surface.

## Upstream findings (source-of-truth)

### gemma4
- **Native format** (Google docs):
  `<|tool_call>call:NAME{k:<|"|>v<|"|>,...}<tool_call|>`. Six new
  special tokens: `<|tool>`, `<|tool_call>`, `<|tool_result>` + closers.
  String values quoted with `<|"|>...<|"|>`.
- **Reasoning leak** (asf0/gemma4_jinja): the stock HF template replays
  `message.reasoning` / `message.reasoning_content` and forces an empty
  `<|channel>thought ... <channel|>` block on assistant turns —
  llama.cpp/OpenWebUI/etc. surface this as literal text. Fix: strip the
  reasoning replay + the forced-empty thought block; keep the
  tool-call/tool-response sections intact.
- **llama.cpp server-side parser**: PR #21326 added a PEG grammar
  parser for the `<|tool_call>...<tool_call|>` token pair; activated by
  `--jinja` on `llama-server`. No separate `--tool-call-parser` flag.
- **Common foot-gun** (llama-cpp-python #2227): tool-call tokens
  emitted as raw text in `content` instead of being lifted into
  OpenAI-shape `tool_calls[]`. That's the bug we already half-have:
  `Gemma4ToolCallParser::new()` exists in
  `crates/server/src/tool_call_parser/gemma4.rs` (823 lines) but
  evidently doesn't match what the real model emits.
- **Caveat to avoid** (llama.cpp #20198): emit `arguments` as a JSON
  **string**, not a JSON object. OpenAI compat breaks otherwise.

### qwen3.6
- **Three coexisting formats** in the wild:
  - **Hermes JSON** — `<tool_call>\n{"name":...,"arguments":...}\n</tool_call>`.
    Default expected by most stacks; default in the official
    `qwen35moe` HF templates.
  - **Qwen3-Coder XML** —
    `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.
    Used by Qwen3-Coder and by the Unsloth `UD-Q*_K_XL` GGUF builds
    even when `general.architecture == qwen35moe`.
  - **qwen3_xml** parser (vLLM) — newer, handles the Coder shape +
    Qwen3.6 quirks.
- **Symptoms across stacks** (HF discussion #13, abysslover repo):
  - Random `<|im_sep_user|>` injection mid-response.
  - Empty `tool_calls: []` while the prose says "I'll call X".
  - Issues persist across qwen-code, claude-code, Ollama, FP8 quant.
  - **Qwen3.5 family is fine**; only Qwen3.6 (27B + 35B-A3B) regresses.
- **vLLM stream bugs** (#19056, #21544, #31871): hermes parser returns
  raw text in streaming mode on Qwen3.
- **vLLM reasoning conflict** (#19513): enabling reasoning breaks
  tool-call parsing.
- **NVIDIA forum fix**: `--tool-call-parser qwen3_xml` +
  `qwen3.5-enhanced.jinja` template.

## Current Flambeau surface

```
crates/server/src/tool_call_parser/
├── mod.rs             — ToolCallFormat enum, dispatcher, auto-detect
├── gemma4.rs          — 823 LOC, Gemma4ToolCallParser
├── hermes.rs          —  61 LOC, HermesJsonParser (thin)
└── qwen3_coder.rs     — 920 LOC, QwenCoderXmlParser
```

- `ToolCallFormat { Hermes, QwenCoder, Gemma4 }`.
- `detect_format_from_template(&str)` walks GGUF chat-template source
  for `<|tool_call>` / `<|"|>` (→ Gemma4), `<function=` /
  `<parameter=` (→ QwenCoder), else Hermes.
- `dispatcher(request_override, server_default)` returns a
  `Box<dyn ToolCallParser>`.
- Per-request `tool_call_format: "auto" | "hermes" | "qwen3_coder" |
  "gemma4"` honoured first; falls back to server default; server
  default is `detect_format_from_template` over the loaded GGUF.
- Cert harness already exists for qwen35moe under
  `certs/chat_template/qwen35moe_tools/` (regenerate.sh + fixtures).
- Three quant-side smoke tests
  (`crates/quant/tests/chat_template_{smoke,gemma4_smoke,parity}.rs`)
  — these cover *rendering*, not *parsing*.

So we have *structure* but not *correctness*. The parsers compile and
have unit tests; what's missing is end-to-end coverage against the
exact byte sequences the real models emit on the real rig.

## Plan

Five slices. Each ends with a green parity test against a captured
real-model fixture and a live `/v1/chat/completions` round-trip on
pp2tp2 / hip:0,2,1,3 / `--ctx-cap 4096`.

### T1 — Capture real fixtures (no code change) — COMPLETE

Harness at `scripts/tool_test/{run.py, scenarios.py, assertions.py}`;
24 fixtures (4 models × S1–S6) under `scripts/tool_test/fixtures/`.

**Per-model results, S1–S6:**

| Model                         | S1   | S2   | S3   | S4   | S5   | S6   |
|-------------------------------|------|------|------|------|------|------|
| qwen3.6-27b-q4_0              | pass | pass | pass | pass | FAIL | pass |
| qwen3.6-35b-a3b-q4_0          | pass | pass | pass | pass | FAIL | pass |
| gemma4-26b-a4b-q8_0 (MoE)     | pass | pass | FAIL | pass | FAIL | pass |
| gemma4-31b-q4_0 (dense)       | FAIL | FAIL | skip | FAIL | FAIL | pass |

**Diagnoses (sorted by impact):**

1. **gemma4-31B dense — wrong tool-call format**. The model emits
   `<|call:NAME{location: 'Paris, France'}>` instead of the documented
   `<|tool_call>call:NAME{location:<|"|>Paris<|"|>}<tool_call|>` —
   missing the `<|tool_call>` open token, missing the `<|"|>` string
   quote wrapper, using Python-style single quotes. S1 fixture shows
   the model also auto-hallucinates the tool *response* and the final
   answer in one go ("…<|response:...> The current weather in Paris is
   15°C and cloudy."), so the model never stops on `<tool_call|>` to
   let the server call the tool. The existing `Gemma4ToolCallParser`
   correctly does not lift this — it doesn't match the spec it was
   built for. Either the model is undertrained on tool calling, or our
   chat template is rendering the tools section in a way the model
   doesn't recognize. Likely both — T2 addresses the template,
   T3 must extend the parser to recognize the `<|call:NAME{...}>`
   shape if the template fix doesn't make 31B switch to the documented
   format. Compare T3 work against gemma4-26B-A4B fixture (which DOES
   produce native format — parser path is OK there).

2. **gemma4-26B-A4B MoE — channel-leak in non-tool-call turns**
   (PARTIALLY FIXED in T2). The chat template ends the prompt with
   `<|channel>thought\n<channel|>` (a closed empty thought block) when
   `add_generation_prompt=true` and reasoning is disabled. The model
   often *echoes* `thought\n<channel|>` (or just `<channel|>`) as the
   first decoded bytes, leaking `thought\n` into `content`. T2 ships
   `Gemma4ToolCallParser::with_pending_channel_close()` plus a
   `prompt_ends_with_channel_close()` detector; the dispatcher
   (`dispatcher_with_prompt`) inspects the rendered prompt and
   primes the parser to drop the leading echo. Net on gemma4-26B:
   - S1, S4, S5, S6: PASS (echo cleanly absorbed; tool calls land
     intact; plain chat free of channel markers).
   - S2: still fails sometimes — model decides not to call any tool
     for the "Find me the latest news" prompt and repeats the user
     query back verbatim. Not a parser bug; it's a chat-template
     issue (the tools section may be confusing the model). Defer to
     a follow-up template slice.
   - S3 (round-trip): still fails on a finish-reason=length cut-off
     because the model produces ~200 tokens of inline narrative
     (`thought\nThe user is asking...\nNow I should formulate a
     response...\nResponse: "The current weather in Paris..."`)
     before reaching the final answer, exceeding max_tokens=128.
     This is gemma4's *inline reasoning* pattern — distinct from the
     `<|channel>`-marker leak — and isn't reachable from the echo
     guard. Same follow-up template work as S2.

3. **All 4 models — `tool_choice="none"` was ignored** (FIXED in T2.5).
   `req.tool_choice` was parsed but never read in either
   `routes/chat.rs` or `routes/messages.rs` — the server always
   rendered the tools section regardless of the client's request and
   lifted `tool_calls[0]` if the model emitted anything. New helpers
   `ToolChoice::forbids_tools()` and
   `AnthropicToolChoice::forbids_tools()` recognize `"none"` (case-
   insensitive on OpenAI, exact tagged-variant on Anthropic); both
   routes now set `merged_tools = None` when forbidden so the chat
   template doesn't render the tools section, plus the OpenAI path
   blanks `final_tool_calls` post-parse as defence-in-depth. S5
   turns green for all 4 models; S1–S4/S6 unaffected.

4. **qwen3.6 family is fine on the V1 rig**. Both 27B dense and 35B
   MoE pass S1, S2, S3 (round-trip), S4 (parallel — actually emits
   both calls), and S6. The upstream HF/vLLM regression chatter
   doesn't reproduce here — likely because we're on V1 of the
   official Qwen3.6 GGUFs and the model+template combination
   happens to work through our Hermes parser path. T4 is downgraded
   to "verify the Unsloth UD-Q*_K_XL GGUF builds also pass on the
   same harness" — if they do, T4 is no-op; if not, ship the
   `qwen3.5-enhanced.jinja` template variant per the NVIDIA forum.

**Net effect on slice ordering:**

- T2 (gemma4 template strip) and T3 (gemma4 parser extension for
  the `<|call:NAME{...}>` shape) become co-dependent: ship T2 first,
  re-capture gemma4-31B fixtures, then scope T3 against whatever
  format the 31B model emits POST-template-fix.
- T2.5 added for `tool_choice="none"` enforcement.
- T4 demoted from "implement qwen3_xml parser" to "verify Unsloth
  build passes the same harness" — the qwen3.6 base GGUFs already
  work.
- T5 unchanged: full 4-model × 6-scenario green gate before merge.

**Re-baseline cadence**: after each fix slice lands, re-run
`scripts/tool_test/run.py --all --capture` to overwrite fixtures.
The diff between captures is the gate evidence.

### T2 — Gemma4 channel-leak template fix

Goal: stop `<|channel>thought\n` from appearing in visible content.

This is purely a chat-template rendering bug — visible in the live
test today. Two paths:

**T2a — Patch the GGUF-embedded template at render time.**
Add a `crates/quant/src/chat_template/gemma4_strip_thought.rs`
pre-processor that runs on the template source before minijinja
renders it. Strips two patterns:
1. The reasoning-replay branch
   (`{% if message.reasoning %}...{% endif %}` and the
   `message.reasoning_content` sibling).
2. The forced-empty `<|channel>thought ... <channel|>` emission in the
   assistant-generation prompt.

Keep all tool-call / tool-response branches verbatim. Mirror
asf0/gemma4_jinja precisely — that's the proven-clean shape.

**T2b — Ship the patched template under
`certs/chat_template/gemma4_thought_strip/`** with a `regenerate.sh`
and a side-by-side diff doc, same shape as `qwen35moe_tools/`.

**Test**:
- Quant-side unit test: render a gemma4 assistant turn through the
  stripped template and assert no `<|channel>thought` substring in the
  output.
- Live test: re-run the `long_test.sh` prompt and confirm the
  technical-analysis output no longer begins with `vie vie vie ...`
  noise + `<channel|>thought\n`.

**No kernel work.**

### T3 — Gemma4 parser fixture parity

Goal: lift the existing `Gemma4ToolCallParser` (823 LOC) to
bit-equal correctness against the captured T1 fixture.

Inspect what `gemma4.rs` currently parses vs. what the live capture
shows. Hypothesis (from llama-cpp-python #2227): the parser is
matching a documented spec but the model emits a slightly different
byte stream (probably `<|"|>` quoting vs raw quote handling). The
fixture-driven test will surface the exact divergence.

**Sub-slices**:
- T3a: build a `cargo run -p bench -- tool-call-cert --arch gemma4`
  that replays the fixture through the parser and emits a structured
  diff (`expected_tool_calls` vs `got_tool_calls`).
- T3b: fix whatever the diff surfaces (likely: `<|"|>` string-quote
  state machine, `,` separator inside `{...}`, multi-arg ordering,
  or the closing `<tool_call|>` boundary scan).
- T3c: lock the parity test in
  `crates/server/src/tool_call_parser/gemma4.rs` `#[cfg(test)]` so
  future template changes can't silently regress.

**Hard rule from llama.cpp #20198**: emit `arguments` as a JSON string,
not a JSON object — keep our existing serialisation if it already
matches OpenAI; verify in the parity diff.

### T4 — Qwen3.6 parser regression hunt

Goal: tool calls actually get emitted on Qwen3.6-27B + Qwen3.6-35B-A3B.

Qwen3.5 worked, Qwen3.6 doesn't, framework-wide. The HF discussion
narrows the bug to:
- Random `<|im_sep_user|>` injection during decode.
- Empty `tool_calls: []` while prose claims a call.

Both look like a **stop-token / chat-template mismatch** in the
generation-prompt path, not the parser per se. Inspect the GGUF chat
template diff between Qwen3.5 and Qwen3.6 (T1 captures both); the fix
candidates are:
- Adopt `qwen3.5-enhanced.jinja` semantics (per NVIDIA forum).
- Ensure `<|im_sep_user|>` is in the EOS-marker set so it stops decode
  cleanly instead of getting emitted as a content token.
- Confirm `detect_format_from_template` lands on the right format for
  the Unsloth `UD-Q*_K_XL` builds (already-documented Coder-XML
  surprise in `mod.rs:208`).

**Sub-slices**:
- T4a: render Qwen3.6 template → confirm assistant-generation prompt
  matches what the official Qwen reference produces. Patch any
  divergence.
- T4b: extend the EOS-marker set per arch (qwen35 / qwen35moe) to
  include `<|im_sep_user|>` and any sibling separators that should
  stop decode but don't today.
- T4c: parity test via `tool-call-cert --arch qwen35moe` against the
  T1 fixture; expect `tool_calls[]` populated and the prose ending
  cleanly before any separator token.

### T5 — End-to-end live gate

Goal: all four models pass an agentic-shape live test.

Extend `/tmp/long_test.sh` (or a new `/tmp/tool_test.sh`) with:
1. A single OpenAI-shaped `tools` array.
2. A prompt that demands a call.
3. Assertion: `response.choices[0].message.tool_calls` is a non-empty
   list, each item has `function.name` ∈ declared tools and
   `function.arguments` is parseable JSON.

Run on all 4 models, pp2tp2 / hip:0,2,1,3 / `--ctx-cap 4096`.

This is the green gate before merge to main.

## Architectural-rule check (project root CLAUDE.md)

- **Rule 11**: server stays a JSON HTTP API, no upstream-MCP loop.
  The parser is purely "decode → emit tool_calls → return"; no change.
- **Rule 12**: no `Hip*`/`Gemma4*`-prefixed types on shared server
  surface. The existing `Gemma4ToolCallParser` already lives in
  `tool_call_parser/gemma4.rs` behind the `ToolCallParser` trait — fine.
- **Rule 13**: arch-specific glue stays out of shared handlers; the
  dispatcher already polymorphises through `Box<dyn ToolCallParser>`.
- **Rule 10** (no narrative comments): the per-slice work must not
  carry `// T2a` / `// asf0 fix —` style markers in the code. Slice
  numbering lives in commit messages, not source.

## Out of scope / explicitly NOT in this plan

- Streaming SSE delta emission of `tool_calls` (vLLM #31871 + #21544
  shape). Open as a follow-up `TOOL_CALLING_STREAMING_PLAN.md` if T1–T5
  surface that the non-streaming path is the main user pain.
- A new "qwen3_xml" format on top of the existing
  `QwenCoder` variant. Re-evaluate after T4 — Qwen3.6 may simply work
  via the existing Hermes/QwenCoder dispatcher once the template +
  stop-tokens are right.
- Per-model OpenAI `parallel_tool_calls` semantics. The flag is
  already plumbed; the parsers either support multiple calls per
  turn or they don't, and that's a fixture-driven judgement.

## References

- [Function calling with Gemma 4 — Google AI](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
- [Gemma 4 Prompt Formatting — Google AI](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [llama.cpp PR #21326 — Gemma 4 server PEG tool-call parser](https://github.com/ggml-org/llama.cpp/pull/21326)
- [asf0/gemma4_jinja — channel-leak fix template](https://github.com/asf0/gemma4_jinja)
- [Daniel Farina — llama.cpp + Gemma 4 26B working command](https://gist.github.com/daniel-farina/87dc1c394b94e45bb700d27e9ea03193)
- [llama-cpp-python #2227 — Gemma 4 tool calls as raw tokens](https://github.com/abetlen/llama-cpp-python/issues/2227)
- [vLLM Gemma 4 Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [HF Qwen3.6-27B discussion #13 — tool calling broken](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/13)
- [abysslover/qwen36_tool_calling_failure](https://github.com/abysslover/qwen36_tool_calling_failure)
- [NVIDIA forum — Qwen3.5 tool calling fix](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451)
- [vLLM Tool Calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM #19056 — Hermes stream output on Qwen3](https://github.com/vllm-project/vllm/issues/19056)
- [vLLM #19513 — Qwen3 reasoning breaks tool-call parsing](https://github.com/vllm-project/vllm/issues/19513)
- [vLLM #31871 — Hermes streaming returns raw text](https://github.com/vllm-project/vllm/issues/31871)
- [llama.cpp #20198 — arguments as object breaks OpenAI](https://github.com/ggml-org/llama.cpp/issues/20198)
- [llama.cpp #20164 — multi-optional-param failures on Qwen](https://github.com/ggml-org/llama.cpp/issues/20164)
- [llama.cpp #20837 — Qwen3.5 XML inside thinking block](https://github.com/ggml-org/llama.cpp/issues/20837)
