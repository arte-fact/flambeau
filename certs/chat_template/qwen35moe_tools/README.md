# Chat-template parity cert — `qwen35moe` with tools

Ships the V2.12 (ROADMAP-V2-TOOL-CALLING-AND-MCP §T1.3) parity gate: for a
fixed set of `(messages, tools, add_generation_prompt)` inputs, the prompt
string flambeau's Jinja render produces must be byte-identical to
`llama.cpp`'s `--jinja` render on the same template extracted from the
same GGUF.

Why byte-identical matters: Qwen3.6's Hermes-style tool-calling is
trained on a specific prompt shape. Any whitespace drift between our
render and llama.cpp's — the reference oracle — means our `tool_calls`
field arrives at the model differently than during training, and the
downstream parser (T2) inherits the drift. Catch it here, in one place,
once.

## Layout

- `fixtures/*.json` — inputs in `llama.cpp`'s `test-chat-template` format:
  `{messages, tools?, bos_token, eos_token, add_generation_prompt}`.
  `tool_choice` is *not* rendered by the Jinja template on any model we
  ship (it affects grammar/sampling, not the prompt), so it's not
  represented in these fixtures.
- `expected/*.txt` — reference renders produced by
  `llama.cpp/build-mi50/bin/test-chat-template`. Regenerate via
  `regenerate.sh` (not committed as golden — they're derived from the
  GGUF's embedded template + llama.cpp at a pinned commit). The parity
  test either compares against these when present, or skips with a clear
  message when they aren't.

## Running the parity test

```bash
# 1. Point at a Qwen3.6 GGUF (must contain tokenizer.chat_template).
export FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
# 2. Point at llama.cpp's test-chat-template binary.
export FLAMBEAU_LLAMACPP_TEST_CHAT_TEMPLATE=/artefact/llama.cpp/build-mi50/bin/test-chat-template
# ROCm libs:
source /artefact/flambeau/.env
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
# 3. Run.
cargo test -p flambeau-quant --test chat_template_parity -- --nocapture
```

With neither env var set, the test skips with an explanatory message.

## Regenerating the `expected/` references

```bash
certs/chat_template/qwen35moe_tools/regenerate.sh \
    /artefact/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
    /artefact/llama.cpp/build-mi50/bin/test-chat-template
```

Commit the resulting `expected/*.txt` if you want the parity test to run
without llama.cpp present (golden-file mode). Otherwise keep them gitignored
and re-derive on demand.

## Finding from T1.3 implementation (2026-04-24)

The Qwen3.6-35B-A3B-UD-Q8_K_XL GGUF's embedded `tokenizer.chat_template`
teaches the model to emit tool calls in **Qwen3-Coder XML format**
(`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
not the Hermes-JSON format the V2 roadmap assumed.

This is an Unsloth "unified" template — it matches the Qwen3-Coder family
on the wire. On this rig, the V1 parser landing path in T2 should
prioritise `QwenCoderXmlParser` or land both parsers together (the
trait surface in T2.1 already anticipates this).

The parity cert itself is format-agnostic: it validates that flambeau's
Jinja render matches llama.cpp's byte-for-byte, whatever the template
teaches. Parser work is downstream.

Validations that had to be shipped to hit byte-parity:

1. `minijinja` needs the `json` feature (for the `tojson` filter the
   template uses on line 63) AND the `preserve_order` feature (so object
   field order survives round-tripping through the Value layer).
2. `serde_json`'s `preserve_order` feature (workspace-level) — required
   so `Vec<serde_json::Value>` tool inputs keep author-supplied field
   order rather than alphabetising.
3. A custom `tojson` filter with Python-style (`, ` / `: `) separators:
   llama.cpp's `nlohmann::json::dump` emits those by default, minijinja's
   stock `tojson` emits compact no-space JSON.
4. Arguments-string → object normalisation on `tool_calls[].function.arguments`
   before handing messages to the template (llama.cpp's
   `common_chat_msgs_parse_oaicompat` does the same).
5. `enable_thinking` left unbound (not `false`) when matching
   llama.cpp's `test-chat-template` default — the new
   `render_with_tools(…, enable_thinking: Option<bool>)` parameter
   exposes this choice.

## Fixtures

| # | Name | Exercises |
|---|------|-----------|
| 01 | no_tools_baseline | V1.8 regression — empty-tools path must match pre-V2.12 render byte-for-byte |
| 02 | simple_user_one_tool | single tool rendered into prompt, no system |
| 03 | system_user_one_tool | system + user + tool — typical chat-completions shape |
| 04 | prior_assistant_tool_calls | assistant turn with `tool_calls` + null content |
| 05 | prior_tool_response | `role="tool"` with `tool_call_id` + JSON body |
| 06 | multiturn_with_tool_roundtrip | full loop: user → tool_call → tool response → assistant → user |
| 07 | two_tools_different_signatures | two distinct function schemas in one prompt |
| 08 | nested_object_schema | deeply-nested `parameters` (object with object/array children) |
| 09 | array_param | array-typed param with `minItems`/`uniqueItems` constraints |
| 10 | parallel_tool_calls | one assistant turn emitting two `tool_calls` (OpenAI `parallel_tool_calls`) |
