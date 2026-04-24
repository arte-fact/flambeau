# Tool-call parser fixture corpus

Ships the regression surface for `QwenCoderXmlParser` (T2.4). Each
fixture is a pair:

- `<name>.txt` — raw model-output bytes that would arrive from the
  decoder-text stream.
- `<name>.expected.jsonl` — one JSON per line, each line is a serialised
  `ParserEvent` after the parser has coalesced adjacent
  `TextDelta` / `ThinkDelta` / `ToolCallArgumentsDelta(same index)`
  events into single entries.

The harness (`crates/server/tests/tool_call_parser_fixtures.rs`)
runs every fixture through the parser twice — once as a single `push`
and once char-by-char — and asserts both coalesced event streams match
the committed `.expected.jsonl`. Streaming-vs-one-shot equivalence is a
load-bearing invariant.

## Adding a new fixture

1. Drop the raw bytes into a new `.txt`.
2. Run `cargo test -p flambeau-server --test tool_call_parser_fixtures -- --nocapture`
   — the test will print what the parser produced for the new file.
3. If that output is correct, save it as `<name>.expected.jsonl`. If
   not, fix the parser, not the expectation.

**New bugs land as new fixture pairs, never as new assertion code.** The
explicit failure modes of llama.cpp (#20837, #21118), vLLM (#31871),
and community Reddit reports are each one fixture here.

## Fixture index

| File | What it exercises |
|---|---|
| `qwen_coder_simple.txt` | Baseline: one tool_call, one parameter. |
| `qwen_coder_parallel_calls.txt` | Two sequential `<tool_call>` blocks in one turn — monotonic `index`. |
| `qwen_coder_multiline_param.txt` | Parameter value spanning 3 lines. |
| `qwen_coder_angle_chars_in_value.txt` | `<T>` inside a param value — must not be mistaken for a tag. |
| `qwen_coder_think_then_tool.txt` | `</think>` immediately followed by `<tool_call>` — llama.cpp #21118. |
| `qwen_coder_tool_inside_think.txt` | Tool-call syntax inside `<think>` — must NOT fire (llama.cpp #20837). |
| `qwen_coder_plain_text.txt` | No tool call at all — passthrough. |
| `qwen_coder_text_prefix_suffix.txt` | Free text before and after the tool_call. |
| `qwen_coder_unterminated.txt` | Truncated mid-body; content must surface on finish. |
