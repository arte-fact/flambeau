# Tool-Calling Live Test Protocol

Companion to `TOOL_CALLING_FIX_PLAN.md`. Defines the end-to-end live
test that:
1. Captures the broken state (T1, baseline).
2. Gates each fix slice (T2–T4) as it lands.
3. Becomes the merge gate for the branch (T5).

Uses the official OpenAI Python client against our
`/v1/chat/completions` endpoint — same shape any real downstream
client uses. If the client likes it, Claude Desktop / Cursor / vLLM
proxies / SGLang clients will too.

## Setup

### Server

Boot the server once per model under test (pp2tp2 / hip:0,2,1,3
remains the production topology):

```bash
flambeau serve \
    --model "$GGUF_PATH" \
    --port 8080 \
    --mesh-mode pp+tp --pp-size 2 --tp-size 2 \
    --devices hip:0,2,1,3 \
    --ctx-cap 4096
```

### Client

```bash
python3 -m venv /tmp/tool_test_venv
source /tmp/tool_test_venv/bin/activate
pip install 'openai>=1.0' pydantic
```

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",   # server doesn't check
)
```

### Sampler params (fixed for repeatability)

`temperature=0.0`, `top_p=1.0`, `seed=12345`. Tool-calling is a
correctness test, not a creativity test — pin the sampler to make
the fixture diffs meaningful.

## Test scenarios

Six scenarios, run in order against each of the four target models.
Each scenario carries an explicit **pass criterion** that the test
harness asserts.

### S1 — Single-tool, single-call (the canonical case)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, e.g. 'Paris, France'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}]

resp = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[{"role": "user",
               "content": "What's the weather in Paris right now?"}],
    tools=tools,
    tool_choice="auto",
    temperature=0.0,
    seed=12345,
    max_tokens=256,
)
```

**Pass criteria** (ALL must hold):
1. `resp.choices[0].finish_reason == "tool_calls"`.
2. `resp.choices[0].message.tool_calls` is a list of length ≥ 1.
3. `tc.id` is a non-empty string for every `tc` in the list.
4. `tc.type == "function"`.
5. `tc.function.name == "get_current_weather"`.
6. `json.loads(tc.function.arguments)` succeeds (arguments is a JSON
   **string**, not an object — llama.cpp #20198 trap).
7. The parsed arguments dict has `location` and the value mentions
   Paris (substring match, case-insensitive).
8. `resp.choices[0].message.content` is None **or** an empty/whitespace
   string. No raw tool-call tokens (`<|tool_call>`, `<tool_call>`,
   `<function=`) leak into `content`.

### S2 — Multi-tool, model must choose one

```python
tools = [
    weather_tool_from_S1,
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the public web for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]
resp = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[{"role": "user",
               "content": "Find me the latest news about ROCm 7.2."}],
    tools=tools,
    tool_choice="auto",
    ...
)
```

**Pass criteria**:
1. `finish_reason == "tool_calls"`.
2. Exactly one `tool_call` (parallel calls covered in S4).
3. `tc.function.name == "search_web"` (NOT `get_current_weather` —
   tests routing).
4. `json.loads(arguments)["query"]` mentions "ROCm" or "7.2".

### S3 — Round-trip: tool result feeds back

```python
# First turn (same as S1) ↑
tc = resp.choices[0].message.tool_calls[0]

# Simulate the tool result and re-call.
followup = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[
        {"role": "user",
         "content": "What's the weather in Paris right now?"},
        resp.choices[0].message,    # assistant message with tool_calls
        {"role": "tool",
         "tool_call_id": tc.id,
         "name": "get_current_weather",
         "content": json.dumps({"temperature_c": 18, "sky": "overcast"})},
    ],
    tools=tools,
    temperature=0.0,
    seed=12345,
    max_tokens=128,
)
```

**Pass criteria**:
1. `followup.choices[0].finish_reason == "stop"` (model is done, no
   second tool call).
2. `followup.choices[0].message.content` is a non-empty string.
3. The content mentions both `Paris` and a temperature near 18°C
   (substring `18` is enough — we're testing the round-trip, not
   chain-of-thought).
4. No raw tool/channel marker tokens in `content`.

### S4 — Parallel tool calls (optional per arch)

Only run if the architecture is known to support it. The OpenAI
shape allows multiple entries in `tool_calls[]`.

```python
resp = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[{"role": "user",
               "content": "Compare the weather in Paris and Tokyo."}],
    tools=[weather_tool],
    tool_choice="auto",
    parallel_tool_calls=True,
    ...
)
```

**Pass criteria** (lenient):
1. `finish_reason == "tool_calls"`.
2. If `len(tool_calls) >= 2`: each `arguments.location` mentions a
   different city ({Paris, Tokyo}), each `tc.id` is unique.
3. If `len(tool_calls) == 1`: don't fail (model chose sequential).
   Record as "parallel-not-supported" in the fixture and move on.

### S5 — `tool_choice="none"` — model must NOT call

```python
resp = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[{"role": "user",
               "content": "What's the weather in Paris?"}],
    tools=[weather_tool],
    tool_choice="none",
    ...
)
```

**Pass criteria**:
1. `finish_reason == "stop"`.
2. `tool_calls` is None or `[]`.
3. `content` is a non-empty natural-language reply (no leaked tokens).

### S6 — No tools declared, plain chat (regression guard)

```python
resp = client.chat.completions.create(
    model=MODEL_LABEL,
    messages=[{"role": "user",
               "content": "What's the capital of France?"}],
    temperature=0.0,
    seed=12345,
    max_tokens=64,
)
```

**Pass criteria**:
1. `finish_reason == "stop"`.
2. `content` contains "Paris" (case-insensitive).
3. No tool-call markers leak (regression check — the parser's
   sentinel detection mustn't fire on plain chat).
4. No `<|channel>thought` / `<channel|>` marker leaks (gemma4 channel-
   leak regression guard from T2).

## Models under test

| Label                       | GGUF path                                                  |
|-----------------------------|------------------------------------------------------------|
| `qwen3.6-27b-q4_0`          | `/artefact/models/Qwen3.6-27B-Q4_0.gguf`                   |
| `qwen3.6-35b-a3b-q4_0`      | `/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`          |
| `gemma4-31b-q4_0`           | `/artefact/models/gemma-4-31B-it-Q4_0.gguf`                |
| `gemma4-26b-a4b-q8_0`       | `/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf`            |

## Harness layout

```
scripts/tool_test/
├── run.py                — orchestrator: loops models × scenarios
├── scenarios.py          — S1–S6 as plain Python functions
├── assertions.py         — typed pass-criterion checks (Pydantic)
└── fixtures/
    ├── qwen3.6-27b-q4_0_S1.json
    ├── qwen3.6-27b-q4_0_S2.json
    ├── ...
```

`run.py` accepts:
- `--model qwen3.6-27b-q4_0` (single model) or `--all` (all four).
- `--scenarios S1,S3,S6` (subset) or default (all six).
- `--capture` — write the full request/response under `fixtures/` for
  later regression checks; do NOT assert. Use this to take the T1
  baseline.
- `--assert` (default) — fail the harness on any pass-criterion
  violation, exit non-zero. Used by T2–T5 gates.

Each scenario writes one JSON file per model on `--capture`:

```json
{
  "model": "qwen3.6-27b-q4_0",
  "scenario": "S1",
  "request": { ... full chat.completions.create kwargs ... },
  "response": { ... full OpenAI response, .model_dump() ... },
  "assertions": [
    {"name": "finish_reason_tool_calls", "expected": "tool_calls",
     "got": "stop", "pass": false},
    ...
  ],
  "verdict": "fail"
}
```

The fixture is the regression oracle: every fix slice asserts the
post-fix output matches the captured fixture for the scenarios it
should pass, and diverges only where the fix legitimately changes
behavior.

## Running

### Baseline capture (T1)

```bash
# For each model:
flambeau serve --model "$GGUF" ... &
python scripts/tool_test/run.py --model "$LABEL" --capture
kill %1
```

Commit the resulting `fixtures/*.json` files. These pin the broken
state. Diffing them across slices is the "what changed" oracle.

### Gate run (T2 / T3 / T4)

```bash
# After landing a fix slice, re-run:
python scripts/tool_test/run.py --model "$LABEL" --assert
# Exit 0 → all scenarios pass for that model.
```

### Final merge gate (T5)

```bash
python scripts/tool_test/run.py --all --assert
# Must exit 0 with every model passing every scenario it's expected to
# pass. Parallel-tool-calls (S4) is lenient per arch.
```

## What "broken" looks like in fixtures (today, on `main`)

Predicted shape based on upstream issues (see `TOOL_CALLING_FIX_PLAN.md`):

- **gemma4 S1**: `tool_calls` is None; `content` contains literal
  `<|tool_call>call:get_current_weather{...}<tool_call|>` plus
  channel marker leak `<channel|>thought`. `finish_reason == "stop"`.
- **gemma4 S6**: `content` starts with `vie vie vie ...` noise or
  `<channel|>thought\n` before the actual answer.
- **qwen3.6 S1**: `tool_calls == []` (empty list); `content` claims
  "I'll call get_current_weather" but no call gets parsed. Random
  `<|im_sep_user|>` token may inject mid-response.
- **qwen3.6-35B-A3B S1**: same shape; MoE doesn't help.

Each of these maps to one of the slice fixes:
- Channel marker → T2 (gemma4 template strip).
- `<|tool_call>` in content but no `tool_calls[]` → T3 (gemma4 parser).
- Empty `tool_calls[]` but prose says "calling X" + `<|im_sep_user|>`
  injection → T4 (qwen3.6 template + EOS-marker set).

## Out of scope

- **Streaming SSE tool-call deltas.** Separate protocol; gate by a
  follow-up streaming-aware harness if needed (vLLM #31871 shape).
  Today the test only exercises non-streaming `/v1/chat/completions`.
- **Anthropic `/v1/messages` tool round-trip.** Cover after the OpenAI
  path is green; the format adapter on `/v1/messages` is thin enough
  that fixing the underlying parser fixes both.
- **`tool_choice={"type":"function","function":{"name":"X"}}`** (force
  one named tool). Add as S7 if a downstream client needs it.
