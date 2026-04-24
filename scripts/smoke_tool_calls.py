#!/usr/bin/env python3
"""T3.3 — openai-python streaming vs non-streaming tool-calls parity smoke.

Calls `flambeau serve` twice with identical (seed, prompt, tools) — once
`stream=False` and once `stream=True` — using the official `openai`
SDK, and asserts the reconstructed `tool_calls[]` and `finish_reason`
are equal. This is the guard vs llama.cpp #12601 ("Cannot use tools
with stream") and vLLM #31871 ("stream returns raw text not parsed
tool_calls"); flambeau must not drift into either bug.

Usage:
    # Start the server separately, pointing at a Qwen3.6 GGUF:
    #   flambeau serve --model <gguf> --devices hip:0,1,2,3 --port 8080
    python scripts/smoke_tool_calls.py --base-url http://localhost:8080/v1

Requirements:
    pip install openai

Exit code: 0 on success, 1 on any assertion failure or SDK error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr)
    sys.exit(2)


GET_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}

PROMPT = "What's the current weather in San Francisco? Use the tool."


def normalise_tool_call(tc: Any) -> dict[str, Any]:
    """Strip `id` (per-request random) and canonicalise arguments JSON
    so two calls are comparable byte-for-byte."""
    fn = tc.function if hasattr(tc, "function") else tc["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    raw_args = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
    # Arguments MUST be a string on the wire (guards llama.cpp #20198).
    # If it came through as an object, fail explicitly — that's the bug
    # we're here to prevent.
    if not isinstance(raw_args, str):
        raise AssertionError(
            f"tool_call.function.arguments must be a JSON string, got {type(raw_args).__name__} "
            f"(guards llama.cpp #20198)"
        )
    # Canonicalise arg JSON (whitespace / key order).
    args_obj = json.loads(raw_args)
    args_canon = json.dumps(args_obj, sort_keys=True)
    return {"name": name, "arguments": args_canon}


def run_non_stream(client: OpenAI, model: str, seed: int) -> tuple[list[dict], str]:
    resp = client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=0.0,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[GET_WEATHER_TOOL],
        tool_choice="auto",
        stream=False,
    )
    choice = resp.choices[0]
    tcs = choice.message.tool_calls or []
    return [normalise_tool_call(tc) for tc in tcs], choice.finish_reason


def run_stream(client: OpenAI, model: str, seed: int) -> tuple[list[dict], str]:
    """Reconstruct tool_calls from the SSE delta stream."""
    stream = client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=0.0,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[GET_WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )

    # index → {"name": str, "arguments": str}
    partials: dict[int, dict[str, str]] = {}
    finish_reason = ""
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        if getattr(delta, "tool_calls", None):
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                cur = partials.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc_delta.function
                if fn is not None:
                    if getattr(fn, "name", None):
                        cur["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        cur["arguments"] += fn.arguments

    reconstructed: list[dict] = []
    for idx in sorted(partials):
        p = partials[idx]
        # Arguments must be a valid JSON string, canonicalise.
        args_obj = json.loads(p["arguments"])
        reconstructed.append(
            {"name": p["name"], "arguments": json.dumps(args_obj, sort_keys=True)}
        )
    return reconstructed, finish_reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL of the running flambeau server.",
    )
    ap.add_argument("--model", default="flambeau", help="Model ID.")
    ap.add_argument("--seed", type=int, default=9419, help="Seed for parity.")
    ap.add_argument("--api-key", default="not-needed", help="Dummy API key.")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"→ non-stream call  (seed={args.seed}) ...", flush=True)
    ns_tcs, ns_fr = run_non_stream(client, args.model, args.seed)
    print(f"  tool_calls={ns_tcs}  finish_reason={ns_fr!r}")

    print(f"→ stream call     (seed={args.seed}) ...", flush=True)
    s_tcs, s_fr = run_stream(client, args.model, args.seed)
    print(f"  tool_calls={s_tcs}  finish_reason={s_fr!r}")

    failed = 0
    if ns_tcs != s_tcs:
        print(
            f"FAIL: tool_calls differ.\n  non-stream: {ns_tcs}\n  stream:     {s_tcs}",
            file=sys.stderr,
        )
        failed += 1
    if ns_fr != s_fr:
        print(
            f"FAIL: finish_reason differs. non-stream={ns_fr!r} stream={s_fr!r}",
            file=sys.stderr,
        )
        failed += 1
    if ns_fr != "tool_calls":
        print(
            f"FAIL: finish_reason is {ns_fr!r}, expected 'tool_calls' "
            f"(model didn't emit a tool call — check chat template / prompt)",
            file=sys.stderr,
        )
        failed += 1
    if not ns_tcs:
        print("FAIL: no tool calls in non-stream response", file=sys.stderr)
        failed += 1
    if not s_tcs:
        print("FAIL: no tool calls reconstructed from stream", file=sys.stderr)
        failed += 1

    if failed == 0:
        print("✓ stream ↔ non-stream tool-call parity OK")
        return 0
    print(f"✗ {failed} assertion(s) failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
