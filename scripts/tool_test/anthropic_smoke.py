#!/usr/bin/env python3
"""Anthropic Messages API live smoke — official `anthropic` SDK vs flambeau.

The server exposes `/v1/messages` (Anthropic-compatible). The in-tree Rust
tests only exercise the request *translation* and the parser in isolation;
this script proves the actual HTTP envelope deserialises in the real
`anthropic` client and that the round-trip (tool_use -> tool_result ->
answer) holds end-to-end.

Usage:
    # Start a server first, e.g. a tool-capable Qwen:
    #   flambeau serve --model <gguf> --devices hip:0 --port 8080
    python3 scripts/tool_test/anthropic_smoke.py --base-url http://localhost:8080

Pass `--model` to set the `model` field (echoed by the server).

Exit code: 0 if all hard checks pass, 1 otherwise. Thinking (A5) is a
recorded-not-failed probe: the server does not yet emit `thinking` blocks
on the Anthropic path, so its absence is logged as a known gap, not a
failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    import anthropic
except ImportError:
    print("ERROR: pip install --break-system-packages anthropic", file=sys.stderr)
    sys.exit(2)

WEATHER_TOOL = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City, e.g. 'Paris, France'."},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}


def _checks() -> list[dict]:
    return []


def _add(out: list[dict], name: str, ok: bool, detail: str = "") -> None:
    out.append({"name": name, "pass": bool(ok), "detail": detail})


def _text_of(resp: Any) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def _tool_uses(resp: Any) -> list[Any]:
    return [b for b in resp.content if b.type == "tool_use"]


def a1_basic(client: anthropic.Anthropic, model: str) -> list[dict]:
    out = _checks()
    resp = client.messages.create(
        model=model,
        max_tokens=64,
        temperature=0.0,
        messages=[{"role": "user", "content": "What is the capital of France? One word."}],
    )
    txt = _text_of(resp)
    _add(out, "type=message", resp.type == "message", f"type={resp.type!r}")
    _add(out, "role=assistant", resp.role == "assistant", f"role={resp.role!r}")
    _add(out, "stop_reason in {end_turn,max_tokens,stop_sequence}",
         resp.stop_reason in ("end_turn", "max_tokens", "stop_sequence"),
         f"got {resp.stop_reason!r}")
    _add(out, "content mentions Paris", "paris" in txt.lower(), f"text={txt[:80]!r}")
    _add(out, "usage.output_tokens > 0", resp.usage.output_tokens > 0,
         f"out={resp.usage.output_tokens}")
    return out


def a2_tool_use(client: anthropic.Anthropic, model: str) -> tuple[list[dict], Any]:
    out = _checks()
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        tools=[WEATHER_TOOL],
        messages=[{"role": "user", "content": "What's the weather in Paris right now?"}],
    )
    tus = _tool_uses(resp)
    _add(out, "stop_reason=tool_use", resp.stop_reason == "tool_use", f"got {resp.stop_reason!r}")
    _add(out, "exactly one tool_use block", len(tus) == 1, f"len={len(tus)}")
    if tus:
        tu = tus[0]
        _add(out, "tool_use.name=get_current_weather",
             tu.name == "get_current_weather", f"name={tu.name!r}")
        _add(out, "tool_use.id present", bool(tu.id), f"id={tu.id!r}")
        _add(out, "tool_use.input is dict", isinstance(tu.input, dict),
             f"input type={type(tu.input).__name__}")
        loc = (tu.input or {}).get("location", "") if isinstance(tu.input, dict) else ""
        _add(out, "input.location mentions Paris", "paris" in str(loc).lower(),
             f"location={loc!r}")
    return out, resp


def a3_roundtrip(client: anthropic.Anthropic, model: str, prior: Any) -> list[dict]:
    out = _checks()
    tus = _tool_uses(prior)
    if not tus:
        _add(out, "precondition: A2 produced a tool_use", False, "no tool_use to echo back")
        return out
    tu = tus[0]
    resp = client.messages.create(
        model=model,
        max_tokens=128,
        temperature=0.0,
        tools=[WEATHER_TOOL],
        messages=[
            {"role": "user", "content": "What's the weather in Paris right now?"},
            {"role": "assistant", "content": prior.content},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps({"temperature_c": 18, "sky": "overcast"}),
            }]},
        ],
    )
    txt = _text_of(resp)
    _add(out, "stop_reason=end_turn", resp.stop_reason == "end_turn", f"got {resp.stop_reason!r}")
    _add(out, "final answer non-empty", bool(txt.strip()), f"len={len(txt)}")
    _add(out, "answer mentions 18 or overcast",
         ("18" in txt) or ("overcast" in txt.lower()), f"text={txt[:100]!r}")
    return out


def a4_streaming(client: anthropic.Anthropic, model: str) -> list[dict]:
    out = _checks()
    chunks: list[str] = []
    saw_stop = False
    with client.messages.stream(
        model=model,
        max_tokens=64,
        temperature=0.0,
        messages=[{"role": "user", "content": "Name three primary colors, comma separated."}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
        final = stream.get_final_message()
        saw_stop = final.stop_reason in ("end_turn", "max_tokens", "stop_sequence")
    joined = "".join(chunks)
    _add(out, "stream produced text deltas", len(chunks) > 0, f"n_deltas={len(chunks)}")
    _add(out, "final stop_reason valid", saw_stop, "")
    _add(out, "streamed text non-empty", bool(joined.strip()), f"text={joined[:80]!r}")
    return out


def a5_thinking_probe(client: anthropic.Anthropic, model: str) -> list[dict]:
    """Recorded-not-failed: does the Anthropic path surface a thinking block?"""
    out = _checks()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=1.0,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user",
                       "content": "A bat and ball cost $1.10. The bat costs $1 more "
                                  "than the ball. How much is the ball?"}],
        )
    except Exception as e:
        _add(out, "thinking request accepted (recorded)", True,
             f"server rejected thinking param: {e!r} — known gap")
        return out
    has_thinking = any(getattr(b, "type", "") == "thinking" for b in resp.content)
    txt = _text_of(resp)
    _add(out, "thinking block present (recorded, known gap)", True,
         f"emitted={has_thinking} (server drops reasoning on Anthropic path today)")
    _add(out, "answer present", bool(txt.strip()), f"text={txt[:80]!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--model", default="flambeau")
    args = ap.parse_args()

    client = anthropic.Anthropic(base_url=args.base_url, api_key="not-used")

    suites: list[tuple[str, list[dict]]] = []
    a1 = a1_basic(client, args.model)
    suites.append(("A1 basic text", a1))
    a2, prior = a2_tool_use(client, args.model)
    suites.append(("A2 tool_use", a2))
    suites.append(("A3 tool_result round-trip", a3_roundtrip(client, args.model, prior)))
    suites.append(("A4 streaming", a4_streaming(client, args.model)))
    suites.append(("A5 thinking probe", a5_thinking_probe(client, args.model)))

    all_pass = True
    for title, checks in suites:
        verdict = all(c["pass"] for c in checks)
        all_pass = all_pass and verdict
        print(f"\n[{'PASS' if verdict else 'FAIL'}] {title}")
        for c in checks:
            mark = "  ok " if c["pass"] else " FAIL"
            print(f"  {mark} {c['name']}" + (f"  ({c['detail']})" if c["detail"] else ""))

    print(f"\n=== {'ALL PASS' if all_pass else 'FAILURES PRESENT'} ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
