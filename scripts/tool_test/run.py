#!/usr/bin/env python3
"""Tool-calling live test harness.

Usage:
    python3 scripts/tool_test/run.py --model qwen3.6-27b-q4_0 --capture
    python3 scripts/tool_test/run.py --model gemma4-31b-q4_0 --assert
    python3 scripts/tool_test/run.py --all --assert

Expects a flambeau server already running on the default port.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys
from typing import Any

from openai import OpenAI

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assertions  # noqa: E402
import scenarios  # noqa: E402

MODELS = ["qwen3.6-27b-q4_0", "qwen3.6-35b-a3b-q4_0",
          "gemma4-31b-q4_0", "gemma4-26b-a4b-q8_0"]

SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def fixture_path(model: str, scenario: str) -> pathlib.Path:
    return HERE / "fixtures" / f"{model}_{scenario}.json"


def to_dict(obj: Any) -> Any:
    """Serialize OpenAI pydantic model to plain JSON-ready dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def run_one(client: OpenAI, model: str, scenario: str,
            capture: bool, do_assert: bool) -> tuple[bool, dict]:
    """Run a single scenario. Returns (passed, fixture_dict)."""
    print(f"  [{scenario}] ", end="", flush=True)
    if scenario == "S3":
        # S3 needs S1 to fire first.
        s1_kwargs = scenarios.s1(model)
        try:
            s1_resp = client.chat.completions.create(**s1_kwargs)
        except Exception as e:
            print(f"S1 precondition FAILED: {e}")
            return False, {"scenario": scenario, "error": f"S1 precond: {e}"}
        s1_msg = s1_resp.choices[0].message
        tcs = s1_msg.tool_calls or []
        if not tcs:
            print("SKIP (S1 produced no tool_calls)")
            fixture = {
                "model": model, "scenario": scenario,
                "skipped": True,
                "reason": "S1 precondition produced no tool_calls",
                "verdict": "skip",
            }
            return True, fixture
        kwargs = scenarios.s3_followup(model, to_dict(s1_msg), tcs[0].id)
    else:
        kwargs = getattr(scenarios, scenario.lower())(model)
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        return False, {"scenario": scenario, "error": str(e),
                       "verdict": "error"}
    checks = assertions.CHECKS[scenario](resp)
    verdict = "pass" if all(c["pass"] for c in checks) else "fail"
    fixture = {
        "model": model,
        "scenario": scenario,
        "request": kwargs,
        "response": to_dict(resp),
        "checks": checks,
        "verdict": verdict,
    }
    fails = [c for c in checks if not c["pass"]]
    print(f"{verdict.upper()}"
          + (f" ({len(fails)} failed: "
             + ", ".join(c["name"] for c in fails[:3]) + ")"
             if fails else ""))
    return verdict == "pass" or not do_assert, fixture


def run_model(client: OpenAI, model: str, scens: list[str],
              capture: bool, do_assert: bool) -> int:
    print(f"=== model: {model} ===")
    failed = 0
    for s in scens:
        ok, fix = run_one(client, model, s, capture, do_assert)
        if capture:
            fp = fixture_path(model, s)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(fix, indent=2, default=str))
        if not ok:
            failed += 1
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help=f"one of {MODELS} or omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="run every model in MODELS")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help="comma-separated subset of S1..S6")
    ap.add_argument("--capture", action="store_true",
                    help="write fixtures, never fail")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit non-zero on any failed check")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    args = ap.parse_args()

    if not args.all and not args.model:
        ap.error("specify --model X or --all")
    models = MODELS if args.all else [args.model]
    scens = [s.strip().upper() for s in args.scenarios.split(",")]
    for s in scens:
        if s not in SCENARIOS:
            ap.error(f"unknown scenario: {s}")

    client = OpenAI(base_url=args.base_url, api_key="not-used")
    total_failed = 0
    for m in models:
        total_failed += run_model(client, m, scens, args.capture,
                                  args.do_assert)
    if args.do_assert and total_failed:
        print(f"\nFAILED: {total_failed} scenario(s) failed")
        return 1
    print(f"\nDone. {total_failed} failed (capture mode)" if args.capture
          else "\nAll asserted scenarios passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
