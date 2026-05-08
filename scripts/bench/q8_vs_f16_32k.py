#!/usr/bin/env python3
"""
Q8 vs F16 KV at 32k context — Phase 4 follow-up perf cert.

Prediction (per `certs/perf/q8_kv_dp4a_2026_05_07.md`):
  At ctx=300, KV is ~0.2% of per-step HBM traffic — Q8 ≈ F16 wall.
  At ctx=32k, KV grows ~110×, becomes a meaningful share — Q8 should
  finally beat F16 by ~5-10% on decode wall.

Workload: ~30k-token prompt + 64-token greedy decode. Compare F16
and Q8 wall + decode rate.

Boots two servers sequentially (same harness as `phase3_q8_splitk_check.py`).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore
import run_env_impact as rei  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 32768
TG_LEN = 64

# Construct a ~30k-token prompt by repeating a substantive paragraph.
# Average token ≈ 3.5 chars; 110k chars ≈ 31k tokens.
PARA = (
    "The Linux kernel scheduler has gone through several major redesigns "
    "since its inception, each motivated by changes in workload patterns, "
    "hardware capabilities, and theoretical understanding of fairness. "
    "The original O(n) scheduler, present until kernel 2.4, walked the "
    "entire runqueue on every scheduling decision — fine for a few "
    "processes, untenable for hundreds. Linux 2.6 introduced the O(1) "
    "scheduler, with two priority arrays per CPU and a clever expired/active "
    "swap. Then in 2007, Ingo Molnar replaced O(1) with the Completely "
    "Fair Scheduler, modelling each task's CPU consumption as a virtual "
    "runtime tracked in a red-black tree, with the leftmost node always "
    "being the next task to run. CFS aimed for proportional fairness "
    "rather than wall-clock fairness, treating priorities as weights "
    "rather than absolute time slices. The 6.6 kernel introduced EEVDF — "
    "Earliest Eligible Virtual Deadline First — which adds latency "
    "guarantees on top of CFS's fairness model. EEVDF computes a virtual "
    "deadline based on requested slice length and runs whichever eligible "
    "task has the earliest deadline. This better serves interactive "
    "workloads that prefer short bursts over long quanta, while keeping "
    "the proportional-fairness invariant CFS provided. "
)
# Repeat to reach ~30k tokens (PARA is ~1100 chars ≈ 320 tokens, want ~95×)
PROMPT = PARA * 95


def fire(port: int, max_tokens: int = TG_LEN) -> dict:
    body = {
        "model": "Qwen3.6-27B-Q4_0",
        "messages": [
            {"role": "user",
             "content": PROMPT + "\n\nSummarise the above in two sentences."}
        ],
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    arrivals = []
    t0 = time.time()
    with requests.post(url, json=body, stream=True, timeout=1200.0) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            arrivals.append(time.time())
    if len(arrivals) < 2:
        return {"err": "no events"}
    return {
        "wall_ms":           (arrivals[-1] - t0) * 1000.0,
        "first_event_ms":    (arrivals[0]  - t0) * 1000.0,
        "decode_ms_per_tok": (arrivals[-1] - arrivals[0]) * 1000.0
                              / max(len(arrivals) - 1, 1),
        "n_events":          len(arrivals),
    }


def run_arm(arm: str, profile_env: dict[str, str]) -> dict:
    log_path = ROOT / "scripts" / "bench" / "logs" / f"q8_vs_f16_32k__{arm}.log"
    print(f"\n=== {arm} ===", flush=True)
    proc, port = rei.boot_with_env(
        "Qwen3.6-27B-Q4_0", MODEL_PATH, TOPO,
        env_extras={"FLAMBEAU_INFLIGHT_SLOTS": "8", "FLAMBEAU_KV": arm},
        slots=8,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id="Qwen3.6-27B-Q4_0"):
        rm.kill_server(proc)
        return {"arm": arm, "err": "boot timeout"}
    try:
        # warmup chat (16 tokens, short prompt) — primes JIT, doesn't grow KV
        # to anything significant.
        body = {
            "model": "Qwen3.6-27B-Q4_0",
            "messages": [{"role": "user", "content": "Hello."}],
            "stream": False, "temperature": 0.0,
            "seed": 0, "max_tokens": 16,
        }
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        requests.post(url, json=body, timeout=120.0)
        # measured: ~30k-token prompt + 64-tok decode
        result = fire(port, max_tokens=TG_LEN)
        result["arm"] = arm
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return result


def main() -> None:
    print(f"Estimated prompt length: {len(PROMPT):,} chars (~{len(PROMPT)/3.5:.0f} tokens)")
    profile_env = rei.load_profile_env(PROFILE)
    f16 = run_arm("f16", profile_env)
    q8  = run_arm("q8",  profile_env)

    print("\n========== summary ==========")
    for r in (f16, q8):
        if r.get("err"):
            print(f"{r['arm']:>4s}: ERR {r['err']}")
        else:
            ttft = r['first_event_ms']
            dec = r['decode_ms_per_tok']
            wall = r['wall_ms']
            n = r['n_events']
            print(f"{r['arm']:>4s}: wall {wall:7.0f} ms  ttft {ttft:6.0f} ms  "
                  f"decode {dec:6.2f} ms/tok  n {n}")

    if not f16.get("err") and not q8.get("err"):
        f16_dec = f16["decode_ms_per_tok"]
        q8_dec  = q8["decode_ms_per_tok"]
        f16_wall = f16["wall_ms"]
        q8_wall  = q8["wall_ms"]
        print(f"\nDecode speedup Q8/F16: {f16_dec/q8_dec:.3f}× "
              f"({f16_dec - q8_dec:+.2f} ms/tok)")
        print(f"Wall speedup Q8/F16:   {f16_wall/q8_wall:.3f}× "
              f"({f16_wall - q8_wall:+.0f} ms)")


if __name__ == "__main__":
    main()
