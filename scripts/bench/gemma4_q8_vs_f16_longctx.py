#!/usr/bin/env python3
"""
gemma4 Q8 vs F16 KV at multiple ctx — validates d=512 globals Q8
bandwidth win at long ctx.

Per `feedback_q8_kv_head_dim_512_two_wave`: short-ctx Q8 ties F16 (d=512
globals are a small KV share at ~700 tokens). Structural win is at
long ctx where each global layer's full-causal read scales with ctx.

Sweep:
  Model: gemma-4-31B-it-Q4_0 (dense, 50 SWA d=256 + 10 global d=512)
  Topo:  pp2tp2 (hip:0,2,1,3)
  KV:    {f16, q8}
  ctx:   {3000, 8000}

Smoke + single ctx for gemma-4-26B-A4B-it-Q8_0 (MoE sibling, validates
Q8 d=512 across both gemma4 variants).
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

TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}

TG_LEN = 64

PARA = (
    "The Linux kernel scheduler has gone through several major redesigns "
    "since its inception, each motivated by changes in workload patterns, "
    "hardware capabilities, and theoretical understanding of fairness. "
    "The original O(n) scheduler, present until kernel 2.4, walked the "
    "entire runqueue on every scheduling decision -- fine for a few "
    "processes, untenable for hundreds. Linux 2.6 introduced the O(1) "
    "scheduler, with two priority arrays per CPU and a clever expired/active "
    "swap. Then in 2007, Ingo Molnar replaced O(1) with the Completely "
    "Fair Scheduler, modelling each task's CPU consumption as a virtual "
    "runtime tracked in a red-black tree, with the leftmost node always "
    "being the next task to run. CFS aimed for proportional fairness "
    "rather than wall-clock fairness, treating priorities as weights "
    "rather than absolute time slices. The 6.6 kernel introduced EEVDF -- "
    "Earliest Eligible Virtual Deadline First -- which adds latency "
    "guarantees on top of CFS's fairness model. EEVDF computes a virtual "
    "deadline based on requested slice length and runs whichever eligible "
    "task has the earliest deadline. This better serves interactive "
    "workloads that prefer short bursts over long quanta, while keeping "
    "the proportional-fairness invariant CFS provided. "
)


def make_prompt(target_tokens: int) -> str:
    # PARA ~1100 chars ~ 320 tokens. Pad to target.
    n_reps = max(1, target_tokens // 320)
    return PARA * n_reps


def fire(port: int, model_id: str, prompt: str,
         max_tokens: int = TG_LEN) -> dict:
    body = {
        "model": model_id,
        "messages": [
            {"role": "user",
             "content": prompt + "\n\nSummarise the above in two sentences."}
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
        import json as _j
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = _j.loads(payload)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content:
                continue
            arrivals.append(time.time())
    if len(arrivals) < 2:
        return {"err": "no events"}
    wall_ms = (arrivals[-1] - t0) * 1000.0
    ttft_ms = (arrivals[0] - t0) * 1000.0
    decode_ms_tok = (arrivals[-1] - arrivals[0]) * 1000.0 \
        / max(len(arrivals) - 1, 1)
    return {
        "wall_ms":           wall_ms,
        "ttft_ms":           ttft_ms,
        "decode_ms_per_tok": decode_ms_tok,
        "decode_tps":        1000.0 / decode_ms_tok,
        "n_events":          len(arrivals),
    }


def smoke_paris(port: int, model_id: str) -> str:
    body = {
        "model": model_id,
        "messages": [{"role": "user",
                      "content": "What is the capital of France? Answer in one word."}],
        "stream": False,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 16,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    r = requests.post(url, json=body, timeout=120.0)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def boot_and_run(model_id: str, model_path: str, kv: str,
                 ctxs: list[int], ctx_cap: int,
                 profile_env: dict[str, str]) -> list[dict]:
    arm = f"{model_id}__{kv}"
    log_path = ROOT / "scripts" / "bench" / "logs" / f"gemma4_q8_f16__{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== boot {arm} ctx_cap={ctx_cap} ===", flush=True)
    proc, port = rei.boot_with_env(
        model_id, model_path, TOPO,
        env_extras={"FLAMBEAU_INFLIGHT_SLOTS": "1", "FLAMBEAU_KV": kv},
        slots=1,
        log_path=log_path, ctx_cap=ctx_cap,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=model_id):
        rm.kill_server(proc)
        return [{"arm": arm, "err": "boot timeout"}]
    out = []
    try:
        smoke = smoke_paris(port, model_id)
        smoke_ok = "Paris" in smoke
        print(f"  smoke: {smoke!r} -- {'OK' if smoke_ok else 'FAIL'}",
              flush=True)
        if not smoke_ok:
            out.append({"arm": arm, "ctx": 0, "err": f"smoke {smoke!r}"})
            return out
        for ctx in ctxs:
            prompt = make_prompt(ctx)
            reps = []
            for rep in range(3):
                r = fire(port, model_id, prompt, max_tokens=TG_LEN)
                r["arm"] = arm
                r["ctx_target"] = ctx
                r["rep"] = rep
                if r.get("err"):
                    print(f"  ctx≈{ctx} rep{rep}: ERR {r['err']}", flush=True)
                else:
                    print(f"  ctx≈{ctx} rep{rep}: wall {r['wall_ms']:7.0f} ms  "
                          f"ttft {r['ttft_ms']:6.0f} ms  "
                          f"decode {r['decode_ms_per_tok']:6.2f} ms/tok  "
                          f"= {r['decode_tps']:5.2f} tps  n {r['n_events']}",
                          flush=True)
                reps.append(r)
            ok = [r for r in reps if not r.get("err")]
            if not ok:
                out.append({"arm": arm, "ctx_target": ctx, "err": "all reps err"})
                continue
            import statistics as _st
            median = {
                "arm": arm, "ctx_target": ctx,
                "wall_ms":           _st.median(r["wall_ms"]           for r in ok),
                "ttft_ms":           _st.median(r["ttft_ms"]           for r in ok),
                "decode_ms_per_tok": _st.median(r["decode_ms_per_tok"] for r in ok),
                "decode_tps":        _st.median(r["decode_tps"]        for r in ok),
                "n_events":          ok[-1]["n_events"],
                "n_reps":            len(ok),
            }
            print(f"  ctx≈{ctx} MEDIAN({len(ok)}): decode "
                  f"{median['decode_ms_per_tok']:6.2f} ms/tok "
                  f"= {median['decode_tps']:5.2f} tps", flush=True)
            out.append(median)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return out


def main() -> None:
    profile_env = rei.load_profile_env(PROFILE)

    all_rows: list[dict] = []

    # gemma4-31B-Q4_0 dense: 2 contexts, 2 KV modes
    m31 = "/artefact/models/gemma-4-31B-it-Q4_0.gguf"
    m31_id = "gemma-4-31B-it-Q4_0"
    for kv in ("f16", "q8"):
        rows = boot_and_run(m31_id, m31, kv,
                            ctxs=[3000, 8000], ctx_cap=10240,
                            profile_env=profile_env)
        all_rows.extend(rows)

    # gemma4-26B-A4B-Q8_0 MoE: smoke + 1 ctx, 2 KV modes
    m26 = "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf"
    m26_id = "gemma-4-26B-A4B-it-Q8_0"
    for kv in ("f16", "q8"):
        rows = boot_and_run(m26_id, m26, kv,
                            ctxs=[3000], ctx_cap=4096,
                            profile_env=profile_env)
        all_rows.extend(rows)

    # ---------- summary ----------
    print("\n========== summary ==========", flush=True)
    print(f"{'arm':50s} {'ctx':>6s} {'tps':>7s} {'ttft':>8s}")
    for r in all_rows:
        if r.get("err"):
            print(f"{r['arm']:50s} {r.get('ctx_target', 0):>6d} ERR {r['err']}")
        else:
            print(f"{r['arm']:50s} {r['ctx_target']:>6d} "
                  f"{r['decode_tps']:>7.2f} {r['ttft_ms']:>8.0f}")

    # Speedup table per (model, ctx)
    by_key: dict[tuple, dict] = {}
    for r in all_rows:
        if r.get("err"):
            continue
        model = r["arm"].rsplit("__", 1)[0]
        kv = r["arm"].rsplit("__", 1)[1]
        by_key.setdefault((model, r["ctx_target"]), {})[kv] = r

    print("\n========== Q8 vs F16 ==========", flush=True)
    print(f"{'model':40s} {'ctx':>6s} {'F16 tps':>8s} {'Q8 tps':>8s} {'Q8/F16':>8s}")
    for (model, ctx), arms in sorted(by_key.items()):
        if "f16" in arms and "q8" in arms:
            f = arms["f16"]["decode_tps"]
            q = arms["q8"]["decode_tps"]
            print(f"{model:40s} {ctx:>6d} {f:>8.2f} {q:>8.2f} {q/f:>8.3f}x")


if __name__ == "__main__":
    main()
