#!/usr/bin/env python3
"""
Quant × KV × ctx × topo perf matrix for Qwen3.6-27B.

Phase 1 of `doc/QUANT_PERF_PLAN.md`: 9 quants × 2 KV × 2 ctx × 2 topo
= 72 cells. Single-stream greedy decode, 64 tg, 3-rep median.

Boots per (quant, KV, topo); measures both ctx within one boot.
Skips cells where the model doesn't fit. Writes raw rows to
`certs/perf/quant_matrix_qwen36_27b_<DATE>.json` and a markdown
summary to `.md` next to it.

A cell is one (quant, kv, ctx, topo) triple. An "arm" is one boot =
one (quant, kv, topo). Each arm produces up to len(CTXS) cell rows.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore
import run_env_impact as rei  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"
LOG_DIR = ROOT / "scripts" / "bench" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID_FOR_API = "qwen3.6-27b"

QUANTS = [
    # (name, gguf_path_under_/artefact/models, weight_GB)
    # Q4_K_M-mtp.gguf is excluded: it's the multi-token-prediction
    # variant with an extra blk.64.ssm_norm head tensor that
    # flambeau qwen35 doesn't consume. Need a non-MTP Q4_K_M GGUF.
    ("Q4_0",       "Qwen3.6-27B-Q4_0.gguf",       15.0),
    ("Q4_1",       "Qwen3.6-27B-Q4_1.gguf",       17.0),
    ("Q8_0",       "Qwen3.6-27B-Q8_0.gguf",       27.0),
    ("UD-Q3_K_XL", "Qwen3.6-27B-UD-Q3_K_XL.gguf", 14.0),
    ("UD-Q4_K_XL", "Qwen3.6-27B-UD-Q4_K_XL.gguf", 17.0),
    ("UD-Q6_K_XL", "Qwen3.6-27B-UD-Q6_K_XL.gguf", 24.0),
    ("UD-Q8_K_XL", "Qwen3.6-27B-UD-Q8_K_XL.gguf", 33.0),
]

KV_MODES = ["f16", "q8"]
CTXS = [512, 4096]
TG_LEN = 64
REPS = 3

TOPOS = {
    "pp2tp2": {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
               "pp_size": 2, "tp_size": 2, "weight_div": 4},
}

MI50_VRAM_GB = 32.0
# headroom for KV scratch + decode scratch + rocBLAS workspace
TOPO_HEADROOM_GB = 4.0


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
    n = max(1, target_tokens // 320)
    return PARA * n


def fits(weight_gb: float, topo: dict, ctx_cap: int) -> bool:
    per_gpu = weight_gb / topo["weight_div"]
    # Crude KV slab estimate: 2048 (kv_width worst case) × ctx_cap × 2 (KV)
    # × 2 (F16) bytes / 1e9 GB. Cap by per-GPU since KV is sharded TP-side.
    kv_gb_per_gpu = 2 * 2048 * ctx_cap * 2 / 1e9 / max(topo["weight_div"], 1)
    return per_gpu + kv_gb_per_gpu + TOPO_HEADROOM_GB <= MI50_VRAM_GB


def fire(port: int, prompt: str, max_tokens: int = TG_LEN,
         model_id: str = "qwen-model") -> dict:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt
                      + "\n\nSummarise the above in one sentence."}],
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
            try:
                obj = json.loads(payload)
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
        "wall_ms": wall_ms,
        "ttft_ms": ttft_ms,
        "decode_ms_per_tok": decode_ms_tok,
        "decode_tps": 1000.0 / decode_ms_tok,
        "n_events": len(arrivals),
    }


def smoke(port: int, model_id: str) -> str:
    body = {
        "model": model_id,
        "messages": [{"role": "user",
                      "content": "What is the capital of France? "
                                 "Answer in one word."}],
        "stream": False,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 16,
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=body, timeout=120.0)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def run_arm(quant: str, gguf: str, kv: str, topo_name: str, topo: dict,
            ctxs: list[int], ctx_cap: int,
            profile_env: dict[str, str]) -> list[dict]:
    arm = f"{quant}__{kv}__{topo_name}"
    log_path = LOG_DIR / f"qmtx__{arm}.log"
    model_id = Path(gguf).stem
    print(f"\n=== boot {arm} ctx_cap={ctx_cap} ===", flush=True)
    proc, port = rei.boot_with_env(
        model_id, f"/artefact/models/{gguf}", topo,
        env_extras={"FLAMBEAU_INFLIGHT_SLOTS": "1", "FLAMBEAU_KV": kv},
        slots=1,
        log_path=log_path, ctx_cap=ctx_cap,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=model_id):
        rm.kill_server(proc)
        return [{"quant": quant, "kv": kv, "topo": topo_name,
                 "err": "boot timeout"}]
    out = []
    try:
        s = smoke(port, model_id)
        smoke_ok = "Paris" in s or "paris" in s.lower()
        print(f"  smoke: {s!r} -- {'OK' if smoke_ok else 'FAIL'}",
              flush=True)
        if not smoke_ok:
            return [{"quant": quant, "kv": kv, "topo": topo_name,
                     "err": f"smoke {s!r}"}]
        for ctx in ctxs:
            prompt = make_prompt(ctx)
            reps = []
            for rep in range(REPS):
                r = fire(port, prompt, max_tokens=TG_LEN,
                         model_id=model_id)
                r.update({"quant": quant, "kv": kv, "topo": topo_name,
                          "ctx_target": ctx, "rep": rep})
                if r.get("err"):
                    print(f"  ctx≈{ctx} rep{rep}: ERR {r['err']}",
                          flush=True)
                else:
                    print(f"  ctx≈{ctx} rep{rep}: decode "
                          f"{r['decode_ms_per_tok']:6.2f} ms/tok "
                          f"= {r['decode_tps']:5.2f} tps "
                          f"ttft {r['ttft_ms']:5.0f}",
                          flush=True)
                reps.append(r)
            ok = [r for r in reps if not r.get("err")]
            if not ok:
                out.append({"quant": quant, "kv": kv, "topo": topo_name,
                            "ctx_target": ctx, "err": "all reps err"})
                continue
            import statistics as st
            median = {
                "quant": quant, "kv": kv, "topo": topo_name,
                "ctx_target": ctx,
                "wall_ms":           st.median(r["wall_ms"]           for r in ok),
                "ttft_ms":           st.median(r["ttft_ms"]           for r in ok),
                "decode_ms_per_tok": st.median(r["decode_ms_per_tok"] for r in ok),
                "decode_tps":        st.median(r["decode_tps"]        for r in ok),
                "n_reps":            len(ok),
            }
            print(f"  ctx≈{ctx} MEDIAN({len(ok)}): "
                  f"{median['decode_tps']:5.2f} tps", flush=True)
            out.append(median)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return out


def write_summary(rows: list[dict], cert_path: Path) -> None:
    import statistics as st
    # Pivot: rows indexed by (quant, kv, topo, ctx) → decode_tps
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if r.get("err") or r.get("decode_tps") is None:
            continue
        if r.get("ctx_target") is None:
            continue
        by_key[(r["quant"], r["kv"], r["topo"], r["ctx_target"])] = r

    lines = []
    lines.append("# Qwen3.6-27B quant × KV × ctx × topo perf matrix\n")
    lines.append(f"Bench script: `scripts/bench/quant_perf_matrix.py`. "
                 f"Single-stream greedy, 64 tg, {REPS}-rep median.\n")
    for topo_name in TOPOS:
        lines.append(f"\n## {topo_name}\n")
        for ctx in CTXS:
            lines.append(f"\n### ctx ≈ {ctx}\n")
            lines.append("| quant       | F16 KV tps | Q8 KV tps | Q8/F16 |")
            lines.append("|-------------|-----------:|----------:|-------:|")
            for q, _, _ in QUANTS:
                f = by_key.get((q, "f16", topo_name, ctx))
                q8 = by_key.get((q, "q8", topo_name, ctx))
                f_str  = f"{f['decode_tps']:6.2f}"  if f  else "    --"
                q8_str = f"{q8['decode_tps']:6.2f}" if q8 else "    --"
                ratio = (f"{q8['decode_tps']/f['decode_tps']:.3f}×"
                         if f and q8 else "    --")
                lines.append(f"| {q:11s} | {f_str:>10s} | {q8_str:>9s} | {ratio:>6s} |")
    cert_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {cert_path}", flush=True)


def main() -> None:
    profile_env = rei.load_profile_env(PROFILE)
    all_rows: list[dict] = []
    skipped: list[tuple[str, str, str]] = []

    for topo_name, topo in TOPOS.items():
        for quant, gguf, weight_gb in QUANTS:
            # Max ctx_cap we want for this arm: largest CTXS value.
            ctx_cap = max(CTXS)
            if not fits(weight_gb, topo, ctx_cap):
                print(f"SKIP {quant} {topo_name}: {weight_gb} GB "
                      f"won't fit at ctx_cap={ctx_cap} on "
                      f"{topo['weight_div']}×MI50", flush=True)
                for kv in KV_MODES:
                    skipped.append((quant, kv, topo_name))
                continue
            for kv in KV_MODES:
                rows = run_arm(quant, gguf, kv, topo_name, topo,
                               CTXS, ctx_cap, profile_env)
                all_rows.extend(rows)

    # Cert paths
    today = time.strftime("%Y_%m_%d", time.gmtime())
    cert_dir = ROOT / "certs" / "perf"
    cert_dir.mkdir(parents=True, exist_ok=True)
    json_path = cert_dir / f"quant_matrix_qwen36_27b_{today}.json"
    md_path   = cert_dir / f"quant_matrix_qwen36_27b_{today}.md"
    json_path.write_text(json.dumps(
        {"quants": [q[0] for q in QUANTS], "kv": KV_MODES,
         "ctx": CTXS, "topo": list(TOPOS.keys()),
         "rows": all_rows, "skipped": skipped},
        indent=2))
    write_summary(all_rows, md_path)

    print("\n========== summary ==========", flush=True)
    print(f"cells run: {sum(1 for r in all_rows if not r.get('err') and r.get('ctx_target'))}")
    print(f"cells skipped (no fit): {len(skipped)}")
    print(f"cells with errors: {sum(1 for r in all_rows if r.get('err'))}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
