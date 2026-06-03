#!/usr/bin/env python3
"""End-to-end validation bench for the new Phase 3 dp4a kernels.

For each quant: boot pp2tp2, smoke (must contain 'Paris'), then
3-rep median decode tps at ctx 4096 with f16 KV. Single-stream
greedy, 64 tg. Compares to a per-quant bandwidth-ceiling reference
where one is available (Phase 1 scalar-equivalent baseline).
"""
from __future__ import annotations

import json
import statistics
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

TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
TG_LEN = 64

# (label, file, phase, scalar-baseline tps if known)
QUANTS = [
    ("UD-Q2_K_XL",  "Qwen3.6-27B-UD-Q2_K_XL.gguf",   "3e Q2_K",        None),
    ("IQ4_NL",      "Qwen3.6-27B-IQ4_NL.gguf",       "3d IQ4_NL",      None),
    ("UD-IQ3_XXS",  "Qwen3.6-27B-UD-IQ3_XXS.gguf",   "3g IQ3_XXS",     None),
    ("UD-IQ2_XXS",  "Qwen3.6-27B-UD-IQ2_XXS.gguf",   "3j IQ2_XXS",     None),
]

PARA = (
    "The Linux kernel scheduler has gone through several major redesigns "
    "since its inception, each motivated by changes in workload patterns, "
    "hardware capabilities, and theoretical understanding of fairness. "
    "Linux 2.6 introduced the O(1) scheduler with two priority arrays "
    "per CPU. Ingo Molnar replaced O(1) with the Completely Fair "
    "Scheduler in 2007, modelling each task's CPU consumption as a "
    "virtual runtime tracked in a red-black tree. The 6.6 kernel "
    "introduced EEVDF, adding latency guarantees on top of CFS's "
    "fairness model. "
)
PROMPT = PARA * 12  # ~3500 tokens


def smoke(port: int, model_id: str) -> str:
    body = {"model": model_id,
            "messages": [{"role": "user",
                          "content": "What is the capital of France? Answer in one word."}],
            "stream": False, "temperature": 0.0, "seed": 0, "max_tokens": 16}
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=body, timeout=120.0)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def fire(port: int, model_id: str) -> dict:
    body = {"model": model_id,
            "messages": [{"role": "user",
                          "content": PROMPT + "\n\nSummarise the above in one sentence."}],
            "stream": True, "temperature": 0.0, "seed": 0, "max_tokens": TG_LEN}
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
    return {"wall_ms": wall_ms, "ttft_ms": ttft_ms,
            "decode_ms_per_tok": decode_ms_tok,
            "decode_tps": 1000.0 / decode_ms_tok,
            "n_events": len(arrivals)}


def run_arm(label: str, gguf: str, phase: str,
            profile_env: dict[str, str]) -> dict:
    model_path = f"/artefact/models/{gguf}"
    model_id = Path(gguf).stem
    log_path = LOG_DIR / f"phase3_validate__{label}.log"
    print(f"\n=== {label} ({phase}) ===", flush=True)
    proc, port = rei.boot_with_env(
        model_id, model_path, TOPO,
        env_extras={"FLAMBEAU_INFLIGHT_SLOTS": "1", "FLAMBEAU_KV": "f16"},
        slots=1, log_path=log_path, ctx_cap=4096,
        profile_env=profile_env,
    )
    out = {"label": label, "phase": phase}
    try:
        if not rm.wait_ready(port, timeout_s=600.0, model_id=model_id):
            out["err"] = "boot timeout"
            return out
        sm = smoke(port, model_id)
        smoke_ok = "Paris" in sm or "paris" in sm.lower()
        print(f"  smoke: {sm!r} -- {'OK' if smoke_ok else 'FAIL'}", flush=True)
        out["smoke"] = sm
        out["smoke_ok"] = smoke_ok
        if not smoke_ok:
            return out
        reps = []
        for rep in range(3):
            r = fire(port, model_id)
            if r.get("err"):
                print(f"  rep{rep}: ERR {r['err']}", flush=True)
            else:
                print(f"  rep{rep}: decode {r['decode_ms_per_tok']:.2f} ms/tok "
                      f"= {r['decode_tps']:.2f} tps  ttft {r['ttft_ms']:.0f} ms",
                      flush=True)
            reps.append(r)
        ok = [r for r in reps if not r.get("err")]
        if not ok:
            out["err"] = "all reps err"
            return out
        out["decode_tps_median"] = statistics.median(r["decode_tps"] for r in ok)
        out["decode_ms_per_tok_median"] = statistics.median(
            r["decode_ms_per_tok"] for r in ok)
        out["ttft_ms_median"] = statistics.median(r["ttft_ms"] for r in ok)
        print(f"  MEDIAN: {out['decode_tps_median']:.2f} tps", flush=True)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return out


def main() -> None:
    profile_env = rei.load_profile_env(PROFILE)
    rows = []
    for label, gguf, phase, _ in QUANTS:
        rows.append(run_arm(label, gguf, phase, profile_env))
    print("\n========== summary ==========", flush=True)
    print(f"{'quant':14s} {'phase':14s} {'smoke':>5s} {'tps':>8s}")
    for r in rows:
        if r.get("err"):
            print(f"{r['label']:14s} {r['phase']:14s} ERR {r['err']}")
        else:
            ok = "OK" if r["smoke_ok"] else "FAIL"
            tps = r.get("decode_tps_median")
            tps_s = f"{tps:.2f}" if tps else "-"
            print(f"{r['label']:14s} {r['phase']:14s} {ok:>5s} {tps_s:>8s}")


if __name__ == "__main__":
    main()
