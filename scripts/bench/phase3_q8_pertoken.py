#!/usr/bin/env python3
"""
Per-token wall-clock distribution: F16 vs Q8 KV.

If JIT/cold-cache warmup is the cause, we'll see a spike on the
first N tokens then flatten. If the cost is uniform across tokens,
something steady-state differs between layouts.
"""
from __future__ import annotations

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

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 80


def one_chat_pertoken(port: int) -> tuple[float, list[float]]:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user",
                       "content": "Write a long, detailed essay on the "
                                   "history of operating-system schedulers."}],
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": TG_LEN,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    arrivals: list[float] = []
    t0 = time.time()
    with requests.post(url, json=body, stream=True, timeout=600.0) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            arrivals.append(time.time())
    if not arrivals:
        return 0.0, []
    prefill_ms = (arrivals[0] - t0) * 1000.0
    deltas_ms = [(arrivals[i] - arrivals[i - 1]) * 1000.0
                  for i in range(1, len(arrivals))]
    return prefill_ms, deltas_ms


def run_arm(arm: str, env_extras: dict[str, str],
            profile_env: dict[str, str]):
    log_path = ROOT / "scripts" / "bench" / "logs" / f"phase3_q8_pertoken__{arm}.log"
    print(f"\n=== {arm} (env={env_extras}) ===", flush=True)
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=8,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID):
        rm.kill_server(proc)
        return None, None
    try:
        prefill, deltas = one_chat_pertoken(port)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return prefill, deltas


def report(label: str, prefill: float, deltas: list[float]) -> None:
    if not deltas:
        print(f"{label}: no data")
        return
    n = len(deltas)
    print(f"\n--- {label} (prefill={prefill:.1f} ms, {n} inter-token gaps) ---")
    # First 16 individually
    print("  first 16 inter-token gaps (ms):")
    for i, d in enumerate(deltas[:16]):
        print(f"    [{i:2d}] {d:7.2f}")
    if n > 16:
        rest = deltas[16:]
        print(f"  rest [{16}..{n - 1}]:")
        print(f"    n={len(rest)}  min={min(rest):.2f}  "
              f"median={statistics.median(rest):.2f}  "
              f"mean={statistics.mean(rest):.2f}  "
              f"max={max(rest):.2f}  "
              f"stdev={statistics.stdev(rest):.2f}")


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    common = {"FLAMBEAU_INFLIGHT_SLOTS": "8"}

    f16_p, f16_d = run_arm("f16", {**common, "FLAMBEAU_KV": "f16"}, profile_env)
    q8_p,  q8_d  = run_arm("q8",  {**common, "FLAMBEAU_KV": "q8"},  profile_env)

    print("\n========== PER-TOKEN DELTAS ==========")
    if f16_p is not None and f16_d is not None:
        report("F16", f16_p, f16_d)
    if q8_p is not None and q8_d is not None:
        report("Q8",  q8_p,  q8_d)


if __name__ == "__main__":
    main()
