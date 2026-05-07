#!/usr/bin/env python3
"""
Confirm the Q8 KV slowdown is JIT-warmup-dominated, not steady-state.

Boot the server ONCE with --kv q8, fire 3 sequential 80-token chats
on the same connection. If JIT warmup is the cause, run #1 will be
slow (~78 ms/tok), runs #2 and #3 will land at ~36 ms/tok.

Same harness for --kv f16 as a control.
"""
from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass
class Run:
    arm: str
    run_idx: int
    prefill_ms: float
    decode_ms: float
    total_ms: float


def one_chat(port: int) -> tuple[float, float, float]:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user",
                       "content": "Write a long, detailed essay on the "
                                   "history of operating-system schedulers "
                                   "from the 1960s through today. Cover at "
                                   "least: round-robin, multi-level feedback "
                                   "queues, priority inheritance, the BSD "
                                   "scheduler, Linux O(1) and CFS, and modern "
                                   "EEVDF. Include code-level details where "
                                   "relevant. Keep going until you've "
                                   "exhausted the topic — at least several "
                                   "thousand words."}],
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": TG_LEN,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    first_t = None
    last_t = None
    n = 0
    t0 = time.time()
    with requests.post(url, json=body, stream=True, timeout=600.0) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            if first_t is None:
                first_t = time.time()
            last_t = time.time()
            n += 1
    total = (last_t - t0) * 1000.0 if last_t else 0.0
    prefill = (first_t - t0) * 1000.0 if first_t else 0.0
    decode = ((last_t - first_t) * 1000.0 / max(n - 1, 1)) if first_t and last_t and n > 1 else 0.0
    return prefill, decode, total


def run_arm(arm: str, env_extras: dict[str, str],
            profile_env: dict[str, str], n_chats: int = 3) -> list[Run]:
    log_path = ROOT / "scripts" / "bench" / "logs" / f"phase3_q8_warmup__{arm}.log"
    print(f"\n=== {arm} (env={env_extras}) ===", flush=True)
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=8,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID):
        rm.kill_server(proc)
        return []
    out = []
    try:
        for i in range(n_chats):
            p, d, t = one_chat(port)
            out.append(Run(arm, i, p, d, t))
            print(f"  chat #{i}: prefill={p:.0f} ms  decode={d:.2f} ms/tok  total={t:.0f} ms",
                  flush=True)
            time.sleep(0.3)  # let server settle between chats
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return out


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    common = {"FLAMBEAU_INFLIGHT_SLOTS": "8"}

    f16 = run_arm("f16", {**common, "FLAMBEAU_KV": "f16"}, profile_env)
    q8 = run_arm("q8",  {**common, "FLAMBEAU_KV": "q8"}, profile_env)

    print("\n--- summary ---")
    for r in f16 + q8:
        print(f"  {r.arm} #{r.run_idx}: decode {r.decode_ms:.2f} ms/tok  total {r.total_ms:.0f} ms")


if __name__ == "__main__":
    main()
