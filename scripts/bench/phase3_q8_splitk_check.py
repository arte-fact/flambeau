#!/usr/bin/env python3
"""
Phase 3 Q8-splitK perf check. Boots flambeau twice (--kv f16 then --kv q8),
warms up with a "Hello" chat, then fires one measured chat with a LONG
prompt (~400 tokens, stream=true, 80-token decode) so n_tokens_kv stays
above 256 across the whole decode loop and split-K kicks in.

Output: F16 vs Q8 wall + decode ms/tok, side by side.
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
CTX_CAP = 4096
TG_LEN = 80

LONG_PROMPT = (
    "Please write a long detailed essay on the history of operating-"
    "system schedulers, covering round-robin, multi-level feedback "
    "queues, priority inheritance, the BSD scheduler, Linux O(1) and "
    "CFS, and modern EEVDF. Include concrete code-level details where "
    "relevant, and provide quantitative comparisons of dispatch latency "
    "across decades of hardware. Aim for at least 2000 words. "
) * 4  # ~400 tokens


def fire(port: int, prompt: str, max_tokens: int = TG_LEN,
         stream: bool = True) -> tuple[float, float, int, str]:
    body = {
        "model": "Qwen3.6-27B-Q4_0",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    if not stream:
        t0 = time.time()
        r = requests.post(url, json=body, timeout=600.0)
        wall_ms = (time.time() - t0) * 1000.0
        d = r.json()
        return wall_ms, 0.0, d["usage"]["completion_tokens"], d["choices"][0]["message"]["content"][:80]
    arrivals = []
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
    if len(arrivals) < 2:
        return 0.0, 0.0, 0, ""
    wall_ms = (arrivals[-1] - t0) * 1000.0
    decode_ms_per_tok = (arrivals[-1] - arrivals[0]) * 1000.0 / max(len(arrivals) - 1, 1)
    return wall_ms, decode_ms_per_tok, len(arrivals), ""


def run_arm(arm: str, profile_env: dict[str, str]) -> dict:
    log_path = ROOT / "scripts" / "bench" / "logs" / f"phase3_q8_splitk__{arm}.log"
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
        # warmup chat (16 tokens, short prompt)
        fire(port, "Hello.", max_tokens=16, stream=False)
        # measured: long prompt + 80-tok decode (n_tokens_kv ≥ ~400 → split-K)
        wall, dec, n, _ = fire(port, LONG_PROMPT, max_tokens=TG_LEN, stream=True)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return {"arm": arm, "wall_ms": wall, "decode_ms_per_tok": dec, "n_tokens": n}


def main() -> None:
    profile_env = rei.load_profile_env(PROFILE)
    f16 = run_arm("f16", profile_env)
    q8  = run_arm("q8",  profile_env)
    print("\n========== summary ==========")
    for r in (f16, q8):
        if r.get("err"):
            print(f"{r['arm']:>4s}: ERR {r['err']}")
        else:
            print(f"{r['arm']:>4s}: wall {r['wall_ms']:7.0f} ms  "
                  f"decode {r['decode_ms_per_tok']:5.2f} ms/tok  "
                  f"n {r['n_tokens']}")


if __name__ == "__main__":
    main()
