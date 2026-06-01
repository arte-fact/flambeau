#!/usr/bin/env python3
"""Single-request baseline: fire one streaming chat completion N times,
report TTFT, decode tok/s, and total wall (median of N). No concurrency,
no `batched_pending` activity — measures the K4c lock-acquire overhead
when FLAMBEAU_MIXED_BATCH=1 doesn't have anything to engage with.

Env:
  BENCH_URL    (default http://localhost:8080)
  BENCH_MODEL  (default Qwen3.6-35B-A3B-Q4_0)
  BENCH_REPS   (default 5)
  BENCH_TG     (default 128)
"""
import json, os, time, statistics, urllib.request

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-35B-A3B-Q4_0")
REPS = int(os.environ.get("BENCH_REPS", "5"))
TG = int(os.environ.get("BENCH_TG", "128"))

PROMPT_BASE = (
    "You are a senior compiler engineer at a large GPU vendor. "
    "Write an exhaustive technical explanation that covers, in order: "
    "(1) how modern AMD GPUs of the CDNA-1 / gfx906 generation implement "
    "integer dot-product acceleration via the V_DOT4_I32_I8 instruction. "
    "(2) Walk through the per-VGPR register pressure implications when this "
    "instruction is fused into a quantized matrix-vector multiplication "
    "kernel, including the impact on waves-per-EU occupancy. "
    "(3) Describe the practical role of dp4a in Q4_0 and Q8_0 mmvq kernels. "
    "(4) Compare CDNA-1 dp4a with NVIDIA's __dp4a intrinsic. "
    "(5) Cover memory-system implications: LDS bank layout, L2 prefetcher, "
    "typical MemBusy / MemStall / VALUBusy profile in the bandwidth-bound "
    "regime. "
    "(6) End with practical kernel-author advice. Be exhaustive."
)


def parse_sse(line):
    if not line.startswith(b"data: "):
        return None
    payload = line[len(b"data: "):].strip()
    if payload == b"[DONE]":
        return "__done__"
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    choices = obj.get("choices") or []
    if not choices:
        return None
    return (choices[0].get("delta") or {}).get("content")


def fire_one():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_BASE}],
        "max_tokens": TG,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None
    n_tok = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            c = parse_sse(raw.rstrip())
            if c == "__done__":
                break
            if c is None or c == "":
                continue
            if t_first is None:
                t_first = time.monotonic()
            n_tok += 1
    wall = time.monotonic() - t0
    ttft = t_first - t0 if t_first is not None else None
    decode_tps = n_tok / (wall - ttft) if (ttft is not None and wall > ttft) else 0.0
    return {"ttft_ms": ttft * 1000, "wall_s": wall, "n_tok": n_tok, "decode_tps": decode_tps}


def main():
    print(f"BENCH-single: model={MODEL} reps={REPS} tg={TG}")
    runs = []
    for i in range(REPS):
        r = fire_one()
        runs.append(r)
        print(f"  rep {i}: TTFT={r['ttft_ms']:.0f}ms wall={r['wall_s']:.2f}s "
              f"n_tok={r['n_tok']} decode={r['decode_tps']:.2f} tok/s")

    print()
    print(f"  median TTFT:      {statistics.median(r['ttft_ms'] for r in runs):.0f} ms")
    print(f"  median wall:      {statistics.median(r['wall_s'] for r in runs):.2f} s")
    print(f"  median decode:    {statistics.median(r['decode_tps'] for r in runs):.2f} tok/s")
    print(f"  min/max decode:   {min(r['decode_tps'] for r in runs):.2f} / {max(r['decode_tps'] for r in runs):.2f} tok/s")


if __name__ == "__main__":
    main()
