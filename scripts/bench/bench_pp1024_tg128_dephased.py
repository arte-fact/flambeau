#!/usr/bin/env python3
"""Dephased pp~1024 / tg=128. Same ~1024-token prompt as the
concurrent variant but staggers stream submissions by STAGGER_S so
each new prefill arrives while older streams are mid-decode — the
K4c mixed-batch engagement regime. Per-consumer + cumulated.

Env:
  BENCH_URL    (default http://localhost:8080)
  BENCH_MODEL  (default Qwen3.6-27B-Q4_0)
  BENCH_N      (default 4)  — concurrent streams
  BENCH_TG     (default 128) — max_tokens per stream
  BENCH_STAGGER_S (default 5.0) — gap between successive submissions
"""
import json, os, time, threading, urllib.request, statistics

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")
N = int(os.environ.get("BENCH_N", "4"))
TG = int(os.environ.get("BENCH_TG", "128"))
STAGGER_S = float(os.environ.get("BENCH_STAGGER_S", "5.0"))

PROMPT_BASE = (
    "You are a senior compiler engineer at a large GPU vendor. "
    "Write an exhaustive technical explanation that covers, in order: "
    "(1) how modern AMD GPUs of the CDNA-1 / gfx906 generation implement "
    "integer dot-product acceleration via the V_DOT4_I32_I8 instruction. "
    "Cover the instruction's binary encoding, the sign-extension semantics "
    "for each int8 lane, the operand layout, and the per-cycle throughput "
    "characteristics on a wave64 SIMD. "
    "(2) Walk through the per-VGPR register pressure implications when this "
    "instruction is fused into a quantized matrix-vector multiplication "
    "kernel, including the impact on waves-per-EU occupancy at varying "
    "block tile shapes (32-element, 64-element, 128-element K tiles), and "
    "explain why crossing the 32-VGPR boundary forces a drop from 10 to 8 "
    "waves per SIMD. "
    "(3) Describe the practical role of dp4a in Q4_0 and Q8_0 mmvq kernels, "
    "with concrete numerics: how the per-block scale factor is folded into "
    "the accumulation, why this typically reduces the per-element FMA count "
    "by roughly 4x, and why the unsigned-Q4 case still needs the -8 offset "
    "correction. "
    "(4) Compare CDNA-1 dp4a with NVIDIA's __dp4a intrinsic on sm_75 (T4) "
    "and sm_80 (A100): operand types, throughput, and effective TOPS per "
    "SIMD/SM under typical mmvq launch configurations. "
    "(5) Cover memory-system implications: how the LDS bank layout interacts "
    "with the per-warp activation strip in a row-tiled MMVQ kernel; how the "
    "L2 prefetcher behaves on the Q-quant fetch pattern; and what a typical "
    "MemBusy / MemStall / VALUBusy profile looks like in the bandwidth-bound "
    "regime. "
    "(6) End with practical kernel-author advice: which of these factors "
    "matter most when tuning a new mmvq kernel for gfx906, what the typical "
    "ceiling is, and how to diagnose stalls with rocprofv3. "
    "Be exhaustive, do not skip steps, and use concrete instruction encodings "
    "wherever appropriate."
)


def parse_sse_line(line: bytes):
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


def fire(idx, results, t_ref, delay_s):
    if delay_s > 0:
        time.sleep(delay_s)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_BASE}],
        "max_tokens": TG,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t_sub = time.monotonic() - t_ref
    t0 = time.monotonic()
    t_first = None
    n_tok = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                c = parse_sse_line(raw.rstrip())
                if c == "__done__":
                    break
                if c is None or c == "":
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                n_tok += 1
    except Exception as e:
        results[idx] = {"submit_s": t_sub, "error": str(e)}
        return
    wall = time.monotonic() - t0
    ttft = (t_first - t0) if t_first is not None else None
    decode_tps = n_tok / (wall - ttft) if (ttft is not None and wall > ttft) else 0.0
    results[idx] = {
        "submit_s": t_sub,
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "wall_s": wall,
        "n_tok": n_tok,
        "decode_tps": decode_tps,
    }


def main():
    print(f"BENCH-pp1024-tg128-dephased: N={N} tg={TG} stagger={STAGGER_S}s model={MODEL}")
    results = [None] * N
    t_ref = time.monotonic()
    threads = [
        threading.Thread(target=fire, args=(i, results, t_ref, i * STAGGER_S), name=f"c{i}")
        for i in range(N)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    total_wall = time.monotonic() - t_ref

    ok = [r for r in results if r and "error" not in r]
    err = [r for r in results if r and "error" in r]
    if not ok:
        print("all streams failed")
        for r in err:
            print(f"  ERROR: {r['error']}")
        return

    print()
    print(f"=== total wall={total_wall:.2f}s  N_ok={len(ok)}/{N}  errors={len(err)}")
    print()
    print("per-consumer:")
    print(f"  {'#':>3}  {'sub_s':>6}  {'TTFT (ms)':>10}  {'wall (s)':>9}  {'n_tok':>6}  {'decode tok/s':>13}")
    for i, r in enumerate(results):
        if r is None or "error" in r:
            print(f"  {i:>3}  {r}")
        else:
            ttft = f"{r['ttft_ms']:.0f}" if r["ttft_ms"] is not None else "-"
            print(f"  {i:>3}  {r['submit_s']:>6.2f}  {ttft:>10}  {r['wall_s']:>9.2f}  {r['n_tok']:>6}  {r['decode_tps']:>13.2f}")

    def stats(label, vals, unit=""):
        if not vals:
            print(f"  {label}: -")
            return
        s = sorted(vals)
        p50 = statistics.median(vals)
        idx99 = int(0.99 * (len(s) - 1))
        print(f"  {label}: min={min(vals):.2f}  p50={p50:.2f}  p99={s[idx99]:.2f}  max={max(vals):.2f}  {unit}")

    print()
    print("per-consumer distribution:")
    stats("  TTFT", [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None], "ms")
    stats("  wall", [r["wall_s"] for r in ok], "s")
    stats("  decode", [r["decode_tps"] for r in ok], "tok/s")

    total_tok = sum(r["n_tok"] for r in ok)
    sum_decode_tps = sum(r["decode_tps"] for r in ok)
    median_wall = statistics.median([r["wall_s"] for r in ok])
    print()
    print("CUMULATED:")
    print(f"  total decode tokens delivered:  {total_tok}")
    print(f"  cumulated decode throughput:    {sum_decode_tps:.2f} tok/s  (sum of per-stream)")
    print(f"  median per-stream wall:         {median_wall:.2f} s")
    print(f"  total experiment wall:          {total_wall:.2f} s")
    print(f"  effective decode tps over wall: {total_tok / total_wall:.2f} tok/s")


if __name__ == "__main__":
    main()
