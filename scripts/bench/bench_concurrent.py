#!/usr/bin/env python3
"""Fan out N concurrent streaming requests; report aggregate decode tok/s and
per-stream median inter-token latency. Use to measure pp/tp concurrent
throughput (the win you can't see at inflight_slots=1).
"""
import json, os, sys, time, threading, urllib.request, statistics

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")
N = int(os.environ.get("BENCH_N", "4"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "256"))

PROMPTS = [
    "Write a 500-word essay on the history of compilers, from Algol-60 to LLVM. Cover lexers, parsers, optimization passes, register allocation.",
    "Explain how RocksDB's level-tiered compaction works. Cover bloom filters, write amplification, and the L0->L1 burst problem.",
    "Describe the AMD CDNA architecture (gfx906): wave64, LDS layout, dp4a, MFMA, register pressure tradeoffs. Be technical.",
    "Walk through how a CPU branch predictor like TAGE evolves over a workload. Cover history tables, tag matching, and useful-bit aging.",
    "Explain the linker's job: relocation, symbol resolution, GOT/PLT, and how lazy binding works.",
    "Write a technical essay on PCIe Gen4 vs Gen3: signal integrity, equalization, and why long traces matter.",
    "Cover the design of LSM-tree storage engines — compaction, write/read/space amplification, RocksDB-specific tunables.",
    "Explain CUDA/HIP graph capture: when it helps, what it costs, and how launch overhead amortizes.",
]


def run_one(stream_id, prompt, results):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0 = time.monotonic()
    ttft = None
    prev = None
    intervals_ms = []
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line.startswith("data: "): continue
                payload = line[6:]
                if payload == "[DONE]": break
                try: obj = json.loads(payload)
                except: continue
                now = time.monotonic()
                choices = obj.get("choices") or []
                if not choices: continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = now - t0; prev = now
                    else:
                        intervals_ms.append((now - prev) * 1000); prev = now
                    n += 1
    except Exception as e:
        results[stream_id] = {"error": str(e)}
        return
    total = time.monotonic() - t0
    results[stream_id] = {
        "n": n,
        "ttft_ms": ttft * 1000 if ttft else None,
        "total_s": total,
        "intervals_ms": intervals_ms,
    }


def main():
    print(f"BENCH: N={N} concurrent streams, max_tokens={MAX_TOKENS}, model={MODEL}")
    results = [None] * N
    threads = []
    t_start = time.monotonic()
    for i in range(N):
        prompt = PROMPTS[i % len(PROMPTS)]
        t = threading.Thread(target=run_one, args=(i, prompt, results))
        threads.append(t); t.start()
    for t in threads:
        t.join()
    t_end = time.monotonic()
    wall = t_end - t_start

    ok = [r for r in results if r and "error" not in r]
    failed = [r for r in results if r and "error" in r]
    if failed:
        for r in failed: print(f"  ERROR: {r['error']}")

    total_tokens = sum(r["n"] for r in ok)
    aggregate_tps = total_tokens / wall
    per_stream_tps = [r["n"] / r["total_s"] for r in ok]
    per_stream_ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    flat_intervals = [iv for r in ok for iv in r["intervals_ms"]]

    print(f"\n=== Results ({len(ok)} streams completed, {len(failed)} failed):")
    print(f"  wall time:         {wall:.2f} s")
    print(f"  total tokens:      {total_tokens}")
    print(f"  aggregate decode:  {aggregate_tps:.2f} tok/s   (sum across all streams)")
    if per_stream_tps:
        print(f"  per-stream decode: median={statistics.median(per_stream_tps):.2f}  "
              f"mean={statistics.mean(per_stream_tps):.2f}  "
              f"min={min(per_stream_tps):.2f}  max={max(per_stream_tps):.2f} tok/s")
    if per_stream_ttft:
        print(f"  per-stream TTFT:   median={statistics.median(per_stream_ttft):.0f}ms  "
              f"max={max(per_stream_ttft):.0f}ms")
    if flat_intervals:
        s = sorted(flat_intervals)
        print(f"  inter-token ms:    p50={statistics.median(flat_intervals):.1f}  "
              f"p90={s[int(0.9*len(s))]:.1f}  "
              f"p99={s[int(0.99*len(s))]:.1f}")


if __name__ == "__main__":
    main()
