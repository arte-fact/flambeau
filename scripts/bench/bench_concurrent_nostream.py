#!/usr/bin/env python3
"""Non-streaming concurrent bench: N parallel /v1/chat/completions calls
with stream=false. Required to exercise flambeau's scheduler-batched
decode path — the streaming path falls back to single-slot dispatch_decode_one
and never reaches `Session::forward_decode_batched`. Use to measure
aggregate decode tok/s under N>1 with the batched-slots GDN/MMVQ path.
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
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body_bytes = resp.read()
    except Exception as e:
        results[stream_id] = {"error": str(e)}
        return
    total = time.monotonic() - t0
    try:
        obj = json.loads(body_bytes)
        n = obj.get("usage", {}).get("completion_tokens") or 0
        first_chars = (obj["choices"][0]["message"]["content"] or "")[:80]
    except Exception as e:
        results[stream_id] = {"error": f"parse: {e}"}
        return
    results[stream_id] = {
        "n": n,
        "total_s": total,
        "preview": first_chars,
    }


def main():
    print(f"BENCH-nostream: N={N} concurrent /chat/completions stream=false, max_tokens={MAX_TOKENS}, model={MODEL}")
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

    print(f"\n=== Results ({len(ok)} streams completed, {len(failed)} failed):")
    print(f"  wall time:         {wall:.2f} s")
    print(f"  total tokens:      {total_tokens}")
    print(f"  aggregate decode:  {aggregate_tps:.2f} tok/s   (sum across all streams)")
    if per_stream_tps:
        print(f"  per-stream decode: median={statistics.median(per_stream_tps):.2f}  "
              f"mean={statistics.mean(per_stream_tps):.2f}  "
              f"min={min(per_stream_tps):.2f}  max={max(per_stream_tps):.2f} tok/s")
    for i, r in enumerate(ok):
        print(f"  stream {i} preview: {r['preview']!r}")


if __name__ == "__main__":
    main()
