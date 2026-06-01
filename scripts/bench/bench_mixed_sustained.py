#!/usr/bin/env python3
"""Phase K5b — sustained-traffic cert for the Sarathi-Serve mixed-batch
path. Open-loop request injection at a fixed inter-arrival interval,
mix of short and long prompts; runs for `DURATION_S` seconds then
collects in-flight requests and computes aggregate output tokens/sec.

The mixed-batch win shows up here because every long-prompt arrival
that lands while shorts are decoding gets its prefill chunks combined
with the active decodes (Phase K4c) — versus the baseline where
prefill and decode never share a kernel launch.

Run pair with `FLAMBEAU_MIXED_BATCH=1` and unset for the comparison.

Env knobs:
  BENCH_URL         (default http://localhost:8080)
  BENCH_MODEL       (default Qwen3.5-9B-Q4_1)
  BENCH_DURATION_S  (default 30) — open-loop injection window
  BENCH_ARRIVAL_S   (default 1.0) — interval between request submissions
  BENCH_LONG_RATIO  (default 0.33) — fraction of arrivals that are long
  BENCH_SHORT_TOKENS (default 80)
  BENCH_LONG_TOKENS  (default 40)
  BENCH_TIMEOUT_S    (default 120) — per-request hard cap
"""
import json, os, sys, time, threading, urllib.request, statistics

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q4_1")
DURATION_S = float(os.environ.get("BENCH_DURATION_S", "30"))
ARRIVAL_S = float(os.environ.get("BENCH_ARRIVAL_S", "1.0"))
LONG_RATIO = float(os.environ.get("BENCH_LONG_RATIO", "0.33"))
SHORT_TOKENS = int(os.environ.get("BENCH_SHORT_TOKENS", "80"))
LONG_TOKENS = int(os.environ.get("BENCH_LONG_TOKENS", "40"))
TIMEOUT_S = float(os.environ.get("BENCH_TIMEOUT_S", "120"))

LONG_BODY = (
    "Below is a long fragment of background reading to summarise.\n\n"
    + ("Compilers convert source code to executable form. They lex, parse, "
       "type-check, optimize, and emit code. Register allocation is critical. "
       "RocksDB is an LSM-tree storage engine. PCIe Gen4 doubles Gen3 bandwidth. ") * 40
    + "\n\nWrite a single paragraph summary."
)
SHORT_PROMPTS = [
    "Count from 1 to 30 with brief notes on each number.",
    "Describe how a CPU branch predictor works.",
    "Explain what a typical compiler does in detail.",
    "Walk through how PageRank works step by step.",
    "Name three classic sort algorithms with their complexities.",
    "Explain TCP three-way handshake.",
]


def fire(idx, prompt, kind, max_tokens, results, t_ref):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t_sub = time.monotonic() - t_ref
    t_req = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body_bytes = resp.read()
    except Exception as e:
        results[idx] = {"kind": kind, "submit_s": t_sub, "error": str(e)}
        return
    wall_s = time.monotonic() - t_req
    try:
        obj = json.loads(body_bytes)
        n_tok = obj.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        results[idx] = {"kind": kind, "submit_s": t_sub, "error": f"parse: {e}"}
        return
    results[idx] = {
        "kind": kind,
        "submit_s": t_sub,
        "wall_s": wall_s,
        "n_tok": n_tok,
    }


def main():
    print(f"BENCH-mixed-sustained: model={MODEL} duration={DURATION_S}s "
          f"arrival={ARRIVAL_S}s long_ratio={LONG_RATIO}")
    print(f"  short_max_tokens={SHORT_TOKENS} long_max_tokens={LONG_TOKENS}")

    results = {}
    threads = []
    t_start = time.monotonic()
    short_round = 0
    long_round = 0
    next_arrival = 0.0
    while True:
        now = time.monotonic() - t_start
        if now >= DURATION_S:
            break
        if now < next_arrival:
            time.sleep(next_arrival - now)
            continue
        idx = len(threads)
        # Pseudo-deterministic mix: cycle through using arrival count.
        is_long = ((idx + 1) * 100) % 1000 < int(LONG_RATIO * 1000)
        if is_long:
            prompt = LONG_BODY
            kind = "long"
            max_tokens = LONG_TOKENS
            long_round += 1
        else:
            prompt = SHORT_PROMPTS[short_round % len(SHORT_PROMPTS)]
            kind = "short"
            max_tokens = SHORT_TOKENS
            short_round += 1
        t = threading.Thread(
            target=fire,
            args=(idx, prompt, kind, max_tokens, results, t_start),
            name=f"{kind}-{idx}",
        )
        threads.append(t)
        t.start()
        next_arrival += ARRIVAL_S
    inject_wall = time.monotonic() - t_start
    print(f"  injection complete at t={inject_wall:.1f}s "
          f"(N_fired={len(threads)}: {short_round} short + {long_round} long); "
          f"awaiting completions...")

    for t in threads:
        t.join()
    total_wall = time.monotonic() - t_start

    ok = [r for r in results.values() if r and "error" not in r]
    err = [r for r in results.values() if r and "error" in r]
    short = [r for r in ok if r["kind"] == "short"]
    longs = [r for r in ok if r["kind"] == "long"]
    n_short = len(short)
    n_long = len(longs)
    tok_short = sum(r["n_tok"] for r in short)
    tok_long = sum(r["n_tok"] for r in longs)
    tok_total = tok_short + tok_long

    print()
    print(f"=== total wall={total_wall:.2f}s (inject={inject_wall:.2f}s + tail)")
    print(f"  completed: short={n_short}/{short_round}  long={n_long}/{long_round}  errors={len(err)}")
    print(f"  output tokens: short={tok_short}  long={tok_long}  total={tok_total}")
    print(f"  aggregate over wall: {tok_total / total_wall:.1f} tok/s")
    print(f"  aggregate over injection window: {tok_total / inject_wall:.1f} tok/s")

    def stats(label, vals):
        if not vals:
            print(f"  {label}: no samples")
            return
        s = sorted(vals)
        p50 = statistics.median(vals)
        p99 = s[int(0.99 * (len(s) - 1))]
        print(f"  {label}: n={len(vals)}  min={min(vals):.0f}  p50={p50:.0f}  "
              f"p99={p99:.0f}  max={max(vals):.0f}  (ms)")

    print()
    print("--- per-request wall ---")
    stats("short wall", [r["wall_s"] * 1000 for r in short])
    stats("long  wall", [r["wall_s"] * 1000 for r in longs])

    if err:
        print(f"\n{len(err)} errors:")
        for r in err[:5]:
            print(f"  [{r['kind']}] +{r['submit_s']*1000:.0f}ms: {r['error']}")


if __name__ == "__main__":
    main()
