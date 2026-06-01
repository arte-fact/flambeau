#!/usr/bin/env python3
"""Mixed-workload bench for Sarathi-Serve scheduler validation.

Fires N concurrent requests: half with long prompts (heavy prefill),
half with short prompts that arrive ~staggered. Reports TTFT
distribution. The Sarathi-Serve win shows up as bounded p99 TTFT
for the short requests — they don't have to wait for the long
prefills to complete before getting their first token.

Tune FLAMBEAU_PREFILL_CHUNK_TOKENS server-side to see the trade-off.
"""
import json, os, time, threading, urllib.request, statistics, sys

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")
N_LONG = int(os.environ.get("BENCH_N_LONG", "2"))
N_SHORT = int(os.environ.get("BENCH_N_SHORT", "2"))
SHORT_DELAY_S = float(os.environ.get("BENCH_SHORT_DELAY_S", "0.5"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "64"))


LONG_BODY = (
    "Below is a fragment of background reading. Please use it to answer the question at the end.\n\n"
    + ("Compilers convert source code to executable form. They lex, parse, type-check, optimize, "
       "and emit code. Each phase has been studied for decades. The history starts with Algol-60, "
       "moves through Fortran and Pascal, then to C and C++, and arrives at modern LLVM-backed "
       "languages. Lexical analysis groups characters into tokens. Parsing builds the AST. "
       "Type checking enforces the language's static guarantees. Optimization rewrites IR for "
       "speed or size. Register allocation maps virtuals to physicals. Instruction selection "
       "picks the target encoding. Linking resolves cross-module references. ") * 12
    + "\n\nQuestion: in one paragraph, summarise the role of register allocation in a modern compiler."
)
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Explain Big-O notation in two sentences.",
    "Name three classic sort algorithms.",
    "What does TCP stand for?",
]


def run_one(idx: int, prompt: str, results: list, t_start_ref: float, delay_s: float):
    if delay_s > 0:
        time.sleep(delay_s)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t_submit = time.monotonic() - t_start_ref
    t_request = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body_bytes = resp.read()
    except Exception as e:
        results[idx] = {"error": str(e), "kind": "?", "submit_s": t_submit}
        return
    wall = time.monotonic() - t_request
    try:
        obj = json.loads(body_bytes)
        n = obj.get("usage", {}).get("completion_tokens") or 0
    except Exception as e:
        results[idx] = {"error": f"parse: {e}", "submit_s": t_submit}
        return
    # Non-streaming: TTFT == request wall (no intra-response timing).
    # The bench is still useful for measuring "did short prompt
    # complete before long prompts blocked it" — short requests
    # should return earlier than long ones if Sarathi-style chunking
    # interleaved their decode with the long prefills.
    results[idx] = {
        "submit_s": t_submit,
        "wall_s": wall,
        "n_tokens": n,
    }


def main():
    n_total = N_LONG + N_SHORT
    print(f"BENCH-mixed-chat: long={N_LONG} short={N_SHORT} short_delay={SHORT_DELAY_S}s "
          f"max_tokens={MAX_TOKENS} model={MODEL}")
    results: list = [None] * n_total
    threads = []
    t_start = time.monotonic()
    # Long requests fire immediately (idx 0..N_LONG).
    for i in range(N_LONG):
        t = threading.Thread(
            target=run_one,
            args=(i, LONG_BODY, results, t_start, 0.0),
            name=f"long-{i}",
        )
        threads.append(t)
        t.start()
    # Short requests fire after SHORT_DELAY_S to land mid-prefill (idx N_LONG..N_LONG+N_SHORT).
    for j in range(N_SHORT):
        idx = N_LONG + j
        prompt = SHORT_PROMPTS[j % len(SHORT_PROMPTS)]
        t = threading.Thread(
            target=run_one,
            args=(idx, prompt, results, t_start, SHORT_DELAY_S),
            name=f"short-{j}",
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start

    long_walls = [r["wall_s"] * 1000.0 for i, r in enumerate(results)
                  if i < N_LONG and r and "error" not in r]
    short_walls = [r["wall_s"] * 1000.0 for i, r in enumerate(results)
                   if i >= N_LONG and r and "error" not in r]

    print(f"\n=== total wall={wall:.2f}s")

    def summarise(label, ts):
        if not ts:
            print(f"  {label}: no samples")
            return
        s = sorted(ts)
        p50 = statistics.median(ts)
        p99 = s[int(0.99 * (len(s) - 1))]
        print(f"  {label}: n={len(ts)}  min={min(ts):.0f}  p50={p50:.0f}  "
              f"p99={p99:.0f}  max={max(ts):.0f}  (ms)")

    summarise("long  request_wall", long_walls)
    summarise("short request_wall", short_walls)

    for i, r in enumerate(results):
        kind = "long " if i < N_LONG else "short"
        if r is None:
            print(f"  [{i}] {kind}: no result")
        elif "error" in r:
            print(f"  [{i}] {kind}: ERROR {r['error']}")
        else:
            print(f"  [{i}] {kind}: submit=+{r['submit_s']*1000:.0f}ms "
                  f"wall={r['wall_s']*1000:.0f}ms tokens={r['n_tokens']}")


if __name__ == "__main__":
    main()
