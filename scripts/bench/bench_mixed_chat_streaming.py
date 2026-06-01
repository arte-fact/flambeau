#!/usr/bin/env python3
"""Phase 5 S3 — TTFT-measuring mixed-chat bench.

Fires N_LONG long-prompt + N_SHORT short-prompt streaming completions
with `SHORT_DELAY_S` between cohorts. Captures TTFT from the first SSE
content delta and inter-token latency across the response.

The Phase 5 S2 win signal: short TTFT stays bounded under concurrent
long-prefill load. Without chunked prefill, a short request fired
mid-long-prefill blocks until the long prefill's full single-shot call
completes — its TTFT grows to (long_prefill_remaining_ms +
short_prefill_ms). With chunked prefill, short TTFT is bounded to
roughly (one_chunk_prefill_ms + short_prefill_ms).

Compare:
  FLAMBEAU_PREFILL_CHUNK_TOKENS=512  python3 scripts/bench/bench_mixed_chat_streaming.py
  FLAMBEAU_PREFILL_CHUNK_TOKENS=99999 python3 scripts/bench/bench_mixed_chat_streaming.py

(set the env on the SERVER side, not here — chunk size is read at
each prefill call.)
"""
import json, os, time, threading, urllib.request, statistics

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


def parse_sse_line(line: bytes):
    """Pull `content` out of one SSE `data: {...}` JSON line. Returns
    `None` for keepalives, `[DONE]`, role-only deltas, or anything that
    doesn't carry visible content."""
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
    delta = choices[0].get("delta") or {}
    return delta.get("content")


def run_one(idx: int, prompt: str, results: list, t_start_ref: float, delay_s: float):
    if delay_s > 0:
        time.sleep(delay_s)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t_submit = time.monotonic() - t_start_ref
    t_request = time.monotonic()
    token_times: list[float] = []
    text_chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                content = parse_sse_line(raw.rstrip())
                if content == "__done__":
                    break
                if content is None or content == "":
                    continue
                token_times.append(time.monotonic())
                text_chunks.append(content)
    except Exception as e:
        results[idx] = {"error": str(e), "submit_s": t_submit}
        return
    if not token_times:
        results[idx] = {"error": "no tokens received", "submit_s": t_submit}
        return
    ttft = (token_times[0] - t_request) * 1000.0
    wall = (time.monotonic() - t_request) * 1000.0
    inter_token_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]
    results[idx] = {
        "submit_s": t_submit,
        "ttft_ms": ttft,
        "wall_ms": wall,
        "n_tokens": len(token_times),
        "inter_token_ms": inter_token_ms,
        "preview": ("".join(text_chunks))[:80],
    }


def main():
    n_total = N_LONG + N_SHORT
    print(f"BENCH-mixed-chat-streaming: long={N_LONG} short={N_SHORT} "
          f"short_delay={SHORT_DELAY_S}s max_tokens={MAX_TOKENS} model={MODEL}")
    results: list = [None] * n_total
    threads = []
    t_start = time.monotonic()
    for i in range(N_LONG):
        t = threading.Thread(
            target=run_one,
            args=(i, LONG_BODY, results, t_start, 0.0),
            name=f"long-{i}",
        )
        threads.append(t)
        t.start()
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

    def cohort(predicate):
        return [r for i, r in enumerate(results)
                if r and "error" not in r and predicate(i)]

    long_rs = cohort(lambda i: i < N_LONG)
    short_rs = cohort(lambda i: i >= N_LONG)

    def summarise(label, vals, unit="ms"):
        if not vals:
            print(f"  {label}: no samples")
            return
        s = sorted(vals)
        p50 = statistics.median(vals)
        idx99 = int(0.99 * (len(s) - 1))
        p99 = s[idx99]
        print(f"  {label}: n={len(vals)}  min={min(vals):.0f}  p50={p50:.0f}  "
              f"p99={p99:.0f}  max={max(vals):.0f}  ({unit})")

    print(f"\n=== total wall={wall:.2f}s")
    print("--- TTFT (first SSE content delta) ---")
    summarise("long  ttft", [r["ttft_ms"] for r in long_rs])
    summarise("short ttft", [r["ttft_ms"] for r in short_rs])
    print("--- inter-token latency across response ---")
    long_it = [v for r in long_rs for v in r["inter_token_ms"]]
    short_it = [v for r in short_rs for v in r["inter_token_ms"]]
    summarise("long  inter_token", long_it)
    summarise("short inter_token", short_it)

    print()
    for i, r in enumerate(results):
        kind = "long " if i < N_LONG else "short"
        if r is None:
            print(f"  [{i}] {kind}: no result")
        elif "error" in r:
            print(f"  [{i}] {kind}: ERROR {r['error']}")
        else:
            print(f"  [{i}] {kind}: submit=+{r['submit_s']*1000:.0f}ms "
                  f"ttft={r['ttft_ms']:.0f}ms wall={r['wall_ms']:.0f}ms "
                  f"n_tok={r['n_tokens']}  '{r['preview']}'")


if __name__ == "__main__":
    main()
