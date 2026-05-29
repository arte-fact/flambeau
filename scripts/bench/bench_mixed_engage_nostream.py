#!/usr/bin/env python3
"""K4c-focused bench (non-streaming). The scheduler path only fills
`batched_pending` from `decode_via_scheduler_into`, which fires from
non-streaming greedy requests. With FLAMBEAU_MIXED_BATCH=1 on a
supported arch, the long's chunked prefill should engage mixed
once the shorts are mid-decode.

Compare total wall and per-request wall vs the same config without
FLAMBEAU_MIXED_BATCH.
"""
import json, os, time, threading, urllib.request, sys

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q4_1")
N_SHORT = int(os.environ.get("BENCH_N_SHORT", "3"))
DELAY_S = float(os.environ.get("BENCH_DELAY_S", "3.0"))
SHORT_MAX_TOKENS = int(os.environ.get("BENCH_SHORT_MAX_TOKENS", "200"))
LONG_MAX_TOKENS = int(os.environ.get("BENCH_LONG_MAX_TOKENS", "32"))

LONG_BODY = (
    "Below is a fragment of background reading. Please use it to answer the question.\n\n"
    + ("Compilers convert source code to executable form. They lex, parse, type-check, optimize, "
       "and emit code. Register allocation is a critical phase. ") * 30
    + "\n\nQuestion: in one paragraph, summarise the role of register allocation."
)
SHORT_PROMPTS = [
    "Count slowly from 1 to 50 with brief commentary on each number.",
    "Describe how a CPU branch predictor works in detail.",
    "Explain what a typical compiler does, phase by phase, in detail.",
    "Walk through how PageRank works step by step, in detail.",
]


def run_one(idx, prompt, results, t_ref, delay_s, max_tokens):
    if delay_s > 0:
        time.sleep(delay_s)
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
        with urllib.request.urlopen(req, timeout=600) as resp:
            body_bytes = resp.read()
    except Exception as e:
        results[idx] = {"error": str(e), "submit_s": t_sub}
        return
    wall = time.monotonic() - t_req
    try:
        obj = json.loads(body_bytes)
        n_tok = obj.get("usage", {}).get("completion_tokens", 0)
        preview = (obj["choices"][0]["message"]["content"] or "")[:50]
    except Exception as e:
        results[idx] = {"error": f"parse: {e}", "submit_s": t_sub}
        return
    results[idx] = {
        "submit_s": t_sub,
        "wall_s": wall,
        "n_tok": n_tok,
        "preview": preview,
    }


def main():
    print(f"BENCH-mixed-engage-nostream: short={N_SHORT} delay={DELAY_S}s "
          f"short_max={SHORT_MAX_TOKENS} long_max={LONG_MAX_TOKENS} model={MODEL}")
    n_total = N_SHORT + 1
    results = [None] * n_total
    threads = []
    t_start = time.monotonic()
    for j in range(N_SHORT):
        prompt = SHORT_PROMPTS[j % len(SHORT_PROMPTS)]
        t = threading.Thread(
            target=run_one,
            args=(j, prompt, results, t_start, 0.0, SHORT_MAX_TOKENS),
            name=f"short-{j}",
        )
        threads.append(t)
        t.start()
    long_idx = N_SHORT
    t_long = threading.Thread(
        target=run_one,
        args=(long_idx, LONG_BODY, results, t_start, DELAY_S, LONG_MAX_TOKENS),
        name="long",
    )
    threads.append(t_long)
    t_long.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start

    print(f"\n=== total wall={wall:.2f}s")
    for i in range(n_total):
        kind = "long" if i == long_idx else "short"
        r = results[i]
        if r is None:
            print(f"  [{i}] {kind}: no result")
        elif "error" in r:
            print(f"  [{i}] {kind}: ERROR {r['error']}")
        else:
            print(f"  [{i}] {kind}: sub=+{r['submit_s']*1000:.0f}ms "
                  f"wall={r['wall_s']*1000:.0f}ms n_tok={r['n_tok']}  '{r['preview']}'")


if __name__ == "__main__":
    main()
