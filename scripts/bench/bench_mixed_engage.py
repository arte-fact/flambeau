#!/usr/bin/env python3
"""K4c-focused bench: trigger the mixed-batch engagement pattern.

Fires N_SHORT short streaming requests at t=0 (they prefill in <1s,
then enter steady decode). At t=DELAY_S (when shorts are mid-decode)
fires one long-prefill streaming request. With FLAMBEAU_MIXED_BATCH=1
on a supported arch, the long's per-chunk prefill should pick up the
N short decode pendings and fire `Model::forward_mixed_decode`,
delivering per-chunk overlap.

Report per-request TTFT and the long's wall, plus the short
inter-token p99 during the long's prefill window — that's where
mixed-batch's overlap shows up as fewer decode-step stalls.
"""
import json, os, time, threading, urllib.request, statistics, sys

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q4_1")
N_SHORT = int(os.environ.get("BENCH_N_SHORT", "3"))
DELAY_S = float(os.environ.get("BENCH_DELAY_S", "3.0"))
SHORT_MAX_TOKENS = int(os.environ.get("BENCH_SHORT_MAX_TOKENS", "200"))
LONG_MAX_TOKENS = int(os.environ.get("BENCH_LONG_MAX_TOKENS", "32"))

LONG_BODY = (
    "Below is a fragment of background reading. Please use it to answer the question at the end.\n\n"
    + ("Compilers convert source code to executable form. They lex, parse, type-check, optimize, "
       "and emit code. Each phase has been studied for decades. The history starts with Algol-60, "
       "moves through Fortran and Pascal, then to C and C++, and arrives at modern LLVM-backed "
       "languages. ") * 20
    + "\n\nQuestion: in one paragraph, summarise the role of register allocation."
)
SHORT_PROMPTS = [
    "Count slowly from 1 to 50 with brief commentary on each number.",
    "Describe how a CPU branch predictor works in detail.",
    "Explain what a typical compiler does, phase by phase, in detail.",
    "Walk through how PageRank works step by step, in detail.",
]


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


def run_one(idx, prompt, results, t_ref, delay_s, max_tokens):
    if delay_s > 0:
        time.sleep(delay_s)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t_sub = time.monotonic() - t_ref
    t_req = time.monotonic()
    times = []
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                c = parse_sse_line(raw.rstrip())
                if c == "__done__":
                    break
                if c is None or c == "":
                    continue
                times.append(time.monotonic())
                chunks.append(c)
    except Exception as e:
        results[idx] = {"error": str(e), "submit_s": t_sub}
        return
    if not times:
        results[idx] = {"error": "no tokens", "submit_s": t_sub}
        return
    ttft = (times[0] - t_req) * 1000.0
    wall = (time.monotonic() - t_req) * 1000.0
    inter = [(times[i] - times[i-1]) * 1000.0 for i in range(1, len(times))]
    inter_with_t = [(times[i], (times[i] - times[i-1]) * 1000.0) for i in range(1, len(times))]
    results[idx] = {
        "submit_s": t_sub,
        "ttft_ms": ttft,
        "wall_ms": wall,
        "n_tok": len(times),
        "inter_ms": inter,
        "inter_with_t": inter_with_t,
        "preview": ("".join(chunks))[:60],
        "first_t_abs": times[0],
        "last_t_abs": times[-1],
    }


def summarise(label, vals, unit="ms"):
    if not vals:
        print(f"  {label}: no samples")
        return
    s = sorted(vals)
    print(f"  {label}: n={len(vals)}  min={min(vals):.0f}  p50={statistics.median(vals):.0f}  "
          f"p99={s[int(0.99*(len(s)-1))]:.0f}  max={max(vals):.0f}  ({unit})")


def main():
    print(f"BENCH-mixed-engage: short={N_SHORT} delay={DELAY_S}s "
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
    print("--- TTFT ---")
    short_ttft = [results[j]["ttft_ms"] for j in range(N_SHORT)
                  if results[j] and "error" not in results[j]]
    long_ttft = [results[long_idx]["ttft_ms"]] if results[long_idx] and "error" not in results[long_idx] else []
    summarise("short ttft", short_ttft)
    summarise("long  ttft", long_ttft)

    # The mixed-engage window: short inter-token times that occur DURING
    # the long request's "prefill" phase (from long submit → long TTFT).
    if long_ttft:
        long_sub = results[long_idx]["submit_s"]
        long_first_abs = results[long_idx]["first_t_abs"]
        # Window covers `[t_start + long_sub, long_first_abs)`.
        win_start = t_start + long_sub
        win_end = long_first_abs
        print(f"--- short inter-token during long's prefill window "
              f"[{(win_start - t_start)*1000:.0f}..{(win_end - t_start)*1000:.0f}ms] ---")
        in_window = []
        for j in range(N_SHORT):
            r = results[j]
            if not r or "error" in r:
                continue
            for (t_abs, ms) in r["inter_with_t"]:
                if win_start <= t_abs < win_end:
                    in_window.append(ms)
        summarise("short inter (in window)", in_window)

    print()
    for i in range(n_total):
        kind = "long" if i == long_idx else "short"
        r = results[i]
        if r is None:
            print(f"  [{i}] {kind}: no result")
        elif "error" in r:
            print(f"  [{i}] {kind}: ERROR {r['error']}")
        else:
            print(f"  [{i}] {kind}: sub=+{r['submit_s']*1000:.0f}ms "
                  f"ttft={r['ttft_ms']:.0f}ms wall={r['wall_ms']:.0f}ms "
                  f"n_tok={r['n_tok']}")


if __name__ == "__main__":
    main()
