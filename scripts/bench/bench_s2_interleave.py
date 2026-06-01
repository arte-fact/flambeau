#!/usr/bin/env python3
"""S2 interleave check: fire a long-prompt completion (legacy/streaming
path — gemma4 routes here since the scheduler skips when sampling != greedy
or arch != qwen3). 250 ms later fire a short-prompt request. Measure when
each starts emitting tokens.

Without S2: short request blocks until long request's full prefill +
all decode completes (or at least until prefill completes). With S2:
short request's prefill interleaves between the long request's chunks
of 512 tokens, so TTFT_short << total_long.

Run:
  BENCH_URL=http://localhost:8080 BENCH_MODEL=Qwen3.6-27B-Q4_0 \\
    python3 scripts/bench/bench_s2_interleave.py
"""
import json, os, sys, time, threading, urllib.request

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")

LONG_BLOCK = ("RocksDB is a high-performance LSM-tree key-value store. " * 200)
LONG_PROMPT = LONG_BLOCK + " Summarise the above in three bullets."
SHORT_PROMPT = "What is 2 + 2?"

def fire(stream_id, prompt, max_tokens, results, sample_temp=0.7):
    # logprobs=True forces the LEGACY path. The scheduler skips when
    # `collect_logprobs.is_some()` so this is the most reliable way to
    # exercise the chunked-prefill behaviour we just shipped.
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens,
        "temperature": sample_temp,
        "top_p": 0.95,
        "logprobs": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body_bytes = resp.read()
    except Exception as e:
        results[stream_id] = {"error": str(e), "elapsed": time.monotonic() - t0}
        return
    elapsed = time.monotonic() - t0
    try:
        obj = json.loads(body_bytes)
        n_tok = obj.get("usage", {}).get("completion_tokens", 0)
        first = (obj["choices"][0]["message"]["content"] or "")[:60]
    except Exception as e:
        results[stream_id] = {"error": f"parse: {e}", "elapsed": elapsed}
        return
    results[stream_id] = {"elapsed": elapsed, "n_tok": n_tok, "first": first}

def main():
    # sample_temp=0.7 forces the LEGACY path (scheduler engages only on greedy).
    results = {}
    t0 = time.monotonic()
    t_long = threading.Thread(target=fire,
        args=("long", LONG_PROMPT, 64, results, 0.7))
    t_short = threading.Thread(target=fire,
        args=("short", SHORT_PROMPT, 32, results, 0.7))
    t_long.start()
    long_started_at = time.monotonic() - t0
    time.sleep(0.25)
    t_short.start()
    short_started_at = time.monotonic() - t0
    t_long.join()
    t_short.join()

    print(f"long started at {long_started_at*1000:.0f} ms, finished at "
          f"{(long_started_at + results['long']['elapsed'])*1000:.0f} ms "
          f"(elapsed {results['long']['elapsed']*1000:.0f} ms)")
    print(f"short started at {short_started_at*1000:.0f} ms, finished at "
          f"{(short_started_at + results['short']['elapsed'])*1000:.0f} ms "
          f"(elapsed {results['short']['elapsed']*1000:.0f} ms)")

    if "error" in results.get("long", {}):
        print(f"LONG ERROR: {results['long']['error']}")
        sys.exit(1)
    if "error" in results.get("short", {}):
        print(f"SHORT ERROR: {results['short']['error']}")
        sys.exit(1)

    # The S2 win signal: short finishes BEFORE long. Without S2, short
    # blocks on long's mutex until long's decode loop also exits.
    long_finished_at = long_started_at + results["long"]["elapsed"]
    short_finished_at = short_started_at + results["short"]["elapsed"]
    print()
    if short_finished_at < long_finished_at:
        gap_ms = (long_finished_at - short_finished_at) * 1000
        print(f"PASS: short finished {gap_ms:.0f} ms BEFORE long — "
              "S2 interleave working")
    else:
        gap_ms = (short_finished_at - long_finished_at) * 1000
        print(f"FAIL: short finished {gap_ms:.0f} ms AFTER long — "
              "mutex held across full request (S2 not engaged)")
        sys.exit(2)

    print(f"\nlong preview:  {results['long']['first']!r}")
    print(f"short preview: {results['short']['first']!r}")

if __name__ == "__main__":
    main()
