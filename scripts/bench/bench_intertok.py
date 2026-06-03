#!/usr/bin/env python3
"""Per-token latency on a single long stream — direct decode-rate sanity check.

Reads ${BENCH_MODEL:-Qwen3.6-27B-Q4_0} from ${BENCH_URL:-localhost:8080}.
"""
import json, os, time, urllib.request, statistics

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")

body = json.dumps({
    "model": MODEL,
    "messages": [{"role":"user","content":
        "Write a very long, exhaustive, technical essay (about 1000 words) on "
        "the design of LSM-tree storage engines. Cover compaction strategies, "
        "write amplification, read amplification, space amplification, bloom "
        "filters, level-tiered vs size-tiered, RocksDB-specific tunables, and "
        "the cost model behind compaction choices. Be exhaustive."}],
    "max_tokens": 512,
    "temperature": 0.0,
    "stream": True,
}).encode()
req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
t0 = time.monotonic()
ttft = None
prev = None
intervals_ms = []
n = 0
prompt_tokens = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if not line.startswith("data: "): continue
        payload = line[6:]
        if payload == "[DONE]": break
        try: obj = json.loads(payload)
        except: continue
        now = time.monotonic()
        if obj.get("usage") and obj["usage"].get("prompt_tokens"):
            prompt_tokens = obj["usage"]["prompt_tokens"]
        choices = obj.get("choices") or []
        if not choices: continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            if ttft is None:
                ttft = now - t0; prev = now
            else:
                intervals_ms.append((now - prev) * 1000); prev = now
            n += 1
total = time.monotonic() - t0
if intervals_ms:
    median = statistics.median(intervals_ms)
    s = sorted(intervals_ms)
    p10 = s[int(0.10*len(s))]
    p90 = s[int(0.90*len(s))]
    mean = statistics.mean(intervals_ms)
    print(f"prompt_tokens={prompt_tokens}  tokens_emitted={n}  ttft={ttft*1000:.1f} ms  total={total:.2f}s")
    print(f"inter-token ms  p10={p10:.1f}  median={median:.1f}  mean={mean:.1f}  p90={p90:.1f}")
    print(f"decode_rate from median = {1000.0/median:.2f} tok/s")
    print(f"decode_rate from mean   = {1000.0/mean:.2f} tok/s")
    print(f"decode_rate over (n-1)/(total-ttft) = {(n-1)/(total-ttft):.2f} tok/s")
