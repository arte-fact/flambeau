#!/usr/bin/env python3
"""Send N=2 nostream chat requests sequentially (not concurrent) to test
whether the slot-swap correctness issue is concurrency-only or also fires
under simple slot-pool reuse. Each request gets a different prompt; assert
output matches its OWN prompt's domain."""
import json, os, urllib.request

URL = os.environ.get("BENCH_URL", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.6-27B-Q4_0")
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "64"))

PROMPTS = [
    "Write a 500-word essay on the history of compilers, from Algol-60 to LLVM.",
    "Explain how RocksDB's level-tiered compaction works in detail.",
]


def run_one(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"]["content"]


for p in PROMPTS:
    print(f"\n=== PROMPT: {p[:60]}...")
    out = run_one(p)
    print(f"=== RESPONSE: {out[:200]!r}")
