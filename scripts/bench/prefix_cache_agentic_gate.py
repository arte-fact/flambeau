#!/usr/bin/env python3
"""Prefix-cache agentic gate: a growing multi-turn conversation, run under
cache-off and cache-on server boots. Cache-on turns >= 2 should hit an
intermediate chunk-boundary entry and prefill only the tail (TTFT drop),
with greedy replies identical to the cache-off run (restored state must
decode exactly like recomputed state)."""
import json, os, signal, subprocess, sys, time, urllib.request

BIN = "/artefact/flambeau/target/release/flambeau"
MODEL = "/artefact/models/Qwen3.6-27B-Q8_0.gguf"
PORT = 8100

SYS = ("You are a careful, concise assistant for a software team. "
       "Answer precisely and avoid speculation. ") * 60  # ~1100 tokens
TURNS = [
    "List three programming languages, one per line.",
    "Now list three databases, one per line.",
    "Which database is best for time-series data? One short sentence.",
]


def boot(cache_on):
    args = [BIN, "serve", "--model", MODEL, "--devices", "hip:0,2,1,3",
            "--mesh-mode", "pp+tp", "--pp-size", "2", "--tp-size", "2",
            "--port", str(PORT), "--inflight-slots", "1", "--ctx-cap", "8192",
            "--kv", "q8"]
    if cache_on:
        args += ["--prefix-cache", "--prefix-cache-max-gb", "4"]
    env = dict(os.environ); env["FLAMBEAU_KV"] = "q8"
    log = open(f"/tmp/pc_agentic_{'on' if cache_on else 'off'}.log", "w")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(240):
        if p.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            pass
        time.sleep(2)
    return None


def turn(messages):
    body = json.dumps({"model": "x", "messages": messages, "temperature": 0.0,
                       "max_tokens": 40, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter(); tf = None; txt = ""; ptok = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            pl = s[5:].strip()
            if pl == "[DONE]":
                break
            o = json.loads(pl); ch = o.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if tf is None:
                    tf = time.perf_counter()
                txt += ch[0]["delta"]["content"]
            if o.get("usage"):
                ptok = o["usage"].get("prompt_tokens")
    return ptok, (tf - t0) * 1000.0, txt.strip()


def convo():
    msgs = [{"role": "system", "content": SYS}]
    rows = []
    for t in TURNS:
        msgs.append({"role": "user", "content": t})
        ptok, ttft, txt = turn(msgs)
        msgs.append({"role": "assistant", "content": txt})
        rows.append({"ptok": ptok, "ttft_ms": ttft, "reply": txt})
    return rows


def shutdown(p):
    if p and p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()


results = {}
for cache_on in (False, True):
    tag = "on" if cache_on else "off"
    p = boot(cache_on)
    if p is None:
        print(f"cache-{tag}: BOOT FAILED"); sys.exit(1)
    try:
        results[tag] = convo()
    finally:
        shutdown(p)
    print(f"=== cache {tag} ===")
    for i, r in enumerate(results[tag]):
        print(f"  turn {i+1}: ptok={r['ptok']:>5} TTFT={r['ttft_ms']:>8.1f}ms reply={r['reply'][:48]!r}")

print("\n=== gate ===")
ok = True
for i in range(len(TURNS)):
    a, b = results["off"][i], results["on"][i]
    same = a["reply"] == b["reply"]
    ok &= same
    speed = a["ttft_ms"] / b["ttft_ms"] if b["ttft_ms"] > 0 else 0
    print(f"  turn {i+1}: replies {'MATCH' if same else 'DIFFER'}  TTFT off {a['ttft_ms']:8.1f}ms / on {b['ttft_ms']:8.1f}ms = {speed:.2f}x")
    if not same:
        print(f"     off: {a['reply'][:90]!r}")
        print(f"     on : {b['reply'][:90]!r}")
print(f"\nPARITY: {'PASS' if ok else 'FAIL'}")
