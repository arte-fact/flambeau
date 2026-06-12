#!/usr/bin/env python3
"""Live-like prefix-cache A/B: a multi-turn chat whose context grows turn
by turn (the client re-sends the full transcript each turn, exactly like
OpenWebUI / a chat UI). Run identically under cache-off and cache-on boots.

Reports per-turn prompt_tokens + TTFT off/on + speedup, so the cache's win
should grow with conversation length (cache-on only prefills the new tail;
cache-off re-prefills the whole transcript). Asserts the assistant
transcript is identical across boots (deterministic temp=0)."""
import json, os, signal, subprocess, sys, time, urllib.request

BIN = "./target/release/flambeau"
MODEL = "/artefact/models/Qwen3.6-27B-Q8_0.gguf"
PORT = 8080
ANSWER_TOKENS = 400
TURNS = 10

# A substantial fixed system prompt so context starts high, plus a long
# technical paragraph appended each turn (deterministic growth independent
# of how verbose the model is) followed by a question that elicits a real
# answer. The transcript is append-only, so each turn shares an identical
# token prefix with the previous — exactly what the prefix cache exploits.
SYS = (
    "You are a senior systems engineer embedded in a code review. The "
    "project is a HIP/CUDA LLM inference server. Answer precisely and "
    "concretely, citing mechanisms and trade-offs. Keep prior context in "
    "mind across the whole conversation. " + ("Background note. " * 120)
)

PARA = (
    "Consider the following subsystem under discussion: a paged KV cache "
    "with per-layer ring buffers for sliding-window attention, a split-K "
    "decode kernel that fans the K/V loop across blocks, a host-snapshot "
    "prefix cache keyed by rolling chunk hashes, and a tensor-parallel "
    "all-reduce routed over a coherent copy engine. Each interacts with "
    "the others in non-obvious ways under growing context. "
)
QUESTIONS = [
    "Explain in detail how the split-K decode kernel keeps throughput "
    "stable as context grows, and where it can still degrade.",
    "Now contrast that with the prefix cache: what exactly does it save, "
    "and when does it NOT help?",
    "Walk through the failure modes of the per-layer ring buffer for "
    "sliding-window attention at high context.",
    "How would a tensor-parallel all-reduce over a copy engine interact "
    "with the prefix cache restore path? Be specific about ordering.",
    "Describe the memory accounting you'd want for the host-snapshot "
    "prefix cache, and the eviction policy trade-offs.",
    "What instrumentation would you add to attribute a TTFT regression to "
    "one of these subsystems rather than another?",
    "Suppose decode throughput drops at 32k context. Enumerate the "
    "candidate causes in priority order and how to discriminate them.",
    "Explain how chunk-aligned capture boundaries affect cache hit rate "
    "in a growing multi-turn conversation.",
    "What correctness invariants must the snapshot/restore preserve for a "
    "GDN recurrent-state hybrid model specifically?",
    "Summarize the end-to-end TTFT budget for turn N of a long chat and "
    "which term the prefix cache removes.",
]


def boot(cache_on):
    args = [BIN, "serve", "--model", MODEL, "--devices", "hip:0,2,1,3",
            "--mesh-mode", "pp+tp", "--pp-size", "2", "--tp-size", "2",
            "--port", str(PORT), "--inflight-slots", "1", "--ctx-cap", "16384",
            "--kv", "q8"]
    if cache_on:
        args += ["--prefix-cache", "--prefix-cache-max-gb", "12"]
    log = open(f"/tmp/pc_live_{'on' if cache_on else 'off'}.log", "w")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
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


def call(messages):
    body = json.dumps({"model": "x", "messages": messages, "temperature": 0.0,
                       "max_tokens": ANSWER_TOKENS, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter(); tf = None; ptok = None; content = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            pl = s[5:].strip()
            if pl == "[DONE]":
                break
            o = json.loads(pl); ch = o.get("choices") or []
            if ch:
                d = ch[0].get("delta", {})
                if d.get("content"):
                    if tf is None:
                        tf = time.perf_counter()
                    content += d["content"]
            if o.get("usage"):
                ptok = o["usage"].get("prompt_tokens")
    ttft = (tf - t0) * 1000.0 if tf else -1.0
    return ptok, ttft, content.strip()


def chat_session():
    msgs = [{"role": "system", "content": SYS}]
    rows = []          # (turn, ptok, ttft)
    transcript = []    # assistant answers, for parity
    for i in range(TURNS):
        q = QUESTIONS[i % len(QUESTIONS)]
        msgs.append({"role": "user", "content": PARA + q})
        ptok, ttft, content = call(msgs)
        rows.append((i + 1, ptok, ttft))
        transcript.append(content)
        msgs.append({"role": "assistant", "content": content})
    return rows, transcript


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
        rows, transcript = chat_session()
    finally:
        shutdown(p)
    results[tag] = (rows, transcript)
    print(f"=== cache {tag} done ===")
    for (t, ptok, ttft) in rows:
        print(f"  turn {t:>2}: ptok={ptok:>6} TTFT={ttft:>8.1f}ms")

off_rows, off_tr = results["off"]
on_rows, on_tr = results["on"]
print("\n=== A/B (growing-context chat, Qwen3.6-27B-Q8_0 pp2tp2 --kv q8) ===")
print(f"  {'turn':>4} {'ptok':>6} {'TTFT_off':>10} {'TTFT_on':>10} {'speedup':>8}")
tot_off = tot_on = 0.0
for (t, po, to), (_, _, tn) in zip(off_rows, on_rows):
    tot_off += to; tot_on += tn
    print(f"  {t:>4} {po:>6} {to:>9.1f}m {tn:>9.1f}m {to/tn if tn>0 else 0:>7.2f}x")
print(f"  TTFT sum: off {tot_off/1000:.1f}s / on {tot_on/1000:.1f}s = {tot_off/tot_on:.2f}x")
parity = off_tr == on_tr
print(f"\nTRANSCRIPT PARITY: {'PASS' if parity else 'FAIL'}")
if not parity:
    for i, (a, b) in enumerate(zip(off_tr, on_tr)):
        if a != b:
            print(f"  turn {i+1} differs:\n   off: {a[:120]!r}\n   on : {b[:120]!r}")
            break
