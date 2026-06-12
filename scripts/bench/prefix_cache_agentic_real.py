#!/usr/bin/env python3
"""Real-agentic prefix-cache test: a tool-calling loop (OpenAI SDK shape)
whose context grows by tool results each round, run identically under
cache-off and cache-on boots. Reports per-call TTFT and asserts the full
transcript (tool-call sequence + answers) is identical across boots."""
import json, os, signal, subprocess, sys, time, urllib.request

BIN = "/artefact/flambeau/target/release/flambeau"
MODEL = "/artefact/models/Qwen3.6-27B-Q8_0.gguf"
PORT = 8101

SYS = ("You are a coding agent for the flambeau repository. Use the provided "
       "tools to inspect files before answering. Be concise and precise. "
       "Cite line numbers when referencing code. Never guess file contents — "
       "always read them with the tools first. ") * 8

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file and return its contents with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path"}},
            "required": ["path"]}}},
]

TASKS = [
    "List the contents of the crates directory.",
    "Read crates/server/src/prefix_cache.rs and tell me in one sentence what ChunkKey is.",
    "Read crates/forward/src/core/scratch.rs and tell me what KvLayout::Q8Contig's bytes_per_row formula is.",
    "Based on everything you've read so far, summarize in two sentences how a prefix-cache entry is keyed.",
]

# Deterministic canned tool outputs, consumed in call order. Sized like
# real tool results (~300-500 tokens each) so the context grows the way an
# agent trace does.
CANNED = [
    "backend-hip/\nbench/\ncli/\ncore/\nforward/\nkernels-cuda/\nkernels-hip/\n"
    "kernels-shared/\nmodel-ops/\nmodels/\nops/\nquant/\nruntime/\nserver/\nserver-core/\n",
    "\n".join(f"{i:>4}  {line}" for i, line in enumerate([
        "//! Prompt prefix cache: index of chunk-hash chains to KV snapshots.",
        "pub struct ChunkKey(pub u64);",
        "impl ChunkKey {",
        "    pub const SEED: ChunkKey = ChunkKey(0);",
        "    pub fn extend(self, tokens: &[u32]) -> ChunkKey {",
        "        let mut h = Hasher::new(self.0);",
        "        tokens.len().hash(&mut h);",
        "        for &t in tokens { t.hash(&mut h); }",
        "        ChunkKey(h.finish())",
        "    }",
        "}",
        "pub struct PrefixKeys {",
        "    pub chunk_keys: Vec<ChunkKey>,",
        "    pub chunk_tokens: usize,",
        "    pub prompt_tokens: usize,",
        "}",
    ] * 8, 1)),
    "\n".join(f"{i:>4}  {line}" for i, line in enumerate([
        "pub enum KvLayout {",
        "    F16Contig,",
        "    Q8Contig,",
        "}",
        "pub const Q8_0_BLOCK_BYTES: usize = 34;",
        "impl KvLayout {",
        "    pub fn bytes_per_row(self, kv_width: usize) -> usize {",
        "        match self {",
        "            KvLayout::F16Contig => 2 * kv_width,",
        "            KvLayout::Q8Contig => (kv_width / 32) * Q8_0_BLOCK_BYTES,",
        "        }",
        "    }",
        "}",
    ] * 10, 1)),
    "(no further tool output)",
    "(no further tool output)",
    "(no further tool output)",
]


def boot(cache_on):
    args = [BIN, "serve", "--model", MODEL, "--devices", "hip:0,2,1,3",
            "--mesh-mode", "pp+tp", "--pp-size", "2", "--tp-size", "2",
            "--port", str(PORT), "--inflight-slots", "1", "--ctx-cap", "8192",
            "--kv", "q8"]
    if cache_on:
        args += ["--prefix-cache", "--prefix-cache-max-gb", "6"]
    env = dict(os.environ); env["FLAMBEAU_KV"] = "q8"
    log = open(f"/tmp/pc_real_{'on' if cache_on else 'off'}.log", "w")
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


def call(messages):
    """One streaming chat call with tools. Returns (ptok, ttft_ms,
    content, tool_calls) where tool_calls = [(id, name, arguments)]."""
    body = json.dumps({"model": "x", "messages": messages, "tools": TOOLS,
                       "temperature": 0.0, "max_tokens": 256, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter(); tf = None; ptok = None
    content = ""
    calls = {}  # index -> {id, name, args}
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
                if d.get("content") or d.get("tool_calls"):
                    if tf is None:
                        tf = time.perf_counter()
                if d.get("content"):
                    content += d["content"]
                for tc in d.get("tool_calls") or []:
                    e = calls.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        e["name"] = fn["name"]
                    if fn.get("arguments"):
                        e["args"] += fn["arguments"]
            if o.get("usage"):
                ptok = o["usage"].get("prompt_tokens")
    ttft = (tf - t0) * 1000.0 if tf else -1.0
    tool_calls = [(calls[i]["id"], calls[i]["name"], calls[i]["args"])
                  for i in sorted(calls)]
    return ptok, ttft, content.strip(), tool_calls


def agent_session():
    msgs = [{"role": "system", "content": SYS}]
    rows = []          # per-API-call: (ptok, ttft, kind)
    transcript = []    # parity record: ("tool", name, args) / ("answer", text)
    canned = iter(CANNED)
    for task in TASKS:
        msgs.append({"role": "user", "content": task})
        for _round in range(3):
            ptok, ttft, content, tcs = call(msgs)
            if tcs:
                rows.append((ptok, ttft, "tool_call"))
                msgs.append({"role": "assistant", "content": content or None,
                             "tool_calls": [{"id": i or f"call_{n}", "type": "function",
                                             "function": {"name": n, "arguments": a}}
                                            for (i, n, a) in tcs]})
                for (i, n, a) in tcs:
                    transcript.append(("tool", n, a))
                    msgs.append({"role": "tool", "tool_call_id": i or f"call_{n}",
                                 "content": next(canned, "(no output)")})
                continue
            rows.append((ptok, ttft, "answer"))
            transcript.append(("answer", content))
            msgs.append({"role": "assistant", "content": content})
            break
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
    t_session = time.perf_counter()
    try:
        rows, transcript = agent_session()
    finally:
        shutdown(p)
    wall = time.perf_counter() - t_session
    results[tag] = (rows, transcript, wall)
    print(f"=== cache {tag} (session wall {wall:.1f}s) ===")
    for i, (ptok, ttft, kind) in enumerate(rows):
        print(f"  call {i+1:>2}: ptok={ptok:>5} TTFT={ttft:>8.1f}ms  {kind}")

print("\n=== gate ===")
rows_off, tr_off, wall_off = results["off"]
rows_on, tr_on, wall_on = results["on"]
parity = tr_off == tr_on and len(rows_off) == len(rows_on)
total_off = sum(t for (_, t, _) in rows_off)
total_on = sum(t for (_, t, _) in rows_on)
for i in range(min(len(rows_off), len(rows_on))):
    po, to, _ = rows_off[i]
    pn, tn, k = rows_on[i]
    print(f"  call {i+1:>2}: ptok={po:>5} TTFT off {to:>8.1f} / on {tn:>8.1f} = {to/tn if tn>0 else 0:5.2f}x  {k}")
print(f"\n  TTFT sum: off {total_off/1000:.1f}s / on {total_on/1000:.1f}s = {total_off/total_on:.2f}x")
print(f"  session wall: off {wall_off:.1f}s / on {wall_on:.1f}s")
print(f"TRANSCRIPT PARITY: {'PASS' if parity else 'FAIL'}")
if not parity:
    for i in range(max(len(tr_off), len(tr_on))):
        a = tr_off[i] if i < len(tr_off) else None
        b = tr_on[i] if i < len(tr_on) else None
        if a != b:
            print(f"  diverged at step {i}:\n    off: {str(a)[:140]}\n    on : {str(b)[:140]}")
            break
