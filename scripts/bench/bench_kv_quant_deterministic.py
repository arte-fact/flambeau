#!/usr/bin/env python3
"""Deterministic-decode bench rig for KV-quant A/B (Phase 4).

Boots one flambeau server with the requested --kv layout, sends N
greedy-decode requests over SSE streaming, measures per-token wall
times. Reports TTFT + decode-tps median, decoupled from ct variance:
the per-token rate is computed inside the actual decode window
(timestamp of last delta - timestamp of first delta) / (n_tokens - 1).

That number is robust to early-EOS — F16 and Q8 KV can stop at
different ct counts but their per-step decode rate is still a clean
A/B signal.

Env knobs (all optional except MODEL):
  BENCH_MODEL  : abs path to GGUF                       (required)
  BENCH_KV     : "f16" or "q8"                          (default: f16)
  BENCH_CTX    : ctx_cap                                (default: 4096)
  BENCH_PROMPT_TOKENS : target prompt length            (default: 256)
  BENCH_MAX_TOKENS    : per-request max decode length   (default: 128)
  BENCH_REPS          : repetition count                (default: 3)
  BENCH_DEVICES       : --devices CSV                   (default: 0,2,1,3)
  BENCH_MESH          : pp / tp / pp+tp                 (default: pp+tp)
  BENCH_PP / BENCH_TP : ranks per axis                  (default: 2 / 2)
  BENCH_TAG           : log file tag                    (default: kvq_<KV>)

Output (stdout):
  Per-rep: ttft_ms, total_wall_ms, decode_tokens, decode_tps,
           interval-only tps (last-first)/n
  Summary: median + min + max for ttft_ms, decode_tps
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx

BIN = Path("/artefact/flambeau/target/release/flambeau")
MODEL = os.environ.get("BENCH_MODEL")
KV = os.environ.get("BENCH_KV", "f16")
CTX = int(os.environ.get("BENCH_CTX", "4096"))
PROMPT_TOKENS = int(os.environ.get("BENCH_PROMPT_TOKENS", "256"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "128"))
REPS = int(os.environ.get("BENCH_REPS", "3"))
DEVICES = os.environ.get("BENCH_DEVICES", "0,2,1,3")
MESH = os.environ.get("BENCH_MESH", "pp+tp")
PP = int(os.environ.get("BENCH_PP", "2"))
TP = int(os.environ.get("BENCH_TP", "2"))
TAG = os.environ.get("BENCH_TAG", f"kvq_{KV}")
LOG = Path(f"/artefact/flambeau/scripts/bench/logs/{TAG}.log")

if not MODEL:
    print("BENCH_MODEL not set", file=sys.stderr)
    sys.exit(2)


BASE_FILLER = (
    "The Linux kernel scheduler has evolved through major redesigns "
    "since CFS replaced O(1) in 2007. EEVDF added deadline guarantees "
    "in kernel 6.6. Understanding scheduler behavior requires modeling "
    "latency, throughput, fairness, and starvation under varied workloads. "
)


def make_prompt(target_tokens: int) -> str:
    # ~60 tokens per BASE_FILLER repetition for these models.
    reps = max(1, target_tokens // 60)
    return (BASE_FILLER * reps) + "Summarize the main scheduler design decisions in 2 sentences."


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def boot():
    port = free_port()
    env = os.environ.copy()
    env["FLAMBEAU_CTX_CAP"] = str(CTX)
    env["FLAMBEAU_GPU_SAMPLER"] = "1"
    env["FLAMBEAU_BATCHED_DECODE"] = "1"
    env["FLAMBEAU_PREFILL_UBATCH"] = "512"
    env["RUST_LOG"] = "info"
    args = [str(BIN), "serve",
            "--model", MODEL,
            "--devices", DEVICES,
            "--mesh-mode", MESH,
            "--kv", KV,
            "--port", str(port)]
    if MESH != "pp":
        args.extend(["--pp-size", str(PP), "--tp-size", str(TP)])
    log = open(LOG, "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, port


def wait_ready(port: int, timeout_s: float = 900.0) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if LOG.exists():
            tail = LOG.read_text()[-2000:]
            if "panicked" in tail or "stack backtrace" in tail:
                return False
        time.sleep(3.0)
    return False


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_sse_content(raw: bytes) -> str | None:
    """Extract `choices[0].delta.content` from an SSE line, or sentinels."""
    line = raw.decode("utf-8", errors="replace")
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
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


def run_one(port: int, prompt: str, max_tokens: int) -> dict:
    """One greedy streaming request. Returns per-step + summary stats."""
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    token_times: list[float] = []
    text_chunks: list[str] = []
    t_request = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                content = parse_sse_content(raw.rstrip())
                if content == "__done__":
                    break
                if content is None or content == "":
                    continue
                token_times.append(time.monotonic())
                text_chunks.append(content)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    if not token_times:
        return {"error": "no SSE tokens received"}

    wall_ms = (time.monotonic() - t_request) * 1000.0
    ttft_ms = (token_times[0] - t_request) * 1000.0
    n_tokens = len(token_times)
    if n_tokens >= 2:
        decode_span_ms = (token_times[-1] - token_times[0]) * 1000.0
        decode_tps_interval = 1000.0 * (n_tokens - 1) / decode_span_ms
    else:
        decode_span_ms = 0.0
        decode_tps_interval = 0.0
    text = "".join(text_chunks)
    return {
        "ttft_ms": ttft_ms,
        "wall_ms": wall_ms,
        "n_tokens": n_tokens,
        "decode_span_ms": decode_span_ms,
        "decode_tps_interval": decode_tps_interval,
        "preview": text[:80],
    }


def summarise(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: (no samples)")
        return
    vs = sorted(values)
    n = len(vs)
    median = vs[n // 2]
    print(f"  {label}: median={median:.2f}  min={vs[0]:.2f}  max={vs[-1]:.2f}  n={n}")


def main():
    print(f"=== Deterministic KV-quant bench ===", flush=True)
    print(f"  tag={TAG}  model={Path(MODEL).name}  kv={KV}  ctx={CTX}", flush=True)
    print(f"  prompt_tokens≈{PROMPT_TOKENS}  max_tokens={MAX_TOKENS}  reps={REPS}", flush=True)
    print(f"  mesh={MESH}  pp={PP}  tp={TP}  devices={DEVICES}", flush=True)
    print()

    prompt = make_prompt(PROMPT_TOKENS)
    proc, port = boot()
    print(f"booted pid={proc.pid} port={port}", flush=True)
    if not wait_ready(port):
        print("FAIL: server not ready")
        kill(proc)
        sys.exit(1)
    print("server ready", flush=True)
    print()

    # Warm-up: one greedy request that's discarded (first call often has
    # JIT/cache effects that skew the median).
    print("--- warm-up ---", flush=True)
    warm = run_one(port, prompt, max_tokens=16)
    if "error" in warm:
        print(f"  warm-up FAIL: {warm['error']}", flush=True)
    else:
        print(f"  warm-up ttft={warm['ttft_ms']:.1f}ms n={warm['n_tokens']}", flush=True)
    print()

    print("--- measurement runs ---", flush=True)
    ttfts = []
    decode_tps = []
    walls = []
    for i in range(REPS):
        r = run_one(port, prompt, MAX_TOKENS)
        if "error" in r:
            print(f"  rep{i}: ERROR {r['error']}", flush=True)
            continue
        print(
            f"  rep{i}: ttft={r['ttft_ms']:7.1f}ms  wall={r['wall_ms']:7.1f}ms  "
            f"n={r['n_tokens']:3d}  decode_span={r['decode_span_ms']:6.1f}ms  "
            f"decode_tps={r['decode_tps_interval']:5.2f}  | {r['preview']!r}",
            flush=True,
        )
        ttfts.append(r["ttft_ms"])
        decode_tps.append(r["decode_tps_interval"])
        walls.append(r["wall_ms"])
    print()
    print("--- summary ---", flush=True)
    summarise("ttft_ms", ttfts)
    summarise("decode_tps (interval)", decode_tps)
    summarise("wall_ms", walls)

    kill(proc)
    print(f"\n=== {TAG} done ===", flush=True)


if __name__ == "__main__":
    main()
