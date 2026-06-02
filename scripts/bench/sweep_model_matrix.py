#!/usr/bin/env python3
"""Model × context sweep — prefill + decode tps per cell.

Boots one flambeau server per model in sequence, sends one small-ctx
greedy request + one large-ctx greedy request, records:

  - prefill_tps = prompt_tokens / ttft_seconds
                  (includes prefill kernel time + launch overhead)
  - decode_tps  = (n_tokens - 1) / (last_delta_t - first_delta_t)
                  (interval-only, robust to early-EOS / ct variance)
  - ttft_ms / wall_ms summary

Each model is benched at 2 prompt lengths and 3 reps each. Reports a
tidy markdown table on stdout.
"""
from __future__ import annotations

import json
import os
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
DEVICES = "0,2,1,3"
PP, TP = 2, 2
LOG_DIR = Path("/artefact/flambeau/scripts/bench/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILLER = (
    "The Linux kernel scheduler has evolved through major redesigns "
    "since CFS replaced O(1) in 2007. EEVDF added deadline guarantees "
    "in kernel 6.6. Understanding scheduler behavior requires modeling "
    "latency, throughput, fairness, and starvation under varied workloads. "
)


def make_prompt(target_tokens: int) -> str:
    reps = max(1, target_tokens // 60)
    return (BASE_FILLER * reps) + "Summarize the main scheduler design decisions in 2 sentences."


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def boot(model: Path, ctx_cap: int, log_path: Path):
    port = free_port()
    env = os.environ.copy()
    env["FLAMBEAU_CTX_CAP"] = str(ctx_cap)
    env["FLAMBEAU_GPU_SAMPLER"] = "1"
    env["FLAMBEAU_BATCHED_DECODE"] = "1"
    env["FLAMBEAU_PREFILL_UBATCH"] = "512"
    env["RUST_LOG"] = "info"
    args = [str(BIN), "serve",
            "--model", str(model),
            "--devices", DEVICES,
            "--mesh-mode", "pp+tp",
            "--pp-size", str(PP), "--tp-size", str(TP),
            "--kv", "f16",
            "--port", str(port)]
    fh = open(log_path, "w")
    proc = subprocess.Popen(args, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, port


def wait_ready(port: int, log_path: Path, timeout_s: float = 900.0) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if log_path.exists():
            tail = log_path.read_text()[-2000:]
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
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_sse_content(raw: bytes) -> str | None:
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
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    if not token_times:
        return {"error": "no SSE tokens"}

    wall_ms = (time.monotonic() - t_request) * 1000.0
    ttft_ms = (token_times[0] - t_request) * 1000.0
    n = len(token_times)
    if n >= 2:
        decode_span_ms = (token_times[-1] - token_times[0]) * 1000.0
        decode_tps = 1000.0 * (n - 1) / decode_span_ms
    else:
        decode_span_ms = 0.0
        decode_tps = 0.0
    return {"ttft_ms": ttft_ms, "wall_ms": wall_ms, "n": n,
            "decode_span_ms": decode_span_ms, "decode_tps": decode_tps}


def get_pt_from_log(log_path: Path) -> int | None:
    """Pull the last `prompt_tokens=N` value emitted by the server log
    for the request we just sent."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    import re
    pts = re.findall(r"completion request accepted.*?prompt_tokens=(\d+)", text)
    if pts:
        return int(pts[-1])
    pts = re.findall(r"prompt_tokens=(\d+)", text)
    if pts:
        return int(pts[-1])
    return None


def bench_model(model: Path, label: str) -> list[dict]:
    """Returns a list of cells, one per (model, ctx) combination."""
    cells = []
    log_path = LOG_DIR / f"sweep_matrix_{label}.log"
    print(f"\n=== {label} — {model.name} ===", flush=True)
    print(f"  booting...", flush=True)
    proc, port = boot(model, ctx_cap=16384, log_path=log_path)
    print(f"  pid={proc.pid} port={port}", flush=True)
    if not wait_ready(port, log_path):
        print(f"  FAIL: server not ready", flush=True)
        kill(proc)
        return [{"label": label, "ctx": "boot", "error": "server not ready"}]
    print(f"  ready, running cells", flush=True)

    # Warm-up so JIT / first-call doesn't skew the median.
    _ = run_one(port, make_prompt(200), max_tokens=8)

    for ctx_name, target_pt, max_tok in [
        ("small", 200, 64),
        ("large", 5000, 64),
    ]:
        prompt = make_prompt(target_pt)
        prefill_tps_runs = []
        decode_tps_runs = []
        ttft_runs = []
        wall_runs = []
        actual_pt = None
        for rep in range(3):
            r = run_one(port, prompt, max_tok)
            if "error" in r:
                print(f"  {ctx_name} rep{rep}: ERROR {r['error']}", flush=True)
                continue
            if actual_pt is None:
                actual_pt = get_pt_from_log(log_path) or target_pt
            prefill_tps_runs.append(actual_pt / (r["ttft_ms"] / 1000.0))
            decode_tps_runs.append(r["decode_tps"])
            ttft_runs.append(r["ttft_ms"])
            wall_runs.append(r["wall_ms"])

        def med(xs):
            return statistics.median(xs) if xs else 0.0

        cell = {
            "label": label,
            "ctx": ctx_name,
            "pt": actual_pt,
            "ttft_ms_med": med(ttft_runs),
            "wall_ms_med": med(wall_runs),
            "prefill_tps_med": med(prefill_tps_runs),
            "decode_tps_med": med(decode_tps_runs),
            "reps": len(prefill_tps_runs),
        }
        cells.append(cell)
        print(
            f"  {ctx_name:5s}: pt={cell['pt']:5d}  "
            f"prefill={cell['prefill_tps_med']:7.1f} tps  "
            f"decode={cell['decode_tps_med']:6.2f} tps  "
            f"ttft={cell['ttft_ms_med']:7.1f} ms  reps={cell['reps']}",
            flush=True,
        )

    kill(proc)
    return cells


def main():
    matrix = [
        # (label, model file)
        ("qwen3.6-27B Q4_0",         "/artefact/models/Qwen3.6-27B-Q4_0.gguf"),
        ("qwen3.6-27B UD-Q4_K_XL",   "/artefact/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"),
        ("qwen3.6-27B Q8_0",         "/artefact/models/Qwen3.6-27B-Q8_0.gguf"),
        ("qwen3.6-35B-A3B UD-Q6_K_XL", "/artefact/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"),
        ("gemma-4-26B-A4B Q8_0",     "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf"),
        ("qwen3.5-122B-A10B UD-Q2_K_XL", "/artefact/models/Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf"),
    ]

    all_cells = []
    for label, model_path in matrix:
        model = Path(model_path)
        if not model.exists():
            print(f"SKIP {label}: file not found", flush=True)
            continue
        cells = bench_model(model, label.replace(" ", "_").replace("/", "_"))
        for c in cells:
            c["display_label"] = label
            all_cells.append(c)

    # Pretty markdown summary.
    print("\n\n## Sweep summary\n")
    print("| Model | ctx | prompt_tokens | prefill_tps | decode_tps | ttft_ms | wall_ms |")
    print("|-------|-----|--------------:|------------:|-----------:|--------:|--------:|")
    for c in all_cells:
        if "error" in c:
            print(f"| {c['display_label']} | {c['ctx']} | — | — | — | — | ERROR: {c['error']} |")
            continue
        print(
            f"| {c['display_label']} | {c['ctx']} | {c['pt']} | "
            f"{c['prefill_tps_med']:.1f} | {c['decode_tps_med']:.2f} | "
            f"{c['ttft_ms_med']:.1f} | {c['wall_ms_med']:.1f} |"
        )

    out = Path("/tmp/sweep_matrix_results.json")
    out.write_text(json.dumps(all_cells, indent=2))
    print(f"\nDetailed cells → {out}")


if __name__ == "__main__":
    main()
