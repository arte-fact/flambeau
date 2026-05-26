#!/usr/bin/env python3
"""Long-prompt long-response sweep across 4 Qwen variants on pp2tp2.

For each model the harness:
  1. Boots `flambeau serve` with the full feature stack
     (FLAMBEAU_BATCHED_DECODE=1, GPU sampler, prefix cache,
      4 inflight slots, 32-deep admission queue, 512-token chunked
      prefill).
  2. Fires one greedy /v1/chat/completions request with a ~2k-token
     prompt and `max_tokens=512`.
  3. Concurrently samples rocm-smi for per-GPU VRAM + utilisation%
     during the prefill+decode window.
  4. Records prefill tok/s (TTFT-derived), decode tok/s
     (post-TTFT wall), and per-GPU peak VRAM + average util.
  5. Stops the server and moves on.

Topology fixed at pp2tp2 on devices `0,2,1,3` so stages straddle the
two whitehaven dies (memory: project_rig_whitehaven_topology + the
known-bad 2↔3 peer link is avoided by this pairing).
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "flambeau"
ROCM_SMI = "/opt/rocm/bin/rocm-smi"
DEVICES = "hip:0,2,1,3"
PP_SIZE = 2
TP_SIZE = 2

# (id, gguf_path, ctx_cap, prefill_ubatch, max_tokens)
MODELS = [
    ("qwen35_9B_q4_1",
     "/artefact/models/Qwen3.5-9B-Q4_1.gguf", 16384, 512, 512),
    ("qwen36_27B_q4_1",
     "/artefact/models/Qwen3.6-27B-Q4_1.gguf", 8192, 512, 512),
    ("qwen36_35B_a3b_q4_0",
     "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", 8192, 512, 512),
    ("qwen3_coder_next_q4_0",
     "/artefact/models/Qwen3-Coder-Next-Q4_0.gguf", 8192, 512, 512),
]

# Long prompt taken verbatim from run_matrix.py — ~2k tokens, exercises
# chunked prefill at FLAMBEAU_PREFILL_UBATCH=512 (4 chunks).
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from run_matrix import (
    LONG_PROMPT_SYSTEM, LONG_PROMPT_USER_TEMPLATE, LONG_PROMPT_PADDING,
)

PROMPT_PAD = LONG_PROMPT_PADDING


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def boot(model_path: str, ctx_cap: int, prefill_ubatch: int, log_path: Path):
    port = free_port()
    env = os.environ.copy()
    env["FLAMBEAU_CTX_CAP"] = str(ctx_cap)
    env["FLAMBEAU_INFLIGHT_SLOTS"] = "4"
    env["FLAMBEAU_GPU_SAMPLER"] = "1"
    env["FLAMBEAU_BATCHED_DECODE"] = "1"
    env["FLAMBEAU_PREFIX_CACHE"] = "1"
    env["FLAMBEAU_PREFIX_CACHE_MAX_GB"] = "4"
    env["FLAMBEAU_MAX_QUEUE_DEPTH"] = "32"
    env["FLAMBEAU_PREFILL_UBATCH"] = str(prefill_ubatch)
    env["RUST_LOG"] = "info"
    args = [str(BIN), "serve",
            "--model", model_path,
            "--devices", DEVICES,
            "--mesh-mode", "pp+tp",
            "--pp-size", str(PP_SIZE),
            "--tp-size", str(TP_SIZE),
            "--port", str(port)]
    log = open(log_path, "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, port


def wait_ready(port: int, log_path: Path, timeout_s: float = 600.0) -> bool:
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
            try:
                tail = log_path.read_text()[-2000:]
                if "panicked" in tail or "stack backtrace" in tail:
                    return False
            except Exception:
                pass
        time.sleep(2.0)
    return False


def kill(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---- rocm-smi samplers --------------------------------------------------

VRAM_RE = re.compile(r"GPU\[(\d+)\].*?(?:VRAM|Memory)\s*Used.*?:\s*(\d+)\s*MiB", re.IGNORECASE)
USE_RE = re.compile(r"GPU\[(\d+)\].*?GPU use.*?:\s*(\d+)", re.IGNORECASE)


def smi_snapshot() -> dict:
    """Return {gpu_idx: {vram_mib, util_pct}} from rocm-smi."""
    out_v = subprocess.run([ROCM_SMI, "--showmemuse"], capture_output=True, text=True, timeout=5.0)
    out_u = subprocess.run([ROCM_SMI, "--showuse"], capture_output=True, text=True, timeout=5.0)
    snap: dict[int, dict] = {}
    for m in VRAM_RE.finditer(out_v.stdout):
        snap.setdefault(int(m.group(1)), {})["vram_mib"] = int(m.group(2))
    for m in USE_RE.finditer(out_u.stdout):
        snap.setdefault(int(m.group(1)), {})["util_pct"] = int(m.group(2))
    return snap


def smi_use_meminfo() -> dict:
    """Fallback parser: rocm-smi --showmeminfo vram + --showuse, in case
    the showmemuse parser misses the format on this rocm version."""
    out_v = subprocess.run([ROCM_SMI, "--showmeminfo", "vram"], capture_output=True, text=True, timeout=5.0).stdout
    out_u = subprocess.run([ROCM_SMI, "--showuse"], capture_output=True, text=True, timeout=5.0).stdout
    snap: dict[int, dict] = {}
    used_re = re.compile(r"GPU\[(\d+)\].*?VRAM Total Used Memory\s*\(B\)\s*:\s*(\d+)", re.IGNORECASE)
    for m in used_re.finditer(out_v):
        snap.setdefault(int(m.group(1)), {})["vram_mib"] = int(m.group(2)) // (1024 * 1024)
    for m in USE_RE.finditer(out_u):
        snap.setdefault(int(m.group(1)), {})["util_pct"] = int(m.group(2))
    return snap


class SmiSampler(threading.Thread):
    def __init__(self, gpus: list[int], interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.gpus = gpus
        self.interval = interval_s
        self.stop_evt = threading.Event()
        self.peak_vram: dict[int, int] = {g: 0 for g in gpus}
        self.util_samples: dict[int, list[int]] = {g: [] for g in gpus}

    def run(self):
        while not self.stop_evt.is_set():
            try:
                snap = smi_use_meminfo()
                for g in self.gpus:
                    info = snap.get(g, {})
                    if "vram_mib" in info:
                        self.peak_vram[g] = max(self.peak_vram[g], info["vram_mib"])
                    if "util_pct" in info:
                        self.util_samples[g].append(info["util_pct"])
            except Exception:
                pass
            self.stop_evt.wait(self.interval)


# ---- one chat call ------------------------------------------------------

def stream_one(port: int, max_tokens: int) -> dict:
    body = {
        "model": "anything",
        "messages": [
            {"role": "system", "content": LONG_PROMPT_SYSTEM},
            {"role": "user",
             "content": LONG_PROMPT_USER_TEMPLATE.format(pad=PROMPT_PAD)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "seed": 42,
        "stream_options": {"include_usage": True},
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    completion_tokens = None
    prompt_tokens = None
    err = None
    try:
        with httpx.Client(timeout=httpx.Timeout(900.0)) as cli:
            with cli.stream("POST", url, json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("choices"):
                        delta = obj["choices"][0].get("delta", {})
                        if delta.get("content"):
                            if t_first is None:
                                t_first = time.perf_counter()
                            t_last = time.perf_counter()
                    if obj.get("usage"):
                        completion_tokens = obj["usage"].get("completion_tokens")
                        prompt_tokens = obj["usage"].get("prompt_tokens")
    except Exception as e:
        err = repr(e)
    if t_first is None:
        return {"err": err or "no first chunk"}
    return {
        "err": err,
        "prefill_ms": (t_first - t0) * 1000.0,
        "decode_ms": ((t_last or t_first) - t_first) * 1000.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }


# ---- main ---------------------------------------------------------------

def run_one(model_id: str, model_path: str, ctx_cap: int, prefill_ubatch: int,
            max_tokens: int, log_dir: Path) -> dict:
    if not Path(model_path).exists():
        return {"id": model_id, "err": f"GGUF missing: {model_path}"}
    log_path = log_dir / f"{model_id}.log"
    print(f"[{model_id}] booting → {log_path}", flush=True)
    t_boot = time.perf_counter()
    proc, port = boot(model_path, ctx_cap, prefill_ubatch, log_path)
    try:
        if not wait_ready(port, log_path):
            return {"id": model_id, "err": "boot failed", "log": str(log_path)}
        load_s = time.perf_counter() - t_boot
        print(f"[{model_id}] ready in {load_s:.1f}s; warming up", flush=True)
        # First-decode-after-load hits cold caches / first-touch paths and
        # under-reports steady-state throughput by ~40% on this rig. Always
        # warm up before the timed measurement (see task #36 papercut #3).
        try:
            _ = stream_one(port, 32)
        except Exception as e:
            print(f"[{model_id}] warmup failed: {e}", flush=True)
        print(f"[{model_id}] firing request", flush=True)
        sampler = SmiSampler([0, 1, 2, 3], interval_s=0.5)
        sampler.start()
        try:
            r = stream_one(port, max_tokens)
        finally:
            sampler.stop_evt.set()
            sampler.join(timeout=2.0)
        r["load_s"] = load_s
        r["peak_vram"] = sampler.peak_vram
        r["util_samples"] = sampler.util_samples
        r["id"] = model_id
        return r
    finally:
        kill(proc)


def fmt_table(rows: list[dict]) -> str:
    out: list[str] = []
    out.append("| model | load (s) | prompt tok | gen tok | prefill tok/s | decode tok/s | peak VRAM (MiB) GPU 0/1/2/3 | avg GPU% 0/1/2/3 |")
    out.append("|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        if r.get("err"):
            out.append(f"| {r['id']} | — | — | — | — | — | ERROR: {r['err']} | |")
            continue
        prompt_tok = r.get("prompt_tokens") or 0
        gen_tok = r.get("completion_tokens") or 0
        prefill_s = (r["prefill_ms"]) / 1000.0 if r.get("prefill_ms") else 0
        decode_s = (r["decode_ms"]) / 1000.0 if r.get("decode_ms") else 0
        prefill_tps = (prompt_tok / prefill_s) if prefill_s > 0 else 0
        decode_tps = (gen_tok / decode_s) if decode_s > 0 else 0
        vram = r.get("peak_vram", {})
        vram_str = "/".join(str(vram.get(g, 0)) for g in (0, 1, 2, 3))
        utl = r.get("util_samples", {})
        avg_util = []
        for g in (0, 1, 2, 3):
            xs = utl.get(g, [])
            avg_util.append(f"{(sum(xs) // len(xs)) if xs else 0}")
        util_str = "/".join(avg_util)
        out.append(f"| {r['id']} | {r.get('load_s', 0):.1f} | {prompt_tok} | {gen_tok} | {prefill_tps:.1f} | {decode_tps:.1f} | {vram_str} | {util_str} |")
    return "\n".join(out)


def main():
    out_root = ROOT / "certs" / "perf" / "matrix_pp2tp2_4models"
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in MODELS:
        rows.append(run_one(*spec, log_dir))
        # GPU thermal cooldown between models — MI50 sustained
        # benches push temp into 80°C-90°C; give 20s of idle.
        time.sleep(20)
    raw_path = out_root / "raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, default=str))
    md_path = out_root / "cert.md"
    md = [
        "# pp2tp2 4-model matrix (long prompt + long response)",
        "",
        f"Topology: pp2tp2 on devices `{DEVICES}` (pp_size={PP_SIZE}, tp_size={TP_SIZE}).",
        "Features: FLAMBEAU_BATCHED_DECODE=1, GPU_SAMPLER=1, PREFIX_CACHE=1 (4 GiB),",
        "4 inflight slots, 32 max-queue depth, 512-token prefill ubatch.",
        "Prompt: ~2 k-token long-prompt fixture (chunked to 4 prefill chunks).",
        "Sampling: greedy (temperature=0, seed=42), max_tokens=512.",
        "VRAM: peak per-GPU MiB observed via rocm-smi during prefill+decode.",
        "GPU%: average rocm-smi GPU-use% sample across the prefill+decode window.",
        "",
        fmt_table(rows),
        "",
        f"Raw JSON: `{raw_path.relative_to(ROOT)}`",
    ]
    md_path.write_text("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\ncert: {md_path}")


if __name__ == "__main__":
    main()
