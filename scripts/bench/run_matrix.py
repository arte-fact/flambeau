#!/usr/bin/env python3
"""
Bench matrix harness — concurrent decode throughput across:
  - models       : Qwen3.5-9B-Q4_1, Qwen3.6-27B-Q4_1, Qwen3.6-35B-A3B-Q4_0
  - topologies   : single (1 GPU), TP2 (2 GPU), PP4 (4 GPU), pp2tp2 (4 GPU)
  - paths        : batched (FLAMBEAU_BATCHED_DECODE=1) vs no-batched
  - concurrency  : 1, 2, 4, 8

Per cell: spawn N concurrent /v1/chat/completions calls (long prompt + long
response), capture per-stream prefill (TTFT) + decode tok/s, aggregate.

For each (model, topology, path) we boot one server process, warm it,
then sweep all concurrency levels in sequence, then tear down.

Usage:
    python3 scripts/bench/run_matrix.py --out certs/perf/<name>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

ROCM_SMI = "/opt/rocm-7.1.1/core-7.13/bin/rocm-smi"
TEMP_RE = re.compile(r"Temperature.*?:\s*([\d.]+)")


def gpu_max_temp_c() -> float:
    """Return the max GPU temperature (any sensor, any GPU) in °C, or 0 on failure."""
    try:
        out = subprocess.run(
            [ROCM_SMI, "--showtemp"],
            capture_output=True, text=True, timeout=5.0,
        ).stdout
    except Exception:
        return 0.0
    temps = [float(m.group(1)) for m in TEMP_RE.finditer(out)]
    return max(temps) if temps else 0.0


def cooldown_until_safe(threshold_c: float, max_wait_s: float, poll_s: float = 10.0) -> float:
    """Block until max GPU temp < threshold, or max_wait elapses. Returns seconds slept."""
    t0 = time.perf_counter()
    while True:
        t = gpu_max_temp_c()
        elapsed = time.perf_counter() - t0
        if t == 0.0:  # rocm-smi unavailable; skip cooldown entirely
            return elapsed
        if t < threshold_c or elapsed >= max_wait_s:
            print(f"   cooldown: {t:.0f}°C, slept {elapsed:.0f}s", flush=True)
            return elapsed
        if elapsed == 0.0 or int(elapsed) % 30 == 0:
            print(f"   cooldown: {t:.0f}°C ≥ {threshold_c:.0f}°C, waiting...", flush=True)
        time.sleep(poll_s)

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "flambeau"

LONG_PROMPT_SYSTEM = (
    "You are a thorough technical writer. Produce a long, well-structured, "
    "informative answer to the user's question. Aim for at least 300 words. "
    "Use plain prose and prefer concrete examples over high-level slogans."
)
LONG_PROMPT_PADDING = """
Reference excerpt (background only, do not summarise verbatim):
The Linux kernel scheduler is implemented primarily in kernel/sched/core.c,
kernel/sched/fair.c, and kernel/sched/rt.c. Each runqueue (struct rq,
defined in kernel/sched/sched.h) is a per-CPU structure containing a
red-black tree of cfs_rq entries plus rt_rq, dl_rq, and stop scheduler
classes. The scheduler tick fires from the timer subsystem via
scheduler_tick() in kernel/sched/core.c, which delegates to the current
class's task_tick callback. CFS computes virtual runtime via
calc_delta_fair() and picks the leftmost task in the rb-tree; vruntime
accumulates in proportion to task weight derived from nice value via
sched_prio_to_weight[]. Wakeup checks (check_preempt_curr) compare
vruntime gaps against sysctl_sched_min_granularity. Load balancing fires
from the timer-driven trigger run_rebalance_domains() and the scheduling
domain hierarchy stored per CPU in sd. Each domain encodes flags such as
SD_BALANCE_NEWIDLE, SD_BALANCE_FORK, SD_NUMA, SD_SHARE_PKG_RESOURCES;
this hierarchy is what decides whether a wake-up migrates and whether
imbalance is corrected via active_load_balance_cpu_stop. NUMA balancing
piggybacks on do_numa_page() PTE faults to record task locality and
schedule-time choices then walk numa_node sched-domain weight tables.
The deadline class (kernel/sched/deadline.c) implements EDF with budget
enforcement under SCHED_FLAG_DL_OVERRUN and constraints set via
sched_setattr(). Real-time priority inheritance is implemented in
kernel/locking/rtmutex.c; the rt_mutex chain walks the lock holder
graph and boosts the holder's effective priority via
rt_mutex_setprio(). Idle is governed by cpuidle_governor and the
do_idle() loop in kernel/sched/idle.c which integrates with cpufreq via
the schedutil governor (kernel/sched/cpufreq_schedutil.c) so frequency
selection participates in scheduling decisions. EEVDF (introduced in
v6.6) replaced CFS's vruntime ordering with a virtual-deadline metric:
each task gets a vd = ve + slice/weight and the scheduler picks the
task with the smallest vd subject to having lag (eligibility). Power
management interactions include EAS (Energy Aware Scheduling) on
heterogeneous big.LITTLE topologies which uses em_cpu_energy() to
estimate the power cost of placing a task on each performance domain.
Cgroup-v2 cpu.weight and cpu.max are enforced through the per-task-group
hierarchy in fair.c; the cfs_bandwidth machinery throttles a task_group
when it exhausts its quota in a period. SCHED_DEADLINE admission control
runs through __dl_overflow() and is verified against the global
bandwidth at sched_setattr() time. Priority inversion mitigation under
RT-PI uses the rt_mutex implementation that attaches a pi_blocked_on
record to the task; the chain walk re-evaluates effective priority up
to the depth of contention. Per-CPU scheduler statistics are exported
via /proc/schedstat and per-task via /proc/<pid>/sched.
"""

LONG_PROMPT_USER_TEMPLATE = (
    "Write a comprehensive explanation of how a modern operating-system "
    "scheduler decides which thread to run next on a multi-core CPU. "
    "Cover at minimum: (1) the data structures the scheduler maintains "
    "(run queues, priority bands, cgroups), (2) the trigger events that "
    "invoke the scheduler (timer ticks, blocking syscalls, IPIs, "
    "preemption), (3) how fairness is balanced against throughput "
    "(CFS-style virtual runtime, EEVDF, deadline classes), (4) per-CPU "
    "vs global run queues and the resulting load-balancing rebalances, "
    "(5) interactions with NUMA topology and last-level cache locality, "
    "(6) interactions with the kernel's idle / power-management governor, "
    "(7) priority inheritance and priority inversion, and (8) the role of "
    "real-time vs normal classes. Use Linux as the running example. "
    "Cite the relevant kernel files where you can.\n\n"
    + (LONG_PROMPT_PADDING * 4)
    + "\n\nProceed with the answer now. Padding tag: {pad}"
)

# ----------------------------------------------------------------------
# Models + topologies

@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    path: str
    ctx_cap: int
    prefill_ubatch: int

MODELS = [
    ModelSpec("qwen35_9B_q4_1",  "/artefact/models/Qwen3.5-9B-Q4_1.gguf",       16384, 512),
    # Qwen3.5-9B re-quantised from Q4_1 → Q3_K_S via llama-quantize
    # (`--allow-requantize` Q3_K_S). Exists to exercise the Q3_K dispatch
    # rows on a model that fits a single GPU; main rejects this dtype.
    ModelSpec("qwen35_9B_q3_k_s", "/artefact/models/Qwen3.5-9B-Q3_K_S.gguf",    16384, 512),
    # 27B context dropped from 16384 to 4096 so PP2 (2 GPUs × 32 layers
    # per rank × 8 inflight slots) fits in VRAM. Long-prompt
    # characterisation is still meaningful at ctx=4096.
    ModelSpec("qwen36_27B_q4_0", "/artefact/models/Qwen3.6-27B-Q4_0.gguf",      4096, 512),
    ModelSpec("qwen36_27B_q4_1", "/artefact/models/Qwen3.6-27B-Q4_1.gguf",      4096, 512),
    ModelSpec("qwen36_35B_a3b_q4_0", "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", 16384, 512),
    # Qwen3.6-35B-A3B MoE re-quantised from Q4_0 → Q3_K_S to exercise the
    # Q3_K MoE indexed-MMVQ + tile8 kernels added in tier-1. main rejects
    # Q3_K weights at qmatmul-dispatch time so this is a branch-only path.
    ModelSpec("qwen36_35B_a3b_q3_k_s", "/artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf", 16384, 512),
    ModelSpec("qwen36_35B_a3b_ud_q4_k_s", "/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", 4096, 512),
    # gemma4 dense — added 2026-05-23 after the BOS-prepend fix landed.
    # E4B fits on one GPU; 31B needs PP across all 4 (per-layer KV at
    # ctx=4096 already pushes ~12 GB on the heaviest rank).
    ModelSpec("gemma4_E4B_q4_0", "/artefact/models/gemma-4-E4B-it-Q4_0.gguf",   4096, 512),
    ModelSpec("gemma4_31B_q4_0", "/artefact/models/gemma-4-31B-it-Q4_0.gguf",   4096, 512),
]

@dataclasses.dataclass(frozen=True)
class TopoSpec:
    id: str
    devices: str    # comma-separated hip:N
    mesh_mode: str  # "pp", "tp", "pp+tp"
    pp_size: int
    tp_size: int
    n_gpus: int

TOPOLOGIES = [
    TopoSpec("single",  "hip:0",         "pp",    0, 0, 1),
    # **2026-05-05** — pp2/tp2 use cross-die hip:0,2 (one GPU per die)
    # for two reasons: (a) memory note `project_rig_whitehaven_topology`
    # says Mesh<2> should prefer {0,1} intra-die-0 OR cross-die {0,2},
    # and (b) cross-die spreads thermal load between dies so a sustained
    # bench doesn't cook one die while the other idles.
    TopoSpec("pp2",     "hip:0,2",       "pp",    0, 0, 2),
    TopoSpec("tp2",     "hip:0,2",       "tp",    0, 2, 2),
    TopoSpec("pp4",     "hip:0,1,2,3",   "pp",    0, 0, 4),
    TopoSpec("pp2tp2",  "hip:0,2,1,3",   "pp+tp", 2, 2, 4),
]

CONCURRENCIES = [1, 2, 4, 8]
# Path: batched vs no-batched. In both, FLAMBEAU_INFLIGHT_SLOTS = max(CONCURRENCIES).
PATHS = ["no_batched", "batched"]

# Skip cells that won't fit in 16 GB / GPU.
def is_feasible(model: ModelSpec, topo: TopoSpec) -> bool:
    # Single-GPU 27B (~14-17 GB) doesn't fit in 16 GB MI50 above Q4_0.
    # Single-GPU 35B A3B (~20 GB+) doesn't fit either.
    # 2-GPU TP for 35B (~10 GB / GPU + KV) marginal but tries.
    if topo.id == "single":
        if model.id in ("qwen36_27B_q4_0", "qwen36_27B_q4_1",
                        "qwen36_35B_a3b_q4_0", "qwen36_35B_a3b_ud_q4_k_s",
                        "gemma4_31B_q4_0"):
            return False
    if topo.id == "tp2":
        if model.id in ("qwen36_35B_a3b_q4_0", "qwen36_35B_a3b_ud_q4_k_s",
                        "gemma4_31B_q4_0"):
            # 35B / 2 ≈ 10 GB; gemma4-31B Q4_0 ≈ 17 GB so /2 ≈ 8.5 GB
            # weights but per-layer KV across 64 layers at ctx=4096 with
            # head_dim 256/512 mix pushes ~16+ GB / GPU. Skip.
            return False
    if topo.id in ("pp2", "pp4", "pp2tp2") and model.id == "gemma4_31B_q4_0":
        # PP2 (2 GPUs) for 31B Q4_0 — 17 GB / 2 ≈ 8.5 GB weights but
        # heavy per-layer KV (gemma4 SWA+global at head_dim 256/512)
        # OOMs on PP2. Confirmed via prior session OOM at default ctx;
        # ctx=4096 still tight. Keep PP4 and skip PP2.
        if topo.id == "pp2":
            return False
    return True

# ----------------------------------------------------------------------
# Server lifecycle

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def boot_server(model: ModelSpec, topo: TopoSpec, batched: bool, slots: int,
                log_path: Path) -> tuple[subprocess.Popen, int]:
    port = free_port()
    env = os.environ.copy()
    env["FLAMBEAU_CTX_CAP"] = str(model.ctx_cap)
    env["FLAMBEAU_INFLIGHT_SLOTS"] = str(slots)
    env["FLAMBEAU_GPU_SAMPLER"] = "1"
    env["FLAMBEAU_PREFILL_UBATCH"] = str(model.prefill_ubatch)
    env["RUST_LOG"] = "info"
    if batched:
        env["FLAMBEAU_BATCHED_DECODE"] = "1"
    else:
        env.pop("FLAMBEAU_BATCHED_DECODE", None)

    args = [str(BIN), "serve",
            "--model", model.path,
            "--devices", topo.devices,
            "--mesh-mode", topo.mesh_mode,
            "--port", str(port)]
    if topo.mesh_mode == "pp+tp":
        args += ["--pp-size", str(topo.pp_size), "--tp-size", str(topo.tp_size)]
    elif topo.mesh_mode == "tp":
        args += ["--tp-size", str(topo.tp_size)]

    log = open(log_path, "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, port


def wait_ready(port: int, timeout_s: float, model_id: str) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def kill_server(proc: subprocess.Popen) -> None:
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

# ----------------------------------------------------------------------
# One streaming chat call → returns (prefill_ms, decode_tokens, decode_ms,
# completion_tokens_reported)

def stream_one_call(port: int, prompt_pad: str, max_tokens: int, seed: int) -> dict:
    body = {
        "model": "anything",
        "messages": [
            {"role": "system", "content": LONG_PROMPT_SYSTEM},
            {"role": "user",
             "content": LONG_PROMPT_USER_TEMPLATE.format(pad=prompt_pad)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "seed": seed,
        "stream_options": {"include_usage": True},
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t_send = time.perf_counter()
    t_first = None
    t_last = None
    n_chunks = 0
    n_content_chunks = 0
    completion_tokens = None
    prompt_tokens = None
    err = None
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as cli:
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
                    if "choices" in obj and obj["choices"]:
                        delta = obj["choices"][0].get("delta", {})
                        # TTFT = time until first content byte. The role
                        # chunk is emitted at request acceptance (before
                        # prefill compute), so do not count it.
                        if delta.get("content"):
                            if t_first is None:
                                t_first = time.perf_counter()
                            n_content_chunks += 1
                        n_chunks += 1
                        t_last = time.perf_counter()
                    if "usage" in obj and obj["usage"]:
                        u = obj["usage"]
                        completion_tokens = u.get("completion_tokens")
                        prompt_tokens = u.get("prompt_tokens")
    except Exception as e:
        err = repr(e)

    if t_first is None:
        return {"err": err or "no first chunk", "wall_ms": (time.perf_counter() - t_send) * 1000}

    prefill_ms = (t_first - t_send) * 1000.0
    decode_ms = ((t_last or t_first) - t_first) * 1000.0
    return {
        "err": err,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "n_content_chunks": n_content_chunks,
        "n_chunks": n_chunks,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "wall_ms": (time.perf_counter() - t_send) * 1000,
    }

# ----------------------------------------------------------------------
# One concurrency cell

def run_cell(port: int, n_concurrent: int, max_tokens: int) -> dict:
    # Pick distinct pad strings so each request has the same prompt length
    # but generation diverges (decode-speed should be identical at temp=0 with
    # same prompt, but we want to avoid any prefix-cache hits in the future).
    results = [None] * n_concurrent
    threads = []
    pad = "x" * 64

    def worker(i: int):
        results[i] = stream_one_call(port, pad, max_tokens, seed=42 + i)

    t0 = time.perf_counter()
    for i in range(n_concurrent):
        th = threading.Thread(target=worker, args=(i,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=900.0)
    t_total = (time.perf_counter() - t0) * 1000.0

    ok = [r for r in results if r and not r.get("err") and r.get("completion_tokens")]
    errs = [r for r in results if not r or r.get("err")]

    summary = {
        "n_concurrent": n_concurrent,
        "wall_ms": t_total,
        "n_ok": len(ok),
        "n_err": len(errs),
        "errs": [r.get("err") if r else "missing" for r in errs],
        "per_stream": [],
    }

    if ok:
        prefill_mss = [r["prefill_ms"] for r in ok]
        decode_mss = [r["decode_ms"] for r in ok]
        comp_tokens = [r["completion_tokens"] for r in ok]
        per_stream_tps = [
            (ct / (dm / 1000.0)) if dm > 0 else 0.0
            for ct, dm in zip(comp_tokens, decode_mss)
        ]
        prompt_tokens_first = ok[0].get("prompt_tokens")
        summary.update({
            "prompt_tokens": prompt_tokens_first,
            "prefill_ms_mean":   sum(prefill_mss) / len(prefill_mss),
            "prefill_ms_min":    min(prefill_mss),
            "prefill_ms_max":    max(prefill_mss),
            "decode_ms_mean":    sum(decode_mss) / len(decode_mss),
            "completion_tokens_total": sum(comp_tokens),
            "decode_tps_per_stream_mean": sum(per_stream_tps) / len(per_stream_tps),
            "decode_tps_aggregate": sum(comp_tokens) / (max(decode_mss) / 1000.0),
            "per_stream": [
                {"prefill_ms": r["prefill_ms"],
                 "decode_ms": r["decode_ms"],
                 "completion_tokens": r["completion_tokens"],
                 "tps": tps}
                for r, tps in zip(ok, per_stream_tps)
            ],
        })

    return summary

# ----------------------------------------------------------------------
# Driver

def run_matrix(out_path: Path, max_tokens: int, only_models=None,
               only_topos=None, only_paths=None, only_concs=None,
               warmup_tokens: int = 32,
               cooldown_threshold_c: float = 85.0,
               cooldown_max_wait_s: float = 300.0,
               resume: bool = False) -> None:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for model in MODELS:
        if only_models and model.id not in only_models:
            continue
        for topo in TOPOLOGIES:
            if only_topos and topo.id not in only_topos:
                continue
            if not is_feasible(model, topo):
                cells.append({"model": model.id, "topo": topo.id, "skipped": "infeasible"})
                continue
            for path in PATHS:
                if only_paths and path not in only_paths:
                    continue
                cells.append({"model": model.id, "topo": topo.id, "path": path})

    # **--resume** — if the JSON already exists, load completed cells
    # and skip them. A cell counts as complete when it has at least
    # one entry in `concurrency_results` (any partial cells are
    # re-run from scratch — we don't merge mid-cell). Skipped/
    # infeasible cells preserved as-is.
    completed_keys: set[tuple[str, str, str]] = set()
    out: dict
    if resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            for c in existing.get("cells", []):
                if "skipped" in c:
                    completed_keys.add((c["model"], c["topo"], "_skipped"))
                elif c.get("concurrency_results"):
                    completed_keys.add((c["model"], c["topo"], c["path"]))
            out = existing
            print(f"resuming: {len(completed_keys)} cells already done", flush=True)
        except Exception as e:
            print(f"resume: ignoring unreadable {out_path}: {e}", flush=True)
            out = None
    else:
        out = None

    if out is None:
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        out = {
            "started": started,
            "max_tokens": max_tokens,
            "concurrencies": only_concs or CONCURRENCIES,
            "cells": [],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-write so partial results survive a crash.
    out_path.write_text(json.dumps(out, indent=2))

    for cell in cells:
        if "skipped" in cell:
            if (cell["model"], cell["topo"], "_skipped") not in completed_keys:
                out["cells"].append(cell)
                out_path.write_text(json.dumps(out, indent=2))
            continue
        if (cell["model"], cell["topo"], cell["path"]) in completed_keys:
            print(
                f"\n=== SKIP (resume) {cell['model']} / {cell['topo']} / {cell['path']} ===",
                flush=True,
            )
            continue
        model = next(m for m in MODELS if m.id == cell["model"])
        topo = next(t for t in TOPOLOGIES if t.id == cell["topo"])
        batched = (cell["path"] == "batched")

        slots = max(only_concs or CONCURRENCIES)

        log_name = f"{model.id}__{topo.id}__{cell['path']}.log"
        log_path = log_dir / log_name
        print(f"\n=== BOOT {model.id} / {topo.id} / {cell['path']} (slots={slots}) ===",
              flush=True)
        proc, port = boot_server(model, topo, batched, slots, log_path)
        boot_t0 = time.time()
        ready = wait_ready(port, timeout_s=600.0, model_id=model.id)
        boot_secs = time.time() - boot_t0
        if not ready:
            print(f"!! boot timeout after {boot_secs:.1f}s — log: {log_path}", flush=True)
            kill_server(proc)
            cell["err"] = f"boot timeout after {boot_secs:.1f}s"
            cell["log"] = str(log_path)
            out["cells"].append(cell)
            out_path.write_text(json.dumps(out, indent=2))
            continue

        cell["boot_secs"] = boot_secs
        cell["port"] = port
        cell["log"] = str(log_path)
        cell["concurrency_results"] = []

        try:
            # Warmup: one quick call (32 tokens) to shake out JIT.
            print(f"   warmup ({warmup_tokens} tok)...", flush=True)
            warm = stream_one_call(port, "x" * 64, warmup_tokens, seed=0)
            cell["warmup"] = {
                "prefill_ms": warm.get("prefill_ms"),
                "decode_ms": warm.get("decode_ms"),
                "completion_tokens": warm.get("completion_tokens"),
                "err": warm.get("err"),
            }

            for n_idx, n in enumerate(only_concs or CONCURRENCIES):
                # **2026-05-05** — temp gate between N variants too. Server
                # stays up across N=1,2,4,8 within a cell; without this,
                # GPUs heat across the cell and later N values measure on
                # a hotter (potentially throttled) chip than earlier N.
                if n_idx > 0 and cooldown_threshold_c > 0:
                    cooldown_until_safe(cooldown_threshold_c, cooldown_max_wait_s)
                print(f"   N={n}", end=" ", flush=True)
                t0 = time.perf_counter()
                res = run_cell(port, n, max_tokens)
                dur = time.perf_counter() - t0
                agg = res.get("decode_tps_aggregate", 0.0)
                per = res.get("decode_tps_per_stream_mean", 0.0)
                pref = res.get("prefill_ms_mean", 0.0)
                ok = res.get("n_ok", 0)
                err = res.get("n_err", 0)
                # Cumulative tps = aggregate (server's view, all streams).
                # Per-stream tps = what one user perceives (slower the more
                # concurrent users there are unless the path actually batches).
                print(
                    f"-> ok={ok} err={err} prefill_ms={pref:.0f} "
                    f"cum_tps={agg:.1f} per_stream_tps={per:.1f} ({dur:.1f}s)",
                    flush=True,
                )
                cell["concurrency_results"].append(res)
        finally:
            print("   killing server...", flush=True)
            kill_server(proc)
            time.sleep(3.0)

        out["cells"].append(cell)
        out_path.write_text(json.dumps(out, indent=2))

        # **2026-05-05** — temperature-gated GPU cooldown. MI50s under
        # sustained load climb 80→90+°C and start thermal-throttling,
        # which contaminates later cells. Block until any-GPU max temp
        # drops below `cooldown_threshold_c`, capped at
        # `cooldown_max_wait_s`. No sleep when temps are already cool.
        if cooldown_threshold_c > 0:
            cooldown_until_safe(cooldown_threshold_c, cooldown_max_wait_s)

    out["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--models", help="comma-separated subset of model ids")
    ap.add_argument("--topos", help="comma-separated subset of topo ids")
    ap.add_argument("--paths", help="comma-separated subset of paths")
    ap.add_argument("--concs", help="comma-separated subset of concurrencies")
    ap.add_argument("--cooldown-threshold-c", type=float, default=70.0,
                    help="°C — wait between cells until max GPU temp drops below this (default 70)")
    ap.add_argument("--cooldown-max-wait-s", type=float, default=300.0,
                    help="cap the cooldown wait per cell (default 300s)")
    ap.add_argument("--resume", action="store_true",
                    help="if --out already exists, skip cells that are already complete")
    args = ap.parse_args()

    only_models = set(args.models.split(",")) if args.models else None
    only_topos = set(args.topos.split(",")) if args.topos else None
    only_paths = set(args.paths.split(",")) if args.paths else None
    only_concs = [int(x) for x in args.concs.split(",")] if args.concs else None

    if not BIN.exists():
        print(f"missing {BIN} — build with: cargo build --release --features hip_serve -p flambeau-cli",
              file=sys.stderr)
        sys.exit(2)

    run_matrix(Path(args.out), args.max_tokens, only_models, only_topos,
               only_paths, only_concs,
               cooldown_threshold_c=args.cooldown_threshold_c,
               cooldown_max_wait_s=args.cooldown_max_wait_s,
               resume=args.resume)


if __name__ == "__main__":
    main()
