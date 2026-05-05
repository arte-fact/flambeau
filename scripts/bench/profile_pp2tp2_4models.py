#!/usr/bin/env python3
"""rocprofv3 kernel-trace profile across 4 Qwen variants on pp2tp2.

Companion to `matrix_pp2tp2_4models.py`: where that one measured wall-
clock + VRAM + GPU%, this one collects per-kernel GPU time so we can
identify common levers (kernels that dominate across the family) and
model-specific levers.

Per model:
  1. Boot `flambeau serve` under
        rocprofv3 --kernel-trace --stats -d <out>/<model>
     with the same feature set used in the matrix bench.
  2. Wait for ready, fire one short profiling request
     (L≈512 prompt, max_tokens=64) so the trace covers exactly one
     prefill + decode cycle without ballooning.
  3. SIGTERM the server (cleanly via the process group); rocprofv3
     finalises *_kernel_stats.csv per agent.
  4. Aggregate stats across the 4 GPU agents into a single per-model
     top-N table.
  5. Stop server, cooldown, next model.

Final cert: per-model top-20 + a cross-model comparison column
(% of total kernel time taken by each kernel-family on each model)
so common levers stand out.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "flambeau"
ROCPROFV3 = "/opt/rocm-7.1.1/core-7.13/bin/rocprofv3"
DEVICES = "hip:0,2,1,3"
PP_SIZE = 2
TP_SIZE = 2

# (id, gguf_path, ctx_cap)
MODELS = [
    ("qwen35_9B_q4_1",
     "/artefact/models/Qwen3.5-9B-Q4_1.gguf", 8192),
    ("qwen36_27B_q4_1",
     "/artefact/models/Qwen3.6-27B-Q4_1.gguf", 8192),
    ("qwen36_35B_a3b_q4_0",
     "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", 8192),
    ("qwen3_coder_next_q4_0",
     "/artefact/models/Qwen3-Coder-Next-Q4_0.gguf", 8192),
]

PROFILE_PROMPT_TOKENS = 512   # short enough to keep one prefill chunk
PROFILE_MAX_TOKENS = 64       # decode steps captured in trace

# A 512-token-ish dummy prompt — content doesn't matter for kernel
# profiling, only shape does. We keep it lexically simple so every
# tokenizer agrees on the count within ±5 %.
PROFILE_PROMPT = (
    "Write a thorough, structured explanation, of the following topic. "
    "The topic is: how a modern operating-system scheduler decides which "
    "thread to run next on a multi-core CPU. Cover the data structures, "
    "the runqueue model, the tick callback, vruntime in CFS, EEVDF in "
    "Linux 6.6 and later, the load-balancer hierarchy, NUMA balancing, "
    "the deadline class with EDF and budget enforcement, real-time "
    "priority inheritance via rt_mutex, the idle path with cpuidle and "
    "schedutil, EAS for big.LITTLE, cgroup-v2 cpu.weight and cpu.max, "
    "and SCHED_DEADLINE admission control via __dl_overflow. Each "
    "section should give one concrete example referencing a kernel "
    "source path. Pay attention to interactions between scheduling and "
    "frequency selection, and between scheduling and PMU events. The "
    "answer should be at least 600 words and use plain prose. Avoid "
    "bullet points; use numbered subsections and short paragraphs. "
    "Begin the answer immediately, with no preface or apology. "
) * 6  # ~600 short tokens — close to PROFILE_PROMPT_TOKENS once tokenized


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def boot_under_rocprofv3(model_path: str, ctx_cap: int, out_dir: Path,
                          log_path: Path):
    port = free_port()
    env = os.environ.copy()
    env["FLAMBEAU_CTX_CAP"] = str(ctx_cap)
    env["FLAMBEAU_INFLIGHT_SLOTS"] = "4"
    env["FLAMBEAU_GPU_SAMPLER"] = "1"
    env["FLAMBEAU_BATCHED_DECODE"] = "1"
    env["FLAMBEAU_PREFIX_CACHE"] = "1"
    env["FLAMBEAU_PREFIX_CACHE_MAX_GB"] = "4"
    env["FLAMBEAU_MAX_QUEUE_DEPTH"] = "32"
    env["FLAMBEAU_PREFILL_UBATCH"] = "512"
    env["RUST_LOG"] = "info"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        ROCPROFV3,
        "--kernel-trace",
        "--stats",
        "--truncate-kernels",
        "--summary-units", "msec",
        # rocprofv3 defaults to the `rocpd` SQLite DB; force CSV so
        # the per-agent kernel_stats.csv files are emitted alongside.
        "-f", "csv",
        "-d", str(out_dir),
        "-o", "trace",
        "--",
        str(BIN), "serve",
        "--model", model_path,
        "--devices", DEVICES,
        "--mesh-mode", "pp+tp",
        "--pp-size", str(PP_SIZE),
        "--tp-size", str(TP_SIZE),
        "--port", str(port),
    ]
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


def fire_profile_request(port: int) -> dict:
    body = {
        "model": "anything",
        "messages": [
            {"role": "user", "content": PROFILE_PROMPT},
        ],
        "max_tokens": PROFILE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
        "seed": 42,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as cli:
            r = cli.post(url, json=body)
            r.raise_for_status()
            data = r.json()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        u = data.get("usage", {})
        return {
            "wall_ms": wall_ms,
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
        }
    except Exception as e:
        return {"err": repr(e)}


def stop_server(proc: subprocess.Popen):
    """SIGTERM the whole process group so rocprofv3 finalises stats
    cleanly, then wait for it to exit."""
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


# ---- stats parsing ------------------------------------------------------

def find_kernel_stats_csvs(out_dir: Path) -> list[Path]:
    """rocprofv3 emits per-agent files like
    `<out>/trace_kernel_stats.csv` and `<out>/<host>/<pid>/trace_kernel_stats.csv`
    depending on version. Walk the directory tree and grab any file
    matching `*kernel_stats.csv`."""
    return sorted(out_dir.rglob("*kernel_stats.csv"))


def aggregate_kernel_stats(csv_paths: list[Path]) -> list[dict]:
    """Combine kernel_stats.csv across all agents/ranks into one
    per-kernel rollup of (calls, total_ns, avg_ns)."""
    by_name: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_ns": 0})
    for p in csv_paths:
        with p.open() as f:
            r = csv.DictReader(f)
            name_col = "Name" if "Name" in (r.fieldnames or []) else "Kernel_Name"
            calls_col = "Calls" if "Calls" in (r.fieldnames or []) else "Count"
            total_col = None
            for cand in ("TotalDurationNs", "TotalDuration_ns", "Total Duration (ns)",
                         "TotalDurationMsec", "Total Duration (msec)",
                         "Total Duration (ms)"):
                if cand in (r.fieldnames or []):
                    total_col = cand
                    break
            if total_col is None:
                # fall back to summing PercentageDuration if present (less ideal)
                total_col = "PercentageDuration"
            unit_ns = 1
            if total_col and "msec" in total_col.lower() or "ms" in (total_col or "").lower():
                unit_ns = 1_000_000
            for row in r:
                name = row.get(name_col, "?")
                try:
                    calls = int(row.get(calls_col, "0") or "0")
                except ValueError:
                    calls = 0
                try:
                    total = float(row.get(total_col, "0") or "0") * unit_ns
                except ValueError:
                    total = 0
                by_name[name]["calls"] += calls
                by_name[name]["total_ns"] += int(total)
    out = [
        {"name": n, "calls": d["calls"], "total_ns": d["total_ns"],
         "avg_ns": (d["total_ns"] // d["calls"]) if d["calls"] else 0}
        for n, d in by_name.items()
    ]
    out.sort(key=lambda r: r["total_ns"], reverse=True)
    return out


def fmt_per_model_table(stats: list[dict], top_n: int = 20) -> str:
    if not stats:
        return "(no stats)"
    grand = sum(s["total_ns"] for s in stats) or 1
    lines = ["| rank | kernel | calls | total ms | avg µs | pct |",
             "|---:|---|---:|---:|---:|---:|"]
    for i, s in enumerate(stats[:top_n], 1):
        pct = 100.0 * s["total_ns"] / grand
        lines.append(
            f"| {i} | `{s['name']}` | {s['calls']} | "
            f"{s['total_ns'] / 1_000_000:.2f} | {s['avg_ns'] / 1_000:.2f} | {pct:.2f}% |"
        )
    lines.append(f"\nTotal kernel time across all ranks: **{grand / 1_000_000:.0f} ms**.")
    return "\n".join(lines)


# ---- kernel family classifier (cross-model lens) ------------------------

KERNEL_FAMILIES = [
    ("attn_decode",     ("attention_decode_f", "gqa_decode_gemv", "fattn", "flash_attn_decode")),
    ("attn_prefill",    ("attention_prefill", "fattn_prefill")),
    ("gdn",             ("gdn_state_step", "gated_delta_net", "gdn_")),
    ("mmvq_q4",         ("mmvq_q4_", "moe_mmvq_q4_")),
    ("mmvq_q5",         ("mmvq_q5_",)),
    ("mmvq_q6_q8",      ("mmvq_q6_", "mmvq_q8_")),
    ("mmq_4warp_lds",   ("mmq_q4_0_4warp_lds", "mmq_q4_1_4warp_lds", "mmq_q5_", "mmq_q6_", "mmq_q8_0_wave64")),
    ("moe_mmq_prefill", ("indexed_moe_mmq", "moe_sort_scatter")),
    ("moe_mmvq_decode", ("indexed_moe_mmvq",)),
    ("rmsnorm",         ("rmsnorm",)),
    ("rope",            ("rope_",)),
    ("quantize",        ("quantize_row", "swiglu_f32_to_q8")),
    ("cast",            ("cast_f32_f16", "cast_f16_f32", "cast_f32_bf16")),
    ("p2p_allreduce",   ("p2p_allreduce", "allreduce")),
    ("topk_softmax",    ("topk_softmax", "softmax_apply")),
    ("dense_gemv",      ("dense_gemv", "f16_f16_batched")),
    ("output_head",     ("output_head", "head_decode")),
    ("misc",            ()),
]

def family_of(name: str) -> str:
    for fam, prefixes in KERNEL_FAMILIES:
        for p in prefixes:
            if p in name:
                return fam
    return "misc"


def family_rollup(stats: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {fam: {"total_ns": 0, "calls": 0} for fam, _ in KERNEL_FAMILIES}
    for s in stats:
        fam = family_of(s["name"])
        out[fam]["total_ns"] += s["total_ns"]
        out[fam]["calls"] += s["calls"]
    return out


def fmt_family_matrix(per_model_families: dict[str, dict[str, dict]]) -> str:
    """Cross-model comparison: rows = kernel families, cols = models,
    values = % of that model's total kernel time."""
    model_ids = list(per_model_families.keys())
    grand_per_model = {
        m: max(1, sum(d["total_ns"] for d in fams.values()))
        for m, fams in per_model_families.items()
    }
    lines = ["| family | " + " | ".join(model_ids) + " |",
             "|---|" + "---:|" * len(model_ids)]
    for fam, _ in KERNEL_FAMILIES:
        cells = []
        for m in model_ids:
            ns = per_model_families[m].get(fam, {}).get("total_ns", 0)
            pct = 100.0 * ns / grand_per_model[m]
            cells.append(f"{pct:.1f}%" if ns else "—")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- main ---------------------------------------------------------------

def run_one(model_id: str, model_path: str, ctx_cap: int, out_root: Path) -> dict:
    if not Path(model_path).exists():
        return {"id": model_id, "err": f"GGUF missing: {model_path}"}
    out_dir = out_root / model_id
    log_path = out_dir / "server.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_id}] booting under rocprofv3 → {out_dir}", flush=True)
    t_boot = time.perf_counter()
    proc, port = boot_under_rocprofv3(model_path, ctx_cap, out_dir, log_path)
    try:
        if not wait_ready(port, log_path):
            return {"id": model_id, "err": "boot failed", "log": str(log_path)}
        load_s = time.perf_counter() - t_boot
        # Warmup (untraced kernels are still traced — but the GPU now
        # holds rocBLAS-compiled gemm Tensile, JIT etc; first request
        # would be artificially slow and inflate launch counts).
        _ = fire_profile_request(port)
        time.sleep(0.5)
        print(f"[{model_id}] ready in {load_s:.1f}s; firing profile request", flush=True)
        r = fire_profile_request(port)
        r["load_s"] = load_s
        r["id"] = model_id
        return r
    finally:
        # SIGTERM forces rocprofv3 to finalise the per-agent stats CSVs.
        stop_server(proc)


def main():
    out_root = ROOT / "certs" / "perf" / "profile_pp2tp2_4models"
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    per_model_top: dict[str, list[dict]] = {}
    per_model_families: dict[str, dict[str, dict]] = {}
    for spec in MODELS:
        meta = run_one(*spec, out_root)
        summary.append(meta)
        if meta.get("err"):
            print(f"[{meta['id']}] ERROR: {meta['err']}")
            continue
        csvs = find_kernel_stats_csvs(out_root / meta["id"])
        if not csvs:
            print(f"[{meta['id']}] no kernel_stats.csv emitted; check log")
            continue
        stats = aggregate_kernel_stats(csvs)
        per_model_top[meta["id"]] = stats
        per_model_families[meta["id"]] = family_rollup(stats)
        time.sleep(15)  # cooldown
    # Write per-model + cross-model cert.
    md = ["# pp2tp2 4-model kernel profile (rocprofv3 --kernel-trace)",
          "",
          f"Topology: pp2tp2 on `{DEVICES}` (pp_size={PP_SIZE}, tp_size={TP_SIZE}).",
          "Profile request shape: ~512-token prompt + 64 decode tokens, greedy, "
          "one warmup request before the traced one.",
          "Server features: FLAMBEAU_BATCHED_DECODE=1, GPU_SAMPLER=1, "
          "PREFIX_CACHE=1, 4 inflight slots, 512-token prefill ubatch.",
          "",
          "## Cross-model kernel-family share (% of total kernel time per model)",
          "",
          fmt_family_matrix(per_model_families),
          "",
          "_Read this matrix to spot **common levers**: a row hot on every "
          "model is a family-wide lever; a row hot on one model is a "
          "model-specific lever._",
          "",
          "## Per-model top-20 kernels",
          "",
          ]
    for mid, stats in per_model_top.items():
        md.append(f"### {mid}")
        md.append("")
        md.append(fmt_per_model_table(stats, top_n=20))
        md.append("")
    md.append("## Run summary")
    md.append("")
    md.append("| model | load (s) | wall ms (traced) | prompt tok | gen tok |")
    md.append("|---|---:|---:|---:|---:|")
    for r in summary:
        if r.get("err"):
            md.append(f"| {r['id']} | — | — | — | — | (err: {r['err']}) |")
            continue
        md.append(
            f"| {r['id']} | {r.get('load_s', 0):.1f} | "
            f"{r.get('wall_ms', 0):.0f} | {r.get('prompt_tokens', 0)} | "
            f"{r.get('completion_tokens', 0)} |"
        )
    cert_path = out_root / "cert.md"
    cert_path.write_text("\n".join(md))
    raw_path = out_root / "raw.json"
    raw_path.write_text(json.dumps({
        "summary": summary,
        "per_model_top": per_model_top,
        "per_model_families": per_model_families,
    }, indent=2, default=str))
    print("\n" + "\n".join(md))
    print(f"\ncert: {cert_path}")


if __name__ == "__main__":
    main()
