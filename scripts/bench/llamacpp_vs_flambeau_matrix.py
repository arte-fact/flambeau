#!/usr/bin/env python3
"""
Comparative bench: flambeau vs llama.cpp, OpenAI-compatible chat
completions, across (model, topology) cells. Each cell boots the
target engine, warms it, runs ONE long-prompt single-stream
streaming completion, measures TTFT + decode rate, then tears down.

Models (one of each {gemma, qwen} × {dense, MoE}):
  - Qwen3.5-9B-Q4_1                 (small dense Qwen)
  - gemma-4-31B-it-Q4_0             (medium dense Gemma, head_dim=512)
  - Qwen3.6-35B-A3B-Q4_0            (medium MoE Qwen)
  - gemma-4-26B-A4B-it-Q8_0         (medium MoE Gemma)

Topologies per model — only those that fit:
  - 9B-Q4_1:        tp2, pp2, pp4
  - 31B-Q4_0:       tp2, pp2, pp4, pp2tp2
  - 35B-A3B-Q4_0:   pp2, pp4, pp2tp2   (tp2 marginal at 10 GB/rank + KV)
  - 26B-A4B-Q8_0:   pp2, pp4, pp2tp2   (tp2 OOMs at ~14 GB/rank)

llama.cpp topology mapping:
  - tp2  : -sm row   -ts 1,1     HIP_VISIBLE_DEVICES=0,2
  - pp2  : -sm layer -ts 1,1     HIP_VISIBLE_DEVICES=0,2
  - pp4  : -sm layer -ts 1,1,1,1 HIP_VISIBLE_DEVICES=0,1,2,3
  - pp2tp2: no native equivalent — approximated as -sm layer on 4 GPUs
            (= pp4). Labelled as `pp2tp2 (llama≈pp4)` in the output.

Output: markdown table on stdout; full JSON dumped to --out.

Usage:
    python3 scripts/bench/llamacpp_vs_flambeau_matrix.py \
        --out certs/perf/llamacpp_vs_flambeau_matrix.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ROCM_LD = "/opt/rocm-host/lib:/artefact/llama.cpp/build-mi50/bin"
ROCBLAS_TENSILE = "/opt/rocm-host/lib/rocblas/library"
LLAMA_BIN = "/artefact/llama.cpp/build-mi50/bin/llama-server"
FLAMBEAU_BIN = ROOT / "target/release/flambeau"

# Long prompt: ~600+ tokens after tokenization. Same prompt across all
# cells so prefill timings are directly comparable.
PROMPT = (
    "/no_think You are a senior compiler engineer. Write a detailed "
    "technical explanation of how modern AMD GPUs (gfx906 / CDNA-1) "
    "implement integer dot-product acceleration via the V_DOT4_I32_I8 "
    "instruction. Cover: (1) the instruction's encoding and operand "
    "layout including the sign-extension semantics for each int8 lane; "
    "(2) how it maps to the wave64 SIMD execution model and the per-VGPR "
    "register pressure implications when fused into a matrix-vector "
    "multiplication kernel; (3) the practical role of dp4a in quantized "
    "matrix multiplication kernels (e.g., Q4_0 / Q8_0 GEMV), with "
    "particular attention to how the per-block scale factor is folded "
    "into the accumulation, why this typically reduces the per-element "
    "FMA count by ~4×, and what the resulting arithmetic intensity "
    "implies for memory-bound vs compute-bound regimes on HBM2 silicon; "
    "(4) the comparison with NVIDIA's __dp4a intrinsic on sm_61+ and "
    "the trade-offs between the two ISAs for portable mixed-precision "
    "kernels; (5) interactions with LDS double-buffering, software "
    "pipelining, and the wave/SIMD scheduler's ability to hide HBM "
    "latency with overlapping dp4a chains; (6) a step-by-step worked "
    "example showing a single 32-element block of int8 K against a "
    "Q8_0-quantized Q vector, tracing each dp4a invocation, partial-sum "
    "accumulation, scalar dequant of K.d × Q.d at the block boundary, "
    "and the final reduction. Be exhaustive and quote concrete numbers "
    "for both the AMD and NVIDIA paths where they differ. Aim for "
    "several hundred words."
)
MAX_TOKENS = 192
WARMUP_TOKENS = 16
SETTLE_SLEEP_S = 4.0


# -------------------- Topology + model specs --------------------

@dataclasses.dataclass(frozen=True)
class Topo:
    id: str
    flambeau_devices: str
    flambeau_mesh_mode: str
    flambeau_pp: int
    flambeau_tp: int
    llama_hip_visible: str
    llama_split_mode: str
    llama_tensor_split: str
    note: str = ""


TOPOLOGIES = {
    "tp2": Topo(
        id="tp2",
        flambeau_devices="hip:0,2",
        flambeau_mesh_mode="tp",
        flambeau_pp=0,
        flambeau_tp=2,
        llama_hip_visible="0,2",
        llama_split_mode="row",
        llama_tensor_split="1,1",
    ),
    "pp2": Topo(
        id="pp2",
        flambeau_devices="hip:0,2",
        flambeau_mesh_mode="pp",
        flambeau_pp=0,
        flambeau_tp=0,
        llama_hip_visible="0,2",
        llama_split_mode="layer",
        llama_tensor_split="1,1",
    ),
    "pp4": Topo(
        id="pp4",
        flambeau_devices="hip:0,1,2,3",
        flambeau_mesh_mode="pp",
        flambeau_pp=0,
        flambeau_tp=0,
        llama_hip_visible="0,1,2,3",
        llama_split_mode="layer",
        llama_tensor_split="1,1,1,1",
    ),
    "pp2tp2": Topo(
        id="pp2tp2",
        flambeau_devices="hip:0,2,1,3",
        flambeau_mesh_mode="pp+tp",
        flambeau_pp=2,
        flambeau_tp=2,
        llama_hip_visible="0,1,2,3",
        llama_split_mode="layer",
        llama_tensor_split="1,1,1,1",
        note="llama.cpp has no native 2D mesh; approximated as -sm layer (pp4-equivalent)",
    ),
}


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str       # "qwen" or "gemma"
    kind: str         # "dense" or "moe"
    path: str
    topologies: tuple[str, ...]
    ctx_cap: int = 4096


MODELS = [
    ModelSpec(
        id="qwen35-9b-q4_1",
        family="qwen",
        kind="dense",
        path="/artefact/models/Qwen3.5-9B-Q4_1.gguf",
        topologies=("tp2", "pp2", "pp4"),
    ),
    ModelSpec(
        id="gemma4-31b-q4_0",
        family="gemma",
        kind="dense",
        path="/artefact/models/gemma-4-31B-it-Q4_0.gguf",
        topologies=("tp2", "pp2", "pp4", "pp2tp2"),
    ),
    ModelSpec(
        id="qwen36-35b-a3b-q4_0",
        family="qwen",
        kind="moe",
        path="/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
        topologies=("tp2", "pp2", "pp4", "pp2tp2"),
    ),
    ModelSpec(
        id="gemma4-26b-a4b-q8_0",
        family="gemma",
        kind="moe",
        path="/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf",
        topologies=("pp2", "pp4", "pp2tp2"),
    ),
]


# -------------------- Server lifecycle --------------------

def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def boot_flambeau(model: ModelSpec, topo: Topo, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + ROCM_LD
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["FLAMBEAU_KV"] = "f16"
    args = [
        str(FLAMBEAU_BIN), "serve",
        "--model", model.path,
        "--devices", topo.flambeau_devices,
        "--mesh-mode", topo.flambeau_mesh_mode,
        "--port", str(port),
        "--ctx-cap", str(model.ctx_cap),
        # Single-stream bench — 1 inflight slot for all models (also
        # required for gemma4, which still gates `slots > 1`).
        "--inflight-slots", "1",
    ]
    if topo.flambeau_mesh_mode == "pp+tp":
        args += ["--pp-size", str(topo.flambeau_pp),
                 "--tp-size", str(topo.flambeau_tp)]
    elif topo.flambeau_mesh_mode == "tp":
        args += ["--tp-size", str(topo.flambeau_tp)]
    log = open(f"/tmp/bench_matrix_flambeau_{model.id}_{topo.id}.log", "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def boot_llamacpp(model: ModelSpec, topo: Topo, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ROCM_LD
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["HIP_VISIBLE_DEVICES"] = topo.llama_hip_visible
    args = [
        LLAMA_BIN,
        "--model", model.path,
        "--port", str(port),
        "-ngl", "999",
        "--split-mode", topo.llama_split_mode,
        "--tensor-split", topo.llama_tensor_split,
        "--ctx-size", str(model.ctx_cap),
        "--flash-attn", "on",
        "--threads", "8",
        "--no-mmap",
    ]
    log = open(f"/tmp/bench_matrix_llamacpp_{model.id}_{topo.id}.log", "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, timeout_s: float = 600.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def kill_proc(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        pass


# -------------------- Per-cell measurement --------------------

@dataclasses.dataclass
class CellResult:
    engine: str
    model: str
    topo: str
    ptok: int = 0
    ctok: int = 0
    ttft_ms: float = 0.0
    decode_ms_per_tok: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    err: str | None = None


def discover_model_id(port: int) -> str:
    r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
    return r.json()["data"][0]["id"]


def stream_one(port: int, model_id: str, prompt: str, max_tokens: int
               ) -> tuple[int, int, float, float]:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.time()
    first = last = None
    n = 0
    ptok = ctok = 0
    with requests.post(url, json=body, stream=True, timeout=600.0) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if first is None:
                        first = time.time()
                    last = time.time()
                    n += 1
            usage = obj.get("usage")
            if usage:
                ctok = usage.get("completion_tokens", 0)
                ptok = usage.get("prompt_tokens", 0)
    if first is None or last is None or n < 2:
        return ptok, ctok, 0.0, 0.0
    if ctok == 0:
        ctok = n
    ttft_ms = (first - t0) * 1000.0
    decode_ms_per_tok = ((last - first) * 1000.0) / max(n - 1, 1)
    return ptok, ctok, ttft_ms, decode_ms_per_tok


def measure_cell(engine: str, model: ModelSpec, topo: Topo, boot_fn
                 ) -> CellResult:
    label = f"{engine} | {model.id} | {topo.id}"
    print(f"\n=== {label} booting ...", flush=True)
    port = free_port()
    proc = boot_fn(model, topo, port)
    if not wait_ready(port, timeout_s=600.0):
        kill_proc(proc)
        return CellResult(engine=engine, model=model.id, topo=topo.id,
                          err="boot timeout")
    try:
        model_id = discover_model_id(port)
        # Warm
        try:
            requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": model_id,
                      "messages": [{"role": "user", "content": "Hello."}],
                      "stream": False, "max_tokens": WARMUP_TOKENS,
                      "temperature": 0},
                timeout=120.0,
            )
        except Exception as e:
            print(f"  warmup: {e}", flush=True)

        ptok, ctok, ttft_ms, dec_ms = stream_one(port, model_id, PROMPT, MAX_TOKENS)
        if ttft_ms == 0.0 or dec_ms == 0.0:
            return CellResult(engine=engine, model=model.id, topo=topo.id,
                              err="empty stream")
        prefill_tps = ptok / (ttft_ms / 1000.0) if ttft_ms > 0 else 0.0
        decode_tps = 1000.0 / dec_ms if dec_ms > 0 else 0.0
        print(f"  ptok={ptok:5d} ctok={ctok:4d} "
              f"ttft={ttft_ms:7.0f}ms "
              f"prefill={prefill_tps:7.1f} tok/s "
              f"decode={decode_tps:6.2f} tok/s",
              flush=True)
        return CellResult(
            engine=engine, model=model.id, topo=topo.id,
            ptok=ptok, ctok=ctok, ttft_ms=ttft_ms,
            decode_ms_per_tok=dec_ms,
            prefill_tps=prefill_tps, decode_tps=decode_tps,
        )
    except Exception as e:
        return CellResult(engine=engine, model=model.id, topo=topo.id,
                          err=f"exception: {e}")
    finally:
        kill_proc(proc)
        time.sleep(SETTLE_SLEEP_S)


# -------------------- Report --------------------

def render_markdown(cells: list[CellResult]) -> str:
    by_key: dict[tuple[str, str], dict[str, CellResult]] = {}
    for c in cells:
        by_key.setdefault((c.model, c.topo), {})[c.engine] = c

    lines = []
    lines.append("# llama.cpp vs flambeau — single-stream chat completions\n")
    lines.append(f"_prompt ≈ 600+ tokens (see PROMPT); max_tokens={MAX_TOKENS}; "
                 f"temp=0.7 top_p=0.9 seed=0; streaming SSE._\n")
    lines.append("\n| model | topo | engine | ptok | ctok | "
                 "prefill (tok/s) | decode (tok/s) | note |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---|\n")
    for (model, topo) in sorted(by_key.keys()):
        for engine in ("flambeau", "llama.cpp"):
            c = by_key[(model, topo)].get(engine)
            if c is None or c.err:
                err = c.err if c else "missing"
                lines.append(f"| {model} | {topo} | {engine} | – | – | – | – | "
                             f"_{err}_ |\n")
            else:
                note = TOPOLOGIES[topo].note if (engine == "llama.cpp" and
                                                  TOPOLOGIES[topo].note) else ""
                lines.append(f"| {model} | {topo} | {engine} | "
                             f"{c.ptok} | {c.ctok} | "
                             f"{c.prefill_tps:.1f} | {c.decode_tps:.2f} | "
                             f"{note} |\n")
    lines.append("\n## Ratios (flambeau / llama.cpp)\n")
    lines.append("\n| model | topo | prefill ratio | decode ratio |\n")
    lines.append("|---|---|---:|---:|\n")
    for (model, topo) in sorted(by_key.keys()):
        fb = by_key[(model, topo)].get("flambeau")
        lc = by_key[(model, topo)].get("llama.cpp")
        if not (fb and lc and not fb.err and not lc.err):
            continue
        pr_s = (f"{fb.prefill_tps / lc.prefill_tps:.2f}×"
                if (lc.prefill_tps > 0 and fb.prefill_tps > 0) else "n/a")
        dr_s = (f"{fb.decode_tps / lc.decode_tps:.2f}×"
                if lc.decode_tps > 0 else "n/a")
        lines.append(f"| {model} | {topo} | {pr_s} | {dr_s} |\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="Output JSON path (markdown printed to stdout)")
    ap.add_argument("--engines", default="flambeau,llama.cpp",
                    help="Comma-separated subset of {flambeau, llama.cpp}")
    ap.add_argument("--models", default=",".join(m.id for m in MODELS),
                    help="Comma-separated subset of model ids")
    ap.add_argument("--topos", default="",
                    help="Optional comma-separated topo filter "
                         "(e.g. 'tp2,pp4'); default: each model's feasible set")
    args = ap.parse_args()
    if not FLAMBEAU_BIN.exists():
        print(f"missing {FLAMBEAU_BIN}", file=sys.stderr); sys.exit(2)
    if not Path(LLAMA_BIN).exists():
        print(f"missing {LLAMA_BIN}", file=sys.stderr); sys.exit(2)

    engines = set(args.engines.split(","))
    keep_models = set(args.models.split(","))
    topo_filter = set(args.topos.split(",")) if args.topos else None

    cells: list[CellResult] = []
    for model in MODELS:
        if model.id not in keep_models:
            continue
        for topo_id in model.topologies:
            if topo_filter and topo_id not in topo_filter:
                continue
            topo = TOPOLOGIES[topo_id]
            if "flambeau" in engines:
                cells.append(measure_cell("flambeau", model, topo, boot_flambeau))
            if "llama.cpp" in engines:
                cells.append(measure_cell("llama.cpp", model, topo, boot_llamacpp))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([dataclasses.asdict(c) for c in cells],
                                    indent=2))
    md = render_markdown(cells)
    print("\n" + md)


if __name__ == "__main__":
    main()
