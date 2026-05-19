#!/usr/bin/env python3
"""
Comparative bench: flambeau v2 vs flambeau legacy vs llama.cpp on
gemma-4 models. Long technical prompt (~700 tok prefill) + 128 decode.
Same shape as `real_task_pp1024_tg128.py` so numbers are directly
comparable to the qwen-side runs.

Cases (override via --only):
  - gemma-4-E4B-it-Q4_0   SD  (hip:0)
  - gemma-4-31B-it-Q4_0   TP2 (hip:0,1)
  - gemma-4-26B-A4B-it-Q8_0  PP2 (hip:0,1) — MoE variant

Each case spawns flambeau (v2 then legacy) and llama.cpp sequentially
on a fresh process so VRAM doesn't carry between arms.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ROCM_LD = "/opt/rocm-host/lib"
ROCBLAS_TENSILE = "/opt/rocm-host/lib/rocblas/library"
LLAMA_BIN = "/artefact/llama.cpp/build-mi50/bin/llama-server"
FLAMBEAU_BIN = ROOT / "target/release/flambeau"

PROMPT_BASE = (
    "You are a senior compiler engineer at a large GPU vendor. "
    "Write an exhaustive technical explanation that covers, in order: "
    "(1) how modern AMD GPUs of the CDNA-1 / gfx906 generation implement "
    "integer dot-product acceleration via the V_DOT4_I32_I8 instruction. "
    "Cover the instruction's binary encoding, the sign-extension semantics "
    "for each int8 lane, the operand layout, and the per-cycle throughput "
    "characteristics on a wave64 SIMD. "
    "(2) Walk through the per-VGPR register pressure implications when this "
    "instruction is fused into a quantized matrix-vector multiplication "
    "kernel, including the impact on waves-per-EU occupancy at varying "
    "block tile shapes (32-element, 64-element, 128-element K tiles), and "
    "explain why crossing the 32-VGPR boundary forces a drop from 10 to 8 "
    "waves per SIMD. "
    "(3) Describe the practical role of dp4a in Q4_0 and Q8_0 mmvq kernels, "
    "with concrete numerics: how the per-block scale factor is folded into "
    "the accumulation, why this typically reduces the per-element FMA count "
    "by approximately 4x, and what the resulting arithmetic intensity "
    "(measured in FLOP per byte of HBM2 traffic) implies for whether such a "
    "kernel is memory-bound or compute-bound on gfx906 with its 1 TB/s HBM2 "
    "and 13.4 TFLOPS F32 peak. "
    "(4) Compare against the NVIDIA __dp4a intrinsic on sm_61+ Pascal and "
    "later, contrasting the encoding, the throughput per SM, and the "
    "implications for portable mixed-precision quantized kernels that have "
    "to dispatch on both ISAs without sacrificing more than a small constant "
    "factor of theoretical peak on either side. "
    "(5) Discuss the interactions with LDS (local data share) double "
    "buffering and software pipelining: how loading the next K tile into LDS "
    "while the dp4a chain on the current tile is in flight enables full "
    "memory-throughput overlap, the role of the wave-scheduler's ability to "
    "swap waves on stall, and the typical asm-level patterns (LDS read into "
    "VGPR pair, dp4a, LDS write of next tile, repeat) for sustained "
    "throughput. "
    "(6) Provide a fully worked numerical example: a single 32-element block "
    "of int8 K weights against a Q8_0-quantized Q activation vector. "
    "Trace each of the 8 dp4a invocations, the per-invocation int32 partial "
    "sum, the scalar dequant step at the block boundary where K.d (the FP16 "
    "scale factor) and Q.d (the FP32 activation scale) are multiplied into "
    "the int32 accumulator and accumulated as F32, and the final reduction "
    "to produce the dot product. "
    "(7) Finally, summarise the practical takeaways for kernel authors: "
    "when dp4a wins on gfx906 (high arithmetic intensity, large K dimension, "
    "tightly fused Q-quant), when it loses (single-row mmvq decode where "
    "weight HBM traffic dominates), and the typical performance signatures "
    "to look for in rocprofv3 PMC traces to confirm a dp4a kernel is "
    "running near peak. "
    "Be exhaustive, quote concrete numbers wherever they differ between "
    "ISAs, and don't skim sections — each subsection deserves its own "
    "multi-paragraph treatment."
)


@dataclass
class Case:
    label: str
    model: str
    flambeau_mesh: str    # "sd" | "tp2" | "pp2" | "pp4" | "pp2tp2"
    flambeau_devices: str # e.g. "0" | "0,1" | "0,2,1,3"
    # llama.cpp can't do TP-of-PP; for `pp2tp2` we map to llama's `pp4`
    # (`--split-mode layer`). TP4 (`--split-mode row` across 4 GPUs) is
    # banned — it routes AllReduce through the {2,3} link-faulted pair
    # on this rig. Other meshes round-trip identically.
    llamacpp_mesh: str
    llamacpp_devices: str


CASES: list[Case] = [
    Case("gemma4-E4B-Q4_0",   "/artefact/models/gemma-4-E4B-it-Q4_0.gguf",
         flambeau_mesh="sd",  flambeau_devices="0",
         llamacpp_mesh="sd",  llamacpp_devices="0"),
    Case("gemma4-31B-Q4_0",   "/artefact/models/gemma-4-31B-it-Q4_0.gguf",
         flambeau_mesh="tp2", flambeau_devices="0,1",
         llamacpp_mesh="tp2", llamacpp_devices="0,1"),
    # 4-GPU Q8_0: flambeau picks pp2tp2 (the user's prod topology);
    # llama.cpp uses pp4 on the same physical 4 GPUs.
    Case("gemma4-31B-Q8_0-pp2tp2", "/artefact/models/gemma-4-31B-it-Q8_0.gguf",
         flambeau_mesh="pp2tp2", flambeau_devices="0,2,1,3",
         llamacpp_mesh="pp4",    llamacpp_devices="0,2,1,3"),
    # 4-GPU Q8_0 apples-to-apples PP4 on both stacks.
    Case("gemma4-31B-Q8_0-pp4", "/artefact/models/gemma-4-31B-it-Q8_0.gguf",
         flambeau_mesh="pp4", flambeau_devices="0,1,2,3",
         llamacpp_mesh="pp4", llamacpp_devices="0,1,2,3"),
    Case("gemma4-26B-A4B-Q8_0", "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf",
         flambeau_mesh="pp2", flambeau_devices="0,1",
         llamacpp_mesh="pp2", llamacpp_devices="0,1"),
]


@dataclass
class Turn:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    decode_ms_per_tok: float = 0.0
    wall_ms: float = 0.0


@dataclass
class ArmResult:
    label: str
    err: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    text_head: str = ""


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def boot_flambeau(model_path: str, port: int, mesh: str, devices: str, v2: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ROCM_LD}:" + env.get("LD_LIBRARY_PATH", "")
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["FLAMBEAU_INFLIGHT_SLOTS"] = "1"
    env["FLAMBEAU_KV"] = "f16"
    env["FLAMBEAU_V2"] = "1" if v2 else "0"
    args = [
        str(FLAMBEAU_BIN), "serve",
        "--model", model_path,
        "--devices", devices,
        "--port", str(port),
        "--ctx-cap", "4096",
    ]
    n_dev = len([d for d in devices.split(",") if d.strip()])
    if mesh == "tp2":
        args += ["--mesh-mode", "tp", "--tp-size", str(n_dev)]
    elif mesh in ("pp2", "pp4"):
        args += ["--mesh-mode", "pp"]
    elif mesh == "pp2tp2":
        args += ["--mesh-mode", "pp+tp", "--pp-size", "2", "--tp-size", "2"]
    elif mesh == "sd":
        args += ["--mesh-mode", "pp"]
    tag = "v2" if v2 else "legacy"
    log = open(f"/tmp/gemma4_bench_{tag}_{port}.log", "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def boot_llamacpp(model_path: str, port: int, mesh: str, devices: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ROCM_LD
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["HIP_VISIBLE_DEVICES"] = devices
    n_dev = len([d for d in devices.split(",") if d.strip()])
    args = [
        LLAMA_BIN,
        "--model", model_path,
        "--port", str(port),
        "-ngl", "999",
        "--ctx-size", "4096",
        "--flash-attn", "on",
        "--threads", "8",
        "--no-mmap",
    ]
    if mesh == "tp2":
        # row-split = TP-style: shards each tensor across GPUs.
        tensor_split = ",".join(["1"] * n_dev)
        args += ["--split-mode", "row", "--tensor-split", tensor_split]
    elif mesh in ("pp2", "pp4"):
        # layer-split = PP-style.
        tensor_split = ",".join(["1"] * n_dev)
        args += ["--split-mode", "layer", "--tensor-split", tensor_split]
    # SD: default is single-device.
    log = open(f"/tmp/gemma4_bench_llamacpp_{port}.log", "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 600.0) -> bool:
    """Poll /v1/models; bail early if the server process exits."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def kill_proc(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        p.wait(timeout=10.0)
    except Exception:
        pass


def stream_one(port: int, model_id: str, prompt: str, max_tokens: int) -> tuple[Turn, str]:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.time()
    first_tok_t: float | None = None
    last_tok_t: float | None = None
    n_tokens = 0
    reply_chunks: list[str] = []
    completion_tokens = 0
    prompt_tokens = 0
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
                content   = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content or reasoning:
                    if first_tok_t is None:
                        first_tok_t = time.time()
                    last_tok_t = time.time()
                    n_tokens += 1
                    if content:
                        reply_chunks.append(content)
                    elif reasoning:
                        reply_chunks.append(reasoning)
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
    if first_tok_t is None or last_tok_t is None or n_tokens < 1:
        return Turn(), ""
    reply = "".join(reply_chunks)
    ttft_ms  = (first_tok_t - t0) * 1000.0
    decode_ms = (last_tok_t - first_tok_t) * 1000.0
    decode_ms_per_tok = decode_ms / max(n_tokens - 1, 1)
    wall_ms = (last_tok_t - t0) * 1000.0
    if completion_tokens == 0:
        completion_tokens = n_tokens
    return Turn(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        decode_ms_per_tok=decode_ms_per_tok,
        wall_ms=wall_ms,
    ), reply


def discover_model_id(port: int) -> str:
    r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
    return r.json()["data"][0]["id"]


def tokenize_count(port: int, text: str) -> int | None:
    """Best-effort prompt tokenization via the server's `/tokenize`
    endpoint. flambeau and llama.cpp both expose it but the response
    shape differs: flambeau returns `{"tokens": [...]}`, llama.cpp
    returns `{"tokens": [...]}` too on recent builds. Returns None on
    failure so callers can fall back to streaming `usage`.
    """
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/tokenize",
            json={"content": text, "add_special": False},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        toks = body.get("tokens")
        if isinstance(toks, list):
            return len(toks)
    except Exception:
        pass
    return None


def run_arm(label: str, boot_fn, max_tokens: int) -> ArmResult:
    port = free_port()
    print(f"\n=== {label} (port {port}) booting ...", flush=True)
    proc = boot_fn(port)
    if not wait_ready(port, proc, timeout_s=600.0):
        kill_proc(proc)
        err = "process exited" if proc.poll() is not None else "boot timeout"
        return ArmResult(label, err=err)
    try:
        model_id = discover_model_id(port)
        print(f"  ready as {model_id!r}", flush=True)
        try:
            stream_one(port, model_id, "Hello.", 8)  # warmup
        except Exception:
            pass
        turn, reply = stream_one(port, model_id, PROMPT_BASE, max_tokens)
        if not reply:
            return ArmResult(label, err="no tokens")
        # llama.cpp doesn't ship `usage` mid-stream — backfill via /tokenize
        # so prefill rates are comparable across stacks.
        if turn.prompt_tokens == 0:
            n_pt = tokenize_count(port, PROMPT_BASE)
            if n_pt:
                turn.prompt_tokens = n_pt
        prefill_tps = turn.prompt_tokens / (turn.ttft_ms / 1000.0) if turn.ttft_ms > 0 and turn.prompt_tokens > 0 else 0
        decode_tps = (turn.completion_tokens - 1) / (turn.decode_ms_per_tok * (turn.completion_tokens - 1) / 1000.0) if turn.completion_tokens > 1 and turn.decode_ms_per_tok > 0 else 0
        return ArmResult(
            label=label,
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            ttft_ms=turn.ttft_ms,
            decode_ms=turn.decode_ms_per_tok * max(turn.completion_tokens - 1, 1),
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            text_head=reply[:80],
        )
    finally:
        kill_proc(proc)
        time.sleep(5.0)


def fmt_row(arm: ArmResult) -> str:
    if arm.err:
        return f"| {arm.label:<24} | ERR: {arm.err}"
    return (
        f"| {arm.label:<24} | {arm.prompt_tokens:>5} "
        f"| {arm.prefill_tps:>8.1f} | {arm.completion_tokens:>4} "
        f"| {arm.decode_tps:>8.2f} | {arm.text_head!r:<60}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Comma list of case labels to run")
    ap.add_argument("--tg", type=int, default=128)
    args = ap.parse_args()

    if not FLAMBEAU_BIN.exists():
        print(f"missing {FLAMBEAU_BIN}", file=sys.stderr)
        return 2
    if not Path(LLAMA_BIN).exists():
        print(f"missing {LLAMA_BIN}", file=sys.stderr)
        return 2

    selected = CASES
    if args.only:
        keep = set(args.only.split(","))
        selected = [c for c in CASES if c.label in keep]
        if not selected:
            print(f"no cases matched --only={args.only!r}", file=sys.stderr)
            return 2

    out: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tg": args.tg,
        "results": [],
    }

    for case in selected:
        if not Path(case.model).exists():
            print(f"SKIP {case.label}: {case.model} not present", flush=True)
            continue
        print(
            f"\n################ {case.label}  "
            f"flambeau={case.flambeau_mesh} on {case.flambeau_devices}  "
            f"llamacpp={case.llamacpp_mesh} on {case.llamacpp_devices}"
        )
        case_results: list[ArmResult] = []
        for stack in ("flambeau-v2", "flambeau-legacy", "llama.cpp"):
            label = f"{stack}__{case.label}"
            if stack == "llama.cpp":
                boot = lambda port, c=case: boot_llamacpp(c.model, port, c.llamacpp_mesh, c.llamacpp_devices)
            else:
                v2 = (stack == "flambeau-v2")
                boot = lambda port, c=case, vv=v2: boot_flambeau(c.model, port, c.flambeau_mesh, c.flambeau_devices, vv)
            try:
                arm = run_arm(label, boot, args.tg)
            except Exception as exc:
                arm = ArmResult(label, err=f"exception: {exc}")
                print(f"  {label}: exception {exc}", flush=True)
            case_results.append(arm)
            out["results"].append({
                "case": case.label,
                "stack": stack,
                "label": arm.label,
                "err": arm.err,
                "prompt_tokens": arm.prompt_tokens,
                "completion_tokens": arm.completion_tokens,
                "ttft_ms": arm.ttft_ms,
                "decode_ms": arm.decode_ms,
                "decode_tps": arm.decode_tps,
                "prefill_tps": arm.prefill_tps,
                "text_head": arm.text_head,
            })
        print(f"\n--- {case.label} summary ---")
        print(f"| {'stack':<24} | ptok  | prefill  | ctok | decode   | text head")
        print(f"|{'-'*26}|------:|---------:|-----:|---------:|-----------")
        for arm in case_results:
            print(fmt_row(arm))

    cert_dir = ROOT / "certs" / "perf"
    cert_dir.mkdir(parents=True, exist_ok=True)
    out_path = cert_dir / "gemma4_v2_vs_legacy_vs_llamacpp_2026_05_19.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
