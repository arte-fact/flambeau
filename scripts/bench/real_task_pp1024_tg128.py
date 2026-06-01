#!/usr/bin/env python3
"""Real-task bench: pp≈1024, tg=128 on Qwen3.6-35B-A3B-Q4_0 / PP2.

Boots flambeau in v2 mode then legacy mode; sends ONE streaming chat
completion with a long technical prompt (~1024 tokens) asking for a
detailed 128-token response. Measures:

- prompt_tokens (server-reported)
- TTFT (time from request send → first content delta)
- decode tok/s (completion_tokens / (wall - ttft))
- total wall

Report as a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "flambeau"

# ~1024-token prompt: ~5000-char technical instruction tiled with
# concrete sub-questions so the model doesn't bail early.
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
    "vendors and architectures, and aim for a thorough multi-paragraph "
    "answer that a graduate student in computer architecture could learn "
    "from. Begin now."
)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_health(port: int, deadline_s: float = 120.0) -> None:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            ) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server :{port} never healthy")


def stream_one(port: int, prompt: str, max_tokens: int) -> dict:
    """Streaming chat completion; returns ttft_ms, decode_tps,
    prompt_tokens, completion_tokens, total_ms."""
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    last_t = None
    prompt_tokens = 0
    completion_tokens = 0
    full_text_chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage") or {}
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                completion_tokens = int(usage.get("completion_tokens", completion_tokens) or 0)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                last_t = now
                full_text_chunks.append(delta)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    ttft_ms = (ttft or 0) * 1000
    decode_ms = (t1 - (t0 + (ttft or 0))) * 1000 if ttft is not None else 0
    decode_tps = completion_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "text_head": "".join(full_text_chunks)[:80],
    }


def run_one(model: str, devices: str, mesh: str, v2: bool, warmup: int, max_tokens: int) -> dict:
    port = free_port()
    cmd = [
        str(BIN),
        "serve",
        "--model", model,
        "--devices", devices,
        "--mesh-mode", mesh,
        "--port", str(port),
        "--inflight-slots", "1",
        "--ctx-cap", "4096",
    ]
    if mesh == "tp":
        ndev = len([d for d in devices.split(",") if d.strip()])
        cmd += ["--tp-size", str(ndev)]
    elif mesh == "pp+tp":
        cmd += ["--pp-size", "2", "--tp-size", "2"]

    env = os.environ.copy()
    if v2:
        env["FLAMBEAU_V2"] = "1"
    else:
        env.pop("FLAMBEAU_V2", None)
    log = open(f"/tmp/realtask_{'v2' if v2 else 'legacy'}_{port}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    try:
        wait_health(port)
        for _ in range(warmup):
            stream_one(port, PROMPT_BASE, 8)
        result = stream_one(port, PROMPT_BASE, max_tokens)
        return result
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf")
    ap.add_argument("--devices", default="0,2")
    ap.add_argument("--mesh", default="pp")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="model")
    ap.add_argument("--v2-only", action="store_true")
    ap.add_argument("--legacy-only", action="store_true")
    args = ap.parse_args()

    print(f"model: {Path(args.model).name}")
    print(f"topo: {args.mesh} devices={args.devices}")
    print(f"tg target: {args.tg}")
    print()

    v2 = None
    legacy = None
    if not args.legacy_only:
        v2 = run_one(args.model, args.devices, args.mesh, v2=True, warmup=args.warmup, max_tokens=args.tg)
    if not args.v2_only:
        legacy = run_one(args.model, args.devices, args.mesh, v2=False, warmup=args.warmup, max_tokens=args.tg)
    if v2 is None or legacy is None:
        target = v2 or legacy
        label = "v2" if v2 else "legacy"
        if target:
            pp = target["prompt_tokens"] / (target["ttft_ms"] / 1000) if target["ttft_ms"] > 0 else 0
            print(f"{label} solo: pp={pp:.1f} t/s decode={target['decode_tps']:.2f} t/s total={target['total_ms']:.1f}ms")
        return 0

    def fmt(r: dict, label: str) -> str:
        pp_tps = r["prompt_tokens"] / (r["ttft_ms"] / 1000) if r["ttft_ms"] > 0 else 0
        return (
            f"| {label:<8} | {r['prompt_tokens']:>5} | {r['ttft_ms']:>8.1f} | "
            f"{pp_tps:>9.1f} | {r['completion_tokens']:>4} | {r['decode_ms']:>8.1f} | "
            f"{r['decode_tps']:>9.2f} | {r['total_ms']:>8.1f} |"
        )

    print("| stack    | pp_tok | ttft_ms  | prefill   | tg_tok | decode_ms | decode    | total_ms |")
    print("|----------|-------:|---------:|----------:|-------:|----------:|----------:|---------:|")
    print(fmt(v2, "v2"))
    print(fmt(legacy, "legacy"))
    print()
    # Ratios.
    v2_dec = v2["decode_tps"]
    leg_dec = legacy["decode_tps"]
    v2_pp = v2["prompt_tokens"] / (v2["ttft_ms"] / 1000) if v2["ttft_ms"] > 0 else 0
    leg_pp = legacy["prompt_tokens"] / (legacy["ttft_ms"] / 1000) if legacy["ttft_ms"] > 0 else 0
    print(f"decode v2/legacy:  {v2_dec/leg_dec:.3f}×" if leg_dec else "")
    print(f"prefill v2/legacy: {v2_pp/leg_pp:.3f}×" if leg_pp else "")
    print(f"total v2/legacy:   {legacy['total_ms']/v2['total_ms']:.3f}× (closer to 1 = faster)")
    print()
    print(f"v2 head:     {v2['text_head']!r}")
    print(f"legacy head: {legacy['text_head']!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
