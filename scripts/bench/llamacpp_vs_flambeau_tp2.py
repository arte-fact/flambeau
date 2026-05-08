#!/usr/bin/env python3
"""
Comparative bench: flambeau vs llama.cpp at TP2 on Qwen3.6-27B and
Qwen3.6-35B-A3B. Three-turn conversation with long technical prompts;
average prefill rate (tok/s) and decode rate (tok/s) across turns.

Topology:
  flambeau:  --mesh-mode tp --tp-size 2   (devices 0,1)
  llama.cpp: --split-mode row --tensor-split 1,1   (devices 0,1)

Per turn: streaming chat completion. TTFT measured from POST to first
SSE byte; decode rate from first to last SSE byte. The conversation
accumulates so each turn has progressively longer prompt context.
"""
from __future__ import annotations

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
ROCM_LD = "/opt/rocm-host/lib:/artefact/llama.cpp/build-mi50/bin"
ROCBLAS_TENSILE = "/opt/rocm-host/lib/rocblas/library"

LLAMA_BIN = "/artefact/llama.cpp/build-mi50/bin/llama-server"
FLAMBEAU_BIN = ROOT / "target/release/flambeau"

# Use devices 0 and 1 for both engines (TP2). Flambeau supports
# arbitrary device indices; llama.cpp respects HIP_VISIBLE_DEVICES.
DEVICES = "0,1"

MODELS = {
    "qwen36-27b-q4_0":     "/artefact/models/Qwen3.6-27B-Q4_0.gguf",
    "qwen36-35b-a3b-q4_0": "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
}

# Three-turn conversation. Each turn nudges for a LONG technical
# response so decode is well-amortised. Conversation accumulates.
# `/no_think` is the Qwen3 convention to suppress the <think>...</think>
# block — keeps the response 100% in `delta.content` for both engines.
TURNS = [
    "/no_think Explain the architecture of modern Linux kernel CPU schedulers, "
    "covering CFS, EEVDF, and the trade-offs between throughput and "
    "interactivity. Include concrete numerical examples of vruntime "
    "calculations and the red-black tree invariants. Be thorough — "
    "aim for several hundred words.",
    "/no_think Now write a complete Rust implementation sketch of an EEVDF-style "
    "scheduler for a userspace task pool. Include the priority queue, "
    "deadline tracking, and the rebalancing logic. Add inline comments "
    "explaining the invariants. Be exhaustive.",
    "/no_think Compare your implementation to the kernel one: what corners did "
    "you cut, where is your version structurally different, and what "
    "would change if we needed hard real-time guarantees? Walk through "
    "the trade-offs in detail.",
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
    turns: list[Turn] = field(default_factory=list)


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def boot_flambeau(model_path: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    env["LD_LIBRARY_PATH"] = (
        env.get("LD_LIBRARY_PATH", "") + ":" + ROCM_LD
    )
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["FLAMBEAU_INFLIGHT_SLOTS"] = "8"
    env["FLAMBEAU_KV"] = "f16"
    args = [
        str(FLAMBEAU_BIN), "serve",
        "--model", model_path,
        "--devices", DEVICES,
        "--mesh-mode", "tp",
        "--tp-size", "2",
        "--port", str(port),
        "--ctx-cap", "8192",
    ]
    log = open(f"/tmp/llamacmp_bench_flambeau_{port}.log", "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def boot_llamacpp(model_path: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ROCM_LD
    env["ROCBLAS_TENSILE_LIBPATH"] = ROCBLAS_TENSILE
    env["HIP_VISIBLE_DEVICES"] = DEVICES
    # Use 8k ctx — 3-turn conversation never exceeds ~1700 tokens; smaller
    # ctx leaves headroom for split-row's larger memory layout. Flash-attn
    # on for KV-cache memory + decode speed.
    args = [
        LLAMA_BIN,
        "--model", model_path,
        "--port", str(port),
        "-ngl", "999",
        "--split-mode", "row",
        "--tensor-split", "1,1",
        "--ctx-size", "8192",
        "--flash-attn", "on",
        "--threads", "8",
        "--no-mmap",
    ]
    log = open(f"/tmp/llamacmp_bench_llamacpp_{port}.log", "w")
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


def chat_one_turn(port: int, model_id: str,
                   messages: list[dict]) -> tuple[Turn, str]:
    """Stream one chat completion; return Turn metrics + reply text."""
    body = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 512,
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
            # OpenAI-style chat-completion-chunk:
            #   choices[0].delta.content        — visible text
            #   choices[0].delta.reasoning_content — Qwen3 think-block (counted
            #                                          as decode work even if
            #                                          we strip it from reply)
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
            # Some servers attach usage at end-of-stream
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


def run_arm(label: str, boot_fn, model_path: str) -> ArmResult:
    port = free_port()
    print(f"\n=== {label} (port {port}) booting ...", flush=True)
    proc = boot_fn(model_path, port)
    if not wait_ready(port, timeout_s=600.0):
        kill_proc(proc)
        return ArmResult(label, err="boot timeout")
    try:
        model_id = discover_model_id(port)
        print(f"  ready as {model_id!r}", flush=True)

        # Warmup chat (16 tok), no measurement.
        warmup_msgs = [{"role": "user", "content": "Hello."}]
        try:
            requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": model_id, "messages": warmup_msgs,
                      "stream": False, "max_tokens": 16, "temperature": 0},
                timeout=120.0,
            )
        except Exception as e:
            print(f"  warmup failed: {e}", flush=True)

        result = ArmResult(label)
        msgs: list[dict] = []
        for i, user_text in enumerate(TURNS):
            msgs.append({"role": "user", "content": user_text})
            print(f"  turn {i+1}/{len(TURNS)} ...", flush=True)
            try:
                t, reply = chat_one_turn(port, model_id, msgs)
            except Exception as exc:
                print(f"    turn {i+1} exception: {exc}", flush=True)
                result.err = f"turn {i+1} exception: {exc}"
                break
            if not reply:
                print(f"    turn {i+1} empty reply (skipping turn)", flush=True)
                # don't add an empty assistant — but record the timing if any
                if t.completion_tokens or t.ttft_ms:
                    result.turns.append(t)
                # skip remaining turns since context can't accumulate
                break
            result.turns.append(t)
            msgs.append({"role": "assistant", "content": reply})
            prefill_rate = (t.prompt_tokens / (t.ttft_ms / 1000.0)
                            if t.ttft_ms > 0 else 0.0)
            decode_rate  = (1000.0 / t.decode_ms_per_tok
                            if t.decode_ms_per_tok > 0 else 0.0)
            print(f"    ptok={t.prompt_tokens:5d}  ctok={t.completion_tokens:4d}  "
                  f"ttft={t.ttft_ms:7.0f} ms  prefill={prefill_rate:7.1f} tok/s  "
                  f"decode={decode_rate:6.2f} tok/s",
                  flush=True)
        return result
    finally:
        kill_proc(proc)
        time.sleep(5.0)


def summarise(arm: ArmResult) -> dict:
    if arm.err or not arm.turns:
        return {"label": arm.label, "err": arm.err or "no turns"}
    total_p_tok = sum(t.prompt_tokens for t in arm.turns)
    total_p_sec = sum(t.ttft_ms / 1000.0 for t in arm.turns)
    total_c_tok = sum(t.completion_tokens for t in arm.turns)
    total_c_sec = sum(t.decode_ms_per_tok * (t.completion_tokens or 1)
                      / 1000.0 for t in arm.turns)
    return {
        "label": arm.label,
        "n_turns": len(arm.turns),
        "avg_prefill_rate_tok_per_s": (total_p_tok / total_p_sec) if total_p_sec > 0 else 0.0,
        "avg_decode_rate_tok_per_s":  (total_c_tok / total_c_sec) if total_c_sec > 0 else 0.0,
        "total_ptok": total_p_tok,
        "total_ctok": total_c_tok,
        "per_turn": [
            {
                "ptok": t.prompt_tokens,
                "ctok": t.completion_tokens,
                "ttft_ms": round(t.ttft_ms, 1),
                "decode_ms_per_tok": round(t.decode_ms_per_tok, 3),
                "wall_ms": round(t.wall_ms, 1),
            } for t in arm.turns
        ],
    }


def main() -> None:
    if not FLAMBEAU_BIN.exists():
        print(f"missing {FLAMBEAU_BIN}", file=sys.stderr)
        sys.exit(2)
    if not Path(LLAMA_BIN).exists():
        print(f"missing {LLAMA_BIN}", file=sys.stderr)
        sys.exit(2)

    out: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "topology": "TP2",
                  "devices": DEVICES,
                  "results": []}

    for model_id, model_path in MODELS.items():
        for engine, boot in [("flambeau", boot_flambeau),
                              ("llama.cpp", boot_llamacpp)]:
            label = f"{engine}__{model_id}"
            try:
                arm = run_arm(label, boot, model_path)
            except Exception as exc:
                arm = ArmResult(label, err=f"run_arm exception: {exc}")
                print(f"  {label}: exception {exc}", flush=True)
            out["results"].append(summarise(arm))

    print("\n========== summary ==========")
    print(json.dumps(out, indent=2))
    cert_dir = ROOT / "certs" / "perf"
    cert_dir.mkdir(parents=True, exist_ok=True)
    out_path = cert_dir / "tp2_3turn_flambeau_vs_llamacpp_2026_05_08.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
