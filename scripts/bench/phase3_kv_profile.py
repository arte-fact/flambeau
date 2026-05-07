#!/usr/bin/env python3
"""
Phase 3 follow-up — kernel-level profile of decode under F16 vs Q8 KV.

Method: boot once per arm with `FLAMBEAU_PROFILE_DECODE=64`, drive a
short prompt (~32 tokens) so prefill cost doesn't dwarf decode, then
let the server emit per-section ms-per-token to stderr. Captures the
profile log into the cert; diff F16 vs Q8 lets us name the per-kernel
cost of Q8 KV.

Output: certs/env_impact/PHASE3_KV_PROFILE.md
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore
import run_env_impact as rei  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "PHASE3_KV_PROFILE.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 80         # 8 warmup-skip + 64 measured + 8 cushion
PROFILE_N = 64
SEED = 0


@dataclasses.dataclass
class ArmResult:
    label: str
    prefill_ms: float
    decode_ms: float
    profile_log: str
    err: str | None = None


def run_arm(label: str, env_extras: dict[str, str],
            profile_env: dict[str, str]) -> ArmResult:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"phase3_kv_profile__{label}.log"
    print(f"\n=== {label} env_extras={env_extras} ===", flush=True)
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=8,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID):
        rm.kill_server(proc)
        return ArmResult(label, 0, 0, "", "boot timeout")
    try:
        # short ~32-token prompt fragment so prefill stays brief
        # (Q8 prefill is per-token; long prompts dominate wall).
        # _run_concurrent_batch uses the harness's standard chunked prompt
        # which is ~3.3k tokens — too long for Q8. Use a tiny payload.
        import requests
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = {
            "model": MODEL_ID,
            "messages": [{"role": "user",
                           "content": "Write a long, detailed essay on the "
                                       "history of operating-system schedulers "
                                       "from the 1960s through today. Cover at "
                                       "least: round-robin, multi-level feedback "
                                       "queues, priority inheritance, the BSD "
                                       "scheduler, Linux O(1) and CFS, and modern "
                                       "EEVDF. Include code-level details where "
                                       "relevant. Keep going until you've "
                                       "exhausted the topic — at least several "
                                       "thousand words."}],
            "stream": True,
            "temperature": 0.0,
            "seed": SEED,
            "max_tokens": TG_LEN,
        }
        prefill_ms = 0.0
        decode_ms = 0.0
        first_token_t = None
        last_token_t = None
        n_tokens = 0
        t0 = time.time()
        with requests.post(url, json=body, stream=True, timeout=600.0) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                if first_token_t is None:
                    first_token_t = time.time()
                last_token_t = time.time()
                n_tokens += 1
        if first_token_t and last_token_t and n_tokens > 1:
            prefill_ms = (first_token_t - t0) * 1000.0
            decode_ms = (last_token_t - first_token_t) * 1000.0 / max(n_tokens - 1, 1)
        time.sleep(0.5)  # let stderr flush
        rm.kill_server(proc)
        time.sleep(2.0)
    except Exception as e:
        rm.kill_server(proc)
        return ArmResult(label, 0, 0, "", str(e))
    finally:
        rm.cooldown_until_safe(70.0, 240.0)

    log_text = log_path.read_text(errors="replace")
    profile_log = ""
    if "=== TP decode profile ===" in log_text:
        idx = log_text.find("=== TP decode profile ===")
        seg = log_text[idx:]
        end = seg.find("TOTAL_RECORDED")
        if end != -1:
            line_end = seg.find("\n", end)
            profile_log = seg[: line_end + 1 if line_end != -1 else len(seg)]
    if "=== HOST decode profile" in log_text:
        idx = log_text.find("=== HOST decode profile")
        seg = log_text[idx:]
        end = seg.find("TOTAL_per_step")
        if end != -1:
            line_end = seg.find("\n", end)
            profile_log += "\n" + seg[: line_end + 1 if line_end != -1 else len(seg)]
    return ArmResult(label, prefill_ms, decode_ms, profile_log)


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    common_env = {
        "FLAMBEAU_INFLIGHT_SLOTS": "8",
        "FLAMBEAU_PROFILE_DECODE": str(PROFILE_N),
        "FLAMBEAU_HOST_PROFILE": "1",
    }

    f16 = run_arm(
        "f16",
        env_extras={**common_env, "FLAMBEAU_KV": "f16"},
        profile_env=profile_env,
    )
    q8 = run_arm(
        "q8",
        env_extras={**common_env, "FLAMBEAU_KV": "q8"},
        profile_env=profile_env,
    )

    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True
                         ).stdout.strip() or "unknown"
    lines: list[str] = []
    lines.append("# Phase 3 follow-up — F16 vs Q8 KV decode profile")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Commit:** {rev}")
    lines.append(f"- **Model:** {MODEL_ID} on pp2tp2 / slots=8")
    lines.append(f"- **Workload:** short prompt ('Count from one to ten.'), "
                 f"max_tokens={TG_LEN}, decode-profile window=64 (skip 8 warmup)")
    lines.append("")
    lines.append("## Wall-clock summary")
    lines.append("")
    lines.append("| arm | prefill ms | decode ms/tok | err |")
    lines.append("|---|---:|---:|---|")
    for r in (f16, q8):
        if r.err:
            lines.append(f"| `{r.label}` | — | — | `{r.err}` |")
        else:
            lines.append(f"| `{r.label}` | {r.prefill_ms:.0f} | "
                         f"{r.decode_ms:.2f} | |")

    lines.append("")
    lines.append("## Per-kernel decode profile")
    for r in (f16, q8):
        lines.append("")
        lines.append(f"### `{r.label}`")
        lines.append("")
        if r.err or not r.profile_log:
            lines.append(f"_no profile captured_ {('('+r.err+')') if r.err else ''}")
        else:
            lines.append("```")
            lines.append(r.profile_log.rstrip())
            lines.append("```")

    CERT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CERT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
