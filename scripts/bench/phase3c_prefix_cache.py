#!/usr/bin/env python3
"""
Phase 3c — re-cert that `--prefix-cache` actually shortens TTFT on a
multi-turn workload.

Method:
  - Boot Qwen3.6-27B-Q4_0 / pp2tp2 with `--prefix-cache=true`.
  - Send the same long prompt 3× sequentially. First call primes the
    cache; subsequent calls should hit the entire prefix.
  - Repeat with `--prefix-cache=false` for a baseline.
  - Cert delta-prefill_ms (cold vs warm), assert ≥ 5× saving warm.

Output: certs/env_impact/PHASE3C_PREFIX_CACHE.md
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore
import run_env_impact as rei  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "PHASE3C_PREFIX_CACHE.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 32  # short decode — we care about TTFT/prefill, not throughput
SEED = 0


@dataclasses.dataclass
class Turn:
    label: str
    prefill_ms: float
    decode_ms: float
    err: str | None = None


def measure_turns(label: str, env_extras: dict[str, str],
                  profile_env: dict[str, str], n_turns: int = 3) -> list[Turn]:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"phase3c__{label}.log"
    print(f"\n=== {label} env_extras={env_extras} ===", flush=True)
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=8,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID):
        rm.kill_server(proc)
        return [Turn(f"{label}#{i}", 0, 0, "boot timeout") for i in range(n_turns)]
    out: list[Turn] = []
    try:
        for i in range(n_turns):
            r = rei._run_concurrent_batch(port, 1, TG_LEN,
                                          base_seed=SEED,
                                          sample_vram=False)
            if r.get("err"):
                out.append(Turn(f"{label}#{i}", 0, 0, r["err"]))
            else:
                out.append(Turn(
                    label=f"{label}#{i}",
                    prefill_ms=r["prefill_ms_median"],
                    decode_ms=r["decode_ms_median"],
                ))
            print(f"  turn {i}: prefill={r.get('prefill_ms_median', 0):.0f}ms "
                  f"decode={r.get('decode_ms_median', 0):.0f}ms",
                  flush=True)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(70.0, 240.0)
    return out


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)

    cold = measure_turns(
        "cold (prefix-cache off)",
        env_extras={"FLAMBEAU_INFLIGHT_SLOTS": "8"},
        profile_env=profile_env,
        n_turns=3,
    )
    warm = measure_turns(
        "warm (prefix-cache on)",
        env_extras={
            "FLAMBEAU_INFLIGHT_SLOTS": "8",
            "FLAMBEAU_PREFIX_CACHE": "1",
            "FLAMBEAU_PREFIX_CACHE_MAX_GB": "2",
        },
        profile_env=profile_env,
        n_turns=3,
    )

    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    import subprocess
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True
                         ).stdout.strip() or "unknown"

    lines: list[str] = []
    lines.append("# Phase 3c — `--prefix-cache` cold/warm cert")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Commit:** {rev}")
    lines.append(f"- **Model:** {MODEL_ID} on pp2tp2")
    lines.append(f"- **Workload:** same prompt 3× sequentially, max_tokens={TG_LEN}")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| arm | turn | prefill ms | decode ms | err |")
    lines.append("|---|---:|---:|---:|---|")
    for t in cold + warm:
        if t.err:
            lines.append(f"| `{t.label}` | — | — | — | `{t.err}` |")
        else:
            lines.append(f"| `{t.label}` | {t.label.split('#')[1]} | "
                         f"{t.prefill_ms:.0f} | {t.decode_ms:.0f} | |")

    cold_first = cold[0].prefill_ms if cold and not cold[0].err else 0
    warm_first = warm[0].prefill_ms if warm and not warm[0].err else 0
    warm_second = warm[1].prefill_ms if len(warm) > 1 and not warm[1].err else 0
    warm_third = warm[2].prefill_ms if len(warm) > 2 and not warm[2].err else 0

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if warm_second > 0 and warm_first > 0:
        speedup = warm_first / warm_second if warm_second > 0 else 0
        lines.append(f"- **Warm-turn speedup:** {speedup:.1f}× "
                     f"(turn 1 = {warm_first:.0f} ms → turn 2 = {warm_second:.0f} ms)")
        lines.append(f"- **Turn 3:** {warm_third:.0f} ms")
        if speedup >= 5.0:
            lines.append("- **PASS** — prefix cache is effective.")
        elif speedup >= 1.5:
            lines.append("- **PARTIAL** — some saving but below 5× target.")
        else:
            lines.append("- **FAIL** — no measurable warm-turn benefit.")
    else:
        lines.append("- **INSUFFICIENT-DATA** (one or more arms errored)")

    CERT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CERT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
