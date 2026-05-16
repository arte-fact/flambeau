#!/usr/bin/env python3
"""
Correctness pass over the 4 sweep_27b_config.py configs.

For each config (baseline, prefill_ubatch_1024, ar_fuse_q8_1, all_optins),
run a deterministic chat at greedy/seed=0 with N=1 and N=4, and compare
the sha256 of generated text. The expected result is bit-identity for
configs that should be perf-only (no kernel correctness change), and
controlled drift for configs that swap kernel paths (e.g. all_optins
includes FLAMBEAU_VARIANT=fused which engages a different SSM path).

Output: certs/env_impact/CORRECTNESS_27B_CONFIGS.md
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore
import run_env_impact as rei  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "CORRECTNESS_27B_CONFIGS.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64
SEED = 0
COOLDOWN_C = 70.0
COOLDOWN_MAX_S = 240.0

CONFIGS: dict[str, dict[str, str]] = {
    "baseline":              {},
    "prefill_ubatch_1024":   {"FLAMBEAU_PREFILL_UBATCH": "1024"},
    "ar_fuse_q8_1":          {"FLAMBEAU_AR_FUSE_Q8_1": "on"},
    "all_optins":            {
        "FLAMBEAU_AR_FUSE_Q8_1":         "on",
        "FLAMBEAU_Q4_0_GU_T128":         "on",
        "FLAMBEAU_Q4_0_GU_WARPCOOP":     "on",
        "FLAMBEAU_BATCHED_MMVQ":         "1",
        "FLAMBEAU_SSM_OUT_F16_DST":      "on",
        "FLAMBEAU_VARIANT":              "fused",
    },
}
NS = [1, 4]


@dataclasses.dataclass
class Cell:
    config: str
    n: int
    sha: str
    text_head: str
    err: str | None = None


def measure(config: str, n: int, env_extras: dict[str, str],
            profile_env: dict[str, str]) -> Cell:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"correctness27b__{config}__n{n}.log"
    slots = max(n, 8)
    print(f"\n=== {config} N={n} (slots={slots}) ===", flush=True)
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=slots,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    if not rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID):
        rm.kill_server(proc)
        return Cell(config, n, "", "", "boot timeout")
    try:
        r = rei._run_concurrent_batch(port, n, TG_LEN,
                                      base_seed=SEED,
                                      sample_vram=False)
        if r.get("err"):
            return Cell(config, n, "", "", r["err"])
        sha = r.get("text_sha256", "")[:16]
        return Cell(config, n, sha, "")
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(COOLDOWN_C, COOLDOWN_MAX_S)


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN} — build first", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)

    cells: list[Cell] = []
    for cfg_name, overlay in CONFIGS.items():
        for n in NS:
            cells.append(measure(cfg_name, n,
                                 env_extras=overlay,
                                 profile_env=profile_env))

    # Render cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    import subprocess
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True
                         ).stdout.strip() or "unknown"

    by_n_cfg: dict[tuple[int, str], Cell] = {(c.n, c.config): c for c in cells}

    lines: list[str] = []
    lines.append("# Correctness pass — 27B/pp2tp2 sweep configs")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Commit:** {rev}")
    lines.append(f"- **Model:** {MODEL_ID} on pp2tp2 (slots=8)")
    lines.append(f"- **Sampling:** greedy / seed={SEED} / max_tokens={TG_LEN}")
    lines.append("")
    lines.append("## Per-cell sha256[:16]")
    lines.append("")
    lines.append("| config | N | sha256[:16] | err |")
    lines.append("|---|---:|---|---|")
    for c in cells:
        if c.err:
            lines.append(f"| `{c.config}` | {c.n} | — | `{c.err}` |")
        else:
            lines.append(f"| `{c.config}` | {c.n} | `{c.sha}` | |")

    # Verdict — compare each config's sha vs baseline at the same N.
    lines.append("")
    lines.append("## Verdict per config")
    lines.append("")
    lines.append("| config | N=1 vs baseline | N=4 vs baseline |")
    lines.append("|---|---|---|")
    for cfg_name in CONFIGS:
        if cfg_name == "baseline":
            continue
        verdicts = []
        for n in NS:
            b = by_n_cfg.get((n, "baseline"))
            c = by_n_cfg.get((n, cfg_name))
            if not b or not c or not b.sha or not c.sha:
                verdicts.append("—")
            elif b.sha == c.sha:
                verdicts.append("**identical**")
            else:
                verdicts.append("DIFFERENT")
        lines.append(f"| `{cfg_name}` | {verdicts[0]} | {verdicts[1]} |")

    CERT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CERT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
