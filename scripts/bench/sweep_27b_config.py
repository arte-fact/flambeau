#!/usr/bin/env python3
"""
Config sweep on Qwen3.6-27B-Q4_0 / pp2tp2 — find the env config that
maximises prefill + decode at single (N=1) and concurrent (N=4) loads.

Runs at the current commit (expected: 27cdabc, pre-collapse, fast-path
active) on top of `bench/profiles/optimized.toml`.

Each candidate is a named env-overlay dict. We measure prefill_ms and
aggregate_tps median over RUNS_PER_CELL post-warmup batches, at N=1
and N=4. Output: certs/env_impact/SWEEP_27B_CONFIG.md with one row
per (config, N) cell + the winners per metric.
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
CERT = ROOT / "certs" / "env_impact" / "SWEEP_27B_CONFIG.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64
SEED = 0
WARMUP_RUNS = 1
RUNS_PER_CELL = 2
COOLDOWN_C = 70.0
COOLDOWN_MAX_S = 240.0
CONCURRENCIES = [1, 4]

# Each config is profile (always-on optimized.toml) + the listed overlay.
# Keys map to env vars set on the server child process.
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


@dataclasses.dataclass
class Cell:
    config: str
    n: int
    n_runs: int
    prefill_ms: float
    decode_ms: float
    agg_tps: float
    text_sha256: str
    err: str | None = None


def measure(label: str, n: int, env_extras: dict[str, str],
            profile_env: dict[str, str]) -> Cell:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sweep27b__{label}__n{n}.log"

    # INFLIGHT_SLOTS=8 is itself a perf lever (BASELINE_27B_PP2TP2 cert
    # measured 2.58× delta vs slots=1 at N=1 even though streaming
    # bypasses the scheduler). Fix slots=8 across all cells so the
    # sweep isolates the env-overlay effect from slot-count.
    slots = max(n, 8)
    print(f"\n=== {label} N={n} (slots={slots}) ===", flush=True)
    boot_t0 = time.time()
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras,
        slots=slots,
        log_path=log_path,
        ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    ready = rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID)
    boot_secs = time.time() - boot_t0
    if not ready:
        rm.kill_server(proc)
        return Cell(label, n, 0, 0, 0, 0, "",
                    f"boot timeout after {boot_secs:.0f}s")
    print(f"   boot {boot_secs:.0f}s — measuring...", flush=True)
    try:
        for _ in range(WARMUP_RUNS):
            _ = rei._run_concurrent_batch(port, n, TG_LEN,
                                          base_seed=SEED,
                                          sample_vram=False)
        prefills, decodes, aggs = [], [], []
        text_hash = ""
        last_err = None
        for run_i in range(RUNS_PER_CELL):
            r = rei._run_concurrent_batch(port, n, TG_LEN,
                                          base_seed=SEED,
                                          sample_vram=False)
            if r.get("err"):
                last_err = r["err"]
                continue
            prefills.append(r["prefill_ms_median"])
            decodes.append(r["decode_ms_median"])
            aggs.append(r["aggregate_tps"])
            if run_i == 0:
                text_hash = r.get("text_sha256", "")[:16]
        if not aggs:
            return Cell(label, n, 0, 0, 0, 0, "",
                        last_err or "no successful batches")
        return Cell(label, n, len(aggs),
                    statistics.median(prefills),
                    statistics.median(decodes),
                    statistics.median(aggs),
                    text_hash)
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(COOLDOWN_C, COOLDOWN_MAX_S)


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN} — build first", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    print(f"profile baseline ({len(profile_env)} keys): "
          f"{list(profile_env.keys())}", flush=True)

    cells: list[Cell] = []
    for cfg_name, overlay in CONFIGS.items():
        for n in CONCURRENCIES:
            cells.append(measure(cfg_name, n,
                                 env_extras=overlay,
                                 profile_env=profile_env))

    # Render cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# 27B/pp2tp2 config sweep — pre-collapse fast-path active")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    import subprocess
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True
                         ).stdout.strip() or "unknown"
    lines.append(f"- **Commit:** {rev}")
    lines.append(f"- **Profile baseline:** `bench/profiles/optimized.toml`")
    lines.append(f"- **Configs:** {', '.join(CONFIGS.keys())}")
    lines.append(f"- **Concurrencies:** {CONCURRENCIES}")
    lines.append(f"- **Runs/cell:** {RUNS_PER_CELL} measured + "
                 f"{WARMUP_RUNS} warmup")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| config | N | runs | prefill ms | decode ms | agg tg t/s | "
                 "Δ prefill | Δ tg | err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    by_n_cfg: dict[tuple[int, str], Cell] = {(c.n, c.config): c for c in cells}
    base_by_n = {n: by_n_cfg.get((n, "baseline")) for n in CONCURRENCIES}
    for c in cells:
        b = base_by_n.get(c.n)
        if c.err:
            lines.append(f"| `{c.config}` | {c.n} | 0 | — | — | — | — | — | "
                         f"`{c.err}` |")
            continue
        if b and b.prefill_ms > 0 and c.config != "baseline":
            d_pref = (b.prefill_ms - c.prefill_ms) / b.prefill_ms * 100.0
            d_tg = (c.agg_tps - b.agg_tps) / b.agg_tps * 100.0
            d_pref_s = f"{d_pref:+.1f}%"
            d_tg_s = f"{d_tg:+.1f}%"
        else:
            d_pref_s = "—"
            d_tg_s = "—"
        lines.append(f"| `{c.config}` | {c.n} | {c.n_runs} | "
                     f"{c.prefill_ms:.0f} | {c.decode_ms:.0f} | "
                     f"{c.agg_tps:.1f} | {d_pref_s} | {d_tg_s} | |")

    # Winners per metric per N.
    lines.append("")
    lines.append("## Winners per metric per N")
    lines.append("")
    lines.append("| N | best prefill (config, ms) | best decode-tg (config, t/s) |")
    lines.append("|---:|---|---|")
    for n in CONCURRENCIES:
        cands = [c for c in cells if c.n == n and c.n_runs > 0]
        if not cands:
            lines.append(f"| {n} | — | — |")
            continue
        bp = min(cands, key=lambda c: c.prefill_ms)
        bt = max(cands, key=lambda c: c.agg_tps)
        lines.append(f"| {n} | `{bp.config}` ({bp.prefill_ms:.0f} ms) | "
                     f"`{bt.config}` ({bt.agg_tps:.1f} t/s) |")

    CERT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CERT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
