#!/usr/bin/env python3
"""
Cumulative opt-in test — does setting all the small-positive-lean
opt-in gates simultaneously add up to a measurable improvement over
baseline, even though each gate individually measured null?

Hypothesis: 5 gates × +1-2% = net 5-10% if they're additive.
Falsification: if cumulative is also null (≤ noise floor 2.0%),
the deletion plan stands — these gates have no compound effect.

Test matrix:
  - Model:    Qwen3.6-27B-Q4_0
  - Topology: pp2tp2
  - N:        [1, 2, 4, 8]
  - Cells:    {baseline, cumulative} × N = 8 cells

Cumulative env (set on top of bench/profiles/optimized.toml):
  - FLAMBEAU_AR_FUSE_Q8_1 = on        # TP boundary AllReduce fusion
  - FLAMBEAU_Q4_0_GU_T128 = on        # Q4_0 gate-up T=128 tile
  - FLAMBEAU_Q4_0_GU_WARPCOOP = on    # Q4_0 gate-up warpcoop
  - FLAMBEAU_BATCHED_MMVQ = 1         # batched MMVQ kernel
  - FLAMBEAU_SSM_OUT_F16_DST = on     # GDN SSM-out F16 fusion
    + FLAMBEAU_VARIANT = fused        # required by SSM_OUT_F16_DST
  - FLAMBEAU_Q8_0_MMVQ_T128_VDR2 = on # no-op on Q4_0 (set anyway)
  - FLAMBEAU_Q8_0_GU_T128_VDR2 = on   # no-op on Q4_0 (set anyway)

Output: certs/env_impact/CUMULATIVE_OPTINS_27B_PP2TP2_N1,2,4,8.md
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore[import-not-found]
import run_env_impact as rei  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "CUMULATIVE_OPTINS_27B_PP2TP2_N1,2,4,8.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-27b-q4_0"
MODEL_PATH = "/artefact/models/Qwen3.6-27B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64
SEED = 0
WARMUP_RUNS = 2
RUNS_PER_CELL = 3
NOISE_FLOOR_PCT = 2.0
COOLDOWN_C = 70.0
COOLDOWN_MAX_S = 240.0

CONCURRENCIES = [1, 2, 4, 8]

CUMULATIVE_ENV: dict[str, str] = {
    "FLAMBEAU_AR_FUSE_Q8_1":         "on",
    "FLAMBEAU_Q4_0_GU_T128":         "on",
    "FLAMBEAU_Q4_0_GU_WARPCOOP":     "on",
    "FLAMBEAU_BATCHED_MMVQ":         "1",
    "FLAMBEAU_SSM_OUT_F16_DST":      "on",
    "FLAMBEAU_VARIANT":              "fused",   # precondition for SSM_OUT_F16_DST
    "FLAMBEAU_Q8_0_MMVQ_T128_VDR2":  "on",      # no-op on Q4_0 (no harm)
    "FLAMBEAU_Q8_0_GU_T128_VDR2":    "on",      # no-op on Q4_0 (no harm)
}


@dataclasses.dataclass
class CellRow:
    label: str          # "baseline" or "cumulative"
    n: int
    n_runs: int
    prefill_ms: float
    decode_ms: float
    agg_tps: float
    vram_peak_gb: float
    text_sha256: str
    err: str | None = None


def measure(label: str, n: int, env_extras: dict[str, str],
            profile_env: dict[str, str]) -> CellRow:
    """Boot server, warm, measure 3 batches at N concurrent. Returns
    median prefill/decode/agg-tps + peak VRAM + correctness hash."""
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cumulative_optin__{label}__n{n}.log"

    # slots = max(N, 1) so the inflight pool can hold all concurrent calls.
    slots = max(n, 1)

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
        return CellRow(label=label, n=n, n_runs=0,
                       prefill_ms=0, decode_ms=0, agg_tps=0,
                       vram_peak_gb=0, text_sha256="",
                       err=f"boot timeout after {boot_secs:.0f}s")
    print(f"   boot {boot_secs:.0f}s — measuring...", flush=True)

    try:
        # warmup
        for _ in range(WARMUP_RUNS):
            _ = rei._run_concurrent_batch(port, n, TG_LEN,
                                          base_seed=SEED,
                                          sample_vram=False)

        # measured
        prefills, decodes, aggs, vrams = [], [], [], []
        text_hash = ""
        last_err = None
        for run_i in range(RUNS_PER_CELL):
            r = rei._run_concurrent_batch(port, n, TG_LEN,
                                          base_seed=SEED,
                                          sample_vram=True)
            if r.get("err"):
                last_err = r["err"]
                continue
            prefills.append(r["prefill_ms_median"])
            decodes.append(r["decode_ms_median"])
            aggs.append(r["aggregate_tps"])
            if r.get("vram_peak_gb", 0) > 0:
                vrams.append(r["vram_peak_gb"])
            if run_i == 0:
                text_hash = r.get("text_sha256", "")[:16]

        if not aggs:
            return CellRow(label=label, n=n, n_runs=0,
                           prefill_ms=0, decode_ms=0, agg_tps=0,
                           vram_peak_gb=max(vrams) if vrams else 0,
                           text_sha256="",
                           err=last_err or "no successful batches")

        return CellRow(
            label=label, n=n, n_runs=len(aggs),
            prefill_ms=statistics.median(prefills),
            decode_ms=statistics.median(decodes),
            agg_tps=statistics.median(aggs),
            vram_peak_gb=max(vrams) if vrams else 0,
            text_sha256=text_hash,
        )
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(COOLDOWN_C, COOLDOWN_MAX_S)


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN} — build with: cargo build --release "
              f"--features hip_serve -p flambeau-cli", file=sys.stderr)
        sys.exit(2)

    profile_env = rei.load_profile_env(PROFILE)
    if not profile_env:
        print("WARNING: no profile baseline loaded — running raw env",
              file=sys.stderr)
    print(f"profile baseline ({len(profile_env)} keys): {profile_env}",
          flush=True)
    print(f"cumulative overlay ({len(CUMULATIVE_ENV)} keys): "
          f"{list(CUMULATIVE_ENV.keys())}", flush=True)

    rows: list[CellRow] = []
    for n in CONCURRENCIES:
        # Baseline cell: profile only.
        rows.append(measure("baseline", n, env_extras={},
                            profile_env=profile_env))
        # Cumulative cell: profile + 7 opt-ins.
        rows.append(measure("cumulative", n, env_extras=dict(CUMULATIVE_ENV),
                            profile_env=profile_env))

    # Group by N for pairwise compare.
    by_n: dict[int, dict[str, CellRow]] = {}
    for r in rows:
        by_n.setdefault(r.n, {})[r.label] = r

    # Write cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Cumulative opt-in test — Qwen3.6-27B-Q4_0 / pp2tp2 / N=[1,2,4,8]")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Profile:** `bench/profiles/optimized.toml` "
                 f"({len(profile_env)} baseline keys)")
    lines.append(f"- **Cumulative env:** {len(CUMULATIVE_ENV)} opt-in gates "
                 f"set simultaneously")
    lines.append(f"- **Runs/cell:** {RUNS_PER_CELL} measured + "
                 f"{WARMUP_RUNS} warmup, greedy / temp=0 / seed={SEED}")
    lines.append(f"- **Noise floor:** ±{NOISE_FLOOR_PCT}%")
    lines.append("")
    lines.append("**Hypothesis.** Each opt-in individually certed null "
                 "(≤ ±2%) at N=1. If they're additive, setting all 7 "
                 "together should give a measurable cumulative gain.")
    lines.append("")
    lines.append("**Cumulative env applied:**")
    lines.append("")
    for k, v in CUMULATIVE_ENV.items():
        lines.append(f"- `{k}={v}`")
    lines.append("")
    lines.append("## Measured cells")
    lines.append("")
    lines.append("| N | label | n | prefill ms | decode ms | aggregate tg t/s | "
                 "Δ vs baseline | correct? | VRAM peak GB | err |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|---:|---|")

    triage_rows: list[str] = []
    for n in CONCURRENCIES:
        b = by_n[n].get("baseline")
        c = by_n[n].get("cumulative")
        if not b or not c:
            continue
        for label, row, ref in (("baseline", b, None), ("cumulative", c, b)):
            if ref is None:
                delta = "—"
                correct = "—"
            else:
                if ref.agg_tps > 0 and row.agg_tps > 0:
                    pct = (row.agg_tps - ref.agg_tps) / ref.agg_tps * 100.0
                    delta = f"{pct:+.1f}%"
                else:
                    delta = "n/a"
                if ref.text_sha256 and row.text_sha256:
                    correct = ("match" if ref.text_sha256 == row.text_sha256
                               else "DIVERGENT")
                else:
                    correct = "missing"
            err_s = row.err or ""
            lines.append(
                f"| {row.n} | {label} | {row.n_runs} | "
                f"{row.prefill_ms:.0f} | {row.decode_ms:.0f} | "
                f"{row.agg_tps:.2f} | {delta} | {correct} | "
                f"{row.vram_peak_gb:.2f} | {err_s} |"
            )
        # triage
        if b.agg_tps > 0 and c.agg_tps > 0:
            pct = (c.agg_tps - b.agg_tps) / b.agg_tps * 100.0
            if c.err:
                verdict = f"BROKEN ({c.err})"
            elif (b.text_sha256 and c.text_sha256
                  and b.text_sha256 != c.text_sha256):
                verdict = f"DIVERGENT ({pct:+.1f}%)"
            elif abs(pct) < NOISE_FLOOR_PCT:
                verdict = f"null ({pct:+.1f}%)"
            elif pct > 0:
                verdict = f"win ({pct:+.1f}%)"
            else:
                verdict = f"loss ({pct:+.1f}%)"
            triage_rows.append(f"- N={n} → **{verdict}**")

    lines.append("")
    lines.append("## Triage")
    lines.append("")
    if triage_rows:
        lines.extend(triage_rows)
    else:
        lines.append("_no triage rows produced_")
    lines.append("")

    # Disposition
    lines.append("## Disposition")
    lines.append("")
    wins = [r for r in triage_rows if "win" in r]
    losses = [r for r in triage_rows if "loss" in r]
    divergents = [r for r in triage_rows if "DIVERGENT" in r]
    if divergents:
        lines.append(f"- **HALT-DIVERGENT** — {len(divergents)} cells "
                     "produce different output text")
    elif wins and not losses:
        lines.append(f"- **CUMULATIVE-WIN** — {len(wins)}/{len(triage_rows)} "
                     "N values show net positive over baseline; "
                     "individual gates are NOT safe to delete in isolation")
    elif losses and not wins:
        lines.append(f"- **CUMULATIVE-LOSS** — {len(losses)}/{len(triage_rows)} "
                     "N values regress; the kitchen-sink hurts overall")
    elif wins and losses:
        lines.append(f"- **CUMULATIVE-MIXED** — wins on {len(wins)}, "
                     f"losses on {len(losses)}; N-dependent")
    else:
        lines.append(f"- **CUMULATIVE-NULL** — no N value beats noise floor; "
                     "confirms the deletion plan, gates have no additive value")
    lines.append("")

    CERT.write_text("\n".join(lines))
    print(f"\nwrote {CERT}", flush=True)


if __name__ == "__main__":
    main()
