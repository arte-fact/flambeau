#!/usr/bin/env python3
"""
N=1 single-stream perf comparison: legacy fast-path (`decode_logits`)
vs batched-hybrid forced (`forward_decode_batched_hybrid` at N=1)
across the V1-target models on pp2tp2.

Question: how much do we lose if we make `forward_decode_batched_*`
the only decode path and delete the legacy fast-path branch in
`routes.rs::decode_via_scheduler_into`?

If the loss is small enough (≤ ~3%) on production models (27B, 35B),
we collapse the dual-path infrastructure: drop `decode_logits` for
hybrid, drop the fast-path branch, drop `FLAMBEAU_NO_FAST_PATH`.
Side-benefit: eliminates the cross-slot prefill-decode overlap as a
race surface, which is the leading suspect for the 35B-A3B / pp2tp2
multi-slot divergence (#16).

Method: per (model, path) cell, boot the server with the appropriate
env, warmup, run 3 measured single-stream calls, record median
prefill_ms / decode_ms / per-stream tg-tps / VRAM peak. Compare
batched vs legacy per model.

Output: certs/env_impact/N1_FAST_VS_BATCHED_PP2TP2.md
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
CERT = ROOT / "certs" / "env_impact" / "N1_FAST_VS_BATCHED_PP2TP2.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64
SEED = 0
SLOTS = 1            # single inflight slot
WARMUP_RUNS = 2
RUNS_PER_CELL = 3
NOISE_FLOOR_PCT = 2.0
COOLDOWN_C = 70.0
COOLDOWN_MAX_S = 240.0

MODELS = [
    ("qwen35-9b-q4_1",      "/artefact/models/Qwen3.5-9B-Q4_1.gguf"),
    ("qwen36-27b-q4_0",     "/artefact/models/Qwen3.6-27B-Q4_0.gguf"),
    ("qwen36-35b-a3b-q4_0", "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"),
]

# Two paths, same single-stream workload.
PATHS = [
    ("legacy_fast_path", {}),                          # default — fast path active
    ("batched_forced",   {"FLAMBEAU_NO_FAST_PATH": "1"}),  # forces batched-hybrid even at N=1
]


@dataclasses.dataclass
class CellRow:
    model: str
    path_label: str
    n_runs: int
    prefill_ms: float
    decode_ms: float
    tps: float
    vram_peak_gb: float
    text_sha256: str
    err: str | None = None


def measure(model_id: str, model_path: str,
            path_label: str, env_extras: dict[str, str],
            profile_env: dict[str, str]) -> CellRow:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"n1_fast_vs_batched__{model_id}__{path_label}.log"

    print(f"\n=== {model_id} / {path_label}  env={env_extras} ===", flush=True)
    boot_t0 = time.time()
    proc, port = rei.boot_with_env(
        model_id, model_path, TOPO,
        env_extras=env_extras, slots=SLOTS,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    ready = rm.wait_ready(port, timeout_s=600.0, model_id=model_id)
    if not ready:
        rm.kill_server(proc)
        return CellRow(model=model_id, path_label=path_label,
                       n_runs=0, prefill_ms=0, decode_ms=0, tps=0,
                       vram_peak_gb=0, text_sha256="",
                       err=f"boot timeout {time.time()-boot_t0:.0f}s")
    print(f"   boot {time.time()-boot_t0:.0f}s — measuring...", flush=True)

    try:
        # warmup
        for _ in range(WARMUP_RUNS):
            _ = rei._run_concurrent_batch(port, 1, TG_LEN,
                                          base_seed=SEED, sample_vram=False)
        # measure
        prefills, decodes, tpss, vrams = [], [], [], []
        text_hash = ""
        last_err = None
        for run_i in range(RUNS_PER_CELL):
            r = rei._run_concurrent_batch(port, 1, TG_LEN,
                                          base_seed=SEED, sample_vram=True)
            if r.get("err"):
                last_err = r["err"]
                continue
            prefills.append(r["prefill_ms_median"])
            decodes.append(r["decode_ms_median"])
            tpss.append(r["aggregate_tps"])
            if r.get("vram_peak_gb", 0) > 0:
                vrams.append(r["vram_peak_gb"])
            if run_i == 0:
                text_hash = r.get("text_sha256", "")[:16]

        if not tpss:
            return CellRow(model=model_id, path_label=path_label,
                           n_runs=0, prefill_ms=0, decode_ms=0, tps=0,
                           vram_peak_gb=max(vrams) if vrams else 0,
                           text_sha256="",
                           err=last_err or "no successful batches")
        return CellRow(
            model=model_id, path_label=path_label,
            n_runs=len(tpss),
            prefill_ms=statistics.median(prefills),
            decode_ms=statistics.median(decodes),
            tps=statistics.median(tpss),
            vram_peak_gb=max(vrams) if vrams else 0,
            text_sha256=text_hash,
        )
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        rm.cooldown_until_safe(COOLDOWN_C, COOLDOWN_MAX_S)


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    print(f"profile baseline ({len(profile_env)} keys): {profile_env}",
          flush=True)

    rows: list[CellRow] = []
    for model_id, model_path in MODELS:
        for path_label, env_extras in PATHS:
            rows.append(measure(model_id, model_path,
                                path_label, env_extras, profile_env))

    # Group by model for pairwise compare.
    by_model: dict[str, dict[str, CellRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, {})[r.path_label] = r

    # Cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# N=1 fast-path vs batched-hybrid forced — pp2tp2")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Topology:** pp2tp2 ({TOPO['devices']})")
    lines.append(f"- **Slots:** {SLOTS} | **N:** 1 | "
                 f"**ctx_cap:** {CTX_CAP} | **tg_len:** {TG_LEN}")
    lines.append(f"- **Profile:** `{PROFILE.name}` "
                 f"({len(profile_env)} baseline keys)")
    lines.append(f"- **Runs/cell:** {RUNS_PER_CELL} measured + "
                 f"{WARMUP_RUNS} warmup, greedy / temp=0 / seed={SEED}")
    lines.append("")
    lines.append("**Question.** Can we delete the legacy `decode_logits` "
                 "fast-path and force `forward_decode_batched_hybrid` for "
                 "all decode (including N=1)? Decision rule:")
    lines.append("")
    lines.append("- batched_forced loss vs legacy ≤ ~3% on prod models "
                 "(27B / 35B) → **collapse paths**: delete the fast-path branch")
    lines.append("- batched_forced loss > 3% on a prod model → keep the gate "
                 "or build an N=1 fast path **inside** the batched function")
    lines.append("")
    lines.append("Side-benefit if collapsed: eliminates the cross-slot "
                 "prefill-decode overlap as a race surface (leading suspect "
                 "for #16 35B-A3B/pp2tp2 multi-slot divergence — slot 1's "
                 "prefill currently overlaps slot 0's legacy decode).")
    lines.append("")
    lines.append("## Measured cells")
    lines.append("")
    lines.append("| model | path | n | prefill ms | decode ms | tg t/s | "
                 "Δ vs legacy | correct? | VRAM peak GB | err |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---|")

    triage: list[str] = []
    for model_id, _ in MODELS:
        legacy = by_model[model_id].get("legacy_fast_path")
        batched = by_model[model_id].get("batched_forced")
        if not legacy or not batched:
            continue
        for label, row, ref in (
            ("legacy_fast_path", legacy, None),
            ("batched_forced",   batched, legacy),
        ):
            if ref is None:
                delta_str = "—"
                correct = "—"
            else:
                if ref.tps > 0 and row.tps > 0:
                    pct = (row.tps - ref.tps) / ref.tps * 100.0
                    delta_str = f"{pct:+.1f}%"
                else:
                    delta_str = "n/a"
                correct = (
                    "match" if (ref.text_sha256 and row.text_sha256
                                 and ref.text_sha256 == row.text_sha256)
                    else ("DIVERGENT" if (ref.text_sha256 and row.text_sha256)
                          else "missing"))
            err_s = row.err or ""
            lines.append(
                f"| {row.model} | {label} | {row.n_runs} | "
                f"{row.prefill_ms:.0f} | {row.decode_ms:.0f} | "
                f"{row.tps:.2f} | {delta_str} | {correct} | "
                f"{row.vram_peak_gb:.2f} | {err_s} |"
            )
        # triage line
        if legacy.tps > 0 and batched.tps > 0:
            pct = (batched.tps - legacy.tps) / legacy.tps * 100.0
            if batched.err or legacy.err:
                v = f"BROKEN ({batched.err or legacy.err})"
            elif (legacy.text_sha256 and batched.text_sha256
                  and legacy.text_sha256 != batched.text_sha256):
                v = f"DIVERGENT ({pct:+.1f}%)"
            elif abs(pct) <= 3.0:
                v = f"acceptable ({pct:+.1f}% within ±3%)"
            elif pct > 3.0:
                v = f"batched-wins ({pct:+.1f}%)"
            else:
                v = f"batched-loses ({pct:+.1f}%)"
            triage.append(f"- **{model_id}** → {v}")

    lines.append("")
    lines.append("## Triage")
    lines.append("")
    if triage:
        lines.extend(triage)
    else:
        lines.append("_no triage rows produced_")
    lines.append("")

    # Decision recommendation.
    lines.append("## Disposition")
    lines.append("")
    losses_3pct = []
    wins_or_acceptable = []
    for model_id, _ in MODELS:
        legacy = by_model[model_id].get("legacy_fast_path")
        batched = by_model[model_id].get("batched_forced")
        if not (legacy and batched and legacy.tps > 0 and batched.tps > 0):
            continue
        pct = (batched.tps - legacy.tps) / legacy.tps * 100.0
        if pct < -3.0:
            losses_3pct.append((model_id, pct))
        else:
            wins_or_acceptable.append((model_id, pct))

    if losses_3pct:
        lines.append(f"- **KEEP-FAST-PATH** — batched_forced loses > 3% on "
                     f"{len(losses_3pct)} model(s) "
                     f"({', '.join(f'{m} {p:+.1f}%' for m,p in losses_3pct)}). ")
        lines.append("  Either keep the gate or build an N=1 internal fast "
                     "path inside `forward_decode_batched_hybrid`.")
    else:
        lines.append("- **COLLAPSE-PATHS** — batched_forced is within ±3% on "
                     "all measured models. Plan: delete `decode_logits` for "
                     "hybrid, delete the routes.rs:734 fast-path branch, "
                     "delete `FLAMBEAU_NO_FAST_PATH`.")
    lines.append("")

    CERT.write_text("\n".join(lines))
    print(f"\nwrote {CERT}", flush=True)


if __name__ == "__main__":
    main()
