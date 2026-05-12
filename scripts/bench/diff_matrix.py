#!/usr/bin/env python3
"""Diff two run_matrix.py output JSONs and emit a per-cell ratio table.

Usage:
    python3 scripts/bench/diff_matrix.py \
        --baseline certs/perf/branch_vs_main/MAIN.json \
        --candidate certs/perf/branch_vs_main/HEAD.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def index_cells(doc: dict) -> dict[tuple[str, str, str, int], dict]:
    """Index per-(model, topo, path, n_concurrent) summaries.

    run_matrix.py groups cells by (model, topo, path) and stores
    one summary per concurrency inside `concurrency_results[]`. We
    flatten that into a single dict keyed by 4-tuple.
    """
    out = {}
    for cell in doc.get("cells", []):
        if "concurrency_results" not in cell or cell.get("skipped"):
            continue
        for r in cell["concurrency_results"]:
            key = (cell["model"], cell["topo"], cell["path"], r["n_concurrent"])
            out[key] = r
    return out


def fmt_ratio(cand: float, base: float) -> str:
    if base <= 0:
        return "  n/a   "
    r = cand / base
    arrow = "+" if r > 1.02 else ("-" if r < 0.98 else "=")
    return f"{r:.3f}x {arrow}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    base_doc = load(Path(args.baseline))
    cand_doc = load(Path(args.candidate))

    base = index_cells(base_doc)
    cand = index_cells(cand_doc)

    keys = sorted(set(base.keys()) | set(cand.keys()))

    print(f"baseline:  {args.baseline}  (cells={len(base)})")
    print(f"candidate: {args.candidate}  (cells={len(cand)})")
    print()
    print(f"{'model':<30} {'topo':<8} {'path':<11} {'N':<3}"
          f"  {'pref_ms_base':>12} {'pref_ms_cand':>12} {'pref_ratio':>10}"
          f"  {'tg_per_base':>11} {'tg_per_cand':>11} {'tg_per_ratio':>12}"
          f"  {'tg_agg_base':>11} {'tg_agg_cand':>11} {'tg_agg_ratio':>12}")

    pp_wins = pp_losses = pp_flats = 0
    tg_wins = tg_losses = tg_flats = 0

    for key in keys:
        model, topo, path, n = key
        b = base.get(key)
        c = cand.get(key)
        if b is None or c is None:
            present = "BASE only" if b else "CAND only"
            print(f"{model:<30} {topo:<8} {path:<11} {n:<3}  ({present})")
            continue
        bpp = b.get("prefill_ms_mean", 0.0) or 0.0
        cpp = c.get("prefill_ms_mean", 0.0) or 0.0
        btg_per = b.get("decode_tps_per_stream_mean", 0.0) or 0.0
        ctg_per = c.get("decode_tps_per_stream_mean", 0.0) or 0.0
        btg_agg = b.get("decode_tps_aggregate", 0.0) or 0.0
        ctg_agg = c.get("decode_tps_aggregate", 0.0) or 0.0
        # prefill ratio: lower ms = better, so invert
        pref_ratio = (bpp / cpp) if cpp > 0 else 0.0
        tg_per_ratio = (ctg_per / btg_per) if btg_per > 0 else 0.0
        tg_agg_ratio = (ctg_agg / btg_agg) if btg_agg > 0 else 0.0

        if bpp > 0 and cpp > 0:
            if pref_ratio > 1.02: pp_wins += 1
            elif pref_ratio < 0.98: pp_losses += 1
            else: pp_flats += 1
        if btg_agg > 0:
            if tg_agg_ratio > 1.02: tg_wins += 1
            elif tg_agg_ratio < 0.98: tg_losses += 1
            else: tg_flats += 1

        print(
            f"{model:<30} {topo:<8} {path:<11} {n:<3}"
            f"  {bpp:>12.1f} {cpp:>12.1f} {fmt_ratio(bpp, cpp):>10}"
            f"  {btg_per:>11.2f} {ctg_per:>11.2f} {fmt_ratio(ctg_per, btg_per):>12}"
            f"  {btg_agg:>11.2f} {ctg_agg:>11.2f} {fmt_ratio(ctg_agg, btg_agg):>12}"
        )

    print()
    print(f"prefill: {pp_wins} faster / {pp_flats} flat / {pp_losses} slower (>2% gate)")
    print(f"decode (aggregate): {tg_wins} faster / {tg_flats} flat / {tg_losses} slower (>2% gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
