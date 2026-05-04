#!/usr/bin/env python3
"""Read matrix.json -> emit a markdown summary table per (model, topology)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def fmt_int(x):
    return "—" if x is None else f"{int(round(x))}"

def fmt_float(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.matrix_json).read_text())
    cells = data["cells"]
    concs = data["concurrencies"]

    # Group by (model, topo). For each, render a table:
    #   N | path no_batched: prefill_ms | agg_tps | per-stream tps
    #     | path batched   : prefill_ms | agg_tps | per-stream tps
    #     | speedup (batched/no_batched aggregate)
    by_mt: dict[tuple[str, str], dict[str, dict]] = {}
    for c in cells:
        if "skipped" in c:
            continue
        if "concurrency_results" not in c:
            continue
        key = (c["model"], c["topo"])
        by_mt.setdefault(key, {})[c["path"]] = c

    out_lines = []
    out_lines.append(f"# Bench matrix — {data.get('started','')[:19]} → {data.get('finished','')[:19]}")
    out_lines.append("")
    out_lines.append(f"max_tokens={data['max_tokens']}, concurrencies={concs}")
    out_lines.append("")

    for (model, topo), paths in sorted(by_mt.items()):
        out_lines.append(f"## {model} / {topo}")
        out_lines.append("")
        prompt_tokens = None
        for p in paths.values():
            if p["concurrency_results"]:
                prompt_tokens = p["concurrency_results"][0].get("prompt_tokens")
                break
        if prompt_tokens:
            out_lines.append(f"prompt_tokens={prompt_tokens}, completion_tokens_per_stream={data['max_tokens']}")
            out_lines.append("")

        out_lines.append("| N | path | prefill_ms (mean) | per-stream tps (mean) | aggregate tps | n_ok | n_err |")
        out_lines.append("|---|------|-------------------|------------------------|---------------|------|-------|")
        for n in concs:
            for path in ("no_batched", "batched"):
                cell = paths.get(path)
                if not cell:
                    continue
                results = {r["n_concurrent"]: r for r in cell["concurrency_results"]}
                r = results.get(n)
                if not r:
                    continue
                out_lines.append(
                    f"| {n} | {path} | {fmt_int(r.get('prefill_ms_mean'))} | "
                    f"{fmt_float(r.get('decode_tps_per_stream_mean'))} | "
                    f"{fmt_float(r.get('decode_tps_aggregate'))} | "
                    f"{r.get('n_ok','—')} | {r.get('n_err','—')} |"
                )

        # Speedup table
        out_lines.append("")
        out_lines.append("| N | aggregate tps no_batched | aggregate tps batched | batched / no_batched |")
        out_lines.append("|---|--------------------------|-----------------------|----------------------|")
        for n in concs:
            nb = paths.get("no_batched", {}).get("concurrency_results", [])
            ba = paths.get("batched", {}).get("concurrency_results", [])
            nb_r = next((r for r in nb if r["n_concurrent"] == n), None)
            ba_r = next((r for r in ba if r["n_concurrent"] == n), None)
            nb_tps = nb_r.get("decode_tps_aggregate") if nb_r else None
            ba_tps = ba_r.get("decode_tps_aggregate") if ba_r else None
            speedup = (ba_tps / nb_tps) if (nb_tps and ba_tps) else None
            out_lines.append(
                f"| {n} | {fmt_float(nb_tps)} | {fmt_float(ba_tps)} | "
                f"{fmt_float(speedup, 2) if speedup else '—'}× |"
            )
        out_lines.append("")

    # Skipped cells
    skipped = [c for c in cells if "skipped" in c]
    if skipped:
        out_lines.append("## Skipped (infeasible)")
        out_lines.append("")
        for c in skipped:
            out_lines.append(f"- {c['model']} / {c['topo']}: {c['skipped']}")
        out_lines.append("")

    text = "\n".join(out_lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
