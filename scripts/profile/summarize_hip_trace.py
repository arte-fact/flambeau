#!/usr/bin/env python3
"""Summarise a rocprofv3 hip_api_trace.csv: per-function total ns +
calls. Sorted by total time descending."""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <hip_api_trace.csv> [--top N]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    top_n = 15
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        top_n = int(sys.argv[i + 1])
    by_fn: dict[str, list[int]] = defaultdict(list)
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fn = row["Function"].strip().strip('"')
            try:
                s = int(row["Start_Timestamp"])
                e = int(row["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            by_fn[fn].append(e - s)
    rows = [
        (fn, sum(d), len(d), sum(d) // max(1, len(d)))
        for fn, d in by_fn.items()
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    total = sum(r[1] for r in rows)
    calls = sum(r[2] for r in rows)
    print(f"file: {path}")
    print(f"total HIP API time: {total/1e6:.2f} ms across {calls} calls")
    print("-" * 80)
    print(f"{'HIP function':<45}  {'total(ms)':>10}  {'calls':>8}  {'mean(us)':>10}  {'%':>6}")
    print("-" * 80)
    for fn, t, c, m in rows[:top_n]:
        pct = t / total * 100 if total else 0
        print(f"{fn[:45]:<45}  {t/1e6:>10.3f}  {c:>8}  {m/1e3:>10.2f}  {pct:>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
