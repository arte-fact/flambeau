#!/usr/bin/env python3
"""Summarise a rocprofv3 kernel_trace.csv: per-kernel-name aggregate
ns and call count. Output sorted by total time descending."""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def short_name(raw: str) -> str:
    # Strip __device_stub__, template args, return type, args.
    s = raw.strip().strip('"')
    s = re.sub(r"^.*?(?:::|\s)", "", s, count=0)  # noop placeholder
    # Drop template args.
    depth = 0
    out = []
    for c in s:
        if c == "<":
            depth += 1
            continue
        if c == ">":
            depth -= 1
            continue
        if depth == 0:
            out.append(c)
    s = "".join(out)
    # Drop arg list.
    if "(" in s:
        s = s.split("(", 1)[0]
    # Strip leading void/etc.
    s = s.replace("__device_stub__", "")
    return s.strip()


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <kernel_trace.csv> [--top N]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    top_n = 20
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        top_n = int(sys.argv[i + 1])

    totals: dict[str, list[int]] = defaultdict(list)
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = short_name(row.get("Kernel_Name", ""))
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            totals[name].append(end - start)

    rows = [
        (name, sum(d), len(d), sum(d) // max(1, len(d)))
        for name, d in totals.items()
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    grand_total = sum(r[1] for r in rows)
    grand_calls = sum(r[2] for r in rows)
    print(f"file: {path}")
    print(f"total kernel time: {grand_total/1e6:.2f} ms across {grand_calls} launches")
    print("-" * 90)
    print(f"{'kernel':<60}  {'total(ms)':>10}  {'calls':>8}  {'mean(us)':>10}  {'%':>6}")
    print("-" * 90)
    for name, total_ns, calls, mean_ns in rows[:top_n]:
        pct = total_ns / grand_total * 100 if grand_total else 0
        print(
            f"{name[:60]:<60}  {total_ns/1e6:>10.3f}  {calls:>8}  "
            f"{mean_ns/1e3:>10.2f}  {pct:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
