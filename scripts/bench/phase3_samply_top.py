#!/usr/bin/env python3
"""
Parse a samply-saved gecko profile (.json.gz) and print top-N
functions by self-CPU-time across all threads, plus per-thread
totals. Self-CPU-time = sum of inter-sample wall-time intervals
where the function is the leaf frame.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter


def schema_idx(samples: dict, key: str) -> int:
    sch = samples["schema"]
    return sch[key]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase3_samply_top.py <profile.json.gz> [topN=40]")
        sys.exit(2)
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    with gzip.open(path) as f:
        prof = json.load(f)

    threads = prof["threads"]
    func_self: Counter[tuple[str, str]] = Counter()
    thread_total: Counter[str] = Counter()

    for th in threads:
        name = th.get("name", "?")
        st = th["stackTable"]
        fr = th["frameTable"]
        fu = th["funcTable"]
        S = th["stringArray"] if "stringArray" in th else th["stringTable"]
        samples = th["samples"]
        ti = schema_idx(samples, "time")
        si = schema_idx(samples, "stack")
        rows = samples["data"]

        st_frame = st["frame"]
        fr_func = fr["func"]
        fu_name = fu["name"]
        st_prefix = st["prefix"]

        prev_t = None
        for r in rows:
            s = r[si]
            t = r[ti]
            dt = (t - prev_t) if prev_t is not None else 0.0
            prev_t = t
            if s is None:
                continue
            # leaf
            leaf_frame = st_frame[s]
            leaf_func = fr_func[leaf_frame]
            leaf_name = S[fu_name[leaf_func]]
            func_self[(name, leaf_name)] += dt
            thread_total[name] += dt

    print(f"\n=== profile {path} ===")
    print(f"threads: {len(threads)}")
    print("\n--- per-thread total time (ms) ---")
    for n, ms in thread_total.most_common(20):
        print(f"  {ms:9.1f}  {n}")

    print(f"\n--- top {top_n} by SELF time (ms) ---")
    for (thr, fn), ms in func_self.most_common(top_n):
        # demangle-friendly trim
        sn = fn
        if len(sn) > 110:
            sn = sn[:107] + "..."
        print(f"  {ms:9.1f}  [{thr[:18]:<18}]  {sn}")


if __name__ == "__main__":
    main()
