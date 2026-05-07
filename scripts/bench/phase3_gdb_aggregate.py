#!/usr/bin/env python3
"""
Aggregate gdb-batch all-thread backtrace samples. Each (sample, thread)
contributes ONE stack. Two views:
- raw leaf (frame 0): where the syscall sleep actually parked.
- first "interesting" (skip libc / std::sync / tokio::runtime infra):
  the first frame in flambeau / hip / app code, attribution.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

NOISE_PATTERNS = [
    r"^__GI_|^syscall\b|^__libc|^_Fork|^libc-2",
    r"futex_wait|wait_on_address|do_futex_wait|FUTEX_",
    r"std::sys::pal::|std::sys::sync::",
    r"std::sync::poison::condvar::|std::sync::poison::mutex::",
    r"std::sync::mpsc::|std::sync::mpmc::",
    r"std::thread::park\b|std::thread::Builder",
    r"tokio::runtime::scheduler::|tokio::runtime::task::core::",
    r"tokio::runtime::task::raw::|tokio::runtime::task::harness::",
    r"tokio::runtime::park::|tokio::runtime::driver::|tokio::runtime::time::",
    r"tokio::runtime::io::driver::",
    r"tokio::park::|tokio::loom::",
    r"core::ops::function::|core::future::poll::",
    r"core::pin::|core::ptr::",
    r"unwind_phase|_Unwind_",
    r"^_start\b|^__libc_start_main|^start_thread\b|^clone3\b",
    r"^pthread_cond|^pthread_mutex_lock|^pthread_mutex_unlock",
    r"^_int_malloc|^malloc|^free$|^realloc$|^calloc$",
    r"^\?\?\b",
    r"^mio::|^epoll_wait\b|^epoll_pwait\b",
    r"^poll\b",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS))


def is_interesting(sym: str) -> bool:
    if not sym or sym in ("??", ""):
        return False
    return not NOISE_RE.search(sym)


def parse_frame(line: str) -> str | None:
    m = re.match(r"^#\d+\s+(?:0x[0-9a-f]+\s+in\s+)?(\S[^(]*)", line)
    if not m:
        return None
    sym = m.group(1).strip()
    sym = re.split(r"\s+at\s+", sym, 1)[0]
    return sym


def shorten(sym: str, n: int = 110) -> str:
    if len(sym) <= n:
        return sym
    return sym[:n - 3] + "..."


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        text = f.read()
    samples = text.split("----SAMPLE")
    print(f"input has {len(samples) - 1} sample blocks")

    leaf_count: Counter = Counter()
    interesting_count: Counter = Counter()
    per_thread_interesting: Counter = Counter()
    stacks_seen = 0

    thread_re = re.compile(
        r"^Thread\s+\d+\s+\(Thread\s+0x[0-9a-f]+\s+\(LWP\s+\d+\)\s+\"([^\"]+)\"\):\s*$"
    )

    for sblock in samples:
        if not sblock.strip():
            continue
        # iterate line-by-line, accumulating frames per current thread
        current_name: str | None = None
        frames: list[str] = []

        def flush(name: str | None, fr: list[str]) -> None:
            nonlocal stacks_seen
            if not name or not fr:
                return
            stacks_seen += 1
            leaf_count[(name, fr[0])] += 1
            chosen = next((f for f in fr if is_interesting(f)), None)
            if chosen:
                interesting_count[chosen] += 1
                per_thread_interesting[(name, chosen)] += 1

        for ln in sblock.splitlines():
            mt = thread_re.match(ln)
            if mt:
                flush(current_name, frames)
                current_name = mt.group(1)
                frames = []
                continue
            sym = parse_frame(ln)
            if sym:
                frames.append(sym)
        flush(current_name, frames)

    print(f"\n--- aggregated {stacks_seen} stacks ---")
    print(f"\n--- raw leaf frame top-30 (per (thread, frame) — where the wait syscall is) ---")
    for (tname, sym), n in leaf_count.most_common(30):
        print(f"  {n:5d}  [{tname[:18]:<18}]  {shorten(sym)}")

    print(f"\n--- first interesting (in-app) frame top-50 (across all threads, where work/wait is attributed) ---")
    for sym, n in interesting_count.most_common(50):
        print(f"  {n:5d}  {shorten(sym)}")

    print(f"\n--- first interesting frame top-50 BY THREAD ---")
    for (tname, sym), n in per_thread_interesting.most_common(50):
        print(f"  {n:5d}  [{tname[:18]:<18}]  {shorten(sym)}")


if __name__ == "__main__":
    main()
