#!/usr/bin/env python3
"""
Localize the 35B-A3B-Q4_0 / pp2tp2 multi-slot race to a specific
(stage, layer, slot, metric) by comparing per-layer state dumps
across two boots.

The reproducer (`repro_35b_divergence.py`) confirmed:
  - slot 0 is bit-deterministic across boots
  - slot 1 produces different output across boots
  - per-token GDN path is stable, batched-GDN is racy

Method:
  Boot 35B-A3B/pp2tp2/slots=2 with FLAMBEAU_LAYER_STATE_DUMP=1 and
  send 2 concurrent /v1/chat/completions calls. The server emits
  `[STATE-DUMP] N=2 slot=S stage=X layer=L ...` lines per layer
  per slot, after each batched-decode call.

  Repeat the boot. Diff the [STATE-DUMP] lines positionally — the
  first index where two boots disagree localises the layer.

Output: certs/env_impact/REPRO_35B_LAYER_DIFF.md with the first
divergent dump line + context around it.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore[import-not-found]
import run_env_impact as rei  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "REPRO_35B_LAYER_DIFF.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-35b-a3b-q4_0"
MODEL_PATH = "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64
# At slots=2 / N=2, prefills serialise (slot 0 finishes decode before
# slot 1 prefill done) and the batched-decode path is never called.
# slots=8 / N=8 was the cell that originally hit DIVERGENT in the
# 2026-05-06 N-sweep cert, so use that here.
SLOTS = 8
N_CONCURRENT = 8
SEED_BASE = 0


# Pattern to extract the structured fields. Order-preserving — we just
# diff string-equality of (slot, stage, layer, metric_string).
DUMP_RE = re.compile(r"\[STATE-DUMP\]\s+(.*)")


@dataclasses.dataclass
class BootRecord:
    label: str
    log_path: Path
    dump_lines: list[str]   # ordered, one per layer-per-slot-per-step
    seed0_text: str
    seed1_text: str
    err: str | None = None


def boot_run(label: str, profile_env: dict[str, str]) -> BootRecord:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"repro_35b_layer_diff__{label}.log"

    print(f"\n=== Boot {label} ===", flush=True)
    boot_t0 = time.time()
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras={
            "FLAMBEAU_LAYER_STATE_DUMP": "1",
            # routes.rs:771 fast-path bypasses forward_decode_batched_hybrid
            # when no other slot is concurrently decoding. Even at N=8
            # prefills serialize, so each decode runs alone via legacy
            # decode_logits. This env forces the batched-hybrid path
            # so our state dump fires.
            "FLAMBEAU_NO_FAST_PATH": "1",
            # Enable scheduler tracing so we see the dispatch events.
            "FLAMBEAU_TRACE_BATCH": "1",
        },
        slots=SLOTS,
        log_path=log_path,
        ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    ready = rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID)
    if not ready:
        rm.kill_server(proc)
        return BootRecord(label=label, log_path=log_path,
                          dump_lines=[], seed0_text="", seed1_text="",
                          err=f"boot timeout {time.time()-boot_t0:.0f}s")
    print(f"   boot {time.time()-boot_t0:.0f}s", flush=True)

    # Run N_CONCURRENT concurrent streams.
    import threading
    results: list[dict] = [None] * N_CONCURRENT  # type: ignore[list-item]

    def worker(i: int, seed: int) -> None:
        results[i] = rei.stream_one_call_capture_text(
            port, "x" * 64, max_tokens=TG_LEN, seed=seed)

    threads: list[threading.Thread] = []
    for i in range(N_CONCURRENT):
        t = threading.Thread(target=worker, args=(i, SEED_BASE + i), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=900.0)
    rm.kill_server(proc)
    time.sleep(1.0)

    s0 = results[0] or {}
    s1 = (results[1] if len(results) > 1 else {}) or {}

    # Parse dump lines from log.
    dump_lines: list[str] = []
    err = s0.get("err") or s1.get("err")
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "[STATE-DUMP]" in line:
                    m = DUMP_RE.search(line)
                    if m:
                        dump_lines.append(m.group(1).strip())
    except Exception as e:
        err = err or f"log read: {e!r}"

    print(f"   captured {len(dump_lines)} dump lines", flush=True)
    return BootRecord(
        label=label, log_path=log_path,
        dump_lines=dump_lines,
        seed0_text=s0.get("text", "") if not s0.get("err") else "",
        seed1_text=s1.get("text", "") if not s1.get("err") else "",
        err=err,
    )


def first_diff_index(a: list[str], b: list[str]) -> int | None:
    """Return the index of the first differing line, or None if identical
    over min(len(a), len(b))."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def slot1_first_diff(a: list[str], b: list[str]) -> int | None:
    """Same but only over slot=1 lines — the racy slot per the prior
    reproducer. Used to confirm the divergence is exactly there."""
    a_idx = [(i, l) for i, l in enumerate(a) if "slot=1" in l]
    b_idx = [(i, l) for i, l in enumerate(b) if "slot=1" in l]
    for k, ((ia, la), (ib, lb)) in enumerate(zip(a_idx, b_idx)):
        if la != lb:
            return ia
    return None


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN}", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    print(f"profile baseline: {profile_env}", flush=True)

    boot1 = boot_run("v1", profile_env=profile_env)
    boot2 = boot_run("v2", profile_env=profile_env)

    diff_idx_all = first_diff_index(boot1.dump_lines, boot2.dump_lines)
    diff_idx_slot1 = slot1_first_diff(boot1.dump_lines, boot2.dump_lines)

    seed0_match = boot1.seed0_text == boot2.seed0_text
    seed1_match = boot1.seed1_text == boot2.seed1_text

    # Cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# 35B-A3B-Q4_0 / pp2tp2 — per-layer state-dump localisation")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Model:** {MODEL_ID} on {TOPO['devices']} ({TOPO['mesh_mode']})")
    lines.append(f"- **Slots:** {SLOTS} | **Concurrent:** 2 | "
                 f"**ctx_cap:** {CTX_CAP} | **tg_len:** {TG_LEN}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Two boots with `FLAMBEAU_LAYER_STATE_DUMP=1`, 2 concurrent")
    lines.append("greedy chat calls per boot, parse `[STATE-DUMP]` lines from")
    lines.append("server log, position-diff the two dump streams.")
    lines.append("")
    lines.append("## Top-level outcome")
    lines.append("")
    lines.append(f"- Boot v1 captured **{len(boot1.dump_lines)}** dump lines (errs: {boot1.err or 'none'})")
    lines.append(f"- Boot v2 captured **{len(boot2.dump_lines)}** dump lines (errs: {boot2.err or 'none'})")
    lines.append(f"- Generated text seed=0 match: `{seed0_match}`")
    lines.append(f"- Generated text seed=1 match: `{seed1_match}`")
    lines.append("")
    if diff_idx_all is None:
        lines.append("- **No layer-state divergence detected** — both boots' dump streams identical line-for-line. ")
        lines.append("  If the generated text *did* diverge (seed=1 match=False above), the race is downstream of the dumped state ")
        lines.append("  (e.g. in the output head, sampler, or in a buffer not currently dumped).")
    else:
        a_line = boot1.dump_lines[diff_idx_all]
        b_line = boot2.dump_lines[diff_idx_all]
        lines.append(f"- **First divergent dump line at index {diff_idx_all}** (out of "
                     f"{min(len(boot1.dump_lines), len(boot2.dump_lines))}):")
        lines.append("")
        lines.append("```")
        lines.append(f"v1: {a_line}")
        lines.append(f"v2: {b_line}")
        lines.append("```")
    lines.append("")

    if diff_idx_slot1 is not None:
        a_line = boot1.dump_lines[diff_idx_slot1]
        b_line = boot2.dump_lines[diff_idx_slot1]
        lines.append(f"## First slot=1 divergence (at index {diff_idx_slot1})")
        lines.append("")
        lines.append("```")
        lines.append(f"v1: {a_line}")
        lines.append(f"v2: {b_line}")
        lines.append("```")
        lines.append("")

    lines.append("## Context window around first divergence")
    lines.append("")
    if diff_idx_all is not None:
        ctx_lo = max(0, diff_idx_all - 4)
        ctx_hi = min(len(boot1.dump_lines), diff_idx_all + 5)
        lines.append("```")
        for i in range(ctx_lo, ctx_hi):
            mark = " ←" if i == diff_idx_all else "  "
            a = boot1.dump_lines[i] if i < len(boot1.dump_lines) else ""
            b = boot2.dump_lines[i] if i < len(boot2.dump_lines) else ""
            if a == b:
                lines.append(f"   [{i:4d}] {a}{mark}")
            else:
                lines.append(f"v1 [{i:4d}] {a}{mark}")
                lines.append(f"v2 [{i:4d}] {b}{mark}")
        lines.append("```")
    else:
        lines.append("_no divergence_")
    lines.append("")

    lines.append("## Generated text per boot")
    lines.append("")
    for r in (boot1, boot2):
        lines.append(f"### Boot {r.label}")
        lines.append("")
        lines.append("seed=0:")
        lines.append("")
        lines.append("```")
        lines.append(r.seed0_text or "<empty>")
        lines.append("```")
        lines.append("")
        lines.append("seed=1:")
        lines.append("")
        lines.append("```")
        lines.append(r.seed1_text or "<empty>")
        lines.append("```")
        lines.append("")

    CERT.write_text("\n".join(lines))
    print(f"\nwrote {CERT}", flush=True)


if __name__ == "__main__":
    main()
