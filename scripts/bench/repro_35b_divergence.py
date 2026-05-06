#!/usr/bin/env python3
"""
Reproduce + characterize the 35B-A3B-Q4_0 / pp2tp2 multi-slot output
divergence flagged by the N=[1,2,4,8] env-impact sweep.

Hypothesis to test:
  H1 (within-env race) — same env, same prompt, same seed, two
    independent boots produce different text. Implies a real race
    in the multi-slot decode path.
  H2 (across-env numerical drift) — same env produces identical text
    across boots, but two semantically-equivalent code paths
    (batched-GDN vs per-token-GDN) produce different text. Implies
    FP-reduction-order divergence amplified by MoE routing.

Method:
  - Boot 35B-A3B-Q4_0 / pp2tp2 / slots=2 with INFLIGHT_SLOTS=2.
  - Run identical chat request 3 times with seed=0:
      Run A: env baseline (default batched-GDN)
      Run B: env baseline AGAIN (control for H1)
      Run C: FLAMBEAU_GDN_NO_BATCHED=1 (force per-token-GDN)
  - Capture full content text per run.
  - Print: (A vs B) text equality + (A vs C) common-prefix length +
    Levenshtein distance + sample diff.

Verdict:
  - A == B and A != C  → H2 confirmed; fix = pick canonical GDN path.
  - A != B (any extent) → H1 confirmed; race exists; localize next.

Output: certs/env_impact/REPRO_35B_DIVERGENCE.md with the three
captured texts + verdict.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore[import-not-found]
import run_env_impact as rei  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "certs" / "env_impact" / "REPRO_35B_DIVERGENCE.md"
PROFILE = ROOT / "bench" / "profiles" / "optimized.toml"

MODEL_ID = "qwen36-35b-a3b-q4_0"
MODEL_PATH = "/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
TOPO = {"devices": "0,2,1,3", "mesh_mode": "pp+tp",
        "pp_size": 2, "tp_size": 2}
CTX_CAP = 4096
TG_LEN = 64        # short-ish output to keep diffs readable
SEED = 0
N_CONCURRENT = 2   # smallest N at which divergence appeared
SLOTS = 2


@dataclasses.dataclass
class RunOutput:
    label: str
    env_extras: dict
    seed0_text: str
    seed0_sha: str
    seed1_text: str   # second concurrent stream — also captured for completeness
    seed1_sha: str
    err: str | None = None


def _two_streams_with_text(port: int) -> dict:
    """Run N=2 concurrent streams at fixed seeds; capture both texts."""
    import threading
    results: list[dict] = [None, None]  # type: ignore[list-item]

    def worker(i: int, seed: int) -> None:
        results[i] = rei.stream_one_call_capture_text(
            port, "x" * 64, max_tokens=TG_LEN, seed=seed)

    t0 = threading.Thread(target=worker, args=(0, SEED), daemon=True)
    t1 = threading.Thread(target=worker, args=(1, SEED + 1), daemon=True)
    t0.start(); t1.start()
    t0.join(timeout=600.0); t1.join(timeout=600.0)
    return {"seed0": results[0], "seed1": results[1]}


def boot_and_run(label: str, env_extras: dict,
                 profile_env: dict) -> RunOutput:
    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"repro_35b_divergence__{label}.log"

    print(f"\n=== Run {label}  env_extras={env_extras} ===", flush=True)
    boot_t0 = time.time()
    proc, port = rei.boot_with_env(
        MODEL_ID, MODEL_PATH, TOPO,
        env_extras=env_extras, slots=SLOTS,
        log_path=log_path, ctx_cap=CTX_CAP,
        profile_env=profile_env,
    )
    ready = rm.wait_ready(port, timeout_s=600.0, model_id=MODEL_ID)
    if not ready:
        rm.kill_server(proc)
        return RunOutput(label=label, env_extras=env_extras,
                         seed0_text="", seed0_sha="",
                         seed1_text="", seed1_sha="",
                         err=f"boot timeout {time.time()-boot_t0:.0f}s")
    print(f"   boot {time.time()-boot_t0:.0f}s", flush=True)

    # Warmup once (kernels JIT, prefix cache primes).
    _ = _two_streams_with_text(port)
    # Measure once — we want bit-deterministic capture, not median.
    pair = _two_streams_with_text(port)
    rm.kill_server(proc)
    time.sleep(2.0)

    s0 = pair["seed0"] or {}
    s1 = pair["seed1"] or {}
    s0_text = s0.get("text", "") if not s0.get("err") else ""
    s1_text = s1.get("text", "") if not s1.get("err") else ""
    return RunOutput(
        label=label, env_extras=env_extras,
        seed0_text=s0_text,
        seed0_sha=hashlib.sha256(s0_text.encode()).hexdigest()[:16],
        seed1_text=s1_text,
        seed1_sha=hashlib.sha256(s1_text.encode()).hexdigest()[:16],
        err=s0.get("err") or s1.get("err"),
    )


def common_prefix_chars(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def levenshtein(a: str, b: str) -> int:
    """Iterative DP, O(len(a) * len(b)) — fine for <=10k chars."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    return prev[-1]


def fmt_diff_window(a: str, b: str, around: int = 80) -> str:
    """Find first divergent char position; show ~around chars of each."""
    pos = common_prefix_chars(a, b)
    a_seg = a[max(0, pos - 20):pos + around]
    b_seg = b[max(0, pos - 20):pos + around]
    a_seg = a_seg.replace("\n", "\\n")
    b_seg = b_seg.replace("\n", "\\n")
    return (f"divergence at char {pos}\n"
            f"  A: ...{a_seg!r}\n"
            f"  B: ...{b_seg!r}")


def main() -> None:
    if not rm.BIN.exists():
        print(f"missing {rm.BIN} — build with: cargo build --release "
              f"--features hip_serve -p flambeau-cli", file=sys.stderr)
        sys.exit(2)
    profile_env = rei.load_profile_env(PROFILE)
    print(f"profile baseline: {profile_env}", flush=True)

    runs: list[RunOutput] = []
    runs.append(boot_and_run("A_baseline_v1", env_extras={},
                             profile_env=profile_env))
    runs.append(boot_and_run("B_baseline_v2", env_extras={},
                             profile_env=profile_env))
    runs.append(boot_and_run("C_per_token_gdn",
                             env_extras={"FLAMBEAU_GDN_NO_BATCHED": "1"},
                             profile_env=profile_env))

    a, b, c = runs

    # H1 test: A vs B (same env, two boots)
    h1 = a.seed0_text == b.seed0_text and a.seed1_text == b.seed1_text
    # H2 test: A vs C (different code path)
    a_vs_c_pref = common_prefix_chars(a.seed0_text, c.seed0_text)
    a_vs_c_lev = levenshtein(a.seed0_text, c.seed0_text)
    a_vs_c_eq = a.seed0_text == c.seed0_text

    # Verdict.
    if a.err or b.err or c.err:
        verdict = "ERROR"
    elif not h1:
        verdict = "H1-RACE"
    elif h1 and not a_vs_c_eq:
        verdict = "H2-PATH-DIVERGENCE"
    else:
        verdict = "NO-DIVERGENCE"

    # Cert.
    CERT.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# 35B-A3B-Q4_0 / pp2tp2 multi-slot divergence — reproduction")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Model:** {MODEL_ID} on {TOPO['devices']} ({TOPO['mesh_mode']})")
    lines.append(f"- **Slots:** {SLOTS} | **Concurrent:** {N_CONCURRENT} | "
                 f"**ctx_cap:** {CTX_CAP} | **tg_len:** {TG_LEN} | seed=0")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if verdict == "H1-RACE":
        lines.append("- **H1-RACE confirmed** — same env, two boots produce "
                     "different text. There is a true within-env race in "
                     "the multi-slot decode path.")
    elif verdict == "H2-PATH-DIVERGENCE":
        lines.append("- **H2-PATH-DIVERGENCE confirmed** — same env is "
                     "deterministic across boots. The two GDN code paths "
                     "(batched vs per-token) produce different output text. "
                     "Most likely FP-reduction-order divergence amplified "
                     "by MoE routing argmax sensitivity.")
    elif verdict == "NO-DIVERGENCE":
        lines.append("- **NO-DIVERGENCE** — could not reproduce. Possible "
                     "causes: rig state changed, harness cooldown affected "
                     "FP precision, or the original divergence was an "
                     "intermittent race.")
    else:
        lines.append(f"- **{verdict}** — see Errors below.")
    lines.append("")
    lines.append("## H1 test (same env, two boots)")
    lines.append("")
    lines.append("| run | seed | sha256[:16] | err |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A baseline v1 | 0 | `{a.seed0_sha}` | {a.err or ''} |")
    lines.append(f"| A baseline v1 | 1 | `{a.seed1_sha}` | {a.err or ''} |")
    lines.append(f"| B baseline v2 | 0 | `{b.seed0_sha}` | {b.err or ''} |")
    lines.append(f"| B baseline v2 | 1 | `{b.seed1_sha}` | {b.err or ''} |")
    lines.append("")
    lines.append(f"- **A vs B (seed 0) text equal:** "
                 f"`{a.seed0_text == b.seed0_text}`")
    lines.append(f"- **A vs B (seed 1) text equal:** "
                 f"`{a.seed1_text == b.seed1_text}`")
    lines.append("")
    lines.append("## H2 test (default batched-GDN vs per-token GDN)")
    lines.append("")
    lines.append("| run | sha256[:16] | err |")
    lines.append("|---|---|---|")
    lines.append(f"| A baseline (batched) | `{a.seed0_sha}` | {a.err or ''} |")
    lines.append(f"| C per-token (=1)     | `{c.seed0_sha}` | {c.err or ''} |")
    lines.append("")
    lines.append(f"- **A vs C text equal:** `{a_vs_c_eq}`")
    lines.append(f"- **A vs C common prefix chars:** {a_vs_c_pref}")
    lines.append(f"- **A vs C Levenshtein distance:** {a_vs_c_lev}")
    lines.append(f"- **A length:** {len(a.seed0_text)}")
    lines.append(f"- **C length:** {len(c.seed0_text)}")
    lines.append("")
    if not a_vs_c_eq:
        lines.append("### First-divergence window")
        lines.append("")
        lines.append("```")
        lines.append(fmt_diff_window(a.seed0_text, c.seed0_text))
        lines.append("```")
        lines.append("")
    lines.append("## Captured texts")
    lines.append("")
    for r in runs:
        lines.append(f"### {r.label}  env_extras={r.env_extras}")
        lines.append("")
        lines.append("**seed=0:**")
        lines.append("")
        lines.append("```")
        lines.append(r.seed0_text or "<empty>")
        lines.append("```")
        lines.append("")
        lines.append("**seed=1:**")
        lines.append("")
        lines.append("```")
        lines.append(r.seed1_text or "<empty>")
        lines.append("```")
        lines.append("")
    CERT.write_text("\n".join(lines))
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"wrote {CERT}", flush=True)


if __name__ == "__main__":
    main()
