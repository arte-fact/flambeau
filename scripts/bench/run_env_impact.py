#!/usr/bin/env python3
"""
Env-var impact audit harness.

Loads bench/env_impact.toml, iterates every (var × test_value × model ×
topology) cell in the spec, boots a flambeau server with the env set,
captures one chat call (greedy, deterministic seed) → first-token text +
prefill_ms + decode_ms, computes delta vs the default cell, emits one
cert markdown per var under certs/env_impact/.

Reuses run_matrix.py primitives (long-prompt template, server lifecycle,
cooldown, stream_one_call) so this harness only adds the env-sweep loop +
cert writer. ~350 LOC.

Usage:
    python3 scripts/bench/run_env_impact.py
    python3 scripts/bench/run_env_impact.py --vars FLAMBEAU_DECODE_GRAPH
    python3 scripts/bench/run_env_impact.py --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# Stdlib tomllib (Python 3.11+); fall back to `toml` package if needed.
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import toml as _toml  # type: ignore[import-not-found]
    class tomllib:  # type: ignore[no-redef]
        @staticmethod
        def loads(s: str) -> dict:
            return _toml.loads(s)

# Reuse run_matrix.py primitives.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "bench" / "env_impact.toml"
PROFILE_PATH = ROOT / "bench" / "profiles" / "optimized.toml"
CERT_DIR = ROOT / "certs" / "env_impact"


def load_profile_env(path: Path) -> dict[str, str]:
    """Load the [env] section of a profile TOML as a flat dict.
    Missing file → empty dict (caller falls back to no-profile baseline)."""
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text())
    env_block = raw.get("env", {})
    return {str(k): str(v) for k, v in env_block.items()}


# ----------------------------------------------------------------------
# Spec data

@dataclasses.dataclass(frozen=True)
class VarSpec:
    name: str
    var_class: str
    description: str
    default_state: str
    test_values: list[str]
    models: list[str]
    topologies: list[str]
    concurrencies: list[int]    # N concurrent calls per measure-batch; default [1]
    metrics: list[str]
    preconditions: dict[str, str]
    notes: str
    known_result: Optional[str]

    @property
    def runnable(self) -> bool:
        return (bool(self.test_values) and bool(self.models)
                and bool(self.topologies) and bool(self.concurrencies))


@dataclasses.dataclass(frozen=True)
class SpecMeta:
    runs_per_cell: int
    warmup_runs: int
    noise_floor_pct: float
    prompt_len: int
    tg_len: int
    seed: int
    correctness_check: str
    model_paths: dict[str, str]
    topologies: dict[str, dict]
    ctx_cap: int                # default per-cell context cap (overridable)
    sample_vram: bool           # if True, harness records peak VRAM per cell


def load_spec(path: Path) -> tuple[SpecMeta, list[VarSpec]]:
    raw = tomllib.loads(path.read_text())
    m = raw["meta"]
    meta = SpecMeta(
        runs_per_cell=m["runs_per_cell"],
        warmup_runs=m["warmup_runs"],
        noise_floor_pct=m["noise_floor_pct"],
        prompt_len=m["prompt_len"],
        tg_len=m["tg_len"],
        seed=m["seed"],
        correctness_check=m["correctness_check"],
        model_paths=dict(m["models"]),
        topologies=dict(m["topologies"]),
        ctx_cap=int(m.get("ctx_cap", 4096)),
        sample_vram=bool(m.get("sample_vram", False)),
    )
    vars_: list[VarSpec] = []
    for name, body in raw.get("var", {}).items():
        vars_.append(VarSpec(
            name="FLAMBEAU_" + name if not name.startswith("FLAMBEAU_") else name,
            var_class=body.get("class", "?"),
            description=body.get("description", ""),
            default_state=body.get("default_state", "unset"),
            test_values=list(body.get("test_values", [])),
            models=list(body.get("models", [])),
            topologies=list(body.get("topologies", [])),
            concurrencies=[int(n) for n in body.get("concurrencies", [1])],
            metrics=list(body.get("metrics", [])),
            preconditions=dict(body.get("preconditions", {})),
            notes=body.get("notes", "").strip(),
            known_result=body.get("known_result"),
        ))
    return meta, vars_


# ----------------------------------------------------------------------
# Server lifecycle with arbitrary env overrides

def boot_with_env(model_id: str, model_path: str, topo: dict,
                  env_extras: dict[str, str], slots: int,
                  log_path: Path, ctx_cap: int = 4096,
                  profile_env: dict[str, str] | None = None) -> tuple:
    """
    Like rm.boot_server but takes arbitrary env overrides on top of a
    profile baseline. Returns (proc, port).

    Layering (lower → higher precedence):
      1. inherited os.environ
      2. ctx_cap + slots (computed per-cell from spec preconditions /
         var slot requirements)
      3. profile_env (the [env] block from bench/profiles/<name>.toml,
         encoding the best-known production baseline)
      4. env_extras (per-cell var-under-test value + spec preconditions)

    `env_extras` semantics: each key is set in the child env unless the
    value is the literal `"__UNSET__"`, which removes it.
    """
    import subprocess, signal, socket
    port_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_sock.bind(("127.0.0.1", 0))
    port = port_sock.getsockname()[1]
    port_sock.close()

    env = os.environ.copy()
    # Per-cell-derived knobs that the profile shouldn't dictate
    # (these depend on the model and the spec's slot count).
    env["FLAMBEAU_CTX_CAP"] = str(ctx_cap)
    env["FLAMBEAU_INFLIGHT_SLOTS"] = str(slots)

    # Profile baseline (best-known config for this rig).
    if profile_env:
        for k, v in profile_env.items():
            # Slot count is per-cell — never let the profile override it.
            if k == "FLAMBEAU_INFLIGHT_SLOTS":
                continue
            env[k] = v
    # Always RUST_LOG=info during a sweep so the cell logs capture
    # per-call telemetry, regardless of the profile's serving choice.
    env["RUST_LOG"] = "info"

    # Per-cell env extras (the var under test + preconditions).
    # "__UNSET__" sentinel removes the key.
    for k, v in env_extras.items():
        if v == "__UNSET__":
            env.pop(k, None)
        else:
            env[k] = str(v)

    args = [
        str(rm.BIN), "serve",
        "--model", model_path,
        "--devices", topo["devices"],
        "--mesh-mode", topo["mesh_mode"],
        "--port", str(port),
    ]
    if topo["mesh_mode"] == "pp+tp":
        args += ["--pp-size", str(topo["pp_size"]),
                 "--tp-size", str(topo["tp_size"])]
    elif topo["mesh_mode"] == "tp":
        args += ["--tp-size", str(topo.get("tp_size", topo.get("world", 2)))]

    log = open(log_path, "w")
    proc = subprocess.Popen(
        args, env=env, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, port


# ----------------------------------------------------------------------
# Single measurement

@dataclasses.dataclass
class CellResult:
    var_value: str                  # "unset", "1", "on", etc.
    model: str
    topo: str
    concurrency: int                # N concurrent calls per batch (1 = single-stream)
    n_runs: int                     # number of successful measure-batches
    prefill_ms_median: float        # median across all calls in all batches
    decode_ms_median: float
    decode_tps_median: float        # at N=1: per-stream tps; at N>1: aggregate (sum_ct / max_dm)
    completion_tokens: int
    text_sha256: str                # hash of generated content text (seed=0 call only)
    err: Optional[str] = None
    boot_secs: Optional[float] = None
    vram_peak_gb: float = 0.0       # peak total-used VRAM (sum across GPUs) during measure window
    ttft_p99_ms: float = 0.0        # 99th-percentile time-to-first-token across all calls (only meaningful at N>1)


# ROCm-SMI binary location (matches scripts/bench/run_matrix.py).
ROCM_SMI = "/opt/rocm-7.1.1/core-7.13/bin/rocm-smi"
_VRAM_USED_RE = None  # lazy-init


def _vram_total_used_gb() -> float:
    """Sum of `VRAM Total Used Memory (B)` across every GPU, in GB.
    Returns 0.0 on rocm-smi failure (caller treats as missing data,
    not as zero-VRAM)."""
    global _VRAM_USED_RE
    if _VRAM_USED_RE is None:
        import re as _re
        _VRAM_USED_RE = _re.compile(
            r"GPU\[\d+\]\s*: VRAM Total Used Memory \(B\):\s*(\d+)")
    import subprocess
    try:
        out = subprocess.run([ROCM_SMI, "--showmeminfo", "vram"],
                             capture_output=True, text=True,
                             timeout=3.0).stdout
    except Exception:
        return 0.0
    bytes_total = sum(int(m.group(1)) for m in _VRAM_USED_RE.finditer(out))
    return bytes_total / (1024.0 ** 3)


def _vram_sampler_loop(stop_event, peak_ref: list, poll_s: float = 1.0) -> None:
    """Background poller: every poll_s seconds while stop_event isn't set,
    sample current VRAM and update peak_ref[0] in place."""
    while not stop_event.is_set():
        used = _vram_total_used_gb()
        if used > peak_ref[0]:
            peak_ref[0] = used
        if stop_event.wait(poll_s):
            return


def stream_one_call_capture_text(port: int, prompt_pad: str,
                                 max_tokens: int, seed: int) -> dict:
    """Like rm.stream_one_call but accumulates the generated `content`
    chunks so the cert can hash the actual output text instead of a
    proxy. Same timing semantics as the original."""
    import httpx, json as _json
    body = {
        "model": "anything",
        "messages": [
            {"role": "system", "content": rm.LONG_PROMPT_SYSTEM},
            {"role": "user",
             "content": rm.LONG_PROMPT_USER_TEMPLATE.format(pad=prompt_pad)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,           # greedy → deterministic across same env
        "stream": True,
        "seed": seed,
        "stream_options": {"include_usage": True},
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t_send = time.perf_counter()
    t_first = None
    t_last = None
    content_parts: list[str] = []
    completion_tokens = None
    prompt_tokens = None
    err = None
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as cli:
            with cli.stream("POST", url, json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        obj = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue
                    if "choices" in obj and obj["choices"]:
                        delta = obj["choices"][0].get("delta", {})
                        if delta.get("content"):
                            if t_first is None:
                                t_first = time.perf_counter()
                            content_parts.append(delta["content"])
                        t_last = time.perf_counter()
                    if "usage" in obj and obj["usage"]:
                        u = obj["usage"]
                        completion_tokens = u.get("completion_tokens")
                        prompt_tokens = u.get("prompt_tokens")
    except Exception as e:
        err = repr(e)

    if t_first is None:
        return {"err": err or "no first chunk",
                "wall_ms": (time.perf_counter() - t_send) * 1000}

    text = "".join(content_parts)
    return {
        "err": err,
        "prefill_ms": (t_first - t_send) * 1000.0,
        "decode_ms": ((t_last or t_first) - t_first) * 1000.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _run_concurrent_batch(port: int, n: int, max_tokens: int,
                          base_seed: int,
                          sample_vram: bool = False) -> dict:
    """Spawn N threads each calling stream_one_call_capture_text with
    a distinct seed (so generations diverge but prompt is identical),
    return per-stream + aggregate metrics. Mirrors run_matrix.run_cell
    shape but uses our text-capturing helper.

    If sample_vram=True, also runs a background rocm-smi poller during
    the batch and reports peak VRAM (sum across GPUs, GB).
    """
    import threading
    results: list[Optional[dict]] = [None] * n

    def worker(i: int) -> None:
        results[i] = stream_one_call_capture_text(
            port, "x" * 64, max_tokens=max_tokens, seed=base_seed + i)

    # VRAM sampler (optional).
    peak_gb = [0.0]
    stop_event = threading.Event()
    sampler: Optional[threading.Thread] = None
    if sample_vram:
        sampler = threading.Thread(
            target=_vram_sampler_loop,
            args=(stop_event, peak_gb),
            daemon=True,
        )
        sampler.start()

    threads: list[threading.Thread] = []
    for i in range(n):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=900.0)

    if sampler is not None:
        stop_event.set()
        sampler.join(timeout=5.0)

    ok = [r for r in results if r and not r.get("err")
          and r.get("completion_tokens")]
    if not ok:
        first_err = next((r.get("err") for r in results
                          if r and r.get("err")), "no successful streams")
        return {"err": first_err, "ok": [], "vram_peak_gb": peak_gb[0]}

    prefill_ms_list = [r["prefill_ms"] for r in ok]
    decode_ms_list = [r["decode_ms"] for r in ok]
    completion_total = sum(r["completion_tokens"] for r in ok)
    # Aggregate tps: server's POV — total tokens served per max-decode-window.
    max_decode_ms = max(decode_ms_list) if decode_ms_list else 0.0
    aggregate_tps = (completion_total / (max_decode_ms / 1000.0)
                     if max_decode_ms > 0 else 0.0)

    # TTFT p99: meaningful only at N>1 (at N=1 it equals prefill_ms_median).
    sorted_ttfts = sorted(prefill_ms_list)
    ttft_p99 = (sorted_ttfts[int(0.99 * len(sorted_ttfts)) - 1]
                if len(sorted_ttfts) >= 100 else max(sorted_ttfts))

    # The seed=base_seed (i=0) result is the one we hash for correctness:
    # at fixed seed + same env, the first stream is reproducible.
    seed0 = ok[0] if ok else None
    return {
        "err": None,
        "ok_count": len(ok),
        "fail_count": len(results) - len(ok),
        "prefill_ms_median": statistics.median(prefill_ms_list),
        "decode_ms_median": statistics.median(decode_ms_list),
        "aggregate_tps": aggregate_tps,
        "completion_tokens_total": completion_total,
        "text_sha256": (seed0 or {}).get("text_sha256", ""),
        "vram_peak_gb": peak_gb[0],
        "ttft_p99_ms": ttft_p99,
    }


def measure_cell(meta: SpecMeta, var: VarSpec, value: str,
                 model_id: str, topo_id: str, n: int = 1,
                 profile_env: dict[str, str] | None = None,
                 cooldown_threshold_c: float = 70.0,
                 cooldown_max_wait_s: float = 240.0) -> CellResult:
    """Boot server with the requested env, run warmup + N measured calls,
    compute median metrics, hash first-token text for correctness check."""
    model_path = meta.model_paths[model_id]
    topo = meta.topologies[topo_id]

    # Build env for this cell.
    env_extras: dict[str, str] = {}
    if value == "unset":
        env_extras[var.name] = "__UNSET__"
    else:
        env_extras[var.name] = value
    # Apply preconditions, expanding "$N" → current concurrency value.
    # Lets a spec write `FLAMBEAU_INFLIGHT_SLOTS = "$N"` so slots track
    # the workload size automatically across the concurrency sweep.
    for k, v in var.preconditions.items():
        env_extras[k] = str(v).replace("$N", str(n))

    # Slot count: spec precondition wins; else profile; else 1.
    slots = int(env_extras.get(
        "FLAMBEAU_INFLIGHT_SLOTS",
        (profile_env or {}).get("FLAMBEAU_INFLIGHT_SLOTS", "1"),
    ))

    log_dir = ROOT / "scripts" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"env_impact__{var.name}__{model_id}__{topo_id}__{value}__n{n}.log"
    log_path = log_dir / log_name

    print(f"   boot {var.name}={value} on {model_id}/{topo_id} "
          f"(N={n}, slots={slots}, ctx_cap={meta.ctx_cap})...", flush=True)
    boot_t0 = time.time()
    proc, port = boot_with_env(
        model_id, model_path, topo, env_extras,
        slots=slots,
        log_path=log_path,
        ctx_cap=meta.ctx_cap,
        profile_env=profile_env,
    )
    ready = rm.wait_ready(port, timeout_s=600.0, model_id=model_id)
    boot_secs = time.time() - boot_t0
    if not ready:
        rm.kill_server(proc)
        return CellResult(
            var_value=value, model=model_id, topo=topo_id, concurrency=n,
            n_runs=0, prefill_ms_median=0.0, decode_ms_median=0.0,
            decode_tps_median=0.0, completion_tokens=0, text_sha256="",
            err=f"boot timeout after {boot_secs:.0f}s (log: {log_path.name})",
            boot_secs=boot_secs,
        )

    try:
        # Warmup batches (discard). VRAM sampling deliberately off here —
        # warmup VRAM is artificially low (caches not populated yet).
        for _ in range(meta.warmup_runs):
            _ = _run_concurrent_batch(port, n, meta.tg_len,
                                      base_seed=meta.seed,
                                      sample_vram=False)

        # Measured batches. Each batch is N concurrent calls;
        # we run runs_per_cell batches and median across batches.
        batch_prefill_med, batch_decode_med, batch_aggr_tps = [], [], []
        batch_vram_peak: list[float] = []
        batch_ttft_p99: list[float] = []
        text_hash = ""
        completion_total = 0
        last_err = None
        for run_i in range(meta.runs_per_cell):
            r = _run_concurrent_batch(port, n, meta.tg_len,
                                      base_seed=meta.seed,
                                      sample_vram=meta.sample_vram)
            if r.get("err"):
                last_err = r["err"]
                continue
            batch_prefill_med.append(r["prefill_ms_median"])
            batch_decode_med.append(r["decode_ms_median"])
            batch_aggr_tps.append(r["aggregate_tps"])
            if r.get("vram_peak_gb", 0) > 0:
                batch_vram_peak.append(r["vram_peak_gb"])
            if r.get("ttft_p99_ms", 0) > 0:
                batch_ttft_p99.append(r["ttft_p99_ms"])
            completion_total = r["completion_tokens_total"]
            # Hash the seed-0 stream from the first measured batch only.
            if run_i == 0:
                text_hash = r.get("text_sha256", "")[:16]

        if not batch_aggr_tps:
            return CellResult(
                var_value=value, model=model_id, topo=topo_id, concurrency=n,
                n_runs=0, prefill_ms_median=0.0, decode_ms_median=0.0,
                decode_tps_median=0.0, completion_tokens=0, text_sha256="",
                err=last_err or "no successful batches", boot_secs=boot_secs,
                vram_peak_gb=max(batch_vram_peak) if batch_vram_peak else 0.0,
            )

        return CellResult(
            var_value=value, model=model_id, topo=topo_id, concurrency=n,
            n_runs=len(batch_aggr_tps),
            prefill_ms_median=statistics.median(batch_prefill_med),
            decode_ms_median=statistics.median(batch_decode_med),
            decode_tps_median=statistics.median(batch_aggr_tps),
            completion_tokens=completion_total,
            text_sha256=text_hash,
            boot_secs=boot_secs,
            vram_peak_gb=max(batch_vram_peak) if batch_vram_peak else 0.0,
            ttft_p99_ms=statistics.median(batch_ttft_p99) if batch_ttft_p99 else 0.0,
        )
    finally:
        rm.kill_server(proc)
        time.sleep(2.0)
        if cooldown_threshold_c > 0:
            rm.cooldown_until_safe(cooldown_threshold_c, cooldown_max_wait_s)


# ----------------------------------------------------------------------
# Cert markdown writer

def triage_decision(default: CellResult, test: CellResult,
                    noise_floor_pct: float) -> str:
    """Return one of: 'divergent', 'prefill-collapse', 'null', 'win',
    'loss', 'broken', 'unknown'.

    Severity order:
      broken (test crashes)
        > divergent (text mismatch at greedy / fixed seed)
        > prefill-collapse (prefill ≥ 5× slower; tg can mask this)
        > win/loss/null (tg-tps delta vs noise floor)
    """
    if test.err:
        return "broken"
    if default.err:
        return "unknown"
    # Correctness-first: bit-different output at temp=0/seed=fixed is a bug.
    if (default.text_sha256 and test.text_sha256
            and default.text_sha256 != test.text_sha256):
        return "divergent"
    # Prefill regression. tg64 measures only the decode portion, so a
    # 10× slower prefill can read as "null" on tg alone. Catch the
    # cliff explicitly: ≥5× prefill regression with bit-identical output
    # is a dead-path-slow, regardless of tg.
    d_pf = default.prefill_ms_median
    t_pf = test.prefill_ms_median
    if d_pf > 0 and t_pf > 0 and t_pf >= 5.0 * d_pf:
        return "prefill-collapse"
    # Compare median tg-tps deltas.
    d_tps = default.decode_tps_median
    t_tps = test.decode_tps_median
    if d_tps <= 0 or t_tps <= 0:
        return "unknown"
    delta_pct = (t_tps - d_tps) / d_tps * 100.0
    if abs(delta_pct) < noise_floor_pct:
        return "null"
    return "win" if delta_pct > 0 else "loss"


def correctness_status(default: CellResult, test: CellResult) -> str:
    if default.err or test.err:
        return "n/a"
    if not default.text_sha256 or not test.text_sha256:
        return "missing"
    return "match" if default.text_sha256 == test.text_sha256 else "DIVERGENT"


def write_cert(var: VarSpec, meta: SpecMeta,
               cells: list[CellResult]) -> Path:
    """Group cells by (model, topo, concurrency); for each group emit a
    row table comparing default vs each test value; write a markdown cert.

    Cert filename includes a `_NX` or `_NX,Y` suffix when concurrencies
    other than [1] are involved, so a re-run at N=4 doesn't overwrite
    the prior N=1 cert. Single-stream sweeps keep the bare name.
    """
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    distinct_n = sorted(set(c.concurrency for c in cells))
    if distinct_n == [1]:
        out = CERT_DIR / f"{var.name}.md"
    else:
        n_suffix = ",".join(str(n) for n in distinct_n)
        out = CERT_DIR / f"{var.name}_N{n_suffix}.md"
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append(f"# Env-impact cert: `{var.name}`")
    lines.append("")
    lines.append(f"- **Class:** {var.var_class}")
    lines.append(f"- **Default state:** `{var.default_state}`")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Spec runs/cell:** {meta.runs_per_cell} (+ {meta.warmup_runs} warmup)")
    lines.append("")
    lines.append(f"**Description.** {var.description}")
    lines.append("")
    if var.preconditions:
        lines.append("**Preconditions:** " + ", ".join(
            f"`{k}={v}`" for k, v in var.preconditions.items()))
        lines.append("")

    # Group: (model, topo, n) → {value: CellResult}
    groups: dict[tuple[str, str, int], dict[str, CellResult]] = {}
    for c in cells:
        groups.setdefault((c.model, c.topo, c.concurrency), {})[c.var_value] = c

    # Surface VRAM column if any cell has data.
    has_vram = any(c.vram_peak_gb > 0 for c in cells)

    decisions: list[str] = []
    lines.append("## Measured cells")
    lines.append("")
    # tg-tps column header changes meaning per-N: at N=1 it's per-stream
    # tps, at N>1 it's the aggregate tps (sum_completion / max_decode).
    for (model, topo, n_concurrent), by_value in sorted(groups.items()):
        n_label = f"N={n_concurrent} " if n_concurrent > 1 else ""
        tps_col = ("aggregate tg64 tok/s" if n_concurrent > 1
                   else "tg64 tok/s")
        lines.append(f"### {model} / {topo} / {n_label}".rstrip(" /"))
        lines.append("")
        header = ["value", "n", "prefill ms", "decode ms", tps_col,
                  "Δ vs default", "correct?"]
        if has_vram:
            header.append("VRAM peak GB")
        header.append("err")
        lines.append("| " + " | ".join(header) + " |")
        sep = ["---"] + ["---:"] * 4 + ["---:", "---"]
        if has_vram:
            sep.append("---:")
        sep.append("---")
        lines.append("|" + "|".join(sep) + "|")
        default_key = "unset" if "unset" in by_value else next(iter(by_value))
        default = by_value[default_key]
        for value in var.test_values:
            if value not in by_value:
                continue
            c = by_value[value]
            if value == default_key:
                delta_str = "—"
            else:
                if default.decode_tps_median > 0:
                    delta_pct = ((c.decode_tps_median - default.decode_tps_median)
                                 / default.decode_tps_median * 100.0)
                    delta_str = f"{delta_pct:+.1f}%"
                else:
                    delta_str = "n/a"
                triage = triage_decision(default, c, meta.noise_floor_pct)
                tag = f"N={n_concurrent}" if n_concurrent > 1 else "N=1"
                decisions.append(f"{model}/{topo} [{tag}] `{value}` "
                                 f"→ **{triage}** ({delta_str})")
            correct = correctness_status(default, c) if value != default_key else "—"
            err = c.err or ""
            row = [
                f"`{value}`",
                str(c.n_runs),
                f"{c.prefill_ms_median:.0f}",
                f"{c.decode_ms_median:.0f}",
                f"{c.decode_tps_median:.2f}",
                delta_str,
                correct,
            ]
            if has_vram:
                row.append(f"{c.vram_peak_gb:.2f}" if c.vram_peak_gb > 0 else "—")
            row.append(err)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Triage")
    lines.append("")
    if not decisions:
        lines.append("_No comparison rows produced._")
    else:
        for d in decisions:
            lines.append(f"- {d}")
    lines.append("")

    # Disposition aggregate — count the verdict mix across cells.
    # A single var can land differently on different (model, topology) cells;
    # the aggregate decides the migration path:
    #
    #   - any broken             → HALT-BROKEN
    #   - all win                → CANDIDATE-DEFAULT (flip)
    #   - all loss               → KEEP-DEFAULT      (alternate is dead path)
    #   - all null               → CANDIDATE-DELETE  (gate has no impact)
    #   - mixed win/loss         → SHAPE-DEPENDENT   (dispatch-table candidate)
    #   - mixed null + win/loss  → CONTEXT-DEPENDENT (default mostly fine,
    #                               keep gate or migrate to dispatch row)
    counts = {"win": 0, "loss": 0, "null": 0, "broken": 0,
              "divergent": 0, "prefill-collapse": 0}
    for d in decisions:
        for k in counts:
            if f"**{k}**" in d:
                counts[k] += 1
                break
    n = sum(counts.values())
    lines.append("## Disposition")
    lines.append("")
    if n == 0:
        lines.append("- INSUFFICIENT-DATA")
    elif counts["divergent"] > 0:
        lines.append(f"- **HALT-DIVERGENT** — {counts['divergent']}/{n} cells "
                     "produce different output at greedy/fixed-seed; "
                     "correctness bug, not a perf gate — file before migration")
    elif counts["broken"] > 0:
        lines.append(f"- **HALT-BROKEN** — {counts['broken']}/{n} cells crashed; "
                     "do not migrate before fix")
    elif counts["prefill-collapse"] > 0:
        lines.append(f"- **HALT-DEAD-PATH-SLOW** — {counts['prefill-collapse']}/{n} cells "
                     "show ≥5× prefill regression with unchanged decode; "
                     "alternate is a dead path, delete gate or fix path")
    elif counts["win"] == n:
        lines.append(f"- **CANDIDATE-DEFAULT** — non-default wins on all {n} cells; "
                     "flip the default and delete the gate")
    elif counts["loss"] == n:
        lines.append(f"- **KEEP-DEFAULT** — current default wins on all {n} cells; "
                     "alternate is a dead path, migrate to tracing or delete")
    elif counts["null"] == n:
        lines.append(f"- **CANDIDATE-DELETE** — no measured impact on any of "
                     f"{n} cells; bake default and remove gate")
    elif counts["win"] > 0 and counts["loss"] > 0:
        lines.append(f"- **SHAPE-DEPENDENT** — wins on {counts['win']}, "
                     f"loses on {counts['loss']}, null on {counts['null']} "
                     "(of {n}); migrate to dispatch table per (dtype, shape)".format(n=n))
    else:
        # null + win  OR  null + loss — directional but inconsistent
        direction = "win" if counts["win"] > 0 else "loss"
        lines.append(f"- **CONTEXT-DEPENDENT** — {counts[direction]}/{n} cells "
                     f"{direction}, rest null; "
                     "consider dispatch-table row for the affected cells")
    lines.append("")

    if var.notes:
        lines.append("## Notes from spec")
        lines.append("")
        lines.append(var.notes)
        lines.append("")

    out.write_text("\n".join(lines))
    return out


# ----------------------------------------------------------------------
# Driver

def run_var(meta: SpecMeta, var: VarSpec,
            profile_env: dict[str, str] | None = None,
            dry_run: bool = False) -> Path | None:
    if not var.runnable:
        print(f"=== SKIP {var.name} (no test_values / known_result only) ===",
              flush=True)
        return None
    if dry_run:
        cell_count = (len(var.test_values) * len(var.models)
                      * len(var.topologies) * len(var.concurrencies))
        boot_min = cell_count * 0.5  # rough: ~30s/boot avg
        n_label = ",".join(str(n) for n in var.concurrencies)
        print(f"=== {var.name}: {cell_count} cells (N=[{n_label}]), "
              f"~{boot_min:.0f} min ===",
              flush=True)
        return None

    print(f"\n=== AUDIT {var.name} ({var.var_class}) — "
          f"{var.description} ===", flush=True)
    cells: list[CellResult] = []
    for model in var.models:
        if model not in meta.model_paths:
            print(f"   !! unknown model id: {model} (skip)", flush=True)
            continue
        for topo in var.topologies:
            if topo not in meta.topologies:
                print(f"   !! unknown topo id: {topo} (skip)", flush=True)
                continue
            for n_concurrent in var.concurrencies:
                for value in var.test_values:
                    t0 = time.perf_counter()
                    cell = measure_cell(meta, var, value, model, topo,
                                        n=n_concurrent,
                                        profile_env=profile_env)
                    dt_s = time.perf_counter() - t0
                    label = f"{model}/{topo} N={n_concurrent} {value}"
                    if cell.err:
                        print(f"   ✗ {label}: ERR {cell.err} "
                              f"({dt_s:.0f}s)", flush=True)
                    else:
                        print(f"   ✓ {label}: "
                              f"prefill={cell.prefill_ms_median:.0f}ms "
                              f"tg={cell.decode_tps_median:.1f}t/s "
                              f"({dt_s:.0f}s)", flush=True)
                    cells.append(cell)

    if not cells:
        print(f"   no cells produced — skipping cert write", flush=True)
        return None
    cert = write_cert(var, meta, cells)
    print(f"   wrote {cert.relative_to(ROOT)}", flush=True)
    return cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(SPEC_PATH))
    ap.add_argument("--profile", default=str(PROFILE_PATH),
                    help="optimized-config profile TOML; its [env] block "
                         "becomes the per-cell baseline. Pass empty string "
                         "to disable.")
    ap.add_argument("--vars", help="comma-separated var names to run "
                    "(default: all runnable vars)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, don't run")
    args = ap.parse_args()

    if not rm.BIN.exists():
        print(f"missing {rm.BIN} — build with: cargo build --release "
              f"--features hip_serve -p flambeau-cli", file=sys.stderr)
        sys.exit(2)

    profile_env: dict[str, str] = {}
    if args.profile:
        profile_path = Path(args.profile)
        profile_env = load_profile_env(profile_path)
        if profile_env:
            print(f"loaded profile {profile_path.name}: "
                  f"{len(profile_env)} baseline keys", flush=True)
            for k, v in profile_env.items():
                print(f"  {k}={v}", flush=True)
        else:
            print(f"profile {profile_path} empty/missing — running "
                  f"with no baseline", flush=True)

    meta, vars_ = load_spec(Path(args.spec))
    if args.vars:
        wanted = set(args.vars.split(","))
        # Allow either short ("DECODE_GRAPH") or full ("FLAMBEAU_DECODE_GRAPH") names.
        wanted_full = set()
        for w in wanted:
            wanted_full.add(w if w.startswith("FLAMBEAU_") else "FLAMBEAU_" + w)
        vars_ = [v for v in vars_ if v.name in wanted_full]

    runnable = [v for v in vars_ if v.runnable]
    print(f"loaded spec: {len(vars_)} vars, {len(runnable)} runnable",
          flush=True)
    if args.dry_run:
        for v in runnable:
            run_var(meta, v, profile_env=profile_env, dry_run=True)
        return

    written: list[Path] = []
    for v in runnable:
        try:
            cert = run_var(meta, v, profile_env=profile_env)
            if cert:
                written.append(cert)
        except KeyboardInterrupt:
            print("\n!! interrupted; partial certs preserved", flush=True)
            break
        except Exception as e:
            print(f"!! {v.name}: harness error: {e!r}", flush=True)

    print(f"\nWrote {len(written)} certs:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
