#!/usr/bin/env python3
"""Like-for-like flambeau vs llama.cpp on one GGUF, over HTTP.

Both engines get the same GGUF, the same GPUs, the same KV dtype and the
same exact-length prompts (built by tokenize/detokenize round-trip, then
checked against the server's own token count). Each request carries a
unique nonce so no prefix cache can serve it.

Per (arm, prompt length N):
  prefill t/s, decode t/s at depth N, from each server's own clocks:
    llama.cpp  response `timings` (prompt_per_second / predicted_per_second)
    flambeau   info logs: N / ttft_ms, (gen - 1) / (total_ms - ttft_ms)
Non-streaming: flambeau's /v1/completions has no SSE.

Usage:
  scripts/bench/qwen38_27b_vs_llamacpp.py \
      --model ~/models/Qwen3.8-27B-UD-Q4_K_M.gguf --arms all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
FLAMBEAU = ROOT / "target/release/flambeau"
LLAMA = HOME / "llama.cpp/build-hip/bin/llama-server"


def arms(model: str, ctx: int, kv: str) -> dict[str, list[str]]:
    lkv = {"q8": "q8_0", "f16": "f16"}[kv]
    fl = [str(FLAMBEAU), "serve", "--model", model, "--ctx-cap", str(ctx),
          "--inflight-slots", "1", "--kv", kv, "--prefill-ubatch", "512"]
    lc = [str(LLAMA), "-m", model, "-ngl", "999", "-c", str(ctx), "-fa", "on",
          "-ctk", lkv, "-ctv", lkv, "-np", "1", "-ub", "512", "-b", "2048"]
    return {
        "flambeau-tp2": fl + ["--devices", "hip:0,1", "--mesh-mode", "tp", "--tp-size", "2"],
        "flambeau-pp2": fl + ["--devices", "hip:0,1", "--mesh-mode", "pp"],
        "llamacpp-layer": lc + ["-sm", "layer", "-ts", "1/1"],
    }


def post(port: int, path: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(port: int, text: str) -> list[int]:
    return post(port, "/tokenize", {"content": text})["tokens"]


def detokenize(port: int, ids: list[int]) -> str:
    return post(port, "/detokenize", {"tokens": ids})["content"]


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 900) -> None:
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server not ready")


def corpus() -> str:
    parts = []
    for p in sorted((ROOT / "crates").rglob("*.rs")):
        parts.append(p.read_text(errors="ignore"))
        if sum(map(len, parts)) > 400_000:
            break
    return "\n".join(parts)


def build_prompt(port: int, base_ids: list[int], n: int, offset: int) -> str:
    nonce = f"// run {uuid.uuid4().hex}\n"
    head = tokenize(port, nonce)
    ids = head + base_ids[offset: offset + n - len(head)]
    text = detokenize(port, ids)
    got = len(tokenize(port, text))
    if got != n:
        ids = ids[: n - (got - n)] if got > n else ids
        text = detokenize(port, ids)
    return text


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def flambeau_timings(log_path: Path) -> tuple[float, float] | None:
    """Last request's (ttft_ms, total_ms) from flambeau's info logs."""
    ttft = total = None
    for line in ANSI.sub("", log_path.read_text(errors="ignore")).splitlines()[-400:]:
        if "server.completion.first_token" in line and (m := re.search(r"ttft_ms=([\d.]+)", line)):
            ttft = float(m.group(1))
        elif "server.completion.finish" in line and (m := re.search(r"total_ms=([\d.]+)", line)):
            total = float(m.group(1))
    return (ttft, total) if ttft is not None and total is not None else None


def measure(port: int, prompt: str, tg: int, log_path: Path) -> tuple[int, int, float, float]:
    """One request → (prompt_tokens, gen_tokens, prefill t/s, decode t/s), server-side clocks."""
    body = {"prompt": prompt, "max_tokens": 1 + tg, "temperature": 0, "cache_prompt": False}
    obj = post(port, "/v1/completions", body, timeout=1800)
    u = obj.get("usage") or {}
    ptok, gen = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    if t := obj.get("timings"):
        return ptok, gen, t["prompt_per_second"], t["predicted_per_second"]
    time.sleep(0.2)
    ft = flambeau_timings(log_path)
    if ft is None:
        raise RuntimeError("no server timings (llama.cpp `timings` / flambeau info logs)")
    ttft_ms, total_ms = ft
    dec = (gen - 1) / ((total_ms - ttft_ms) / 1000) if gen > 1 and total_ms > ttft_ms else float("nan")
    return ptok, gen, ptok / (ttft_ms / 1000), dec


def run_arm(name: str, cmd: list[str], port: int, lengths: list[int], tg: int, reps: int,
            logdir: Path) -> list[dict]:
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    if name.startswith("llamacpp"):
        env["HIP_VISIBLE_DEVICES"] = "0,1"
    log = open(logdir / f"{name}.log", "w")
    full = cmd + ["--port", str(port)]
    print(f"[{name}] {' '.join(full)}", flush=True)
    proc = subprocess.Popen(full, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    rows = []
    try:
        wait_ready(port, proc)
        base = tokenize(port, corpus())
        log_path = logdir / f"{name}.log"
        measure(port, build_prompt(port, base, 256, 0), 16, log_path)
        for n in lengths:
            pp, tgs = [], []
            for rep in range(reps):
                prompt = build_prompt(port, base, n, 1000 * (rep + 1))
                ptok, gen, p, d = measure(port, prompt, tg, log_path)
                if ptok and abs(ptok - n) > 1:
                    print(f"  warn: server counted {ptok} prompt tokens, expected {n}", flush=True)
                pp.append(p)
                if d == d:
                    tgs.append(d)
                print(f"  {name} N={ptok} rep={rep} pp={p:.1f} t/s gen={gen} tg={d:.2f} t/s", flush=True)
            rows.append({"arm": name, "n": n, "pp": statistics.median(pp),
                         "tg": statistics.median(tgs) if tgs else None})
    except Exception as e:
        print(f"  {name} FAILED: {e} (see {logdir / (name + '.log')})", flush=True)
        rows.append({"arm": name, "n": None, "pp": None, "tg": None, "err": str(e)})
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=30)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        time.sleep(5)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--lengths", default="512,4096,16384")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kv", default="q8", choices=["q8", "f16"])
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--out", default=str(ROOT / "scripts/bench/logs"))
    a = ap.parse_args()

    table = arms(a.model, a.ctx, a.kv)
    names = list(table) if a.arms == "all" else a.arms.split(",")
    lengths = [int(x) for x in a.lengths.split(",")]
    logdir = Path(a.out) / f"qwen38_vs_llamacpp_{time.strftime('%Y%m%d_%H%M%S')}"
    logdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        rows += run_arm(name, table[name], a.port, lengths, a.tg, a.reps, logdir)

    hdr = f"model={Path(a.model).name} kv={a.kv} ctx={a.ctx} tg={a.tg} reps={a.reps} (median)"
    lines = [hdr, f"{'arm':<16}{'N':>7}{'prefill t/s':>13}{'decode t/s':>12}"]
    for r in rows:
        if r.get("err"):
            lines.append(f"{r['arm']:<16}  FAILED: {r['err']}")
            continue
        tg = f"{r['tg']:.2f}" if r["tg"] else "-"
        lines.append(f"{r['arm']:<16}{r['n']:>7}{r['pp']:>13.1f}{tg:>12}")
    report = "\n".join(lines)
    print("\n" + report)
    (logdir / "summary.txt").write_text(report + "\n")
    (logdir / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"\nlogs: {logdir}")


if __name__ == "__main__":
    sys.exit(main())
