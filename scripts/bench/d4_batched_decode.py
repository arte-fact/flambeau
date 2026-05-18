#!/usr/bin/env python3
"""D4 — batched-decode throughput bench for the v2 shared-Session stack.

Boots flambeau serve with `--inflight-slots N_MAX --decode-batch-window-us W`,
then fires N concurrent /v1/chat/completions requests for each N in
N_RUN. Each request asks for `K` tokens of greedy generation. Reports:
- wall (s) for N concurrent requests to all complete
- per-stream tokens/s
- aggregate tokens/s
- batched speedup vs N=1 baseline

Run:
    python3 scripts/bench/d4_batched_decode.py \\
        --model /artefact/models/Qwen3.5-9B-Q4_1.gguf \\
        --topology pp --devices 0,2 --tokens 64 --runs 3
"""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "flambeau"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_post_json(url: str, payload: dict, timeout: float) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    t1 = time.perf_counter()
    return (t1 - t0, data)


def wait_health(port: int, deadline_s: float = 90.0) -> None:
    t_end = time.time() + deadline_s
    while time.time() < t_end:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            ) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server on :{port} never became healthy")


def make_request(prompt_idx: int, n_tokens: int) -> dict:
    # Identical prompt across streams. Output may match but that's fine —
    # the bench measures wall and tok/s, not output diversity. Force
    # length-stop (not EOS) by feeding a list-continuation prompt the
    # model extends predictably for many tokens.
    _ = prompt_idx
    return {
        "model": "bench",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Continue the list with the next 70 short lowercase nouns, "
                    "one per line, no commentary: cat, dog, tree, river, bird, "
                    "fish, mountain, star, moon, sun,"
                ),
            }
        ],
        "max_tokens": n_tokens,
        "temperature": 0,
    }


def run_concurrent(
    port: int, n: int, n_tokens: int, timeout: float
) -> tuple[float, list[int]]:
    payloads = [make_request(i, n_tokens) for i in range(n)]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        futs = [ex.submit(http_post_json, url, p, timeout) for p in payloads]
        per_stream = []
        tokens_emitted = []
        for f in as_completed(futs):
            elapsed, body = f.result()
            per_stream.append(elapsed)
            tok = body.get("usage", {}).get("completion_tokens", 0)
            tokens_emitted.append(int(tok))
        wall = time.perf_counter() - t0
    return wall, tokens_emitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topology", default="pp", choices=["pp", "tp", "pp+tp"])
    ap.add_argument("--devices", default="0")
    ap.add_argument("--tp-size", type=int, default=0)
    ap.add_argument("--pp-size", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=4, help="--inflight-slots")
    ap.add_argument(
        "--n-run",
        default="1,2,4",
        help="comma-separated N values (must be <= n-max)",
    )
    ap.add_argument("--tokens", type=int, default=64, help="completion tokens per stream")
    ap.add_argument("--runs", type=int, default=3, help="measurement runs per N")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--decode-batch-window-us", type=int, default=1500)
    ap.add_argument("--ctx-cap", type=int, default=4096)
    ap.add_argument("--out", default=None, help="optional path to write cert markdown")
    args = ap.parse_args()

    n_run = [int(x) for x in args.n_run.split(",") if x.strip()]
    for n in n_run:
        if n > args.n_max:
            print(f"--n-run value {n} > --n-max {args.n_max}", file=sys.stderr)
            return 2

    if not BIN.exists():
        print(f"binary missing: {BIN} (run cargo build --release)", file=sys.stderr)
        return 2

    port = free_port()
    cmd = [
        str(BIN),
        "serve",
        "--model", args.model,
        "--devices", args.devices,
        "--mesh-mode", args.topology,
        "--port", str(port),
        "--inflight-slots", str(args.n_max),
        "--decode-batch-window-us", str(args.decode_batch_window_us),
        "--ctx-cap", str(args.ctx_cap),
    ]
    if args.topology == "tp":
        ndev = len([d for d in args.devices.split(",") if d.strip()])
        cmd += ["--tp-size", str(args.tp_size or ndev)]
    elif args.topology == "pp+tp":
        cmd += ["--pp-size", str(args.pp_size), "--tp-size", str(args.tp_size)]

    env = os.environ.copy()
    env["FLAMBEAU_V2"] = "1"
    log_path = Path(f"/tmp/d4_bench_{port}.log")
    log_f = open(log_path, "w")
    print(f"server log: {log_path}", file=sys.stderr)
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    try:
        wait_health(port)
        print(f"server up :{port} (pid {proc.pid})", file=sys.stderr)

        results: list[dict] = []
        for n in n_run:
            timeout = max(60.0, args.tokens * 0.5 * n)
            print(f"\n=== N={n} ===", file=sys.stderr)
            for _ in range(args.warmup):
                run_concurrent(port, n, args.tokens, timeout)
            walls = []
            tok_totals = []
            for r in range(args.runs):
                wall, toks = run_concurrent(port, n, args.tokens, timeout)
                walls.append(wall)
                tok_totals.append(sum(toks))
                print(
                    f"  run {r}: wall={wall:.3f}s tok_total={sum(toks)} "
                    f"per_stream={toks}",
                    file=sys.stderr,
                )
            best_wall = min(walls)
            tok_at_best = tok_totals[walls.index(best_wall)]
            agg_tps = tok_at_best / best_wall
            per_stream_tps = (tok_at_best / n) / best_wall
            results.append({
                "n": n,
                "wall_best_s": best_wall,
                "wall_walls": walls,
                "tok_total_at_best": tok_at_best,
                "per_stream_tps": per_stream_tps,
                "agg_tps": agg_tps,
            })

        # Render summary.
        baseline = next((r for r in results if r["n"] == 1), None)
        print("\n--- D4 batched-decode summary ---")
        print(f"model={Path(args.model).name} topology={args.topology} "
              f"devices={args.devices} tokens={args.tokens}")
        hdr = f"{'N':>3}  {'wall(s)':>8}  {'per-stream':>12}  {'aggregate':>12}  {'speedup':>8}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            speedup = (r["agg_tps"] / baseline["agg_tps"]) if baseline else 0.0
            print(
                f"{r['n']:>3}  {r['wall_best_s']:>8.3f}  "
                f"{r['per_stream_tps']:>10.2f} t/s  "
                f"{r['agg_tps']:>10.2f} t/s  "
                f"{speedup:>7.2f}×"
            )

        if args.out:
            write_cert(
                Path(args.out),
                model=Path(args.model).name,
                topology=args.topology,
                devices=args.devices,
                tokens=args.tokens,
                runs=args.runs,
                window_us=args.decode_batch_window_us,
                results=results,
            )
            print(f"\ncert: {args.out}", file=sys.stderr)
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        log_f.close()


def write_cert(
    out: Path,
    *,
    model: str,
    topology: str,
    devices: str,
    tokens: int,
    runs: int,
    window_us: int,
    results: list[dict],
) -> None:
    baseline = next((r for r in results if r["n"] == 1), None)
    lines = []
    lines.append(f"# D4 — batched-decode throughput ({model}, {topology})\n")
    lines.append("## Setup")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Topology: `{topology}`  Devices: `{devices}`")
    lines.append(f"- Generation: `{tokens}` tokens per stream, greedy (temp=0)")
    lines.append(f"- Runs per N: {runs} (best-of)")
    lines.append(f"- Decode batch window: {window_us} µs")
    lines.append(f"- Stack: v2 shared `Session<A>` with `max_slots = N_max`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("|   N | wall (s) | per-stream tok/s | aggregate tok/s | speedup vs N=1 |")
    lines.append("|----:|---------:|-----------------:|----------------:|---------------:|")
    for r in results:
        speedup = (r["agg_tps"] / baseline["agg_tps"]) if baseline else 0.0
        lines.append(
            f"| {r['n']:>3} | {r['wall_best_s']:>8.3f} | "
            f"{r['per_stream_tps']:>14.2f} | {r['agg_tps']:>13.2f} | {speedup:>10.2f}× |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Aggregate tok/s is total generated tokens (N × tokens) divided by "
        "wall time for ALL N concurrent requests to complete. Per-stream "
        "tok/s is the slowest stream's rate (= aggregate / N when streams "
        "finish together, which they do under greedy + uniform request shape)."
    )
    lines.append("")
    lines.append(
        "Speedup vs N=1 indicates how well the shared-session batched-decode "
        "scheduler hides per-stream serialisation. Perfect batching is N×; "
        "real-world hits a per-slot ceiling (per-slot KV-append + attention "
        "loops, GDN state-step loop for hybrid archs)."
    )
    lines.append("")
    lines.append("## Architecture context")
    lines.append("")
    lines.append(
        "This cert measures the D3-A shared-session pool: one `Session<A>` "
        "with `max_slots = inflight_slots`, N `V2Conv` ticket handles, and "
        "`V2Model::forward_decode_batched` actually batching through one "
        "Session forward. Pre-D3-A the v2 pool held N independent sessions "
        "(N× weight copy, no batching)."
    )
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
