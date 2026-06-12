#!/usr/bin/env python3
"""Long-context prefill+decode speed sweep.

Boots one `flambeau serve` per context length (so KV is sized to that length
and per-length VRAM feasibility is exercised), fires a single padded prompt,
and records prefill throughput (prompt_tokens / TTFT) and decode throughput
(generated_tokens / decode_time) via the streaming OpenAI API.

Usage:
  python3 scripts/bench/long_ctx_sweep.py \
    --model /artefact/models/Qwen3.6-27B-Q8_0.gguf --kv q8 \
    --ctx 4096 16384 32768 65536 131072 --decode-tokens 48
"""
import argparse, json, os, signal, subprocess, sys, time
import urllib.request

BIN = "/artefact/flambeau/target/release/flambeau"
FILLER = (
    "In the study of large language model inference, the throughput of both the "
    "prefill and the decode phase depends on memory bandwidth, kernel occupancy, "
    "and the size of the key-value cache that must be read at every decode step. "
)


def post_stream(port, prompt, max_tokens, seed):
    """Stream a chat completion; return prefill_ms, decode_tokens, decode_ms, prompt_tokens."""
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"content-type": "application/json"})
    t_send = time.perf_counter()
    t_first = None
    t_last = None
    n_chunks = 0
    prompt_tokens = completion_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            ch = obj.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                t_last = time.perf_counter()
                n_chunks += 1
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens")
                completion_tokens = obj["usage"].get("completion_tokens")
    if t_first is None:
        return None
    prefill_ms = (t_first - t_send) * 1000.0
    decode_ms = (t_last - t_first) * 1000.0
    dec_tok = (completion_tokens or n_chunks) - 1  # first token is prefill's product
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tokens": max(dec_tok, 1),
        "prompt_tokens": prompt_tokens,
    }


def make_prompt(target_tokens):
    # ~ 0.28 tokens/char for this filler; overshoot then the server reports actual.
    reps = max(1, int(target_tokens / 55))  # ~55 tokens per filler block
    return (FILLER * reps) + "\n\nReply with a single short sentence."


def wait_health(port, proc, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def boot(model, ctx_cap, kv, port, devices, mesh, pp, tp):
    env = dict(os.environ)
    env["FLAMBEAU_KV"] = kv
    args = [BIN, "serve", "--model", model, "--port", str(port),
            "--devices", devices, "--inflight-slots", "1",
            "--ctx-cap", str(ctx_cap), "--kv", kv]
    if mesh:
        args += ["--mesh-mode", mesh, "--pp-size", str(pp), "--tp-size", str(tp)]
    log = open(f"/tmp/lctx_serve_{ctx_cap}.log", "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc


def shutdown(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", default="q8")
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072])
    ap.add_argument("--decode-tokens", type=int, default=48)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--devices", default="hip:0,2,1,3")
    ap.add_argument("--mesh-mode", default="pp+tp")
    ap.add_argument("--pp-size", type=int, default=2)
    ap.add_argument("--tp-size", type=int, default=2)
    a = ap.parse_args()

    print(f"# model={a.model} kv={a.kv} topo={a.mesh_mode} pp{a.pp_size}tp{a.tp_size} dev={a.devices}")
    print(f"{'ctx_req':>8} {'ptok':>7} {'prefill_tok/s':>13} {'decode_tok/s':>12} {'prefill_s':>9} {'status'}")
    rows = []
    for ctx in a.ctx:
        cap = ctx + 512
        proc = boot(a.model, cap, a.kv, a.port, a.devices, a.mesh_mode, a.pp_size, a.tp_size)
        if not wait_health(a.port, proc):
            print(f"{ctx:>8} {'-':>7} {'-':>13} {'-':>12} {'-':>9} BOOT_FAILED(see /tmp/lctx_serve_{cap}.log)")
            shutdown(proc)
            continue
        prompt = make_prompt(ctx)
        try:
            post_stream(a.port, prompt, 4, seed=1)  # warmup
            r = post_stream(a.port, prompt, a.decode_tokens, seed=2)
        except Exception as e:
            print(f"{ctx:>8} {'-':>7} {'-':>13} {'-':>12} {'-':>9} REQ_ERR:{e}")
            shutdown(proc)
            continue
        shutdown(proc)
        if r is None:
            print(f"{ctx:>8} {'-':>7} {'-':>13} {'-':>12} {'-':>9} NO_OUTPUT")
            continue
        pt = r["prompt_tokens"] or ctx
        prefill_s = r["prefill_ms"] / 1000.0
        prefill_tps = pt / prefill_s if prefill_s > 0 else 0
        decode_tps = r["decode_tokens"] / (r["decode_ms"] / 1000.0) if r["decode_ms"] > 0 else 0
        rows.append((ctx, pt, prefill_tps, decode_tps, prefill_s))
        print(f"{ctx:>8} {pt:>7} {prefill_tps:>13.1f} {decode_tps:>12.2f} {prefill_s:>9.2f} ok", flush=True)

    print("\n# JSON")
    print(json.dumps([
        {"ctx": c, "prompt_tokens": pt, "prefill_tok_s": round(ptps, 1),
         "decode_tok_s": round(dtps, 2), "prefill_s": round(ps, 2)}
        for (c, pt, ptps, dtps, ps) in rows], indent=2))


if __name__ == "__main__":
    main()
