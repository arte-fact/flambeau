# MoE Q3_K bench: 35B-A3B-Q3_K_S (2026-05-12)

Companion to `cert_q3k.md` (dense Q3_K via 9B-Q3_K_S). This cert
exercises the **MoE Q3_K** path — `indexed_moe_mmvq_q3_k` —
on an actual mixture-of-experts model that main physically cannot
serve. Needed split-GGUF support to be staged ahead of it (so the
loader could handle multi-part Q3_K_XL distributions); the bench
model itself is single-file (a re-quantised 35B-A3B-Q4_0).

## Setup

- Same as `cert_q3k.md`: 4× MI50 / gfx906 / ROCm 7.1.1, pp2tp2,
  N ∈ {1, 2}, both batched + no_batched paths, `max_tokens=256`,
  prompt `'x'*1024`. Inflight slots = max(N) = 2.
- main was patched with `af189c0` (qwen35 arch registration);
  cherry-pick reverted post-bench.
- Branch built at commit `6cccd6b` (split GGUF + bench wiring).

## Model

`Qwen3.6-35B-A3B-Q3_K_S` — Qwen3.6-35B-A3B MoE re-quantised from
the Qwen-published Q4_0 to Q3_K_S via:

```
llama-quantize --allow-requantize \
  /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  /artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf \
  Q3_K_S 8
```

- Single file, 15.18 GB / 3.0–3.4 BPW (down from 22 GB / 4.5 BPW).
- 733 tensors, `qwen35moe` arch, 256 experts × top-8 routing,
  expert FFN intermediate 1024, hidden 3072, 48 hybrid layers.
- Quality degraded by requant-from-Q4_0; correctness check passed
  (smoke "What is 2+2?" → "2 + 2 = 4\nSo the answer is 4.").
- Per-tensor dtype mix produced by `llama-quantize Q3_K_S`: most
  FFN expert weights drop to Q3_K, with the usual Q4_K / Q5_K
  preservation on `ffn_down` per the standard Q3_K_S template.
  This exercises `indexed_moe_mmvq_q3_k` for gate/up and the
  `q4_k_r2` / `q5_k` paths for down.

## Results — Branch (`feature/batched-mmvq-decode @ 6cccd6b`)

| path | N | prefill_ms | cum_tps | per_stream_tps | wall (s) |
|---|---|---|---|---|---|
| no_batched | 1 | 24872 | 30.1 | 30.1 | 33.4 |
| no_batched | 2 | 40031 | 16.5 | 19.1 | 63.6 |
| batched    | 1 | 25147 | 30.0 | 30.0 | 33.7 |
| batched    | 2 | 40676 | 16.0 | 19.0 | 64.6 |

Per-token prefill rate (3003-token prompt): **120 tok/s** at N=1.
Decode: **30 tok/s** single-stream, **19 tok/s per-stream** at N=2.

At N=2, batched vs no_batched is a wash (within bench noise) — the
Q3_K MoE path doesn't yet have a batched scheduler aggregator
specific to its kernel shape (the per-N batched MMVQ family covers
Q4_K / Q6_K / Q8_0 but not Q3_K). That's a tier-2 follow-up.

## Results — main (`0bbaa14` + `af189c0` cherry-pick)

| path | N | result |
|---|---|---|
| no_batched | 1 | **err** (0/1 ok) |
| no_batched | 2 | **err** (0/2 ok) |
| batched    | 1 | **err** (0/1 ok) |
| batched    | 2 | **err** (0/2 ok) |

Every request errors out before producing a token. Server log:

```
err="prefill logits: ... gdn prefill TP layer 0:
     weight dtype Q3K not supported by V1 qmatmul dispatch"
```

Weights upload to GPU successfully on main (the GGUF reader has no
issue, model layout is fine), but the first matmul in the hybrid
forward path hits main's `qdtype_of` allowlist which doesn't accept
Q3_K. Same failure mode as the dense Q3_K bench (`cert_q3k.md`),
this time on the MoE forward path.

## Headline

**MoE Q3_K_S: branch 30 t/s, main 0 t/s.** End-to-end working
mixture-of-experts inference on Q3_K weights — main rejects this
class of model outright. The kernels that fire on each forward
step include `indexed_moe_mmvq_q3_k` (gate + up legs of expert
FFN, dispatched twice unfused), plus the existing Q4_K r2 / Q5_K
down kernels for `ffn_down_exps` (Q3_K_S keeps those at higher
precision).

## Split GGUF support (prerequisite)

Committed in `6cccd6b`. The bench model is a single-file requant,
but the same loader now handles multi-part GGUFs (`-NNNNN-of-MMMMM.gguf`
with `split.count > 1`). Verified by splitting the 9B-Q3_K_S via
`llama-gguf-split` into 3 parts and re-serving — coherent output,
same `/v1/chat/completions` round-trip.

This unblocks future Q3_K_XL / UD-Q3_K_XL models that ship in the
split convention. (The 122B-A10B-UD-Q3_K_XL still won't load
because it includes MXFP4 / IQ tensors which flambeau V1 doesn't
support yet — separate dtype-coverage gap, unrelated to the split
loader.)

## Files

- `certs/perf/branch_vs_main/HEAD_moe_q3k.json` — branch run (4 cells, all ok)
- `certs/perf/branch_vs_main/HEAD_moe_q3k.log` — branch driver log
- `certs/perf/branch_vs_main/MAIN_moe_q3k.json` — main run (4 cells, all err)
- `certs/perf/branch_vs_main/MAIN_moe_q3k.log` — main driver log
- `certs/perf/branch_vs_main/cert_moe_q3k.md` — this write-up

## Reproducibility

Requantise 35B-A3B:
```
LD_LIBRARY_PATH=/opt/rocm-host/lib \
  /artefact/llama.cpp/build-mi50/bin/llama-quantize --allow-requantize \
  /artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  /artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf \
  Q3_K_S 8
```

Branch bench:
```
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py \
  --out certs/perf/branch_vs_main/HEAD_moe_q3k.json \
  --models qwen36_35B_a3b_q3_k_s --topos pp2tp2 \
  --paths batched,no_batched --concs 1,2
```

main bench (cherry-pick arch fix + take branch's script):
```
git checkout main && git cherry-pick af189c0
git checkout feature/batched-mmvq-decode -- scripts/bench/run_matrix.py
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py \
  --out certs/perf/branch_vs_main/MAIN_moe_q3k.json [same args]
git checkout -- scripts/bench/run_matrix.py
git reset --hard HEAD~1
git checkout feature/batched-mmvq-decode
```
