# Q3_K bench: branch enables a model main rejects (2026-05-12)

Companion to `cert2.md`. cert2 covered the no-regression case
(branch vs main on Q4_0 / Q4_1 / Q4_K_S models that both can run);
this cert measures the **enablement** case: a Q3_K-weighted model
that main physically cannot dispatch.

## Model

Qwen3.5-9B re-quantised from `Qwen3.5-9B-Q4_1.gguf` → `Qwen3.5-9B-Q3_K_S.gguf`
via:
```
llama-quantize --allow-requantize \
  /artefact/models/Qwen3.5-9B-Q4_1.gguf \
  /artefact/models/Qwen3.5-9B-Q3_K_S.gguf \
  Q3_K_S 8
```

Output: 4051 MiB / 3.80 BPW (down from 5556 MiB / 5.21 BPW).
Quality is necessarily degraded by requant-from-Q4_1, but the bench
measures kernel throughput, not generation quality. Coherence smoke
test passed (`"Hi! How can I help you today"` for input `"Hi"`).

## Setup

- Same as `cert2.md`: 4× MI50 gfx906, ROCm 7.1.1, pp2tp2, N ∈ {1, 2},
  both batched and no_batched paths, `max_tokens=256`, prompt `'x'*1024`.
- Branch built with `cargo build --release -p flambeau-cli --features hip_serve`.
- main was patched with `af189c0` (qwen35 arch registration) so it
  could attempt model load; cherry-pick reverted post-bench.

## Results

### Branch (`feature/batched-mmvq-decode` @ edf9563 + qdtype_of fix)

| path | N | prefill_ms | cum_tps | per_stream_tps | prompt_tok | wall (s) |
|---|---|---|---|---|---|---|
| no_batched | 1 | 10347 | 20.5 | 20.5 | 3003 | 22.8 |
| no_batched | 2 | 18184 | 26.2 | 16.4 | 3003 | 38.9 |
| batched    | 1 | 10468 | 20.5 | 20.5 | 3003 | 23.0 |
| batched    | 2 | 18291 | 25.9 | 16.3 | 3003 | 39.1 |

Per-token prefill rate: ~290 tok/s at N=1, ~165 tok/s aggregate at N=2.
Decode: 20.5 tok/s single-stream, 26 tok/s aggregate at N=2 (1.28× scaling).

### main (`0bbaa14` + `af189c0` cherry-pick)

| path | N | result |
|---|---|---|
| no_batched | 1 | **err** (0/1 ok) |
| no_batched | 2 | **err** (0/2 ok) |
| batched    | 1 | **err** (0/1 ok) |
| batched    | 2 | **err** (0/2 ok) |

Every request errors out before producing a token. Server log
(see `MAIN_q3k.log` + `scripts/bench/logs/qwen35_9B_q3_k_s__*.log`):

```
err="prefill logits: ... gdn prefill TP layer 0:
     weight dtype Q3K not supported by V1 qmatmul dispatch"
```

The model **loads** on main (the weight upload succeeds), but the
first matmul fails because main has no Q3_K dispatch row.

## Headline

**Branch: 20.5 t/s decode. Main: 0 t/s decode** (load succeeds, first
matmul throws). The tier-1 K-quant work is what turns this model from
unrunnable on the production path into a fully working one — the
"speedup ratio" is undefined; the branch enables a workload main
cannot serve.

The qwen3-moe / hybrid model code path needed one additional change to
let Q3_K kernels actually fire from a loaded model: extending
`qdtype_of` + `run_indexed_moe_gate_up` + `run_indexed_moe_down` +
`validate_moe_dtypes` in `crates/models/qwen3-moe/src/forward/common.rs`
to accept Q2_K / Q3_K (and Q8_K for `qdtype_of`). This is committed
on the branch alongside this cert.

## Files

- `certs/perf/branch_vs_main/HEAD_q3k.json` — branch run (4 cells, all ok)
- `certs/perf/branch_vs_main/HEAD_q3k.log` — branch driver log
- `certs/perf/branch_vs_main/MAIN_q3k.json` — main run (4 cells, all err)
- `certs/perf/branch_vs_main/MAIN_q3k.log` — main driver log
- `certs/perf/branch_vs_main/cert_q3k.md` — this write-up

## Reproducibility

Branch:
```
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py \
  --out certs/perf/branch_vs_main/HEAD_q3k.json \
  --models qwen35_9B_q3_k_s --topos pp2tp2 \
  --paths batched,no_batched --concs 1,2
```

main (with cherry-pick + branch's script):
```
git checkout main && git cherry-pick af189c0
git checkout feature/batched-mmvq-decode -- scripts/bench/run_matrix.py
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py \
  --out certs/perf/branch_vs_main/MAIN_q3k.json [same args]
git checkout -- scripts/bench/run_matrix.py
git reset --hard HEAD~1 && git checkout feature/batched-mmvq-decode
```
