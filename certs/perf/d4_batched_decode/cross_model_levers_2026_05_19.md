# Cross-model v2-vs-legacy lever audit (2026-05-19)

## Bench matrix (PP, hip:0,2 or hip:0,2,1,3)

Single-stream wall-clock + per-kernel trace on every v2+legacy model:

| model | arch | topo | v2 prefill | leg prefill | v2 decode | leg decode | v2/leg total |
|-------|------|------|-----------:|------------:|----------:|-----------:|-------------:|
| Qwen3.6-35B-A3B-Q4_0 | qwen35moe (MoE+GDN) | PP4 | 232 t/s | 686 t/s | 30.8 | 59.0 | 0.44× |
| gemma-4-31B-Q4_0     | gemma4 (dense, head_dim=512) | PP2 | 184 t/s | 180 t/s | 6.9† | 14.2 | 0.58× |
| gemma-4-31B-Q4_0     | gemma4 dense | PP4 | 186 t/s | 181 t/s | 8.9† | 15.6 | 0.67× |

† gemma4-v2 emits **garbage tokens** ("la own la own ownS…") at all
PP fan-outs ≥ 1 chunk. Correctness regression, not a perf gap. The
decode timing is the model looping in a degenerate state.

## Kernel-level (matching kernel = matching cost)

**gemma-31B-Q4_0 / PP2 / K=4 + prefill** (both stacks run the same
qmatmul.rs dispatch, gemma4-only paths):

| kernel | v2 ms (calls) | legacy ms (calls) | per-call delta |
|--------|--------------:|------------------:|---------------:|
| `mmq_q4_0_wave64_q8_1` (prefill MMQ) | 842 (706) | 884 (806) | **v2 faster** |
| `mmvq_q4_0_q8_1` | 213 (2118) | 181 (1398) | v2 51% more calls, faster per-call |
| `mmvq_q4_1_t128_q8_1` | 70.6 (490) | 71.2 (490) | parity |
| `mmvq_q4_k_r2_q8_1` (embed/lm_head) | 60 (8) | 61 (8) | parity |
| **Total kernel time** | **1332 ms / 14 949** | **1387 ms / 15 144** | **v2 SLIGHTLY FASTER** |

The gemma4-v2 forward path runs the same kernels for the same wall
cost as legacy at the kernel level. The wall-clock total_ms gap on
this model is a **correctness regression masked as a perf gap**.

**Qwen3.6-35B-A3B / PP2 / K=4 + 24-tok prefill** (per post-#242 cert):

| kernel | v2 ms (calls) | legacy ms (calls) | observation |
|--------|--------------:|------------------:|-------------|
| `indexed_moe_mmq_q4_0_gate_up_tile8` | 38 (80) | 40 (80) | parity |
| `indexed_moe_mmq_q4_0_down_tile8` | 14 (70) | 14 (70) | parity |
| `mmq_q8_0_oracle_q8_1` (attn prefill) | 34 (268) | 52 (308) | v2 fewer (faster) |
| `mmvq_q4_0_q8_1` | 91 (4554) | 109 (4524) | v2 faster per-call |
| `mmvq_q6_k_dp4a_q8_1` | 55 (278) | 14 (248) | v2 30 extra calls; tail-skewed mean |
| `mmvq_q5_0_q8_1` | 50 (3240) | 27 (3240) | v2 1.8× slower per-call, SAME shape — **launch geometry?** |

## Recalibrated lever list

What the data actually says:

### 1. **gemma4-v2 correctness regression** (BIGGEST single lever)
The gemma-31B v2 wall is 1.5× legacy not because the kernels are
slower (they're parity-or-faster) but because the **model produces
garbage tokens** under v2. Fixing this makes gemma4-v2 immediately
competitive at the kernel level it already runs at. The bug appears
at all prompt lengths (even <100 tokens). Files: #222 was the
chained-prefill regression but evidently the gemma4 path has a
broader correctness issue. Filed as **#245**.

### 2. **Qwen3.6 MoE per-token decode path**
On the real-task bench (pp=708 / tg=128 / PP4), v2 decode is
30.84 t/s vs legacy 59.0 t/s = **0.52× legacy**. K=24 micro-bench
masked this because the wall is dominated by per-stream startup.

The trace breakdown shows the remaining gap is in the per-token
decode pattern:
- attention output_proj (mmvq_q4_0_q8_1): v2 fires it slightly more
  often than legacy, possibly because legacy's decode-aware
  attention has a custom kernel sequence v2 doesn't use.
- per-token shared-expert (block has no prefill variant; v2 loops
  per token).
- `mmvq_q5_0_q8_1`: v2 1.8× per-call slower at IDENTICAL call count
  (3240). Worth investigating launch grid.

### 3. **NOT a lever**: prefill MoE batching (already shipped #242)
The tile8 prefill kernels now match legacy 1:1. The qwen35moe
prefill rate of 232 t/s at PP4 vs legacy 686 t/s is **not** a
batched-vs-per-token gap any more — those are equal. The 3× gap is
elsewhere. Hypothesis: shared-expert per-token loop during the
1024-token prefill chunked path costs significant wall.

### 4. **NOT a lever** (post-#239): `mmvq_q6_k` per-call cost
The 3.6× mean per-call gap is likely a shape-mix artefact across
separate runs (v2 fires 30 more Q6_K calls at smaller shapes than
legacy, pulling the mean down on legacy). A controlled per-shape
microbench would confirm; not worth the engineering until other
levers are paid down.

## Recommended order of operations

1. **#245 (gemma4-v2 correctness)** — unlocks accurate gemma perf
   measurements AND likely fixes a wider class of arch-specific
   bugs lurking under "smoke tests pass".
2. **#244 (qwen35moe-v2 TP loader)** — unlocks pp2tp2 production
   topology measurement.
3. **#243 (Q5_0/Q6_K launch geometry audit)** — small win on
   qwen35moe per-token decode; needs a microbench harness.
4. **shared-expert prefill batching** — port `SharedExpert::forward_prefill`
   into blocks; saves the per-token loop in v2 moe_ffn prefill.
   Filed as #246.

## Reproduce

```
python3 scripts/bench/real_task_pp1024_tg128.py \\
    --model <gguf> --mesh pp --devices <csv> --tg 128

scripts/profile/decode_step_trace_param.sh <v2|legacy> /tmp/out \\
    <gguf> <devices> <mesh>
python3 scripts/profile/summarize_kernel_trace.py \\
    /tmp/out/threadreaper/*kernel_trace.csv --top 12
```
