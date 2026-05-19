# v2 vs legacy TP2 kernel-by-kernel — Qwen3.6-35B-A3B-Q4_0, real task (2026-05-19)

## Setup

- Model: `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`
- Topology: TP2 on `hip:0,1`
- Prompt: ~700-token technical instruction
- Tokenized: `prompt_tokens = 287`, `completion_tokens = 128`
- Generation: greedy (`temperature=0`)
- One warmup + one measurement under `rocprofv3 --kernel-trace`
- Build: `target/release/flambeau`, post-#244

## Headline

|                       | v2 TP2     | legacy TP2 | ratio          |
|-----------------------|-----------:|-----------:|---------------:|
| Wall time             | 9028 ms    | 4802 ms    | **1.88× slower** |
| Total GPU kernel time | 6071 ms    | 4246 ms    | 1.43×          |
| Total launches        | **807 486**| **374 006**| **2.16× more** |
| Per-token GPU         | 14.6 ms    | 10.2 ms    | 1.43×          |

v2 wall ratio (1.88×) > v2 GPU ratio (1.43×): the extra ~0.45× is
CPU-side launch overhead from 2.16× more `hipModuleLaunchKernel`
calls. Output coherent on both stacks.

## Top per-kernel deltas (sorted by absolute v2-cost)

| kernel | v2 ms (calls) | legacy ms (calls) | observation |
|--------|-------------:|------------------:|-------------|
| `attention_decode_f16` | **1187 (2600)** | 0 (60) | **v2 plain decode-attn**; legacy uses `splitk` |
| `attention_decode_f16_splitk_chunk` | — | **445 (2540)** | legacy split-K decode |
| `attention_decode_f16_splitk_combine` | — | (2540, smaller) | legacy split-K combine |
| `__amd_rocclr_copyBuffer` | 568 (165 286) | not in top | **v2 hipMemcpy launches 4×+** |
| `indexed_moe_mmvq_q4_0_q8_1` | 473 (9100) | 541 (9170) | parity |
| `gdn_state_step_alphabeta` | 414 (7920) | 437 (7920) | parity |
| `mmvq_q5_0_q8_1` | **295 (52 080)** | 103 (17 640) | **v2 fires 2.95× more** |
| `cast_f32_f16` | 240 (98 640) | 125 (37 000) | v2 2.67× more casts |
| `mmvq_q6_k_dp4a_q8_1` | 239 (1734) | 100 (302) | v2 calls 5.7× more (LM head / etc.) |
| `mmvq_q8_0_t128_vdr2` | 220 (36 600) | 82 (8840) | v2 4.14× more |
| `shared_expert_scale_f32` | 176 (34 720) | not in top | v2 per-token loop overhead |
| `swiglu_f32_to_q8_1` | 154 (52 920) | 106 (28 600) | v2 1.85× more |
| `cast_f16_f32` | 149 (69 440) | not in top | v2-only |

## What v2 is missing (legacy-only kernels)

| kernel | legacy ms (calls) | v2 has? | what it does |
|--------|-----------:|---------|-------|
| `attention_decode_f16_splitk_chunk` + `_combine` | ~445 ms / 5080 | ✗ | Split-K flash-decoding: partitions KV across grid.y. Memory cert quotes **7.78×** speedup vs single-pass at n_tokens=2048. v2 leaves this on the table for every decode step. |
| `p2p_allreduce_residual_rmsnorm_tp2` | 123 ms / 10 400 | ✗ | Single launch: AR + add residual + rmsnorm. v2 does these as 3 separate kernels with intermediate HBM round-trips. |
| `p2p_allreduce_residual_tp2` | 82 ms / 10 720 | ✗ | Single launch: AR + add residual. Same story. |
| `topk_softmax_f32` | 166 ms / 10 560 | ✗ | GPU-side MoE router top-k. v2 does host top-k with DtoH + sync + HtoD upload per layer per token. |

Sum of legacy-only kernels: **~816 ms** that v2 has to make up
through replacement-kernel work. v2 burns ~568 ms of additional
`copyBuffer` calls + 91 ms of extra `cast_*_*` calls — roughly even
at the GPU level, but the launch-count difference (807k vs 374k)
costs CPU overhead on top.

## Levers, ranked by expected impact

### 1. **Split-K decode attention** (biggest single lever)
v2 spends 1187 ms in `attention_decode_f16`; legacy spends 445 ms
in `splitk_chunk` + `_combine`. Per-call cost: v2 456 µs vs legacy
175 µs = **2.6× per-call slowdown**. Both stacks fire the kernel
the same number of times (~2540).

The `attention_decode_f16_splitk` op already exists in
`crates/ops/src/hip/attention.rs`. v2's `standard_attn` composite
needs to dispatch to it when `n_tokens_kv` is large enough that the
single-pass kernel under-utilises CUs (legacy's threshold is around
n_tokens_kv ≥ 512 on Qwen3.6 with head_dim=256).

Expected wall delta: ~750 ms saved (≈ 8% off the 9 s total at this
prompt length; grows with context).

### 2. **Fused AR+residual+rmsnorm kernel for TP**
Legacy's `flambeau_p2p_allreduce_residual_rmsnorm_tp2` fuses three
ops into one launch on each TP step. v2 does this as
`ar_sum_f32` → `add_f16` → `rmsnorm_f16`, three kernels at three
HBM round-trips each.

Wiring: a new `TopologyHooks::ar_sum_residual_rmsnorm` hook +
matching op + dispatch from the v2 standard_attn / dense_ffn /
moe_ffn composites at the AR-fold sites. Estimated save: ~200 ms
of GPU + the 30k launch-overhead reduction.

### 3. **GPU-side `topk_softmax_f32` for MoE router**
v2 currently does host top-k + softmax per layer per token (and
per-token in the prefill loop). Legacy fires `topk_softmax_f32`
directly. v2's reason: router weight may be quantised; the block's
`route_decode` only handles F16/F32. Fix: dispatch quantised-router
via existing `qmatmul` first into `router_logits_f32` (already
done), then call `topk_softmax_f32` instead of the host topk + HtoD
upload. Saves the host-sync per layer per token + 1 HtoD.
Estimated save: ~166 ms of legacy-kernel-time + the host sync.

### 4. **Reduce small `copyBuffer` and per-slot loop launches**
v2 fires `__amd_rocclr_copyBuffer` 165k times for 415 tokens of
work — ~400 copies per token. These are the per-slot KV-append
copies, the various staging memcpys inside composites. Each is a
~3 µs HIP call but the host-side overhead adds up. Trace shows v2
has 2.16× more launches overall.

Specific suspects:
- `shared_expert_scale_f32` fires 34 720 times in v2 — that's the
  shared-expert per-token loop (already filed as #245).
- `cast_f16_f32` fires 69 440 times in v2 (zero in legacy top-20)
  — likely the v2 `moe_ffn` shared-expert TP AR staging
  introduced in #244, plus various delta-staging casts.

### 5. **Per-call Q5_0 / Q6_K MMVQ — call-count not per-call cost**
The `mmvq_q5_0_q8_1` 3× call-count gap is a SHAPE-MIX effect, not
a dispatch-row bug. v2 fires it from more layers' alpha/beta
matmuls because the per-slot loops aren't batched. Closing this
falls out from levers #1-4; not a separate audit.

## Why this matters

The qwen35moe-v2 stack is correct and now spans the production
pp2tp2 topology (#244), but per-real-task wall is ~1.88× legacy on
TP2. The four levers above account for nearly all of that gap.
The biggest single one (split-K attention, #248 candidate) is
already-shipped op-side; only the v2 composite dispatch is missing.

## Reproduce

```
scripts/profile/real_task_trace.sh v2     /tmp/rt_v2_tp2  <gguf> 0,1 tp 2
scripts/profile/real_task_trace.sh legacy /tmp/rt_leg_tp2 <gguf> 0,1 tp 2
python3 scripts/profile/summarize_kernel_trace.py \
    /tmp/rt_<mode>_tp2/threadreaper/*kernel_trace.csv --top 20
```
