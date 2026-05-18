# D4 — v2 vs legacy comparison + GPU/CPU profile (2026-05-18)

## TL;DR

D3-A's shared-Session scheduler engages correctly (D4 cert). But on
`Qwen3.6-35B-A3B-Q4_0` / PP2, **v2 is 6.6× slower than legacy at
N=1** — and batched-decode (N=2/4) inherits that gap.

Root cause: the v2 `moe_ffn` composite does NOT dispatch to the
batched-expert kernels (`flambeau_indexed_moe_*`) that the legacy
qwen3-moe forward path uses. Instead it loops experts per token,
calling plain `flambeau_mmvq_q4_0_q8_1` ~8× per token. This costs
both GPU time (kernel work doubles) and CPU time (10× more
`hipModuleLaunchKernel` calls).

D3-A itself is fine. The gap is upstream in the composite.

## Wall-clock comparison

Same model, same prompt, same K=24 / temp=0 / PP2 on hip:0,2:

|  N  | legacy agg t/s | v2 D3-A agg t/s | v2 ÷ legacy |
|----:|---------------:|----------------:|------------:|
|   1 |        42.95   |          6.48   |   0.151×    |
|   2 |        42.83   |          6.36   |   0.149×    |
|   4 |        43.84   |          6.50   |   0.148×    |

Both stacks hit the same ~1× aggregate ceiling at N>1 (per-slot
serial composite loops, as documented in D4 cert). v2 is just 6.6×
behind from N=1.

Re-confirmed: legacy `--no-gpu-sampler` at N=1 = 43.40 t/s (matches
default-gpu-sampler 42.95 t/s). At greedy temp=0, the GPU sampler
is not the variable. The gap is in the forward path.

## GPU kernel profile (rocprofv3 --kernel-trace)

One single-request K=4 capture, post-warmup, K=4 tokens of decode +
~24 tokens of prefill. Sum of all GPU kernel time:

|                                    | legacy   | v2 D3-A   | ratio  |
|------------------------------------|---------:|----------:|-------:|
| Total GPU kernel time              |  382 ms  |  1719 ms  |  4.5×  |
| Total kernel launches              | 19 411   |  199 415  | 10.3×  |
| `flambeau_mmvq_q4_0_q8_1` calls    |  4 524   |  51 354   | 11.4×  |
| `flambeau_mmvq_q4_0_q8_1` total ms |  109     |   957     |  8.8×  |
| `flambeau_indexed_moe_*` calls     |  ~600    |       0   |   ∞    |
| `flambeau_indexed_moe_*` total ms  |   ~72    |       0   |   ∞    |

The legacy stack uses `flambeau_indexed_moe_mmq_q4_0_gate_up_tile8`
(40 ms / 80 calls) + `flambeau_indexed_moe_mmq_q4_0_down_tile8`
(14 ms / 70 calls) + `flambeau_indexed_moe_mmvq_q4_0` (12 ms /
210 calls). These are the batched-expert MoE kernels.

The v2 stack uses NONE of these. Instead it issues ~8 per-token
launches of plain `flambeau_mmvq_q4_0_q8_1` (one per active expert),
plus `flambeau_swiglu_f32_to_f16` (17k calls / 74 ms) and
`flambeau_scale_f16` (17k / 57 ms) — sub-ops that legacy fuses into
the indexed_moe kernel.

## CPU profile (rocprofv3 --hip-trace)

Host-side HIP API time on the same capture:

| HIP function              | legacy   | v2 D3-A  | ratio |
|---------------------------|---------:|---------:|------:|
| `hipModuleLaunchKernel`   |  186 ms  | 1727 ms  | 9.3×  |
| `hipModuleLaunchKernel` n | 18 332   | 188 756  | 10.3× |
| `hipMemcpyAsync` calls    |  1 893   | 11 421   |  6.0× |
| `hipStreamSynchronize`    |  123 ms  |   89 ms  | 0.72× |

Translation:

- CPU burns **~1.5 sec more** in `hipModuleLaunchKernel` itself —
  pure submission overhead. Each launch costs ~9 µs; v2 just makes
  10× more of them.
- Memcpy volume is similar, but v2 splits it across 6× more calls
  (many small per-token int32 position uploads, scratch zero-fills,
  etc.). Per-call cost is fine (avg 398 µs); aggregate is similar.
- v2 actually has fewer sync points than legacy (less driver-stall
  time), so the gap really is launch + compute density.

## Why this is happening in v2

Reading `crates/forward/src/core/composites/moe_ffn.rs`:

The v2 MoE composite, after routing, runs a per-expert loop:

```
for each active expert e in top-k:
    expert_input = gather indices belonging to e
    gate_up = mmvq(W_gate_up[e], expert_input)   ← per-expert MMVQ
    swiglu(...)
    down = mmvq(W_down[e], gated)                ← per-expert MMVQ
    scatter_add into output
```

At decode (n_tokens=1) on Qwen3.6 (top-k=8 experts, 128 expert
pool), that's 8 expert MMVQs per layer × 32 layers per token =
256 MMVQ launches per token — versus legacy's 1 batched
`indexed_moe_mmq_q4_0_gate_up_tile8` launch per layer.

This is exactly the architectural pattern memory entry
`project_quant_coverage_post_phase4` describes for legacy:
"HIP kernels (MMVQ, wave64 MMQ, indexed MoE MMVQ + MMQ tile8)".
The v2 composite simply never adopted the indexed-MoE path.

## D3-A correctness check

D3-A's shared-session scheduler is verified by the D4 cert:
`pending=N` batched dispatches confirmed in trace, all N streams
complete together with isolated state. The aggregate ceiling at
N>1 (~1×) is universal to both stacks — same per-slot kernel
serial loops in decode composites. D3-A is correct as-shipped.

The 6.6× v2 vs legacy gap is **inherited** from the v2 forward
path's MoE composite, not introduced by D3-A. Without D3-A the v2
stack would have been 6.6× slower AND held N copies of weights in
VRAM. With D3-A it's still 6.6× slower but at 1× weight cost — the
VRAM architectural win stands.

## Next levers (separate slices)

In rough impact order:

1. **MoE composite indexed-expert dispatch.** Port the
   `flambeau_indexed_moe_mmvq_q4_0_q8_1` /
   `flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1` /
   `flambeau_indexed_moe_mmq_q4_0_down_tile8_dp4a_q8_1` dispatch
   into `crates/forward/src/core/composites/moe_ffn.rs`. Expected
   gain: ~4–5× on MoE archs (closes most of the gap).
   Filed as `#239` candidate.

2. **Batched-decode attention kernel.** Per D4 cert: replaces the
   per-slot KV-append + per-slot `attention_decode_f16(pos)` loop
   with one batched launch. Lifts N>1 aggregate from 1× toward N×.
   Filed as `#240` candidate.

3. **Batched-GDN companion** for hybrid archs (qwen3-moe family).
   Same shape as #240 for the GDN composite. Filed as `#241`
   candidate.

Dense archs (Qwen3.5-9B) at v2 N=1 hit 30–32 t/s on PP2/TP2 —
likely the legacy-comparable level, but no apples-to-apples
counterpart exists because qwen35 dense isn't in the legacy stack.

## Reproduce

```
scripts/profile/decode_step_trace.sh v2     /tmp/v2_trace
scripts/profile/decode_step_trace.sh legacy /tmp/legacy_trace
python3 scripts/profile/summarize_kernel_trace.py \\
    /tmp/v2_trace/threadreaper/*kernel_trace.csv --top 15
python3 scripts/profile/summarize_hip_trace.py \\
    /tmp/v2_trace/threadreaper/*hip_api_trace.csv --top 12
```
