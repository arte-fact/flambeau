# MTP-5e profile — cost-of-implementation vs decode-speedup balance

**Question:** the perf cert showed spec is 18 % slower than baseline on PP4.
But how much of that gap is structural (rank-sync wall on PCIe) and how
much is implementation overhead we can chip away at? Answered with HipEvent
section attribution.

## Headline numbers (32 baseline steps + 32 spec macros, after 4-step warmup)

| | baseline L=1 | spec L=2 + MTP |
|---|---:|---:|
| wall ms/iteration | 53.80 ms/step | 108.27 ms/macro |
| tokens per iteration | 1.0 | 1.875 (87.5 % of macros gave 2 tokens) |
| **ms/token** | **53.80** | **57.74** (spec is +7.3 %) |
| profileable sections | 50.36 ms | 100.46 ms |
| accept rate | n/a | 90.6 % |

(Earlier MTP-5e A/B reported 50.33 vs 59.33 → 18 % slower. This run is
warmed-up which slightly favours both branches; +7 % is the warm-cache
ratio. Headline 18 % includes cold-cache/allocator effects.)

## Where spec macro's 108 ms actually go

| section | ms/macro | per token (÷ 1.875) | cumulative |
|---|---:|---:|---|
| `l2_stage_end` (peer-copy + 16-layer body × 4 ranks, L=2) | 89.86 | 47.92 | structural |
| `l2_output_pos1_done` (LM head pass 2) | 3.05 | 1.63 | optimisable |
| `l2_output_pos0_done` (LM head pass 1) | 2.66 | 1.42 | structural |
| reject path (rollback + restore + L=1 redo, ~9.4 % rejects) | ~6.25 | 3.33 | optimisable |
| GDN snapshot (always paid) | 2.0 | 1.07 | optimisable |
| MTP draft | 1.5 | 0.80 | structural |
| `l2_embed_done` | 0.14 | 0.07 | structural |
| **measured wall total** | **108.27** | **57.74** | |

Compare to baseline's 50.36 ms attributable per step (53.80 ms wall).

## The L=2 body cost

`l2_stage_end` is **89.86 ms/macro = 1.78× the L=1 baseline body of 50.36 ms**.
Per-token: 89.86 / 2 (work-token slots) = **44.93 ms** — the L=2 body
itself is *more efficient per slot* than baseline L=1 (44.93 vs 50.36),
which is the batching working as designed.

**The waste comes from rejected slots**, not from the L=2 body being slow:

- L=2 body at 89.86 ms / 1.875 effective-tokens = 47.92 ms/tok (vs 50.36
  L=1) → savings of **2.44 ms/tok**.
- Reject overhead 6.25 / 1.875 = 3.33 ms/tok cost.
- Output head ×2 + embed: 4.85 / 1.875 = 2.59 ms/tok cost (vs baseline ~1.6).
- GDN snap + MTP draft: 3.5 / 1.875 = 1.87 ms/tok cost.

Net: −2.44 + 3.33 + 1.0 + 1.87 = **+3.76 ms/tok over baseline**.
Predicted: 53.80 + 3.76 = 57.56. Measured: 57.74 ✓.

## Optimisation levers, in priority order

### Lever 1: GDN-only reject re-step — SHIPPED, measured savings match prediction

**Implemented and validated 2026-04-29** (commit landed in
`crates/models/qwen3-moe/src/forward/{pp,spec}.rs` + `sharded.rs`).

| | pre-Lever-1 | post-Lever-1 | delta |
|---|---:|---:|---:|
| baseline ms/tok | 50.33 | 49.95 | (variance, ±0.4) |
| spec ms/tok | 59.33 | **56.46** | **−2.87 ms/tok** |
| spec/baseline | +18 % slower | +13 % slower | |
| accept rate | 87.5 % | 87.5 % | unchanged |
| first 16 tokens vs baseline | bit-identical | bit-identical | correctness preserved |

Predicted savings: 2.9 ms/tok. Measured: 2.87 ms/tok. Match within 1 %.

What the implementation looks like:
- `RankForwardPrefillScratch` gained `gdn_input_snapshots: Vec<Option<DevicePtr>>`
  — one device buffer per local layer, allocated at scratch construction
  for GDN-bearing layers only (`hidden * 2` bytes each, ~120 KB / rank).
- `forward_prefill_pp_logits_paired_l2` issues a DtoD memcpy_async
  (position 0, hidden_bytes) on each rank's default stream BEFORE the
  GDN layer's `forward_layer_prefill` call, populating the snapshot
  buffer.
- `Qwen3MoEShardedSession::redo_gdn_only_pp` runs `forward_gdn_layer_decode`
  per-rank-parallel using the snapshots. Each rank processes its own
  GDN layers concurrently across devices. Discards delta_out (we only
  need the side-effect: GDN state advance).
- `forward_speculative_pp_step` reject branch now does:
  `rollback_full_attn(1)` + `restore_gdn_snapshot` + `redo_gdn_only_pp`
  (instead of `rollback_full_attn(2)` + `restore_gdn_snapshot` +
  `forward_one_token_pp_logits`). Committed token = argmax of L=2's
  p_pos0 (already computed). h_for_next = `prefill_scratch.hidden_a[0]`.

The reject path now costs ~7 ms (per-rank-parallel GDN-only forward)
vs the prior ~50 ms full L=1 redo. Fixed cost of saving snapshots on
every macro (always-paid even on accept): ~12 DtoD memcpys × 0.01 ms
each on each rank, ~0.5 ms/macro across all ranks. Net savings on
12.5 % rejects ≈ 5.4 ms/macro = 2.9 ms/tok.

### Lever 1 (original analysis, kept for reference) — GDN-only reject re-step — REVISED, smaller payoff than first claimed

**Corrected savings: ~2.9 ms/tok** (not 3.3 as first stated).

The reject path currently runs a full `forward_one_token_pp_logits` (50 ms/reject)
to re-advance GDN state by 1 step. The actual purpose of that redo is to
re-update GDN state with last_token's input — KV slot=position was already
correctly written by L=2 verify (input was `last_token`), so it doesn't
need re-writing. The redo also computes residuals for full-attn + MoE
which we don't need (their L=2 outputs at position 0 are correct).

A clean implementation needs a per-layer GDN-input snapshot during L=2
verify so the reject path can run GDN-only forward (`forward_gdn_layer_decode`
on each GDN-bearing layer in parallel across ranks). Each rank then runs
~12 GDN-layer-decode calls × 0.6 ms ≈ 7 ms in parallel.

| | current | with Lever 1 |
|---|---:|---:|
| reject cost | ~50 ms × 12.5 % = 6.25 ms/macro | ~7 ms × 12.5 % = 0.9 ms/macro |
| saved | — | 5.4 ms/macro = **2.9 ms/tok** |

**Implementation cost:** ~300 LOC structural change.
- `RankForwardPrefillScratch` gains a per-GDN-layer snapshot buffer (~120 KB / rank, lazy alloc).
- `forward_prefill_pp_logits_paired_l2` saves x_in (position 0 only) before each
  GDN-layer call.
- `Qwen3MoEShardedSession::redo_gdn_only_pp` runs `forward_gdn_layer_decode` per
  rank-parallel using the snapshots.
- `forward_speculative_pp_step` reject branch swaps in the new path.

**Verdict:** worth doing — flips spec from −7 % wall to ~−2 % wall on this
rig. Combined with Lever 2 fix it tips net-positive. Filed as #197.

(Earlier draft of this cert claimed 3.3 ms/tok savings. That was a
miscount — assumed 100 % of L=1 redo cost recovered, but in practice the
GDN-only re-step still pays per-rank state-step kernel launches + per-rank
sync, which is ~7 ms on this 4-rank setup, not 0.)

### Lever 2 — RE-ANALYSED, the 21 ms gap is genuine 2× compute, not implementation overhead

After adding intra-`forward_gdn_prefill` marks (#199), the per-substage
L=2 / L=1 ratios are:

| sub-stage | L=2 ms/call | L=1 ms/call | ratio |
|---|---:|---:|---:|
| `gdn_ssm_out_p` (Q5_K matmul → 5120) | 0.476 | 0.227 | **2.10×** |
| `gdn_proj_qkv_gate_p` (Q4_1 matmul) | 0.222 | 0.117 | 1.89× |
| `gdn_state_step_p` | 0.041 | 0.034 | 1.21× |
| `gdn_proj_alpha_beta_p` | 0.027 | 0.016 | 1.66× |
| `gdn_norm_quant_p` | 0.026 | 0.034 | **0.76×** (better at L=2) |
| `gdn_conv1d_p` / `gdn_l2norm_qk_p` | 0.019 | 0.015 | 1.28× |
| `gdn_swiglu_quant_p` | 0.016 | 0.009 | 1.74× |
| `gdn_ssm_norm_p` / `gdn_silu_p` / `gdn_cast_f16_p` | ~0.01 | ~0.01 | ~1.05× |

The matmul kernels (ssm_out, proj_qkv_gate, etc.) dominate the L=2
body cost. They're **memory-bandwidth-bound at small m** but the L=2
case effectively runs 2× MMVQ calls (one per row), so launch-overhead
is ~7–14 % of call time per
`project_v2_31_b_e_multirow_dead_lever.md` — calling MMVQ twice at
m=2 isn't pathological, it's genuinely 2× the activation traffic.

**The "21 ms layer_start_p gap" I claimed earlier was a mis-attribution.**
`layer_moe_done_p` mysteriously does not appear in the profile output
(despite being in source, with the right `&'static str` literal — likely
a pairwise-window edge case in `profile.rs::flush` that couldn't be
isolated without deeper instrumentation). The MoE work between
`layer_post_norm_p` and the next layer's `layer_start_p` therefore gets
attributed to `layer_start_p`. At L=2 that MoE work doubles (2 tokens
× experts), explaining the 0.305 → 0.633 ms/call jump exactly.

**There is no big implementation-side lever hiding in the L=2 body.**
The 1.78× L=2/L=1 ratio is the structural cost of running 2 tokens
sequentially through the same per-rank PCIe pipeline. The paper's
"concurrency-not-free" assumption violation (Section 3.4) IS the
binding constraint on this rig.

**Implication:** with Lever 1 shipped (−2.87 ms/tok) and Lever 2
re-classified as structural, the spec-decode wall floor on PP4 is
~56 ms/tok ≈ +13 % vs baseline. Closing further requires:

1. **Topology change** — NVLink/xGMI eliminates the per-rank PCIe
   sync wall that scales L=2 cost linearly
2. **Higher acceptance** — FastMTP fine-tune (multi-week training
   work) lifts α from 0.875 to ~0.95+, reducing wasted-slot cost
3. **Concurrent draft + verify** — overlap the MTP draft (1.5 ms,
   on the last rank) with L=2 verify body kernels on the same rank;
   small but free win, ~0.5–1.0 ms/macro
4. **Different verify shape** — K=2 spec instead of K=1 (paper's
   γ=2 with optimal γ at α=0.85 is 4–5; but each γ adds another L
   to the verify, so on PCIe it's compounding 2×)

(Old Lever 2 mid-analysis below kept for reference.)

### Lever 2 (mid-analysis, superseded): investigate the 27 ms unaccounted in L=2 body — DONE, attributed

Per-substage marks added to `forward_layer_prefill` (`layer_start_p`,
`layer_attn_gdn_p`, `layer_attn_full_p`, `layer_post_norm_p`,
`layer_moe_done_p`). Re-run gives this L=2 body breakdown per macro:

| section | ms/macro | calls | mean per call (ms) |
|---|---:|---:|---:|
| `layer_start_p` (between layers, loop overhead) | 40.49 | 64 | **0.633** |
| `layer_attn_gdn_p` (full GDN prefill body) | 39.09 | 48 | 0.815 |
| `layer_attn_full_p` (full-attn prefill body) | 5.99 | 16 | 0.374 |
| `layer_post_norm_p` (residual + post-rmsnorm) | 1.42 | 64 | 0.022 |
| `l2_stage_end` (per-rank end-of-stage memcpy + sync) | 3.06 | 4 | 0.766 |
| (sum) L=2 body proper | 90.05 | | |

Compare per-call to L=1 baseline:

| | L=1 baseline | L=2 prefill | L=2 / L=1 ratio |
|---|---:|---:|---:|
| layer_start_* (loop iter, per layer) | 0.305 ms | 0.633 ms | **2.07×** |
| layer_attn_gdn (full GDN) | ~0.6 ms (sum of gdn_* subs) | 0.815 ms | 1.36× |
| layer_attn_full | 0.22 ms | 0.374 ms | 1.70× |

The expected L=2 / L=1 ratio for compute-bound work is 2× (linear in L);
for memory-bound work it's closer to 1× (re-uses cached weights). What we
see:
- **GDN attention scales 1.36×** — better than naive 2× because GDN is
  state-update-bound (state-step kernel is cheap per L), not weight-bound.
- **Full-attn scales 1.70×** — close to expected 2×.
- **Loop-iteration overhead scales 2.07×** — *worse than expected*, this
  is the smoking gun.

**The 21 ms / macro gap is in `layer_start_p`** — the inter-layer
transition. L=1 baseline pays 0.305 ms × 64 = 19.5 ms/step on inter-layer
overhead; L=2 pays 0.633 × 64 = 40.5 ms/macro. Doubling makes some sense
(L=2 has 2× the per-kernel-launch latency to digest) but *all the way to
2.07×* for a section that's mostly host-side loop work suggests the
bottleneck is **kernel-launch latency on cold-state-step kernels**, not
host loop overhead.

Hypotheses (need targeted instrumentation to confirm):
1. The L=2 GDN state-step kernel's first-launch latency per layer is
   higher than L=1's because the kernel touches more L1 working set
   (different launch config) and isn't warm.
2. `LayerPrefillScratch` does an allocator path that `LayerForwardScratch`
   doesn't on the L=1 path.
3. The L=2 path uses different `forward_*_prefill` sub-functions that
   each do a few extra setup memcpys vs the decode versions.

**Next step (ticketed as MTP-5h-2b):** add intra-`forward_gdn_prefill`
marks (mirror gdn.rs's existing decode marks) to localise which sub-step
of GDN is paying the extra. Should pinpoint whether it's setup overhead
or actual compute slack.

Combined with Lever 1: spec at ~47 ms/tok = 1.14× speedup. Net-positive.

### Lever 3: lazy GDN snapshot — false economy

GDN snapshot is 2 ms always paid. Could we snapshot only on demand? No —
by the time we know we need to reject, GDN has advanced 2 steps. The
alternative (shadow-GDN copy-on-write) costs the same 2 ms per macro
because the copy itself is what dominates. **Skip this lever.**

### Lever 4: better MTP head (FastMTP) — caps below baseline anyway

At ideal α = 1.0 with current structure: ms/tok = 108 / 2 = **54 ms/tok**.
Even perfect acceptance leaves spec slower than baseline. The fundamental
ceiling is the 89.86 ms L=2 body, not the head's quality. FastMTP only
helps if Levers 1+2 land first.

## Verdict on cost-vs-speedup balance

The implementation overhead split:

- **Structural** (paper's "L=2 wall multiplier", PCIe rank-sync): ~85 % of
  the gap. Won't budge without a topology change (NVLink) or much deeper
  PP changes.
- **Implementation-optimisable** (Lever 1 + Lever 2): ~10 ms/macro =
  ~5 ms/tok of headroom = enough to flip spec from net-negative to
  modestly net-positive (~1.1× speedup).

So the answer is **yes, there is real implementation headroom**, mostly
in two specific places:
1. Reject path doesn't need the full L=1 redo (Lever 1).
2. L=2 body has 27 ms of unaccounted latency to attribute (Lever 2).

Together they could move us from −7 % wall (warm) / −18 % wall (cold) to
+10–14 % speedup, putting spec genuinely net-positive on PCIe Mesh<4> at
this acceptance rate. Both are tractable next-session work.

## Reproducer

```
cargo build --tests --release -p flambeau-qwen3-moe --features hip \
    --test mtp_spec_decode_profile
FLAMBEAU_PROFILE_BASE_STEPS=32 FLAMBEAU_PROFILE_SPEC_MACROS=32 \
    cargo test --release -p flambeau-qwen3-moe --features hip \
    --test mtp_spec_decode_profile -- --nocapture
```

Test: `crates/models/qwen3-moe/tests/mtp_spec_decode_profile.rs`.
Rig: 4× MI50 / PCIe 3.0 x16 / 100 W / ROCm 7.1.1.
Model: `Qwen3.6-27B-Q4_0.gguf` + `Qwen3.6-27B-mtp.gguf` (Q8_0 MTP linears).
