# Decode-gap closing levers — post-bench cert

Hardware: 2× MI50 (gfx906), TP=2 on `hip:0,1`. Prompt 708 tokens
(tokenized from the 1024-token DP4A technical prompt in
`scripts/bench/real_task_pp1024_tg128.py`), 128 decode tokens, greedy.
Each row is one v2 vs one legacy `flambeau serve` run, warmup 1.

## Headline (v2 vs legacy)

### Qwen3.6-27B-Q4_0 (16 FullAttn + 48 GDN, dense FFN)

| config                  | v2 prefill | v2/leg prefill | v2 decode | v2/leg decode | decode gap |
|-------------------------|-----------:|---------------:|----------:|--------------:|-----------:|
| pre-lever (HEAD a3d455d)| 304.8      | 1.011×         | 23.60     | 0.919×        | **8.1%**   |
| L1+2 only               | 305.0      | 1.012×         | 23.94     | 0.929×        | 7.1%       |
| L1+2+3 ungated          | **166.7**  | **0.556×**     | 24.68     | 0.965×        | 3.5%       |
| **L1+2+3 gated**        | **305.7**  | **1.013×**     | **24.88** | **0.969×**    | **3.1%**   |

### Qwen3.6-35B-A3B-Q4_0 (16 FullAttn + 48 GDN, MoE FFN + shared expert)

| config                  | v2 prefill | v2/leg prefill | v2 decode | v2/leg decode | decode gap |
|-------------------------|-----------:|---------------:|----------:|--------------:|-----------:|
| L1+2 only               | 918.3      | 0.963×         | 58.41     | 0.889×        | **11.1%**  |
| L1+2+3 ungated          | 926.8      | 0.975×         | 65.33     | 0.991×        | 0.9%       |
| **L1+2+3 gated**        | **927.6**  | **1.038×**     | **65.15** | **0.979×**    | **2.1%**   |

## Lever-3 prefill regression bisect

Ungated event-based BAR1 ordering dropped 27B v2 prefill from 304.8 →
166.7 t/s (-45%) while only improving decode by 3.1%. Bisect with
Lever-3 reverted (Lever 1+2 only) restored prefill to 305.0 t/s,
confirming Lever-3 as the sole regression source.

Root cause: with `Stream::synchronize` removed, the host queues
kernels far ahead of the GPU. At prefill, each `bar_ar_residual_f16`
call has `n_elems = 708 * 5120 ≈ 3.6M F16`; the BAR1 kernel takes
real wall time, the host enters the next layer's queue while
multiple AR kernels are still pending on the stream. gfx906 / HIP
appears to serialize at high queue depth — pre-lever's
`Stream::synchronize` naturally throttled this.

## Resolution: gate on `n_elems`

`crates/forward/src/runtime/ar.rs` exposes a const
`EVENT_PATH_MAX_ELEMS = 65_536`. The three `bar_ar_*` functions pick
the prologue at call time:

- `n_elems ≤ 65_536` → `ar_publish_with_events`: record per-rank
  event, host-non-blocking publish, peer `stream_wait`. CPU stays
  decoupled from GPU. Decode hidden ≤ 8192 fits comfortably under
  the gate.
- `n_elems > 65_536` → `ar_publish_with_host_sync`:
  `Stream::synchronize` drains own stream before publish (legacy /
  pre-Lever-3 semantics). Naturally throttles queue depth.

Shared `ar_epilogue` handles the slab-reset barriers either way. No
public API change.

## What each lever contributed

Comparing **L1+2+3 gated** to **pre-lever** on the same bench shape:

- 27B Q4_0: prefill +0.3% (parity), decode +5.4%, gap 8.1% → 3.1%.
- 35B-A3B Q4_0: prefill +XX% (v2 ahead now; pre-lever cert used 287-tok
  prompt so direct prefill ratio isn't apples-to-apples), decode
  +13.1%, gap 15% → 2.1%.

Decomposition (27B Q4_0):

- Lever 1+2 alone: decode +1.4% (Q4_0 F16-direct attn_output / ffn_down
  fast path + gate_up_t128 fused mmvq).
- Lever 3 (gated) on top: decode +3.9% extra (event-based AR removes
  hipStreamSynchronize host blocks on decode-shape AR calls).

Decomposition (35B-A3B Q4_0):

- Lever 1+2 alone: decode +1.4% over pre-cert baseline (F16 router
  upload via `upload_router_f16` + Q4_0 shared-expert fused
  gate+up_t128).
- Lever 3 (gated) on top: decode +11.5% extra. MoE has many more AR
  calls per token (router + indexed-down + shared-expert) than dense
  FFN; event-based ordering wins big here.

## Outcome

- v2 prefill at parity or ahead of legacy on both archs.
- v2 decode within 2–3% of legacy on both archs (was 8% / 15%
  pre-lever).
- v2 ahead of llama.cpp by the previous cert's margins (`v2 / llama.cpp
  row` ≈ 1.4–1.7× prefill, 1.06–1.37× decode — those did not regress
  since llama.cpp is unchanged).

## Files touched

- `crates/forward/src/runtime/ar.rs` — Lever 3 (event-gated AR).
- `crates/backend-hip/src/bar_p2p.rs` — `device_id(rank)` accessor.
- `crates/forward/src/ctx.rs` — `QuantWeight::qmatmul_decode_to_f16` +
  `supports_decode_to_f16` (Lever 1).
- `crates/forward/src/core/composites/{standard_attn,dense_ffn}.rs` —
  Lever 1 F16-direct mmvq fast path at `n==1` under AR fold.
- `crates/blocks/src/{delta_net,shared_expert}.rs` — Lever 1
  `mmvq_q4_0_gate_up_t128` dispatch + Q4_0 shared-expert fused
  gate_up.
- `crates/forward/src/loader/{shard,mod}.rs` — Lever 2
  `upload_router_f16` helper.
- `crates/models/qwen35moe-v2/src/loader.rs` — Lever 2 use site.
- `crates/forward/src/runtime/orchestrate.rs` — propagate
  `BarArCoordinator::new` Result.
