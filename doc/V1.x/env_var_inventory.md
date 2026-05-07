# Env-Var Read-Site Inventory (Phase 0b)

Method: `rg 'env::var\("FLAMBEAU_' --type rust` from `/artefact/flambeau/`.
**242 read sites, 113 distinct vars** across 73 files. No `env::var_os`
or direct `getenv` calls match `FLAMBEAU_*`.

Vars are grouped by category:

1. **Production** — read by the live `flambeau serve` / inference path.
2. **Debug / dump** — gates an `if std::env::var(...).is_ok()` block
   that has zero side-effect when unset (printing, dumping tensors,
   probe lines).
3. **Test-only** — exclusively read from `crates/*/tests/*` or
   `crates/*/examples/*`; never reached by the production binary.

Each production entry has: read sites, default behavior, alternate
behavior, forward-path component(s), and cross-ref to
`bench/profiles/optimized.toml`.

The summary table at the end gives one row per var with the
preliminary Phase-1 verdict. The final verdict is decided in
Phase 1 by cross-referencing this document with
`doc/V1.x/optimal_paths.md`.

---

## Section 1 — Production vars

### FLAMBEAU_AR_FUSE_Q8_1

- **Read sites:** `crates/models/qwen3-moe/src/forward/tp.rs:1840`,
  `crates/models/qwen3-moe/src/forward/tp.rs:2610`
- **Default (unset):** `use_q8_1_fused_ar = false`. Path runs unfused
  AR + RMSNorm + separate quantize.
- **Alternate (`=on`):** Engages fused AR+RMSNorm+Q8_1 kernel path.
  Only fires when `world > 1 && cfg.is_dense_ffn()` (TP/Hybrid only,
  dense-FFN path). Both call sites guarded by the same predicate.
- **Component:** `models::qwen3_moe::forward::tp` — full-attn and
  GDN-layer drivers' post-AR path.
- **Cross-ref:** `optimized.toml` `[delete_candidates]` (cert
  measured −1.1% vs unfused on Qwen3.6-27B-Q4_0/TP w=2).

### FLAMBEAU_BATCHED_DECODE

- **Read sites:** `crates/server/src/routes.rs:3358` (in
  `scheduler_can_engage`)
- **Default (unset):** `scheduler_can_engage()` returns `false`;
  the chat handler falls through to the legacy
  `run_completion_blocking_streaming`/`_ids` path. Decode steps go
  direct to `decode_logits`.
- **Alternate (set, any value):** Scheduler-aware path engages —
  routes through `decode_via_scheduler_into` (with fast-path branch
  + race-safe lock at `routes.rs:771`) and the batched leader
  (`dispatch_batched_pending`).
- **Component:** `server::routes` request gating.
- **Cross-ref:** `optimized.toml` `[env]` — `=1`. Production-on.

### FLAMBEAU_BATCH_MAX

- **Read sites:** `crates/server/src/routes.rs:859`
- **Default (unset):** `batch_max = usize::MAX` — leader drains all
  pending decodes into one batched dispatch.
- **Alternate (positive integer):** Cap pending-queue chunk size at
  N. Larger chunks → more cross-stream batching, larger transient
  scratch.
- **Component:** `server::routes::dispatch_batched_pending` leader.
- **Cross-ref:** `optimized.toml` `[delete_candidates]` (only
  measured at N=1 sweep; flagged caveat).

### FLAMBEAU_BATCH_WINDOW_US

- **Read sites:** `crates/server/src/routes.rs:819`
- **Default (unset):** 1500 µs window between leader try-lock and
  drain. The leader sleeps this long if any other slot is active.
- **Alternate (integer µs):** Custom window. `0` disables the
  pre-drain sleep entirely.
- **Component:** `server::routes::decode_via_scheduler_into` leader.
- **Cross-ref:** Not listed in `optimized.toml` (default-acceptable).

### FLAMBEAU_BATCHED_MMVQ

- **Read sites:** `crates/ops/src/hip/qmatmul.rs:100`
- **Default (unset):** Per-row MMVQ loop (production default).
- **Alternate values:**
  - `=v1` — calls `mmvq_q4_1_batched_launch` (decode-batched MMVQ).
  - `=wave64` — forces `qmatmul_q4_1_mmq_wave64_gfx906` recipe.
  - any other truthy → shape-aware: `n>=8192` → wave64 MMQ; else
    falls through to per-row MMVQ.
- **Component:** `ops::hip::qmatmul::dispatch_qmatmul` (Q4_1 only,
  `2 <= m <= MMVQ_Q4_1_BATCHED_MAX_N`).
- **Cross-ref:** Not in `optimized.toml`; cert in
  `certs/perf/mmvq_q4_1_batched_v1_2026_05_04.md`.

### FLAMBEAU_CTX_CAP

- **Read sites:** `crates/server/src/serve.rs:161`,
  `crates/models/qwen3-moe/tests/profile_point_tp.rs:70`,
  `crates/models/qwen3-moe/tests/bench_tp2_any_model.rs:124`,
  `crates/models/qwen3-moe/tests/mtp_spec_decode_perf_ab.rs:79`
- **Default (unset):** Use the model's GGUF `context_length` field
  (often 262144 for Qwen3.x).
- **Alternate (positive integer < model_ctx):** Clamp KV-cache
  provisioning to this token count. Save VRAM at the cost of max
  prompt+gen length.
- **Component:** `server::serve` boot path.
- **Cross-ref:** Not in `optimized.toml`; operator-config knob.

### FLAMBEAU_DECODE_GRAPH

- **Read sites:** `crates/models/qwen3-moe/src/forward/pp.rs:224`,
  `crates/models/qwen3-moe/src/forward/hybrid.rs:578`
- **Default (unset):** No graph-capture; replay disabled. Each
  decode pass issues fresh kernel launches.
- **Alternate (set, any value):** Capture each rank's per-(rank,
  lane) layer chain on first ubatch (PP) or per-stage (Hybrid),
  then replay. Iter-1 limitation: no slot binding (broken at N>1).
- **Component:** `models::qwen3_moe::forward::pp` and `…::hybrid`
  decode driver.
- **Cross-ref:** `optimized.toml` `[do_not_set]` — HALT-BROKEN
  (3/9 cells crash on hybrid).

### FLAMBEAU_DEFAULT_SYSTEM

- **Read sites:** `crates/server/src/serve.rs:401`
- **Default (unset):** No default system prompt; chat template's
  built-in default is used.
- **Alternate (string):** Boot-time default system prompt for chat
  requests with no `system` message. Empty string treated as unset.
- **Component:** `server::serve` boot config.
- **Cross-ref:** Operator-config; not in `optimized.toml`.

### FLAMBEAU_DENSE_GATE_UP

- **Read sites:** `crates/models/qwen3-moe/src/forward/dense_ffn.rs:165`,
  `crates/models/qwen3-moe/src/forward/dense_ffn_tp.rs:129`
- **Default (unset):** `specific_off = false`. Fused gate+up MMVQ
  kernel runs (Q8_0 / Q4_0 / Q4_1 dense-FFN paths).
- **Alternate (`=unfused`):** Disables the fused gate+up path —
  separate MMVQ kernels for gate and up. Combines with
  `FLAMBEAU_VARIANT=baseline` (which is the global-baseline opt-out).
- **Component:** `models::qwen3_moe::forward::dense_ffn` and `…_tp`.
- **Cross-ref:** `optimized.toml` `[do_not_set]` — HALT-DIVERGENT
  on TP2 (correctness bug in unfused-TP2 path).

### FLAMBEAU_EMBEDDING_MAX_TOKENS

- **Read sites:** `crates/server/src/serve.rs:544`
- **Default (unset):** 4096 tokens (clamped to [16, 32768]).
- **Alternate (integer):** Override the longest input the
  `/v1/embeddings` endpoint accepts.
- **Component:** `server::serve` embedding boot path.
- **Cross-ref:** Operator-config; not in `optimized.toml`.

### FLAMBEAU_FORCE_BATCH_WINDOW

- **Read sites:** `crates/server/src/routes.rs:833`
- **Default (unset):** `force_sleep = false`. Leader skips
  `BATCH_WINDOW_US` sleep when no other slot is active.
- **Alternate (set, any value):** Always sleep the full window even
  on single-user dispatches. Debug knob to make scheduler timing
  deterministic.
- **Component:** `server::routes::decode_via_scheduler_into` leader.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_GDN_NO_BATCHED

- **Read sites:** `crates/models/qwen3-moe/src/forward/hybrid.rs:1147`
- **Default (unset):** Batched-GDN path (one kernel call per stage
  spans all N slots).
- **Alternate (set, any value):** Per-slot legacy fallback —
  iterates `layer_states.iter_mut()` and calls per-slot. A/B
  regression knob.
- **Component:** `models::qwen3_moe::forward::hybrid` decode driver.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_GDN_QKV_FUSE_Q8_0

- **Read sites:** `crates/models/qwen3-moe/src/forward/gdn_tp.rs:229`
- **Default (unset):** `gdn_fuse_q8_0_off = false`. When weights
  are Q8_0, fuse `attn_qkv + attn_gate` into one mmvq_q8_0_gate_up
  call.
- **Alternate (`=off`):** Disables the fused kernel; runs two
  separate Q8_0 MMVQs.
- **Component:** `models::qwen3_moe::forward::gdn_tp` (TP-GDN
  path, hybrid arch only).
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_GPU_SAMPLER

- **Read sites:** `crates/server/src/routes.rs:3714`
- **Default (unset):** `use_gpu_sampler = false`. Sampler runs on
  host (full-vocab DtoH then partial-sort + softmax).
- **Alternate (set, any value):** Engages GPU top-K + softmax +
  apply-penalties kernel path. Only fires for TP/Hybrid models with
  non-greedy sampling.
- **Component:** `server::routes::run_completion_blocking_ids`
  decode loop.
- **Cross-ref:** `optimized.toml` `[env]` — `=1`. Production-on.

### FLAMBEAU_HOST_PROFILE

- **Read sites:** `crates/server/src/routes.rs:4577`
- **Default (unset):** No host-side per-section accumulator. Decode
  loop runs unprofiled.
- **Alternate (set, any value):** Per-section host wall accumulator
  (decode/mask/sample/emit/stopstr). Dumps one-line summary at end
  of decode loop.
- **Component:** `server::routes::run_completion_blocking_streaming`
  diagnostic.
- **Cross-ref:** Debug knob; not in `optimized.toml`.

### FLAMBEAU_INFLIGHT_SLOTS

- **Read sites:** `crates/server/src/serve.rs:424`
- **Default (unset):** 1 (clamped to [1, 32]).
- **Alternate (integer):** Number of pre-allocated `Inflight` slots
  in the pool. Each slot has its own KV cache + decode scratch
  (VRAM-linear).
- **Component:** `server::serve` boot config; consumed across
  `routes` (slot acquisition, scheduler).
- **Cross-ref:** `optimized.toml` `[env]` — `=4`. Production-on.

### FLAMBEAU_KV

- **Read sites:** `crates/models/qwen3-moe/src/session.rs:55`
- **Default (unset / "f16"):** `KvLayout::F16` — per-layer F16 KV
  cache.
- **Alternate (`=q8`/`=q8_0`/`=Q8`/`=Q8_0`):** `KvLayout::Q8` —
  per-layer Q8 KV cache (~2× HBM saving).
- **Component:** `models::qwen3_moe::session::Session::new` ctor.
  Propagates through every Inflight + decode kernel selection.
- **Cross-ref:** `optimized.toml` `[opt_in]` — set when memory-pressed.

### FLAMBEAU_KV_F16_DST

- **Read sites:** `crates/models/qwen3-moe/src/forward/attn_tp.rs:210`
- **Default (unset / not "off"):** Engages fused Q4_0 K+V matmul
  with F16-dest output (`mmvq_q4_0_kv_f16dst`). Only fires for
  Q4_0 K and V with equal row counts and `FLAMBEAU_VARIANT != baseline`.
- **Alternate (`=off`):** Disables fused K+V; falls back to
  separate K and V matmuls.
- **Component:** `models::qwen3_moe::forward::attn_tp` (TP only).
- **Cross-ref:** `optimized.toml` `[dispatch_table_candidates]`
  (TP2-only loss when off; null elsewhere).

### FLAMBEAU_MAX_CTX

- **Read sites:** `crates/models/qwen3-moe/src/config.rs:223`
- **Default (unset):** Use GGUF's `context_length` (model native).
- **Alternate (positive integer):** Clamp the parsed
  `Qwen3MoEConfig::context_length` (only shrinks).
- **Component:** `models::qwen3_moe::config::Qwen3MoEConfig::from_gguf`.
- **Cross-ref:** Operator-config; sibling of `FLAMBEAU_CTX_CAP`.

### FLAMBEAU_MAX_QUEUE_DEPTH

- **Read sites:** `crates/server/src/serve.rs:432`
- **Default (unset):** 16.
- **Alternate (integer; 0 disables):** Admission control cap.
  Total admitted requests = `inflight_slots + max_queue_depth`.
  Beyond that, returns 503 + Retry-After.
- **Component:** `server::serve` boot config; consumed in
  `ServerState::try_admit`.
- **Cross-ref:** Operator-config; not in `optimized.toml`.

### FLAMBEAU_MBATCH

- **Read sites:** `crates/ops/src/hip/moe.rs:1251`
- **Default (unset):** Standard indexed-MoE MMVQ Q4_K gate+up.
- **Alternate (set, any value):** Switches to llama.cpp-style
  top_k-warps mbatch kernel (`indexed_moe_mmvq_q4_k_gate_up_mbatch`).
- **Component:** `ops::hip::moe::indexed_moe_mmvq_q4_k_gate_up`
  dispatcher.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_MOE_SCATTER

- **Read sites:** `crates/ops/src/hip/moe.rs:1677`
- **Default (unset):** `flambeau_moe_sort_scatter_det` —
  deterministic single-thread scatter (TP-bit-reproducible).
- **Alternate (`=race`):** `flambeau_moe_sort_scatter` —
  racing-atomic scatter (faster on long prefills, breaks TP
  determinism).
- **Component:** `ops::hip::moe` sort scatter.
- **Cross-ref:** Not in `optimized.toml`. Production safe-default
  on; `=race` is a perf-risk opt-in.

### FLAMBEAU_MOE_SORTED

- **Read sites:** `crates/models/qwen3-moe/src/forward/moe.rs:985`
- **Default (unset):** Falls through to default `tile8` variant.
- **Alternate (`=0`):** Switches `moe_variant_cached` fallback to
  `r4` (when `FLAMBEAU_MOE_VARIANT` is also unset).
- **Component:** `models::qwen3_moe::forward::moe::moe_variant_cached`.
- **Cross-ref:** `optimized.toml` `[delete_candidates]` (legacy
  compat shortcut for MOE_VARIANT=r4 — confirmed loser).

### FLAMBEAU_MOE_VARIANT

- **Read sites:** `crates/models/qwen3-moe/src/forward/moe.rs:984`
- **Default (unset):** `moe_variant_cached()` returns `tile8`
  (production default, unless `MOE_SORTED=0` flips to `r4`).
- **Alternate (string):** Selects MoE prefill variant:
  - `tile8` (default), `sorted`, `r4`, `turbo`.
- **Component:** `models::qwen3_moe::forward::moe::moe_variant_cached`
  → consumed by every MoE prefill kernel selection.
- **Cross-ref:** `optimized.toml` `[dispatch_table_candidates]`
  (cert measured no win across 16 cells; tile8 default global).

### FLAMBEAU_MTP_BF16

- **Read sites:** `crates/models/qwen3-moe/src/mtp.rs:1296,1403,1513`
- **Default (unset):** F16/Q8_1 MTP forward path.
- **Alternate (truthy: not `""`/`0`/`off`/`false`):** Routes MTP
  step to `forward_mtp_step_bf16` (BF16 linears).
- **Component:** `models::qwen3_moe::mtp::forward_mtp_step*`
  (only invoked when `FLAMBEAU_SPEC_MTP` is configured).
- **Cross-ref:** Not in `optimized.toml`; spec-decode opt-in lever.

### FLAMBEAU_NO_FAST_PATH

- **Read sites:** `crates/server/src/routes.rs:757`
- **Default (unset):** Fast-path engages when `n_others_active==0`.
- **Alternate (set, any value):** Disables the fast-path branch in
  `decode_via_scheduler_into`. All decodes go through the batched
  leader path.
- **Component:** `server::routes::decode_via_scheduler_into`.
- **Cross-ref:** `optimized.toml` `[do_not_set]` — Debug-only,
  −2.0% with bit-identical output.

### FLAMBEAU_PREFILL_UBATCH

- **Read sites:** `crates/server/src/serve.rs:419`,
  `crates/server/src/routes.rs:332` (lock_tp_prefill_scratch),
  `crates/server/src/model.rs:135,525,621`
- **Default (unset):** 512 (clamped to ≥128).
- **Alternate (integer ≥128):** Tokens per chunked-prefill ubatch.
  Smaller → lower TTFT, higher per-token launch overhead. Larger
  → fewer launches but bigger transient scratch.
- **Component:** `server::serve` boot, `server::model::prefill_logits`
  (PP, TP, Hybrid chunked prefill), pooled-scratch sizing.
- **Cross-ref:** `optimized.toml` `[env]` — `=512`. Production-on.

### FLAMBEAU_PREFIX_CACHE

- **Read sites:** `crates/server/src/prefix_cache.rs:280`
  (`PrefixCache::enabled()`)
- **Default (unset / `""` / `=0`):** Disabled. Server still
  constructs the index but never reads/writes.
- **Alternate (any other value):** Engages the
  process-local prompt-prefix KV cache for chat-style workloads.
- **Component:** `server::prefix_cache::PrefixCache` gating; used
  by both streaming and non-streaming chat handlers.
- **Cross-ref:** `optimized.toml` `[opt_in]`.

### FLAMBEAU_PREFIX_CACHE_MAX_GB

- **Read sites:** `crates/server/src/prefix_cache.rs:270`
  (`PrefixCache::budget_from_env()`)
- **Default (unset):** 2.0 GB.
- **Alternate (float):** Cap on prefix-cache LRU size.
- **Component:** `server::prefix_cache::PrefixCache::new` budget.
- **Cross-ref:** `optimized.toml` `[opt_in]`; sibling of
  `FLAMBEAU_PREFIX_CACHE`.

### FLAMBEAU_PROFILE_DECODE

- **Read sites:** `crates/server/src/routes.rs:4567`
- **Default (unset):** 0 — no per-section HipEvent profiling.
- **Alternate (positive integer N):** Enable HipEvent recording for
  N decode steps after `profile_skip=8`. Flush and dump aggregate
  per-section ms to stderr.
- **Component:** `server::routes::run_completion_blocking_streaming`
  (CN-80B-22 TP-decode profiling).
- **Cross-ref:** Debug knob; not in `optimized.toml`.

### FLAMBEAU_Q4_0_GU_T128

- **Read sites:** `crates/models/qwen3-moe/src/forward/gdn_tp.rs:272`,
  `crates/models/qwen3-moe/src/forward/dense_ffn_tp.rs:165`
- **Default (unset):** Shape-aware: `symmetric (n_rows_gate ==
  n_rows_up)` → t128, asymmetric → 256t. Fired only on Q4_0
  fused gate+up paths.
- **Alternate values:**
  - `=on` — force 128 t/block.
  - `=off` — force 256 t/block.
- **Component:** `models::qwen3_moe::forward::{gdn_tp,dense_ffn_tp}`
  Q4_0 gate+up dispatch.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_Q4_0_GU_WARPCOOP

- **Read sites:** `crates/models/qwen3-moe/src/forward/gdn_tp.rs:270`,
  `crates/models/qwen3-moe/src/forward/dense_ffn_tp.rs:163`
- **Default (unset):** Standard 256 t/block kernel.
- **Alternate (`=on`):** Engages 64 t/block warpcoop kernel
  (`mmvq_q4_0_gate_up_warpcoop64`). C6 opt-in.
- **Component:** Same call sites as `Q4_0_GU_T128`.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_Q8_0_GU_T128_VDR2

- **Read sites:** `crates/ops/src/hip/qmatmul.rs:548`
  (in `mmvq_q8_0_gate_up`)
- **Default (unset / not "off"):** `t128_vdr2` schedule
  (`flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1`, threads=128).
- **Alternate (`=off`):** Reverts to baseline `dp4a` schedule
  (`flambeau_mmvq_q8_0_gate_up_dp4a_q8_1`, threads=256).
- **Component:** `ops::hip::qmatmul::mmvq_q8_0_gate_up`.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_Q8_0_MMVQ_T128

- **Read sites:** `crates/ops/src/hip/qmatmul.rs:988`
  (in `from_impl_id`)
- **Default (unset):** Off (don't engage the t128 lever).
- **Alternate (`=on`):** Routes Q8_0 single-row MMVQ to t128
  schedule.
- **Component:** `ops::hip::qmatmul::Recipe::from_impl_id`.
- **Cross-ref:** `optimized.toml` `[do_not_set]` — `=on` is
  −4.8% / −7.0% on Q8_0 across pp4/tp2.

### FLAMBEAU_Q8_0_MMVQ_T128_VDR2

- **Read sites:** `crates/ops/src/hip/qmatmul.rs:995`
  (in `from_impl_id`)
- **Default (unset / not "off"):** t128_vdr2 default (combined
  occupancy + VDR=2 lever).
- **Alternate (`=off`):** Opt out to pre-c9-followup vdr2.
  3-way test: `=off` is -5.1% on 27b-q8_0/tp2 vs unset; `=on` is
  null.
- **Component:** `ops::hip::qmatmul::Recipe::from_impl_id`.
- **Cross-ref:** `optimized.toml` `[dispatch_table_candidates]`
  (hidden branch divergence — investigate before deleting).

### FLAMBEAU_QKV_FUSED

- **Read sites:** `crates/models/qwen3-moe/src/forward/gdn.rs:1175`
- **Default (unset / not "0"):** Fused `gdn_split_qkv_f32` kernel
  (V2.4.d) replaces 3×L memcpy loop.
- **Alternate (`=0`):** Reverts to per-token `gather_qkv_strided`
  memcpy loop (regression compare only).
- **Component:** `models::qwen3_moe::forward::gdn` GDN prefill QKV
  split.
- **Cross-ref:** `optimized.toml` `[dispatch_table_candidates]`
  (wins +2.1% on 9B/pp2tp2, loses -3.6% on 9B/pp4 — model+topology
  dependent).

### FLAMBEAU_SPEC_MTP

- **Read sites:** `crates/server/src/serve.rs:207`
- **Default (unset / `""` / `=0` / `=off`):** No MTP head loaded;
  spec-decode disabled.
- **Alternate (path string):** Loads MTP head GGUF, attaches to
  last rank. Enables greedy spec-decode in chat handler.
- **Component:** `server::serve` boot path; consumed by chat
  handlers' spec-decode branch.
- **Cross-ref:** Operator-config; not in `optimized.toml`.

### FLAMBEAU_SSM_OUT_F16_DST

- **Read sites:** `crates/models/qwen3-moe/src/forward/gdn_tp.rs:609`
- **Default (unset / not "on"):** F32+cast pair (separate
  cast_f32_to_f16) for SSM output.
- **Alternate (`=on`):** Uses `mmvq_q5_k_r2_f16dst` kernel — fuses
  cast into kernel epilogue. Only fires when `ssm_out.dtype == Q5K`
  AND `FLAMBEAU_VARIANT != baseline`.
- **Component:** `models::qwen3_moe::forward::gdn_tp` SSM output.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

### FLAMBEAU_TP_BATCHED

- **Read sites:** `crates/models/qwen3-moe/src/forward/tp.rs:797,860`,
  `crates/models/qwen3-moe/src/forward/hybrid.rs:76`
- **Default (unset / not "0"):** Batched-TP prefill engages when
  `prompt_ids.len() >= 8` and no Q8 KV.
- **Alternate (`=0`):** Force per-token loop. **13× prefill
  regression** (149s vs 11s on Qwen3.6-27B-Q4_0/tp2).
- **Component:** `models::qwen3_moe::forward::tp` and `…::hybrid`
  prefill drivers.
- **Cross-ref:** `optimized.toml` `[do_not_set]` — KEEP-DEFAULT
  (alternate is purely dead).

### FLAMBEAU_TP_SKIP_SHARED

- **Read sites:** `crates/models/qwen3-moe/src/forward/tp.rs:1291`,
  `crates/models/qwen3-moe/src/forward/tp.rs:2163`,
  `crates/models/qwen3-moe/src/forward/tp.rs:3052`,
  `crates/models/qwen3-moe/src/forward/hybrid.rs:1349`
- **Default (unset):** Shared-expert path runs when
  `cfg.shared_expert_intermediate_size.is_some()`.
- **Alternate (set, any value):** Skip shared-expert (pure-MoE
  delta only — B5 bisect knob).
- **Component:** TP and Hybrid MoE drivers.
- **Cross-ref:** Debug bisect; not in `optimized.toml`. Disabling
  the shared expert is incorrect for arches that have one.

### FLAMBEAU_UBATCH

- **Read sites:** `crates/models/qwen3-moe/src/forward/pp.rs:1182`
  (also in tests).
- **Default (unset):** Falls back to `scratch.max_tokens`.
- **Alternate (positive integer):** Override async-prefill ubatch
  size. Only consulted when `FLAMBEAU_ASYNC_UBATCH` is set.
- **Component:** `models::qwen3_moe::forward::pp::forward_prefill_pp`.

### FLAMBEAU_VARIANT

Global kernel/forward variant override. Production-relevant read
sites (the same env is checked from many spots):

- **Read sites:**
  - `crates/ops/src/hip/qmatmul.rs:974` — `Recipe::from_impl_id`
    (selects MMVQ kernel: baseline / dp4a / llamacpp_style /
    q8_r4 / q4_1_wave64 / q4_1_tile16 / q4_k_r4 / q8_tile32).
  - `crates/ops/src/hip/moe.rs:129,1245` — MoE MMVQ
    (baseline / dp4a_r1 / dp4a_r2 / dp4a_r4 / dp4a_r8).
  - `crates/models/qwen3-moe/src/forward/attn.rs:356,540,1767,1901`
    — full-attn fuse_kv + split-K gates.
  - `crates/models/qwen3-moe/src/forward/attn_tp.rs:209,348` — TP
    fuse_kv + split-K.
  - `crates/models/qwen3-moe/src/forward/dense_ffn.rs:164` — dense
    FFN gate+up fuse.
  - `crates/models/qwen3-moe/src/forward/dense_ffn_tp.rs:128` —
    dense-FFN TP fuse.
  - `crates/models/qwen3-moe/src/forward/gdn.rs:310,355,479,569,1245`
    — GDN qkv+gate fuse, alpha+beta fuse, state_step fuse, tail fuse.
  - `crates/models/qwen3-moe/src/forward/gdn_tp.rs:222,346,488,569,608,1060,1364`
    — TP-GDN equivalents.
  - `crates/models/qwen3-moe/src/forward/moe.rs:478,529` — shared-expert
    gate+up fuse, swiglu+quant fuse.
  - `crates/models/qwen3-moe/src/forward/moe_tp.rs:124,245,286,733`
    — TP MoE swiglu+quant fuse, shared-expert fuse.

- **Default (unset / not "baseline"):** All "fused" / "default-on"
  paths engage (production hot path: dp4a / VDR=2 / Q8_0 fused /
  Q4_0 fused / GDN fused / MoE tile8).
- **Alternate values (`baseline`, `dp4a`, `dp4a_r1`, `dp4a_r2`,
  `dp4a_r4`, `dp4a_r8`, `dp4a_vdr2`, `llamacpp_style`, `q8_r4`,
  `q4_1_wave64`, `q4_1_tile16`, `q4_k_r4`, `q8_tile32`, `fused`):**
  Each forces a specific kernel path; `=baseline` reverts to the
  pre-DP4A scalar kernels.
- **Component:** Spans `ops::hip::{qmatmul,moe}` + the entire
  `models::qwen3_moe::forward` tree.
- **Cross-ref:** Not directly listed in `optimized.toml` (the
  default unset is what production runs), but conceptually it is
  the master switch the cumulative-optins cert references and that
  every per-fusion gate (`DENSE_GATE_UP`, `KV_F16_DST`,
  `GDN_QKV_FUSE_Q8_0`, `SSM_OUT_F16_DST`) ANDs against. Setting
  `=baseline` simultaneously disables ~25 fused paths.

### FLAMBEAU_ASYNC_GRAPH

- **Read sites:** `crates/models/qwen3-moe/src/forward/pp.rs:1477`
- **Default (unset):** Synchronous PP prefill — each rank>0's
  layer chain runs eagerly per ubatch.
- **Alternate (set, any value):** Capture rank>0's per-(rank,
  lane) layer chain on first ubatch and replay (V2.26.a-i5c).
- **Component:** `models::qwen3_moe::forward::pp::forward_prefill_pp_async`.
- **Cross-ref:** `optimized.toml` `[dispatch_table_candidates]`
  (loses -2.4% on 35B-A3B/pp4; null elsewhere).

### FLAMBEAU_ASYNC_UBATCH

- **Read sites:** `crates/models/qwen3-moe/src/forward/pp.rs:1179`
- **Default (unset):** Synchronous PP prefill at ubatch
  granularity.
- **Alternate (set, any value):** Engages async PP prefill.
  Requires `u_lanes >= 2` from `Scratch::new_with_lanes`.
- **Component:** `models::qwen3_moe::forward::pp` prefill driver.
- **Cross-ref:** `optimized.toml` `[delete_candidates]`.

---

## Section 2 — Debug / dump vars

These all gate `if std::env::var(...).is_ok() { ... print/dump ... }`
blocks. The "default" arm is a no-op; the alternate arm side-effects
(stderr, file dump) without affecting forward-path correctness.

| Var | Read sites |
|---|---|
| `FLAMBEAU_AR_DUMP` | `models/qwen3-moe/src/forward/hybrid.rs:1214` |
| `FLAMBEAU_BATCHED_DECODE_DUMP` | `models/qwen3-moe/src/forward/hybrid.rs:1472` |
| `FLAMBEAU_DEBUG_TOOL_RAW` | `server/src/routes.rs:1811` |
| `FLAMBEAU_DUMP_PROMPT` | `server/src/routes.rs:1711,1785` |
| `FLAMBEAU_DUMP_RAW_REQ` | `server/src/routes.rs:1493` |
| `FLAMBEAU_GRAPH_TRACE` | `backend-hip/src/device.rs:624` |
| `FLAMBEAU_KV_PROJ_DUMP` | `models/qwen3-moe/src/forward/attn_tp.rs:878` |
| `FLAMBEAU_KV_ROPE_DUMP` | `models/qwen3-moe/src/forward/attn_tp.rs:937` |
| `FLAMBEAU_LAYER_STATE_DUMP` | `models/qwen3-moe/src/forward/hybrid.rs:1515` |
| `FLAMBEAU_LOAD_TRACE` | `models/qwen3-moe/src/sharded.rs:481,1126`, `quant/src/gguf.rs:367` |
| `FLAMBEAU_PARITY_LAYER_DUMP` | `models/qwen3-moe/src/forward/{pp.rs:476,1323; layer.rs:413; tp.rs:2189}` |
| `FLAMBEAU_PARITY_TOPK_LOGITS` | `models/qwen3-moe/src/forward/io.rs:445` |
| `FLAMBEAU_PP_PROBE` | `models/qwen3-moe/src/forward/pp.rs:662` |
| `FLAMBEAU_STAGE_ENTRY_DUMP` | `models/qwen3-moe/src/forward/hybrid.rs:1002` |
| `FLAMBEAU_TP_LAYER0_BISECT` | `models/qwen3-moe/src/forward/{layer.rs:301; tp.rs:2188,2284,2304,2672}` |
| `FLAMBEAU_TP_LAYER_LIMIT` | `models/qwen3-moe/src/forward/tp.rs:1607` |
| `FLAMBEAU_TP_PROBE` | `models/qwen3-moe/src/forward/{tp.rs:1611,1760,1808,1834,2516; gdn_tp.rs:162}` |
| `FLAMBEAU_TRACE_BATCH` | `server/src/routes.rs:758,929` |

All are debug-class. Cross-ref: none in `optimized.toml`.

---

## Section 3 — Test / example / bench-only vars

These are read exclusively from `crates/*/tests/*.rs` and
`crates/*/examples/*.rs` — never reached by the production
`flambeau serve` binary. They drive parity tests, perf benches,
and dev-only A/B harnesses.

| Var | Used by tests/examples |
|---|---|
| `FLAMBEAU_AB_ONLY` | `coder_next_pp2tp2_decode_graph_ab` |
| `FLAMBEAU_AB_STEPS` | `examples/ab_decode.rs` |
| `FLAMBEAU_A_DEVICES`, `FLAMBEAU_B_DEVICES`, `FLAMBEAU_A_VARIANT`, `FLAMBEAU_B_VARIANT` | `examples/ab_decode.rs` |
| `FLAMBEAU_BENCH_CTX_LENGTHS` | `q8_kv_sustained_decode_bench` |
| `FLAMBEAU_BENCH_GGUF` | `v1_bench_matrix`, `bench_tp2_any_model` |
| `FLAMBEAU_BENCH_SCRATCH_TOKENS` | `v1_bench_matrix` |
| `FLAMBEAU_BENCH_SKIP_TP` | `v1_bench_matrix` |
| `FLAMBEAU_BENCH_TAG` | `v1_bench_matrix` |
| `FLAMBEAU_BENCH_TOPOLOGY_TAG` | `v1_bench_matrix` |
| `FLAMBEAU_BENCH_WARM_TOKEN` | `forward_one_token_pp_real` |
| `FLAMBEAU_DECODE_ONLY` | `perf_baseline_qwen35_9b` |
| `FLAMBEAU_DEVICES` | `examples/decode_profile.rs` |
| `FLAMBEAU_FUSE_AB_ONLY` | `coder_next_pp2tp2_decode_fuse_ab` |
| `FLAMBEAU_HYBRID` | `parity_hybrid_qwen35_9b` |
| `FLAMBEAU_HYBRID_DEVICES` | `parity_hybrid_qwen35_9b`, `parity_tp_batched_prefill` |
| `FLAMBEAU_LOB_ITERS`, `FLAMBEAU_LOB_SYNC_EVERY` | `examples/launch_overhead_bench.rs` |
| `FLAMBEAU_LONG_TEXT` | `perf_baseline_qwen3_moe` |
| `FLAMBEAU_MESH_RANKS` | `decode_profile`, `perf_baseline_qwen35_9b`, `perf_baseline_qwen3_moe` |
| `FLAMBEAU_MTP_ACCEPT_STEPS`, `FLAMBEAU_MTP_BASE`, `FLAMBEAU_MTP_HEAD`, `FLAMBEAU_MTP_KV_ACCUM`, `FLAMBEAU_MTP_PREFILL_PRIME`, `FLAMBEAU_MTP_PROMPT` | `mtp_acceptance_passive` |
| `FLAMBEAU_PERF_AB_TOKENS`, `FLAMBEAU_PP_RANKS` | `mtp_spec_decode_perf_ab` |
| `FLAMBEAU_PP_LAYERS` | `perf_baseline_qwen3_moe` |
| `FLAMBEAU_PREFILL_L`, `FLAMBEAU_PREFILL_ONLY`, `FLAMBEAU_TG_LEN` | `perf_baseline_qwen3_moe` |
| `FLAMBEAU_PROFILE_BASE_STEPS`, `FLAMBEAU_PROFILE_SPEC_MACROS` | `mtp_spec_decode_profile` |
| `FLAMBEAU_PROFILE_GGUF`, `FLAMBEAU_PROFILE_L`, `FLAMBEAU_PROFILE_TG`, `FLAMBEAU_PROFILE_TP_DEVICES`, `FLAMBEAU_PROFILE_TP_SIZE`, `FLAMBEAU_PROFILE_STEPS` | `profile_point_tp`, `profile_tp_decode`, `decode_profile` |
| `FLAMBEAU_PROFILE_MESH` | `profile_point` |
| `FLAMBEAU_QWEN3_GGUF`, `FLAMBEAU_QWEN35_GGUF`, `FLAMBEAU_QWEN36_GGUF` | many (~30 tests) — fixture-path env |
| `FLAMBEAU_RESULT_JSON` | `examples/decode_profile.rs` |
| `FLAMBEAU_SPEC_MACRO_STEPS` | `mtp_spec_decode_smoke`, `mtp_spec_decode_sampling_smoke` |
| `FLAMBEAU_TEST_L`, `FLAMBEAU_TEST_CHUNK` | `chunked_prefill_kv_parity_tp` |
| `FLAMBEAU_TOPOLOGY_COMPARE`, `FLAMBEAU_TOPOLOGY_TAG` | `topology_compare_qwen35_9b` |
| `FLAMBEAU_TP_BATCHED_PARITY` | `parity_tp_batched_prefill` |
| `FLAMBEAU_TP_DEVICES`, `FLAMBEAU_TP_RANKS` | `bench_tp2_any_model`, `mtp_spec_decode_tp_perf_ab` |
| `FLAMBEAU_U_LANES` | `perf_baseline_qwen35_9b`, `perf_baseline_qwen3_moe` |

All test-only. Cross-ref: none in `optimized.toml`.

A few of the same names ALSO have a production read site
(`FLAMBEAU_CTX_CAP`, `FLAMBEAU_UBATCH`, `FLAMBEAU_VARIANT`,
`FLAMBEAU_ASYNC_UBATCH`); those are listed in §1 and the test
read sites are non-load-bearing duplicates.

---

## Summary table

Production vars first (sorted by category), then debug, then test-only.
Preliminary verdict ∈ {**KEEP**, **DISPATCH**, **DELETE**, **DELETE-DEAD-PATH**, **HALT**, **DEBUG-FEATURE**, **TEST-ONLY**}.

| Var | Read sites | optimized.toml | Preliminary verdict |
|---|---:|---|---|
| FLAMBEAU_AR_FUSE_Q8_1 | 2 | delete_candidates | DELETE |
| FLAMBEAU_BATCHED_DECODE | 1 | env | KEEP (production-on; single read) |
| FLAMBEAU_BATCH_MAX | 1 | delete_candidates | DELETE |
| FLAMBEAU_BATCH_WINDOW_US | 1 | — | DELETE-DEAD-PATH (default 1500 µs is fine) |
| FLAMBEAU_BATCHED_MMVQ | 1 | — | DISPATCH (Q4_1 batched MMVQ — shape-aware) |
| FLAMBEAU_CTX_CAP | 1 | — | KEEP (operator clamp; promote to clap arg) |
| FLAMBEAU_DECODE_GRAPH | 2 | do_not_set | DELETE-DEAD-PATH (HALT-BROKEN) |
| FLAMBEAU_DEFAULT_SYSTEM | 1 | — | KEEP (operator config; promote to clap arg) |
| FLAMBEAU_DENSE_GATE_UP | 2 | do_not_set | DELETE-DEAD-PATH (HALT-DIVERGENT) |
| FLAMBEAU_EMBEDDING_MAX_TOKENS | 1 | — | KEEP (promote to clap arg) |
| FLAMBEAU_FORCE_BATCH_WINDOW | 1 | delete_candidates | DELETE |
| FLAMBEAU_GDN_NO_BATCHED | 1 | delete_candidates | DELETE |
| FLAMBEAU_GDN_QKV_FUSE_Q8_0 | 1 | delete_candidates | DELETE |
| FLAMBEAU_GPU_SAMPLER | 1 | env | KEEP (production-on) |
| FLAMBEAU_HOST_PROFILE | 1 | — | DEBUG-FEATURE (gate behind `cfg(dev_trace)`) |
| FLAMBEAU_INFLIGHT_SLOTS | 1 | env | KEEP (operator config; promote to clap arg) |
| FLAMBEAU_KV | 1 | opt_in | KEEP (operator quant choice; promote to clap arg) |
| FLAMBEAU_KV_F16_DST | 1 | dispatch_table_candidates | DISPATCH |
| FLAMBEAU_MAX_CTX | 1 | — | KEEP-DUPLICATE (sibling of CTX_CAP — collapse to one) |
| FLAMBEAU_MAX_QUEUE_DEPTH | 1 | — | KEEP (promote to clap arg) |
| FLAMBEAU_MBATCH | 1 | delete_candidates | DELETE |
| FLAMBEAU_MOE_SCATTER | 1 | — | DELETE (default `det` is correct; race-mode wins are TP-incorrect) |
| FLAMBEAU_MOE_SORTED | 1 | delete_candidates | DELETE |
| FLAMBEAU_MOE_VARIANT | 1 | dispatch_table_candidates | DISPATCH |
| FLAMBEAU_MTP_BF16 | 3 | — | DELETE-DEAD-PATH (only matters when SPEC_MTP set; F16 is production default) |
| FLAMBEAU_NO_FAST_PATH | 1 | do_not_set | DELETE-DEAD-PATH |
| FLAMBEAU_PREFILL_UBATCH | 5 | env | KEEP (operator config; **promote to single shared resolution + clap arg**) |
| FLAMBEAU_PREFIX_CACHE | 1 | opt_in | KEEP (operator config; promote to clap arg) |
| FLAMBEAU_PREFIX_CACHE_MAX_GB | 1 | opt_in | KEEP (sibling of PREFIX_CACHE) |
| FLAMBEAU_PROFILE_DECODE | 1 | — | DEBUG-FEATURE (gate behind `cfg(dev_trace)`) |
| FLAMBEAU_Q4_0_GU_T128 | 2 | delete_candidates | DELETE |
| FLAMBEAU_Q4_0_GU_WARPCOOP | 2 | delete_candidates | DELETE |
| FLAMBEAU_Q8_0_GU_T128_VDR2 | 1 | delete_candidates | DELETE |
| FLAMBEAU_Q8_0_MMVQ_T128 | 1 | do_not_set | DELETE-DEAD-PATH |
| FLAMBEAU_Q8_0_MMVQ_T128_VDR2 | 1 | dispatch_table_candidates | DISPATCH (after investigating off-vs-unset divergence) |
| FLAMBEAU_QKV_FUSED | 1 | dispatch_table_candidates | DISPATCH |
| FLAMBEAU_SPEC_MTP | 1 | — | KEEP (operator config; promote to clap arg) |
| FLAMBEAU_SSM_OUT_F16_DST | 1 | delete_candidates | DELETE |
| FLAMBEAU_TP_BATCHED | 3 | do_not_set | DELETE-DEAD-PATH (alternate is 13× slow path) |
| FLAMBEAU_TP_SKIP_SHARED | 4 | — | DEBUG-FEATURE (B5 bisect; gate behind `cfg(dev_trace)`) |
| FLAMBEAU_UBATCH | 1 (+ tests) | — | DELETE (only consulted when ASYNC_UBATCH set; both die together) |
| FLAMBEAU_VARIANT | ~25 | — | KEEP-MASTER-SWITCH then DISPATCH-MIGRATE (touches everything; the cleanup target itself) |
| FLAMBEAU_ASYNC_GRAPH | 1 | dispatch_table_candidates | DISPATCH |
| FLAMBEAU_ASYNC_UBATCH | 1 | delete_candidates | DELETE |
| --- DEBUG vars (18) --- | | | DEBUG-FEATURE (move behind `cfg(dev_trace)` feature flag) |
| FLAMBEAU_AR_DUMP, FLAMBEAU_BATCHED_DECODE_DUMP, FLAMBEAU_DEBUG_TOOL_RAW, FLAMBEAU_DUMP_PROMPT, FLAMBEAU_DUMP_RAW_REQ, FLAMBEAU_GRAPH_TRACE, FLAMBEAU_KV_PROJ_DUMP, FLAMBEAU_KV_ROPE_DUMP, FLAMBEAU_LAYER_STATE_DUMP, FLAMBEAU_LOAD_TRACE, FLAMBEAU_PARITY_LAYER_DUMP, FLAMBEAU_PARITY_TOPK_LOGITS, FLAMBEAU_PP_PROBE, FLAMBEAU_STAGE_ENTRY_DUMP, FLAMBEAU_TP_LAYER0_BISECT, FLAMBEAU_TP_LAYER_LIMIT, FLAMBEAU_TP_PROBE, FLAMBEAU_TRACE_BATCH | | | |
| --- TEST-ONLY vars (~30) --- | | | TEST-ONLY (move to `BenchConfig` / test fixtures; out of binary) |
| (see Section 3) | | | |

### Triage breakdown

- **KEEP (production-on, promote to clap or stays as env):** 11
  (BATCHED_DECODE, GPU_SAMPLER, INFLIGHT_SLOTS, PREFILL_UBATCH,
  PREFIX_CACHE, PREFIX_CACHE_MAX_GB, KV, CTX_CAP, MAX_QUEUE_DEPTH,
  DEFAULT_SYSTEM, EMBEDDING_MAX_TOKENS, SPEC_MTP, MAX_CTX —
  the last two collapse with siblings).
- **DISPATCH (move to dispatch table):** 6
  (KV_F16_DST, MOE_VARIANT, Q8_0_MMVQ_T128_VDR2, QKV_FUSED,
  ASYNC_GRAPH, BATCHED_MMVQ).
- **DELETE (default-bake, remove read site):** 14
  (AR_FUSE_Q8_1, BATCH_MAX, BATCH_WINDOW_US, FORCE_BATCH_WINDOW,
  GDN_NO_BATCHED, GDN_QKV_FUSE_Q8_0, MBATCH, MOE_SCATTER,
  MOE_SORTED, Q4_0_GU_T128, Q4_0_GU_WARPCOOP, Q8_0_GU_T128_VDR2,
  SSM_OUT_F16_DST, ASYNC_UBATCH; plus UBATCH which dies with
  ASYNC_UBATCH).
- **DELETE-DEAD-PATH (alternate is broken/slow/divergent — delete
  alt code as well):** 6
  (DECODE_GRAPH, DENSE_GATE_UP unfused, MTP_BF16, NO_FAST_PATH,
  Q8_0_MMVQ_T128, TP_BATCHED).
- **DEBUG-FEATURE (gate behind `cfg(dev_trace)`):** 18 (see
  Section 2) + HOST_PROFILE + PROFILE_DECODE + TP_SKIP_SHARED + 4
  bisect/probe vars.
- **TEST-ONLY (out of binary):** ~30 (see Section 3).
- **MASTER-SWITCH (handle last):** 1 (FLAMBEAU_VARIANT — the
  global fuse override). Each `FLAMBEAU_VARIANT != "baseline"`
  guard expansion can be statically simplified once we commit to
  "no global baseline opt-out", reducing ~25 fused-path branches
  to one path.

### What's NOT on this list

Anything not appearing in the §1/§2/§3 grep is not an env var
flambeau reads. Tests that consult `FLAMBEAU_QWEN3_GGUF` etc. are
fixture-path env, not behavior knobs. The `FLAMBEAU_TP_BATCHED_PARITY`
and `FLAMBEAU_HYBRID` test gates control whether parity tests run
at all.

