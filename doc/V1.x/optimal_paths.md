# Optimal Forward-Path Map per Anchor

**Phase 0a deliverable** of `code_path_cleanup_plan.md`. Read-only
audit at `main` HEAD `d7c3cc6`. Walks the actual call sequence each
anchor (model × topology × N-class) traverses under
`bench/profiles/optimized.toml`. Anything not visited by any cell here
is a Phase 1 deletion candidate.

## Anchor configuration

`bench/profiles/optimized.toml` `[env]`:
- `FLAMBEAU_BATCHED_DECODE=1` — gates `scheduler_can_engage` →
  `run_completion_blocking_ids` re-routes to the scheduler handler.
- `FLAMBEAU_GPU_SAMPLER=1` — runtime checked at chat handler.
- `FLAMBEAU_INFLIGHT_SLOTS=4` (4 or 8 in prod) — slot-pool size.
- `FLAMBEAU_PREFILL_UBATCH=512` — chunk size in `prefill_logits`.
- `RUST_LOG=warn`.

`FLAMBEAU_VARIANT` is unset → all `!= Ok("baseline")` checks fall on
the **fused** branch in attn / GDN / MoE / dense-FFN.

The race-safe fast-path lock at `routes.rs:771` is in. Streaming chat
calls `decode_logits` directly and bypasses the scheduler at this
commit; non-streaming chat re-routes through
`run_completion_scheduler_pp_blocking` which uses
`decode_via_scheduler_into`.

## Anchor models

| model | arch | dtype | dense_ffn | recurrent layers |
|---|---|---|---|---|
| Qwen3.5-9B-Q4_1 | `qwen35` | Q4_1 | yes | none (`full_attention_interval` ≤ 1) |
| Qwen3.6-27B-Q4_0 | `qwen36moe` | Q4_0 | no | hybrid GDN + full-attn |
| Qwen3.6-35B-A3B-Q4_0 | `qwen36moe` | Q4_0 | no | hybrid GDN + full-attn |
| Qwen3-Coder-Next-80B-Q4_0 | `qwen3next` | Q4_0 | no | hybrid GDN + full-attn |

Qwen3.6-27B / 35B-A3B / Coder-Next-80B share the **same forward
modules** (`crates/models/qwen3-moe/src/forward/{gdn,attn,moe,
dense_ffn,layer,...}.rs`). The only difference between cells of these
three under the same topology is hyperparameters (hidden, n_layers,
n_experts) — the kernel chain is identical.

Qwen3.5-9B at `qwen35` arch loads through `is_dense_ffn() == true` and
`is_recurrent(il) == false` for every layer. Its forward path is
**full-attn + dense-FFN** with no GDN, no MoE, no shared expert.

## Cells

For each (model-class × topology × N-class) below, a single
ASCII-tree walk from server entry to leaf kernels. Branches
visited under the anchor config; alternates noted in `[gate=…]`
brackets (those alternates are deletion candidates if no anchor cell
selects them).

---

### A. Qwen3.5-9B (qwen35 dense) × `pp4` × N=1 (fast-path)

Streaming chat path:

```
routes::run_completion_blocking_streaming
├─ acquire_inflight_blocking
├─ prefill_logits  (no _prefill_lock — PP-only, see #321 gate)
│   └─ model::prefill_logits @ Inflight::Pp arm
│       └─ chunked over FLAMBEAU_PREFILL_UBATCH (512):
│           └─ pp::forward_prefill_pp_logits  (final chunk)
│               └─ pp::forward_prefill_pp     (non-final, with on_boundary)
│                   └─ per-rank loop:
│                       └─ layer::forward_layer_prefill (×n_layers/rank)
│                           ├─ attn::forward_full_attn_prefill   [is_recurrent=false always]
│                           └─ dense_ffn::forward_dense_ffn_prefill   [is_dense_ffn=true]
│
└─ for step in 1..max_tokens:
    └─ model::decode_logits @ Inflight::Pp arm
        └─ pp::forward_one_token_pp_logits → forward_one_token_pp_inner
            ├─ rank-0 io::forward_embed_decode_host
            ├─ per-rank loop:
            │   ├─ peer_copy_via_host (rank>0)
            │   └─ layer::forward_layer_decode (×n_layers/rank)
            │       ├─ attn::forward_full_attn_layer_decode  [AttnWeights gated; qwen35 → here]
            │       │   └─ rmsnorm + fused QKV + RoPE + softmax-attn + output proj
            │       └─ dense_ffn::forward_dense_ffn_decode   [is_dense_ffn=true]
            │           └─ gate/up/down with optional fused gate+up
            │               [FLAMBEAU_VARIANT, FLAMBEAU_DENSE_GATE_UP — anchor: fused]
            └─ last-rank io::forward_output_head_decode + DtoH logits
```

Non-streaming path adds `state.claim_slot_blocking` →
`run_completion_scheduler_pp_blocking` → `decode_via_scheduler_into`.
At N=1 `n_others_active==0`, so the fast-path branch fires: takes
`prefill_serialiser`, calls `model::decode_logits` (same leaf as
streaming).

### B. Qwen3.5-9B × `tp2` × N=1 (fast-path)

```
streaming: prefill_logits → model::prefill_logits @ Inflight::Tp arm
  └─ holds prefill_serialiser (#321) + tp_prefill_scratch (#324)
  └─ chunked over FLAMBEAU_PREFILL_UBATCH:
      └─ tp::forward_prefill_tp_logits_pooled
          [FLAMBEAU_TP_BATCHED unset → batched on; len ≥ 8 → batched driver]
          └─ tp::forward_prefill_tp_batched_logits → per-layer batched:
              ├─ attn_tp::forward_full_attn_layer_decode_batched_tp
              └─ dense_ffn_tp::forward_dense_ffn_decode_tp     [batched-L]

decode: model::decode_logits @ Tp → tp::forward_one_token_tp_logits → _inner
  ├─ per-layer loop:
  │   ├─ attn_tp::forward_full_attn_layer_tp  (rmsnorm-AR-fused QKV-RoPE-attn-AR)
  │   │   └─ kv_q4_0_fused: FLAMBEAU_KV_F16_DST != "off" AND VARIANT != baseline → fused
  │   └─ dense_ffn_tp::forward_dense_ffn_decode_tp
  │       ├─ FLAMBEAU_VARIANT/FLAMBEAU_DENSE_GATE_UP=unfused → unfused
  │       └─ default (fused): FLAMBEAU_Q4_0_GU_T128 / WARPCOOP gate kernel variants
  └─ io::forward_output_head_decode + DtoH
```

### C. Qwen3.5-9B × `pp2tp2` × N=1 (fast-path)

```
hybrid::forward_one_token_hybrid_logits → _inner
  ├─ stage 0 embed (rank 0 of stage 0)
  ├─ for stage in stages:
  │   ├─ FLAMBEAU_DECODE_GRAPH=1: HALT — broken on hybrid, anchor leaves unset
  │   └─ run_layer_loop:
  │       └─ for il in stage.layer_range:
  │           ├─ is_recurrent → forward_gdn_layer_tp   [unreachable on qwen35 dense]
  │           └─ else        → forward_full_attn_layer_tp  + dense ffn block (tp.rs:1909)
  ├─ inter-stage peer_copy_via_host of [hidden] F16
  └─ head stage: io::forward_output_head_decode + DtoH
```

(All Qwen3.5-9B / pp2tp2 layers go to the full-attn + dense-FFN branch.)

### D. Qwen3.5-9B × {pp4, tp2, pp2tp2} × N≥2 (scheduler)

Non-streaming → `run_completion_scheduler_pp_blocking`. Streaming
stays per-step direct — but `acquire_inflight_blocking` serializes
slots so concurrent streams don't actually batch in the streaming
chat path. Below is the scheduler path:

```
decode_via_scheduler_into [N≥2 → n_others_active>0]
  └─ batched_pending.push(PendingDecode); leader: dispatch_batched_pending
      ├─ acquire prefill_serialiser (PR2.5)
      ├─ blocking_lock all referenced slot mutexes
      └─ match LoadedModel:
          ├─ Pp     → batched::forward_decode_batched_pp
          │   └─ per-rank: forward_layer_decode_batched
          │       ├─ attn::forward_full_attn_layer_decode_batched
          │       └─ dense_ffn::forward_dense_ffn_decode  (per-slot in loop)
          ├─ Tp     → tp::forward_decode_batched_tp
          │   └─ per-layer:
          │       ├─ attn_tp::forward_full_attn_layer_decode_batched_tp
          │       └─ dense_ffn_tp::forward_dense_ffn_decode_tp  (batched n_tokens=N)
          └─ Hybrid → hybrid::forward_decode_batched_hybrid
              └─ per-stage: full-attn-batched + dense-ffn-batched
```

### E. Qwen3.6-27B / 35B-A3B / Coder-Next-80B × `pp4` × N=1 (fast-path)

Streaming + non-streaming-fast-path leaf:

```
pp::forward_one_token_pp_logits → forward_one_token_pp_inner
  ├─ rank-0 io::forward_embed_decode_host
  ├─ per-rank loop:
  │   ├─ peer_copy_via_host (rank>0)
  │   └─ layer::forward_layer_decode (×n_layers/rank):
  │       ├─ if cfg.is_recurrent(il):
  │       │     gdn::forward_gdn_layer_decode
  │       │     ├─ FLAMBEAU_QKV_FUSED=0 → unfused QKV [anchor: unset → fused]
  │       │     └─ FLAMBEAU_VARIANT={fuse_alpha_beta, fuse_state_step, fuse_tail}
  │       │       all default to fused when VARIANT != baseline
  │       └─ else (full-attn):
  │             match AttnWeights:
  │               Dense → forward_dense_attn_layer_decode  [qwen3moe; not in anchors]
  │               _     → attn::forward_full_attn_layer_decode
  │                       └─ fuse_kv: VARIANT != baseline → fused
  │       (FFN selection inside forward_layer_decode):
  │       ├─ shared_expert (qwen36moe/qwen3next): moe::forward_shared_expert_decode
  │       ├─ moe::forward_router_decode
  │       └─ moe::forward_moe_ffn_decode
  │             [FLAMBEAU_MOE_VARIANT (default sorted/tile8 per dispatch);
  │              FLAMBEAU_MOE_SORTED=0 → off path; anchor: tile8 default]
  └─ last-rank io::forward_output_head_decode + DtoH
```

### F. Qwen3.6-27B / 35B-A3B / Coder-Next-80B × `tp2` × N=1 (fast-path)

```
tp::forward_one_token_tp_logits → forward_one_token_tp_inner (LogitsSink::HostLogits)
  ├─ embed (rank-broadcast)
  ├─ per-layer:
  │   ├─ if is_recurrent(il):
  │   │     gdn_tp::forward_gdn_layer_tp
  │   │     ├─ FLAMBEAU_GDN_QKV_FUSE_Q8_0=off → unfused [anchor: unset → fused]
  │   │     ├─ Q4_0 gate-up: FLAMBEAU_Q4_0_GU_T128, FLAMBEAU_Q4_0_GU_WARPCOOP
  │   │     ├─ fuse_alpha_beta / fuse_state_step / fuse_tail: VARIANT-gated, default fused
  │   │     └─ ssm_out_f16_dst: FLAMBEAU_SSM_OUT_F16_DST=on → opt-in fusion
  │   └─ else: attn_tp::forward_full_attn_layer_tp
  │             ├─ kv_q4_0_fused: VARIANT != baseline AND KV_F16_DST != off → fused
  │             └─ FLAMBEAU_AR_FUSE_Q8_1: opt-in AR+Q8_1 fusion (default off)
  │   (FFN block at tp.rs:1909):
  │   ├─ shared expert: moe_tp::forward_shared_expert_decode_tp
  │   │     └─ fuse_shexp_swiglu_quant: VARIANT != baseline
  │   ├─ router (replicated)
  │   └─ moe_tp::forward_moe_ffn_decode_tp
  │         ├─ fuse_swiglu_quant: VARIANT != baseline
  │         └─ fuse_gate_up:      VARIANT != baseline
  └─ output head + DtoH (or KeepOnDevice at GPU sampler)
```

### G. Qwen3.6-27B / 35B-A3B / Coder-Next-80B × `pp2tp2` × N=1 (fast-path)

```
hybrid::forward_one_token_hybrid_logits → _inner
  ├─ stage 0 embed
  ├─ for stage in stages: run_layer_loop
  │   ├─ FLAMBEAU_DECODE_GRAPH=1: graph capture (HALT-BROKEN; anchor leaves unset)
  │   └─ for il in stage.layer_range:
  │       ├─ is_recurrent → tp::forward_gdn_layer_tp
  │       └─ else        → tp::forward_full_attn_layer_tp
  │       (FFN — implicit inside forward_full_attn_layer_tp's tail at tp.rs:2086,
  │        same MoE+shared chain as cell F)
  ├─ inter-stage peer_copy_via_host (one [hidden] F16 per hop)
  └─ head stage: forward_output_head_decode + DtoH
```

### H. Qwen3.6-27B / 35B-A3B / Coder-Next-80B × {pp4, tp2, pp2tp2} × N≥2 (scheduler)

```
dispatch_batched_pending → match LoadedModel:
├─ Pp     → batched::forward_decode_batched_pp
│   └─ per-rank: layer::forward_layer_decode_batched
│       ├─ is_recurrent → gdn::forward_gdn_layer_decode  (per-slot loop, not batched)
│       └─ else → attn::forward_full_attn_layer_decode_batched
│       (FFN: per-slot dense_ffn_decode loop or moe path — see batched.rs:289+)
├─ Tp     → tp::forward_decode_batched_tp
│   └─ per-layer:
│       ├─ is_recurrent → forward_gdn_layer_tp (per-slot loop) [no batched-GDN kernel]
│       └─ else → attn_tp::forward_full_attn_layer_decode_batched_tp
│       └─ FFN: prefill-flavoured kernels at n_tokens=N (dense_ffn_decode_tp /
│              moe_ffn_decode_tp accept N>1)
└─ Hybrid → hybrid::forward_decode_batched_hybrid
    └─ per-stage: same pattern as Tp inside each stage's sub_cluster
```

GDN at N≥2 still loops per-slot — no batched-GDN kernel exists
(`#288` in memory). Aggregate scaling is bounded by GDN per-slot
ceiling under hybrid arch.

## Common subpaths

To avoid repeating the same chain in every cell, these leaves are
shared across all anchors:

### S1. Embed (decode, single token)
`io::forward_embed_decode_host` (host-driven gather; rank 0 of stage
0 owns the embedding table).

### S2. Full-attention decode (single-device / PP rank-local)
`attn::forward_full_attn_layer_decode`:
1. rmsnorm (input)
2. fused QKV proj (`FLAMBEAU_QKV_FUSED=0` opt-out [Phase 1: dead?])
3. partial-NeoX RoPE (Q + K)
4. KV append → KV cache (F16 default; Q8 alt path exists)
5. softmax-attention (GQA, masked, fused-Q8 alt at FLAMBEAU_VARIANT
   non-baseline)
6. output projection (rmsnorm-fused at non-baseline VARIANT)

### S3. Full-attention TP layer
`attn_tp::forward_full_attn_layer_tp`:
- Step 1–2: rmsnorm-fused QKV, KV-quant fused if VARIANT != baseline
  AND `KV_F16_DST` != off.
- Step 3: AR (BarP2pAllReduce) of K|V partial outputs.
- Step 4: attention compute on rank-local KV slice.
- Step 5: AR of output projection (with `AR_FUSE_Q8_1` opt-in
  fusing the AR + Q8_1 quantize into one kernel).

### S4. GDN decode (single-device / rank-local)
`gdn::forward_gdn_layer_decode`:
1. rmsnorm + Q,K,V,β,α projections (`QKV_FUSED` opt-out for Q,K,V
   fusion — anchors keep fused).
2. SSM gates: alpha/beta/state_step/tail kernels — all 4 are
   VARIANT-gated, anchors take the fused path.
3. SSM output projection (rmsnorm-fused at non-baseline).

### S5. GDN TP layer
`gdn_tp::forward_gdn_layer_tp`: same chain as S4 with AR boundaries
between the SSM stages. `FLAMBEAU_GDN_QKV_FUSE_Q8_0=off` opt-out (not
in anchor); `FLAMBEAU_SSM_OUT_F16_DST=on` opt-in fusion.

### S6. Dense FFN
`dense_ffn::forward_dense_ffn_decode` (single-device) /
`dense_ffn_tp::forward_dense_ffn_decode_tp` (TP). Gate / up / down
matmuls; `DENSE_GATE_UP=unfused` opt-out.

### S7. MoE FFN
- Optional shared-expert delta (`moe::forward_shared_expert_decode`).
- Router (`moe::forward_router_decode` — dense F32 GEMV + topk).
- Routed MoE matmul (`moe::forward_moe_ffn_decode`) — variant
  switched by `FLAMBEAU_MOE_VARIANT` (default `tile8`; legacy
  `FLAMBEAU_MOE_SORTED=0` shortcut to `r4`).

### S8. Output head
`io::forward_output_head_decode` (last rank). Two variants on TP /
Hybrid: `HostLogits` (DtoH F32 row) or `KeepOnDevice` (Sampler-D3
Phase B; consumed by `topk_softmax_f32` under
`FLAMBEAU_GPU_SAMPLER=1`).

### S9. Sampler
- `FLAMBEAU_GPU_SAMPLER=1` → `gpu_sampler::run_gpu_topk` (TP/Hybrid only;
  PP not wired, falls through to host).
- Else → host `Sampler::sample` from F32 logits.

### S10. Prefill chunking
`model::prefill_logits` chunks at `FLAMBEAU_PREFILL_UBATCH` (default
512) for PP / TP / Hybrid. Calls `forward_prefill_*_logits` final
chunk + `forward_prefill_*` non-final with `on_boundary` callback for
prefix-cache capture. TP path holds `#321` `prefill_serialiser` and
uses `#324` pooled scratch (`tp_prefill_scratch`).

### S11. Scheduler engagement
`routes::scheduler_can_engage` returns true iff
`FLAMBEAU_BATCHED_DECODE` is set AND model is `Pp{mtp:None}` / `Tp` /
`Hybrid` AND `!params.json_mode` AND `params.collect_logprobs` is
None. All anchor configs match → non-streaming chat goes via the
scheduler path. (Streaming chat does not — it's a known gap.)

## Branches NOT visited by any anchor cell

These code paths are reachable in code but not exercised by any
anchor (model, topology, N-class) under the optimized.toml config.
Phase 1 will classify each as off-path / dead / dispatch-table /
broken.

- **`forward_dense_attn_layer_decode`** (`attn.rs:1966`) — reachable
  only via `AttnWeights::Dense`. Set by `qwen3` (Embedding) and
  `qwen3moe` (Coder-30B, dropped V1.x). No anchor selects.
- **`FLAMBEAU_DECODE_GRAPH=1`** branch in
  `hybrid.rs:578` — HALT-BROKEN per env_impact cert; anchor leaves
  unset.
- **`FLAMBEAU_VARIANT=baseline`** unfused branches across attn / GDN
  / MoE / dense-FFN — opt-out of every fused kernel. No anchor
  selects baseline.
- **`FLAMBEAU_QKV_FUSED=0`** at `gdn.rs:1175` — opt-out of QKV
  fusion. No anchor selects.
- **`FLAMBEAU_TP_BATCHED=0`** at `tp.rs:860` and `hybrid.rs:76` —
  per-token prefill loop, kept for diagnostics + Q8 KV. Anchors all
  use F16 KV → Q8-KV fallback never used; no anchor sets `=0`.
- **`FLAMBEAU_GDN_NO_BATCHED=1`** at `hybrid.rs:1147` — N≥2 path
  forces per-slot GDN (already the default since no batched-GDN
  kernel). Diagnostic only.
- **`FLAMBEAU_DENSE_GATE_UP=unfused`** at `dense_ffn{,_tp}.rs:165` —
  opt-out; no anchor selects.
- **`FLAMBEAU_GDN_QKV_FUSE_Q8_0=off`** at `gdn_tp.rs:229` — opt-out;
  no anchor selects.
- **`FLAMBEAU_MOE_SORTED=0`** at `moe.rs:985` — legacy compat
  shortcut to `r4` (a known-loser). No anchor selects.
- **`FLAMBEAU_KV_F16_DST=off`** at `attn_tp.rs:210` — opt-out; no
  anchor selects.
- **`FLAMBEAU_AR_FUSE_Q8_1`** opt-in at `tp.rs:1840,2610` — anchor
  null on every cell (per env_impact survey).
- **`FLAMBEAU_Q4_0_GU_T128` / `_GU_WARPCOOP`** — kernel-variant
  toggles in dense-FFN-TP and GDN-TP gate-up; anchor null per
  survey.
- **`FLAMBEAU_SSM_OUT_F16_DST=on`** opt-in fusion at `gdn_tp.rs:609`
  — anchor null per survey.
- **All `FLAMBEAU_*_DUMP` / `_PROBE` / `_BISECT` / `LAYER_PROFILE`
  branches** — diagnostic dumpers in attn_tp / gdn_tp / layer / pp /
  hybrid. Phase 1: candidate for migration to `cfg(dev_trace)`.
- **`FLAMBEAU_NO_FAST_PATH=1`** at `routes.rs:757` — debug
  switch, scheduler path (decode_via_scheduler_into). Anchor unset →
  fast-path active. Per env-purge cert, unsetting it is the prod
  path. Keep as debug knob OR delete and rely on the (now)
  bench-evidenced fast-path.
- **`forward_one_token_pp_logits`'s `download_logits=None` arm**
  (`pp.rs:584`) — old host-argmax variant. Used by tests / spec
  decode. Server always passes `Some`. Phase 1: check if any
  non-test caller exists.
- **`forward_one_token_tp_keep_logits_on_device` /
  `forward_one_token_hybrid_keep_logits_on_device`** — Sampler-D3
  Phase B keep-on-device variants. Reachable only when
  `gpu_scratch.is_some()` in `run_completion_blocking_ids`. With
  `FLAMBEAU_GPU_SAMPLER=1` the scheduler handler doesn't currently
  use these (only the legacy ids handler does). Phase 1: confirm
  whether the scheduler should pick them up or whether the legacy
  branch is dead under optimized.toml.
- **`prefill_hybrid` non-batched fallback** at `hybrid.rs:88` — per-
  token loop for prompts < 8 tokens or `TP_BATCHED=0`. Smoke chat at
  prompt ≥ 8 tokens never reaches.
- **Spec-decode (`forward_speculative_pp_step*`,
  `decode_spec_pp{,_sampling}`)** — gated on `FLAMBEAU_SPEC_MTP`
  env at boot. Anchor unset → `LoadedModel::Pp { mtp: None, .. }` →
  scheduler engages directly. Spec is not on any anchor's optimal
  path.

This list anchors Phase 1's triage. Every entry above is a Phase 1
agenda item: confirm "no anchor selects" with a code-reading pass,
then either delete or move behind a feature flag.

