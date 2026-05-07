# Cleanup Triage — Phase 1

**Inputs:**
- `doc/V1.x/optimal_paths.md` — anchor forward paths (cells A–H + S1–S11).
- `doc/V1.x/env_var_inventory.md` — 113 env vars, 242 read sites.
- `doc/V1.x/kernel_impl_inventory.md` — 146 kernels + 64 dispatch rows + 92 certs.

**Anchors:** Qwen3.5-9B-Q4_1, Qwen3.6-27B-Q4_0, Qwen3.6-35B-A3B-Q4_0, Qwen3-Coder-Next-80B-Q4_0 × {pp4, tp2, pp2tp2}.

**Rule:** if 0a's forward-path map disagrees with 0b's preliminary verdict, **0a wins**.

**Slice key (defined in §3):** S1 default-bake / S2 kernel orphans / S3 dead-alt / S4 VARIANT / S5 clap / S6 dispatch-table / S7 debug-feature / S8 non-anchor.

---

## Section 1 — Env var verdicts

| Var | 0b verdict | Final | Justification | Slice |
|---|---|---|---|---|
| AR_FUSE_Q8_1 | DELETE | `delete-default-bake` | `tp.rs:1840,2610` opt-in; no anchor cell selects (cells F/G never set; cert null). | S1 |
| ASYNC_GRAPH | DISPATCH | `keep-dispatch-table` | `pp.rs:1477` PP-only async; loses on 35B-A3B/pp4 → dispatch row keyed on (model, topology). | S6 |
| ASYNC_UBATCH | DELETE | `delete-default-bake` | `pp.rs:1179` opt-in; no anchor cell selects, only consumed when ASYNC_GRAPH set. | S1 |
| BATCHED_DECODE | KEEP | `keep-clap` | `routes.rs:3358` gate for `scheduler_can_engage`; cells D/H require it; production-on. | S5 |
| BATCHED_MMVQ | DISPATCH | `keep-dispatch-table` | `qmatmul.rs:100` Q4_1 batched-MMVQ at 2≤m≤MMVQ_Q4_1_BATCHED_MAX_N — shape predicate. | S6 |
| BATCH_MAX | DELETE | `delete-default-bake` | `routes.rs:859` leader chunk cap; default `usize::MAX` is correct, no anchor sets. | S1 |
| BATCH_WINDOW_US | DELETE-DEAD | `delete-default-bake` | `routes.rs:819` knob for leader sleep; default 1500 µs is the prod path. | S1 |
| CTX_CAP | KEEP | `keep-clap` | `serve.rs:161` operator clamp; promote. **Conflict**: sibling `MAX_CTX` does the same — collapse into one clap arg. | S5 |
| DECODE_GRAPH | DELETE-DEAD | `delete-dead-alt` | `pp.rs:224, hybrid.rs:578`. HALT-BROKEN (3/9 cells crash). Delete the var **and** the graph-capture branch in pp/hybrid decode drivers. | S3 |
| DEFAULT_SYSTEM | KEEP | `keep-clap` | `serve.rs:401` boot-time chat default; promote. | S5 |
| DENSE_GATE_UP | DELETE-DEAD | `delete-dead-alt` | `dense_ffn.rs:165, dense_ffn_tp.rs:129`. HALT-DIVERGENT on TP2 unfused. Delete var + the unfused-TP2 branch (S3-S6 of S2). | S3 |
| EMBEDDING_MAX_TOKENS | KEEP | `keep-clap` | `serve.rs:544` `/v1/embeddings` clamp; promote. | S5 |
| FORCE_BATCH_WINDOW | DELETE | `delete-default-bake` | `routes.rs:833` debug timing knob; default off is prod. | S1 |
| GDN_NO_BATCHED | DELETE | `delete-default-bake` | `hybrid.rs:1147` per-slot fallback (current code already iterates per-slot at N≥2 — fallback is identical). Delete the gate, keep the loop. | S1 |
| GDN_QKV_FUSE_Q8_0 | DELETE | `delete-default-bake` | `gdn_tp.rs:229` opt-out; no anchor sets. Delete + the unfused branch. | S1 |
| GPU_SAMPLER | KEEP | `keep-clap` | `routes.rs:3714`; cell S9 requires it; production-on. | S5 |
| HOST_PROFILE | DEBUG | `debug-feature` | `routes.rs:4577` per-section host wall accumulator; gate behind `cfg(dev_trace)`. | S7 |
| INFLIGHT_SLOTS | KEEP | `keep-clap` | `serve.rs:424` slot-pool size; **load-bearing perf lever** (2.58× at slots=8 vs 1). Promote to clap. | S5 |
| KV | KEEP | `keep-clap` | `session.rs:55` KV layout choice; quality cert exists for Q8 KV. Promote. | S5 |
| KV_F16_DST | DISPATCH | `keep-dispatch-table` | `attn_tp.rs:210`. TP2-only loss when off; null elsewhere. Dispatch row keyed on (topology, dtype). | S6 |
| MAX_CTX | KEEP | `keep-clap` (collapse) | `config.rs:223`. **Duplicate of CTX_CAP** at the model-config layer. Collapse into single clap arg. | S5 |
| MAX_QUEUE_DEPTH | KEEP | `keep-clap` | `serve.rs:432` admission-control cap; promote. | S5 |
| MBATCH | DELETE | `delete-default-bake` | `moe.rs:1251` Q4_K mbatch alt; no anchor uses Q4_K MoE; cert null. | S1 |
| MOE_SCATTER | DELETE | `delete-dead-alt` | `moe.rs:1677`. `=race` opt-in is TP-bit-incorrect. Delete + the race-scatter branch (only kept the deterministic one). | S3 |
| MOE_SORTED | DELETE | `delete-default-bake` | `moe.rs:985` legacy compat shortcut to `r4` (confirmed loser). Tile8 default is correct. | S1 |
| MOE_VARIANT | DISPATCH | `keep-dispatch-table` | `moe.rs:984` master MoE variant gate. tile8 default global; Phase-0c kernels for `sorted/r4/turbo` exist. Either move alt selection to dispatch table **or** delete (cert says no win across 16 cells). **Open question**. | S6 |
| MTP_BF16 | DELETE-DEAD | `delete-dead-alt` | `mtp.rs:1296,1403,1513`. BF16 MTP is non-anchor (no anchor uses MTP); F16 path is the production default. Delete + bf16 mtp branches. | S3 |
| NO_FAST_PATH | DELETE-DEAD | `delete-dead-alt` | `routes.rs:757`. Debug-only; -2.0% with bit-identical output. Delete the var **but keep the fast-path itself** (the alternate path is what's broken-perf). Just remove the env-flag escape. | S3 |
| PREFILL_UBATCH | KEEP | `keep-clap` | `serve.rs:419, model.rs:135,525,621, routes.rs:332`. Five read sites should resolve once at boot. Promote to clap arg. | S5 |
| PREFIX_CACHE | KEEP | `keep-clap` | `prefix_cache.rs:280`. Promote (operator opt-in). | S5 |
| PREFIX_CACHE_MAX_GB | KEEP | `keep-clap` | `prefix_cache.rs:270` budget; sibling of PREFIX_CACHE. | S5 |
| PROFILE_DECODE | DEBUG | `debug-feature` | `routes.rs:4567` HipEvent recording window; gate behind `cfg(dev_trace)`. | S7 |
| Q4_0_GU_T128 | DELETE | `delete-default-bake` | `gdn_tp.rs:272, dense_ffn_tp.rs:165` shape-aware default already correct; no anchor toggles. Delete + collapse to default branch. | S1 |
| Q4_0_GU_WARPCOOP | DELETE | `delete-default-bake` | `gdn_tp.rs:270, dense_ffn_tp.rs:163`. Cert null; no anchor sets. | S1 |
| Q8_0_GU_T128_VDR2 | DELETE | `delete-default-bake` | `qmatmul.rs:548` t128_vdr2 default is correct; the `=off` opt-out is non-anchor. | S1 |
| Q8_0_MMVQ_T128 | DELETE-DEAD | `delete-dead-alt` | `qmatmul.rs:988`. `=on` is -4.8%/-7.0% on Q8_0 across pp4/tp2 — alternate path is purely worse. Delete var + remove t128 Q8_0 single-row recipe row. | S3 |
| Q8_0_MMVQ_T128_VDR2 | DISPATCH | `keep-dispatch-table` | `qmatmul.rs:995`. 3-way [unset / off / on]; `=off` is -5.1% on tp2 vs unset. **Open question**: investigate hidden branch divergence first; default unset is right but the fork suggests stale code. | S6 |
| QKV_FUSED | DISPATCH | `keep-dispatch-table` | `gdn.rs:1175`. +2.1% on 9B/pp2tp2, -3.6% on 9B/pp4 — model+topology dependent. Dispatch row. | S6 |
| SPEC_MTP | KEEP | `keep-clap` | `serve.rs:207` boot-time MTP-head load; operator config. | S5 |
| SSM_OUT_F16_DST | DELETE | `delete-default-bake` | `gdn_tp.rs:609`. Opt-in fusion; cert null on every cell. | S1 |
| TP_BATCHED | DELETE-DEAD | `delete-dead-alt` | `tp.rs:797,860, hybrid.rs:76`. `=0` is **13× prefill regression**. Alternate is purely dead. Delete var + the per-token TP-prefill fallback path. | S3 |
| TP_SKIP_SHARED | DEBUG | `debug-feature` | `tp.rs:1291,2163,3052, hybrid.rs:1349`. B5 bisect knob; gate behind `cfg(dev_trace)`. | S7 |
| UBATCH | DELETE | `delete-default-bake` | `pp.rs:1182`. Only consulted when ASYNC_UBATCH set; both die together. | S1 |
| VARIANT | MASTER | `master-switch-FLAMBEAU_VARIANT` | ~25 read sites across `ops/{qmatmul,moe}` + entire `models/qwen3-moe/forward/` tree. Anchor never sets `=baseline`. Collapsing means ~25 fused-path branches statically simplify. **Single largest cleanup lever — handle in S4, after S1–S3 land cleanly.** | S4 |
| **18 debug/dump vars** (AR_DUMP, BATCHED_DECODE_DUMP, DEBUG_TOOL_RAW, DUMP_PROMPT, DUMP_RAW_REQ, GRAPH_TRACE, KV_PROJ_DUMP, KV_ROPE_DUMP, LAYER_STATE_DUMP, LOAD_TRACE, PARITY_LAYER_DUMP, PARITY_TOPK_LOGITS, PP_PROBE, STAGE_ENTRY_DUMP, TP_LAYER0_BISECT, TP_LAYER_LIMIT, TP_PROBE, TRACE_BATCH) | DEBUG-FEATURE | `debug-feature` | All gate `is_ok()` print/dump blocks; no forward-path effect. Move under `cfg(dev_trace)` feature flag. | S7 |
| **~30 test/example vars** (see 0b §3) | TEST-ONLY | `test-only` | Read only from tests/examples; never reached by production binary. Fixture/A-B knobs. Move to BenchConfig / cfg(test). | (out-of-scope: separate cleanup, no slice) |

**Conflicts with 0b verdicts:** none. 0a confirms the "no anchor cell selects" reasoning for every DELETE / DELETE-DEAD-PATH var.

---

## Section 2 — Kernel impl verdicts

### 2.1 Anchors hit (confirmed live, KEEP):

For every kernel below, optimal_paths.md cells reach the launcher:
`mmvq_q4_0`, `mmvq_q4_0_t128`, `mmvq_q4_0_warpcoop64`, `mmvq_q4_0_gate_up_dp4a`, `mmvq_q4_0_gate_up_t128_dp4a`, `mmvq_q4_0_gate_up_warpcoop64`, `mmvq_q4_0_kv_f16dst_dp4a`, `mmvq_q4_1`, `mmvq_q4_1_t128`, `mmvq_q4_1_r2`, `mmvq_q4_1_r2_dp4a`, `mmvq_q4_1_batched`, `mmvq_q4_1_gate_up_dp4a`, `mmq_q4_0_4warp_lds`, `mmq_q4_1_4warp_lds`, `attention_decode_f16`, `attention_decode_f16_batched`, `attention_decode_f16_splitk`, `attention_prefill_f16`, all `rmsnorm_f16/f32/_q8_1_fused/_add_residual`, `l2_norm_f32`, `silu_f32`, `swiglu_*` (anchor dtypes only), `add_{f16,f32}`, `scale_f32`, `split_q_gate_f16`, `shared_expert_scale_f32`, `rope_f16`, `rope_neox_partial_f16`, `cast_*` (anchor pairs), `quantize_f16_q8_*`, `topk_f32`, `softmax_masked_f16`, all GDN kernels (`gdn_alpha_beta_f32`, `gdn_state_step_f32`, `gdn_state_step_alphabeta_f32`, `gdn_split_qkv_f32`, `gdn_assemble_conv_input_f32`), `causal_conv1d_f32`, `sampler_topk_softmax_f32`, `sampler_apply_penalties_f32`, `moe_combine_f16/_no_residual_f16/_two_residuals_f16`, `moe_sort_by_expert` (all 7 fns), all `dense_gemv_*` variants, `mmq_f16_q8_1`, `mmq_f16_tile`, `indexed_moe_mmvq_q4_0`, `indexed_moe_mmvq_q4_0_gate_up_dp4a`, `indexed_moe_mmq_q4_0_down_tile8_dp4a`, `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a`, `indexed_moe_mmq_q4_1_down_tile8_dp4a`, `mmvq_f16_q8_1`, `attention_decode_q8_kv` (Q8 KV opt-in), `attention_decode_f16_batched`, `mmq_q8_0_oracle` (debug oracle), `mmq_q8_0_wave64_tile16`, `mmq_q5_0_wave64`. **(verdict: keep-on-anchor-path, no slice)**

### 2.2 Orphans / unverified — DELETE candidates (slice S2):

| Stem | File | Status | Final | Justification |
|---|---|---|---|---|
| `mmq_q8_0_wave64_tile32` | `kernels/mmq_q8_0_wave64_tile32.cu` | orphan | `delete-orphan` | In KERNEL_STEMS, no dispatch row, no cert. Recipe alt only; never selected. |
| `mmq_q4_1_wave64_tile16` | `kernels/mmq_q4_1_wave64_tile16.cu` | orphan-with-cert | `delete-orphan` | In KERNEL_STEMS + cert exists, but no dispatch row → `Recipe::from_impl_id` never called. |
| `_unverified/indexed_moe_mmq_q4_k_gate_up_tile8_ylds.cu` | _unverified/ | unverified-V2.24.b NULL | `delete-cfg-unverified` | No callers in code. |
| `_unverified/indexed_moe_mmq_q4_k_gate_up_tile16_dp4a.cu` | _unverified/ | unverified + KERNEL_STEMS anomaly | `delete-cfg-unverified` | Launcher fn `indexed_moe_mmq_q4_k_gate_up_tile16` exists in `ops/moe.rs:623` but has **no callers in models/**. Delete `.cu` + launcher + KERNEL_STEMS entry. |
| `_unverified/mmvq_q4_0_r2.cu` | _unverified/ | unverified-V2.28.d NULL | `delete-cfg-unverified` | No callers (only a doc comment ref in `qmatmul.rs:600`). |

### 2.3 Bench-only kernels — KEEP as-is:

These are intentional A/B references (cited in `BENCH_REFERENCE_KERNELS_GFX906`, used by `sweep_mmq.rs` / PMC-refresh). Anchors don't traverse them but they remain useful for the bench infrastructure. **No slice.**

`mmq_q4_K_4warp`, `mmq_q4_K_turbo`, `mmq_q6_K_4warp`, `mmq_q8_0_4warp`, `mmq_q8_0_wave64`, `quantize_q8_1_mmq`, `attention_prefill_flash_tile_f16`, `peer_copy_via_host`, `indexed_moe_mmvq_q4_k` (single-row baseline).

### 2.4 Non-anchor live kernels — KEEP-NON-ANCHOR (S8 candidates):

These are referenced by forward path **for non-anchor models / dtypes**. They reach the kernel chain only when (model, dtype) ≠ anchor. Per CLAUDE.md scope ("V1.x targets specifically the anchor models"), they are S8 candidates **only if** the user wants to drop non-anchor support. Default = keep.

- **BF16 family** (Gemma-4 / Mistral non-anchor): `mmvq_bf16_bf16`, `attention_decode_bf16`, `rmsnorm_bf16`, `rope_neox_partial_bf16`, `swiglu_f32_to_bf16`, `sigmoid_mul_bf16`, `split_q_gate_bf16`. (~6 stems + 6 dispatch rows.)
- **K-quant MMQ family** (Q4_K_S / Q5_K_S / Q6_K weights, used by UD-Q4_K_S / UD-Q4_K_XL etc., not anchor models): `mmq_q4_K_wave64`, `mmq_q5_K_wave64`, `mmq_q6_K_wave64`. (3 stems + dispatch rows; **NOT contradictions** — 0c was wrong: stems ARE in KERNEL_STEMS at lines 123/124/126.)
- **K-quant MMVQ alt-rows**: `mmvq_q4_k`, `mmvq_q5_k`, `mmvq_q6_k`, `mmvq_q4_k_r4`, `mmvq_q5_k_r2`, `mmvq_q5_k_r2_f16dst`, `mmvq_q6_k_r4`, `mmvq_q6_k_dp4a`. Used for UD-Q*_K_* prefill / decode. Anchors don't hit.
- **Q5_0 / Q5_1 MMVQ**: `mmvq_q5_0`, `mmvq_q5_1`. Unsupported V1.x weight dtypes for anchors (Qwen3.x ships Q4_0/Q4_1/Q4_K_S/Q8_0).
- **Indexed-MoE Q4_K family** (only relevant if a Q4_K MoE arch exists — Qwen3.6 ships Q4_0; Qwen3-Coder-Next ships Q4_0): all `indexed_moe_mmvq_q4_k_*` variants + `indexed_moe_mmq_q4_k_*` variants. ~10 stems gated behind MOE_VARIANT alts (themselves a dispatch-table candidate).
- **Indexed-MoE Q4_1**: `indexed_moe_mmvq_q4_1` — launcher in `common.rs:222` + `moe.rs:1572`. Only fires if a Q4_1 MoE model loads (no anchor). Keep.
- **Indexed-MoE Q5_K / Q6_K / Q8_0 MMVQ**: `indexed_moe_mmvq_q{5_k,6_k,8_0}` — DirectCall registered. Used by Q5_K / Q6_K / Q8_0 weighted MoE archs. Anchors are Q4_0 only. Keep.
- **Indexed-MoE Q5_K / Q6_K / Q8_0 MMQ-down/gate_up**: `indexed_moe_mmq_q{5_k,6_k}_down_tile8_dp4a`, `indexed_moe_mmq_q8_0_*`. Same — non-anchor live.

**Action for §2.4:** these stay until a separate decision drops the non-anchor weight families. **No slice in this plan.**

### 2.5 Dormant rows (impls.rs note):

- `qmatmul_q4_1_mmq_wave64_gfx906` — m_range `(MAX,MAX)`. Lookup-only. Same kernel still in KERNEL_STEMS for cert-refresh. Safe to keep.
- `qmatmul_q4_K_mmq_turbo_gfx906` — `impls.rs` comment claims dormant but actual KernelDescriptor has m_range `(128, MAX)`. **Open question**: stale comment or duplicate row? Confirm in S2 before any K-quant cleanup.

---

## Section 3 — Phase 2 slice plan

### S1 — Default-bake env vars (low risk)

**In scope (14 vars, ~250 LOC removed):**
- AR_FUSE_Q8_1 (2 sites)
- ASYNC_UBATCH (1)
- BATCH_MAX (1)
- BATCH_WINDOW_US (1)
- FORCE_BATCH_WINDOW (1)
- GDN_NO_BATCHED (1)
- GDN_QKV_FUSE_Q8_0 (1)
- MBATCH (1)
- MOE_SORTED (1)
- Q4_0_GU_T128 (2)
- Q4_0_GU_WARPCOOP (2)
- Q8_0_GU_T128_VDR2 (1)
- SSM_OUT_F16_DST (1)
- UBATCH (1, dies with ASYNC_UBATCH)

**Action per var:** find the if/match-branch the var gates, statically pick the production branch, delete the env read + the dead alt branch. Update `bench/profiles/optimized.toml` to drop `[delete_candidates]` entries.

**Verification:**
1. `cargo build --release --features hip_serve` clean.
2. `python3 scripts/bench/repro_35b_divergence.py` → NO-DIVERGENCE.
3. Smoke: `BASELINE_27B_PP2TP2` + `BASELINE_35B_PP2TP2` at N=1, N=4 — within ±5% of d7c3cc6.

**Risk:** LOW. These env vars all measure null on anchors; deleting their alt branches is removing dead code.

**Why first:** smallest risk, biggest line-count reduction. If anything regresses here, we learn the bench oracle is wrong before touching the harder slices.

---

### S2 — Kernel orphans + unverified (low risk)

**In scope (5 kernels, ~3 KB .cu + ~80 LOC of launchers):**
- `mmq_q8_0_wave64_tile32.cu` — orphan
- `mmq_q4_1_wave64_tile16.cu` — orphan with cert (delete cert too)
- `_unverified/indexed_moe_mmq_q4_k_gate_up_tile8_ylds.cu`
- `_unverified/indexed_moe_mmq_q4_k_gate_up_tile16_dp4a.cu` + ops/moe.rs:623 launcher (`indexed_moe_mmq_q4_k_gate_up_tile16`)
- `_unverified/mmvq_q4_0_r2.cu`

**Action:** delete `.cu` + KERNEL_STEMS entries + Recipe rows in qmatmul.rs/moe.rs + cert JSONs.

**Verification:** same as S1.

**Risk:** LOW. No callers. Build will fail loudly if a stale reference remains.

**Why second:** orthogonal to S1; lets us validate the verification gate handles two distinct change shapes.

---

### S3 — Dead-alt env vars + alt code (medium risk)

**In scope (6 vars + their alt branches, ~600 LOC):**
- DECODE_GRAPH (2 sites + graph-capture branch in `pp.rs` and `hybrid.rs`)
- DENSE_GATE_UP (2 sites + unfused-TP2 dense-FFN branches in `dense_ffn.rs:165` and `dense_ffn_tp.rs:129`)
- MOE_SCATTER (1 site + `flambeau_moe_sort_scatter` race-mode kernel in `ops/moe.rs:1677`; **keep the deterministic scatter, delete the racing scatter**)
- MTP_BF16 (3 sites + BF16 MTP step branches in `mtp.rs:1296,1403,1513`; only matters when SPEC_MTP set, so deleting BF16 MTP only loses an alt MTP path — F16 MTP is the production path)
- NO_FAST_PATH (1 site, just delete the env read; **the fast-path itself stays**)
- Q8_0_MMVQ_T128 (1 site + Recipe row + `mmvq_q8_0_t128.cu` deletion candidate — **0a confirms no anchor cell selects t128; cert says alt path is -7%**)
- TP_BATCHED (3 sites + per-token TP-prefill fallback branches in `tp.rs:797,860` and `hybrid.rs:76` — the slow path)

**Action:** delete the var **and the alt branch's code**. Distinguish per var:
- **Code-deletion type A** (env-only): NO_FAST_PATH — just delete the env read and the surrounding `if no_fast_path { ... }` shortcut, keep all code paths.
- **Code-deletion type B** (env + alt branch): all others — delete the env read, delete the broken/divergent branch's code, keep the production branch as the only code path.

**Per-var sub-verification:**
- DENSE_GATE_UP: ensure no anchor cell selects unfused (Phase 0a confirms cells F/G/H stay fused).
- DECODE_GRAPH: confirm no test relies on the graph-capture branch.
- MTP_BF16: confirm no anchor uses MTP (anchors leave SPEC_MTP unset → MTP is loaded into `LoadedModel::Pp { mtp: None, .. }`, never invoked).
- TP_BATCHED: the fallback was kept for diagnostics + Q8 KV. Does Q8 KV path still need it? **Open question** — investigate `model::prefill_logits` path for Q8 KV.

**Verification:** same as S1, plus an extra check: `cargo test -p flambeau-qwen3-moe` (parity tests) — if any test depends on the deleted alt branch, it surfaces here.

**Risk:** MEDIUM. The alt branches are dead-on-anchor but may be exercised by tests; ensure tests use cfg(test) overrides instead of env vars before deletion.

**Why third:** S1 + S2 land first to confirm the verification gate works on simpler cases. S3 deletes more code but every branch we delete is bench-evidenced dead.

---

### S4 — `FLAMBEAU_VARIANT` master-switch collapse (medium-high risk, biggest lever)

**In scope:** ~25 read sites across `ops/{qmatmul,moe}` and `models/qwen3-moe/forward/{attn,attn_tp,dense_ffn,dense_ffn_tp,gdn,gdn_tp,moe,moe_tp}.rs`. Anchor never sets `=baseline` → every `VARIANT != "baseline"` branch is the live one; all `=baseline` branches are dead.

**Action:** for each `match env::var("FLAMBEAU_VARIANT").as_deref() { Ok("baseline") => …, _ => … }` site, statically pick the `_` branch and delete the `Ok("baseline")` branch. Plus delete:
- `Recipe::from_impl_id` baseline-MMVQ entries (`baseline`, `dp4a_*`, `q8_r4`, `q4_1_wave64`, etc.) — **but keep the dispatched-default kernels they alias to**.
- All `if VARIANT == "baseline"` opt-out arms in fuse_kv / fuse_qkv / fuse_alpha_beta / fuse_state_step / fuse_tail / fuse_swiglu_quant / fuse_gate_up.

**Verification:** same as S1, plus:
- All four anchor cells × {N=1, N=4} smoke bench.
- Output-text bit-identity vs d7c3cc6 (we keep d7c3cc6 commit hash as the perf+correctness anchor).

**Risk:** MEDIUM-HIGH. Touches the entire forward tree. If any non-anchor cell relies on `=baseline` (e.g. a debug A/B path used in some test), build fails or test regresses. Also: some `VARIANT != "baseline"` branches may have subtle alts (e.g. `dp4a` vs `q8_r4`) that we'd statically simplify to one — confirm those alts are unused too.

**Why fourth:** the largest line-count + branch-count reduction (~25 fused-path branches collapsed). Wait until S1–S3 land cleanly so the verification gate is well-tested.

---

### S5 — Promote 13 keep-clap vars to `clap` args (low risk)

**In scope (13 vars):**
BATCHED_DECODE, GPU_SAMPLER, INFLIGHT_SLOTS, PREFILL_UBATCH, PREFIX_CACHE, PREFIX_CACHE_MAX_GB, KV, CTX_CAP, MAX_CTX (collapsed into CTX_CAP), MAX_QUEUE_DEPTH, DEFAULT_SYSTEM, EMBEDDING_MAX_TOKENS, SPEC_MTP.

**Action:** add a `ServeConfig` struct in `crates/server` (or `crates/cli`) parsed via clap. Replace each `env::var("FLAMBEAU_X")` site with `cfg.x`. Keep env-var fallback for deployment compat: clap arg → env var → default.

**Verification:** same as S1. Plus: check that `bench/profiles/optimized.toml` is still the production reference (likely rewrite it as a `flambeau serve …` command-line example).

**Risk:** LOW. Pure surface-shape change. `bench/profiles/optimized.toml` and `scripts/bench/run_env_impact.py` will need updating to use `serve --inflight-slots 8 …` style — that's harness work, not production-correctness.

**Why fifth:** the keep-vars don't get smaller until clap migration — the 13 reads stay until that's done.

---

### S6 — Migrate 6 vars to dispatch table rows (medium risk)

**In scope (6 vars):**
ASYNC_GRAPH, BATCHED_MMVQ, KV_F16_DST, MOE_VARIANT, Q8_0_MMVQ_T128_VDR2, QKV_FUSED.

**Action:** for each var, identify the (model, topology, dtype, m) predicate and add a row to `dispatch/hip/gfx906.toml`. Replace the env-var read with a `dispatch::resolve(predicate)` call. Some of these (BATCHED_MMVQ, MOE_VARIANT) span multiple variants — encode each variant as its own row.

**Verification:** same as S1, plus full `cargo run -p bench -- sweep --arch gfx906` + `cargo run -p bench -- matrix` to confirm dispatched variants stay correct.

**Risk:** MEDIUM. Dispatch table changes can move the hot path. Q8_0_MMVQ_T128_VDR2 has the off-vs-unset divergence noted in 0b — investigate before migration.

**Why sixth:** dispatch-table is the right home but this is the largest behavioral migration. Land after S1–S5 clean up the simpler vars.

---

### S7 — Debug-feature gating (low risk)

**In scope (~22 debug vars):**
All 18 debug/dump vars from 0b §2 + HOST_PROFILE + PROFILE_DECODE + TP_SKIP_SHARED + 4 bisect/probe vars (LAYER_LIMIT, LAYER0_BISECT, etc.).

**Action:** add a `dev_trace` feature in the relevant crates' Cargo.toml. Wrap each `if env::var(…).is_ok() { dump }` block in `#[cfg(feature = "dev_trace")]`. Production builds (default features) skip the env reads and the dump code entirely.

**Verification:** build twice — once with `--features hip_serve` (default, no dev_trace) and once with `--features hip_serve,dev_trace`. Both clean.

**Risk:** LOW. Debug vars have no forward-path effect; gating them removes binary size + a small amount of env-read overhead.

**Why seventh:** mechanical. Order doesn't matter much; can be done in parallel with S5 / S6.

---

### S8 — Non-anchor kernel/dtype paths (high risk, OPTIONAL)

**In scope (~25 kernels + corresponding dispatch/cert rows):**
BF16 family, K-quant MMQ family, Q5_0/Q5_1 MMVQ, K-quant MMVQ alts, indexed-MoE K-quant + Q4_1 + Q5/Q6/Q8 MMVQ.

**Action:** delete kernels + dispatch rows + certs + the model-side branches that select them.

**Verification:** same as S1 PLUS confirm no V1.x scope item lists these models (CLAUDE.md says Mistral / Gemma-4 are in scope but not anchors — be careful).

**Risk:** HIGH. These kernels exist to support models in scope (per CLAUDE.md) that are not benched as anchors. Deletion narrows V1.x scope; reverting later means recompiling kernels.

**Why eighth (and OPTIONAL):** the user picked anchors that exclude Mistral / Gemma-4 / Q*_K-weighted models. If V1.x truly doesn't ship them, S8 is a big simplification. If V1.x means to ship them, S8 is a regression. **Open question** — needs explicit user confirmation before any S8 commit.

---

## Section 4 — Open questions (resolved 2026-05-07)

User decisions on all 6 open questions:

1. **MOE_VARIANT** → default-bake `tile8`, delete the ~6 alt kernels. (S1 + S2)
2. **TP_BATCHED `=0`** → delete both the alt per-token path and the env flag. (S3 expanded)
3. **Q8_0_MMVQ_T128_VDR2** → investigate `qmatmul.rs:548` and `qmatmul.rs:995` during S2 prep. If `unset==on` path identical, downgrade to S1; otherwise keep as S6 dispatch row.
4. **`qmatmul_q4_K_mmq_turbo_gfx906`** → investigate during S2. Confirm dormant vs live by reading both call sites; delete if dormant, fix comment if live.
5. **S8 (non-anchor kernels)** → SKIP. CLAUDE.md scope still lists Mistral / Gemma-4 / K-quant; out-of-anchor is not out-of-scope. Don't delete BF16 / K-quant MMQ / Q5_0/Q5_1.
6. **Test-only vars (~30)** → ADD slice S9: move to `BenchConfig` struct in `crates/bench`, removing them from the production binary.

Phase 2 cleared to start.

---

## Section 4 (original) — Open questions for user

1. **MOE_VARIANT (slice S6 vs S1)**: cert says no win across 16 cells. Two reasonable verdicts:
   - Migrate to dispatch table (keeps `tile8`/`sorted`/`r4`/`turbo` selectable per-shape) → S6.
   - Default-bake `tile8` and delete the alt kernels (~6 indexed-MoE Q4_K alt variants) → S1 + S2.
   Recommendation: **default-bake** (S1+S2) — the dispatch-table is for evidenced wins, not for "no-win-but-might-someday".

2. **TP_BATCHED Q8 KV path (slice S3)**: the `=0` per-token fallback is documented as kept for Q8 KV diagnostics. Does production Q8 KV still need it? If yes → keep the alt path, just delete the env-flag escape. If no → delete both.

3. **Q8_0_MMVQ_T128_VDR2 (slice S6)**: 3-way [unset / off / on] suggests stale code at `qmatmul.rs:548` AND `qmatmul.rs:995`. Investigate before migrating: is the `unset → t128_vdr2 default` path actually identical to `=on → t128_vdr2`? If yes the var is redundant and goes to S1.

4. **`qmatmul_q4_K_mmq_turbo_gfx906` dormant note**: `impls.rs` comment says dormant; actual KernelDescriptor has `m_range (128, usize::MAX)`. Stale comment or duplicate row? Affects S2's K-quant cleanup if any.

5. **S8 scope**: anchors exclude Mistral / Gemma-4 / K-quant-weighted models. Are these out of V1.x scope (drop kernels = S8 lands), or kept under "scope but not benched" (S8 stays out)?

6. **Test-only vars (~30, see 0b §3)**: out of this slice plan but a separate cleanup. Move to a `BenchConfig` struct in `crates/bench`, removing them from the production binary. Want this as a separate slice S9, or skip?

Phase 2 should not start any slice with an open question above unresolved.

---

## Slice ordering summary

| Order | Slice | Risk | Vars touched | Kernels touched | LOC |
|---:|---|---|---:|---:|---:|
| 1 | S1 default-bake | LOW | 14 env | 0 | ~250 |
| 2 | S2 kernel orphans | LOW | 0 | 5 | ~3 KB .cu + ~80 LOC |
| 3 | S3 dead-alt | MED | 6 env + alt code | 0 | ~600 |
| 4 | S4 VARIANT collapse | MED-HIGH | 1 env (~25 sites) | 0 (just dead branches) | ~500 |
| 5 | S5 keep → clap | LOW | 13 env | 0 | ~150 |
| 6 | S6 dispatch-table | MED | 6 env | 0 (just predicates) | ~200 |
| 7 | S7 debug-feature | LOW | ~22 env | 0 | ~100 (gates only) |
| 8 | S8 non-anchor | HIGH (OPTIONAL) | 0 env | ~25 | several KB |

**End-state goal** (per `code_path_cleanup_plan.md`):
- ≤ 5 keys in `bench/profiles/optimized.toml` `[env]` (currently 5; production switches to clap args, profile becomes a doc).
- ≤ 10 live `FLAMBEAU_*` env reads (S5+S7 leave only DEBUG-FEATURE-feature-flagged + clap-fallback reads).
- 137 → ~110 `KERNEL_STEMS` entries after S2 + S8.
- 64 → ~55 dispatch rows after S2 + S6 (S6 *adds* rows but S2 *removes* them).
