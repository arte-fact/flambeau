# Tensor Parallelism Plan — flambeau on 4× MI50 (gfx906, PCIe-only)

**Status:** draft / not started
**Created:** 2026-04-25
**Reference rig:** 4× AMD MI50 32 GB, gfx906, PCIe (no XGMI)
**Reference impl:** [`larkinwc/mi50grad`](https://github.com/larkinwc/mi50grad) — proven TP-4 on the same silicon

---

## 0. Why TP, despite CLAUDE.md saying "PP-primary"

CLAUDE.md (V1 vertical-slice posture) takes a strong anti-TP stance:

> **Pipeline parallelism — not wide expert parallelism — is the V1 topology.** Rationale: the V1 rig is PCIe-only … TP>2 and wide-EP all-to-all are bandwidth-bound on PCIe (40–50 % comm overhead measured) …

That stance was **right for V1**, but it conflates two things and excludes a third:

1. **Wide-EP all-to-all** (DeepEP-style, full expert routing across ranks) — genuinely PCIe-bound, correctly excluded.
2. **Megatron-LM TP** with naive ring/star AllReduce — the 40–50 % overhead figure is from this regime; correctly excluded as a default.
3. **Megatron-LM TP with kernel-launched BAR1 P2P AllReduce** (mi50grad's M1+M3 path) — **not* covered* by CLAUDE.md's reasoning.

mi50grad ships (3) on this exact rig and gets:

| Config (Qwen3.5-27B-GPTQ-Int4) | Throughput | vs. flambeau Mesh<4> PP equivalent |
| --- | --- | --- |
| TP=1 (1× MI50) tg128 | 20.4 tok/s | — |
| TP=4 (4× MI50) tg128 | **56.3 tok/s** | flambeau V2.32.a Mesh<4> = **21.1 tok/s** (Qwen3.5-27B-Q4_1) |
| TP=2 (2× MI50) pp512 | **38.9 tok/s** | (flambeau peak at ≥640 with V2.32.a Q4_1 prefill) |
| TP=4 (4× MI50) pp512 | 32.1 tok/s | (TP=4 prefill is *worse* than TP=2 — AR payload dominates) |

The data carries two clean messages:

- **Decode wins go to TP**: 56.3 vs 21.1 tok/s on equivalent 27B Q4 quants, **~2.7× speedup** purely from topology + BAR1 AR. flambeau is leaving this on the table.
- **Prefill wins stay with PP** (or with TP-2 at most): TP-4 prefill regresses vs TP-2 because per-layer AR payload scales with `L × hidden`. flambeau's V2.32.a Mesh<4> Q4_1 prefill (566 tok/s L=8192) is *already faster than mi50grad's TP-2 best (38.9)* on related shapes — so we have *nothing* to gain by switching prefill topology, and risk losing a lot.

**Conclusion:** TP is a decode-only addition. Prefill stays on the V1.7/V2.4 PP path. The two topologies coexist within a single run, switching at the prefill→decode boundary.

---

## 1. The three load-bearing ideas from mi50grad

### 1.1 BAR1 P2P AllReduce kernel (M1, +1.50× over star-AR)

`hipDeviceEnablePeerAccess` makes peer GPU memory addressable through PCIe BAR1. A single kernel on each rank reads peer partial buffers via plain `__half2*` loads and writes a residual-add result locally. No host bounce, no `hipMemcpyPeerAsync`, no per-GPU sequential reduce.

mi50grad's [`kernel_p2p_allreduce.hip`](https://github.com/larkinwc/mi50grad/blob/main/src/kernels/kernel_p2p_allreduce.hip) — quoted, the inner loop is just:

```cpp
__half2 h  = *(__half2*)(hidden + idx);          // local
__half2 pl = *(__half2*)(partial_local + idx);   // local
__half2 p0 = *(__half2*)(partial_peer0 + idx);   // BAR1 P2P read
__half2 p1 = *(__half2*)(partial_peer1 + idx);   // BAR1 P2P read
__half2 p2 = *(__half2*)(partial_peer2 + idx);   // BAR1 P2P read
*(__half2*)(hidden + idx) = h + pl + p0 + p1 + p2;   // FP32-accumulated
```

Cross-rank synchronisation lives at the **launch boundary** (HIP events on each rank's stream), not inside the kernel. Each rank launches its own copy simultaneously; each rank reads peer partials *as they exist at launch time* — caller is responsible for ensuring the producer GEMV completed first (via `hipEventRecord` + cross-stream wait). No in-kernel atomic barrier required.

**Numbers (mi50grad RESEARCH.md):**

- 5120 FP16 = 10 KB payload
- BAR1 effective bandwidth ~12 GB/s per peer, so **theoretical** ~0.8 µs
- Measured 79 µs — synchronisation overhead, not bandwidth, dominates

This is the single most important kernel in the whole plan.

### 1.2 Deferred Attention AllReduce (M3, +35 %)

Standard transformer pre-norm block (one TP rank, dense FFN):

```
x_in
 ├── norm1 → attn_local (Q/K/V/O col-row sharded) → AR1 → attn_full
 │     residual: x_in + attn_full = h_attn
 │                                             ┌── h_attn
 └── norm2(h_attn) → ffn_local (gate/up/down) → AR2 → ffn_full
       residual: h_attn + ffn_full = x_out
```

Two AR per layer × 64 layers (Qwen3.5-27B) = 128 AR/token. At 79 µs each, AR alone costs 10.1 ms — already over the 18.6 ms decode budget for 60 tok/s.

**Deferred AR**: skip AR1, let each rank carry its own `attn_local`, run FFN on a *partial* residual stream:

```
x_in (replicated, AR-correct) → norm1 → attn_local → r_local := x_in + attn_local (rank-local!)
                                                                       │
norm2 sees a rank-local input, but: norm(x) is invariant under x↦x+c IF c is the same on all ranks
   x_in is replicated, so the differing component is attn_local — we must norm on the AR'd value.
```

mi50grad's actual scheme exposes this with one more step. The block becomes:

```
attn_seg:   norm1 → q,k,v,o_local      (ends in attn_local, no AR)
ffn_seg:    AR(attn_local) → r := x_in + AR_attn → norm2 → gate,up,down_local
combined AR: AR(r + ffn_local) once at the end of layer
```

Net: **one** AR per layer instead of two. 128 → 64 AR/token. The "deferred" part is folding the residual-add and the second AR into the same kernel-launched P2P AR. The "fused GEMV+AR+RMSNorm" (M3, +3.8 % on top) further fuses the down-proj GEMV's last-block reduce with the AR write — that's a polish step, not a load-bearing one.

### 1.3 Topology: P2P graph capture + C dispatch — **NOT plan-critical**

mi50grad's M2 is a HIP-graph-captured per-layer dispatch from compiled C, eliminating Python interpreter overhead. flambeau's call site is already Rust + bound HIP; the per-call FFI overhead is bounded and progressive-dispatch already overlaps. M2 transposed to flambeau is "use existing async-stream dispatch + record HIP events between AR and next-layer norm". No new infrastructure needed beyond what V2.30.a's event ordering already supplies.

---

## 2. Sharding axes — what gets cut where

### 2.1 Dense attention (qwen35: Qwen3.5-27B-Q4_1, V2.32.a target)

For TP-N with hidden = `H`, num_q_heads = `nQ`, num_kv_heads = `nKV` (GQA), head_dim = `D`, intermediate = `I`:

| Weight | Shape (full) | TP layout | Per-rank slice | AR? |
| --- | --- | --- | --- | --- |
| `attn_q.weight` | `(nQ·D, H)` | **col-parallel** (split rows = output dim) | `(nQ/N · D, H)` | no (Q is rank-local input to attn) |
| `attn_k.weight` | `(nKV·D, H)` | **col-parallel** | `(nKV/N · D, H)` | no |
| `attn_v.weight` | `(nKV·D, H)` | **col-parallel** | `(nKV/N · D, H)` | no |
| `attn_output.weight` | `(H, nQ·D)` | **row-parallel** (split cols = input dim) | `(H, nQ/N · D)` | yes (or deferred) |
| `ffn_gate.weight` | `(I, H)` | **col-parallel** | `(I/N, H)` | no |
| `ffn_up.weight` | `(I, H)` | **col-parallel** | `(I/N, H)` | no |
| `ffn_down.weight` | `(H, I)` | **row-parallel** | `(H, I/N)` | yes |
| `*_norm.weight` | `(H,)` | **replicated** | `(H,)` | no |
| `token_embd.weight` | `(V, H)` | **replicated** rank 0, or vocab-sharded | (init: replicate; later: split V) | n/a |
| `output.weight` (LM head) | `(V, H)` | **col-parallel** (split V), then arg-max-AR | `(V/N, H)` | only argmax-mask reduce |

Constraint: `nQ % N == 0`, `nKV % N == 0` (mi50grad asserts). Qwen3.5-27B has `nQ=64, nKV=8` → divides cleanly for N ∈ {1,2,4,8}.

**KV cache ramification:** with `nKV/N` heads per rank, the KV cache is **already** sharded as a side-effect — total KV memory is unchanged but spread across ranks. This is the single biggest qualitative reason TP wins decode on small ranks: `KvCache<F16Contig>::per_rank_bytes` shrinks by `N`, easing HBM pressure on the rank-1 case where 27B-Q4_1 + KV barely fits 1 card.

### 2.2 GDN layers (qwen35moe hybrid, V2 second target)

GDN's per-token recurrent state is keyed by `(linear_num_v_heads, linear_value_head_dim)`. Shard heads col-wise → state stays rank-local, no AR needed *inside* GDN. Output projection is row-parallel → AR at exit (or fold into deferred-AR pattern with the next layer).

Subtlety: GDN α/β scalars are *per-head* and computed from rank-local Q/K/V (which already differ per rank under col-parallel). They must NOT be AR'd — heads on rank `r` use rank-`r`'s own α/β. This is the natural Megatron pattern; mention here only because the V2.30.a fusion kernel currently assumes single-device.

### 2.3 MoE layers (qwen35moe + qwen3moe)

Two viable schemes:

**A. Intra-expert TP** (recommended, no AllToAll): each rank holds 1/N of *every* expert's `ffn_gate/up/down` (col/col/row sharded inside the expert). Router runs replicated → top-k indices match across ranks → each rank computes its slice → AR after `ffn_down`. Fits the same pattern as dense FFN. `intermediate_size_per_expert` must divide `N`.

**B. Expert-parallel** (vLLM-style): each rank holds different *whole* experts; AllToAll routes tokens to expert ranks. PCIe-only kills this (mi50grad explicitly avoids it; CLAUDE.md correctly excludes wide-EP).

**Plan: scheme A only.** The shared expert in qwen35moe is dense-shaped — col/row parallelise it like a regular FFN.

### 2.4 Prefill — explicitly out of TP scope

Per §0, prefill stays on the existing V1.7/V2.4 PP path. The TP topology is engaged only when transitioning to decode (after the first `forward_one_token` call). No code path computes prefill TP. If a user runs `serve --tp-size 4` we set up TP shards at load and use PP-on-the-same-shards for prefill (one rank does its own piece linearly and then reduces; equivalent to TP-1-per-rank for prefill but with sharded weights). Acceptable because prefill is bandwidth-bound on weights and 4× weight-bandwidth still helps.

Actually the simpler decision: **prefill uses PP layer-split (current)**, decode uses TP weight-shard. The two require different weight layouts — see §6.

---

## 3. Where this lands in flambeau's existing surface

flambeau already has the right shape:

- `runtime::AllReduce` trait — `crates/runtime/src/collective.rs:149`
- `runtime::Mesh` trait + `RefMesh` reference impl
- `backend-hip::HipMesh` + `HipRankHandle::all_reduce_host` (RCCL host-bounce)
- `backend-hip::cluster::peer_copy_via_host` (current PP hand-off)
- `runtime::LayerAssignment` (PP layer-split)

What's missing:

- `WeightLayout::ColParallel { world } | RowParallel { world } | Replicated` enum (new, in `crates/loaders` or `crates/models/qwen3-moe/weights`)
- `BarP2pAllReduce` impl of `AllReduce` (new, `backend-hip/src/bar_p2p.rs` + `kernels-hip/src/kernels/kernel_p2p_allreduce_*.cu`)
- `forward_one_token_tp` — peer of `forward_one_token_pp` in `crates/models/qwen3-moe/src/forward/`
- TP layout selector at load time (CLI `--tp-size N` or `--mesh-mode {pp,tp}`)

The PP path stays intact; TP is a parallel forward driver, not a replacement.

---

## 4. Targets, gates, and the honest ceiling

Targets are calibrated against mi50grad's measured numbers (same silicon, same model class), not against vLLM-on-A100 fantasies.

| Model | Topology | Current (V2.32.a) | Gate (V2.X.a-tp) | mi50grad evidence |
| --- | --- | --- | --- | --- |
| Qwen3.5-27B-Q4_1 (qwen35 dense) | Mesh<4> decode | 21.1 tok/s | **≥ 50 tok/s** | 56.3 tok/s |
| Qwen3.5-27B-Q4_1 | Mesh<4> prefill | 566 tok/s L=8192 | **≥ 540 tok/s** (no regression) | n/a — keep PP |
| Qwen3.5-9B-Q4_1 (V2.2) | Mesh<1> decode | 65.2 tok/s | n/a — already fits 1 card | n/a |
| Qwen3.6-35B-A3B (qwen35moe hybrid) | Mesh<4> decode | 53.5 tok/s | **≥ 75 tok/s** | extrapolated |
| Qwen3-Coder-30B (qwen3moe) | Mesh<4> decode | 38.9 tok/s | **≥ 60 tok/s** | extrapolated |

Gate methodology: TP cert lands only when (a) parity bit-exact vs Mesh<1> on a 1-token greedy probe at seed 9419, and (b) decode tok/s ≥ gate at tg=64. No gate, no merge.

**Ceiling honesty:** mi50grad got 56.3 / 21.97 = 2.56× over their own single-GPU; we're starting from a **harder** baseline (V1.7.6.f's 65.2 tok/s on 1-card-fits Qwen3.5-9B already implies our single-GPU kernel stack is tighter than mi50grad's INT4-GEMM-v2). Expect 2.0–2.4× over Mesh<4>-PP, not 2.6×.

---

## 5. Session-sized phases

Each phase is a single-PR, single-cert deliverable. **Order is load-bearing — do not reorder.**

### TP-0: BAR1 P2P AllReduce primitive (foundation)

- **TP-0a — peer-access registry.** In `backend-hip/src/cluster.rs`, after `HipCluster::init`, call `hipDeviceEnablePeerAccess(peer)` for every (rank, peer) pair where `hipDeviceCanAccessPeer == 1`. Skip pairs where it's 0 (degenerate; warn). Add `HipMesh::peer_pointer(rank: RankId, peer: RankId, local_ptr: DevicePtr) -> DevicePtr` that returns a BAR1-mapped pointer (which on ROCm is the same numeric address — peer access doesn't remap, it just authorises the read). Test: 10 KB write on rank 0, peer-read from rank 1, byte-equal.
  - Risk: ROCm 7.1.1 patched gfx906 has had peer-access regressions; mi50grad pins to a specific patched docker image. Probe `HIP_PLATFORM=amd hipDeviceCanAccessPeer` matrix at boot, fall back to host-bounce AR with a one-line log if any pair returns 0.
  - **Files:** `backend-hip/src/cluster.rs` (additive).
  - **Cert:** `certs/hip/gfx906/peer_access_matrix.json` — N×N of `can_access_peer` results.

- **TP-0b — kernel_p2p_allreduce_residual_tpN kernels.** New file `kernels-hip/src/kernels/p2p_allreduce_residual.cu`, port mi50grad's [`kernel_p2p_allreduce.hip`](https://github.com/larkinwc/mi50grad/blob/main/src/kernels/kernel_p2p_allreduce.hip) verbatim — both `tp2` and `tp4` variants, and the `_sum` (no residual) variants. Single-source via templating or four `extern "C"` shells, doesn't matter. Block 256 threads, 2 fp16 elem/thread, FP32 accumulate. Build hsaco, register dispatch row.
  - **Cert:** `certs/hip/gfx906/p2p_allreduce_residual_tp4.json` — correctness sweep at N ∈ {1024, 5120, 8192, 16384, 65536} fp16 elements; reference is host-side sum.
  - **PMC:** record `MemBusy` per-rank during the kernel — should be near zero on local HBM (the peer reads are BAR1, don't show up the same way). Latency target ≤ 90 µs for N=5120.
  - **Files:** new kernel file + dispatch row in `dispatch/hip/gfx906.toml`.

- **TP-0c — `BarP2pAllReduce` impl of `runtime::AllReduce`.** New file `backend-hip/src/bar_p2p.rs`. The trait method takes `&mut [u8]`, but our payload is *already* device-local — add a parallel `all_reduce_device(&mut [DevicePtr], hidden_ptrs: &[DevicePtr], elem_count, dtype, stream_per_rank)` extension. Caller passes one device pointer per rank (the partial buffer) plus the per-rank residual target buffer. Each rank launches its own kernel on its own stream, then `hipEventRecord` after launch. Ordering with the producing GEMV is via `hipStreamWaitEvent` on the GEMV's completion event before the AR launch.
  - **Cert (latency):** `certs/perf/p2p_allreduce_residual_tp4_latency.json` — measured 79–95 µs at N=5120. Compare against `HipRankHandle::all_reduce_host` (RCCL or host-bounce) at the same shape; expect ≥ 1.3× speedup.
  - **Files:** `backend-hip/src/bar_p2p.rs`, `backend-hip/src/lib.rs` re-export, `runtime/src/collective.rs` (extend trait if needed — additive method with default `unimplemented!`).

### TP-1: sharded weight loader (no forward yet)

- **TP-1a — `WeightLayout` enum + per-tensor table.** In `crates/loaders/src/lib.rs` (or wherever the gguf weight registry lives), add:
  ```rust
  pub enum WeightLayout {
      Replicated,
      ColParallel { world: u32, dim: usize },  // split output dim
      RowParallel { world: u32, dim: usize },  // split input dim
  }
  ```
  Define a `tp_layout(tensor_name: &str, world: u32) -> WeightLayout` function for `arch=qwen35` (dense path only; MoE layouts come in TP-4). Verify divisibility at registration (loader panics if `nQ % world != 0` etc.).
  - **Cert:** `certs/parity/qwen35_tp_layout_table.json` — for hidden_size=5120, nQ=64, nKV=8, intermediate=27648, world=4 — list of (tensor_name, layout, per_rank_shape). This is the contract.
  - **Files:** `crates/loaders/src/tp_layout.rs` (new).

- **TP-1b — slicing in the GGUF tensor copy path.** Where the loader currently does `device.upload(host_buf)`, branch on `WeightLayout`. ColParallel: each rank uploads only its slice rows (stride-aware; for quantised dtypes this requires slicing on block boundaries — `Q4_1`/`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` all have 32-element blocks, so divisibility carries through if the unsliced row count is divisible by 32×world). RowParallel: each rank takes a column slice — requires reading *all* rows but only `cols_per_rank` columns per row from disk.
  - **Risk:** RowParallel of a quantised tensor splits a single ggml block (32-element row-aligned). Can't slice a Q4_K block down the middle. Solution: row-parallel only over the **outer** rank-major dim of the matmul, which for `attn_output` and `ffn_down` is `H` (the unquantised input-dim direction). Verify in cert that all ColParallel splits land on block boundaries.
  - **Cert:** `certs/parity/qwen35_27b_q4_1_tp4_load.json` — load Qwen3.5-27B-Q4_1 on Mesh<4>, dump per-rank weight bytes total. Expect each rank ≈ 4.0 GiB (16.0 / 4).
  - **Files:** `crates/loaders/src/gguf.rs` and the `Qwen3MoEShardedModel::load` peer in qwen3-moe.

- **TP-1c — TP-aware shard model.** Mirror `Qwen3MoEShardedModel` (PP) with `Qwen3DenseTpModel` (or extend the existing struct with a `topology: Topology` field where `Topology = { Pp(LayerAssignment), Tp(TpLayout) }`). Smoke: load Qwen3.5-27B-Q4_1 with `--mesh-mode tp --tp-size 4`, no forward yet — just verify load completes and `cargo run -p cli -- inspect` reports the per-rank shard sizes.

### TP-2: TP decode forward (dense attention)

- **TP-2a — `RankForwardScratchTp`.** Mirror PP's `RankForwardScratch` but adds `partial_attn_out: DevicePtr` and `partial_ffn_out: DevicePtr` (each `H × max_batch × dtype` bytes). The `hidden` buffer becomes the AR'd-and-residualised value and is replicated across ranks (the AR kernel writes to it).
  - **Files:** `crates/models/qwen3-moe/src/forward/tp.rs` (new — sibling of `pp.rs`).

- **TP-2b — `forward_full_attn_decode_tp`.** Per rank:
  1. RMSNorm (replicated input, replicated output)
  2. Q/K/V proj using ColParallel weights — each rank emits `(nQ/N · D, 1)` Q and `(nKV/N · D, 1)` K, V locally
  3. Update KV cache (per-rank, `nKV/N` heads' worth)
  4. Attention (per-rank, `nQ/N` heads' worth) — the `attention_decode_f16` kernel is already head-parallel; just call with per-rank head count
  5. O proj using RowParallel weight — each rank emits a partial `(H, 1)` into `partial_attn_out`
  6. **Skip the AR for now** (will be added in TP-2c — first cut uses naive AR after both attn and FFN; deferred AR is TP-3a)
  7. Eager AR via `BarP2pAllReduce`, residual-add into `hidden`
  - **Files:** `crates/models/qwen3-moe/src/forward/tp.rs`, plus surgical edits in `forward/attn.rs` to expose a head-count-parametric variant.

- **TP-2c — `forward_dense_ffn_decode_tp`.** Same shape: gate/up ColParallel → SwiGLU on per-rank intermediate slice → down RowParallel → AR with residual into `hidden`. Re-use the V2.4.b shape-aware `indexed_moe_mmvq` r2/r4 variants? No — dense FFN is plain GEMV, use the already-certed `dense_gemv_f32_f16` path with sliced weights.

- **TP-2d — `forward_one_token_tp`.** End-to-end glue: embed (rank 0, then broadcast → use `Broadcast` collective via the same BAR1 P2P pattern; payload is `(H,)` so 10 KB), 64-layer loop alternating attn/FFN with two AR per layer, output norm (replicated), LM head (col-parallel V/N output, AR-arg-max via a small int-reduction kernel — for greedy, equivalent to `argmax` on each rank then `all_reduce_max` of (logit, idx) pairs).
  - **Cert (parity):** `certs/parity/qwen35_27b_q4_1_tp4_decode.json` — first-token greedy match vs Mesh<1> (use `forward_one_token_pp` with Mesh<1>::degenerate as oracle). Bit-exact at first argmax for seed 9419.
  - **Files:** `crates/models/qwen3-moe/src/forward/tp.rs`, `tests/parity_tp.rs`.

- **TP-2e — naive-AR perf cert.** `cargo run -p bench -- matrix --topology tp --tp-size 4 --models qwen35_27b_q4_1`. Expect decode 30–40 tok/s (no deferred AR yet — 128 AR/token at 79 µs is 10 ms over 27 ms decode budget; plus actual compute → tg ≈ 1/(0.027) ≈ 37 tok/s plausible). If we hit ≥ 35 tok/s here, the trajectory to ≥ 50 with TP-3a is real.
  - **Cert:** `certs/perf/v2_X_a_qwen35_27b_q4_1_tp4_decode_naive.json`.

### TP-3: AR optimisations (decode-only)

- **TP-3a — Deferred Attention AllReduce.** Refactor the layer body so that the post-attn AR is *not* issued; instead the partial `attn_local` is kept in a per-rank scratch buffer, the FFN runs on `x_in + AR_only_at_end_of_layer` (correct *only* if we apply RMSNorm on the AR'd-pre-attn-residual, which the standard pre-norm block already does — the FFN's input norm sees the residualised attn-AR'd activation). One AR per layer instead of two. 64 layers × 1 AR = 64 AR/token. Recheck parity bit-exact (it's algebraically equivalent, no fp drift).
  - **Cert (parity):** must remain bit-exact vs TP-2d.
  - **Cert (perf):** `certs/perf/v2_X_b_qwen35_27b_q4_1_tp4_decode_deferred.json`. Expect ≥ 48 tok/s.

- **TP-3b — fused gemv+AR+rmsnorm.** mi50grad's M3-extra. Skip if TP-3a hits the gate. If we're still ≤ 50 tok/s, port [`kernel_p2p_allreduce_rmsnorm.hip`](https://github.com/larkinwc/mi50grad/blob/main/src/kernels/kernel_p2p_allreduce_rmsnorm.hip) — fuses the down-proj reduction with peer-read AR with the next-layer RMSNorm, using a cross-WG atomic barrier. High-risk, high-reward; only if needed.

### TP-4: hybrid arch (Qwen3.6-35B-A3B and Coder-30B MoE)

- **TP-4a — GDN col-parallel by linear_num_v_heads.** Needs `linear_num_v_heads % N == 0`. Qwen3.6-35B-A3B: `linear_num_v_heads = 32` → divides 1/2/4/8. The fused `gdn_state_step_f32_s128` kernel and α/β fusion (V2.30.a, V2.4.d) already operate per-head — sharding heads is transparent at the kernel level. State alloc: each rank's GDN state is `linear_num_v_heads/N × linear_value_head_dim × n_layers_with_gdn`.
  - **Cert (parity):** `certs/parity/qwen36_35b_tp4_decode.json` — 8-token greedy bit-exact vs Mesh<1>.

- **TP-4b — MoE intra-expert sharding.** The router runs replicated (input is the AR'd hidden — same on all ranks → top-k indices identical). Each rank holds 1/N of every expert's gate/up/down. The existing `indexed_moe_mmvq_q4_k_*` kernels emit `(intermediate_per_expert, num_active_experts × n_tokens)` — sharding intermediate by N is a straight slice. Combine kernel runs per-rank, AR after.
  - **Risk:** `intermediate_size_per_expert = 768` for Qwen3.6-A3B. 768 / 4 = 192 — divisible. 768 / 8 = 96 — also OK. For models where it doesn't divide (Coder-30B Q5_K MoE down → check), TP-N restricted at load time with a clear error.
  - **Cert (parity):** `certs/parity/qwen3_coder_30b_tp4_decode.json`.

- **TP-4c — shared expert col/row TP.** Same shape as dense FFN. Trivial after TP-4b.

- **TP-4d — MoE perf cert + V2 milestone close.** `certs/perf/v2_X_c_qwen36_35b_tp4_decode.json`, `..._coder_30b_..._.json`. Targets per §4.

### TP-5: server integration + topology selection

- **TP-5a — CLI flag.** `flambeau serve --mesh-mode {pp,tp,pp+tp} --tp-size N`. Default `pp` (preserves V1 behaviour). Verify `--mesh-mode tp` round-trips through OpenAI smoke tests.
- **TP-5b — prefill-PP + decode-TP coexistence.** The interesting one: prefill on PP weight layout, decode on TP layout. Two paths:
  - (i) Reload between phases (slow, simple, only worth it for very long prefills).
  - (ii) **Default: load with TP layout, do prefill on TP-1-per-rank** (each rank does its own piece sequentially with full sharded weights — equivalent to running without AR and gathering at end). Loses 30-40 % prefill perf but avoids reload. This is what mi50grad does — their "TP-4 prefill 32 tok/s" is exactly this.
  - (iii) **Best: dual-storage.** Keep weights in TP layout. For prefill, broadcast each TP-shard back into a temporary "full" weight buffer on each rank (PCIe BW limited but one-time per layer), use existing PP prefill kernels. Only worth doing if (ii) regresses prefill > 25 %.
  - Defer the choice to measurement at TP-5b time.

### TP-6 (V2++): tooling + tuner

- **TP-6a** — `cli tune --tp-size N` integration. The warmup-tuner already exists for kernel selection; TP adds a topology choice (TP-N for N ∈ {1, 2, 4} given available GPUs). Pick the topology + AR variant that minimises decode latency at the session's measured prompt distribution.
- **TP-6b** — MCP `flambeau_dispatch_ab` extension to compare topologies. Out of session-1 scope.

---

## 6. Open questions / known unknowns

1. **ROCm version.** mi50grad pins `mixa3607/rocm-gfx906:7.1.0-complete`. Our `.env` already pins ROCm 7.1.1 (per project memory). Verify peer-access works on 7.1.1; if not, rebuild against 7.1.0 just for the cluster bring-up (gated by `rocm_version_supports_p2p_gfx906` cert).
2. **BAR1 size.** PCIe BAR1 must be at least as large as the per-peer mapped HBM region. Default BIOS BAR1 on consumer mobos is often 4 GB — too small for 32 GB MI50. Already a documented mi50grad prerequisite (server-class boards expose "Above 4G Decoding" + "Resizable BAR"). The flambeau-on-this-rig setup already has this configured (PP host-bounce works at 6.63 GB/s, which is the BAR1-mapped path); confirm with a `dd`-style probe in TP-0a.
3. **Argmax-AR.** Greedy decoding can do `local_argmax → AR(max(logit, idx_pair))`. With sampling (top-k, top-p, temperature), the right primitive is `gather logits → AR sum → sample on rank 0 → broadcast token`. Slightly more expensive but still small (V × dtype = 152 KB for V=152064 fp16 → ~13 µs at PCIe). Defer the design until TP-2d cert is green.
4. **Long-context (≥ 8K) memory layout.** Q8 KV cache + TP-shard means per-rank KV is `seq_len × nKV/N × D × 1 byte`. For 27B at N=4, seq=16k: 16384 × 2 × 128 × 1 = 4 MiB per layer × 64 = 256 MiB per rank. Comfortable. F16 doubles this — also OK. No new constraint vs. existing single-card budget.
5. **Speculative decoding interaction.** V2.33 was reverted; if it returns, draft + verify both need to be TP-aware (verify is a bigger batch of the same GEMV pattern, no new AR shape). Out of TP-V1 scope.

---

## 7. Glossary deltas

Add to `doc/GLOSSARY.md` (TP-0 deliverable):

- **TP** — tensor parallelism: per-layer weight sharding + AllReduce, decode-time topology.
- **BAR1 P2P** — PCIe Base Address Register 1 mapping that exposes peer GPU HBM as host-addressable; on ROCm enabled by `hipDeviceEnablePeerAccess`. ~12 GB/s effective on this rig.
- **Deferred AR** — the optimisation that skips the post-attention AllReduce by carrying rank-local attn output into the FFN block, reducing 2 AR/layer to 1.
- **Col-parallel** — output dim sharded; each rank computes `Y_r = X · W_r^T`, full Y obtained by concatenation (no AR).
- **Row-parallel** — input dim sharded; each rank computes `Y_r = X_r · W_r^T`, full Y obtained by AR-sum (AR mandatory).
- **Topology** — one of `Pp(LayerAssignment)`, `Tp(TpLayout)`, `PpTp { pp, tp }`. V1 = Pp; V2-tp adds Tp.

---

## 8. What this plan deliberately does NOT do

- No wide expert-parallel (AllToAll over experts). PCIe-bound, mi50grad correctly skips, CLAUDE.md correctly excludes.
- No TP for prefill at TP=4 unless measurement says otherwise. Prefill stays PP.
- No RCCL replacement. Existing `HipRankHandle::all_reduce_host` stays as a fallback / correctness oracle. New BAR1 path is an additional `AllReduce` impl, not a substitute.
- No HIP graph capture (M2 in mi50grad). Flambeau's existing async-stream + event-ordering already does what M2 buys mi50grad.
- No CUDA TP support in this plan. CUDA backend is V2+; TP for CUDA needs a separate plan keyed on NVLink topology — likely *simpler* there because NVLink AR is solved.
- No assumption that we beat mi50grad. Their published 56.3 tok/s is the calibration target; 50 tok/s on our V2.32.a Q4_1 is the aspirational gate. If we meet 50 we ship; if we beat 56 we celebrate.

---

## 9. Order of operations summary

```
TP-0a ── TP-0b ── TP-0c ──┐
                          │
                          ├── TP-1a ── TP-1b ── TP-1c ──┐
                          │                              │
                          │                              ├── TP-2a ── TP-2b ── TP-2c ── TP-2d ── TP-2e
                          │                              │                                          │
                          │                              │                                          ├── TP-3a ── (TP-3b)
                          │                              │                                          │
                          │                              │                                          └── TP-4a ── TP-4b ── TP-4c ── TP-4d
                          │                                                                                                       │
                          └────────────────────────────── TP-5a ── TP-5b ──────────────────────────────────────────────────────────┤
                                                                                                                                  │
                                                                                                                                  └── TP-6a ── (TP-6b)
```

Each box is a single PR. Critical path: **TP-0a → TP-0b → TP-0c → TP-2d** is the smallest sequence that produces a parity-green TP decode (~7 sessions). Adding TP-3a (~1 session) is the smallest sequence that hits the ≥ 50 tok/s gate.

---

## References

- mi50grad README: <https://github.com/larkinwc/mi50grad/blob/main/README.md>
- mi50grad RESEARCH.md: <https://github.com/larkinwc/mi50grad/blob/main/RESEARCH.md>
- BAR1 AR kernel (verbatim source we port): <https://github.com/larkinwc/mi50grad/blob/main/src/kernels/kernel_p2p_allreduce.hip>
- Fused AR+RMSNorm (TP-3b reference): <https://github.com/larkinwc/mi50grad/blob/main/src/kernels/kernel_p2p_allreduce_rmsnorm.hip>
- Tensor-parallel sharding (TP-1 reference): <https://github.com/larkinwc/mi50grad/blob/main/src/runtime/tensor_parallel.py>
- Megatron-LM original sharding paper: Shoeybi et al., 2019 — for the col/row-parallel + AR pattern.
- flambeau current PP: `crates/models/qwen3-moe/src/forward/pp.rs`, `crates/runtime/src/mesh.rs`.
- flambeau current AllReduce trait: `crates/runtime/src/collective.rs:149`.
- flambeau host-bounce reference: `crates/backend-hip/src/cluster.rs`.
