//! MoE building blocks — TopK router, IndexedMoE matmul (MMVQ r2, fused
//! gate+up, MMQ prefill), weighted combine.
//!
//! Expert bucketing (required by `indexed_moe_mmq_q4_k`) lives here as
//! [`build_expert_buckets`]. Byte-compatible with `sweep_moe::build_expert_buckets`;
//! the two will be folded into one helper when V1.7.3 consumes this.

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

/// `MMQ_X` for `indexed_moe_mmq_q4_k` — bucket width. Fixed in the kernel
/// source.
pub const INDEXED_MOE_MMQ_X: usize = 8;

/// `MMQ_Y` for `indexed_moe_mmq_q4_k` — output rows per block.
pub const INDEXED_MOE_MMQ_Y: usize = 16;

/// TopK router over per-token logits. Emits `(token, slot)` → expert index
/// plus normalised softmax weights over the k selected experts per token.
///
/// Shapes: `logits[n_tokens, n_experts]` F32 in; `idx[n_tokens, k]` i32 out;
/// `weights[n_tokens, k]` F32 out.
pub fn topk_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    logits: DevicePtr,
    idx: DevicePtr,
    weights: DevicePtr,
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) -> Result<()> {
    // Kernel's compile-time ceiling is TOPK_MAX_EXPERTS = 256. Launching
    // with more than that silently drops the tail and triggers OOB LDS
    // writes at the cross-warp reduce (V1.7.4.a root cause was this
    // hardcoded at 128, tripping on Qwen3.6's 256 experts).
    assert!(
        n_experts <= 256,
        "topk_f32: n_experts {n_experts} > 256 (bump TOPK_MAX_EXPERTS + add cert shape)"
    );
    let module = reg.expect_module("topk_f32")?;
    let kernel = module.kernel("flambeau_topk_softmax_f32")?;

    let n_tokens_i = n_tokens as i32;
    let n_experts_i = n_experts as i32;
    let k_i = k as i32;
    let l_ptr: u64 = logits.as_usize() as u64;
    let i_ptr: u64 = idx.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&l_ptr);
    args.push(&i_ptr);
    args.push(&w_ptr);
    args.push(&n_tokens_i);
    args.push(&n_experts_i);
    args.push(&k_i);
    // blockDim must be a multiple of wave64 and ≥64 so `n_warps = blockDim>>6`
    // is non-zero. Threads past n_experts load -INF via the kernel's guard.
    let block = ((n_experts.max(64) + 63) / 64 * 64) as u32;
    let cfg = LaunchCfg::one_d(n_tokens as u32, block);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Indexed MoE MMVQ (r2 variant) — the decode-path MoE matmul. Half the
/// launches of single-row, 2 output rows per wave64.
///
/// Shapes:
/// - `w[n_experts, n_rows, n_sb_per_row]` Q4_K blocks
/// - `y[n_tokens, n_sb_per_row * 8]` Q8_1 blocks
/// - `expert_ids[n_tokens, top_k]` i32
/// - `dst[n_tokens, top_k, n_rows]` F32
pub fn indexed_moe_mmvq_q4_k_r2(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    // V1.7.6.f productisation: DP4A is the default. `FLAMBEAU_VARIANT=baseline`
    // reverts to the scalar kernel for regression comparison only.
    let force_baseline = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let (stem, entry) = if force_baseline {
        ("indexed_moe_mmvq_q4_k_r2", "flambeau_indexed_moe_mmvq_q4_k_r2_q8_1")
    } else {
        ("indexed_moe_mmvq_q4_k_r2_dp4a", "flambeau_indexed_moe_mmvq_q4_k_r2_dp4a_q8_1")
    };
    let module = reg.expect_module(stem)?;
    let kernel = module.kernel(entry)?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let grid_x = ((n_rows as u32) + 1) / 2;
    let cfg = LaunchCfg {
        grid: (grid_x, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Q6_K sibling of `indexed_moe_mmvq_q4_k_r2`. Same indexing contract —
/// `[n_tokens, top_k]` expert ids, `[n_tokens, top_k, n_rows]` F32 output,
/// `[n_tokens, n_sb_per_row * 8]` Q8_1 activations — but weights are
/// Q6_K super-blocks. Needed for UD-Q4_K_S-style mixed-quant GGUFs where
/// some `ffn_down_exps` are promoted from Q4_K to Q6_K.
///
/// Single-row kernel (64 threads per block, one wave64, one output row
/// per block); a multi-row r2/r4 variant is the follow-up perf lever.
pub fn indexed_moe_mmvq_q6_k(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmvq_q6_k")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmvq_q6_k_q8_1")?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = LaunchCfg {
        grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
        block: (64, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused gate+up MoE MMVQ (candle P30). One launch does both `gate = W_g · x`
/// and `up = W_u · x` reading `x` only once. Shapes match
/// `indexed_moe_mmvq_q4_k_r2` but with two separate weight tensors and two
/// separate F32 output tensors.
pub fn indexed_moe_mmvq_q4_k_gate_up(
    reg: &OpsRegistry,
    stream: &HipStream,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    y: DevicePtr,
    expert_ids: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_rows: usize,
    n_tokens: usize,
    top_k: usize,
    n_sb_per_row: usize,
) -> Result<()> {
    // V1.7.6.f productisation: DP4A gate+up fusion is the default.
    // FLAMBEAU_VARIANT=baseline → unfused scalar (regression compare only).
    // FLAMBEAU_MBATCH=1 → llama.cpp-style top_k-warps-per-block mbatch (proven
    // null on MI50, kept as an A/B knob).
    let force_baseline = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let mbatch = std::env::var("FLAMBEAU_MBATCH").is_ok();
    let (stem, entry) = if force_baseline {
        (
            "indexed_moe_mmvq_q4_k_gate_up",
            "flambeau_indexed_moe_mmvq_q4_k_gate_up_q8_1",
        )
    } else if mbatch {
        (
            "indexed_moe_mmvq_q4_k_gate_up_mbatch",
            "flambeau_indexed_moe_mmvq_q4_k_gate_up_mbatch_q8_1",
        )
    } else {
        (
            "indexed_moe_mmvq_q4_k_gate_up_dp4a",
            "flambeau_indexed_moe_mmvq_q4_k_gate_up_dp4a_q8_1",
        )
    };
    let module = reg.expect_module(stem)?;
    let kernel = module.kernel(entry)?;

    let n_rows_i = n_rows as i32;
    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let nb_i = n_sb_per_row as i32;
    let g_ptr: u64 = w_gate.as_usize() as u64;
    let u_ptr: u64 = w_up.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let e_ptr: u64 = expert_ids.as_usize() as u64;
    let go_ptr: u64 = gate_out.as_usize() as u64;
    let uo_ptr: u64 = up_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&g_ptr);
    args.push(&u_ptr);
    args.push(&y_ptr);
    args.push(&e_ptr);
    args.push(&go_ptr);
    args.push(&uo_ptr);
    args.push(&n_rows_i);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&nb_i);
    let cfg = if mbatch {
        // llama.cpp-style MoE shape: block (64 × top_k × 1), grid (n_rows × n_tokens × 1).
        // top_k warps per block run independently, one per expert-slot.
        LaunchCfg {
            grid: (n_rows as u32, n_tokens as u32, 1),
            block: (64, top_k as u32, 1),
            shared_bytes: 0,
        }
    } else {
        LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        }
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Indexed MoE MMQ Q4_K — prefill path. Caller pre-sorts (token, slot) pairs
/// into per-expert buckets of up to MMQ_X=8 refs. See [`build_expert_buckets`].
///
/// Shapes:
/// - `w[n_experts, n_rows, n_sb_per_row]` Q4_K blocks
/// - `y[n_tokens, n_sb_per_row * 8]` Q8_1 blocks
/// - `bucket_expert[n_buckets]` i32
/// - `bucket_slots[n_buckets, MMQ_X]` i32 packed `(token << 16 | slot)`, `-1` sentinel
/// - `dst[n_tokens, top_k, n_rows]` F32
pub fn indexed_moe_mmq_q4_k(
    reg: &OpsRegistry,
    stream: &HipStream,
    w: DevicePtr,
    y: DevicePtr,
    bucket_expert: DevicePtr,
    bucket_slots: DevicePtr,
    dst: DevicePtr,
    n_rows: usize,
    n_sb_per_row: usize,
    top_k: usize,
    n_buckets: usize,
) -> Result<()> {
    let module = reg.expect_module("indexed_moe_mmq_q4_k")?;
    let kernel = module.kernel("flambeau_indexed_moe_mmq_q4_k_q8_1")?;

    let n_rows_i = n_rows as i32;
    let nb_i = n_sb_per_row as i32;
    let top_k_i = top_k as i32;
    let w_ptr: u64 = w.as_usize() as u64;
    let y_ptr: u64 = y.as_usize() as u64;
    let be_ptr: u64 = bucket_expert.as_usize() as u64;
    let bs_ptr: u64 = bucket_slots.as_usize() as u64;
    let d_ptr: u64 = dst.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&y_ptr);
    args.push(&be_ptr);
    args.push(&bs_ptr);
    args.push(&d_ptr);
    args.push(&n_rows_i);
    args.push(&nb_i);
    args.push(&top_k_i);
    let grid_x =
        ((n_rows as u32) + INDEXED_MOE_MMQ_Y as u32 - 1) / INDEXED_MOE_MMQ_Y as u32;
    let cfg = LaunchCfg {
        grid: (grid_x, n_buckets as u32, 1),
        block: (128, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Shared-expert gate scaling (Qwen3.5/3.6 hybrid). Computes per-token
/// `gate[t] = sigmoid(Σ_i gate_w[i] · x[t, i])` and multiplies each row of
/// `shared_out` by its token's `gate` in-place.
///
/// Caller typically computes `shared_out` first via gate/up/swiglu/down
/// dense FFN using the existing `qmatmul` + `swiglu_f16` ops on the
/// `ffn_*_shexp` weights, then calls this to apply the learned scalar gate.
pub fn shared_expert_scale_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    shared_out: DevicePtr,  // in-place [n_tokens, hidden]
    x: DevicePtr,           // [n_tokens, hidden] (layer input)
    gate_w: DevicePtr,      // [hidden]
    n_tokens: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("shared_expert_scale_f32")?;
    let kernel = module.kernel("flambeau_shared_expert_scale_f32")?;
    let n_tokens_i = n_tokens as i32;
    let hidden_i = hidden as i32;
    let so_ptr: u64 = shared_out.as_usize() as u64;
    let x_ptr: u64 = x.as_usize() as u64;
    let w_ptr: u64 = gate_w.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&so_ptr);
    args.push(&x_ptr);
    args.push(&w_ptr);
    args.push(&n_tokens_i);
    args.push(&hidden_i);
    let cfg = LaunchCfg::one_d(n_tokens as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Weighted combine + residual — `out = residual + Σ_k w_k · expert_out[k]`.
/// One thread per `(token, hidden)` slot. All tensors F16 except
/// `weights[n_tokens, top_k]` F32.
pub fn moe_combine_f16(
    reg: &OpsRegistry,
    stream: &HipStream,
    expert_outs: DevicePtr,
    weights: DevicePtr,
    residual: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    let module = reg.expect_module("moe_combine_f16")?;
    let kernel = module.kernel("flambeau_moe_combine_f16")?;

    let n_tokens_i = n_tokens as i32;
    let top_k_i = top_k as i32;
    let hidden_i = hidden as i32;
    let e_ptr: u64 = expert_outs.as_usize() as u64;
    let w_ptr: u64 = weights.as_usize() as u64;
    let r_ptr: u64 = residual.as_usize() as u64;
    let o_ptr: u64 = out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&e_ptr);
    args.push(&w_ptr);
    args.push(&r_ptr);
    args.push(&o_ptr);
    args.push(&n_tokens_i);
    args.push(&top_k_i);
    args.push(&hidden_i);
    let total = n_tokens * hidden;
    let cfg = LaunchCfg::one_d(((total + 255) / 256) as u32, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Group `(token, slot)` pairs by expert into buckets of up to [`INDEXED_MOE_MMQ_X`]
/// refs. Returns `(bucket_expert, bucket_slots)` where `bucket_slots[i, col]`
/// is either `token << 16 | slot` or `-1` sentinel padding. Deterministic —
/// experts are emitted in ascending-id order.
pub fn build_expert_buckets(
    expert_ids: &[i32],
    n_tokens: usize,
    top_k: usize,
) -> (Vec<i32>, Vec<i32>) {
    use std::collections::HashMap;
    let mut per_expert: HashMap<i32, Vec<i32>> = HashMap::new();
    for t in 0..n_tokens {
        for slot in 0..top_k {
            let e = expert_ids[t * top_k + slot];
            let packed = ((t as i32) << 16) | (slot as i32);
            per_expert.entry(e).or_default().push(packed);
        }
    }
    let mut experts: Vec<i32> = per_expert.keys().copied().collect();
    experts.sort();
    let mut bucket_expert = Vec::new();
    let mut bucket_slots = Vec::new();
    for e in experts {
        // Safe by construction: `experts` comes from `per_expert.keys()`
        // moments earlier with no concurrent mutation.
        let refs = per_expert
            .remove(&e)
            .expect("expert key enumerated but missing on remove");
        for chunk in refs.chunks(INDEXED_MOE_MMQ_X) {
            bucket_expert.push(e);
            for &r in chunk {
                bucket_slots.push(r);
            }
            for _ in chunk.len()..INDEXED_MOE_MMQ_X {
                bucket_slots.push(-1);
            }
        }
    }
    (bucket_expert, bucket_slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketing_groups_by_expert_and_pads() {
        // 3 tokens × top_k=2. Experts chosen to produce one bucket with a
        // tail-pad and one full bucket.
        let ids = vec![0i32, 1, 0, 1, 0, 2];
        let (be, bs) = build_expert_buckets(&ids, 3, 2);
        // Expert 0 has 3 refs → one bucket with 5× -1 tail pad.
        // Expert 1 has 2 refs → one bucket with 6× -1 tail pad.
        // Expert 2 has 1 ref  → one bucket with 7× -1 tail pad.
        assert_eq!(be, vec![0, 1, 2]);
        assert_eq!(bs.len(), 3 * INDEXED_MOE_MMQ_X);
        // Expert 0 refs: (t=0,slot=0), (t=1,slot=0), (t=2,slot=0).
        assert_eq!(bs[0], (0 << 16) | 0);
        assert_eq!(bs[1], (1 << 16) | 0);
        assert_eq!(bs[2], (2 << 16) | 0);
        assert_eq!(bs[3], -1);
        // Expert 1 refs: (t=0,slot=1), (t=1,slot=1).
        let b1 = INDEXED_MOE_MMQ_X;
        assert_eq!(bs[b1], (0 << 16) | 1);
        assert_eq!(bs[b1 + 1], (1 << 16) | 1);
        assert_eq!(bs[b1 + 2], -1);
    }

    #[test]
    fn bucketing_splits_chunks_over_mmq_x() {
        // One expert hit 10 times → two buckets: full 8 + tail 2.
        let ids: Vec<i32> = (0..10).map(|_| 7i32).collect();
        let (be, bs) = build_expert_buckets(&ids, 5, 2);
        assert_eq!(be, vec![7, 7]);
        assert_eq!(bs.len(), 2 * INDEXED_MOE_MMQ_X);
        // Second bucket has 2 real refs (entries 8 and 9 = (t=4,slot=0)
        // and (t=4,slot=1)) then 6 sentinels.
        let b2 = INDEXED_MOE_MMQ_X;
        assert_eq!(bs[b2], (4 << 16) | 0);
        assert_eq!(bs[b2 + 1], (4 << 16) | 1);
        for i in 2..INDEXED_MOE_MMQ_X {
            assert_eq!(bs[b2 + i], -1);
        }
    }
}
