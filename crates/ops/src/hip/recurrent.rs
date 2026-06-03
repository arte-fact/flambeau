//! Recurrent-layer kernels (Gated-Delta-Net).
//! fused per-(b,h) autoregressive + prefill step kernel. One launch
//! runs the recurrence loop over `L` tokens with the state held in registers,
//! so decode (L=1) and prefill (L>1) share the same dispatch path.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a kernel.launch or memcpy_async over \
              DevicePtrs validated by the caller; the kernel stem + entry are resolved \
              through the validated registry and the ABI matches the kernels extern-C \
              signature."
)]

use anyhow::Result;
use flambeau_backend_hip::{HipStream, KernelArgs, LaunchCfg};
use flambeau_core::DevicePtr;

use super::OpsRegistry;

/// Fused GDN step. S_v = 128 (Qwen3.6-35B: `head_k_dim = head_v_dim = 128`).
/// Shapes (row-major, contiguous, F32):
/// - `q, k`: `(B, H_kv, L, 128)` where `H_kv = H / n_rep`
/// - `v`: `(B, H, L, 128)`
/// - `gate, beta`: `(B, H, L)` — one scalar per (b, h, t)
/// - `state_in`: `(B, H, 128, 128)` stored col-outer (see kernel).
/// - `state_out`: same shape and layout; may alias `state_in` (the kernel
///   loads state_in into registers at entry and writes state_out at exit).
/// - `attn_out`: `(B, H, L, 128)`
///   `n_rep = H_v / H_kv` drives implicit GQA broadcast for Q/K (no caller-side
///   expand). `n_rep = 1` reduces to the no-GQA path.
///   Launch: grid `(H, B, ceil(S_v / warps_per_block))`, block
///   `(WARP_SIZE=64, 4, 1)` — 4 warps per block, each warp owns one output
///   column. `warps_per_block = 4` ⇒ grid_z = `S_v / 4 = 32` at S_v=128.
pub fn gdn_state_step_f32_s128(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnStepBuffers,
    shape: crate::GdnStepShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnStepBuffers { q, k, v, gate, beta, state_in, state_out, attn_out } = bufs;
    let crate::GdnStepShape { b, h_v, l, n_rep, rep_inner_layout } = shape;
    const S_V: u32 = 128;
    const WARP_SIZE: u32 = 64;
    const WARPS_PER_BLOCK: u32 = 4;

    let module = reg.expect_module("gdn_state_step_f32")?;
    let kernel = module.kernel("flambeau_gdn_state_step_f32_s128")?;

    let b_i = b as i32;
    let h_i = h_v as i32;
    let l_i = l as i32;
    let n_rep_i = n_rep as i32;
    let rep_inner_i: i32 = if rep_inner_layout { 1 } else { 0 };
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let gate_ptr: u64 = gate.as_usize() as u64;
    let beta_ptr: u64 = beta.as_usize() as u64;
    let sin_ptr: u64 = state_in.as_usize() as u64;
    let sout_ptr: u64 = state_out.as_usize() as u64;
    let ao_ptr: u64 = attn_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&gate_ptr);
    args.push(&beta_ptr);
    args.push(&sin_ptr);
    args.push(&sout_ptr);
    args.push(&ao_ptr);
    args.push(&b_i);
    args.push(&h_i);
    args.push(&l_i);
    args.push(&n_rep_i);
    args.push(&rep_inner_i);

    let grid_z = S_V / WARPS_PER_BLOCK;
    let cfg = LaunchCfg {
        grid: (h_v as u32, b as u32, grid_z),
        block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused GDN α/β/gate compute over `n_tokens × num_v_heads` F32 elements:
/// gate_out[t, i] = softplus(alpha_in[t, i] + ssm_dt_bias[i]) * ssm_a[i]
/// beta_out[t, i] = sigmoid(beta_in[t, i])
/// Per-head constants `ssm_dt_bias` and `ssm_a` are shared across all
/// `n_tokens` rows. Grid = `(n_tokens, 1, 1)`, block = `(num_v_heads,
/// 1, 1)`. For decode `n_tokens = 1`; for prefill `n_tokens = L`.
/// batched across tokens in one launch. Previously the
/// caller looped L times at one-token-per-launch; at pp512 × 16 GDN
/// layers that fired 12k tiny launches dominated by argument
/// marshalling (profile 2026-04-22: ~50 ms in the kernel + ~150 ms
/// rocclr_copyBuffer overhead).
pub fn gdn_alpha_beta_f32(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnAlphaBetaBuffers,
    shape: crate::GdnAlphaBetaShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnAlphaBetaBuffers {
        alpha_in,
        beta_in,
        ssm_dt_bias,
        ssm_a,
        gate_out,
        beta_out,
    } = bufs;
    let crate::GdnAlphaBetaShape { num_v_heads, n_tokens } = shape;
    assert!(n_tokens >= 1, "gdn_alpha_beta_f32 needs n_tokens >= 1");
    assert!(
        num_v_heads <= 1024,
        "gdn_alpha_beta_f32 expects num_v_heads <= block-size cap 1024"
    );
    let module = reg.expect_module("gdn_alpha_beta_f32")?;
    let kernel = module.kernel("flambeau_gdn_alpha_beta_f32")?;
    let num_v_i = num_v_heads as i32;
    let n_tokens_i = n_tokens as i32;
    let a_ptr: u64 = alpha_in.as_usize() as u64;
    let b_ptr: u64 = beta_in.as_usize() as u64;
    let dt_ptr: u64 = ssm_dt_bias.as_usize() as u64;
    let sa_ptr: u64 = ssm_a.as_usize() as u64;
    let g_ptr: u64 = gate_out.as_usize() as u64;
    let bo_ptr: u64 = beta_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&dt_ptr);
    args.push(&sa_ptr);
    args.push(&g_ptr);
    args.push(&bo_ptr);
    args.push(&num_v_i);
    args.push(&n_tokens_i);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, 1, 1),
        block: (num_v_heads as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// **C10** — fused GDN recurrent step that absorbs the preceding
/// `gdn_alpha_beta_f32` launch. Same launch shape as
/// [`gdn_state_step_f32_s128`] but reads the raw mmvq outputs
/// `alpha_in[B,L,H]` / `beta_in[B,L,H]` plus per-head `ssm_dt_bias` /
/// `ssm_a` constants directly, computing `softplus`/`sigmoid` inline
/// in the per-token loop. Saves one launch per GDN layer per token.
/// Numerically identical to the unfused chain at FP32 (same op order,
/// same warp-reduce signatures); the cert sweep verifies parity
/// against `gdn_state_step_f32_s128 ∘ gdn_alpha_beta_f32` on Qwen3.6
/// shapes.
pub fn gdn_state_step_alphabeta_f32_s128(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnStepAlphaBetaBuffers,
    shape: crate::GdnStepShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnStepAlphaBetaBuffers {
        q,
        k,
        v,
        alpha_in,
        beta_in,
        ssm_dt_bias,
        ssm_a,
        state_in,
        state_out,
        attn_out,
    } = bufs;
    let crate::GdnStepShape { b, h_v, l, n_rep, rep_inner_layout } = shape;
    const S_V: u32 = 128;
    const WARP_SIZE: u32 = 64;
    const WARPS_PER_BLOCK: u32 = 4;

    let module = reg.expect_module("gdn_state_step_alphabeta_f32")?;
    let kernel = module.kernel("flambeau_gdn_state_step_alphabeta_f32_s128")?;

    let b_i = b as i32;
    let h_i = h_v as i32;
    let l_i = l as i32;
    let n_rep_i = n_rep as i32;
    let rep_inner_i: i32 = if rep_inner_layout { 1 } else { 0 };
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let alpha_ptr: u64 = alpha_in.as_usize() as u64;
    let beta_ptr: u64 = beta_in.as_usize() as u64;
    let dt_ptr: u64 = ssm_dt_bias.as_usize() as u64;
    let sa_ptr: u64 = ssm_a.as_usize() as u64;
    let sin_ptr: u64 = state_in.as_usize() as u64;
    let sout_ptr: u64 = state_out.as_usize() as u64;
    let ao_ptr: u64 = attn_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&alpha_ptr);
    args.push(&beta_ptr);
    args.push(&dt_ptr);
    args.push(&sa_ptr);
    args.push(&sin_ptr);
    args.push(&sout_ptr);
    args.push(&ao_ptr);
    args.push(&b_i);
    args.push(&h_i);
    args.push(&l_i);
    args.push(&n_rep_i);
    args.push(&rep_inner_i);

    let grid_z = S_V / WARPS_PER_BLOCK;
    let cfg = LaunchCfg {
        grid: (h_v as u32, b as u32, grid_z),
        block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Batched-slots variant of [`gdn_state_step_alphabeta_f32_s128`]. Same
/// per-(B,H,col) compute and grid shape; the state-in / state-out
/// pointers per batch are dereferenced through a `[B] u64` device
/// pointer array instead of computed by stride from a single base.
/// Lets `forward_gdn_decode_batched_tp` collapse its per-slot
/// state-step loop into one launch when each slot's
/// `GdnLayerState::state` lives in a separate allocation.
///
/// `state_in_ptrs` and `state_out_ptrs` must each point to a `B`-entry
/// `u64` array on the device containing the per-slot
/// `GdnLayerState::state` base pointers. Same buffer for both is fine
/// when in-place (the existing kernel's pattern).
pub fn gdn_state_step_alphabeta_f32_s128_batched_slots(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnStepAlphaBetaBatchedSlotsBuffers,
    shape: crate::GdnStepShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnStepAlphaBetaBatchedSlotsBuffers {
        q,
        k,
        v,
        alpha_in,
        beta_in,
        ssm_dt_bias,
        ssm_a,
        state_in_ptrs,
        state_out_ptrs,
        attn_out,
    } = bufs;
    let crate::GdnStepShape { b, h_v, l, n_rep, rep_inner_layout } = shape;
    const S_V: u32 = 128;
    const WARP_SIZE: u32 = 64;
    const WARPS_PER_BLOCK: u32 = 4;

    let module = reg.expect_module("gdn_state_step_alphabeta_f32_batched_slots")?;
    let kernel = module.kernel("flambeau_gdn_state_step_alphabeta_f32_s128_batched_slots")?;

    let b_i = b as i32;
    let h_i = h_v as i32;
    let l_i = l as i32;
    let n_rep_i = n_rep as i32;
    let rep_inner_i: i32 = if rep_inner_layout { 1 } else { 0 };
    let q_ptr: u64 = q.as_usize() as u64;
    let k_ptr: u64 = k.as_usize() as u64;
    let v_ptr: u64 = v.as_usize() as u64;
    let alpha_ptr: u64 = alpha_in.as_usize() as u64;
    let beta_ptr: u64 = beta_in.as_usize() as u64;
    let dt_ptr: u64 = ssm_dt_bias.as_usize() as u64;
    let sa_ptr: u64 = ssm_a.as_usize() as u64;
    let sin_arr: u64 = state_in_ptrs.as_usize() as u64;
    let sout_arr: u64 = state_out_ptrs.as_usize() as u64;
    let ao_ptr: u64 = attn_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&alpha_ptr);
    args.push(&beta_ptr);
    args.push(&dt_ptr);
    args.push(&sa_ptr);
    args.push(&sin_arr);
    args.push(&sout_arr);
    args.push(&ao_ptr);
    args.push(&b_i);
    args.push(&h_i);
    args.push(&l_i);
    args.push(&n_rep_i);
    args.push(&rep_inner_i);

    let grid_z = S_V / WARPS_PER_BLOCK;
    let cfg = LaunchCfg {
        grid: (h_v as u32, b as u32, grid_z),
        block: (WARP_SIZE, WARPS_PER_BLOCK, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Fused single-token GDN conv trio (assemble + causal_conv1d + shift)
/// across `N` decode slots, each owning its own `conv_history` buffer.
/// One launch replaces 3N per-slot launches (DtoD assemble + conv1d +
/// DtoD shift) at GDN-batched-decode. Each block handles a
/// `(slot, channel_tile)` pair; all 3 ops happen in registers (no LDS,
/// no inter-block sync). Caller writes per-slot history base pointers
/// into a device array sized `[N] u64` and passes its base via
/// `slot_history_ptrs`.
pub fn gdn_conv_trio_decode_f32_batched_slots(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnConvTrioBatchedSlotsBuffers,
    shape: crate::GdnConvTrioShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnConvTrioBatchedSlotsBuffers {
        slot_history_ptrs,
        qkv_mixed,
        weight,
        conv_out,
    } = bufs;
    let crate::GdnConvTrioShape { n_slots, conv_channels, conv_kernel } = shape;
    const THREADS: u32 = 256;
    const KERNEL_MAX: usize = 8;
    assert!(
        conv_kernel <= KERNEL_MAX,
        "gdn_conv_trio_decode_f32_batched_slots: conv_kernel {conv_kernel} > KERNEL_MAX {KERNEL_MAX}"
    );
    let module = reg.expect_module("gdn_conv_trio_decode_f32_batched_slots")?;
    let kernel = module.kernel("flambeau_gdn_conv_trio_decode_f32_batched_slots")?;

    let n_slots_i = n_slots as i32;
    let conv_channels_i = conv_channels as i32;
    let conv_kernel_i = conv_kernel as i32;
    let ptrs_arr: u64 = slot_history_ptrs.as_usize() as u64;
    let q_ptr: u64 = qkv_mixed.as_usize() as u64;
    let w_ptr: u64 = weight.as_usize() as u64;
    let o_ptr: u64 = conv_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&ptrs_arr);
    args.push(&q_ptr);
    args.push(&w_ptr);
    args.push(&o_ptr);
    args.push(&n_slots_i);
    args.push(&conv_channels_i);
    args.push(&conv_kernel_i);
    let cfg = LaunchCfg {
        grid: ((conv_channels as u32).div_ceil(THREADS), n_slots as u32, 1),
        block: (THREADS, 1, 1),
        shared_bytes: 0,
    };
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// 3.d.1 — fused `conv_input = [history, current]`. Replaces the two
/// back-to-back DtoD memcpys in `forward/gdn.rs::assemble_conv_input` (decode
/// path) with a single elementwise kernel. Each GDN layer at decode fires
/// this pattern once per token.
pub fn gdn_assemble_conv_input_f32(
    reg: &OpsRegistry,
    stream: &HipStream,
    history: DevicePtr,
    current: DevicePtr,
    conv_input: DevicePtr,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let module = reg.expect_module("gdn_assemble_conv_input_f32")?;
    let kernel = module.kernel("flambeau_gdn_assemble_conv_input_f32")?;
    let total = conv_channels * conv_kernel;
    let h_ptr: u64 = history.as_usize() as u64;
    let c_ptr: u64 = current.as_usize() as u64;
    let o_ptr: u64 = conv_input.as_usize() as u64;
    let cc_i = conv_channels as i32;
    let ck_i = conv_kernel as i32;
    let mut args = KernelArgs::new();
    args.push(&h_ptr);
    args.push(&c_ptr);
    args.push(&o_ptr);
    args.push(&cc_i);
    args.push(&ck_i);
    const BLOCK: u32 = 256;
    let grid = (total as u32).div_ceil(BLOCK);
    let cfg = LaunchCfg::one_d(grid, BLOCK);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// fused split: replaces the 3×L DtoD memcpy loop in
/// `forward.rs::gather_qkv_strided`. Reads one row of silu_out per token
/// and strided-writes into q_out / k_out / v_out.
pub fn gdn_split_qkv_f32(
    ctx: crate::OpCtx<'_>,
    bufs: crate::GdnSplitQkvBuffers,
    shape: crate::GdnSplitQkvShape,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::GdnSplitQkvBuffers { silu_out, q_out, k_out, v_out } = bufs;
    let crate::GdnSplitQkvShape { n_tokens, qk_size, v_size } = shape;
    let module = reg.expect_module("gdn_split_qkv_f32")?;
    let kernel = module.kernel("flambeau_gdn_split_qkv_f32")?;
    let conv_channels = 2 * qk_size + v_size;
    let total = n_tokens * conv_channels;
    let s_ptr: u64 = silu_out.as_usize() as u64;
    let q_ptr: u64 = q_out.as_usize() as u64;
    let k_ptr: u64 = k_out.as_usize() as u64;
    let v_ptr: u64 = v_out.as_usize() as u64;
    let n_tokens_i = n_tokens as i32;
    let qk_i = qk_size as i32;
    let v_i = v_size as i32;
    let mut args = KernelArgs::new();
    args.push(&s_ptr);
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&n_tokens_i);
    args.push(&qk_i);
    args.push(&v_i);
    const BLOCK: u32 = 256;
    let grid = (total as u32).div_ceil(BLOCK);
    let cfg = LaunchCfg::one_d(grid, BLOCK);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
