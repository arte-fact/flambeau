//! TP-sharded MoE FFN upload for gemma4 26B-A4B.
//!
//! Mirrors [`crate::weights_hip::upload_moe_layer`] but produces a
//! per-rank slice of the MoE expert weights along the intermediate
//! axis. Per-rank shapes:
//!
//! | Tensor | Single-device | Per-rank (TP) |
//! |----------------------------|--------------------------------|----------------------------------|
//! | `ffn_gate_inp` | `[n_experts, hidden]` | `[n_experts, hidden]` Replicated |
//! | pre_router_weight | `[hidden]` F16 | `[hidden]` F16 Replicated |
//! | `ffn_gate_up_exps` | `[n_experts, 2·n_ff_exp, hidden]` | gate + up each `[n_experts, n_ff_exp_local, hidden]` (host-split + ColParallel{dim=1}) |
//! | `ffn_down_exps` | `[n_experts, hidden, n_ff_exp]` | `[n_experts, hidden, n_ff_exp_local]` RowParallel{dim=2} |
//! | 3 extra norms | `[hidden]` F16 | `[hidden]` F16 Replicated |
//!
//! Policy is declared in [`crate::tp_layout::Gemma4TpLayout`]; this
//! module is the upload-side implementation.
//!
//! Block-alignment: the `ffn_down_exps` inner-dim split requires
//! `n_ff_exp_local % 32 == 0` for Q8_0 / Q4_0; checked by
//! [`flambeau_runtime::tp_slice::slice_for_tp`].

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_blocks::{
    ggml_to_qdtype as blocks_ggml_to_qdtype, row_bytes_for_dtype as blocks_row_bytes,
    upload_replicated_norm_f32_to_f16, upload_replicated_tensor, upload_sharded_tensor, Activation,
    DenseMlpDecodeScratch, DenseMlpTp, MoeExperts, MoeExpertsDecodeScratch, RawAllocTracker,
    WeightHandle,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_ops::Ops;
use flambeau_quant::{GgmlDType, GgufFile};
use flambeau_runtime::WeightLayout;
use half::f16;

use crate::config::MoeDims;
use crate::layer::Gemma4LayerWeights;
use crate::names::MoeFfnNames;
use crate::weights_hip::DeviceTensor;

/// Per-rank decode scratch for the gemma4 MoE FFN composer. Sized
/// for `hidden` + `local_inter` + `n_experts` + `top_k`; allocated
/// once per rank when `cfg.moe.is_some()` (all 26B-A4B layers are
/// MoE so sharing one scratch across the layer loop is correct).
///
/// Pointer fields hold the gemma4-specific intermediate buffers
/// (mirroring `Gemma4MoeScratch` for the single-device path) plus
/// the embedded `MoeExpertsDecodeScratch` view consumed by
/// `MoeExperts::forward_decode_tp`.
#[derive(Copy, Clone)]
pub struct Gemma4TpMoeScratch {
    /// F16 `[hidden]` — `rmsnorm_f16(attn_residual, pre_router_weight)`.
    pub router_input_f16: DevicePtr,
    /// F16 `[hidden]` — post-shared-MLP norm output (full hidden, AR'd).
    pub cur_mlp_f16: DevicePtr,
    /// F16 `[hidden]` — pre-MoE input / post-MoE norm output.
    pub cur_moe_f16: DevicePtr,
    /// F16 `[hidden]` — `cur_mlp + cur_moe` intermediate (before final
    /// `post_ffw_norm` + residual add).
    pub cur_combined_f16: DevicePtr,
    /// F16 `[hidden]` — read-only zero buffer fed as `residual` to
    /// `MoeExperts::forward_decode_tp` (no fused residual; the
    /// per-branch norm + residual add happen in the composer).
    pub zero_hidden_f16: DevicePtr,
    /// F16 `[hidden]` — second row-parallel partial. The shared-MLP
    /// branch reuses `stage.partial_ffn`; the MoE branch writes here.
    /// AR'd in Phase 5b (separately from `partial_ffn`) under
    /// `post_ffw_norm_2`.
    pub partial_moe_f16: DevicePtr,
    /// `MoeExperts` decode scratch view, sized for `local_inter`.
    pub moe_scratch: MoeExpertsDecodeScratch,
}

impl Gemma4TpMoeScratch {
    /// Allocate per-rank scratch on `device`. Allocations are tracked
    /// in `raw_alloc` so the stage's `dispose()` frees them in one
    /// pass.
    pub fn alloc(
        device: &HipDevice,
        hidden: usize,
        local_inter: usize,
        n_experts: usize,
        top_k: usize,
        raw_alloc: &mut RawAllocTracker,
    ) -> Result<Self> {
        if hidden % 32 != 0 {
            bail!("Gemma4TpMoeScratch::alloc: hidden={hidden} not a multiple of 32");
        }
        let inter_total = top_k * local_inter;
        if inter_total % 32 != 0 {
            bail!(
                "Gemma4TpMoeScratch::alloc: top_k*local_inter={inter_total} not a multiple of 32"
            );
        }
        let router_input_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let cur_mlp_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let cur_moe_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let cur_combined_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let zero_hidden_f16 = raw_alloc.alloc_f16(device, hidden)?.0; // zeroed by alloc_f16
        let partial_moe_f16 = raw_alloc.alloc_f16(device, hidden)?.0;

        let moe_scratch = MoeExpertsDecodeScratch {
            x_q8_1: raw_alloc.alloc_q8_1(device, hidden)?.0,
            router_logits: raw_alloc.alloc_f32(device, n_experts)?.0,
            expert_ids: raw_alloc.alloc_i32(device, top_k)?.0,
            expert_weights: raw_alloc.alloc_f32(device, top_k)?.0,
            gate_out_f32: raw_alloc.alloc_f32(device, inter_total)?.0,
            up_out_f32: raw_alloc.alloc_f32(device, inter_total)?.0,
            activated_f16: raw_alloc.alloc_f16(device, inter_total)?.0,
            activated_q8_1: raw_alloc.alloc_q8_1(device, inter_total)?.0,
            down_f32: raw_alloc.alloc_f32(device, top_k * hidden)?.0,
            down_f16: raw_alloc.alloc_f16(device, top_k * hidden)?.0,
        };

        Ok(Self {
            router_input_f16,
            cur_mlp_f16,
            cur_moe_f16,
            cur_combined_f16,
            zero_hidden_f16,
            partial_moe_f16,
            moe_scratch,
        })
    }
}

/// Per-rank MoE FFN weights for one TP-sharded gemma4 26B-A4B layer.
/// Same field shape as [`crate::moe::Gemma4MoeFfnWeights`] but the
/// embedded `MoeExperts` block is sized for `local_inter = n_ff_exp /
/// world` and the gate/up/down pointers reference the per-rank slice.
pub struct Gemma4TpMoeFfnWeights {
    /// Routed experts block. `MoeExperts::new` is called with
    /// `intermediate = local_inter`; the gate/up/down `WeightHandle`s
    /// point at the per-rank sliced buffers.
    pub moe: MoeExperts,
    /// F16 `[hidden]` — `(1/sqrt(hidden)) * ffn_gate_inp.scale`
    /// pre-multiplied. Replicated.
    pub pre_router_weight_f16: DevicePtr,
    /// F16 `[hidden]` — pre-MoE-branch RMSNorm weight. Replicated.
    pub pre_ffw_norm_2: DevicePtr,
    /// F16 `[hidden]` — post-shared-MLP RMSNorm weight. Replicated.
    pub post_ffw_norm_1: DevicePtr,
    /// F16 `[hidden]` — post-MoE-branch RMSNorm weight. Replicated.
    pub post_ffw_norm_2: DevicePtr,
}

/// Upload one MoE FFN layer's weights TP-sharded for `rank` of `world`.
///
/// Returns per-rank-shaped [`Gemma4TpMoeFfnWeights`] whose `MoeExperts`
/// block is sized for `local_inter`; the caller's
/// `MoeExperts::forward_decode_tp` produces a row-parallel partial
/// that's AR-summed by the layer composer.
///
/// Mirrors the single-device [`crate::weights_hip::upload_moe_layer`]
/// step-by-step; differences (per-rank slicing of expert tensors) are
/// commented inline.
///
/// # Errors
/// - `world == 0` or `n_ff_exp % world != 0`.
/// - Required MoE tensors missing from the GGUF.
/// - Quant block-alignment failures on the per-rank `n_ff_exp_local`
///   slice for `ffn_down_exps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_moe_layer_tp(
    file: &GgufFile,
    layer_index: usize,
    hidden: usize,
    moe_dims: MoeDims,
    world: u32,
    rank: u32,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<Gemma4TpMoeFfnWeights> {
    if world == 0 {
        bail!("upload_moe_layer_tp: world=0");
    }
    if rank >= world {
        bail!("upload_moe_layer_tp: rank {rank} >= world {world}");
    }
    let names = MoeFfnNames::for_layer(layer_index);
    let n_experts = moe_dims.num_experts;
    let n_ff_exp = moe_dims.moe_intermediate_size;
    let top_k = moe_dims.num_experts_per_tok;
    let w = world as usize;
    if n_ff_exp % w != 0 {
        bail!(
            "upload_moe_layer_tp: n_ff_exp={n_ff_exp} not divisible by world={world}"
        );
    }
    let local_inter = n_ff_exp / w;

    // 1. Router (F32 `[n_experts, hidden]`) — Replicated on every rank.
    let router_info = file
        .tensors
        .get(&names.ffn_gate_inp)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_inp))?;
    if router_info.dtype != GgmlDType::F32 {
        bail!(
            "{} expected F32, got {:?}",
            names.ffn_gate_inp,
            router_info.dtype
        );
    }
    if router_info.dims != [n_experts as u64, hidden as u64] {
        bail!(
            "{} dims {:?} != [{}, {}]",
            names.ffn_gate_inp,
            router_info.dims,
            n_experts,
            hidden
        );
    }
    let router = ut_to_dt(upload_replicated_tensor(
        file, router_info, device, stream, tracker,
    )?);

    // 2. Pre-router weight (`(1/sqrt(hidden)) * ffn_gate_inp.scale`,
    //    cast F16). Per-rank computed identically; values bit-identical
    //    across ranks.
    let pre_router_ptr =
        upload_pre_router_weight(file, &names, hidden, device, stream, tracker)?;

    // 3. Split fused `ffn_gate_up_exps` host-side then per-rank slice
    //    along the intermediate axis (dim 1). Output: per-rank gate
    //    and up each `[n_experts, local_inter, hidden]`.
    let fused_info = file
        .tensors
        .get(&names.ffn_gate_up_exps)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_up_exps))?;
    if fused_info.dims
        != [n_experts as u64, (2 * n_ff_exp) as u64, hidden as u64]
    {
        bail!(
            "{} dims {:?} != [{}, {}, {}]",
            names.ffn_gate_up_exps,
            fused_info.dims,
            n_experts,
            2 * n_ff_exp,
            hidden
        );
    }
    let (gate_ptr, up_ptr) = split_fused_gate_up_tp(
        file,
        fused_info,
        n_experts,
        n_ff_exp,
        local_inter,
        rank,
        hidden,
        device,
        stream,
        tracker,
    )?;

    // 4. Down experts (`[n_experts, hidden, n_ff_exp]`) per-rank slice
    //    along inner dim (RowParallel{dim=2}). The generic
    //    `upload_sharded_tensor` handles the 3D quantised slice +
    //    block-alignment check.
    let down_info = file
        .tensors
        .get(&names.ffn_down_exps)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_down_exps))?;
    if down_info.dims != [n_experts as u64, hidden as u64, n_ff_exp as u64] {
        bail!(
            "{} dims {:?} != [{}, {}, {}]",
            names.ffn_down_exps,
            down_info.dims,
            n_experts,
            hidden,
            n_ff_exp
        );
    }
    let down = ut_to_dt(upload_sharded_tensor(
        file,
        down_info,
        WeightLayout::row_parallel(world, 2),
        rank,
        device,
        stream,
        tracker,
    )?);

    // 5. Three extra MoE norms (F32→F16 cast) — Replicated.
    let pre_ffw_norm_2 = ut_to_dt(upload_replicated_norm_f32_to_f16(
        file,
        file.tensors
            .get(&names.pre_ffw_norm_2)
            .ok_or_else(|| anyhow!("{} missing", names.pre_ffw_norm_2))?,
        hidden,
        device,
        stream,
        tracker,
    )?);
    let post_ffw_norm_1 = ut_to_dt(upload_replicated_norm_f32_to_f16(
        file,
        file.tensors
            .get(&names.post_ffw_norm_1)
            .ok_or_else(|| anyhow!("{} missing", names.post_ffw_norm_1))?,
        hidden,
        device,
        stream,
        tracker,
    )?);
    let post_ffw_norm_2 = ut_to_dt(upload_replicated_norm_f32_to_f16(
        file,
        file.tensors
            .get(&names.post_ffw_norm_2)
            .ok_or_else(|| anyhow!("{} missing", names.post_ffw_norm_2))?,
        hidden,
        device,
        stream,
        tracker,
    )?);

    // 6. Build per-rank `MoeExperts` block. `local_inter` sizes the
    //    block; the gate/up/down handles describe the per-rank sliced
    //    weight shapes. `MoeExperts::forward_decode_tp` produces a
    //    row-parallel partial that the layer composer AR-sums.
    let router_handle = router.as_weight_handle([n_experts, hidden])?;
    let gate_handle = WeightHandle {
        ptr: gate_ptr,
        dtype: blocks_ggml_to_qdtype(fused_info.dtype)?,
        dims: [n_experts * local_inter, hidden],
    };
    let up_handle = WeightHandle {
        ptr: up_ptr,
        dtype: blocks_ggml_to_qdtype(fused_info.dtype)?,
        dims: [n_experts * local_inter, hidden],
    };
    let down_handle = WeightHandle {
        ptr: down.ptr,
        dtype: blocks_ggml_to_qdtype(down_info.dtype)?,
        dims: [n_experts * hidden, local_inter],
    };
    let moe = MoeExperts::new(
        router_handle,
        gate_handle,
        up_handle,
        down_handle,
        hidden,
        local_inter,
        n_experts,
        top_k,
    )
    .context("MoeExperts::new (TP)")?
    .with_activation(Activation::Gelu);

    Ok(Gemma4TpMoeFfnWeights {
        moe,
        pre_router_weight_f16: pre_router_ptr,
        pre_ffw_norm_2: pre_ffw_norm_2.ptr,
        post_ffw_norm_1: post_ffw_norm_1.ptr,
        post_ffw_norm_2: post_ffw_norm_2.ptr,
    })
}

/// Host-side split of fused `ffn_gate_up_exps` with per-rank slice
/// along the intermediate axis. Output: two device buffers each
/// holding this rank's slice — `[n_experts, local_inter, hidden]`
/// for gate and up.
///
/// Source layout (GGUF): `[n_experts, 2·n_ff_exp, hidden]`, with rows
/// `[0..n_ff_exp]` per expert being gate and `[n_ff_exp..2·n_ff_exp]`
/// being up. The fused row stride is `2·n_ff_exp · row_bytes`.
///
/// Per-rank slice picks rows `[rank·local_inter..(rank+1)·local_inter]`
/// from within each expert's gate half (and the matching slice from
/// the up half). Both gate and up output buffers are
/// `n_experts · local_inter · row_bytes` bytes.
#[allow(clippy::too_many_arguments)]
fn split_fused_gate_up_tp(
    file: &GgufFile,
    fused_info: &flambeau_quant::TensorInfo,
    n_experts: usize,
    n_ff_exp: usize,
    local_inter: usize,
    rank: u32,
    hidden: usize,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<(DevicePtr, DevicePtr)> {
    let row_bytes = blocks_row_bytes(fused_info.dtype, hidden)?;
    let gate_half_bytes_per_expert = n_ff_exp * row_bytes;
    let fused_bytes_per_expert = 2 * gate_half_bytes_per_expert;
    let local_bytes_per_expert = local_inter * row_bytes;
    let total_local_bytes = n_experts * local_bytes_per_expert;
    let rank_offset_within_half = (rank as usize) * local_bytes_per_expert;

    let fused_raw = file
        .tensor_raw(&fused_info.name)
        .with_context(|| format!("tensor_raw `{}`", fused_info.name))?;
    let expected_bytes = n_experts * fused_bytes_per_expert;
    if fused_raw.len() < expected_bytes {
        bail!(
            "{} mmap slice {} < expected {}",
            fused_info.name,
            fused_raw.len(),
            expected_bytes
        );
    }

    let gate_ptr = device
        .alloc(total_local_bytes)
        .map_err(|e| anyhow!("alloc gate_exps tp split: {e}"))?;
    let up_ptr = device
        .alloc(total_local_bytes)
        .map_err(|e| anyhow!("alloc up_exps tp split: {e}"))?;
    tracker.track(gate_ptr, total_local_bytes);
    tracker.track(up_ptr, total_local_bytes);

    let src_base = fused_raw.as_ptr() as usize;
    for e in 0..n_experts {
        let expert_src_base = src_base + e * fused_bytes_per_expert;
        let gate_src = expert_src_base + rank_offset_within_half;
        let up_src = expert_src_base + gate_half_bytes_per_expert + rank_offset_within_half;
        let dst_gate = DevicePtr(gate_ptr.0 + e * local_bytes_per_expert);
        let dst_up = DevicePtr(up_ptr.0 + e * local_bytes_per_expert);
        // SAFETY: fused_raw covers expected_bytes ≥ all (gate_src,
        // up_src) + local_bytes_per_expert ranges; gate/up bufs
        // each own total_local_bytes; per-expert offsets are
        // distinct.
        unsafe {
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    dst_gate,
                    DevicePtr(gate_src),
                    local_bytes_per_expert,
                )
                .map_err(|e| anyhow!("memcpy gate expert (tp): {e}"))?;
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    dst_up,
                    DevicePtr(up_src),
                    local_bytes_per_expert,
                )
                .map_err(|e| anyhow!("memcpy up expert (tp): {e}"))?;
        }
    }
    stream.synchronize()?;
    Ok((gate_ptr, up_ptr))
}

fn upload_pre_router_weight(
    file: &GgufFile,
    names: &MoeFfnNames,
    hidden: usize,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<DevicePtr> {
    let scale_info = file
        .tensors
        .get(&names.ffn_gate_inp_scale)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_inp_scale))?;
    if scale_info.dtype != GgmlDType::F32 {
        bail!(
            "{} expected F32, got {:?}",
            names.ffn_gate_inp_scale,
            scale_info.dtype
        );
    }
    if scale_info.dims != [hidden as u64] {
        bail!(
            "{} dims {:?} != [{}]",
            names.ffn_gate_inp_scale,
            scale_info.dims,
            hidden
        );
    }
    let scale_raw = file
        .tensor_raw(&scale_info.name)
        .with_context(|| format!("tensor_raw `{}`", scale_info.name))?;
    let scale_f32: &[f32] = bytemuck::cast_slice(&scale_raw[..hidden * 4]);
    let inv_sqrt = 1.0f32 / (hidden as f32).sqrt();
    let pre_router_host: Vec<f16> = scale_f32
        .iter()
        .map(|&v| f16::from_f32(v * inv_sqrt))
        .collect();
    let bytes = hidden * 2;
    let ptr = device
        .alloc(bytes)
        .map_err(|e| anyhow!("alloc pre_router_weight (tp): {e}"))?;
    // SAFETY: pre_router_host outlives the bounded sync below; ptr
    // owns `bytes`.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(pre_router_host.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow!("memcpy pre_router_weight (tp): {e}"))?;
    }
    stream.synchronize()?;
    drop(pre_router_host);
    tracker.track(ptr, bytes);
    Ok(ptr)
}

fn ut_to_dt(u: flambeau_blocks::UploadedTensor) -> DeviceTensor {
    DeviceTensor {
        ptr: u.ptr,
        dtype: u.dtype,
        bytes: u.bytes,
    }
}

/// Per-rank forward for one gemma4 MoE FFN layer (TP). Writes the two
/// row-parallel partials needed by the MoE-aware composer:
/// - `partial_shared_mlp_out` (`Buffer<F16, RowParallel<0>>`): shared
///   MLP per-rank down-projection. AR'd in Phase 5a under `post_ffw_norm_1`.
/// - `scratch.partial_moe_f16` (same type): routed-experts per-rank
///   sum-over-experts. AR'd in Phase 5b under `post_ffw_norm_2`.
///
/// Pipeline (mirrors single-device `crate::moe::forward_ffn_moe` but
/// sliced for TP):
/// ```text
/// router_input = rmsnorm_f16(attn_residual, pre_router_weight)
/// {ids, w} = tp_moe.moe.route_decode(router_input)    // replicated topk
/// partial_shared_mlp_out = SharedMLP_TP(attn_residual, ffn_norm,
///                                        ffn_gate, ffn_up, ffn_down)
/// cur_moe_input = rmsnorm_f16(attn_residual, pre_ffw_norm_2)
/// scratch.partial_moe_f16 = tp_moe.moe.forward_decode_tp(cur_moe_input,
///                                                        moe_scratch)
/// ```
///
/// The per-branch post-norms (`post_ffw_norm_1`, `post_ffw_norm_2`)
/// and the combine + final norm + residual are all in the composer
/// (Phase 5a-c, 6) — they need full-hidden (post-AR) values.
#[allow(clippy::too_many_arguments)]
pub fn forward_ffn_moe_tp_per_rank<O: Ops>(
    ops: &O,
    layer: &Gemma4LayerWeights,
    tp_moe: &Gemma4TpMoeFfnWeights,
    scratch: &Gemma4TpMoeScratch,
    layer_x_q8_1: DevicePtr,
    layer_gate_f32: DevicePtr,
    layer_up_f32: DevicePtr,
    layer_activated_f16: DevicePtr,
    layer_activated_q8_1: DevicePtr,
    layer_down_f32: DevicePtr,
    attn_residual: DevicePtr,
    partial_shared_mlp_out: DevicePtr,
    hidden: usize,
    ff_len_local: usize,
    rms_norm_eps: f32,
) -> Result<()> {
    // 1. Router input — folded rmsnorm + scale + mul (pre_router_weight
    //    already carries (1/sqrt(hidden)) * ffn_gate_inp.scale).
    ops.rmsnorm_f16(
        attn_residual,
        tp_moe.pre_router_weight_f16,
        scratch.router_input_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE TP router_input rmsnorm")?;

    // 2. Replicated router top-k. Every rank computes the same
    //    expert_ids/weights (same router weights, same input).
    tp_moe
        .moe
        .route_decode(ops, scratch.router_input_f16, scratch.moe_scratch)
        .context("MoE TP route_decode")?;

    // 3. Shared MLP TP forward (parallel branch). Mirrors the dense
    //    `forward_ffn` body in `tp.rs::Gemma4TpDriver::forward_ffn`.
    ops.rmsnorm_quant_q8_1(
        attn_residual,
        layer.ffn_norm,
        layer_x_q8_1,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE TP shared-MLP ffn_norm + quant")?;
    let block = DenseMlpTp::new(
        WeightHandle {
            ptr: layer.ffn_gate.ptr,
            dtype: layer.ffn_gate.dtype,
            dims: [ff_len_local, hidden],
        },
        WeightHandle {
            ptr: layer.ffn_up.ptr,
            dtype: layer.ffn_up.dtype,
            dims: [ff_len_local, hidden],
        },
        WeightHandle {
            ptr: layer.ffn_down.ptr,
            dtype: layer.ffn_down.dtype,
            dims: [hidden, ff_len_local],
        },
        hidden,
        ff_len_local,
        Activation::Gelu,
    )
    .context("MoE TP shared-MLP DenseMlpTp::new")?;
    let dense_scratch = DenseMlpDecodeScratch {
        x_q8_1: layer_x_q8_1,
        gate_f32: layer_gate_f32,
        up_f32: layer_up_f32,
        activated_f16: layer_activated_f16,
        activated_q8_1: layer_activated_q8_1,
        down_f32: layer_down_f32,
        down_f16: DevicePtr(0),
    };
    block
        .forward_decode(
            ops,
            /* x_norm = */ DevicePtr(0),
            partial_shared_mlp_out,
            dense_scratch,
            /* pre_quantized = */ true,
        )
        .context("MoE TP shared-MLP DenseMlpTp::forward_decode")?;

    // 4. Pre-MoE-branch norm: cur_moe_input = rmsnorm(attn_residual,
    //    pre_ffw_norm_2). Per-row norm preserves the replicated layout.
    ops.rmsnorm_f16(
        attn_residual,
        tp_moe.pre_ffw_norm_2,
        scratch.cur_moe_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE TP pre_ffw_norm_2")?;

    // 5. Routed MoE forward (per-rank sliced experts). Writes
    //    `partial_moe_f16` = Σ_k w_k · down_local[k, :].
    //
    // Diagnostic bypass: `FLAMBEAU_TP_MOE_BYPASS_ROUTED=1` skips the
    // routed branch and zeros the partial. Used by Phase 10c-G to
    // isolate whether the F16 overflow at layer 5 originates in the
    // shared MLP TP path or the routed MoE TP path.
    if std::env::var_os("FLAMBEAU_TP_MOE_BYPASS_ROUTED").is_some() {
        ops.scale_f16(
            scratch.partial_moe_f16,
            scratch.partial_moe_f16,
            hidden,
            0.0,
        )
        .context("MoE TP bypass (zero partial_moe)")?;
    } else {
        tp_moe
            .moe
            .forward_decode_tp(ops, scratch.cur_moe_f16, scratch.partial_moe_f16, scratch.moe_scratch)
            .context("MoE TP forward_decode_tp")?;
    }

    Ok(())
}
