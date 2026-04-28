//! Gated-Delta-Net (GDN) forward: decode (one token) + prefill (L tokens).
//!
//! GDN is Qwen3.5-Next / Qwen3.6's hybrid-recurrent layer — it replaces the
//! attention block on even-numbered layer indices with an SSM-style state
//! update. F32 precision end-to-end between MMVQ cast-up and the gated F16
//! output, per-rank state lives in `session::GdnLayerState`.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    conv::causal_conv1d_f32,
    mlp::{scale_f32, silu_f32, swiglu_f32},
    norm::{
        l2_norm_f32, quantize_f16_q8_1, quantize_f16_q8_1_mmq, quantize_q8_1,
        quantize_q8_1_mmq, rmsnorm_f16, rmsnorm_f32, rmsnorm_quant_q8_1,
    },
    qmatmul::mmvq_q8_0_gate_up,
    recurrent::{
        gdn_alpha_beta_f32, gdn_split_qkv_f32, gdn_state_step_alphabeta_f32_s128,
        gdn_state_step_f32_s128,
    },
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::BlockQ8_1;

use super::common::{
    mat_shape, run_mmvq_from_tensor, run_qmatmul_from_tensor,
};
use crate::config::Qwen3MoEConfig;
use crate::session::{GdnLayerState, LayerCache};
use crate::weights::{DeviceTensor, GdnWeights};

// ---------------------------------------------------------------------------
// V1.7.3-c2 — Gated-Delta-Net decode-step forward.
// ---------------------------------------------------------------------------

/// Workspace buffers for one decode step of a GDN layer. Sized against
/// `Qwen3MoEConfig::gdn` (the hybrid arch's SSM dims).
///
/// The GDN path keeps F32 precision end-to-end from the post-MMVQ cast
/// through the state update, the ssm_norm, and the gated output. Only
/// the input activation coming in and the delta output going out are
/// F16 — everything in between is F32.
pub struct GdnScratch {
    // Fused norm-and-quant output for attn_qkv/attn_gate/ssm_alpha/ssm_beta.
    pub x_q8_1: DevicePtr,
    // F32 mmvq outputs (re-used across projections where sizes permit).
    pub qkv_mixed_f32: DevicePtr,   // [conv_channels]
    pub z_f32: DevicePtr,           // [d_inner]
    pub alpha_f32: DevicePtr,       // [num_v_heads]
    pub beta_f32: DevicePtr,        // [num_v_heads]
    // Conv1d staging: [conv_kernel, conv_channels] packed F32. First
    // (conv_kernel-1) rows come from the layer's conv_history; last row
    // is the current qkv_mixed.
    pub conv_input: DevicePtr,
    pub conv_out: DevicePtr,        // [conv_channels] — silu_in
    pub silu_out: DevicePtr,        // [conv_channels] — post-silu conv-out (Q|K|V packed)
    // L2-normalised Q and K per head (F32).
    pub q_norm_f32: DevicePtr,      // [num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,      // [num_k_heads, head_k_dim]
    // State-step output (F32): [num_v_heads, head_v_dim].
    pub state_out: DevicePtr,
    // ssm_norm output and gated output (F32).
    pub out_normed: DevicePtr,      // [num_v_heads, head_v_dim]
    pub gated_f32: DevicePtr,       // [d_inner]
    pub gated_q8_1: DevicePtr,      // Q8_1 blocks for ssm_out mmvq input
    pub ssm_out_f32: DevicePtr,     // [hidden]
    // Device gate + beta, consumed by `gdn_state_step_f32_s128`.
    pub gate_device: DevicePtr,     // [num_v_heads] F32
    pub beta_device: DevicePtr,     // [num_v_heads] F32
    // Bookkeeping for teardown.
    x_q8_1_bytes: usize,
    conv_channels_f32_bytes: usize,
    d_inner_f32_bytes: usize,
    num_v_heads_f32_bytes: usize,
    conv_input_bytes: usize,
    qk_f32_bytes: usize,
    v_f32_bytes: usize,
    gated_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl GdnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let gdn = cfg.gdn.as_ref().context("GdnScratch requires cfg.gdn")?;
        let hidden = cfg.hidden_size;
        let d_inner = gdn.d_inner;
        let num_v_heads = gdn.num_v_heads;
        let num_k_heads = gdn.num_k_heads;
        let head_k_dim = gdn.head_k_dim;
        let head_v_dim = gdn.head_v_dim();
        let conv_channels = gdn.conv_channels();
        let conv_kernel = gdn.conv_kernel;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            head_k_dim == 128 && head_v_dim == 128,
            "V1.7.2.F gdn_state_step kernel only instantiated at S_v=128"
        );

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let conv_channels_f32_bytes = conv_channels * 4;
        let d_inner_f32_bytes = d_inner * 4;
        let num_v_heads_f32_bytes = num_v_heads * 4;
        let conv_input_bytes = conv_kernel * conv_channels * 4;
        let qk_f32_bytes = num_k_heads * head_k_dim * 4;
        let v_f32_bytes = num_v_heads * head_v_dim * 4;
        let gated_q8_1_bytes = (d_inner / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let qkv_mixed_f32 = device.alloc(conv_channels_f32_bytes)?;
        let z_f32 = device.alloc(d_inner_f32_bytes)?;
        let alpha_f32 = device.alloc(num_v_heads_f32_bytes)?;
        let beta_f32 = device.alloc(num_v_heads_f32_bytes)?;
        let conv_input = device.alloc(conv_input_bytes)?;
        let conv_out = device.alloc(conv_channels_f32_bytes)?;
        let silu_out = device.alloc(conv_channels_f32_bytes)?;
        let q_norm_f32 = device.alloc(qk_f32_bytes)?;
        let k_norm_f32 = device.alloc(qk_f32_bytes)?;
        let state_out = device.alloc(v_f32_bytes)?;
        let out_normed = device.alloc(v_f32_bytes)?;
        let gated_f32 = device.alloc(d_inner_f32_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let ssm_out_f32 = device.alloc(hidden_f32_bytes)?;
        let gate_device = device.alloc(num_v_heads_f32_bytes)?;
        let beta_device = device.alloc(num_v_heads_f32_bytes)?;

        Ok(Self {
            x_q8_1,
            qkv_mixed_f32,
            z_f32,
            alpha_f32,
            beta_f32,
            conv_input,
            conv_out,
            silu_out,
            q_norm_f32,
            k_norm_f32,
            state_out,
            out_normed,
            gated_f32,
            gated_q8_1,
            ssm_out_f32,
            gate_device,
            beta_device,
            x_q8_1_bytes,
            conv_channels_f32_bytes,
            d_inner_f32_bytes,
            num_v_heads_f32_bytes,
            conv_input_bytes,
            qk_f32_bytes,
            v_f32_bytes,
            gated_q8_1_bytes,
            hidden_f32_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.qkv_mixed_f32, self.conv_channels_f32_bytes)?;
            device.dealloc(self.z_f32, self.d_inner_f32_bytes)?;
            device.dealloc(self.alpha_f32, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.beta_f32, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.conv_input, self.conv_input_bytes)?;
            device.dealloc(self.conv_out, self.conv_channels_f32_bytes)?;
            device.dealloc(self.silu_out, self.conv_channels_f32_bytes)?;
            device.dealloc(self.q_norm_f32, self.qk_f32_bytes)?;
            device.dealloc(self.k_norm_f32, self.qk_f32_bytes)?;
            device.dealloc(self.state_out, self.v_f32_bytes)?;
            device.dealloc(self.out_normed, self.v_f32_bytes)?;
            device.dealloc(self.gated_f32, self.d_inner_f32_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.ssm_out_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.gate_device, self.num_v_heads_f32_bytes)?;
            device.dealloc(self.beta_device, self.num_v_heads_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for GdnScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "GdnScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// After the conv kernel has read `conv_input`, advance the layer's
/// `conv_history` by one row: `history[0..k-2] = history[1..k-1]`,
/// `history[k-2] = current`. We exploit the fact that `conv_input` now
/// holds exactly `[old_history, current]`; copying `conv_input[1..k]`
/// back to the history slot is a single memcpy.
fn shift_conv_history(
    device: &HipDevice,
    stream: &HipStream,
    conv_input: DevicePtr,
    history: DevicePtr,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: `conv_input` and `history` both have at least
    // `hist_rows * row_bytes` valid device bytes starting from the
    // offsets we read/write.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            history,
            conv_input.offset_bytes(row_bytes),
            hist_rows * row_bytes,
        )?;
    }
    Ok(())
}

/// Decode step for one GDN layer. Consumes `x_in` (F16 `[hidden]`) and
/// writes the pre-residual output to `delta_out` (F16 `[hidden]`). Updates
/// the layer's recurrent state + conv1d history in-place.
///
/// V1.7.3-c2 scope:
/// - Split `ssm_alpha` + `ssm_beta` projections (Qwen3.6 / Qwen3.5 convention).
///   Fused `ssm_ba` (Qwen3-Next) is rejected with a bail for now.
/// - F32 recurrent arithmetic end-to-end from post-MMVQ cast to output
///   projection input (matches candle's `delta_net.rs` precision).
/// - Host alpha/beta/gate compute on `num_v_heads` floats per layer per
///   token. Flagged as follow-up for fusion into a single device kernel
///   (~2 memcpy roundtrips per GDN layer per decode step = 60 roundtrips
///   per decode at 30 GDN layers).
pub fn forward_gdn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &GdnWeights,
    layer_state: &mut GdnLayerState,
    scratch: &mut GdnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
) -> Result<()> {
    let gdn = cfg.gdn.as_ref().context("forward_gdn_decode requires cfg.gdn")?;
    let hidden = cfg.hidden_size;
    let d_inner = gdn.d_inner;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    let head_k_dim = gdn.head_k_dim;
    let head_v_dim = gdn.head_v_dim();
    let conv_channels = gdn.conv_channels();
    let conv_kernel = gdn.conv_kernel;
    let qk_size = num_k_heads * head_k_dim;
    let v_size = num_v_heads * head_v_dim;
    // Sanity — V1 only supports the split-alpha/split-beta arch family.
    if weights.ssm_ba.is_some() {
        bail!("V1 GDN forward expects split ssm_alpha/ssm_beta; fused ssm_ba is unsupported");
    }
    let ssm_alpha = weights
        .ssm_alpha
        .as_ref()
        .context("V1 GDN forward requires ssm_alpha")?;
    let ssm_beta = weights
        .ssm_beta
        .as_ref()
        .context("V1 GDN forward requires ssm_beta")?;

    // V1-BENCH-CN-80B-7 — intra-GDN section markers. No-op when timer
    // disabled. Every section's elapsed_ms aggregates across the GDN
    // layers in a forward pass, surfacing the dominant sub-kernel.
    flambeau_backend_hip::profile::mark("gdn_start", device, stream)?;

    // 1. Fused rmsnorm(x_in) + Q8_1 quantise.
    rmsnorm_quant_q8_1(
        ops,
        stream,
        x_in,
        attn_norm.ptr,
        scratch.x_q8_1,
        1,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("gdn attn_norm + quant")?;
    flambeau_backend_hip::profile::mark("gdn_norm_quant", device, stream)?;

    // 2..5. Four hidden-input projections share the Q8_1 input.
    // attn_qkv + attn_gate fuse when both weights match a supported
    // dtype: Q8_0 (Qwen3.6-x-Q8_0/Q8_K_XL family) or Q4_0 (Qwen3.6-x-
    // Q4_0 + Coder-Next-Q4_0 — the common GDN case per
    // mmvq_q4_0_gate_up's own docstring; pre-iter-4 the V1 dispatcher
    // skipped this branch and ran two separate mmvqs). The extended
    // gate_up kernel handles asymmetric n_rows by grid=max(n1,n2) with
    // per-output early-return.
    let variant_off = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let qkv_dtype = weights.attn_qkv.dtype;
    let gate_dtype = weights.attn_gate.dtype;
    let fuse_qkv_gate_q8_0 = !variant_off
        && qkv_dtype == flambeau_quant::GgmlDType::Q8_0
        && gate_dtype == flambeau_quant::GgmlDType::Q8_0;
    let fuse_qkv_gate_q4_0 = !variant_off
        && qkv_dtype == flambeau_quant::GgmlDType::Q4_0
        && gate_dtype == flambeau_quant::GgmlDType::Q4_0;
    if fuse_qkv_gate_q8_0 {
        mmvq_q8_0_gate_up(
            ops,
            stream,
            weights.attn_qkv.ptr,
            weights.attn_gate.ptr,
            scratch.x_q8_1,
            scratch.qkv_mixed_f32,
            scratch.z_f32,
            conv_channels,
            d_inner,
            hidden,
        )
        .context("attn_qkv + attn_gate fused mmvq_q8_0")?;
    } else if fuse_qkv_gate_q4_0 {
        flambeau_ops::hip::qmatmul::mmvq_q4_0_gate_up(
            ops,
            stream,
            weights.attn_qkv.ptr,
            weights.attn_gate.ptr,
            scratch.x_q8_1,
            scratch.qkv_mixed_f32,
            scratch.z_f32,
            conv_channels,
            d_inner,
            hidden,
        )
        .context("attn_qkv + attn_gate fused mmvq_q4_0 (V1-BENCH-CN-80B-7)")?;
    } else {
        run_mmvq_from_tensor(ops, stream, &weights.attn_qkv, scratch.x_q8_1, scratch.qkv_mixed_f32, conv_channels, hidden, "attn_qkv")?;
        run_mmvq_from_tensor(ops, stream, &weights.attn_gate, scratch.x_q8_1, scratch.z_f32, d_inner, hidden, "attn_gate")?;
    }
    flambeau_backend_hip::profile::mark("gdn_proj_qkv_gate", device, stream)?;
    // ssm_alpha + ssm_beta fuse when FLAMBEAU_VARIANT=dp4a_vdr2 — both Q8_0,
    // same [num_v_heads, hidden] shape, both read x_q8_1 once. Same pattern
    // as shared-expert gate+up fusion.
    let fuse_alpha_beta = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && ssm_alpha.dtype == flambeau_quant::GgmlDType::Q8_0
        && ssm_beta.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_alpha_beta {
        let (a_rows, a_k) = mat_shape(ssm_alpha)?;
        let (b_rows, b_k) = mat_shape(ssm_beta)?;
        if a_rows != num_v_heads || a_k != hidden || b_rows != num_v_heads || b_k != hidden {
            bail!(
                "fused ssm alpha/beta shape mismatch: alpha=[{a_rows},{a_k}] beta=[{b_rows},{b_k}] expected=[{num_v_heads},{hidden}]"
            );
        }
        mmvq_q8_0_gate_up(
            ops,
            stream,
            ssm_alpha.ptr,
            ssm_beta.ptr,
            scratch.x_q8_1,
            scratch.alpha_f32,
            scratch.beta_f32,
            num_v_heads,
            num_v_heads,
            hidden,
        )
        .context("ssm alpha+beta fused mmvq_q8_0")?;
    } else {
        run_mmvq_from_tensor(ops, stream, ssm_alpha, scratch.x_q8_1, scratch.alpha_f32, num_v_heads, hidden, "ssm_alpha")?;
        run_mmvq_from_tensor(ops, stream, ssm_beta, scratch.x_q8_1, scratch.beta_f32, num_v_heads, hidden, "ssm_beta")?;
    }
    flambeau_backend_hip::profile::mark("gdn_proj_alpha_beta", device, stream)?;

    // 6. Conv1d step — assemble [history_{k-1}, qkv_mixed] into conv_input,
    // run causal conv, then shift history forward. V2.23.d.1 uses a single
    // fused kernel in place of the prior two DtoD memcpys.
    let _ = device; // history+current copy now done via kernel, not device
    flambeau_ops::hip::recurrent::gdn_assemble_conv_input_f32(
        ops,
        stream,
        layer_state.conv_history,
        scratch.qkv_mixed_f32,
        scratch.conv_input,
        conv_channels,
        conv_kernel,
    )?;
    causal_conv1d_f32(
        ops,
        stream,
        scratch.conv_input,
        weights.ssm_conv1d.ptr,
        scratch.conv_out,
        1,
        conv_channels,
        conv_kernel,
    )
    .context("causal_conv1d_f32")?;
    shift_conv_history(
        device,
        stream,
        scratch.conv_input,
        layer_state.conv_history,
        conv_channels,
        conv_kernel,
    )?;
    flambeau_backend_hip::profile::mark("gdn_conv1d", device, stream)?;

    // 7. silu(conv_out).
    silu_f32(ops, stream, scratch.conv_out, scratch.silu_out, conv_channels)
        .context("silu_f32(conv_out)")?;
    flambeau_backend_hip::profile::mark("gdn_silu", device, stream)?;

    // 8. Slice silu_out into Q|K|V via pointer offsets. Q and K are
    // adjacent `qk_size` blocks; V follows. No kernel.
    let q_src = scratch.silu_out;
    let k_src = scratch.silu_out.offset_bytes(qk_size * 4);
    let v_src = scratch.silu_out.offset_bytes(2 * qk_size * 4);

    // 9. L2-normalise Q and K per head (row = head, k = head_k_dim).
    l2_norm_f32(
        ops,
        stream,
        q_src,
        scratch.q_norm_f32,
        num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("l2_norm Q")?;
    l2_norm_f32(
        ops,
        stream,
        k_src,
        scratch.k_norm_f32,
        num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("l2_norm K")?;

    // 10. Scale Q by 1/sqrt(head_k_dim) (in-place).
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    scale_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        qk_size,
        q_scale,
    )
    .context("scale_f32 Q")?;
    flambeau_backend_hip::profile::mark("gdn_l2norm_qk", device, stream)?;

    // 11–12. C10 — fused state-step that absorbs the α/β/gate compute
    // (saves one kernel launch per GDN layer per token). Default-on;
    // `FLAMBEAU_VARIANT=baseline` opts back to the unfused chain for
    // regression A/B. n_rep = num_v_heads / num_k_heads.
    //
    // CN-80B-13/14 — q/k repeat layout differs by arch:
    //   qwen35moe (Qwen3.6-35B-A3B): rep-OUTER (cyclic ggml_repeat_4d)
    //     → kernel uses `h_kv = h_idx % H_kv`. rep_inner_layout = false.
    //   qwen3next (Coder-Next-80B): rep-INNER (reshape-interleave per
    //     `qwen3next.cpp:418-431`) → kernel uses `h_kv = h_idx / n_rep`.
    //     rep_inner_layout = true.
    // Wrong choice produces a degenerate logit attractor; see CN-80B-14.
    let n_rep = num_v_heads / num_k_heads;
    let rep_inner_layout = cfg.arch == "qwen3next";
    let fuse_state_step = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline");
    if fuse_state_step {
        gdn_state_step_alphabeta_f32_s128(
            ops,
            stream,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            v_src,
            scratch.alpha_f32,
            scratch.beta_f32,
            weights.ssm_dt_bias.ptr,
            weights.ssm_a.ptr,
            layer_state.state,
            layer_state.state,
            scratch.state_out,
            1,
            num_v_heads,
            1,
            n_rep,
            rep_inner_layout,
        )
        .context("gdn_state_step_alphabeta_f32_s128 (C10 fused)")?;
    } else {
        gdn_alpha_beta_f32(
            ops,
            stream,
            scratch.alpha_f32,
            scratch.beta_f32,
            weights.ssm_dt_bias.ptr,
            weights.ssm_a.ptr,
            scratch.gate_device,
            scratch.beta_device,
            num_v_heads,
            /* n_tokens = */ 1,
        )
        .context("gdn_alpha_beta_f32 fused")?;
        gdn_state_step_f32_s128(
            ops,
            stream,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            v_src,
            scratch.gate_device,
            scratch.beta_device,
            layer_state.state,
            layer_state.state,
            scratch.state_out,
            1,
            num_v_heads,
            1,
            n_rep,
            rep_inner_layout,
        )
        .context("gdn_state_step_f32_s128 (baseline)")?;
    }
    flambeau_backend_hip::profile::mark("gdn_state_step", device, stream)?;

    // 13. ssm_norm per-head on the state-step output.
    let ssm_norm_k = weights
        .ssm_norm
        .dims
        .first()
        .copied()
        .context("ssm_norm missing dim")? as usize;
    if ssm_norm_k != head_v_dim {
        bail!(
            "ssm_norm dim {ssm_norm_k} != head_v_dim {head_v_dim}"
        );
    }
    rmsnorm_f32(
        ops,
        stream,
        scratch.state_out,
        weights.ssm_norm.ptr,
        scratch.out_normed,
        num_v_heads,
        head_v_dim,
        cfg.rms_norm_eps,
    )
    .context("ssm_norm (rmsnorm_f32)")?;
    flambeau_backend_hip::profile::mark("gdn_ssm_norm", device, stream)?;

    // 14+15. CN-80B-19c — fused swiglu(z, out_normed) → Q8_1 directly.
    // Skips the F32 `gated_f32` intermediate buffer + 1 launch. Default-on;
    // FLAMBEAU_VARIANT=baseline opts back to the unfused pair.
    if v_size != d_inner {
        bail!(
            "GDN layout bug: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})"
        );
    }
    let fuse_tail = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && d_inner % 32 == 0;
    if fuse_tail {
        flambeau_ops::hip::mlp::swiglu_f32_to_q8_1(
            ops,
            stream,
            scratch.z_f32,
            scratch.out_normed,
            scratch.gated_q8_1,
            d_inner,
        )
        .context("swiglu_f32_to_q8_1(z, out_normed) (CN-80B-19c)")?;
    } else {
        swiglu_f32(ops, stream, scratch.z_f32, scratch.out_normed, scratch.gated_f32, d_inner)
            .context("swiglu_f32(z, out_normed)")?;
        quantize_q8_1(ops, stream, scratch.gated_f32, scratch.gated_q8_1, d_inner)
            .context("quantize gated → Q8_1")?;
    }
    flambeau_backend_hip::profile::mark("gdn_swiglu_quant", device, stream)?;

    // 16. ssm_out projection.
    run_mmvq_from_tensor(
        ops,
        stream,
        &weights.ssm_out,
        scratch.gated_q8_1,
        scratch.ssm_out_f32,
        hidden,
        d_inner,
        "ssm_out",
    )?;
    flambeau_backend_hip::profile::mark("gdn_ssm_out", device, stream)?;

    // 17. Cast back to F16 for the residual path.
    cast_f32_to_f16(ops, stream, scratch.ssm_out_f32, delta_out, hidden)
        .context("cast ssm_out → f16")?;
    flambeau_backend_hip::profile::mark("gdn_cast_f16", device, stream)?;

    Ok(())
}

// `run_mmvq_from_tensor` moved to `forward::common`.

/// Route a `LayerCache` entry through the GDN forward, pulling the
/// correct `GdnLayerState` out of the enum and the matching `GdnWeights`
/// out of the layer's `AttnWeights` variant.
pub fn forward_gdn_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut GdnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
) -> Result<()> {
    let LayerCache::Gdn(state) = layer_cache else {
        bail!(
            "layer {} is not a GDN layer (cache variant mismatch)",
            layer_weights.layer_idx
        );
    };
    let crate::weights::AttnWeights::Gdn(g) = &layer_weights.attn else {
        bail!("layer {} weights are not Gdn variant", layer_weights.layer_idx);
    };
    forward_gdn_decode(
        ops,
        stream,
        device,
        cfg,
        &layer_weights.attn_norm,
        g,
        state,
        scratch,
        x_in,
        delta_out,
    )
}


// ---------------------------------------------------------------------------
// V1.7.3-f2 — Gated-Delta-Net prefill (L > 1).
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of a GDN layer. Sized once against
/// `(cfg, max_tokens)`. Most buffers scale linearly with L; the state
/// tensor is per-layer (doesn't grow with L) and lives in the session.
pub struct GdnPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,      // F16 [L, hidden] — V2.2.d.P8 unfused rmsnorm sink
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,      // DS4 layout sibling of x_q8_1 for MmqLdsX64
    pub qkv_mixed_f32: DevicePtr,   // [L, conv_channels]
    pub z_f32: DevicePtr,           // [L, d_inner]
    pub alpha_f32: DevicePtr,       // [L, num_v_heads]
    pub beta_f32: DevicePtr,        // [L, num_v_heads]
    pub conv_input: DevicePtr,      // [(conv_kernel-1) + L, conv_channels]
    pub conv_out: DevicePtr,        // [L, conv_channels]
    pub silu_out: DevicePtr,        // [L, conv_channels]
    pub q_norm_f32: DevicePtr,      // [L, num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,      // [L, num_k_heads, head_k_dim]
    pub v_f32: DevicePtr,           // [L, num_v_heads, head_v_dim]
    pub state_out: DevicePtr,       // [L, num_v_heads, head_v_dim]
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,  // DS4 layout sibling of gated_q8_1
    pub ssm_out_f32: DevicePtr,     // [L, hidden]
    pub gate_device: DevicePtr,     // [L, num_v_heads]
    pub beta_device: DevicePtr,     // [L, num_v_heads]
    // Bookkeeping.
    x_norm_f16_bytes: usize,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    qkv_mixed_bytes: usize,
    z_bytes: usize,
    alpha_beta_bytes: usize,
    conv_input_bytes: usize,
    conv_out_bytes: usize,
    silu_out_bytes: usize,
    qk_bytes: usize,
    v_bytes: usize,
    state_out_bytes: usize,
    out_normed_bytes: usize,
    gated_f32_bytes: usize,
    gated_q8_1_bytes: usize,
    gated_q8_1_mmq_bytes: usize,
    ssm_out_bytes: usize,
    gate_device_bytes: usize,
    disposed: bool,
}

impl GdnPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let gdn = cfg.gdn.as_ref().context("GdnPrefillScratch requires cfg.gdn")?;
        let hidden = cfg.hidden_size;
        let d_inner = gdn.d_inner;
        let num_v_heads = gdn.num_v_heads;
        let num_k_heads = gdn.num_k_heads;
        let head_k_dim = gdn.head_k_dim;
        let head_v_dim = gdn.head_v_dim();
        let conv_channels = gdn.conv_channels();
        let conv_kernel = gdn.conv_kernel;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            d_inner % 128 == 0,
            "d_inner must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            head_k_dim == 128 && head_v_dim == 128,
            "V1.7.2.F gdn_state_step kernel only instantiated at S_v=128"
        );

        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_norm_f16_bytes = max_tokens * hidden * 2;
        let x_q8_1_bytes =
            max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let qkv_mixed_bytes = max_tokens * conv_channels * 4;
        let z_bytes = max_tokens * d_inner * 4;
        let alpha_beta_bytes = max_tokens * num_v_heads * 4;
        let conv_input_bytes = ((conv_kernel - 1) + max_tokens) * conv_channels * 4;
        let conv_out_bytes = max_tokens * conv_channels * 4;
        let silu_out_bytes = max_tokens * conv_channels * 4;
        let qk_bytes = max_tokens * num_k_heads * head_k_dim * 4;
        let v_bytes = max_tokens * num_v_heads * head_v_dim * 4;
        let state_out_bytes = max_tokens * num_v_heads * head_v_dim * 4;
        let out_normed_bytes = state_out_bytes;
        let gated_f32_bytes = max_tokens * d_inner * 4;
        let gated_q8_1_bytes =
            max_tokens * (d_inner / 32) * std::mem::size_of::<BlockQ8_1>();
        let gated_q8_1_mmq_bytes = max_tokens * (d_inner / 128) * mmq_block;
        let ssm_out_bytes = max_tokens * hidden * 4;
        let gate_device_bytes = max_tokens * num_v_heads * 4;

        let x_norm_f16 = device.alloc(x_norm_f16_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let qkv_mixed_f32 = device.alloc(qkv_mixed_bytes)?;
        let z_f32 = device.alloc(z_bytes)?;
        let alpha_f32 = device.alloc(alpha_beta_bytes)?;
        let beta_f32 = device.alloc(alpha_beta_bytes)?;
        let conv_input = device.alloc(conv_input_bytes)?;
        let conv_out = device.alloc(conv_out_bytes)?;
        let silu_out = device.alloc(silu_out_bytes)?;
        let q_norm_f32 = device.alloc(qk_bytes)?;
        let k_norm_f32 = device.alloc(qk_bytes)?;
        let v_f32 = device.alloc(v_bytes)?;
        let state_out = device.alloc(state_out_bytes)?;
        let out_normed = device.alloc(out_normed_bytes)?;
        let gated_f32 = device.alloc(gated_f32_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let gated_q8_1_mmq = device.alloc(gated_q8_1_mmq_bytes)?;
        let ssm_out_f32 = device.alloc(ssm_out_bytes)?;
        let gate_device = device.alloc(gate_device_bytes)?;
        let beta_device = device.alloc(gate_device_bytes)?;

        Ok(Self {
            max_tokens,
            x_norm_f16,
            x_q8_1,
            x_q8_1_mmq,
            qkv_mixed_f32,
            z_f32,
            alpha_f32,
            beta_f32,
            conv_input,
            conv_out,
            silu_out,
            q_norm_f32,
            k_norm_f32,
            v_f32,
            state_out,
            out_normed,
            gated_f32,
            gated_q8_1,
            gated_q8_1_mmq,
            ssm_out_f32,
            gate_device,
            beta_device,
            x_norm_f16_bytes,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            qkv_mixed_bytes,
            z_bytes,
            alpha_beta_bytes,
            conv_input_bytes,
            conv_out_bytes,
            silu_out_bytes,
            qk_bytes,
            v_bytes,
            state_out_bytes,
            out_normed_bytes,
            gated_f32_bytes,
            gated_q8_1_bytes,
            gated_q8_1_mmq_bytes,
            ssm_out_bytes,
            gate_device_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // SAFETY: every pointer came from `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(self.x_norm_f16, self.x_norm_f16_bytes)?;
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.qkv_mixed_f32, self.qkv_mixed_bytes)?;
            device.dealloc(self.z_f32, self.z_bytes)?;
            device.dealloc(self.alpha_f32, self.alpha_beta_bytes)?;
            device.dealloc(self.beta_f32, self.alpha_beta_bytes)?;
            device.dealloc(self.conv_input, self.conv_input_bytes)?;
            device.dealloc(self.conv_out, self.conv_out_bytes)?;
            device.dealloc(self.silu_out, self.silu_out_bytes)?;
            device.dealloc(self.q_norm_f32, self.qk_bytes)?;
            device.dealloc(self.k_norm_f32, self.qk_bytes)?;
            device.dealloc(self.v_f32, self.v_bytes)?;
            device.dealloc(self.state_out, self.state_out_bytes)?;
            device.dealloc(self.out_normed, self.out_normed_bytes)?;
            device.dealloc(self.gated_f32, self.gated_f32_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.gated_q8_1_mmq, self.gated_q8_1_mmq_bytes)?;
            device.dealloc(self.ssm_out_f32, self.ssm_out_bytes)?;
            device.dealloc(self.gate_device, self.gate_device_bytes)?;
            device.dealloc(self.beta_device, self.gate_device_bytes)?;
        }
        Ok(())
    }
}

impl Drop for GdnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "GdnPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Gather Q / K / V rows out of a packed `silu_out[L, 2*qk_size + v_size]`
/// tensor into separate contiguous `[L, qk_size]` / `[L, v_size]` buffers.
/// Uses L stream-ordered `memcpy_async(DeviceToDevice)` per output — 3 × L
/// copies per layer per prefill. For Qwen3.6 at L = 128 that's ~400 copies,
/// each ~8 KB — stream-pipelined so no sync penalty. Future fusion: a
/// single `gdn_split_qkv_f32` kernel.
fn gather_qkv_strided(
    device: &HipDevice,
    stream: &HipStream,
    silu_out: DevicePtr,
    q_out: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    n_tokens: usize,
    qk_size: usize,
    v_size: usize,
) -> Result<()> {
    let conv_channels = 2 * qk_size + v_size;
    let row_bytes_in = conv_channels * 4;
    let q_row_bytes = qk_size * 4;
    let v_row_bytes = v_size * 4;
    for t in 0..n_tokens {
        let row_ptr = silu_out.offset_bytes(t * row_bytes_in);
        // SAFETY: silu_out has n_tokens * conv_channels F32s; the three
        // sub-ranges fit inside one row.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                q_out.offset_bytes(t * q_row_bytes),
                row_ptr,
                q_row_bytes,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                k_out.offset_bytes(t * q_row_bytes),
                row_ptr.offset_bytes(qk_size * 4),
                q_row_bytes,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                v_out.offset_bytes(t * v_row_bytes),
                row_ptr.offset_bytes(2 * qk_size * 4),
                v_row_bytes,
            )?;
        }
    }
    Ok(())
}

/// Assemble `conv_input[(K-1) + L, conv_channels]` from the layer's
/// `conv_history[(K-1), conv_channels]` + the fresh `qkv_mixed[L, conv_channels]`.
pub(super) fn assemble_conv_input_prefill(
    device: &HipDevice,
    stream: &HipStream,
    history: DevicePtr,
    qkv_mixed: DevicePtr,
    conv_input: DevicePtr,
    n_tokens: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: all three buffers have at least the bytes we touch.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            conv_input,
            history,
            hist_rows * row_bytes,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            conv_input.offset_bytes(hist_rows * row_bytes),
            qkv_mixed,
            n_tokens * row_bytes,
        )?;
    }
    Ok(())
}

/// After the conv has read `conv_input[(K-1) + L]`, update the layer's
/// history slot to the last `K-1` rows — `conv_input[L..L+K-1]`. One
/// memcpy (may alias if L == 0, but prefill has L ≥ 1).
pub(super) fn shift_conv_history_prefill(
    device: &HipDevice,
    stream: &HipStream,
    conv_input: DevicePtr,
    history: DevicePtr,
    n_tokens: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: conv_input has (K-1 + L) valid rows; history has K-1.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            history,
            conv_input.offset_bytes(n_tokens * row_bytes),
            hist_rows * row_bytes,
        )?;
    }
    Ok(())
}

/// Prefill step for one GDN layer. Consumes `x_in` (F16 `[L, hidden]`),
/// updates `layer_state.state` + `layer_state.conv_history` across all L
/// tokens, and writes the pre-residual `delta_out` (F16 `[L, hidden]`).
///
/// The GDN state-step kernel is already L-aware — it keeps the per-head
/// `[S_v, S_v]` state register-resident across the entire L recurrence
/// loop in a single launch. The α / β / gate compute is currently a L-wide
/// loop over the single-token kernel; future fusion into a proper batched
/// variant saves O(L) launches but is not required for correctness.
pub fn forward_gdn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &GdnWeights,
    layer_state: &mut GdnLayerState,
    scratch: &mut GdnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    state_event: Option<&flambeau_backend_hip::HipEvent>,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_gdn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_gdn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
            scratch.max_tokens
        );
    }

    let gdn = cfg.gdn.as_ref().context("forward_gdn_prefill requires cfg.gdn")?;
    let hidden = cfg.hidden_size;
    let d_inner = gdn.d_inner;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    let head_k_dim = gdn.head_k_dim;
    let head_v_dim = gdn.head_v_dim();
    let conv_channels = gdn.conv_channels();
    let conv_kernel = gdn.conv_kernel;
    let qk_size = num_k_heads * head_k_dim;
    let v_size = num_v_heads * head_v_dim;

    if weights.ssm_ba.is_some() {
        bail!("V1 GDN forward expects split ssm_alpha/ssm_beta; fused ssm_ba is unsupported");
    }
    let ssm_alpha = weights
        .ssm_alpha
        .as_ref()
        .context("V1 GDN forward requires ssm_alpha")?;
    let ssm_beta = weights
        .ssm_beta
        .as_ref()
        .context("V1 GDN forward requires ssm_beta")?;

    // 1. rmsnorm(x_in) → F16 scratch, then quantise to BOTH Q8_1 layouts.
    //    V2.2.d.P8 de-fuses the old rmsnorm_quant_q8_1 so we can emit the
    //    DS4 (BlockQ8_1Mmq) layout consumed by the new Q4_1 MMQ kernel at
    //    m ≥ 128 alongside the standard per-row layout for MMVQ.
    rmsnorm_f16(
        ops,
        stream,
        x_in,
        attn_norm.ptr,
        scratch.x_norm_f16,
        n_tokens,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("gdn prefill attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("gdn prefill x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("gdn prefill x_norm → Q8_1 (MMQ DS4)")?;

    // 2..5. Hidden-input projections at M = L. attn_qkv / attn_gate are Q4_1
    // on Qwen3.5-9B → route to MmqLdsX64 at m ≥ 128. ssm_alpha / ssm_beta are
    // Q5_K / other → never route to MmqLdsX64 (dispatch has no row). The mmq
    // buffer is passed to all four; dispatch picks per-weight-dtype.
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.attn_qkv,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.qkv_mixed_f32,
        n_tokens,
        hidden,
        conv_channels,
        "attn_qkv",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.attn_gate,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.z_f32,
        n_tokens,
        hidden,
        d_inner,
        "attn_gate",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        ssm_alpha,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.alpha_f32,
        n_tokens,
        hidden,
        num_v_heads,
        "ssm_alpha",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        ssm_beta,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.beta_f32,
        n_tokens,
        hidden,
        num_v_heads,
        "ssm_beta",
    )?;

    // 6. Conv1d across L tokens: assemble [history, qkv_mixed] → conv_input,
    //    run conv, then shift history to the last (K-1) rows.
    assemble_conv_input_prefill(
        device,
        stream,
        layer_state.conv_history,
        scratch.qkv_mixed_f32,
        scratch.conv_input,
        n_tokens,
        conv_channels,
        conv_kernel,
    )?;
    causal_conv1d_f32(
        ops,
        stream,
        scratch.conv_input,
        weights.ssm_conv1d.ptr,
        scratch.conv_out,
        n_tokens,
        conv_channels,
        conv_kernel,
    )
    .context("prefill causal_conv1d_f32")?;
    shift_conv_history_prefill(
        device,
        stream,
        scratch.conv_input,
        layer_state.conv_history,
        n_tokens,
        conv_channels,
        conv_kernel,
    )?;

    // 7. silu(conv_out) → silu_out (F32 [L, conv_channels]).
    silu_f32(
        ops,
        stream,
        scratch.conv_out,
        scratch.silu_out,
        n_tokens * conv_channels,
    )
    .context("prefill silu_f32(conv_out)")?;

    // 8. Split silu_out into Q / K / V contiguous buffers. V2.4.d fused
    // `gdn_split_qkv_f32` kernel replaces the 3×L memcpy loop (~1500
    // driver calls per layer at L=512). FLAMBEAU_QKV_FUSED=0 reverts
    // to the memcpy loop for regression comparison.
    if std::env::var("FLAMBEAU_QKV_FUSED").as_deref() == Ok("0") {
        let _ = device; // unused in fused path
        gather_qkv_strided(
            device,
            stream,
            scratch.silu_out,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            n_tokens,
            qk_size,
            v_size,
        )?;
    } else {
        gdn_split_qkv_f32(
            ops,
            stream,
            scratch.silu_out,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            n_tokens,
            qk_size,
            v_size,
        )
        .context("prefill gdn_split_qkv_f32")?;
    }

    // 9. L2-normalise Q and K per head (row = head, k = head_k_dim).
    l2_norm_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill l2_norm Q")?;
    l2_norm_f32(
        ops,
        stream,
        scratch.k_norm_f32,
        scratch.k_norm_f32,
        n_tokens * num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill l2_norm K")?;

    // 10. Scale Q by 1/sqrt(head_k_dim) — in place across all L.
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    scale_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * qk_size,
        q_scale,
    )
    .context("prefill scale_f32 Q")?;

    // 11–12. C10 fused state-step (default) absorbs α/β/gate; baseline
    // chain available via FLAMBEAU_VARIANT=baseline. State-step
    // event-ordering preserved across both branches (V2.30.a).
    // CN-80B-13/14 — q/k repeat layout differs by arch; see decode-path
    // comment in `forward_gdn_decode` for the explanation.
    let n_rep = num_v_heads / num_k_heads;
    let rep_inner_layout = cfg.arch == "qwen3next";
    let fuse_state_step = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline");
    if let Some(ev) = state_event {
        ev.stream_wait(stream)
            .context("gdn state_step stream_wait")?;
    }
    if fuse_state_step {
        gdn_state_step_alphabeta_f32_s128(
            ops,
            stream,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            scratch.alpha_f32,
            scratch.beta_f32,
            weights.ssm_dt_bias.ptr,
            weights.ssm_a.ptr,
            layer_state.state,
            layer_state.state,
            scratch.state_out,
            1,
            num_v_heads,
            n_tokens,
            n_rep,
            rep_inner_layout,
        )
        .context("prefill gdn_state_step_alphabeta_f32_s128 (C10 fused)")?;
    } else {
        gdn_alpha_beta_f32(
            ops,
            stream,
            scratch.alpha_f32,
            scratch.beta_f32,
            weights.ssm_dt_bias.ptr,
            weights.ssm_a.ptr,
            scratch.gate_device,
            scratch.beta_device,
            num_v_heads,
            n_tokens,
        )
        .context("prefill gdn_alpha_beta_f32 (batched)")?;
        gdn_state_step_f32_s128(
            ops,
            stream,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            scratch.gate_device,
            scratch.beta_device,
            layer_state.state,
            layer_state.state,
            scratch.state_out,
            1,
            num_v_heads,
            n_tokens,
            n_rep,
            rep_inner_layout,
        )
        .context("prefill gdn_state_step_f32_s128 (baseline)")?;
    }
    if let Some(ev) = state_event {
        ev.record(stream).context("gdn state_step record")?;
    }

    // 13. ssm_norm per-head over L × num_v_heads rows.
    let ssm_norm_k = weights
        .ssm_norm
        .dims
        .first()
        .copied()
        .context("ssm_norm missing dim")? as usize;
    if ssm_norm_k != head_v_dim {
        bail!("ssm_norm dim {ssm_norm_k} != head_v_dim {head_v_dim}");
    }
    rmsnorm_f32(
        ops,
        stream,
        scratch.state_out,
        weights.ssm_norm.ptr,
        scratch.out_normed,
        n_tokens * num_v_heads,
        head_v_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill ssm_norm (rmsnorm_f32)")?;

    // 14. Gated: `gated = silu(z) * out_normed` across [L, d_inner].
    if v_size != d_inner {
        bail!(
            "GDN layout: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})"
        );
    }
    swiglu_f32(
        ops,
        stream,
        scratch.z_f32,
        scratch.out_normed,
        scratch.gated_f32,
        n_tokens * d_inner,
    )
    .context("prefill swiglu_f32(z, out_normed)")?;

    // 15. Quantise gated → both Q8_1 layouts for the ssm_out matmul.
    quantize_q8_1(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1,
        n_tokens * d_inner,
    )
    .context("prefill quantise gated → Q8_1 (std)")?;
    quantize_q8_1_mmq(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1_mmq,
        d_inner,
        n_tokens,
    )
    .context("prefill quantise gated → Q8_1 (MMQ DS4)")?;

    // 16. ssm_out projection at M = L. ssm_out is typically Q5_K / Q8_0 on
    // V1 models and does not route to MmqLdsX64, but we pass the mmq buffer
    // so dispatch has it available if a Q4_1 ssm_out lands in some future
    // GGUF dtype mix.
    run_qmatmul_from_tensor(
        ops,
        stream,
        &weights.ssm_out,
        scratch.gated_q8_1,
        scratch.gated_q8_1_mmq,
        scratch.ssm_out_f32,
        n_tokens,
        d_inner,
        hidden,
        "ssm_out",
    )?;

    // 17. Cast back to F16 for the outer residual path.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.ssm_out_f32,
        delta_out,
        n_tokens * hidden,
    )
    .context("prefill cast ssm_out → f16")?;

    Ok(())
}

// `run_qmatmul_from_tensor` moved to `forward::common`.

