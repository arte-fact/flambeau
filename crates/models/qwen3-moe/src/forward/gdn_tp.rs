//! TP-4a-i2 — tensor-parallel `forward_gdn_decode`.
//!
//! Sister of [`super::gdn::forward_gdn_decode`] for the TP-sharded
//! forward path. Same 17-op structure, with two differences:
//!
//! 1. Per-rank head counts threaded through every kernel call:
//!    - `local_num_v_heads = num_v_heads / world`
//!    - `local_num_k_heads = num_k_heads / world`
//!    - `local_d_inner = local_num_v_heads · head_v_dim`
//!    - `local_qk_size = local_num_k_heads · head_k_dim`
//!    - `local_conv_channels = local_d_inner + 2 · local_qk_size`
//!    - `n_rep = num_v_heads / num_k_heads` (unchanged — divides identically
//!      on either side of the per-rank split).
//!
//!    Sliced weights (per TP-1a + TP-4a-i1):
//!    - `attn_qkv.weight`     `FusedQkvParallel` → `[local_conv_channels, hidden]`
//!    - `attn_gate.weight`    `ColParallel{dim=0}`  → `[local_d_inner, hidden]`
//!    - `ssm_alpha.weight`    `ColParallel{dim=0}`  → `[local_num_v_heads, hidden]`
//!    - `ssm_beta.weight`     `ColParallel{dim=0}`  → `[local_num_v_heads, hidden]`
//!    - `ssm_a` (1-D)         `ColParallel{dim=0}`  → `[local_num_v_heads]`
//!    - `ssm_dt.bias` (1-D)   `ColParallel{dim=0}`  → `[local_num_v_heads]`
//!    - `ssm_conv1d.weight`   `FusedQkvParallel` → `[local_conv_channels, conv_kernel]`
//!    - `ssm_norm.weight`     Replicated (per-`head_v_dim`)
//!    - `ssm_out.weight`      `RowParallel{dim=1}`  → `[hidden, local_d_inner]`
//!
//! 2. **No residual add.** The PP path writes a full-`H` `delta_out` that
//!    the layer driver folds into the residual; the TP path emits the
//!    rank-local partial into `partial_attn_out` (this rank's
//!    contribution to the AllReduce sum). The caller schedules
//!    `BarP2pAllReduce::residual_tp{2,4}` immediately after to fold the
//!    rank-local partials into `hidden`.
//!
//! ## Layer state sizing
//!
//! `GdnLayerState::state` and `GdnLayerState::conv_history` must be
//! sized for the *per-rank* head count. Caller (TP-2d's KV-cache
//! plumbing) is responsible for allocating per-rank state at world-aware
//! sizes. Re-using PP full-shape allocations is correct (kernels only
//! touch the head of each slab) but ~world× wasteful — TP-4d may revisit.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition; same rationale as super::gdn — every \
              unsafe is a kernel launch / memcpy_async over session-scoped \
              DevicePtrs."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    conv::causal_conv1d_f32,
    mlp::{scale_f32, silu_f32, swiglu_f32},
    norm::{l2_norm_f32, quantize_q8_1, rmsnorm_f32, rmsnorm_quant_q8_1},
    qmatmul::{mmvq_q4_0_gate_up, mmvq_q4_0_gate_up_t128, mmvq_q5_k_r2_f16dst, mmvq_q8_0_gate_up},
    recurrent::{gdn_alpha_beta_f32, gdn_state_step_f32_s128},
    HipDevice, HipStream, OpsRegistry,
};

use super::common::{mat_shape, run_mmvq_from_tensor};
use super::gdn::GdnScratch;
use crate::config::Qwen3MoEConfig;
use crate::session::GdnLayerState;
use crate::weights::DeviceTensor;

/// Per-rank decode for one Gated-Delta-Net layer.
///
/// All weight tensors are *already sliced* per the TP-1a/TP-4a-i1
/// layout table. The caller (TP-2d's `forward_full_attn_layer_tp`'s
/// GDN sibling) is responsible for picking them out of `TpLayerTensor`
/// by name.
///
/// # Errors
/// - `tp_world == 0`, or any of `num_v_heads`/`num_k_heads`/`d_inner`
///   not divisible by `tp_world`.
/// - Sliced weight shape mismatch.
/// - Underlying op-dispatch / kernel-launch failures.
#[expect(
    clippy::too_many_arguments,
    reason = "matches super::gdn::forward_gdn_decode's flat-arg list — same \
              rationale: avoid struct copies on the decode hot path."
)]
pub fn forward_gdn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_qkv: &DeviceTensor,
    attn_gate: &DeviceTensor,
    ssm_alpha: &DeviceTensor,
    ssm_beta: &DeviceTensor,
    ssm_a: &DeviceTensor,
    ssm_dt_bias: &DeviceTensor,
    ssm_conv1d: &DeviceTensor,
    ssm_norm: &DeviceTensor,
    ssm_out: &DeviceTensor,
    layer_state: &mut GdnLayerState,
    scratch: &mut GdnScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let gdn = cfg.gdn.as_ref().context("forward_gdn_decode_tp requires cfg.gdn")?;
    let hidden = cfg.hidden_size;
    let d_inner = gdn.d_inner;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    let head_k_dim = gdn.head_k_dim;
    let head_v_dim = gdn.head_v_dim();
    let conv_kernel = gdn.conv_kernel;
    if num_v_heads % world != 0 {
        bail!("num_v_heads {num_v_heads} not divisible by tp_world {tp_world}");
    }
    if num_k_heads % world != 0 {
        bail!("num_k_heads {num_k_heads} not divisible by tp_world {tp_world}");
    }
    if d_inner % world != 0 {
        bail!("d_inner {d_inner} not divisible by tp_world {tp_world}");
    }
    let local_num_v_heads = num_v_heads / world;
    let local_num_k_heads = num_k_heads / world;
    let local_d_inner = d_inner / world;
    let local_qk_size = local_num_k_heads * head_k_dim;
    let local_v_size = local_num_v_heads * head_v_dim;
    let local_conv_channels = local_d_inner + 2 * local_qk_size;
    if local_v_size != local_d_inner {
        bail!("GDN per-rank dim bug: local_v_size {local_v_size} != local_d_inner {local_d_inner}");
    }
    let n_rep = num_v_heads / num_k_heads;

    let probe = std::env::var("FLAMBEAU_TP_PROBE").is_ok();
    macro_rules! probe_f32 {
        ($label:literal, $ptr:expr, $n:expr) => {
            if probe {
                debug_probe_f32(device, stream, $label, $ptr, $n)?;
            }
        };
    }
    macro_rules! probe_f16 {
        ($label:literal, $ptr:expr, $n:expr) => {
            if probe {
                debug_probe_f16(device, stream, $label, $ptr, $n)?;
            }
        };
    }
    probe_f16!("gdn x_in", x_in, hidden);
    if probe {
        eprintln!(
            "    META attn_norm.dtype={:?} dims={:?} bytes={}",
            attn_norm.dtype, attn_norm.dims, attn_norm.bytes
        );
        eprintln!(
            "    META attn_qkv.dtype={:?} dims={:?} bytes={}",
            attn_qkv.dtype, attn_qkv.dims, attn_qkv.bytes
        );
        if attn_norm.dtype == flambeau_quant::GgmlDType::F32 {
            debug_probe_f32(device, stream, "gdn attn_norm.weight", attn_norm.ptr, attn_norm.dims[0] as usize)?;
        } else if attn_norm.dtype == flambeau_quant::GgmlDType::F16 {
            debug_probe_f16(device, stream, "gdn attn_norm.weight", attn_norm.ptr, attn_norm.dims[0] as usize)?;
        }
    }

    if probe {
        debug_probe_q8_1(device, stream, "gdn x_q8_1 PRE-rmsnorm", scratch.x_q8_1, hidden / 32)?;
    }
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
    .context("gdn (TP) attn_norm + quant")?;
    if probe {
        // x_q8_1 has Q8_1 block layout (4 bytes scale + 4 bytes ds + 32 i8 weights = 40 B/block).
        // hidden=4096 -> 128 blocks. Just dump byte stats.
        debug_probe_q8_1(device, stream, "gdn x_q8_1 POST-rmsnorm", scratch.x_q8_1, hidden / 32)?;
        let n_qkv_bytes = (hidden / attn_qkv.dtype.block_size() as usize)
            * attn_qkv.dtype.type_size() as usize
            * 4;
        debug_probe_quant_bytes(device, stream, "gdn attn_qkv first 4 rows", attn_qkv.ptr, n_qkv_bytes)?;
    }

    // 2..3. attn_qkv (FusedQkvParallel sliced) + attn_gate (ColParallel sliced)
    //       projections. Output dims are local_conv_channels and local_d_inner.
    //       Both Q8_0 and Q4_0 have fused gate+up MMVQ variants; pick one.
    let baseline = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let dt_q = attn_qkv.dtype;
    let dt_g = attn_gate.dtype;
    let fuse_qkv_gate_q8_0 = !baseline
        && dt_q == flambeau_quant::GgmlDType::Q8_0
        && dt_g == flambeau_quant::GgmlDType::Q8_0;
    let fuse_qkv_gate_q4_0 = !baseline
        && dt_q == flambeau_quant::GgmlDType::Q4_0
        && dt_g == flambeau_quant::GgmlDType::Q4_0;
    if fuse_qkv_gate_q8_0 {
        mmvq_q8_0_gate_up(
            ops,
            stream,
            attn_qkv.ptr,
            attn_gate.ptr,
            scratch.x_q8_1,
            scratch.qkv_mixed_f32,
            scratch.z_f32,
            local_conv_channels,
            local_d_inner,
            hidden,
        )
        .context("attn_qkv + attn_gate (TP) fused mmvq_q8_0")?;
    } else if fuse_qkv_gate_q4_0 {
        let use_t128 = std::env::var("FLAMBEAU_Q4_0_GU_T128").as_deref() != Ok("off");
        if use_t128 {
            mmvq_q4_0_gate_up_t128(
                ops,
                stream,
                attn_qkv.ptr,
                attn_gate.ptr,
                scratch.x_q8_1,
                scratch.qkv_mixed_f32,
                scratch.z_f32,
                local_conv_channels,
                local_d_inner,
                hidden,
            )
            .context("attn_qkv + attn_gate (TP) fused mmvq_q4_0_t128")?;
        } else {
            mmvq_q4_0_gate_up(
                ops,
                stream,
                attn_qkv.ptr,
                attn_gate.ptr,
                scratch.x_q8_1,
                scratch.qkv_mixed_f32,
                scratch.z_f32,
                local_conv_channels,
                local_d_inner,
                hidden,
            )
            .context("attn_qkv + attn_gate (TP) fused mmvq_q4_0")?;
        }
    } else {
        run_mmvq_from_tensor(
            ops,
            stream,
            attn_qkv,
            scratch.x_q8_1,
            scratch.qkv_mixed_f32,
            local_conv_channels,
            hidden,
            "attn_qkv (TP)",
        )?;
        run_mmvq_from_tensor(
            ops,
            stream,
            attn_gate,
            scratch.x_q8_1,
            scratch.z_f32,
            local_d_inner,
            hidden,
            "attn_gate (TP)",
        )?;
    }
    probe_f32!("gdn qkv_mixed_f32", scratch.qkv_mixed_f32, local_conv_channels);
    probe_f32!("gdn z_f32", scratch.z_f32, local_d_inner);

    // 4..5. ssm_alpha + ssm_beta (both ColParallel, [local_num_v_heads, hidden]).
    let fuse_alpha_beta = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && ssm_alpha.dtype == flambeau_quant::GgmlDType::Q8_0
        && ssm_beta.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_alpha_beta {
        let (a_rows, a_k) = mat_shape(ssm_alpha)?;
        let (b_rows, b_k) = mat_shape(ssm_beta)?;
        if a_rows != local_num_v_heads
            || a_k != hidden
            || b_rows != local_num_v_heads
            || b_k != hidden
        {
            bail!(
                "fused ssm alpha/beta (TP) shape mismatch: alpha=[{a_rows},{a_k}] \
                 beta=[{b_rows},{b_k}] expected=[{local_num_v_heads},{hidden}]"
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
            local_num_v_heads,
            local_num_v_heads,
            hidden,
        )
        .context("ssm alpha+beta (TP) fused mmvq_q8_0")?;
    } else {
        run_mmvq_from_tensor(
            ops,
            stream,
            ssm_alpha,
            scratch.x_q8_1,
            scratch.alpha_f32,
            local_num_v_heads,
            hidden,
            "ssm_alpha (TP)",
        )?;
        run_mmvq_from_tensor(
            ops,
            stream,
            ssm_beta,
            scratch.x_q8_1,
            scratch.beta_f32,
            local_num_v_heads,
            hidden,
            "ssm_beta (TP)",
        )?;
    }

    // 6. Conv1d step — assemble [history, qkv_mixed] → conv_input, run
    //    causal conv, shift history. All ops parameterised by
    //    local_conv_channels.
    flambeau_ops::hip::recurrent::gdn_assemble_conv_input_f32(
        ops,
        stream,
        layer_state.conv_history,
        scratch.qkv_mixed_f32,
        scratch.conv_input,
        local_conv_channels,
        conv_kernel,
    )?;
    causal_conv1d_f32(
        ops,
        stream,
        scratch.conv_input,
        ssm_conv1d.ptr,
        scratch.conv_out,
        1,
        local_conv_channels,
        conv_kernel,
    )
    .context("causal_conv1d_f32 (TP)")?;
    // Inline shift_conv_history: history = conv_input[1..k] (per-rank
    // local_conv_channels-wide rows).
    {
        let row_bytes = local_conv_channels * 4;
        let hist_rows = conv_kernel - 1;
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                layer_state.conv_history,
                scratch.conv_input.offset_bytes(row_bytes),
                hist_rows * row_bytes,
            )?;
        }
    }

    // 7. silu(conv_out).
    silu_f32(ops, stream, scratch.conv_out, scratch.silu_out, local_conv_channels)
        .context("silu_f32(conv_out) (TP)")?;

    // 8. Slice silu_out into Q|K|V via pointer offsets within the
    //    per-rank conv_channels layout: [Q_local | K_local | V_local].
    let q_src = scratch.silu_out;
    let k_src = scratch.silu_out.offset_bytes(local_qk_size * 4);
    let v_src = scratch.silu_out.offset_bytes(2 * local_qk_size * 4);

    // 9. L2-normalise Q and K per (local) head.
    l2_norm_f32(
        ops,
        stream,
        q_src,
        scratch.q_norm_f32,
        local_num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("l2_norm Q (TP)")?;
    l2_norm_f32(
        ops,
        stream,
        k_src,
        scratch.k_norm_f32,
        local_num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("l2_norm K (TP)")?;

    // 10. Scale Q by 1/sqrt(head_k_dim).
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    scale_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        local_qk_size,
        q_scale,
    )
    .context("scale_f32 Q (TP)")?;

    // 11. Per-rank α/β/gate compute. ssm_dt_bias and ssm_a are 1-D
    //     ColParallel (per-v-head). The fused kernel is parameterised
    //     by num_v_heads — pass local_num_v_heads.
    gdn_alpha_beta_f32(
        ops,
        stream,
        scratch.alpha_f32,
        scratch.beta_f32,
        ssm_dt_bias.ptr,
        ssm_a.ptr,
        scratch.gate_device,
        scratch.beta_device,
        local_num_v_heads,
        /* n_tokens = */ 1,
    )
    .context("gdn_alpha_beta_f32 fused (TP)")?;

    // 12. GDN state step — operates on per-rank head subset.
    gdn_state_step_f32_s128(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        v_src,
        scratch.gate_device,
        scratch.beta_device,
        layer_state.state,
        layer_state.state, // state_in/out alias — kernel handles
        scratch.state_out,
        1, // B = 1
        local_num_v_heads,
        1, // L = 1 (decode)
        n_rep,
    )
    .context("gdn_state_step_f32_s128 (TP)")?;

    // 13. ssm_norm per-(local) head on the state-step output.
    let ssm_norm_k = ssm_norm
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
        ssm_norm.ptr,
        scratch.out_normed,
        local_num_v_heads,
        head_v_dim,
        cfg.rms_norm_eps,
    )
    .context("ssm_norm (rmsnorm_f32) (TP)")?;

    // 14. gated = silu(z) * out_normed (per-rank d_inner).
    swiglu_f32(
        ops,
        stream,
        scratch.z_f32,
        scratch.out_normed,
        scratch.gated_f32,
        local_d_inner,
    )
    .context("swiglu_f32(z, out_normed) (TP)")?;

    // 15. Quantise gated → Q8_1 for ssm_out mmvq.
    quantize_q8_1(ops, stream, scratch.gated_f32, scratch.gated_q8_1, local_d_inner)
        .context("quantize gated → Q8_1 (TP)")?;

    // 16+17. Row-parallel ssm_out projection — weight[hidden, local_d_inner]
    //        × gated[local_d_inner] → partial_attn_out[hidden] (full-H partial).
    //
    // **Cycle-4 lever** — when ssm_out is Q5_K (the common case for
    // Qwen3.5/3.6 hybrids), use the F16-dst variant that fuses the
    // cast_f32_to_f16 into the kernel epilogue. Saves 1 launch + 1
    // F32 buffer round-trip per GDN layer per rank. Falls back to
    // the F32+cast pair for any other dtype.
    // **Cycle-4 null**: F16-dst ssm_out measured +0.3% on Qwen3.6-27B
    // TP w=2 — within noise envelope (±0.13 across 3 runs). Only 1 call
    // per GDN layer per rank → 48 calls saved doesn't move the needle.
    // Default OFF; opt in via `FLAMBEAU_SSM_OUT_F16_DST=on` to A/B on
    // other models.
    let ssm_out_f16dst = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && std::env::var("FLAMBEAU_SSM_OUT_F16_DST").as_deref() == Ok("on")
        && ssm_out.dtype == flambeau_quant::GgmlDType::Q5K;
    if ssm_out_f16dst {
        mmvq_q5_k_r2_f16dst(
            ops,
            stream,
            ssm_out.ptr,
            scratch.gated_q8_1,
            partial_attn_out,
            hidden,
            local_d_inner,
        )
        .context("mmvq_q5_k_r2_f16dst ssm_out (TP)")?;
    } else {
        run_mmvq_from_tensor(
            ops,
            stream,
            ssm_out,
            scratch.gated_q8_1,
            scratch.ssm_out_f32,
            hidden,
            local_d_inner,
            "ssm_out (TP)",
        )?;
        probe_f32!("gdn ssm_out_f32 (pre-cast)", scratch.ssm_out_f32, hidden);
        cast_f32_to_f16(ops, stream, scratch.ssm_out_f32, partial_attn_out, hidden)
            .context("cast ssm_out → partial_attn_out (TP)")?;
    }

    Ok(())
}

fn debug_probe_q8_1(
    device: &HipDevice,
    stream: &HipStream,
    label: &str,
    ptr: DevicePtr,
    n_blocks: usize,
) -> Result<()> {
    use flambeau_core::Stream;
    // Q8_1 block: half scale + half ds + i8[32] = 36 B
    let block_bytes = 36;
    let n = n_blocks * block_bytes;
    let mut host = vec![0u8; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n,
        )?;
    }
    stream.synchronize()?;
    // Read scales of first 4 blocks
    let scales: Vec<f32> = (0..n_blocks.min(4))
        .map(|b| {
            let off = b * block_bytes;
            let bits = u16::from_le_bytes([host[off], host[off + 1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();
    let nan = scales.iter().filter(|v| v.is_nan()).count();
    eprintln!("    Q8_1 {label:30}  blocks={n_blocks}  first4_scales={scales:?}  nan={nan}");
    Ok(())
}

fn debug_probe_quant_bytes(
    device: &HipDevice,
    stream: &HipStream,
    label: &str,
    ptr: DevicePtr,
    n_bytes: usize,
) -> Result<()> {
    use flambeau_core::Stream;
    let mut host = vec![0u8; n_bytes];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n_bytes,
        )?;
    }
    stream.synchronize()?;
    let nonzero = host.iter().filter(|&&b| b != 0).count();
    let first16: Vec<u8> = host.iter().take(16).copied().collect();
    eprintln!(
        "    BYTES {label:30}  bytes={n_bytes}  nonzero={nonzero}  first16={first16:?}"
    );
    Ok(())
}

fn debug_probe_f32(
    device: &HipDevice,
    stream: &HipStream,
    label: &str,
    ptr: DevicePtr,
    n: usize,
) -> Result<()> {
    use flambeau_core::Stream;
    let mut host = vec![0.0f32; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n * 4,
        )?;
    }
    stream.synchronize()?;
    let nan = host.iter().filter(|v| v.is_nan()).count();
    let inf = host.iter().filter(|v| v.is_infinite()).count();
    let zero = host.iter().filter(|&&v| v == 0.0).count();
    let finite: Vec<f32> = host.iter().copied().filter(|v| v.is_finite()).collect();
    let (min, max, mean) = if finite.is_empty() {
        (f32::NAN, f32::NAN, f32::NAN)
    } else {
        let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let m = finite.iter().sum::<f32>() / finite.len() as f32;
        (mn, mx, m)
    };
    eprintln!(
        "    F32 {label:30}  n={n:>5}  nan={nan:>5}  inf={inf:>3}  zero={zero:>5}  min={min:.4}  max={max:.4}  mean={mean:.4}"
    );
    Ok(())
}

fn debug_probe_f16(
    device: &HipDevice,
    stream: &HipStream,
    label: &str,
    ptr: DevicePtr,
    n: usize,
) -> Result<()> {
    use flambeau_core::Stream;
    let mut host = vec![0u16; n];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n * 2,
        )?;
    }
    stream.synchronize()?;
    let mut nan = 0;
    let mut zero = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut finite = 0;
    for &b in &host {
        let v = half::f16::from_bits(b).to_f32();
        if v.is_nan() {
            nan += 1;
        } else {
            if v == 0.0 { zero += 1; }
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
            finite += 1;
        }
    }
    let mean = if finite > 0 { sum / finite as f64 } else { f64::NAN };
    eprintln!(
        "    F16 {label:30}  n={n:>5}  nan={nan:>5}  zero={zero:>5}  min={min:.4}  max={max:.4}  mean={mean:.4}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    // Substantive validation needs GPU + per-rank GdnLayerState
    // allocated at local dims; covered by TP-4a-i2's parity smoke.
}
