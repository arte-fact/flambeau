//! tensor-parallel `forward_gdn_decode`.
//! Sister of [`super::gdn::forward_gdn_decode`] for the TP-sharded
//! forward path. Same 17-op structure, with two differences:
//! 1. Per-rank head counts threaded through every kernel call:
//! - `local_num_v_heads = num_v_heads / world`
//! - `local_num_k_heads = num_k_heads / world`
//! - `local_d_inner = local_num_v_heads · head_v_dim`
//! - `local_qk_size = local_num_k_heads · head_k_dim`
//! - `local_conv_channels = local_d_inner + 2 · local_qk_size`
//! - `n_rep = num_v_heads / num_k_heads` (unchanged — divides identically
//! on either side of the per-rank split).
//! Sliced weights (per + ):
//! - `attn_qkv.weight` `FusedQkvParallel` → `[local_conv_channels, hidden]`
//! - `attn_gate.weight` `ColParallel{dim=0}` → `[local_d_inner, hidden]`
//! - `ssm_alpha.weight` `ColParallel{dim=0}` → `[local_num_v_heads, hidden]`
//! - `ssm_beta.weight` `ColParallel{dim=0}` → `[local_num_v_heads, hidden]`
//! - `ssm_a` (1-D) `ColParallel{dim=0}` → `[local_num_v_heads]`
//! - `ssm_dt.bias` (1-D) `ColParallel{dim=0}` → `[local_num_v_heads]`
//! - `ssm_conv1d.weight` `FusedQkvParallel` → `[local_conv_channels, conv_kernel]`
//! - `ssm_norm.weight` Replicated (per-`head_v_dim`)
//! - `ssm_out.weight` `RowParallel{dim=1}` → `[hidden, local_d_inner]`
//! 2. **No residual add.** The PP path writes a full-`H` `delta_out` that
//! the layer driver folds into the residual; the TP path emits the
//! rank-local partial into `partial_attn_out` (this rank's
//! contribution to the AllReduce sum). The caller schedules
//! `BarP2pAllReduce::residual_tp{2,4}` immediately after to fold the
//! rank-local partials into `hidden`.
//! ## Layer state sizing
//! `GdnLayerState::state` and `GdnLayerState::conv_history` must be
//! sized for the *per-rank* head count. Caller (KV-cache
//! plumbing) is responsible for allocating per-rank state at world-aware
//! sizes. Re-using PP full-shape allocations is correct (kernels only
//! touch the head of each slab) but ~world× wasteful — may revisit.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition; same rationale as super::gdn — every \
              unsafe is a kernel launch / memcpy_async over session-scoped \
              DevicePtrs."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    conv::causal_conv1d_f32,
    mlp::{scale_f32, silu_f32, swiglu_f32},
    norm::{
        l2_norm_f32, quantize_f16_q8_1, quantize_f16_q8_1_mmq, quantize_q8_1, quantize_q8_1_mmq,
        rmsnorm_f16, rmsnorm_f32, rmsnorm_quant_q8_1,
    },
    qmatmul::{
        mmvq_q4_0_gate_up, mmvq_q4_0_gate_up_row_tile_batched, mmvq_q4_0_gate_up_t128,
        mmvq_q8_0_gate_up,
    },
    recurrent::{
        gdn_split_qkv_f32, gdn_state_step_alphabeta_f32_s128,
        gdn_state_step_alphabeta_f32_s128_batched_slots,
    },
    HipDevice, HipStream, OpsRegistry,
};

use super::common::{mat_shape, run_mmvq_from_tensor, run_qmatmul_from_tensor};
use super::gdn::{
    assemble_conv_input_prefill, shift_conv_history_prefill, GdnPrefillScratch, GdnScratch,
};
use crate::config::Qwen3MoEConfig;
use crate::session::GdnLayerState;
use crate::weights::DeviceTensor;

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}


/// Per-rank decode for one Gated-Delta-Net layer.
/// All weight tensors are *already sliced* per the /
/// layout table. The caller ( `forward_full_attn_layer_tp`'s
/// GDN sibling) is responsible for picking them out of `TpLayerTensor`
/// by name.
/// # Errors
/// - `tp_world == 0`, or any of `num_v_heads`/`num_k_heads`/`d_inner`
/// not divisible by `tp_world`.
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
    kq_replicated: bool,
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
    if !kq_replicated && num_k_heads % world != 0 {
        bail!("num_k_heads {num_k_heads} not divisible by tp_world {tp_world}");
    }
    if d_inner % world != 0 {
        bail!("d_inner {d_inner} not divisible by tp_world {tp_world}");
    }
    let local_num_v_heads = num_v_heads / world;
    // kq_replicated keeps the full K/Q head count
    // per rank (rep_outer arches: qwen35moe / qwen36moe). See
    // `WeightLayout::FusedQkvParallel` doc for the rationale.
    let local_num_k_heads = if kq_replicated {
        num_k_heads
    } else {
        num_k_heads / world
    };
    let local_d_inner = d_inner / world;
    let local_qk_size = local_num_k_heads * head_k_dim;
    let local_v_size = local_num_v_heads * head_v_dim;
    let local_conv_channels = local_d_inner + 2 * local_qk_size;
    if local_v_size != local_d_inner {
        bail!("GDN per-rank dim bug: local_v_size {local_v_size} != local_d_inner {local_d_inner}");
    }
    // n_rep is the V→K ratio the kernel walks. When K is replicated,
    // each local V head sees its actual matching K head locally:
    // - rep_outer + kq_replicated: H_v_local / H_k = (H_v/world)/H_k
    // - rep_inner contiguous: H_v / H_k (same on local side)
    // Both reduce to local_num_v_heads / local_num_k_heads.
    let n_rep = local_num_v_heads / local_num_k_heads;

    let probe = dev_flag("FLAMBEAU_TP_PROBE");
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
    // projections. Output dims are local_conv_channels and local_d_inner.
    // Both Q8_0 and Q4_0 have fused gate+up MMVQ variants; pick one.
    let dt_q = attn_qkv.dtype;
    let dt_g = attn_gate.dtype;
    let fuse_qkv_gate_q8_0 = dt_q == flambeau_quant::GgmlDType::Q8_0
        && dt_g == flambeau_quant::GgmlDType::Q8_0;
    let fuse_qkv_gate_q4_0 = dt_q == flambeau_quant::GgmlDType::Q4_0
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
        // Shape-aware Q4_0 fused gate+up dispatch (task #53 fix).
        // Cycle-5 made t128 the default; benching on Qwen3.6-35B-A3B-Q4_0
        // (hidden=2048, GDN asymmetric local_conv_channels=2560 vs
        // local_d_inner=2048) showed t128 regresses -15.5% vs the cycle-1
        // 256t baseline at *that* shape. On Qwen3.6-27B-Q4_0 (hidden=5120,
        // dense FFN symmetric) t128 wins +3.3 %.
        // The discriminator that matches the measured sign-flip is
        // (n_rows_gate == n_rows_up) — symmetric → t128, asymmetric → 256t.
        // GDN's attn_qkv (`local_conv_channels`) ≠ attn_gate
        // (`local_d_inner`) → asymmetric → 256t default.
        if local_conv_channels == local_d_inner {
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
    let fuse_alpha_beta = ssm_alpha.dtype == flambeau_quant::GgmlDType::Q8_0
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
    // causal conv, shift history. All ops parameterised by
    // local_conv_channels.
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
    // per-rank conv_channels layout: [Q_local | K_local | V_local].
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

    // 11–12. C10 — per-rank fused state-step (default-on; baseline
    // chain via FLAMBEAU_VARIANT=baseline). ssm_dt_bias / ssm_a are
    // 1-D ColParallel — the local slice is the right per-rank head
    // subset. Operates on local_num_v_heads.
    // /14 — q/k repeat layout differs by arch; see decode-path
    // comment in `super::gdn::forward_gdn_decode` for the explanation.
    let rep_inner_layout = cfg.arch == "qwen3next";
    gdn_state_step_alphabeta_f32_s128(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        v_src,
        scratch.alpha_f32,
        scratch.beta_f32,
        ssm_dt_bias.ptr,
        ssm_a.ptr,
        layer_state.state,
        layer_state.state,
        scratch.state_out,
        1,
        local_num_v_heads,
        1,
        n_rep,
        rep_inner_layout,
    )
    .context("gdn_state_step_alphabeta_f32_s128 (TP, C10 fused)")?;

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

    // 14+15. fused swiglu(z, out_normed) → Q8_1 directly
    // (TP variant). Skips the F32 `gated_f32` intermediate buffer + 1
    // launch. Default-on; FLAMBEAU_VARIANT=baseline opts back to the
    // unfused pair.
    let fuse_tail = local_d_inner % 32 == 0;
    if fuse_tail {
        flambeau_ops::hip::mlp::swiglu_f32_to_q8_1(
            ops,
            stream,
            scratch.z_f32,
            scratch.out_normed,
            scratch.gated_q8_1,
            local_d_inner,
        )
        .context("swiglu_f32_to_q8_1(z, out_normed) (TP)")?;
    } else {
        swiglu_f32(
            ops,
            stream,
            scratch.z_f32,
            scratch.out_normed,
            scratch.gated_f32,
            local_d_inner,
        )
        .context("swiglu_f32(z, out_normed) (TP)")?;
        quantize_q8_1(ops, stream, scratch.gated_f32, scratch.gated_q8_1, local_d_inner)
            .context("quantize gated → Q8_1 (TP)")?;
    }

    // 16+17. Row-parallel ssm_out projection — weight[hidden, local_d_inner]
    // × gated[local_d_inner] → partial_attn_out[hidden] (full-H partial).
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

/// 1** — L-batched per-rank GDN prefill.
/// Sister of [`forward_gdn_decode_tp`] (M=L instead of M=1) and
/// [`super::gdn::forward_gdn_prefill`] (TP-sliced weights instead of
/// PP-full). Same 17-op chain — every kernel call is the L-aware variant
/// and every dim is the per-rank `local_*` count.
/// Output is `partial_attn_out[L, hidden]`: this rank's contribution to
/// the AllReduce sum, written at hidden-stride. The caller schedules
/// `ar_residual_prefill` immediately after to fold the per-rank partials
/// into the residual.
/// State management: `layer_state.state` and `layer_state.conv_history`
/// are read/written in place; the kernels touch only the leading
/// `local_num_v_heads` slabs (PP-allocation is over-allocated for TP, see
/// the gdn_tp.rs preamble).
/// `n_tokens` must be ≤ `scratch.max_tokens`; caller chunks larger
/// prompts.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_gdn_decode_tp + forward_gdn_prefill arg shapes"
)]
pub fn forward_gdn_prefill_tp(
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
    scratch: &mut GdnPrefillScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
    kq_replicated: bool,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_gdn_prefill_tp called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_gdn_prefill_tp: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
            scratch.max_tokens
        );
    }
    let world = tp_world as usize;
    let gdn = cfg.gdn.as_ref().context("forward_gdn_prefill_tp requires cfg.gdn")?;
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
    if !kq_replicated && num_k_heads % world != 0 {
        bail!("num_k_heads {num_k_heads} not divisible by tp_world {tp_world}");
    }
    if d_inner % world != 0 {
        bail!("d_inner {d_inner} not divisible by tp_world {tp_world}");
    }
    let local_num_v_heads = num_v_heads / world;
    // see `forward_gdn_decode_tp` for the kq_replicated
    // rationale (rep_outer head-mapping + contiguous TP split is broken).
    let local_num_k_heads = if kq_replicated {
        num_k_heads
    } else {
        num_k_heads / world
    };
    let local_d_inner = d_inner / world;
    let local_qk_size = local_num_k_heads * head_k_dim;
    let local_v_size = local_num_v_heads * head_v_dim;
    let local_conv_channels = local_d_inner + 2 * local_qk_size;
    if local_v_size != local_d_inner {
        bail!(
            "GDN per-rank dim bug: local_v_size {local_v_size} != local_d_inner {local_d_inner}"
        );
    }
    let n_rep = local_num_v_heads / local_num_k_heads;

    // 1. rmsnorm + dual Q8_1 quantise (std + DS4 MMQ layouts).
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
    .context("gdn prefill (TP) attn_norm")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.x_norm_f16,
        scratch.x_q8_1,
        n_tokens * hidden,
    )
    .context("gdn prefill (TP) x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops,
        stream,
        scratch.x_norm_f16,
        scratch.x_q8_1_mmq,
        hidden,
        n_tokens,
    )
    .context("gdn prefill (TP) x_norm → Q8_1 (MMQ DS4)")?;

    // 2..5. Per-rank projections at M = L. attn_qkv → [L, local_conv_channels],
    // attn_gate → [L, local_d_inner], ssm_alpha/beta → [L, local_num_v_heads].
    run_qmatmul_from_tensor(
        ops,
        stream,
        attn_qkv,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.qkv_mixed_f32,
        n_tokens,
        hidden,
        local_conv_channels,
        "attn_qkv (TP)",
    )?;
    run_qmatmul_from_tensor(
        ops,
        stream,
        attn_gate,
        scratch.x_q8_1,
        scratch.x_q8_1_mmq,
        scratch.z_f32,
        n_tokens,
        hidden,
        local_d_inner,
        "attn_gate (TP)",
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
        local_num_v_heads,
        "ssm_alpha (TP)",
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
        local_num_v_heads,
        "ssm_beta (TP)",
    )?;

    // 6. Conv1d step over L tokens. All buffers are at local-stride layout
    // (PP-allocated rows are wider but kernels only touch the leading
    // local_conv_channels floats per row).
    assemble_conv_input_prefill(
        device,
        stream,
        layer_state.conv_history,
        scratch.qkv_mixed_f32,
        scratch.conv_input,
        n_tokens,
        local_conv_channels,
        conv_kernel,
    )?;
    causal_conv1d_f32(
        ops,
        stream,
        scratch.conv_input,
        ssm_conv1d.ptr,
        scratch.conv_out,
        n_tokens,
        local_conv_channels,
        conv_kernel,
    )
    .context("gdn prefill (TP) causal_conv1d_f32")?;
    shift_conv_history_prefill(
        device,
        stream,
        scratch.conv_input,
        layer_state.conv_history,
        n_tokens,
        local_conv_channels,
        conv_kernel,
    )?;

    // 7. silu(conv_out) over [L, local_conv_channels].
    silu_f32(
        ops,
        stream,
        scratch.conv_out,
        scratch.silu_out,
        n_tokens * local_conv_channels,
    )
    .context("gdn prefill (TP) silu_f32(conv_out)")?;

    // 8. Split silu_out into Q | K | V at local sizes.
    gdn_split_qkv_f32(
        ops,
        stream,
        scratch.silu_out,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        scratch.v_f32,
        n_tokens,
        local_qk_size,
        local_v_size,
    )
    .context("gdn prefill (TP) gdn_split_qkv_f32")?;

    // 9. L2-normalise Q and K per (local) head.
    l2_norm_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * local_num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("gdn prefill (TP) l2_norm Q")?;
    l2_norm_f32(
        ops,
        stream,
        scratch.k_norm_f32,
        scratch.k_norm_f32,
        n_tokens * local_num_k_heads,
        head_k_dim,
        cfg.rms_norm_eps,
    )
    .context("gdn prefill (TP) l2_norm K")?;

    // 10. Scale Q by 1/sqrt(head_k_dim).
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    scale_f32(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.q_norm_f32,
        n_tokens * local_qk_size,
        q_scale,
    )
    .context("gdn prefill (TP) scale_f32 Q")?;

    // 11–12. C10 fused state-step (default) absorbs α/β/gate; baseline
    // chain via FLAMBEAU_VARIANT=baseline. All operands at per-rank
    // local_num_v_heads; n_tokens = L. Same kernel as decode_tp,
    // just with n_tokens > 1.
    // /14 — q/k repeat layout differs by arch; see decode-path
    // comment in `super::gdn::forward_gdn_decode` for the explanation.
    let rep_inner_layout = cfg.arch == "qwen3next";
    gdn_state_step_alphabeta_f32_s128(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        scratch.v_f32,
        scratch.alpha_f32,
        scratch.beta_f32,
        ssm_dt_bias.ptr,
        ssm_a.ptr,
        layer_state.state,
        layer_state.state,
        scratch.state_out,
        1,
        local_num_v_heads,
        n_tokens,
        n_rep,
        rep_inner_layout,
    )
    .context("gdn prefill (TP) gdn_state_step_alphabeta_f32_s128 (C10 fused)")?;

    // 13. ssm_norm per-(local) head over [L, local_num_v_heads, head_v_dim].
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
        n_tokens * local_num_v_heads,
        head_v_dim,
        cfg.rms_norm_eps,
    )
    .context("gdn prefill (TP) ssm_norm (rmsnorm_f32)")?;

    // 14. gated = silu(z) * out_normed across [L, local_d_inner].
    swiglu_f32(
        ops,
        stream,
        scratch.z_f32,
        scratch.out_normed,
        scratch.gated_f32,
        n_tokens * local_d_inner,
    )
    .context("gdn prefill (TP) swiglu_f32(z, out_normed)")?;

    // 15. Quantise gated → both Q8_1 layouts for ssm_out matmul.
    quantize_q8_1(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1,
        n_tokens * local_d_inner,
    )
    .context("gdn prefill (TP) quantise gated → Q8_1 (std)")?;
    quantize_q8_1_mmq(
        ops,
        stream,
        scratch.gated_f32,
        scratch.gated_q8_1_mmq,
        local_d_inner,
        n_tokens,
    )
    .context("gdn prefill (TP) quantise gated → Q8_1 (MMQ DS4)")?;

    // 16. Row-parallel ssm_out projection: weight [hidden, local_d_inner]
    // × gated_q8_1[L, local_d_inner] → ssm_out_f32[L, hidden]. Each
    // rank emits a partial sum; AR after this layer folds them.
    run_qmatmul_from_tensor(
        ops,
        stream,
        ssm_out,
        scratch.gated_q8_1,
        scratch.gated_q8_1_mmq,
        scratch.ssm_out_f32,
        n_tokens,
        local_d_inner,
        hidden,
        "ssm_out (TP)",
    )?;

    // 17. Cast back to F16 → partial_attn_out[L, hidden].
    cast_f32_to_f16(
        ops,
        stream,
        scratch.ssm_out_f32,
        partial_attn_out,
        n_tokens * hidden,
    )
    .context("gdn prefill (TP) cast ssm_out → f16")?;

    Ok(())
}

/// **#285 batched-GDN decode** — single GDN-layer forward over N
/// (slot, layer-state) pairs. Replaces the per-slot loop in
/// `forward_decode_batched_hybrid`'s GDN branch (and PP twin).
/// Shape map (mirrors `forward_gdn_prefill_tp`, with the per-slot
/// state mutation factored into a per-slot inner loop):
/// - **Stages A–C** (rmsnorm + dual Q8_1 quant + attn_qkv proj +
/// attn_gate proj + ssm_alpha proj + ssm_beta proj): batched
/// over n_tokens=N via the prefill kernels. Same launches as the
/// prefill path; one launch per kernel regardless of N.
/// - **Stage D** (per-slot conv1d + silu + split_qkv + l2_norm +
/// scale + state-step + ssm_norm): looped per slot. Each slot's
/// conv_history and recurrent state mutate in place. Conv-input
/// scratch (`scratch.conv_input`) is REUSED per iteration —
/// each slot's K-row temp lives there only during its loop body.
/// conv_out / silu_out / Q/K/V / state_out / out_normed are
/// indexed per-slot via base+s*stride pointers.
/// - **Stages E–F** (swiglu+quant + ssm_out proj + cast → F16):
/// batched over N. Same launches as prefill.
/// `layer_states.len()` must equal `n_tokens`; each entry's per-rank
/// `state` and `conv_history` are mutated in place.
/// `n_tokens` ≤ `scratch.max_tokens` (= INFLIGHT_SLOTS in the
/// batched-decode caller).
/// `kq_replicated` must match the caller's TP layout — same rule as
/// `forward_gdn_decode_tp`.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_gdn_prefill_tp arg shape; the per-slot state \
              slice is passed alongside the prefill scratch."
)]
pub fn forward_gdn_decode_batched_tp(
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
    layer_states: &mut [&mut GdnLayerState],
    scratch: &mut GdnPrefillScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
    kq_replicated: bool,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_gdn_decode_batched_tp called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_gdn_decode_batched_tp: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    if layer_states.len() != n_tokens {
        bail!(
            "forward_gdn_decode_batched_tp: layer_states.len()={} != n_tokens={n_tokens}",
            layer_states.len()
        );
    }

    let world = tp_world as usize;
    let gdn = cfg
        .gdn
        .as_ref()
        .context("forward_gdn_decode_batched_tp requires cfg.gdn")?;
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
    if !kq_replicated && num_k_heads % world != 0 {
        bail!("num_k_heads {num_k_heads} not divisible by tp_world {tp_world}");
    }
    if d_inner % world != 0 {
        bail!("d_inner {d_inner} not divisible by tp_world {tp_world}");
    }
    let local_num_v_heads = num_v_heads / world;
    let local_num_k_heads = if kq_replicated {
        num_k_heads
    } else {
        num_k_heads / world
    };
    let local_d_inner = d_inner / world;
    let local_qk_size = local_num_k_heads * head_k_dim;
    let local_v_size = local_num_v_heads * head_v_dim;
    let local_conv_channels = local_d_inner + 2 * local_qk_size;
    if local_v_size != local_d_inner {
        bail!(
            "GDN per-rank dim bug: local_v_size {local_v_size} != local_d_inner {local_d_inner}"
        );
    }
    let n_rep = local_num_v_heads / local_num_k_heads;

    // === Stage A: rmsnorm + dual Q8_1 quantise (n_tokens=N) ===
    rmsnorm_f16(
        ops, stream, x_in, attn_norm.ptr, scratch.x_norm_f16,
        n_tokens, hidden, cfg.rms_norm_eps,
    )
    .context("gdn batched-decode (TP) attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("gdn batched-decode (TP) x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("gdn batched-decode (TP) x_norm → Q8_1 (MMQ DS4)")?;

    // === Stages B + C: 4 projections at n_tokens=N ===
    // Fused Q4_0 gate+up via the row-tile batched kernel: one launch
    // produces both `qkv_mixed_f32 = attn_qkv · x_norm` and
    // `z_f32 = attn_gate · x_norm` across all N slots, with the Q8_1
    // activation strip staged into LDS once per (R=4 row tile, outer
    // iter) instead of HBM-fetched per row per call. cert.md:
    // 1.32–1.80× vs the two-separate-launch path on GDN-class shapes.
    // Mirrors the m=1 fused dispatch in `forward_gdn_decode_tp`.
    let fuse_qkv_gate_q4_0 = attn_qkv.dtype == flambeau_quant::GgmlDType::Q4_0
        && attn_gate.dtype == flambeau_quant::GgmlDType::Q4_0
        && (2..=4).contains(&n_tokens)
        && std::env::var("FLAMBEAU_GDN_FUSE_Q4_0_ROWTILE").as_deref() != Ok("0");
    if fuse_qkv_gate_q4_0 {
        mmvq_q4_0_gate_up_row_tile_batched(
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
            n_tokens,
        )
        .context("attn_qkv + attn_gate (TP batched-decode) fused row-tile Q4_0")?;
    } else {
        run_qmatmul_from_tensor(
            ops, stream, attn_qkv,
            scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.qkv_mixed_f32,
            n_tokens, hidden, local_conv_channels, "attn_qkv (TP batched-decode)",
        )?;
        run_qmatmul_from_tensor(
            ops, stream, attn_gate,
            scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.z_f32,
            n_tokens, hidden, local_d_inner, "attn_gate (TP batched-decode)",
        )?;
    }
    run_qmatmul_from_tensor(
        ops, stream, ssm_alpha,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.alpha_f32,
        n_tokens, hidden, local_num_v_heads, "ssm_alpha (TP batched-decode)",
    )?;
    run_qmatmul_from_tensor(
        ops, stream, ssm_beta,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.beta_f32,
        n_tokens, hidden, local_num_v_heads, "ssm_beta (TP batched-decode)",
    )?;

    // === Stage D: per-slot conv1d + silu + split_qkv + l2_norm + scale +
    // state-step + ssm_norm.
    // Each slot has its own `conv_history` and recurrent `state` to
    // mutate. The conv/SSM kernels themselves are SMALL (head_dim=128
    // tiles, N=4 → ~50 µs each); the wins from batching come from the
    // big MMVQ stages (A–C, E–F) — Stage D's 6× per-slot launches
    // are an acceptable cost vs running the full GDN N times.
    let row_qkv_bytes = local_conv_channels * 4;
    let row_z_bytes = local_d_inner * 4;
    let row_alpha_bytes = local_num_v_heads * 4;
    let row_qk_bytes = local_qk_size * 4;
    let row_v_bytes = local_v_size * 4;
    let row_state_out_bytes = local_num_v_heads * head_v_dim * 4;
    let row_out_normed_bytes = row_state_out_bytes;
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
    let rep_inner_layout = cfg.arch == "qwen3next";

    let batch_state_step = std::env::var("FLAMBEAU_GDN_STATE_STEP_BATCHED")
        .as_deref()
        != Ok("0");

    for s in 0..n_tokens {
        // Slot pointers into the batched scratch.
        let slot_qkv =
            DevicePtr(scratch.qkv_mixed_f32.as_usize() + s * row_qkv_bytes);
        let slot_conv_out =
            DevicePtr(scratch.conv_out.as_usize() + s * row_qkv_bytes);
        let slot_silu_out =
            DevicePtr(scratch.silu_out.as_usize() + s * row_qkv_bytes);
        let slot_q_norm =
            DevicePtr(scratch.q_norm_f32.as_usize() + s * row_qk_bytes);
        let slot_k_norm =
            DevicePtr(scratch.k_norm_f32.as_usize() + s * row_qk_bytes);
        let slot_v =
            DevicePtr(scratch.v_f32.as_usize() + s * row_v_bytes);
        let slot_alpha =
            DevicePtr(scratch.alpha_f32.as_usize() + s * row_alpha_bytes);
        let slot_beta =
            DevicePtr(scratch.beta_f32.as_usize() + s * row_alpha_bytes);
        let slot_state_out =
            DevicePtr(scratch.state_out.as_usize() + s * row_state_out_bytes);
        let slot_out_normed =
            DevicePtr(scratch.out_normed.as_usize() + s * row_out_normed_bytes);
        let _ = row_z_bytes; // z_f32 read in Stage E batched; row stride for slicing not needed here.

        // 6a. Assemble conv_input from slot's history + slot's qkv_mixed
        // (n_tokens=1 per slot).
        assemble_conv_input_prefill(
            device, stream,
            layer_states[s].conv_history,
            slot_qkv,
            scratch.conv_input,
            1, local_conv_channels, conv_kernel,
        )?;
        // 6b. Conv1d: K-row input → 1-row output for this slot.
        causal_conv1d_f32(
            ops, stream,
            scratch.conv_input, ssm_conv1d.ptr, slot_conv_out,
            1, local_conv_channels, conv_kernel,
        )
        .context("gdn batched-decode (TP) causal_conv1d_f32 slot")?;
        // 6c. Update slot's conv_history with the new row.
        shift_conv_history_prefill(
            device, stream,
            scratch.conv_input,
            layer_states[s].conv_history,
            1, local_conv_channels, conv_kernel,
        )?;

        // 7. silu(conv_out) on this slot's row.
        silu_f32(
            ops, stream, slot_conv_out, slot_silu_out, local_conv_channels,
        )
        .context("gdn batched-decode (TP) silu_f32(conv_out) slot")?;

        // 8. Split silu_out → Q | K | V at local sizes (1-row).
        gdn_split_qkv_f32(
            ops, stream,
            slot_silu_out, slot_q_norm, slot_k_norm, slot_v,
            1, local_qk_size, local_v_size,
        )
        .context("gdn batched-decode (TP) gdn_split_qkv_f32 slot")?;

        // 9. L2-normalise Q and K per local head.
        l2_norm_f32(
            ops, stream, slot_q_norm, slot_q_norm,
            local_num_k_heads, head_k_dim, cfg.rms_norm_eps,
        )
        .context("gdn batched-decode (TP) l2_norm Q slot")?;
        l2_norm_f32(
            ops, stream, slot_k_norm, slot_k_norm,
            local_num_k_heads, head_k_dim, cfg.rms_norm_eps,
        )
        .context("gdn batched-decode (TP) l2_norm K slot")?;

        // 10. Scale Q by 1/sqrt(head_k_dim).
        scale_f32(
            ops, stream, slot_q_norm, slot_q_norm,
            local_qk_size, q_scale,
        )
        .context("gdn batched-decode (TP) scale_f32 Q slot")?;

        // 11–12. Per-slot state-step (when batched_state_step is off).
        // When on, the per-slot state-step is replaced with one batched
        // call over all N slots after this loop ends.
        if !batch_state_step {
            gdn_state_step_alphabeta_f32_s128(
                ops, stream,
                slot_q_norm, slot_k_norm, slot_v,
                slot_alpha, slot_beta,
                ssm_dt_bias.ptr, ssm_a.ptr,
                layer_states[s].state, layer_states[s].state, slot_state_out,
                1, local_num_v_heads, 1, n_rep, rep_inner_layout,
            )
            .context("gdn batched-decode (TP) gdn_state_step_alphabeta_f32_s128 slot")?;

            // 13. ssm_norm per-(local) head over this slot's [num_v_heads, head_v_dim].
            rmsnorm_f32(
                ops, stream,
                slot_state_out, ssm_norm.ptr, slot_out_normed,
                local_num_v_heads, head_v_dim, cfg.rms_norm_eps,
            )
            .context("gdn batched-decode (TP) ssm_norm slot")?;
        }
    }

    // === Stage D' (batched state-step): one launch over all N slots. ===
    // Per-slot Q/K/V/α/β are already laid out slot-major in scratch as
    // `[N, H, head_dim]`; only the per-slot `GdnLayerState::state`
    // pointers differ, so we HtoD-copy a `[N] u64` array of slot state
    // base pointers and let the kernel dereference indirectly.
    if batch_state_step {
        let slot_ptrs: Vec<u64> =
            layer_states.iter().map(|ls| ls.state.as_usize() as u64).collect();
        // HtoD on the same stream as the state-step kernel; ordering is
        // guaranteed by stream-queue semantics so no host-side sync
        // needed. SAFETY: slot_ptrs lives on the host for the lifetime
        // of this call frame; `memcpy_async` only requires the host
        // buffer to remain valid until the next operation on this
        // stream observes the copy, which happens before this function
        // returns. scratch.slot_state_ptrs was sized for max_tokens*8 B.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                scratch.slot_state_ptrs,
                DevicePtr(slot_ptrs.as_ptr() as usize),
                std::mem::size_of_val(slot_ptrs.as_slice()),
            )?;
        }

        gdn_state_step_alphabeta_f32_s128_batched_slots(
            ops, stream,
            scratch.q_norm_f32, scratch.k_norm_f32, scratch.v_f32,
            scratch.alpha_f32, scratch.beta_f32,
            ssm_dt_bias.ptr, ssm_a.ptr,
            scratch.slot_state_ptrs, scratch.slot_state_ptrs,
            scratch.state_out,
            n_tokens, local_num_v_heads, 1, n_rep, rep_inner_layout,
        )
        .context("gdn batched-decode (TP) batched-slots state-step")?;

        // Per-slot ssm_norm (kept per-slot; batching this is the
        // next-smaller lever, deferred).
        for s in 0..n_tokens {
            let slot_state_out =
                DevicePtr(scratch.state_out.as_usize() + s * row_state_out_bytes);
            let slot_out_normed =
                DevicePtr(scratch.out_normed.as_usize() + s * row_out_normed_bytes);
            rmsnorm_f32(
                ops, stream, slot_state_out, ssm_norm.ptr, slot_out_normed,
                local_num_v_heads, head_v_dim, cfg.rms_norm_eps,
            )
            .context("gdn batched-decode (TP) ssm_norm slot (post-batched)")?;
        }
    }

    // === Stage E: swiglu(z, out_normed) → gated_f32 (batched) ===
    swiglu_f32(
        ops, stream,
        scratch.z_f32, scratch.out_normed, scratch.gated_f32,
        n_tokens * local_d_inner,
    )
    .context("gdn batched-decode (TP) swiglu_f32(z, out_normed)")?;

    // 15. Quantise gated → both Q8_1 layouts.
    quantize_q8_1(
        ops, stream,
        scratch.gated_f32, scratch.gated_q8_1, n_tokens * local_d_inner,
    )
    .context("gdn batched-decode (TP) quantise gated → Q8_1 (std)")?;
    quantize_q8_1_mmq(
        ops, stream,
        scratch.gated_f32, scratch.gated_q8_1_mmq, local_d_inner, n_tokens,
    )
    .context("gdn batched-decode (TP) quantise gated → Q8_1 (MMQ DS4)")?;

    // === Stage F: ssm_out projection (batched) → partial_attn_out[N, hidden] ===
    run_qmatmul_from_tensor(
        ops, stream, ssm_out,
        scratch.gated_q8_1, scratch.gated_q8_1_mmq, scratch.ssm_out_f32,
        n_tokens, local_d_inner, hidden, "ssm_out (TP batched-decode)",
    )?;
    cast_f32_to_f16(
        ops, stream, scratch.ssm_out_f32, partial_attn_out, n_tokens * hidden,
    )
    .context("gdn batched-decode (TP) cast ssm_out → f16")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Substantive validation needs GPU + per-rank GdnLayerState
    // allocated at local dims; covered by parity smoke.
}
