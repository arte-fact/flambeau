//! TP-2b — tensor-parallel `forward_full_attn_decode`.
//!
//! Per-rank decode for one full-attention layer on a `world`-rank TP
//! mesh. The function mirrors [`super::attn::forward_full_attn_decode`]
//! but with two differences:
//!
//! 1. Every head-count parameter is the *per-rank* count
//!    (`local_n_heads = n_heads / world`,
//!    `local_n_kv_heads = n_kv_heads / world`). Sliced weights and the
//!    KV cache must already be sized for these locals — the caller
//!    (TP-2d) is responsible for upstream slicing.
//! 2. Output writes a *partial* `[hidden, 1]` F16 vector into
//!    `partial_attn_out`. This is this rank's contribution to the
//!    AllReduce sum that follows the attention block; the AR kernel
//!    (TP-0c `BarP2pAllReduce::residual_tp4`) folds the 4 contributions
//!    into the replicated `hidden` buffer.
//!
//! ## What this function does NOT do
//!
//! - **No AllReduce.** Caller schedules `BarP2pAllReduce` on the
//!   `partial_attn_out` buffer immediately after this returns. We deliberately
//!   keep the AR out of the per-layer call so TP-3a's deferred-AR refactor
//!   can fold attn-AR + ffn-AR into one launch without touching this body.
//! - **No KV cache slot capture.** TP-2b targets the eager (non-graph)
//!   path; graph capture for TP is V2.
//! - **No fused-K+V mmvq path.** The PP version probes
//!   `FLAMBEAU_VARIANT` for the V2.X K+V fusion; for TP-2b correctness
//!   we stick to the unfused split. TP-3 may revisit if profiling
//!   shows attn launch latency dominates.
//! - **No split-K attention path.** TP-2b targets short-context
//!   correctness; long-context split-K stays single-rank for now.
//!
//! ## VGPR / scratch implications
//!
//! `FullAttnScratch` is sized for the *full* (`n_heads`,
//! `n_kv_heads`, `q_width = n_heads · head_dim`) shapes, so under TP=4
//! the scratch buffers are 4× larger than each rank actually needs.
//! That's wasteful but correct — kernels parameterised by
//! `local_n_heads` only touch the head of each scratch slab. TP-2b-i2
//! (sized scratch) is filed but not on this session's critical path.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    attention::{attention_decode_f16_slots, split_q_gate_f16},
    cast::cast_f32_to_f16,
    mlp::sigmoid_mul_f16,
    norm::{quantize_f16_q8_1, rmsnorm_f16, rmsnorm_quant_q8_1},
    pe::rope_neox_partial_f16,
    qmatmul::{mmvq, mmvq_q4_0_kv_f16dst},
    OpsRegistry,
};
use flambeau_runtime::KvCache;

use super::attn::FullAttnScratch;
use super::common::{mat_shape, qdtype_of};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one full-attention layer.
///
/// # Shape contract
///
/// `tp_world ≥ 1`. The per-rank head counts must divide cleanly:
///   `cfg.num_heads % tp_world == 0` and
///   `cfg.num_kv_heads % tp_world == 0`. The TP-1a layout validator
/// guarantees this at load time; we re-assert here as a debug guard.
///
/// Sliced weight shapes (validated below):
///   - `attn_q.dims  == [2 · local_n_heads · head_dim, hidden]`
///   - `attn_k.dims  == [local_n_kv_heads · head_dim, hidden]`
///   - `attn_v.dims  == [local_n_kv_heads · head_dim, hidden]`
///   - `attn_output.dims == [hidden, local_n_heads · head_dim]`
///
/// `attn_norm` / `attn_q_norm` / `attn_k_norm` are Replicated (full
/// shape); `kv_cache` is sized for `local_n_kv_heads` already.
///
/// # Errors
/// - Shape mismatches (returned early as `anyhow::Error`).
/// - Any underlying op-dispatch / kernel-launch failure (propagated).
#[expect(
    clippy::too_many_arguments,
    reason = "matches super::attn::forward_full_attn_decode's flat parameter list — \
              same rationale: avoid struct copies on the decode hot path."
)]
pub fn forward_full_attn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    attn_q: &DeviceTensor,
    attn_k: &DeviceTensor,
    attn_v: &DeviceTensor,
    attn_output: &DeviceTensor,
    attn_q_norm: &DeviceTensor,
    attn_k_norm: &DeviceTensor,
    kv_cache: &mut KvCache<flambeau_runtime::F16Contig, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    partial_attn_out: DevicePtr,
    position: usize,
    tp_world: u32,
    // TP-4d-i2: when true, K/V run with full cfg.num_kv_heads per rank.
    // attn_k/attn_v weights must be Replicated; KvCache is full-sized.
    kv_replicated: bool,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    if n_heads % world != 0 {
        bail!("num_heads {n_heads} not divisible by tp_world {tp_world}");
    }
    let local_n_heads = n_heads / world;
    // TP-4d-i2: when kv_replicated, every rank uses full nKV.
    let local_n_kv_heads = if kv_replicated {
        n_kv_heads
    } else {
        if n_kv_heads % world != 0 {
            bail!(
                "num_kv_heads {n_kv_heads} not divisible by tp_world {tp_world} \
                 and kv_replicated=false (caller bug — pass kv_replicated=true \
                 from Qwen35DenseTpLayout::kv_replicated)"
            );
        }
        n_kv_heads / world
    };
    let local_q_width = local_n_heads * head_dim;
    let rope = &cfg.rope;

    // 1. Fused RMSNorm(x_in) + Q8_1 quantise. Replicated input + output
    //    (the AR'd hidden state is identical on every rank).
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
    .context("attn_norm + quant")?;

    // 2. Q|gate projection — fused [Q | gate], rows = 2 · local_n_heads · head_dim.
    let dtype_q = qdtype_of(attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(attn_q)?;
    let expect_q_rows = 2 * local_n_heads * head_dim;
    if q_rows != expect_q_rows || q_k != hidden {
        bail!(
            "attn_q (TP) shape [{q_rows}, {q_k}] != expected [{expect_q_rows}, {hidden}]"
        );
    }
    mmvq(ops, stream, attn_q.ptr, scratch.x_q8_1, scratch.mmvq_f32, q_rows, q_k, dtype_q)
        .context("mmvq attn_q (TP)")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.q_fused_f16, q_rows)
        .context("cast attn_q → f16 (TP)")?;

    // 3. Split fused [Q | gate] into per-rank Q + per-rank gate.
    split_q_gate_f16(
        ops,
        stream,
        scratch.q_fused_f16,
        scratch.q_f16,
        scratch.gate_f16,
        1,
        local_n_heads,
        head_dim,
    )
    .context("split_q_gate (TP)")?;

    // 4+5. K and V — both Q8_0 typically; per-rank shape [local_n_kv_heads*head_dim, hidden].
    let dtype_k = qdtype_of(attn_k.dtype)?;
    let dtype_v = qdtype_of(attn_v.dtype)?;
    let (k_rows, k_k) = mat_shape(attn_k)?;
    let (v_rows, v_k) = mat_shape(attn_v)?;
    let expect_kv_rows = local_n_kv_heads * head_dim;
    if k_rows != expect_kv_rows || k_k != hidden {
        bail!(
            "attn_k (TP) shape [{k_rows}, {k_k}] != expected [{expect_kv_rows}, {hidden}]"
        );
    }
    if v_rows != expect_kv_rows || v_k != hidden {
        bail!(
            "attn_v (TP) shape [{v_rows}, {v_k}] != expected [{expect_kv_rows}, {hidden}]"
        );
    }
    // **Cycle-3 lever** — fused K+V Q4_0 with F16-dst when both K and V
    // are Q4_0 and shapes match. Saves 1 MMVQ launch + 2 cast launches
    // per full-attn layer per rank. Falls back to the 4-launch path for
    // any non-Q4_0 dtype (e.g. Q8_0 K/V on other models).
    let kv_q4_0_fused = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && std::env::var("FLAMBEAU_KV_F16_DST").as_deref() != Ok("off")
        && attn_k.dtype == flambeau_quant::GgmlDType::Q4_0
        && attn_v.dtype == flambeau_quant::GgmlDType::Q4_0
        && k_rows == v_rows;
    if kv_q4_0_fused {
        mmvq_q4_0_kv_f16dst(
            ops,
            stream,
            attn_k.ptr,
            attn_v.ptr,
            scratch.x_q8_1,
            scratch.k_f16,
            scratch.v_f16,
            k_rows,
            k_k,
        )
        .context("mmvq attn_k+v fused F16-dst (TP)")?;
    } else {
        mmvq(ops, stream, attn_k.ptr, scratch.x_q8_1, scratch.mmvq_f32, k_rows, k_k, dtype_k)
            .context("mmvq attn_k (TP)")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.k_f16, k_rows)
            .context("cast attn_k → f16 (TP)")?;
        mmvq(ops, stream, attn_v.ptr, scratch.x_q8_1, scratch.mmvq_f32, v_rows, v_k, dtype_v)
            .context("mmvq attn_v (TP)")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.v_f16, v_rows)
            .context("cast attn_v → f16 (TP)")?;
    }

    // 6. Per-head RMSNorm on Q and K. Norms are Replicated (per-head_dim).
    let q_norm_dim = attn_q_norm
        .dims
        .first()
        .copied()
        .context("attn_q_norm missing dim")? as usize;
    if q_norm_dim != head_dim {
        bail!("attn_q_norm dim {q_norm_dim} != head_dim {head_dim}");
    }
    rmsnorm_f16(
        ops,
        stream,
        scratch.q_f16,
        attn_q_norm.ptr,
        scratch.q_f16,
        local_n_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("attn_q_norm (TP)")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.k_f16,
        attn_k_norm.ptr,
        scratch.k_f16,
        local_n_kv_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("attn_k_norm (TP)")?;

    // 7. RoPE on Q and K. Multi-freq partial NeoX. Same RoPE freq base
    //    on every rank — it doesn't depend on the head subset.
    scratch.positions_host[0] = position as i32;
    // SAFETY: scratch.positions has 4 valid bytes; positions_host is a
    // persistent Vec on the scratch.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            4,
        )?;
    }
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.q_f16,
        scratch.positions,
        rope.freq_base,
        1,
        local_n_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("rope Q (TP)")?;
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.k_f16,
        scratch.positions,
        rope.freq_base,
        1,
        local_n_kv_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("rope K (TP)")?;

    // 8. Append per-rank K, V to the per-rank KV cache.
    // SAFETY: scratch.k_f16 / scratch.v_f16 are contiguous F16
    // [local_n_kv_heads, head_dim] and the KV cache was sized for
    // local_n_kv_heads at construction.
    unsafe {
        kv_cache
            .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
            .map_err(|e| anyhow::anyhow!("kv_cache.append (TP): {e}"))?;
    }

    // 9. Per-rank attention against the local KV slab.
    let n_tokens_kv = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    attention_decode_f16_slots(
        ops,
        stream,
        scratch.q_f16,
        kv_cache.k_buffer(),
        kv_cache.v_buffer(),
        scratch.attn_out_f16,
        local_n_heads,
        local_n_kv_heads,
        head_dim,
        n_tokens_kv,
        scale,
        None,
    )
    .context("attention_decode_f16 (TP)")?;

    // 10. Sigmoid-gate (NOT SiLU — see V1.7.4.b root-cause note).
    let gated_elems = local_q_width;
    sigmoid_mul_f16(
        ops,
        stream,
        scratch.gate_f16,
        scratch.attn_out_f16,
        scratch.gated_out_f16,
        gated_elems,
    )
    .context("post-attn sigmoid-gate (TP)")?;

    // 11. Quantise gated_out to Q8_1 for the row-parallel output proj.
    quantize_f16_q8_1(ops, stream, scratch.gated_out_f16, scratch.x_q8_1, gated_elems)
        .context("quantize gated_out → Q8_1 (TP)")?;

    // 12. Row-parallel output projection. attn_output is sliced
    //     [hidden, local_q_width] — full output rows but only this
    //     rank's column slab. Result is the per-rank partial that
    //     contributes to the AllReduce sum.
    let dtype_o = qdtype_of(attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(attn_output)?;
    if o_rows != hidden || o_k != gated_elems {
        bail!(
            "attn_output (TP) shape [{o_rows}, {o_k}] != expected [{hidden}, {gated_elems}]"
        );
    }
    mmvq(
        ops,
        stream,
        attn_output.ptr,
        scratch.x_q8_1,
        scratch.mmvq_f32,
        o_rows,
        o_k,
        dtype_o,
    )
    .context("mmvq attn_output (TP)")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, partial_attn_out, o_rows)
        .context("cast attn_output → f16 (TP)")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Substantive tests are GPU-gated and live in TP-2d's parity smoke.
    // A non-GPU shape-derivation check would just re-assert the same
    // arithmetic this file's preconditions enforce; not pulling its weight.
}
