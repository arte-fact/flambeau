//! Forward-pass composition for one layer of a Qwen3.x MoE model.
//!
//! V1.7.3-b: only the **full-attention decode step** is wired. GDN layers
//! (V1.7.3-c) and MoE FFN (V1.7.3-d) land in follow-on sessions; once
//! `forward_gdn_decode` + `forward_moe_ffn_decode` are in place,
//! `forward_one_token` in V1.7.3-e composes the three per-layer functions
//! into the full 40-layer stack.
//!
//! Shape conventions for one decode step (single token, Qwen3.6-35B):
//! - hidden `H = 2048`, `n_heads = 16`, `n_kv_heads = 2`, `head_dim = 256`
//! - fused Q|gate projection width: `2 * n_heads * head_dim = 8192`
//! - K/V projection width: `n_kv_heads * head_dim = 512`
//! - post-attention intermediate: `n_heads * head_dim = 4096`
//!
//! All intermediates are F16 except MMVQ accumulator outputs, which are
//! F32 and get cast back with `ops::cast::cast_f32_to_f16`.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::hip::{
    attention::{attention_decode_f16, attention_prefill_f16, split_q_gate_f16},
    cast::{cast_f16_to_f32, cast_f32_to_f16},
    conv::causal_conv1d_f32,
    mlp::{add_f16, scale_f32, sigmoid_mul_f16, silu_f32, swiglu_f32},
    moe::{
        indexed_moe_mmq_q4_k_down_tile8, indexed_moe_mmq_q4_k_down_turbo,
        indexed_moe_mmq_q4_k_gate_up_tile8, indexed_moe_mmq_q4_k_gate_up_turbo,
        indexed_moe_mmq_q6_k_down_tile8,
        indexed_moe_mmvq_q4_k_gate_up, indexed_moe_mmvq_q4_k_gate_up_sorted,
        indexed_moe_mmvq_q4_k_r2, indexed_moe_mmvq_q4_k_r2_sorted,
        indexed_moe_mmvq_q4_0, indexed_moe_mmvq_q6_k, indexed_moe_mmvq_q8_0,
        moe_combine_f16, moe_sort_by_expert,
        moe_sort_by_expert_padded, shared_expert_scale_f32, topk_f32,
    },
    norm::{
        l2_norm_f32, quantize_f16_q8_1, quantize_f16_q8_1_mmq, quantize_q8_1,
        quantize_q8_1_mmq, rmsnorm_f16, rmsnorm_f32,
        rmsnorm_quant_q8_1,
    },
    pe::rope_neox_partial_f16,
    qmatmul::{mmvq, mmvq_q8_0_gate_up, qmatmul},
    recurrent::{gdn_alpha_beta_f32, gdn_split_qkv_f32, gdn_state_step_f32_s128},
    router::dense_gemv_f32_f16,
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::{BlockQ8_1, GgmlDType};
use flambeau_runtime::KvCache;

use crate::config::Qwen3MoEConfig;
use crate::session::{GdnLayerState, LayerCache};
use crate::weights::{DenseFfnWeights, DeviceTensor, FullAttnWeights, GdnWeights};

/// Workspace buffers needed by one decode step of a full-attention layer.
/// Sized once at session init against the model config; shared across all
/// full-attn layers (they all have the same intermediate dims).
pub struct FullAttnScratch {
    pub x_norm: DevicePtr,         // F16 [H]
    pub x_q8_1: DevicePtr,         // Q8_1 blocks [H / 32]
    pub mmvq_f32: DevicePtr,       // F32 [max(fused_q_width, H)]
    pub q_fused_f16: DevicePtr,    // F16 [2 * n_heads * head_dim]
    pub q_f16: DevicePtr,          // F16 [n_heads * head_dim]
    pub gate_f16: DevicePtr,       // F16 [n_heads * head_dim]
    pub k_f16: DevicePtr,          // F16 [n_kv_heads * head_dim]
    pub v_f16: DevicePtr,          // F16 [n_kv_heads * head_dim]
    pub attn_out_f16: DevicePtr,   // F16 [n_heads * head_dim]
    pub gated_out_f16: DevicePtr,  // F16 [n_heads * head_dim]
    pub positions: DevicePtr,      // i32 [1] — position for the current token
    // V2.19.b — split-K (flash-decoding) partials. Sized for
    // `MAX_SPLITK_CHUNKS` chunks so the scratch can serve any context up to
    // `MAX_SPLITK_CHUNKS * SPLITK_CHUNK_SIZE_LONG` tokens; dispatch asserts
    // `n_chunks <= MAX_SPLITK_CHUNKS`.
    pub splitk_partials_m: DevicePtr,  // F32 [n_heads * MAX_SPLITK_CHUNKS]
    pub splitk_partials_s: DevicePtr,  // F32 [n_heads * MAX_SPLITK_CHUNKS]
    pub splitk_partials_o: DevicePtr,  // F32 [n_heads * MAX_SPLITK_CHUNKS * head_dim]
    // Sizes for teardown + sanity asserts.
    x_norm_bytes: usize,
    x_q8_1_bytes: usize,
    mmvq_f32_bytes: usize,
    q_fused_bytes: usize,
    qk_bytes: usize,
    kv_bytes: usize,
    attn_bytes: usize,
    positions_bytes: usize,
    splitk_ms_bytes: usize,
    splitk_o_bytes: usize,
    disposed: bool,
}

/// V2.19.b — partials scratch budget. 32 chunks × 512 tokens/chunk = 16 384
/// tokens max context covered by split-K (≥ anything practical on gfx906
/// decode). Bump alongside the dispatch threshold if context ever exceeds.
pub const MAX_SPLITK_CHUNKS: usize = 32;

impl FullAttnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;

        let q_fused_width = 2 * n_heads * head_dim;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");

        let x_norm_bytes = hidden * 2;
        // `x_q8_1` is reused for two activations: the RMSNorm-output quant
        // of `hidden` elements (feeds Q/K/V matmuls), and the post-attn
        // gated_out quant of `n_heads*head_dim` elements (feeds the output
        // matmul). Size for the max — Qwen3.6 has `n_heads*head_dim=4096 >
        // hidden=2048`, so budgeting only `hidden/32` blocks OOB-writes.
        let x_q8_1_elems = hidden.max(q_width);
        assert!(x_q8_1_elems % 32 == 0, "x_q8_1 elems must be multiple of QK8_1=32");
        let x_q8_1_bytes = (x_q8_1_elems / 32) * std::mem::size_of::<BlockQ8_1>();
        // Max MMVQ output width across all layer matmuls:
        //   attn_q: q_fused_width (8192)
        //   attn_output: hidden (2048)
        //   attn_k/v: kv_width (512)
        let mmvq_f32_bytes = q_fused_width.max(hidden) * 4;
        let q_fused_bytes = q_fused_width * 2;
        let qk_bytes = q_width * 2;
        let kv_bytes = kv_width * 2;
        let attn_bytes = q_width * 2;
        let positions_bytes = 4;
        // splitk partials: f32 × [n_heads, MAX_CHUNKS] (m, s) and
        // f32 × [n_heads, MAX_CHUNKS, head_dim] (o).
        let splitk_ms_bytes = n_heads * MAX_SPLITK_CHUNKS * 4;
        let splitk_o_bytes = n_heads * MAX_SPLITK_CHUNKS * head_dim * 4;

        let x_norm = device.alloc(x_norm_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let mmvq_f32 = device.alloc(mmvq_f32_bytes)?;
        let q_fused_f16 = device.alloc(q_fused_bytes)?;
        let q_f16 = device.alloc(qk_bytes)?;
        let gate_f16 = device.alloc(qk_bytes)?;
        let k_f16 = device.alloc(kv_bytes)?;
        let v_f16 = device.alloc(kv_bytes)?;
        let attn_out_f16 = device.alloc(attn_bytes)?;
        let gated_out_f16 = device.alloc(attn_bytes)?;
        let positions = device.alloc(positions_bytes)?;
        let splitk_partials_m = device.alloc(splitk_ms_bytes)?;
        let splitk_partials_s = device.alloc(splitk_ms_bytes)?;
        let splitk_partials_o = device.alloc(splitk_o_bytes)?;

        Ok(Self {
            x_norm,
            x_q8_1,
            mmvq_f32,
            q_fused_f16,
            q_f16,
            gate_f16,
            k_f16,
            v_f16,
            attn_out_f16,
            gated_out_f16,
            positions,
            splitk_partials_m,
            splitk_partials_s,
            splitk_partials_o,
            x_norm_bytes,
            x_q8_1_bytes,
            mmvq_f32_bytes,
            q_fused_bytes,
            qk_bytes,
            kv_bytes,
            attn_bytes,
            positions_bytes,
            splitk_ms_bytes,
            splitk_o_bytes,
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
            device.dealloc(self.x_norm, self.x_norm_bytes)?;
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.mmvq_f32, self.mmvq_f32_bytes)?;
            device.dealloc(self.q_fused_f16, self.q_fused_bytes)?;
            device.dealloc(self.q_f16, self.qk_bytes)?;
            device.dealloc(self.gate_f16, self.qk_bytes)?;
            device.dealloc(self.k_f16, self.kv_bytes)?;
            device.dealloc(self.v_f16, self.kv_bytes)?;
            device.dealloc(self.attn_out_f16, self.attn_bytes)?;
            device.dealloc(self.gated_out_f16, self.attn_bytes)?;
            device.dealloc(self.positions, self.positions_bytes)?;
            device.dealloc(self.splitk_partials_m, self.splitk_ms_bytes)?;
            device.dealloc(self.splitk_partials_s, self.splitk_ms_bytes)?;
            device.dealloc(self.splitk_partials_o, self.splitk_o_bytes)?;
        }
        Ok(())
    }
}

impl Drop for FullAttnScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "FullAttnScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Write `[position]` as an i32 into the 4-byte `positions` scratch slot.
/// Used by RoPE to pick the angle per token.
fn upload_position(device: &HipDevice, stream: &HipStream, dst: DevicePtr, position: i32) -> Result<()> {
    let host = [position];
    // SAFETY: `dst` has 4 valid bytes; `host` is 4 valid host bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            dst,
            DevicePtr(host.as_ptr() as usize),
            4,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Map our `GgmlDType` (from GGUF) to the `QDtype` the qmatmul dispatcher
/// uses. Only the dtypes our V1 kernels support are allowed here; everything
/// else is a load-time error.
fn qdtype_of(dtype: GgmlDType) -> Result<QDtype> {
    Ok(match dtype {
        GgmlDType::Q4K => QDtype::Q4_K,
        GgmlDType::Q5K => QDtype::Q5_K,
        GgmlDType::Q6K => QDtype::Q6_K,
        GgmlDType::Q8_0 => QDtype::Q8_0,
        GgmlDType::Q4_1 => QDtype::Q4_1,
        // V2.21.b — UD-Q8_K_XL reserves F16 for i-matrix-flagged layers
        // (Qwen3.6-27B-UD-Q8_K_XL: all 48 attn_gate + 48 ssm_out + scattered
        // attn_q/k + ffn_gate/up/down). `mmvq()` special-cases F16 to skip
        // the dispatch table and call the direct F16×Q8_1 kernel.
        GgmlDType::F16 => QDtype::F16,
        // V2.23.a — Q4_0 and Q5_0 unblock Qwen3.6-35B-A3B-Q4_0.
        GgmlDType::Q4_0 => QDtype::Q4_0,
        GgmlDType::Q5_0 => QDtype::Q5_0,
        other => bail!("weight dtype {other:?} not supported by V1 qmatmul dispatch"),
    })
}

/// Pull the (n_rows, k) pair out of a weight tensor's GGUF dims.
///
/// `flambeau_quant::GgufFile` reverses the on-wire dim order at parse time,
/// so `dims` is **outermost-first**: for a 2D weight `[n_rows, k]` we have
/// `dims = [n_rows, k]`.
fn mat_shape(w: &DeviceTensor) -> Result<(usize, usize)> {
    if w.dims.len() != 2 {
        bail!(
            "expected a 2D weight tensor for `{}`, got dims {:?}",
            w.name,
            w.dims
        );
    }
    let n_rows = w.dims[0] as usize;
    let k = w.dims[1] as usize;
    Ok((n_rows, k))
}

/// Decode step for one full-attention layer. Consumes `x_in` (F16 `[H]`)
/// and writes the pre-residual output to `delta_out` (F16 `[H]`). The
/// caller is expected to do the residual sum (`out = x_in + delta_out`)
/// outside this function — V1.7.3-e adds the fused residual-add kernel
/// for the top-level compose.
///
/// Appends to `kv_cache` at the current tail. `position` is the 0-based
/// token index used by RoPE and also the `n_tokens_kv` for the attention
/// kernel after the append bumps the cache size by 1.
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    post_attn_norm: Option<&DeviceTensor>,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<flambeau_runtime::F16Contig, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
) -> Result<()> {
    // V1.7.3-b wires the attention block only; the FFN side of the
    // residual is V1.7.3-d. `post_attn_norm` is still unused here; keep
    // the handle so V1.7.3-d can call it without a second signature.
    let _ = post_attn_norm;

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let rope = &cfg.rope;

    // 1. Fused RMSNorm(x_in) + Q8_1 quantise.
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

    // 2. Q|gate projection. `attn_q.weight` rows = `2 * n_heads * head_dim`
    //    (fused). Output lands in F32; cast to F16 for the downstream
    //    F16-only kernels.
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != 2 * n_heads * head_dim || q_k != hidden {
        bail!(
            "attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            2 * n_heads * head_dim,
            hidden
        );
    }
    mmvq(
        ops,
        stream,
        weights.attn_q.ptr,
        scratch.x_q8_1,
        scratch.mmvq_f32,
        q_rows,
        q_k,
        dtype_q,
    )
    .context("mmvq attn_q")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.q_fused_f16, q_rows)
        .context("cast attn_q → f16")?;

    // 3. Split Q and gate halves out of the fused projection.
    split_q_gate_f16(
        ops,
        stream,
        scratch.q_fused_f16,
        scratch.q_f16,
        scratch.gate_f16,
        1,
        n_heads,
        head_dim,
    )
    .context("split_q_gate")?;

    // 4+5. K and V projections. Both Q8_0, both [n_kv_heads*head_dim, hidden],
    // both read the same x_q8_1. Fuse when FLAMBEAU_VARIANT=dp4a_vdr2 via
    // mmvq_q8_0_gate_up kernel (writes to 2 F32 buffers). Avoids one launch
    // per full-attn layer + halves activation HBM reads on this path.
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if k_rows != n_kv_heads * head_dim || k_k != hidden {
        bail!(
            "attn_k shape [{k_rows}, {k_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    if v_rows != n_kv_heads * head_dim || v_k != hidden {
        bail!(
            "attn_v shape [{v_rows}, {v_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    let fuse_kv = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && weights.attn_k.dtype == flambeau_quant::GgmlDType::Q8_0
        && weights.attn_v.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_kv {
        // Fused K+V matmul, then two casts (K, V go to different F16 dsts).
        // K output → mmvq_f32[0..k_rows]; V output → mmvq_f32[k_rows..k_rows+v_rows].
        // scratch.mmvq_f32 is sized for attn_q (8192 rows), so 1024-row K+V fits.
        let v_f32_offset = scratch.mmvq_f32.offset_bytes(k_rows * 4);
        mmvq_q8_0_gate_up(
            ops,
            stream,
            weights.attn_k.ptr,
            weights.attn_v.ptr,
            scratch.x_q8_1,
            scratch.mmvq_f32,
            v_f32_offset,
            k_rows,
            v_rows,
            k_k,
        )
        .context("attn_k + attn_v fused mmvq_q8_0")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.k_f16, k_rows)
            .context("cast attn_k → f16")?;
        cast_f32_to_f16(ops, stream, v_f32_offset, scratch.v_f16, v_rows)
            .context("cast attn_v → f16")?;
    } else {
        mmvq(
            ops, stream, weights.attn_k.ptr, scratch.x_q8_1,
            scratch.mmvq_f32, k_rows, k_k, dtype_k,
        ).context("mmvq attn_k")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.k_f16, k_rows)
            .context("cast attn_k → f16")?;
        mmvq(
            ops, stream, weights.attn_v.ptr, scratch.x_q8_1,
            scratch.mmvq_f32, v_rows, v_k, dtype_v,
        ).context("mmvq attn_v")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.v_f16, v_rows)
            .context("cast attn_v → f16")?;
    }

    // 6. Per-head RMSNorm on Q and K.
    let q_norm_dim = weights
        .attn_q_norm
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
        weights.attn_q_norm.ptr,
        scratch.q_f16,
        n_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("attn_q_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.k_f16,
        weights.attn_k_norm.ptr,
        scratch.k_f16,
        n_kv_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("attn_k_norm")?;

    // 7. RoPE on Q and K. Multi-freq partial NeoX for Qwen3.5/3.6 text-only.
    upload_position(device, stream, scratch.positions, position as i32)
        .context("positions upload")?;
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.q_f16,
        scratch.positions,
        rope.freq_base,
        1,
        n_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("rope Q")?;
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.k_f16,
        scratch.positions,
        rope.freq_base,
        1,
        n_kv_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("rope K")?;

    // 8. Append K, V to the KV cache at the tail slot.
    // SAFETY: scratch.k_f16 and v_f16 are contiguous F16 `[n_kv_heads, head_dim]`.
    unsafe {
        kv_cache
            .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
            .map_err(|e| anyhow::anyhow!("kv_cache.append: {e}"))?;
    }

    // 9. Attention decode against the full cache (includes the token we
    //    just appended — `current_tokens = position + 1`).
    //
    // V2.19.b — split-K (flash-decoding) for long contexts. The single-pass
    // kernel hits 27 % CU occupancy (16 heads × 1 block on 60 CUs) and
    // serialises over n_tokens_kv per block; at n_tokens=2048 that's 2647 µs
    // vs split-K's 340 µs (7.78×). FLAMBEAU_VARIANT=baseline opts out.
    let n_tokens_kv = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    let use_splitk = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && n_tokens_kv > 256;
    if use_splitk {
        let chunk_size = flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv);
        let n_chunks = n_tokens_kv.div_ceil(chunk_size);
        debug_assert!(
            n_chunks <= MAX_SPLITK_CHUNKS,
            "split-K n_chunks={n_chunks} exceeds scratch budget MAX={MAX_SPLITK_CHUNKS}"
        );
        flambeau_ops::hip::attention::attention_decode_f16_splitk(
            ops,
            stream,
            scratch.q_f16,
            kv_cache.k_buffer(),
            kv_cache.v_buffer(),
            scratch.attn_out_f16,
            scratch.splitk_partials_m,
            scratch.splitk_partials_s,
            scratch.splitk_partials_o,
            n_heads,
            n_kv_heads,
            head_dim,
            n_tokens_kv,
            chunk_size,
            scale,
        )
        .context("attention_decode_f16_splitk")?;
    } else {
        attention_decode_f16(
            ops,
            stream,
            scratch.q_f16,
            kv_cache.k_buffer(),
            kv_cache.v_buffer(),
            scratch.attn_out_f16,
            n_heads,
            n_kv_heads,
            head_dim,
            n_tokens_kv,
            scale,
        )
        .context("attention_decode_f16")?;
    }

    // 10. Post-attention sigmoid-gate: gated_out = sigmoid(gate) * attn_out.
    // Qwen3.5/3.6 uses a plain logistic sigmoid (per llama.cpp qwen35moe.cpp
    // `gate_sigmoid = sigmoid(Qcur_full view); attn_gated = attn * gate_sigmoid`)
    // NOT SiLU. Using swiglu here adds an extra factor of `gate`; that was
    // V1.7.4.b's root cause — our full-attn layer 3 diverged 10-25× per element
    // from llama.cpp, cascading through the remaining 37 layers into garbage
    // logits. See `project_v1_7_4_b_sigmoid_gate.md`.
    let gated_elems = n_heads * head_dim;
    sigmoid_mul_f16(
        ops,
        stream,
        scratch.gate_f16,
        scratch.attn_out_f16,
        scratch.gated_out_f16,
        gated_elems,
    )
    .context("post-attn sigmoid-gate")?;

    // 11. Quantise gated_out to Q8_1 for the output projection. Fused
    // F16 → Q8_1 kernel lands from V1.7.3-g; replaces the earlier host
    // roundtrip.
    quantize_f16_q8_1(ops, stream, scratch.gated_out_f16, scratch.x_q8_1, gated_elems)
        .context("quantize gated_out → Q8_1")?;

    // 12. Output projection `[hidden, n_heads*head_dim]`.
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != gated_elems {
        bail!(
            "attn_output shape [{o_rows}, {o_k}] != expected [{}, {}]",
            hidden,
            gated_elems
        );
    }
    mmvq(
        ops,
        stream,
        weights.attn_output.ptr,
        scratch.x_q8_1,
        scratch.mmvq_f32,
        o_rows,
        o_k,
        dtype_o,
    )
    .context("mmvq attn_output")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, delta_out, o_rows)
        .context("cast attn_output → f16")?;

    Ok(())
}

/// Route a `LayerCache` entry through the full-attn forward, pulling the
/// correct `KvCache` out of the enum. Fails if the layer is actually a
/// GDN layer (caller dispatch error).
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attn_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
) -> Result<()> {
    let LayerCache::FullAttn(kv) = layer_cache else {
        bail!(
            "layer {} is not a full-attn layer (cache variant mismatch)",
            layer_weights.layer_idx
        );
    };
    let crate::weights::AttnWeights::FullAttn(fa) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not FullAttn variant",
            layer_weights.layer_idx
        );
    };
    forward_full_attn_decode(
        ops,
        stream,
        device,
        cfg,
        &layer_weights.attn_norm,
        layer_weights.post_attention_norm.as_ref(),
        fa,
        kv,
        scratch,
        x_in,
        delta_out,
        position,
    )
}


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

/// Assemble the conv1d input tile `[conv_kernel, conv_channels]` from the
/// layer's history + a fresh `qkv_mixed` row. Uses device-to-device
/// `memcpy_async` (stream-ordered with the surrounding kernels).
fn assemble_conv_input(
    device: &HipDevice,
    stream: &HipStream,
    history: DevicePtr,
    current: DevicePtr,
    conv_input: DevicePtr,
    conv_channels: usize,
    conv_kernel: usize,
) -> Result<()> {
    let row_bytes = conv_channels * 4;
    let hist_rows = conv_kernel - 1;
    // SAFETY: `conv_input` has `conv_kernel * row_bytes` valid device
    // bytes; the source buffers are at least `hist_rows * row_bytes` and
    // `row_bytes` respectively.
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
            current,
            row_bytes,
        )?;
    }
    Ok(())
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
#[allow(clippy::too_many_arguments)]
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

    // 2..5. Four hidden-input projections share the Q8_1 input.
    // attn_qkv + attn_gate fuse when VARIANT=dp4a_vdr2 — both Q8_0, same K=hidden,
    // different N (conv_channels vs d_inner). The extended gate_up kernel handles
    // asymmetric n_rows by grid=max(n1,n2) with per-output early-return.
    let fuse_qkv_gate = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && weights.attn_qkv.dtype == flambeau_quant::GgmlDType::Q8_0
        && weights.attn_gate.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_qkv_gate {
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
    } else {
        run_mmvq_from_tensor(ops, stream, &weights.attn_qkv, scratch.x_q8_1, scratch.qkv_mixed_f32, conv_channels, hidden, "attn_qkv")?;
        run_mmvq_from_tensor(ops, stream, &weights.attn_gate, scratch.x_q8_1, scratch.z_f32, d_inner, hidden, "attn_gate")?;
    }
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

    // 6. Conv1d step — assemble [history_{k-1}, qkv_mixed] into conv_input,
    // run causal conv, then shift history forward.
    assemble_conv_input(
        device,
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

    // 7. silu(conv_out).
    silu_f32(ops, stream, scratch.conv_out, scratch.silu_out, conv_channels)
        .context("silu_f32(conv_out)")?;

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

    // 11. Alpha/β/gate compute — fused device kernel (V1.7.3-g, V2.2.d fix 3
    // batched across tokens). Decode runs n_tokens = 1.
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

    // 12. GDN state step. V (aliased as v_src) stays in place; Q/K have
    // been l2-normalised and Q scaled. n_rep = num_v_heads / num_k_heads.
    let n_rep = num_v_heads / num_k_heads;
    gdn_state_step_f32_s128(
        ops,
        stream,
        scratch.q_norm_f32,
        scratch.k_norm_f32,
        v_src,
        scratch.gate_device,
        scratch.beta_device,
        layer_state.state,
        layer_state.state,   // state_in and state_out alias — kernel handles it
        scratch.state_out,
        1,                   // B = 1
        num_v_heads,
        1,                   // L = 1 (decode)
        n_rep,
    )
    .context("gdn_state_step_f32_s128")?;

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

    // 14. gated = silu(z) * out_normed.
    if v_size != d_inner {
        bail!(
            "GDN layout bug: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})"
        );
    }
    swiglu_f32(ops, stream, scratch.z_f32, scratch.out_normed, scratch.gated_f32, d_inner)
        .context("swiglu_f32(z, out_normed)")?;

    // 15. Quantise gated → Q8_1 for ssm_out mmvq.
    quantize_q8_1(ops, stream, scratch.gated_f32, scratch.gated_q8_1, d_inner)
        .context("quantize gated → Q8_1")?;

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

    // 17. Cast back to F16 for the residual path.
    cast_f32_to_f16(ops, stream, scratch.ssm_out_f32, delta_out, hidden)
        .context("cast ssm_out → f16")?;

    Ok(())
}

/// Helper: run an MMVQ against a weight `DeviceTensor`, validating dims
/// and dispatching on dtype. Keeps the forward body readable.
#[allow(clippy::too_many_arguments)]
fn run_mmvq_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    expected_rows: usize,
    expected_k: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    mmvq(ops, stream, w.ptr, act_q8_1, dst, rows, k, dtype)
        .with_context(|| format!("mmvq {label}"))
}

/// Route a `LayerCache` entry through the GDN forward, pulling the
/// correct `GdnLayerState` out of the enum and the matching `GdnWeights`
/// out of the layer's `AttnWeights` variant.
#[allow(clippy::too_many_arguments)]
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
// V1.7.3-d1 — routed MoE FFN decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the routed MoE FFN (no shared expert —
/// that lands in V1.7.3-d2, no router — V1.7.3-d3). Sized against
/// `(hidden, moe_intermediate_size, num_experts_per_tok=top_k)`.
pub struct MoeScratch {
    // Q8_1 of layer input, shared across all top_k experts' gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Router logits (F32 [n_experts]) → populated by `forward_router_decode`.
    pub router_logits: DevicePtr,
    // Expert ids / weights — populated by the router (or the caller).
    pub expert_ids: DevicePtr,        // i32 [top_k]
    pub expert_weights: DevicePtr,    // F32 [top_k]
    // Fused gate+up MMVQ outputs: F32 [top_k, moe_inter] each.
    pub gate_out_f32: DevicePtr,
    pub up_out_f32: DevicePtr,
    // swiglu(gate, up) result: F32 [top_k, moe_inter], then cast to F16,
    // then quantised to Q8_1 (flat [top_k, moe_inter/32]) for the down step.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down MMVQ output: F32 [top_k, hidden], then cast to F16 for combine.
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    router_logits_bytes: usize,
    expert_ids_bytes: usize,
    expert_weights_bytes: usize,
    gate_up_bytes: usize,
    activated_f32_bytes: usize,
    activated_f16_bytes: usize,
    activated_q8_1_bytes: usize,
    down_f32_bytes: usize,
    down_f16_bytes: usize,
    disposed: bool,
}

impl MoeScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(inter % 32 == 0, "moe_intermediate_size must be a multiple of QK8_1=32");

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let router_logits_bytes = cfg.num_experts * 4;
        let expert_ids_bytes = top_k * 4;
        let expert_weights_bytes = top_k * 4;
        let gate_up_bytes = top_k * inter * 4;
        let activated_f32_bytes = top_k * inter * 4;
        let activated_f16_bytes = top_k * inter * 2;
        let activated_q8_1_bytes = top_k * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let down_f32_bytes = top_k * hidden * 4;
        let down_f16_bytes = top_k * hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let router_logits = device.alloc(router_logits_bytes)?;
        let expert_ids = device.alloc(expert_ids_bytes)?;
        let expert_weights = device.alloc(expert_weights_bytes)?;
        let gate_out_f32 = device.alloc(gate_up_bytes)?;
        let up_out_f32 = device.alloc(gate_up_bytes)?;
        let activated_f32 = device.alloc(activated_f32_bytes)?;
        let activated_f16 = device.alloc(activated_f16_bytes)?;
        let activated_q8_1 = device.alloc(activated_q8_1_bytes)?;
        let down_f32 = device.alloc(down_f32_bytes)?;
        let down_f16 = device.alloc(down_f16_bytes)?;

        Ok(Self {
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
            x_q8_1_bytes,
            router_logits_bytes,
            expert_ids_bytes,
            expert_weights_bytes,
            gate_up_bytes,
            activated_f32_bytes,
            activated_f16_bytes,
            activated_q8_1_bytes,
            down_f32_bytes,
            down_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.router_logits, self.router_logits_bytes)?;
            device.dealloc(self.expert_ids, self.expert_ids_bytes)?;
            device.dealloc(self.expert_weights, self.expert_weights_bytes)?;
            device.dealloc(self.gate_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.up_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.activated_f32, self.activated_f32_bytes)?;
            device.dealloc(self.activated_f16, self.activated_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.activated_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.down_f32_bytes)?;
            device.dealloc(self.down_f16, self.down_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for MoeScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "MoeScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of the routed MoE FFN. Assumes the caller has:
/// - Run `post_attention_norm` on the residual stream (so `x_norm` is the
///   norm output).
/// - Already filled `scratch.expert_ids` and `scratch.expert_weights` with
///   the router's output. V1.7.3-d3 will land the router; until then the
///   caller is synthetic (test fixture or hand-rolled top-k).
///
/// The final `out` is computed as `residual + Σ_k weight_k · expert_out_k`,
/// matching `moe_combine_f16`'s semantics — so `out` already has the
/// residual fused in and the outer loop can skip a second residual add.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_ffn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn: &crate::weights::FfnWeights,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    let ffn_gate_exps = ffn.ffn_gate_exps.as_ref().expect("MoE forward: ffn_gate_exps missing");
    let ffn_up_exps = ffn.ffn_up_exps.as_ref().expect("MoE forward: ffn_up_exps missing");
    let ffn_down_exps = ffn.ffn_down_exps.as_ref().expect("MoE forward: ffn_down_exps missing");
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let top_k = cfg.num_experts_per_tok;
    let _ = cfg.num_experts; // presently only asserted by dims on ffn_gate_exps

    // 1. Quantise x_norm to Q8_1 for the gate/up matmul.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("moe x_norm → Q8_1")?;

    // 2. Fused gate + up matmul across top_k selected experts in one launch.
    // Weight shape (outermost-first): `[n_experts, inter, hidden]`. The
    // indexed_moe kernels take n_sb_per_row = hidden / QK_K.
    const QK_K: usize = 256;
    if hidden % QK_K != 0 {
        bail!("MoE expects hidden={hidden} divisible by QK_K={QK_K}");
    }
    if inter % QK_K != 0 {
        bail!("MoE expects moe_intermediate_size={inter} divisible by QK_K={QK_K}");
    }
    let nb_per_row_hidden = hidden / QK_K;
    let nb_per_row_inter = inter / QK_K;

    // gate+up may both be Q4_K (standard UD-Q4_K_S) or Q8_0 (V2.22.a:
    // UD-Q8_K_XL). down may be Q4_K, Q6_K (UD-Q4_K_S ffn_down promotion),
    // or Q8_0 (UD-Q8_K_XL). BF16 layers in UD-Q8_K_XL aren't handled here
    // yet — loader converts them to Q8_0 on host.
    let gate_dt = ffn_gate_exps.dtype;
    let up_dt = ffn_up_exps.dtype;
    if !(gate_dt == up_dt
        && (gate_dt == GgmlDType::Q4K
            || gate_dt == GgmlDType::Q8_0
            || gate_dt == GgmlDType::Q4_0))
    {
        bail!(
            "indexed-MoE gate/up dtypes must match and be Q4_K, Q8_0 or Q4_0; got gate={:?}, up={:?}",
            gate_dt, up_dt
        );
    }
    if ffn_down_exps.dtype != GgmlDType::Q4K
        && ffn_down_exps.dtype != GgmlDType::Q6K
        && ffn_down_exps.dtype != GgmlDType::Q8_0
        && ffn_down_exps.dtype != GgmlDType::Q4_0
    {
        bail!(
            "indexed-MoE ffn_down_exps must be Q4_K, Q6_K, Q8_0 or Q4_0; got {:?}",
            ffn_down_exps.dtype
        );
    }

    // Block size per row: Q4_K/Q5_K/Q6_K use 256-elem super-blocks; Q8_0,
    // Q4_0 and Q5_0 use 32-elem blocks (same as the Q8_1 activation block).
    let nb_per_row_hidden_gate = match gate_dt {
        GgmlDType::Q8_0 | GgmlDType::Q4_0 => hidden / 32,
        _ => nb_per_row_hidden,
    };

    match gate_dt {
        GgmlDType::Q4K => indexed_moe_mmvq_q4_k_gate_up(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            inter,
            1,
            top_k,
            nb_per_row_hidden,
        )
        .context("indexed_moe gate+up q4_k")?,
        GgmlDType::Q8_0 => {
            indexed_moe_mmvq_q8_0(
                ops, stream, ffn_gate_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
                scratch.gate_out_f32, inter, 1, top_k, nb_per_row_hidden_gate,
            ).context("indexed_moe gate q8_0")?;
            indexed_moe_mmvq_q8_0(
                ops, stream, ffn_up_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
                scratch.up_out_f32, inter, 1, top_k, nb_per_row_hidden_gate,
            ).context("indexed_moe up q8_0")?;
        }
        GgmlDType::Q4_0 => {
            indexed_moe_mmvq_q4_0(
                ops, stream, ffn_gate_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
                scratch.gate_out_f32, inter, 1, top_k, nb_per_row_hidden_gate,
            ).context("indexed_moe gate q4_0")?;
            indexed_moe_mmvq_q4_0(
                ops, stream, ffn_up_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
                scratch.up_out_f32, inter, 1, top_k, nb_per_row_hidden_gate,
            ).context("indexed_moe up q4_0")?;
        }
        _ => unreachable!("gate dtype was validated above"),
    }

    // 3. SwiGLU(gate, up) → activated, F32.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f32,
        top_k * inter,
    )
    .context("moe swiglu_f32")?;

    // 4. Cast activated F32 → F16, then quantise F16 → Q8_1. Layout is
    // `[top_k, inter]` flat — each row of `activated_q8_1` is one expert's
    // input to the down matmul (re-used as one "effective token" below).
    cast_f32_to_f16(ops, stream, scratch.activated_f32, scratch.activated_f16, top_k * inter)
        .context("cast activated → f16")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        top_k * inter,
    )
    .context("quantise activated → Q8_1")?;

    // 5. Down matmul. Indexed MoE MMVQ dispatches by
    // `expert_ids[token * top_k + slot]`. Treat each of our top_k routed
    // experts as its own "effective token" with `top_k = 1` and its own
    // expert id. The scratch already holds `expert_ids[0..top_k]` which
    // doubles as the flat expert lookup (`flat[i] = expert_ids[0 * 1 + i]`).
    match ffn_down_exps.dtype {
        GgmlDType::Q4K => indexed_moe_mmvq_q4_k_r2(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            top_k, // n_tokens_effective
            1,     // top_k=1 in this re-indexed view
            nb_per_row_inter,
        )
        .context("indexed_moe down q4_k r2")?,
        GgmlDType::Q6K => indexed_moe_mmvq_q6_k(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            top_k,
            1,
            nb_per_row_inter,
        )
        .context("indexed_moe down q6_k")?,
        GgmlDType::Q8_0 => indexed_moe_mmvq_q8_0(
            ops, stream,
            ffn_down_exps.ptr, scratch.activated_q8_1, scratch.expert_ids,
            scratch.down_f32,
            hidden, top_k, 1, inter / 32,
        ).context("indexed_moe down q8_0")?,
        GgmlDType::Q4_0 => indexed_moe_mmvq_q4_0(
            ops, stream,
            ffn_down_exps.ptr, scratch.activated_q8_1, scratch.expert_ids,
            scratch.down_f32,
            hidden, top_k, 1, inter / 32,
        ).context("indexed_moe down q4_0")?,
        other => bail!("unreachable: ffn_down_exps dtype {other:?} should have been rejected"),
    }

    // 6. Cast expert outputs to F16 for the combine kernel.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        top_k * hidden,
    )
    .context("cast down → f16")?;

    // 7. Weighted sum + residual. `moe_combine_f16` writes
    //   out[h] = residual[h] + Σ_k weights[k] · expert_outs[k, h]
    // into the caller's `out`. `expert_outs[n_tokens=1, top_k, hidden]`
    // is `scratch.down_f16` in place.
    moe_combine_f16(
        ops,
        stream,
        scratch.down_f16,
        scratch.expert_weights,
        residual,
        out,
        1,
        top_k,
        hidden,
    )
    .context("moe_combine_f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-d2 — shared expert decode step.
// ---------------------------------------------------------------------------

/// Workspace for one decode step of the shared expert (dense FFN +
/// per-token sigmoid gate scaling). Sized against
/// `(hidden, shared_expert_intermediate_size)`.
pub struct SharedExpertScratch {
    // Q8_1 of `x_norm`, shared across gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Dense gate/up matmul outputs, F32 [shared_inter].
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    // SwiGLU output + F16 round-trip for the down matmul input.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down matmul output (F32), scaled in place by `shared_expert_scale_f32`.
    pub down_f32: DevicePtr,
    // F32 view of `x_norm` — the gate-scale kernel dots it against
    // `ffn_gate_inp_shexp` to produce the per-token gate scalar.
    pub x_norm_f32: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl SharedExpertScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg
            .shared_expert_intermediate_size
            .context("SharedExpertScratch requires cfg.shared_expert_intermediate_size")?;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(
            inter % 32 == 0,
            "shared_expert_intermediate_size must be a multiple of QK8_1=32"
        );

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = inter * 4;
        let inter_f16_bytes = inter * 2;
        let inter_q8_1_bytes = (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let x_norm_f32 = device.alloc(hidden_f32_bytes)?;

        Ok(Self {
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
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
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.x_norm_f32, self.hidden_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for SharedExpertScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "SharedExpertScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of the shared expert (always-on dense FFN), composed
/// with the learned per-token sigmoid-gate scaling:
///
///   gate_scalar[t] = sigmoid(⟨ ffn_gate_inp_shexp, x_norm[t] ⟩)
///   dense[t]       = down_shexp(swiglu(gate_shexp(x_norm[t]), up_shexp(x_norm[t])))
///   shared_out[t]  = gate_scalar[t] * dense[t]
///
/// Output (`shared_out`) is a standalone F16 **delta** — the caller is
/// expected to sum it with the routed-MoE output and the residual in
/// V1.7.3-e. Keeping this delta-only keeps the composition orthogonal:
/// routed and shared contributions flow through the same combine layer in
/// V1.7.3-e without re-using `residual` for a second purpose.
#[allow(clippy::too_many_arguments)]
pub fn forward_shared_expert_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    shared: &crate::weights::SharedExpertWeights,
    scratch: &mut SharedExpertScratch,
    x_norm: DevicePtr,
    shared_out: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_decode requires cfg.shared_expert_intermediate_size")?;

    // 1. Quantise x_norm → Q8_1 for gate/up matmuls.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("shexp x_norm → Q8_1")?;

    // 2+3. Dense gate + up matmuls. Fuse when FLAMBEAU_VARIANT=dp4a_vdr2 so
    // the shared Q8_1 activation is read once, saving one kernel launch per
    // layer per forward. Both weights must be Q8_0 for the fused path.
    let fuse_gate_up = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && shared.ffn_gate_shexp.dtype == flambeau_quant::GgmlDType::Q8_0
        && shared.ffn_up_shexp.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_gate_up {
        let (g_rows, g_k) = mat_shape(&shared.ffn_gate_shexp)?;
        let (u_rows, u_k) = mat_shape(&shared.ffn_up_shexp)?;
        if g_rows != inter || g_k != hidden || u_rows != inter || u_k != hidden {
            bail!(
                "fused shexp gate/up shape mismatch: gate=[{g_rows},{g_k}] up=[{u_rows},{u_k}] expected=[{inter},{hidden}]"
            );
        }
        mmvq_q8_0_gate_up(
            ops,
            stream,
            shared.ffn_gate_shexp.ptr,
            shared.ffn_up_shexp.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            inter,
            inter,
            hidden,
        )
        .context("shexp mmvq_q8_0_gate_up (fused)")?;
    } else {
        run_mmvq_from_tensor(
            ops,
            stream,
            &shared.ffn_gate_shexp,
            scratch.x_q8_1,
            scratch.gate_f32,
            inter,
            hidden,
            "ffn_gate_shexp",
        )?;
        run_mmvq_from_tensor(
            ops,
            stream,
            &shared.ffn_up_shexp,
            scratch.x_q8_1,
            scratch.up_f32,
            inter,
            hidden,
            "ffn_up_shexp",
        )?;
    }

    // 4. swiglu(gate, up) → activated_f32.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f32,
        inter,
    )
    .context("shexp swiglu_f32")?;

    // 5. Cast + quantise activated for the down matmul input.
    // Tried F32→Q8_1 direct (skip F16 intermediate) — regressed ~1% because
    // the F32 quantize kernel is slower per-element than the F16 one (no
    // packed fp16 max-reduction). Two small kernels beat one big one here.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        inter,
    )
    .context("shexp cast activated → f16")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        inter,
    )
    .context("shexp quantise activated → Q8_1")?;

    // 6. Dense down matmul → down_f32 [hidden].
    run_mmvq_from_tensor(
        ops,
        stream,
        &shared.ffn_down_shexp,
        scratch.activated_q8_1,
        scratch.down_f32,
        hidden,
        inter,
        "ffn_down_shexp",
    )?;

    // 7. Apply the learned per-token sigmoid gate scaling in place.
    // The kernel needs F32 views of both `shared_out` (the dense FFN
    // result) and `x_norm` (the layer input the gate learns from).
    cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, hidden)
        .context("shexp cast x_norm → f32")?;
    shared_expert_scale_f32(
        ops,
        stream,
        scratch.down_f32,
        scratch.x_norm_f32,
        shared.ffn_gate_inp_shexp.ptr,
        1,
        hidden,
    )
    .context("shared_expert_scale_f32")?;

    // 8. Cast the scaled output back to F16 for the outer composition.
    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_out, hidden)
        .context("shexp cast → f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V2.2.c — dense FFN decode (arch=qwen35).
//
// One gate+up+down triple per layer (no router, no experts, no shared expert).
// Structurally identical to forward_shared_expert_decode minus the
// sigmoid-gate post-scale. `forward_dense_ffn_decode` writes the residual sum
// `x_out = residual + FFN(x_norm)` in one shot so callers don't need to add
// later.
// ---------------------------------------------------------------------------

pub struct DenseFfnScratch {
    // Q8_1 of `x_norm`, shared across gate/up matmuls.
    pub x_q8_1: DevicePtr,
    // Dense gate/up matmul outputs, F32 [inter].
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    // SwiGLU output + F16 round-trip for the down matmul input.
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // Down matmul outputs (F32 → F16) for the residual add.
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    // Bookkeeping.
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    hidden_f16_bytes: usize,
    disposed: bool,
}

impl DenseFfnScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(inter % 32 == 0, "inter (moe_intermediate_size) must be a multiple of QK8_1=32");

        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = inter * 4;
        let inter_f16_bytes = inter * 2;
        let inter_q8_1_bytes = (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = hidden * 4;
        let hidden_f16_bytes = hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let down_f16 = device.alloc(hidden_f16_bytes)?;

        Ok(Self {
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            down_f16,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            hidden_f32_bytes,
            hidden_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.down_f16, self.hidden_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for DenseFfnScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "DenseFfnScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of a dense FFN block (arch=qwen35). Writes
/// `x_out = residual + ffn_down(swiglu(ffn_gate(x_norm), ffn_up(x_norm)))`
/// into `x_out`. No router, no experts, no sigmoid-gate scaling.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense_ffn_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    dense: &DenseFfnWeights,
    scratch: &mut DenseFfnScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    x_out: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;

    // 1. Quantise x_norm → Q8_1 once, reused for gate/up.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("dense ffn x_norm → Q8_1")?;

    // 2+3. gate + up matmuls share x_q8_1. V2.20.b — when both are Q8_0
    //      (Qwen3.6-27B dense path, 66.3% of decode wall pre-fusion), fuse
    //      into one mmvq_q8_0_gate_up launch. Kernel is the same one the
    //      full-attn layer uses for K+V fusion; parity cert in
    //      `crates/bench/tests/mmvq_q8_0_gate_up_parity.rs` proves bit-exact
    //      equivalence to two independent single-row calls. Disabled only
    //      by the coarse `FLAMBEAU_VARIANT=baseline` or the specific
    //      `FLAMBEAU_DENSE_GATE_UP=unfused`.
    let global_baseline = std::env::var("FLAMBEAU_VARIANT").as_deref() == Ok("baseline");
    let specific_off = std::env::var("FLAMBEAU_DENSE_GATE_UP").as_deref() == Ok("unfused");
    let fuse_gate_up = !global_baseline && !specific_off
        && dense.ffn_gate.dtype == flambeau_quant::GgmlDType::Q8_0
        && dense.ffn_up.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_gate_up {
        mmvq_q8_0_gate_up(
            ops,
            stream,
            dense.ffn_gate.ptr,
            dense.ffn_up.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            inter,
            inter,
            hidden,
        )
        .context("dense ffn gate+up fused mmvq_q8_0")?;
    } else {
        qmatmul(
            ops,
            stream,
            dense.ffn_gate.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.gate_f32,
            1,
            hidden,
            inter,
            qdtype_of(dense.ffn_gate.dtype)?,
        )
        .context("dense ffn gate qmatmul")?;
        qmatmul(
            ops,
            stream,
            dense.ffn_up.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.up_f32,
            1,
            hidden,
            inter,
            qdtype_of(dense.ffn_up.dtype)?,
        )
        .context("dense ffn up qmatmul")?;
    }

    // 4. SwiGLU(gate, up) → activated_f32.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f32,
        inter,
    )
    .context("dense ffn swiglu_f32")?;

    // 5. Cast + quantise activated for the down matmul.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        inter,
    )
    .context("dense ffn cast activated → f16")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        inter,
    )
    .context("dense ffn quantise activated → Q8_1")?;

    // 6. down matmul: weight[hidden, inter] × activated[inter] → down_f32[hidden].
    //    Decode path: m=1 never hits MmqLdsX64.
    qmatmul(
        ops,
        stream,
        dense.ffn_down.ptr,
        scratch.activated_q8_1,
        DevicePtr(0),
        scratch.down_f32,
        1,
        inter,
        hidden,
        qdtype_of(dense.ffn_down.dtype)?,
    )
    .context("dense ffn down qmatmul")?;

    // 7. Cast down F32→F16, residual add into x_out.
    cast_f32_to_f16(ops, stream, scratch.down_f32, scratch.down_f16, hidden)
        .context("dense ffn cast down → f16")?;
    add_f16(ops, stream, residual, scratch.down_f16, x_out, hidden)
        .context("dense ffn residual: residual + down")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V2.2.c — dense FFN prefill (arch=qwen35, L tokens).
// ---------------------------------------------------------------------------

pub struct DenseFfnPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    /// V2.2.d.P8 — DS4 Q8_1 MMQ layout sibling of `x_q8_1`, consumed by the
    /// 4-warp LDS-tiled Q4_1 MMQ (and future Q4_K / Q6_K MMQ turbo kernels)
    /// at m ≥ 128. Populated from `x_q8_1` F16 source via the
    /// `flambeau_quantize_f16_q8_1_mmq` kernel alongside the standard quant.
    pub x_q8_1_mmq: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    /// V2.2.d.P8 — DS4 sibling of `activated_q8_1` for the down-projection.
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    inter_q8_1_mmq_bytes: usize,
    hidden_f32_bytes: usize,
    hidden_f16_bytes: usize,
    disposed: bool,
}

impl DenseFfnPrefillScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice, max_tokens: usize) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        assert!(max_tokens >= 1);
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128"
        );
        assert!(
            inter % 128 == 0,
            "moe_intermediate_size must be a multiple of QK8_1_MMQ=128"
        );
        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let inter_f32_bytes = max_tokens * inter * 4;
        let inter_f16_bytes = max_tokens * inter * 2;
        let inter_q8_1_bytes = max_tokens * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_q8_1_mmq_bytes = max_tokens * (inter / 128) * mmq_block;
        let hidden_f32_bytes = max_tokens * hidden * 4;
        let hidden_f16_bytes = max_tokens * hidden * 2;
        Ok(Self {
            max_tokens,
            x_q8_1: device.alloc(x_q8_1_bytes)?,
            x_q8_1_mmq: device.alloc(x_q8_1_mmq_bytes)?,
            gate_f32: device.alloc(inter_f32_bytes)?,
            up_f32: device.alloc(inter_f32_bytes)?,
            activated_f32: device.alloc(inter_f32_bytes)?,
            activated_f16: device.alloc(inter_f16_bytes)?,
            activated_q8_1: device.alloc(inter_q8_1_bytes)?,
            activated_q8_1_mmq: device.alloc(inter_q8_1_mmq_bytes)?,
            down_f32: device.alloc(hidden_f32_bytes)?,
            down_f16: device.alloc(hidden_f16_bytes)?,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            inter_q8_1_mmq_bytes,
            hidden_f32_bytes,
            hidden_f16_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.activated_q8_1_mmq, self.inter_q8_1_mmq_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.down_f16, self.hidden_f16_bytes)?;
        }
        Ok(())
    }
}

impl Drop for DenseFfnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "DenseFfnPrefillScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward_dense_ffn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    dense: &DenseFfnWeights,
    scratch: &mut DenseFfnPrefillScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    x_out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;

    // Quantise x_norm to BOTH Q8_1 layouts: the standard per-row layout
    // consumed by MMVQ / Mmq4Warp kernels, and the DS4 MMQ layout consumed
    // by the 4-warp LDS-tiled turbo kernel (Q4_1 at m ≥ 128). qmatmul()
    // dispatches to whichever matches the weight dtype + M.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("dense ffn prefill x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(ops, stream, x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
        .context("dense ffn prefill x_norm → Q8_1 (MMQ DS4)")?;

    qmatmul(
        ops, stream,
        dense.ffn_gate.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.gate_f32,
        n_tokens, hidden, inter,
        qdtype_of(dense.ffn_gate.dtype)?,
    ).context("dense ffn prefill gate qmatmul")?;
    qmatmul(
        ops, stream,
        dense.ffn_up.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.up_f32,
        n_tokens, hidden, inter,
        qdtype_of(dense.ffn_up.dtype)?,
    ).context("dense ffn prefill up qmatmul")?;

    swiglu_f32(ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f32, n_tokens * inter)
        .context("dense ffn prefill swiglu_f32")?;
    cast_f32_to_f16(ops, stream, scratch.activated_f32, scratch.activated_f16, n_tokens * inter)
        .context("dense ffn prefill cast activated → f16")?;
    quantize_f16_q8_1(ops, stream, scratch.activated_f16, scratch.activated_q8_1, n_tokens * inter)
        .context("dense ffn prefill quantise activated → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(ops, stream, scratch.activated_f16, scratch.activated_q8_1_mmq, inter, n_tokens)
        .context("dense ffn prefill quantise activated → Q8_1 (MMQ DS4)")?;

    qmatmul(
        ops, stream,
        dense.ffn_down.ptr,
        scratch.activated_q8_1, scratch.activated_q8_1_mmq,
        scratch.down_f32,
        n_tokens, inter, hidden,
        qdtype_of(dense.ffn_down.dtype)?,
    ).context("dense ffn prefill down qmatmul")?;

    cast_f32_to_f16(ops, stream, scratch.down_f32, scratch.down_f16, n_tokens * hidden)
        .context("dense ffn prefill cast down → f16")?;
    add_f16(ops, stream, residual, scratch.down_f16, x_out, n_tokens * hidden)
        .context("dense ffn prefill residual: residual + down")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-d3 — MoE router.
// ---------------------------------------------------------------------------

/// Run the MoE router for one decode token. Reads `x_norm` and the FFN's
/// `ffn_gate_inp` weight; writes the top-k selected expert ids + their
/// softmaxed weights into the MoE scratch buffers that
/// `forward_moe_ffn_decode` consumes.
///
/// Two-stage path:
///   1. `dense_gemv_f32_f16(ffn_gate_inp, x_norm)` → `router_logits` F32 [n_experts]
///   2. `topk_f32(router_logits, expert_ids, expert_weights, 1, n_experts, top_k)`
///
/// The router weight must be F32 — Qwen3.x GGUFs don't quantise this
/// particular tensor (`ffn_gate_inp.weight`) since it's tiny.
pub fn forward_router_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_inp: &DeviceTensor,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let n_experts = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;

    if ffn_gate_inp.dtype != GgmlDType::F32 {
        bail!(
            "router expects F32 ffn_gate_inp; got {:?}",
            ffn_gate_inp.dtype
        );
    }
    // Weight dims (outermost-first): `[n_experts, hidden]`.
    if ffn_gate_inp.dims.len() != 2 {
        bail!(
            "ffn_gate_inp: expected 2D weight, got dims {:?}",
            ffn_gate_inp.dims
        );
    }
    let w_rows = ffn_gate_inp.dims[0] as usize;
    let w_k = ffn_gate_inp.dims[1] as usize;
    if w_rows != n_experts || w_k != hidden {
        bail!(
            "ffn_gate_inp shape [{w_rows}, {w_k}] != expected [{n_experts}, {hidden}]"
        );
    }

    dense_gemv_f32_f16(
        ops,
        stream,
        ffn_gate_inp.ptr,
        x_norm,
        scratch.router_logits,
        n_experts,
        hidden,
    )
    .context("router dense_gemv_f32_f16")?;

    topk_f32(
        ops,
        stream,
        scratch.router_logits,
        scratch.expert_ids,
        scratch.expert_weights,
        1,
        n_experts,
        top_k,
    )
    .context("router topk_f32")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-e2 — token embedding gather.
// ---------------------------------------------------------------------------

/// Byte count per vocabulary row for a 2D weight `[vocab, hidden]` of the
/// given dtype.
fn row_bytes_for_dtype(dtype: GgmlDType, hidden: usize) -> Result<usize> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    if block_size > 1 && hidden % block_size != 0 {
        bail!(
            "token_embd hidden {hidden} is not a multiple of block_size {block_size} for {:?}",
            dtype
        );
    }
    let n_blocks = hidden / block_size;
    Ok(n_blocks * type_size)
}

/// Look up a single token's embedding row and write it as F16 into
/// `out_f16`. Decode-path only (one token). Path: download the row's raw
/// bytes from `token_embd` on device → dequantise on host → cast to F16 →
/// upload to `out_f16`.
///
/// Per-token cost at Qwen3.6 dims: ~1 KB download + 256-block dequant +
/// ~4 KB upload + two stream syncs. Negligible at any realistic tg
/// throughput. Future optimisation: an on-device `gather_q4_k_to_f16`
/// kernel that bypasses the host roundtrip.
pub fn forward_embed_decode_host(
    device: &HipDevice,
    stream: &HipStream,
    token_embd: &DeviceTensor,
    token_id: u32,
    out_f16: DevicePtr,
    hidden: usize,
) -> Result<()> {
    if token_embd.dims.len() != 2 {
        bail!(
            "token_embd: expected 2D weight [vocab, hidden], got dims {:?}",
            token_embd.dims
        );
    }
    let vocab = token_embd.dims[0] as usize;
    let w_k = token_embd.dims[1] as usize;
    if w_k != hidden {
        bail!(
            "token_embd inner dim {w_k} != config hidden {hidden}"
        );
    }
    if (token_id as usize) >= vocab {
        bail!("token_id {token_id} >= vocab {vocab}");
    }

    let row_bytes = row_bytes_for_dtype(token_embd.dtype, hidden)?;
    let offset = token_id as usize * row_bytes;
    if offset + row_bytes > token_embd.bytes {
        bail!(
            "token_embd row out of bounds: token_id={token_id} row_bytes={row_bytes} \
             total_bytes={}",
            token_embd.bytes
        );
    }

    // 1. Download the row's raw bytes.
    let mut row_raw = vec![0u8; row_bytes];
    let src = token_embd.ptr.offset_bytes(offset);
    // SAFETY: `src` points to at least `row_bytes` valid device bytes
    // (checked above); `row_raw` is a host vec of the same length.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(row_raw.as_mut_ptr() as usize),
            src,
            row_bytes,
        )?;
    }
    stream.synchronize()?;

    // 2. Dequantise on host. F16 fast-path avoids the F32 round-trip.
    let row_f16: Vec<half::f16> = if token_embd.dtype == GgmlDType::F16 {
        bytemuck::cast_slice::<u8, half::f16>(&row_raw).to_vec()
    } else {
        let row_f32 =
            flambeau_quant::dequantize_to_vec(token_embd.dtype, &row_raw, hidden)
                .map_err(|e| anyhow::anyhow!("dequant token_embd row {token_id}: {e}"))?;
        row_f32
            .into_iter()
            .map(half::f16::from_f32)
            .collect()
    };
    drop(row_raw);

    // 3. Upload to the F16 scratch slot.
    let upload_bytes = hidden * 2;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            out_f16,
            DevicePtr(row_f16.as_ptr() as usize),
            upload_bytes,
        )?;
    }
    stream.synchronize()?;
    drop(row_f16);
    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-e3 — output norm + LM head + argmax sampling.
// ---------------------------------------------------------------------------

/// Workspace for the output / LM head path. Sized against
/// `(hidden, vocab_size)`.
pub struct OutputHeadScratch {
    pub x_norm_f16: DevicePtr,   // [hidden] F16, rmsnorm(output_norm, x_final)
    pub x_q8_1: DevicePtr,       // Q8_1 of x_norm for the LM head mmvq
    pub logits_f32: DevicePtr,   // [vocab] F32
    // Bookkeeping.
    x_norm_bytes: usize,
    x_q8_1_bytes: usize,
    logits_bytes: usize,
    disposed: bool,
}

impl OutputHeadScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");

        let x_norm_bytes = hidden * 2;
        let x_q8_1_bytes = (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let logits_bytes = vocab * 4;

        let x_norm_f16 = device.alloc(x_norm_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let logits_f32 = device.alloc(logits_bytes)?;

        Ok(Self {
            x_norm_f16,
            x_q8_1,
            logits_f32,
            x_norm_bytes,
            x_q8_1_bytes,
            logits_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_norm_f16, self.x_norm_bytes)?;
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.logits_f32, self.logits_bytes)?;
        }
        Ok(())
    }
}

impl Drop for OutputHeadScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "OutputHeadScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Run the post-last-layer tail: output rmsnorm → LM head mmvq → logits F32.
///
/// `lm_head_weight`: either the untied `output.weight` (when present) or
/// the tied `token_embd.weight`. Expected shape (outermost-first):
/// `[vocab, hidden]`.
///
/// On return, `scratch.logits_f32` holds `[vocab]` F32 logits.
#[allow(clippy::too_many_arguments)]
pub fn forward_output_head_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    output_norm: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    scratch: &mut OutputHeadScratch,
    x_final: DevicePtr,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // 1. Final rmsnorm + Q8_1 quantise in one fused launch.
    rmsnorm_quant_q8_1(
        ops,
        stream,
        x_final,
        output_norm.ptr,
        scratch.x_q8_1,
        1,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("output_norm + quant")?;

    // 2. LM head mmvq → F32 logits.
    let dtype = qdtype_of(lm_head_weight.dtype)?;
    let (rows, k) = mat_shape(lm_head_weight)?;
    if rows != vocab || k != hidden {
        bail!(
            "lm_head weight shape [{rows}, {k}] != expected [{vocab}, {hidden}]"
        );
    }
    mmvq(
        ops,
        stream,
        lm_head_weight.ptr,
        scratch.x_q8_1,
        scratch.logits_f32,
        rows,
        k,
        dtype,
    )
    .context("lm_head mmvq")?;

    Ok(())
}

/// Host-side argmax sampler over the F32 logits produced by
/// [`forward_output_head_decode`]. Downloads `[vocab]` F32s to host and
/// scans for the maximum.
///
/// V1 intentionally keeps sampling CPU-side: vocab × 4 bytes is tiny
/// (Qwen3.6: 248320 × 4 ≈ 970 KB — a single PCIe memcpy + ~1 ms argmax).
/// Temperature / top-p sampling is V2.
pub fn argmax_token_host(
    device: &HipDevice,
    stream: &HipStream,
    logits: DevicePtr,
    vocab: usize,
) -> Result<u32> {
    let mut host = vec![0.0f32; vocab];
    // SAFETY: `logits` points to at least `vocab * 4` valid device bytes
    // (caller's contract — OutputHeadScratch sizes it against cfg.vocab_size).
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            logits,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    let mut best_idx = 0usize;
    let mut best_val = host[0];
    for (i, &v) in host.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    // V1.7.4.a diagnostic: set FLAMBEAU_PARITY_TOPK_LOGITS to dump the
    // top-20 argmax + rank of llama.cpp's top-7 baseline tokens. Lets us
    // tell "F16 noise, llama's #1 is in our top 20" from "systematic bug,
    // llama's #1 is rank 200k+". Left in because the parity gap isn't
    // closed yet and the harness is cheap.
    if std::env::var("FLAMBEAU_PARITY_TOPK_LOGITS").is_ok() {
        let mut idxs: Vec<usize> = (0..host.len()).collect();
        idxs.sort_by(|&a, &b| host[b].partial_cmp(&host[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = idxs.iter().take(20).map(|&i| (i, host[i])).collect();
        eprintln!("[argmax-topk] top-20 = {top:?}");
        // Show where llama.cpp's top-7 ids at pos-0 land in our ranking.
        // Reference generated via `llama-server` POST /completion on the
        // same GGUF with `return_tokens: true, n_probs: 10`.
        let lcpp_top = [11u32, 4858, 0, 1017, 660, 13, 353];
        for tok in lcpp_top {
            let rank = idxs.iter().position(|&i| i == tok as usize).unwrap_or(usize::MAX);
            eprintln!(
                "[argmax-topk] llama.cpp token {tok} → our logit={} rank={}",
                host[tok as usize], rank
            );
        }
    }
    Ok(best_idx as u32)
}

// ---------------------------------------------------------------------------
// V1.7.3-e4 — per-layer composition (residual sums + ffn/post-attn norm +
// routed/shared fan-in).
// ---------------------------------------------------------------------------

/// Every scratch the per-layer composition function touches. Owning them
/// in one struct keeps the `forward_layer_decode` signature readable and
/// lets the session allocate exactly once.
///
/// We always carry a `FullAttnScratch` and a `GdnScratch` even though
/// each layer uses only one; the unused one sits idle. The `SharedExpertScratch`
/// is `Option` because dense qwen3moe arches have no shared expert.
pub struct LayerForwardScratch {
    pub full_attn: Option<FullAttnScratch>,
    pub gdn: Option<GdnScratch>,
    pub moe: Option<MoeScratch>,
    pub shared: Option<SharedExpertScratch>,
    /// Present iff `cfg.is_dense_ffn()` (arch=qwen35); replaces the moe/shared
    /// scratches on that path.
    pub dense_ffn: Option<DenseFfnScratch>,
    /// F16 `[hidden]` — holds the post-attention residual (`x_in + attn_delta`).
    pub mid_f16: DevicePtr,
    /// F16 `[hidden]` — holds rmsnorm(mid, post_attn_norm_or_ffn_norm).
    pub mid_norm_f16: DevicePtr,
    /// F16 `[hidden]` — holds the shared-expert delta (only populated on
    /// hybrid layers with a shared expert).
    pub shared_delta_f16: DevicePtr,
    /// F16 `[hidden]` — holds `mid + shared_delta` fed into moe_combine
    /// as the residual.
    pub moe_residual_f16: DevicePtr,
    hidden_bytes: usize,
    disposed: bool,
}

impl LayerForwardScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let hidden_bytes = hidden * 2;

        let full_attn = Some(FullAttnScratch::new(cfg, device)?);
        let gdn = Some(GdnScratch::new(cfg, device)?);
        // Dense-FFN arches (qwen35) skip the MoE router + shared expert
        // scratch entirely. Allocate dense scratch in its place.
        let (moe, shared, dense_ffn) = if cfg.is_dense_ffn() {
            (None, None, Some(DenseFfnScratch::new(cfg, device)?))
        } else {
            let moe = Some(MoeScratch::new(cfg, device)?);
            let shared = if cfg.shared_expert_intermediate_size.is_some() {
                Some(SharedExpertScratch::new(cfg, device)?)
            } else {
                None
            };
            (moe, shared, None)
        };

        let mid_f16 = device.alloc(hidden_bytes)?;
        let mid_norm_f16 = device.alloc(hidden_bytes)?;
        let shared_delta_f16 = device.alloc(hidden_bytes)?;
        let moe_residual_f16 = device.alloc(hidden_bytes)?;

        Ok(Self {
            full_attn,
            gdn,
            moe,
            shared,
            dense_ffn,
            mid_f16,
            mid_norm_f16,
            shared_delta_f16,
            moe_residual_f16,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.mid_f16, self.hidden_bytes)?;
            device.dealloc(self.mid_norm_f16, self.hidden_bytes)?;
            device.dealloc(self.shared_delta_f16, self.hidden_bytes)?;
            device.dealloc(self.moe_residual_f16, self.hidden_bytes)?;
        }
        if let Some(s) = self.full_attn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.moe.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.shared.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.dense_ffn.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for LayerForwardScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "LayerForwardScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One decode step of a full layer block — dispatches on `cfg.is_recurrent(il)`
/// to the GDN or full-attention path, then runs the MoE FFN (with optional
/// shared expert) and composes all the residual sums.
///
/// Math, per layer:
///   attn_delta = attn(attn_norm(x_in))                 (forward_{full_attn,gdn}_decode)
///   mid        = x_in + attn_delta                     (add_f16)
///   mid_norm   = post_attn_norm(mid)                   (rmsnorm_f16)
///   shared     = shared_expert(mid_norm)               (optional; forward_shared_expert_decode)
///   moe_res    = mid + shared                          (add_f16, if shared)
///   x_out      = moe_res + Σ w_k · expert_k(mid_norm)  (forward_moe_ffn_decode)
///
/// On arches without a shared expert (dense qwen3moe), the `shared` and
/// `moe_res` steps are skipped and `moe_res = mid` is passed directly.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_decode(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut LayerForwardScratch,
    x_in: DevicePtr,
    x_out: DevicePtr,
    position: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let il = layer_weights.layer_idx;

    // 1. Attention (full-attn or GDN) → attn_delta in `mid_f16`.
    // We reuse mid_f16 as the delta slot first, then overwrite it with the
    // post-attention residual sum on the next line.
    if cfg.is_recurrent(il) {
        let gdn = scratch
            .gdn
            .as_mut()
            .context("LayerForwardScratch.gdn missing")?;
        forward_gdn_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            gdn,
            x_in,
            scratch.mid_f16,
        )?;
    } else {
        let full_attn = scratch
            .full_attn
            .as_mut()
            .context("LayerForwardScratch.full_attn missing")?;
        forward_full_attn_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            full_attn,
            x_in,
            scratch.mid_f16,
            position,
        )?;
    }

    // 2. Residual: mid = x_in + attn_delta (in-place on mid_f16).
    add_f16(ops, stream, x_in, scratch.mid_f16, scratch.mid_f16, hidden)
        .context("layer residual: x_in + attn_delta")?;

    // 3. post-attention / ffn norm → mid_norm.
    let post_norm = layer_weights
        .post_attention_norm
        .as_ref()
        .or(layer_weights.ffn_norm.as_ref())
        .context("layer missing both post_attention_norm and ffn_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.mid_f16,
        post_norm.ptr,
        scratch.mid_norm_f16,
        1,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("post-attn rmsnorm")?;

    // 4. FFN. Two flavours:
    //    - arch=qwen35 (dense): single gate/up/down triple, no router. Writes
    //      `x_out = mid + FFN(mid_norm)` directly.
    //    - MoE arches: optional shared expert delta + router + routed MoE
    //      (residual folded into moe_combine).
    if cfg.is_dense_ffn() {
        let dense_w = layer_weights
            .ffn
            .dense
            .as_ref()
            .context("dense FFN forward: layer.ffn.dense missing")?;
        let dense_scratch = scratch
            .dense_ffn
            .as_mut()
            .context("LayerForwardScratch.dense_ffn missing")?;
        forward_dense_ffn_decode(
            ops,
            stream,
            cfg,
            dense_w,
            dense_scratch,
            scratch.mid_norm_f16,
            scratch.mid_f16,
            x_out,
        )?;
        return Ok(());
    }

    // MoE path — optional shared expert delta.
    let moe_residual = if let (Some(shared_w), Some(shared_scratch)) =
        (layer_weights.ffn.shared.as_ref(), scratch.shared.as_mut())
    {
        forward_shared_expert_decode(
            ops,
            stream,
            cfg,
            shared_w,
            shared_scratch,
            scratch.mid_norm_f16,
            scratch.shared_delta_f16,
        )?;
        // moe_res = mid + shared_delta (fused into moe_combine's residual below).
        add_f16(
            ops,
            stream,
            scratch.mid_f16,
            scratch.shared_delta_f16,
            scratch.moe_residual_f16,
            hidden,
        )
        .context("moe residual: mid + shared_delta")?;
        scratch.moe_residual_f16
    } else {
        // Dense arch with no shared expert: moe_combine's residual is just mid.
        scratch.mid_f16
    };

    // 5. Router (dense F32 GEMV + topk).
    let moe = scratch
        .moe
        .as_mut()
        .context("LayerForwardScratch.moe missing")?;
    forward_router_decode(
        ops,
        stream,
        cfg,
        layer_weights.ffn.ffn_gate_inp.as_ref().expect("MoE forward: ffn_gate_inp missing"),
        moe,
        scratch.mid_norm_f16,
    )?;

    // 6. Routed MoE FFN — fuses the residual add in moe_combine.
    forward_moe_ffn_decode(
        ops,
        stream,
        cfg,
        &layer_weights.ffn,
        moe,
        scratch.mid_norm_f16,
        moe_residual,
        x_out,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-e5 — forward_one_token end-to-end.
// ---------------------------------------------------------------------------

/// Complete per-token scratch: two hidden-state buffers (ping/pong for
/// the per-layer loop) + the per-layer composition scratches + the output
/// head scratch. Allocates all once per session.
pub struct ForwardOneTokenScratch {
    pub hidden_a: DevicePtr,    // F16 [hidden]
    pub hidden_b: DevicePtr,    // F16 [hidden]
    pub layer: Option<LayerForwardScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl ForwardOneTokenScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden_bytes = cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = Some(LayerForwardScratch::new(cfg, device)?);
        let output_head = Some(OutputHeadScratch::new(cfg, device)?);
        Ok(Self {
            hidden_a,
            hidden_b,
            layer,
            output_head,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for ForwardOneTokenScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "ForwardOneTokenScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// End-to-end single-token decode. Reads the weights + layout from the
/// model, updates the per-sequence session (KV caches, GDN state, conv
/// history), and returns the argmax-sampled next token id.
///
/// Flow:
/// 1. Embed `token_id` → hidden_a [hidden] F16.
/// 2. For il in 0..num_layers: `forward_layer_decode(il, hidden_{a,b}, ...)` then swap.
/// 3. `forward_output_head_decode(hidden, output_norm, lm_head)` → logits.
/// 4. `argmax_token_host(logits)` → next token id.
///
/// `lm_head`: if `cfg.tied_lm_head` is true, pass `&weights.token_embd` —
/// otherwise the untied `&weights.output`. Wiring this pick is the
/// model-struct's job; we keep the forward path dtype-agnostic.
pub fn forward_one_token(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    weights: &crate::weights::ModelWeights,
    session: &mut crate::session::Qwen3MoESession,
    scratch: &mut ForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let hidden = cfg.hidden_size;

    // 1. Gather the input embedding row. Writes hidden_a.
    forward_embed_decode_host(
        device,
        stream,
        &weights.token_embd,
        token_id,
        scratch.hidden_a,
        hidden,
    )?;

    // 2. Per-layer loop. Ping-pong between hidden_a and hidden_b so each
    // layer reads the previous output without a stream-stalling copy.
    let layer_scratch = scratch
        .layer
        .as_mut()
        .context("ForwardOneTokenScratch.layer missing")?;
    let (mut x_in, mut x_out) = (scratch.hidden_a, scratch.hidden_b);
    for (il, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut session.layers_mut()[il];
        forward_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            position,
        )?;
        std::mem::swap(&mut x_in, &mut x_out);
    }
    // After the swap in the last iter, `x_in` holds the final output.
    let x_final = x_in;

    // 3. Output head: rmsnorm + LM head → logits.
    let lm_head = weights
        .output
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let output_head_scratch = scratch
        .output_head
        .as_mut()
        .context("ForwardOneTokenScratch.output_head missing")?;
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &weights.output_norm,
        lm_head,
        output_head_scratch,
        x_final,
    )?;

    // 4. Sample.
    argmax_token_host(device, stream, output_head_scratch.logits_f32, cfg.vocab_size)
}

// ---------------------------------------------------------------------------
// V1.7.5.C — pipeline-parallel forward_one_token.
// ---------------------------------------------------------------------------

/// Per-rank scratch for a pipeline-parallel single-token decode. Only the
/// last rank in the cluster owns an `OutputHeadScratch` (middle ranks
/// never run the LM head).
pub struct RankForwardScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerForwardScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardScratch {
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch: one `RankForwardScratch` per rank in the cluster.
pub struct ShardedForwardOneTokenScratch {
    pub per_rank: Vec<RankForwardScratch>,
}

impl ShardedForwardOneTokenScratch {
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<Self> {
        let hidden_bytes = model.config.hidden_size * 2;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerForwardScratch::new(&model.config, device)?);
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                hidden_a,
                hidden_b,
                layer,
                output_head,
                hidden_bytes,
                disposed: false,
            });
        }
        Ok(Self { per_rank })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel single-token decode across an N-rank cluster.
///
/// Flow:
///   1. Rank 0 gathers the input embedding into its `hidden_a`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` pulls the previous rank's
///          final hidden (in `hidden_a` by convention — see step 3)
///          into this rank's `hidden_a`.
///        - Run `forward_layer_decode` over the layers the shard owns,
///          ping-ponging `hidden_a ↔ hidden_b`.
///        - Normalise the final hidden back into `hidden_a` so the
///          next peer-copy has a known source.
///   3. Last rank runs `forward_output_head_decode` + `argmax_token_host`.
///
/// Single-token in flight — no micro-batching (the bubble is the sum of
/// each rank's compute; V2 can add 2-micro-batch pipelining). Per-hop
/// cost ≈ 30 µs (V1.7.5.B measurement), 3 hops for N=4 ≈ 0.5% of the
/// 16.7 ms/token budget at 60 tok/s.
#[allow(clippy::too_many_arguments)]
pub fn forward_one_token_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_one_token_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let hidden_bytes = hidden * 2;

    // 1. Embed on rank 0. Bind first so the embed memcpys land on the
    // right device.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        forward_embed_decode_host(
            rank0,
            rank0.default_stream(),
            token_embd,
            token_id,
            scratch0.hidden_a,
            hidden,
        )?;
    }

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are hidden_bytes long on their
            // respective devices; no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        // Bind this rank's device before issuing any kernels through its
        // OpsRegistry — HIP's module-launched kernels use the thread's
        // current device context, not whichever device the module was
        // loaded on.
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_decode(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                position,
            )
            .with_context(|| {
                format!(
                    "rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            // V1.7.4.b per-layer activation dump for A/B vs llama.cpp.
            // Env-gated so the hot path pays zero cost when unset. Pair
            // with `llama-eval-callback` + grep `l_out-<il>` to bisect a
            // future forward divergence.
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() && position == 0 {
                let mut buf = vec![half::f16::from_f32(0.0); hidden];
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToHost,
                        DevicePtr(buf.as_mut_ptr() as usize),
                        x_out,
                        hidden * 2,
                    )?;
                }
                device.default_stream().synchronize()?;
                let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
                let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                let (mn, mx) = vals
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                eprintln!(
                    "[layer-dump] l_out-{} ({}): L2={:.6} min={} max={} head={:?}",
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                    l2, mn, mx, &vals[..4]
                );
            }
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next hand-off.
        // NO sync: subsequent peer_copy_via_host + downstream kernels all run on
        // the same default_stream, so stream ordering guarantees correctness.
        // Removing this sync saves one ~200 µs CPU-wait per rank per forward.
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    hidden_bytes,
                )?;
            }
        }
    }

    // 3. Output head on the last rank.
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_scratch.hidden_a,
    )?;

    // 4. Host argmax.
    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

// ---------------------------------------------------------------------------
// V1.7.5.D — pipeline-parallel forward_prefill.
// ---------------------------------------------------------------------------

/// Per-rank scratch for a pipeline-parallel prefill chunk of up to
/// `max_tokens` tokens. Layout mirrors `RankForwardScratch` with the
/// hidden ping-pong buffers and `LayerPrefillScratch` both sized for `L`
/// tokens. Only the last rank owns an `OutputHeadScratch`.
pub struct RankForwardPrefillScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratch {
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardPrefillScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch for PP prefill: one `RankForwardPrefillScratch` per rank.
pub struct ShardedForwardPrefillScratch {
    pub per_rank: Vec<RankForwardPrefillScratch>,
}

impl ShardedForwardPrefillScratch {
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden_bytes = max_tokens * model.config.hidden_size * 2;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerPrefillScratch::new(&model.config, device, max_tokens)?);
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardPrefillScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens,
                hidden_a,
                hidden_b,
                layer,
                output_head,
                hidden_bytes,
                disposed: false,
            });
        }
        Ok(Self { per_rank })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel prefill over `L` tokens across an N-rank cluster.
///
/// Same per-rank flow as [`forward_one_token_pp`] but each stage processes
/// `L` tokens at once:
///   1. Rank 0 gathers `L` embeddings (one per token) into `hidden_a[0..L*H]`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` moves `L * hidden * 2` bytes of
///          F16 hidden state from r-1's `hidden_a` to r's `hidden_a`.
///        - Bind this rank's device.
///        - Ping-pong through the rank's local layers via `forward_layer_prefill`.
///          KV cache / GDN state / conv-history get `L` tokens of history
///          appended per layer.
///        - If the final swap left the output in `hidden_b`, copy back
///          into `hidden_a` so the next rank's peer-copy has a known source.
///   3. Last rank: output head on the LAST token's hidden row + argmax.
///
/// No microbatching / 1F1B schedule — the pipeline bubble is the sum of
/// each rank's prefill compute. For Qwen3.6-31B at L=512 across 4 ranks,
/// measured prefill is compute-bound enough that microbatching is a V2+
/// perf lever, not a V1 correctness blocker.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp called with empty tokens");
    }
    let max_tokens = scratch.per_rank[0].max_tokens;
    if l > max_tokens {
        bail!(
            "forward_prefill_pp: L={l} > scratch.max_tokens={max_tokens}; caller must chunk"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    // 1. Embed all L tokens on rank 0. Row-by-row host dequant + upload —
    //    matches single-device `forward_prefill`'s embedding path.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }
    }

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are at least `chunk_bytes` long
            // on their respective devices (sized against scratch.max_tokens
            // ≥ L); no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_prefill(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                start_position,
            )
            .with_context(|| {
                format!(
                    "prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() {
                // Dump each token's post-layer hidden for A/B vs llama.cpp's
                // per-layer prefill output (`l_out-<il>` covers the full
                // [L, hidden] tensor in the callback trace).
                for t in 0..l {
                    let mut buf = vec![half::f16::from_f32(0.0); hidden];
                    unsafe {
                        device.memcpy_async(
                            device.default_stream(),
                            CopyDirection::DeviceToHost,
                            DevicePtr(buf.as_mut_ptr() as usize),
                            x_out.offset_bytes(t * row_bytes),
                            hidden * 2,
                        )?;
                    }
                    device.default_stream().synchronize()?;
                    let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
                    let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                    eprintln!(
                        "[prefill-dump] l_out-{} t={} ({}): L2={:.6} head={:?}",
                        layer_weights.layer_idx,
                        t,
                        if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                        l2, &vals[..4]
                    );
                }
            }
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next peer-copy hop
        // (L rows of F16 hidden).
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
    }

    // 3. Output head on the LAST token row on the last rank.
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

// ---------------------------------------------------------------------------
// V1.7.3-f1 — full-attention prefill (L > 1).
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of a full-attention layer. Sized once
/// against `(cfg, max_prefill_tokens)` — the caller chunks long prompts
/// to keep scratch VRAM bounded (V1.7.3-f4 decides the chunk size).
///
/// The buffers scale linearly with `max_prefill_tokens` except `x_q8_1`
/// (which scales in blocks of 32 inputs). At hidden=2048 and L=128:
/// activations + scratch < 10 MB total — comfortable even on 16 GB cards.
pub struct FullAttnPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,      // F16 [max_L, hidden] — rmsnorm output buffer
                                    //                      (V2.2.d.P8: split away from the
                                    //                       D1 fused rmsnorm+quant path so we
                                    //                       can emit both Q8_1 layouts.)
    pub x_q8_1: DevicePtr,          // Q8_1 blocks [max_L, hidden/32]
    pub x_q8_1_mmq: DevicePtr,      // BlockQ8_1Mmq [hidden/128, max_L] — DS4 layout for MmqLdsX64
    pub mmvq_f32: DevicePtr,        // F32 [max_L, max(2*H*D, H_kv*D, hidden)]
    pub q_fused_f16: DevicePtr,     // F16 [max_L, 2*n_heads*head_dim]
    pub q_f16: DevicePtr,           // F16 [max_L, n_heads*head_dim]
    pub gate_f16: DevicePtr,        // F16 [max_L, n_heads*head_dim]
    pub k_f16: DevicePtr,           // F16 [max_L, n_kv_heads*head_dim]
    pub v_f16: DevicePtr,           // F16 [max_L, n_kv_heads*head_dim]
    pub attn_out_f16: DevicePtr,    // F16 [max_L, n_heads*head_dim]
    pub gated_out_f16: DevicePtr,   // F16 [max_L, n_heads*head_dim]
    pub positions: DevicePtr,       // i32 [max_L]
    pub gated_q8_1: DevicePtr,      // Q8_1 [max_L, n_heads*head_dim/32]
    pub gated_q8_1_mmq: DevicePtr,  // BlockQ8_1Mmq [q_width/128, max_L] — DS4 layout
    // Bookkeeping.
    x_norm_f16_bytes: usize,
    x_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    mmvq_f32_bytes: usize,
    q_fused_bytes: usize,
    qk_bytes: usize,
    kv_bytes: usize,
    attn_bytes: usize,
    positions_bytes: usize,
    gated_q8_1_bytes: usize,
    gated_q8_1_mmq_bytes: usize,
    disposed: bool,
}

impl FullAttnPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;

        let q_fused_width = 2 * n_heads * head_dim;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv_heads * head_dim;

        assert!(hidden % 32 == 0, "hidden must be a multiple of QK8_1=32");
        assert!(q_width % 32 == 0, "n_heads * head_dim must be multiple of 32");
        assert!(
            hidden % 128 == 0,
            "hidden must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );
        assert!(
            q_width % 128 == 0,
            "n_heads * head_dim must be a multiple of QK8_1_MMQ=128 for the DS4 layout"
        );

        let mmq_block = std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let x_norm_f16_bytes = max_tokens * hidden * 2;
        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let x_q8_1_mmq_bytes = max_tokens * (hidden / 128) * mmq_block;
        let mmvq_f32_bytes = max_tokens * q_fused_width.max(hidden) * 4;
        let q_fused_bytes = max_tokens * q_fused_width * 2;
        let qk_bytes = max_tokens * q_width * 2;
        let kv_bytes = max_tokens * kv_width * 2;
        let attn_bytes = max_tokens * q_width * 2;
        let positions_bytes = max_tokens * 4;
        let gated_q8_1_bytes =
            max_tokens * (q_width / 32) * std::mem::size_of::<BlockQ8_1>();
        let gated_q8_1_mmq_bytes = max_tokens * (q_width / 128) * mmq_block;

        let x_norm_f16 = device.alloc(x_norm_f16_bytes)?;
        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let mmvq_f32 = device.alloc(mmvq_f32_bytes)?;
        let q_fused_f16 = device.alloc(q_fused_bytes)?;
        let q_f16 = device.alloc(qk_bytes)?;
        let gate_f16 = device.alloc(qk_bytes)?;
        let k_f16 = device.alloc(kv_bytes)?;
        let v_f16 = device.alloc(kv_bytes)?;
        let attn_out_f16 = device.alloc(attn_bytes)?;
        let gated_out_f16 = device.alloc(attn_bytes)?;
        let positions = device.alloc(positions_bytes)?;
        let gated_q8_1 = device.alloc(gated_q8_1_bytes)?;
        let gated_q8_1_mmq = device.alloc(gated_q8_1_mmq_bytes)?;

        Ok(Self {
            max_tokens,
            x_norm_f16,
            x_q8_1,
            x_q8_1_mmq,
            mmvq_f32,
            q_fused_f16,
            q_f16,
            gate_f16,
            k_f16,
            v_f16,
            attn_out_f16,
            gated_out_f16,
            positions,
            gated_q8_1,
            gated_q8_1_mmq,
            x_norm_f16_bytes,
            x_q8_1_bytes,
            x_q8_1_mmq_bytes,
            mmvq_f32_bytes,
            q_fused_bytes,
            qk_bytes,
            kv_bytes,
            attn_bytes,
            positions_bytes,
            gated_q8_1_bytes,
            gated_q8_1_mmq_bytes,
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
            device.dealloc(self.mmvq_f32, self.mmvq_f32_bytes)?;
            device.dealloc(self.q_fused_f16, self.q_fused_bytes)?;
            device.dealloc(self.q_f16, self.qk_bytes)?;
            device.dealloc(self.gate_f16, self.qk_bytes)?;
            device.dealloc(self.k_f16, self.kv_bytes)?;
            device.dealloc(self.v_f16, self.kv_bytes)?;
            device.dealloc(self.attn_out_f16, self.attn_bytes)?;
            device.dealloc(self.gated_out_f16, self.attn_bytes)?;
            device.dealloc(self.positions, self.positions_bytes)?;
            device.dealloc(self.gated_q8_1, self.gated_q8_1_bytes)?;
            device.dealloc(self.gated_q8_1_mmq, self.gated_q8_1_mmq_bytes)?;
        }
        Ok(())
    }
}

impl Drop for FullAttnPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "FullAttnPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Upload `L` i32 positions `[start_position, start_position + L)` into the
/// `positions` scratch slot. Matches `rope_neox_partial_f16`'s expectation.
fn upload_positions_range(
    device: &HipDevice,
    stream: &HipStream,
    dst: DevicePtr,
    start_position: usize,
    n: usize,
) -> Result<()> {
    let host: Vec<i32> = (0..n).map(|i| (start_position + i) as i32).collect();
    // SAFETY: `dst` has at least `n * 4` valid bytes; `host` is the same length.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            dst,
            DevicePtr(host.as_ptr() as usize),
            n * 4,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Prefill step for one full-attention layer over `n_tokens = L` inputs.
/// The KV cache is expected to hold `start_position` tokens of history
/// (0 on a fresh sequence); this call appends the L new tokens and
/// computes causal attention for each new Q row against the combined
/// `[history + L]` KV.
///
/// `x_in` layout: F16 `[L, hidden]`, row-major (rows are tokens).
/// `delta_out` layout: F16 `[L, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<flambeau_runtime::F16Contig, HipDevice>,
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_full_attn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_full_attn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let q_width = n_heads * head_dim;
    let rope = &cfg.rope;

    // 1. rmsnorm(x_in) → F16 scratch, then quantise to BOTH Q8_1 layouts.
    //
    // V2.2.d.P8: the older D1 fused rmsnorm+quant_q8_1 kernel writes only
    // the standard per-row layout. To feed the 4-warp LDS-tiled Q4_1 MMQ
    // kernel at M ≥ 128 we need the DS4 (BlockQ8_1Mmq) layout in parallel.
    // Unfused rmsnorm costs one extra HBM round-trip per token (x_norm_f16
    // buffer, ~n_tokens·hidden·2 B), negligible against attention wall-clock.
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
    .context("prefill attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("prefill attn x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("prefill attn x_norm → Q8_1 (MMQ DS4)")?;

    // 2. Q|gate projection across L rows. `qmatmul` auto-dispatches to
    //    looped MMVQ (mid-M) or MMQ (M ≥ 128) based on the table.
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != 2 * n_heads * head_dim || q_k != hidden {
        bail!(
            "attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            2 * n_heads * head_dim,
            hidden
        );
    }
    qmatmul(
        ops,
        stream,
        weights.attn_q.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        q_k,
        q_rows,
        dtype_q,
    )
    .context("prefill qmatmul attn_q")?;
    cast_f32_to_f16(
        ops,
        stream,
        scratch.mmvq_f32,
        scratch.q_fused_f16,
        n_tokens * q_rows,
    )
    .context("prefill cast attn_q → f16")?;

    // 3. Split Q | gate across L tokens.
    split_q_gate_f16(
        ops,
        stream,
        scratch.q_fused_f16,
        scratch.q_f16,
        scratch.gate_f16,
        n_tokens,
        n_heads,
        head_dim,
    )
    .context("prefill split_q_gate")?;

    // 4. K / V projections.
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    if k_rows != n_kv_heads * head_dim || k_k != hidden {
        bail!(
            "attn_k shape [{k_rows}, {k_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_k.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens, k_k, k_rows, dtype_k,
    )
    .context("prefill qmatmul attn_k")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.k_f16, n_tokens * k_rows,
    )
    .context("prefill cast attn_k → f16")?;

    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if v_rows != n_kv_heads * head_dim || v_k != hidden {
        bail!(
            "attn_v shape [{v_rows}, {v_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_v.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens, v_k, v_rows, dtype_v,
    )
    .context("prefill qmatmul attn_v")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.v_f16, n_tokens * v_rows,
    )
    .context("prefill cast attn_v → f16")?;

    // 5. Per-head Q/K rmsnorm. Flatten the outer dim to L × heads.
    let q_norm_dim = weights
        .attn_q_norm
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
        weights.attn_q_norm.ptr,
        scratch.q_f16,
        n_tokens * n_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill attn_q_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.k_f16,
        weights.attn_k_norm.ptr,
        scratch.k_f16,
        n_tokens * n_kv_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .context("prefill attn_k_norm")?;

    // 6. RoPE on Q / K, with per-token positions.
    upload_positions_range(device, stream, scratch.positions, start_position, n_tokens)?;
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.q_f16,
        scratch.positions,
        rope.freq_base,
        n_tokens,
        n_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("prefill rope Q")?;
    rope_neox_partial_f16(
        ops,
        stream,
        scratch.k_f16,
        scratch.positions,
        rope.freq_base,
        n_tokens,
        n_kv_heads,
        head_dim,
        rope.rotated_dims,
    )
    .context("prefill rope K")?;

    // 7. Append all L tokens to the KV cache.
    // SAFETY: scratch.k_f16 / v_f16 hold `n_tokens * n_kv_heads * head_dim` F16s.
    unsafe {
        kv_cache
            .append(device, stream, scratch.k_f16, scratch.v_f16, n_tokens)
            .map_err(|e| anyhow::anyhow!("kv_cache.append(L={n_tokens}): {e}"))?;
    }

    // 8. Causal prefill attention. `n_k_tokens = start_position + L`
    // (after append); `q_offset = start_position` so Q row i attends to
    // K rows `0..start_position + i + 1`.
    let n_k_tokens = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    attention_prefill_f16(
        ops,
        stream,
        scratch.q_f16,
        kv_cache.k_buffer(),
        kv_cache.v_buffer(),
        scratch.attn_out_f16,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
        n_k_tokens,
        start_position,
        scale,
    )
    .context("attention_prefill_f16")?;

    // 9. Post-attention sigmoid-gate: gated_out = sigmoid(gate) * attn_out,
    // per token. See V1.7.4.b note in the decode path — Qwen3.5/3.6 uses
    // plain sigmoid, not SiLU.
    sigmoid_mul_f16(
        ops,
        stream,
        scratch.gate_f16,
        scratch.attn_out_f16,
        scratch.gated_out_f16,
        n_tokens * q_width,
    )
    .context("prefill post-attn sigmoid-gate")?;

    // 10. Quantise gated_out to BOTH Q8_1 layouts for the output projection.
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.gated_out_f16,
        scratch.gated_q8_1,
        n_tokens * q_width,
    )
    .context("prefill quantise gated → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops,
        stream,
        scratch.gated_out_f16,
        scratch.gated_q8_1_mmq,
        q_width,
        n_tokens,
    )
    .context("prefill quantise gated → Q8_1 (MMQ DS4)")?;

    // 11. Output projection across L tokens.
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != q_width {
        bail!(
            "attn_output shape [{o_rows}, {o_k}] != expected [{}, {}]",
            hidden,
            q_width
        );
    }
    qmatmul(
        ops,
        stream,
        weights.attn_output.ptr,
        scratch.gated_q8_1, scratch.gated_q8_1_mmq,
        scratch.mmvq_f32,
        n_tokens,
        o_k,
        o_rows,
        dtype_o,
    )
    .context("prefill qmatmul attn_output")?;
    cast_f32_to_f16(
        ops,
        stream,
        scratch.mmvq_f32,
        delta_out,
        n_tokens * hidden,
    )
    .context("prefill cast attn_output → f16")?;

    Ok(())
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
fn assemble_conv_input_prefill(
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
fn shift_conv_history_prefill(
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
#[allow(clippy::too_many_arguments)]
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

    // 11. α / β / gate compute — batched across all L tokens in one launch
    // (V2.2.d fix 3). The kernel's grid.x = n_tokens, block = num_v_heads.
    // gate/beta lay out contiguously as [L, num_v_heads]; the per-head
    // constants ssm_dt_bias / ssm_a are shared across the L rows.
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

    // 12. GDN state step — kernel natively handles B=1, H=num_v_heads, L.
    let n_rep = num_v_heads / num_k_heads;
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
    )
    .context("prefill gdn_state_step_f32_s128")?;

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

/// Helper: run a qmatmul against a `DeviceTensor`, validating dims + dispatching on dtype.
///
/// Takes both the standard [`flambeau_quant::BlockQ8_1`] buffer and the DS4
/// [`flambeau_quant::BlockQ8_1Mmq`] buffer. Callers whose weight dtype never
/// routes to the MmqLdsX64 kernel (everything except Q4_1 as of V2.2.d.P8)
/// can legitimately pass [`DevicePtr(0)`] for `act_q8_1_mmq`.
#[allow(clippy::too_many_arguments)]
fn run_qmatmul_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    act_q8_1_mmq: DevicePtr,
    dst: DevicePtr,
    m: usize,
    expected_k: usize,
    expected_rows: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    qmatmul(ops, stream, w.ptr, act_q8_1, act_q8_1_mmq, dst, m, k, rows, dtype)
        .with_context(|| format!("qmatmul {label}"))
}

// ---------------------------------------------------------------------------
// V1.7.3-f3 — MoE + shared expert + router prefill.
// ---------------------------------------------------------------------------

/// Workspace for one prefill chunk of the routed MoE FFN. Sized against
/// `(cfg, max_tokens)`.
pub struct MoePrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub router_logits: DevicePtr,      // F32 [L, n_experts]
    pub expert_ids: DevicePtr,         // i32 [L, top_k]
    pub expert_weights: DevicePtr,     // F32 [L, top_k]
    pub gate_out_f32: DevicePtr,       // F32 [L, top_k, inter]
    pub up_out_f32: DevicePtr,         // F32 [L, top_k, inter]
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    // V2.14.c DS4 Q8_1 activation buffers (turbo MoE variant only).
    // `x_q8_1_mmq`: hidden activation in DS4 layout — [hidden/128, n_tokens].
    // `activated_q8_1_mmq`: per-pair SwiGLU'd activation in DS4 layout — [inter/128, n_pairs].
    pub x_q8_1_mmq: DevicePtr,
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,           // F32 [L, top_k, hidden]
    pub down_f16: DevicePtr,
    // V2.5.a sort-by-expert state. Only populated / used when
    // FLAMBEAU_MOE_SORTED=1 is set on the gate+up path.
    pub sort_counts: DevicePtr,        // i32 [n_experts]
    pub sort_offsets: DevicePtr,       // i32 [n_experts + 1]
    pub sort_cursors: DevicePtr,       // i32 [n_experts]
    pub sort_sorted_pair_idx: DevicePtr, // i32 [L * top_k]
    // V2.6.a padded sort outputs (only touched when tile8 path is on).
    pub sort_padded_offsets: DevicePtr,   // i32 [n_experts + 1]
    pub sort_sorted_pair_idx_padded: DevicePtr, // i32 [max_tokens * top_k + n_experts * 8]
    x_q8_1_bytes: usize,
    router_logits_bytes: usize,
    expert_ids_bytes: usize,
    expert_weights_bytes: usize,
    gate_up_bytes: usize,
    activated_f32_bytes: usize,
    activated_f16_bytes: usize,
    activated_q8_1_bytes: usize,
    x_q8_1_mmq_bytes: usize,
    activated_q8_1_mmq_bytes: usize,
    down_f32_bytes: usize,
    down_f16_bytes: usize,
    sort_counts_bytes: usize,
    sort_offsets_bytes: usize,
    sort_cursors_bytes: usize,
    sort_sorted_pair_idx_bytes: usize,
    sort_padded_offsets_bytes: usize,
    sort_sorted_pair_idx_padded_bytes: usize,
    disposed: bool,
}

impl MoePrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        let n_experts = cfg.num_experts;
        assert!(hidden % 32 == 0);
        assert!(inter % 32 == 0);

        let x_q8_1_bytes =
            max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let router_logits_bytes = max_tokens * n_experts * 4;
        let expert_ids_bytes = max_tokens * top_k * 4;
        let expert_weights_bytes = max_tokens * top_k * 4;
        let gate_up_bytes = max_tokens * top_k * inter * 4;
        let activated_f32_bytes = max_tokens * top_k * inter * 4;
        let activated_f16_bytes = max_tokens * top_k * inter * 2;
        let activated_q8_1_bytes =
            max_tokens * top_k * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        // V2.14.c DS4 activation buffers. 144 bytes per MMQ block (128 elements).
        // hidden/128 big_blocks × max_tokens rows for gate+up (per-token);
        // inter/128 big_blocks × max_tokens*top_k rows for down (per-pair).
        let x_q8_1_mmq_bytes =
            max_tokens * (hidden / 128) * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let activated_q8_1_mmq_bytes =
            max_tokens * top_k * (inter / 128) * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let down_f32_bytes = max_tokens * top_k * hidden * 4;
        let down_f16_bytes = max_tokens * top_k * hidden * 2;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let router_logits = device.alloc(router_logits_bytes)?;
        let expert_ids = device.alloc(expert_ids_bytes)?;
        let expert_weights = device.alloc(expert_weights_bytes)?;
        let gate_out_f32 = device.alloc(gate_up_bytes)?;
        let up_out_f32 = device.alloc(gate_up_bytes)?;
        let activated_f32 = device.alloc(activated_f32_bytes)?;
        let activated_f16 = device.alloc(activated_f16_bytes)?;
        let activated_q8_1 = device.alloc(activated_q8_1_bytes)?;
        let x_q8_1_mmq = device.alloc(x_q8_1_mmq_bytes)?;
        let activated_q8_1_mmq = device.alloc(activated_q8_1_mmq_bytes)?;
        let down_f32 = device.alloc(down_f32_bytes)?;
        let down_f16 = device.alloc(down_f16_bytes)?;

        // V2.5.a sort-by-expert scratch
        let sort_counts_bytes = n_experts * 4;
        let sort_offsets_bytes = (n_experts + 1) * 4;
        let sort_cursors_bytes = n_experts * 4;
        let sort_sorted_pair_idx_bytes = max_tokens * top_k * 4;
        let sort_counts = device.alloc(sort_counts_bytes)?;
        let sort_offsets = device.alloc(sort_offsets_bytes)?;
        let sort_cursors = device.alloc(sort_cursors_bytes)?;
        let sort_sorted_pair_idx = device.alloc(sort_sorted_pair_idx_bytes)?;
        // V2.6.a padded sort outputs. Upper bound on padded total: the
        // real total plus up to 7 padding entries per expert.
        let sort_padded_offsets_bytes = (n_experts + 1) * 4;
        let sort_sorted_pair_idx_padded_bytes =
            (max_tokens * top_k + n_experts * 8) * 4;
        let sort_padded_offsets = device.alloc(sort_padded_offsets_bytes)?;
        let sort_sorted_pair_idx_padded = device.alloc(sort_sorted_pair_idx_padded_bytes)?;

        Ok(Self {
            max_tokens,
            x_q8_1,
            router_logits,
            expert_ids,
            expert_weights,
            gate_out_f32,
            up_out_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            x_q8_1_mmq,
            activated_q8_1_mmq,
            down_f32,
            down_f16,
            sort_counts,
            sort_offsets,
            sort_cursors,
            sort_sorted_pair_idx,
            sort_padded_offsets,
            sort_sorted_pair_idx_padded,
            x_q8_1_bytes,
            router_logits_bytes,
            expert_ids_bytes,
            expert_weights_bytes,
            gate_up_bytes,
            activated_f32_bytes,
            activated_f16_bytes,
            activated_q8_1_bytes,
            x_q8_1_mmq_bytes,
            activated_q8_1_mmq_bytes,
            down_f32_bytes,
            down_f16_bytes,
            sort_counts_bytes,
            sort_offsets_bytes,
            sort_cursors_bytes,
            sort_sorted_pair_idx_bytes,
            sort_padded_offsets_bytes,
            sort_sorted_pair_idx_padded_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.router_logits, self.router_logits_bytes)?;
            device.dealloc(self.expert_ids, self.expert_ids_bytes)?;
            device.dealloc(self.expert_weights, self.expert_weights_bytes)?;
            device.dealloc(self.gate_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.up_out_f32, self.gate_up_bytes)?;
            device.dealloc(self.activated_f32, self.activated_f32_bytes)?;
            device.dealloc(self.activated_f16, self.activated_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.activated_q8_1_bytes)?;
            device.dealloc(self.x_q8_1_mmq, self.x_q8_1_mmq_bytes)?;
            device.dealloc(self.activated_q8_1_mmq, self.activated_q8_1_mmq_bytes)?;
            device.dealloc(self.down_f32, self.down_f32_bytes)?;
            device.dealloc(self.down_f16, self.down_f16_bytes)?;
            device.dealloc(self.sort_counts, self.sort_counts_bytes)?;
            device.dealloc(self.sort_offsets, self.sort_offsets_bytes)?;
            device.dealloc(self.sort_cursors, self.sort_cursors_bytes)?;
            device.dealloc(self.sort_sorted_pair_idx, self.sort_sorted_pair_idx_bytes)?;
            device.dealloc(self.sort_padded_offsets, self.sort_padded_offsets_bytes)?;
            device.dealloc(self.sort_sorted_pair_idx_padded, self.sort_sorted_pair_idx_padded_bytes)?;
        }
        Ok(())
    }
}

impl Drop for MoePrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "MoePrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Run the MoE router for a prefill chunk. Produces L × top_k expert ids and
/// softmaxed weights. `dense_gemv_f32_f16` is currently 1-row; we loop L
/// times (launch overhead ≈ L µs, negligible at typical chunk sizes).
/// V2 fusion candidate: a true M-dimension variant.
pub fn forward_router_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_inp: &DeviceTensor,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let n_experts = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;

    if ffn_gate_inp.dtype != GgmlDType::F32 {
        bail!("router expects F32 ffn_gate_inp; got {:?}", ffn_gate_inp.dtype);
    }
    if ffn_gate_inp.dims.len() != 2
        || ffn_gate_inp.dims[0] as usize != n_experts
        || ffn_gate_inp.dims[1] as usize != hidden
    {
        bail!(
            "ffn_gate_inp shape {:?} != expected [{n_experts}, {hidden}]",
            ffn_gate_inp.dims
        );
    }

    let x_row_bytes = hidden * 2;
    let logits_row_bytes = n_experts * 4;
    for t in 0..n_tokens {
        dense_gemv_f32_f16(
            ops,
            stream,
            ffn_gate_inp.ptr,
            x_norm.offset_bytes(t * x_row_bytes),
            scratch.router_logits.offset_bytes(t * logits_row_bytes),
            n_experts,
            hidden,
        )
        .with_context(|| format!("prefill router dense_gemv token {t}"))?;
    }

    topk_f32(
        ops,
        stream,
        scratch.router_logits,
        scratch.expert_ids,
        scratch.expert_weights,
        n_tokens,
        n_experts,
        top_k,
    )
    .context("prefill router topk_f32")?;

    Ok(())
}

/// Routed MoE FFN prefill. Mirrors `forward_moe_ffn_decode` but parametrised
/// by `n_tokens`; every indexed-MoE op already takes an `n_tokens` arg.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_ffn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn: &crate::weights::FfnWeights,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    residual: DevicePtr,
    out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_moe_ffn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_moe_ffn_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let ffn_gate_exps = ffn.ffn_gate_exps.as_ref().expect("MoE prefill: ffn_gate_exps missing");
    let ffn_up_exps = ffn.ffn_up_exps.as_ref().expect("MoE prefill: ffn_up_exps missing");
    let ffn_down_exps = ffn.ffn_down_exps.as_ref().expect("MoE prefill: ffn_down_exps missing");

    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let top_k = cfg.num_experts_per_tok;
    let n_experts = cfg.num_experts;

    const QK_K: usize = 256;
    if hidden % QK_K != 0 {
        bail!("MoE expects hidden={hidden} divisible by QK_K={QK_K}");
    }
    if inter % QK_K != 0 {
        bail!("MoE expects moe_intermediate_size={inter} divisible by QK_K={QK_K}");
    }
    let nb_per_row_hidden = hidden / QK_K;
    let nb_per_row_inter = inter / QK_K;

    let gate_dt_pre = ffn_gate_exps.dtype;
    let up_dt_pre = ffn_up_exps.dtype;
    let down_dt_pre = ffn_down_exps.dtype;
    if !(gate_dt_pre == up_dt_pre
        && (gate_dt_pre == GgmlDType::Q4K
            || gate_dt_pre == GgmlDType::Q8_0
            || gate_dt_pre == GgmlDType::Q4_0))
    {
        bail!(
            "indexed-MoE prefill gate/up dtypes must match and be Q4_K, Q8_0 or Q4_0; got gate={:?}, up={:?}",
            gate_dt_pre, up_dt_pre
        );
    }
    if down_dt_pre != GgmlDType::Q4K
        && down_dt_pre != GgmlDType::Q6K
        && down_dt_pre != GgmlDType::Q8_0
        && down_dt_pre != GgmlDType::Q4_0
    {
        bail!(
            "indexed-MoE prefill ffn_down_exps must be Q4_K, Q6_K, Q8_0 or Q4_0; got {:?}",
            down_dt_pre
        );
    }

    // 1. Quantise x_norm [L, hidden] → Q8_1.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("prefill moe x_norm → Q8_1")?;

    // V2.22.a / V2.23.a — Q8_0 / Q4_0 fast path. Skip sort/pad + MMQ tile8
    // (not ported yet); use plain indexed MoE MMVQ with n_tokens > 1. Slower
    // than tile8 at prefill but structurally correct — unblocks UD-Q8_K_XL
    // and Qwen3.6-35B-A3B-Q4_0 load-and-run.
    if gate_dt_pre == GgmlDType::Q4_0 {
        let nb_hidden_q4 = hidden / 32;
        let nb_inter_q4 = inter / 32;
        indexed_moe_mmvq_q4_0(
            ops, stream,
            ffn_gate_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
            scratch.gate_out_f32,
            inter, n_tokens, top_k, nb_hidden_q4,
        ).context("prefill indexed_moe gate q4_0")?;
        indexed_moe_mmvq_q4_0(
            ops, stream,
            ffn_up_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
            scratch.up_out_f32,
            inter, n_tokens, top_k, nb_hidden_q4,
        ).context("prefill indexed_moe up q4_0")?;
        swiglu_f32(
            ops, stream,
            scratch.gate_out_f32, scratch.up_out_f32, scratch.activated_f32,
            n_tokens * top_k * inter,
        ).context("prefill moe swiglu_f32 (q4_0 path)")?;
        cast_f32_to_f16(
            ops, stream, scratch.activated_f32, scratch.activated_f16,
            n_tokens * top_k * inter,
        ).context("prefill cast activated → f16 (q4_0 path)")?;
        quantize_f16_q8_1(
            ops, stream, scratch.activated_f16, scratch.activated_q8_1,
            n_tokens * top_k * inter,
        ).context("prefill quantise activated → Q8_1 (q4_0 path)")?;
        match down_dt_pre {
            GgmlDType::Q4_0 => indexed_moe_mmvq_q4_0(
                ops, stream,
                ffn_down_exps.ptr, scratch.activated_q8_1, scratch.expert_ids,
                scratch.down_f32,
                hidden, n_tokens * top_k, 1, nb_inter_q4,
            ).context("prefill indexed_moe down q4_0")?,
            GgmlDType::Q8_0 => indexed_moe_mmvq_q8_0(
                ops, stream,
                ffn_down_exps.ptr, scratch.activated_q8_1, scratch.expert_ids,
                scratch.down_f32,
                hidden, n_tokens * top_k, 1, nb_inter_q4,
            ).context("prefill indexed_moe down q8_0 (mixed q4_0 gate+up)")?,
            _ => bail!(
                "V2.23.a Q4_0 MoE prefill path requires Q4_0 or Q8_0 down_exps; got {:?}",
                down_dt_pre
            ),
        }
        cast_f32_to_f16(
            ops, stream, scratch.down_f32, scratch.down_f16,
            n_tokens * top_k * hidden,
        ).context("prefill cast down → f16 (q4_0 path)")?;
        moe_combine_f16(
            ops, stream,
            scratch.down_f16, scratch.expert_weights, residual, out,
            n_tokens, top_k, hidden,
        ).context("prefill moe_combine (q4_0 path)")?;
        return Ok(());
    }

    // V2.22.a — Q8_0 fast path. Skip sort/pad + MMQ tile8 (not ported yet);
    // use plain indexed MoE MMVQ with n_tokens > 1. Slower than tile8 at
    // prefill (no per-expert batching) but structurally correct and enough
    // to unblock UD-Q8_K_XL load-and-run. V2.22.b will add MMQ tile8 for
    // Q8_0 to recover prefill throughput.
    if gate_dt_pre == GgmlDType::Q8_0 {
        let nb_hidden_q8 = hidden / 32;
        let nb_inter_q8 = inter / 32;
        indexed_moe_mmvq_q8_0(
            ops, stream,
            ffn_gate_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
            scratch.gate_out_f32,
            inter, n_tokens, top_k, nb_hidden_q8,
        ).context("prefill indexed_moe gate q8_0")?;
        indexed_moe_mmvq_q8_0(
            ops, stream,
            ffn_up_exps.ptr, scratch.x_q8_1, scratch.expert_ids,
            scratch.up_out_f32,
            inter, n_tokens, top_k, nb_hidden_q8,
        ).context("prefill indexed_moe up q8_0")?;
        swiglu_f32(
            ops, stream,
            scratch.gate_out_f32, scratch.up_out_f32, scratch.activated_f32,
            n_tokens * top_k * inter,
        ).context("prefill moe swiglu_f32 (q8_0 path)")?;
        cast_f32_to_f16(
            ops, stream, scratch.activated_f32, scratch.activated_f16,
            n_tokens * top_k * inter,
        ).context("prefill cast activated → f16 (q8_0 path)")?;
        quantize_f16_q8_1(
            ops, stream, scratch.activated_f16, scratch.activated_q8_1,
            n_tokens * top_k * inter,
        ).context("prefill quantise activated → Q8_1 (q8_0 path)")?;
        // Down matmul: treat each of (L × top_k) as its own "effective token"
        // with top_k_inner = 1 — same pattern as decode-path down.
        if down_dt_pre != GgmlDType::Q8_0 {
            bail!(
                "V2.22.a Q8_0 MoE prefill path requires Q8_0 ffn_down_exps too; got {:?}",
                down_dt_pre
            );
        }
        indexed_moe_mmvq_q8_0(
            ops, stream,
            ffn_down_exps.ptr, scratch.activated_q8_1, scratch.expert_ids,
            scratch.down_f32,
            hidden, n_tokens * top_k, 1, nb_inter_q8,
        ).context("prefill indexed_moe down q8_0")?;
        cast_f32_to_f16(
            ops, stream, scratch.down_f32, scratch.down_f16,
            n_tokens * top_k * hidden,
        ).context("prefill cast down → f16 (q8_0 path)")?;
        moe_combine_f16(
            ops, stream,
            scratch.down_f16, scratch.expert_weights, residual, out,
            n_tokens, top_k, hidden,
        ).context("prefill moe_combine (q8_0 path)")?;
        return Ok(());
    }

    // 2. Path selection:
    //   tile8  (V2.6.b, default): sort+pad + 64×8-tile MMQ kernel
    //   sorted (V2.5.b): sort + r4 block reorder
    //   none   (V2.4): raw r4
    // FLAMBEAU_MOE_VARIANT in {tile8, sorted, r4}. Default = tile8.
    // FLAMBEAU_MOE_SORTED=0 still works as a shortcut to force r4.
    let moe_variant = std::env::var("FLAMBEAU_MOE_VARIANT")
        .ok()
        .unwrap_or_else(|| {
            if std::env::var("FLAMBEAU_MOE_SORTED").as_deref() == Ok("0") {
                "r4".to_string()
            } else {
                "tile8".to_string()
            }
        });
    let total_pairs = n_tokens * top_k;
    if moe_variant == "turbo" || moe_variant == "tile8" {
        moe_sort_by_expert_padded(
            ops,
            stream,
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            scratch.sort_padded_offsets,
            scratch.sort_sorted_pair_idx_padded,
            total_pairs,
            n_experts,
            scratch.max_tokens,
            top_k,
        )
        .context("prefill moe_sort_by_expert_padded")?;
        // Upper bound on padded_total: real total plus up to 7 padding entries
        // per expert. Kernel early-exits blocks past the actual count.
        let padded_total_ub = total_pairs + n_experts * 8;
    if moe_variant == "turbo" {
        // V2.14.c: DS4 Q8_1 activation for turbo gate_up. Per-TOKEN layout
        // — hidden activation shared across the top_k slots of each token.
        quantize_f16_q8_1_mmq(ops, stream, x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("prefill turbo quantize x_norm → Q8_1_MMQ")?;
        indexed_moe_mmq_q4_k_gate_up_turbo(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1_mmq,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: nb_per_row_hidden,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up turbo")?;
    } else {
        indexed_moe_mmq_q4_k_gate_up_tile8(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx_padded,
            scratch.sort_padded_offsets,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            flambeau_ops::hip::moe::MoeShape {
                n_rows: inter,
                n_tokens,
                top_k,
                n_sb_per_row: nb_per_row_hidden,
                n_experts,
                padded_total_upper_bound: padded_total_ub,
            },
        )
        .context("prefill indexed_moe gate+up tile8")?;
    }
    } else if moe_variant == "sorted" {
        moe_sort_by_expert(
            ops,
            stream,
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            total_pairs,
            n_experts,
        )
        .context("prefill moe_sort_by_expert")?;
        indexed_moe_mmvq_q4_k_gate_up_sorted(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            inter,
            n_tokens,
            top_k,
            nb_per_row_hidden,
        )
        .context("prefill indexed_moe gate+up (sorted)")?;
    } else {
        indexed_moe_mmvq_q4_k_gate_up(
            ops,
            stream,
            ffn_gate_exps.ptr,
            ffn_up_exps.ptr,
            scratch.x_q8_1,
            scratch.expert_ids,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            inter,
            n_tokens,
            top_k,
            nb_per_row_hidden,
        )
        .context("prefill indexed_moe gate+up")?;
    }

    // 3. SwiGLU over [L, top_k, inter] flat.
    swiglu_f32(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f32,
        n_tokens * top_k * inter,
    )
    .context("prefill moe swiglu_f32")?;

    // 4. Cast + Q8_1-quantise activated. Each (token, slot) pair is one
    // "effective token" in the down matmul's input layout.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.activated_f32,
        scratch.activated_f16,
        n_tokens * top_k * inter,
    )
    .context("prefill cast activated → f16")?;
    if moe_variant == "turbo" {
        // V2.14.c turbo path: DS4 Q8_1 activation for down matmul, per-PAIR layout.
        quantize_f16_q8_1_mmq(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1_mmq,
            inter,
            n_tokens * top_k,
        )
        .context("prefill turbo quantise activated → Q8_1_MMQ")?;
    } else {
        quantize_f16_q8_1(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_tokens * top_k * inter,
        )
        .context("prefill quantise activated → Q8_1")?;
    }

    // 5. Down matmul: treat each of L × top_k activations as one
    // "effective token" with top_k_inner = 1 and its own expert id. The
    // scratch's flat `expert_ids` [L, top_k] doubles as the flat lookup
    // [L * top_k] when viewed with stride 1.
    match ffn_down_exps.dtype {
        GgmlDType::Q4K if moe_variant == "turbo" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q4_k_down_turbo(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1_mmq,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q4_k turbo")?;
        }
        GgmlDType::Q4K if moe_variant == "tile8" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q4_k_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q4_k tile8")?;
        }
        GgmlDType::Q4K if moe_variant == "sorted" => indexed_moe_mmvq_q4_k_r2_sorted(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.sort_sorted_pair_idx,
            scratch.down_f32,
            hidden,
            n_tokens * top_k,
            1,
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q4_k r2 sorted")?,
        GgmlDType::Q4K => indexed_moe_mmvq_q4_k_r2(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            n_tokens * top_k, // n_tokens_effective
            1,                // top_k_inner
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q4_k r2")?,
        GgmlDType::Q6K if moe_variant == "tile8" => {
            let padded_total_ub = total_pairs + n_experts * 8;
            indexed_moe_mmq_q6_k_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                flambeau_ops::hip::moe::MoeShape {
                    n_rows: hidden,
                    n_tokens: n_tokens * top_k,
                    top_k: 1,
                    n_sb_per_row: nb_per_row_inter,
                    n_experts,
                    padded_total_upper_bound: padded_total_ub,
                },
            )
            .context("prefill indexed_moe down q6_k tile8")?;
        }
        GgmlDType::Q6K => indexed_moe_mmvq_q6_k(
            ops,
            stream,
            ffn_down_exps.ptr,
            scratch.activated_q8_1,
            scratch.expert_ids,
            scratch.down_f32,
            hidden,
            n_tokens * top_k,
            1,
            nb_per_row_inter,
        )
        .context("prefill indexed_moe down q6_k")?,
        other => bail!("unreachable: ffn_down_exps dtype {other:?} should have been rejected"),
    }

    // 6. Cast expert outputs to F16.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        n_tokens * top_k * hidden,
    )
    .context("prefill cast down → f16")?;

    // 7. Weighted sum + residual. `moe_combine_f16` handles L natively.
    moe_combine_f16(
        ops,
        stream,
        scratch.down_f16,
        scratch.expert_weights,
        residual,
        out,
        n_tokens,
        top_k,
        hidden,
    )
    .context("prefill moe_combine_f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared-expert prefill.
// ---------------------------------------------------------------------------

pub struct SharedExpertPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub x_norm_f32: DevicePtr,
    x_q8_1_bytes: usize,
    inter_f32_bytes: usize,
    inter_f16_bytes: usize,
    inter_q8_1_bytes: usize,
    hidden_f32_bytes: usize,
    disposed: bool,
}

impl SharedExpertPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1);
        let hidden = cfg.hidden_size;
        let inter = cfg
            .shared_expert_intermediate_size
            .context("SharedExpertPrefillScratch requires cfg.shared_expert_intermediate_size")?;
        assert!(hidden % 32 == 0);
        assert!(inter % 32 == 0);

        let x_q8_1_bytes = max_tokens * (hidden / 32) * std::mem::size_of::<BlockQ8_1>();
        let inter_f32_bytes = max_tokens * inter * 4;
        let inter_f16_bytes = max_tokens * inter * 2;
        let inter_q8_1_bytes =
            max_tokens * (inter / 32) * std::mem::size_of::<BlockQ8_1>();
        let hidden_f32_bytes = max_tokens * hidden * 4;

        let x_q8_1 = device.alloc(x_q8_1_bytes)?;
        let gate_f32 = device.alloc(inter_f32_bytes)?;
        let up_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f32 = device.alloc(inter_f32_bytes)?;
        let activated_f16 = device.alloc(inter_f16_bytes)?;
        let activated_q8_1 = device.alloc(inter_q8_1_bytes)?;
        let down_f32 = device.alloc(hidden_f32_bytes)?;
        let x_norm_f32 = device.alloc(hidden_f32_bytes)?;

        Ok(Self {
            max_tokens,
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
            x_q8_1_bytes,
            inter_f32_bytes,
            inter_f16_bytes,
            inter_q8_1_bytes,
            hidden_f32_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.x_q8_1, self.x_q8_1_bytes)?;
            device.dealloc(self.gate_f32, self.inter_f32_bytes)?;
            device.dealloc(self.up_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f32, self.inter_f32_bytes)?;
            device.dealloc(self.activated_f16, self.inter_f16_bytes)?;
            device.dealloc(self.activated_q8_1, self.inter_q8_1_bytes)?;
            device.dealloc(self.down_f32, self.hidden_f32_bytes)?;
            device.dealloc(self.x_norm_f32, self.hidden_f32_bytes)?;
        }
        Ok(())
    }
}

impl Drop for SharedExpertPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "SharedExpertPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Shared-expert prefill. Mirrors `forward_shared_expert_decode` with
/// `n_tokens = L`; every op (`qmatmul`, `swiglu_f32`, `shared_expert_scale_f32`)
/// already accepts L natively.
#[allow(clippy::too_many_arguments)]
pub fn forward_shared_expert_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    shared: &crate::weights::SharedExpertWeights,
    scratch: &mut SharedExpertPrefillScratch,
    x_norm: DevicePtr,
    shared_out: DevicePtr,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 {
        bail!("forward_shared_expert_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_shared_expert_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let inter = cfg.shared_expert_intermediate_size.unwrap();

    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("prefill shexp x_norm → Q8_1")?;

    // Shared-expert FFN weights are Q4_K on all V1 targets (Qwen3.6 MoE);
    // Q4_K has no MmqLdsX64 row in dispatch, so the DS4 Q8_1 buffer is
    // semantically unused here — DevicePtr(0) is honest, not a placeholder.
    // If a future Q4_K turbo kernel lands, add x_q8_1_mmq to
    // SharedExpertPrefillScratch and populate it alongside x_q8_1.
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_gate_shexp, scratch.x_q8_1, DevicePtr(0), scratch.gate_f32,
        n_tokens, hidden, inter, "ffn_gate_shexp",
    )?;
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_up_shexp, scratch.x_q8_1, DevicePtr(0), scratch.up_f32,
        n_tokens, hidden, inter, "ffn_up_shexp",
    )?;
    swiglu_f32(
        ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f32,
        n_tokens * inter,
    )
    .context("prefill shexp swiglu_f32")?;
    cast_f32_to_f16(
        ops, stream, scratch.activated_f32, scratch.activated_f16, n_tokens * inter,
    )
    .context("prefill shexp cast activated → f16")?;
    quantize_f16_q8_1(
        ops, stream, scratch.activated_f16, scratch.activated_q8_1, n_tokens * inter,
    )
    .context("prefill shexp quantise → Q8_1")?;
    run_qmatmul_from_tensor(
        ops, stream, &shared.ffn_down_shexp, scratch.activated_q8_1, DevicePtr(0),
        scratch.down_f32,
        n_tokens, inter, hidden, "ffn_down_shexp",
    )?;
    cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, n_tokens * hidden)
        .context("prefill shexp cast x_norm → f32")?;
    shared_expert_scale_f32(
        ops, stream, scratch.down_f32, scratch.x_norm_f32,
        shared.ffn_gate_inp_shexp.ptr, n_tokens, hidden,
    )
    .context("prefill shared_expert_scale_f32")?;
    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_out, n_tokens * hidden)
        .context("prefill shexp cast → f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V1.7.3-f4 — per-layer prefill + forward_prefill end-to-end.
// ---------------------------------------------------------------------------

/// Scratches a per-layer prefill step touches: sibling of
/// `LayerForwardScratch` sized against `(cfg, max_tokens)`.
pub struct LayerPrefillScratch {
    pub max_tokens: usize,
    pub full_attn: Option<FullAttnPrefillScratch>,
    pub gdn: Option<GdnPrefillScratch>,
    pub moe: Option<MoePrefillScratch>,
    pub shared: Option<SharedExpertPrefillScratch>,
    /// Present iff `cfg.is_dense_ffn()`.
    pub dense_ffn: Option<DenseFfnPrefillScratch>,
    pub mid_f16: DevicePtr,           // F16 [L, hidden] — post-attn residual
    pub mid_norm_f16: DevicePtr,      // F16 [L, hidden] — rmsnorm(mid)
    pub shared_delta_f16: DevicePtr,  // F16 [L, hidden]
    pub moe_residual_f16: DevicePtr,  // F16 [L, hidden] — mid + shared
    hidden_bytes: usize,
    disposed: bool,
}

impl LayerPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1);
        let hidden_bytes = max_tokens * cfg.hidden_size * 2;

        let full_attn = Some(FullAttnPrefillScratch::new(cfg, device, max_tokens)?);
        let gdn = Some(GdnPrefillScratch::new(cfg, device, max_tokens)?);
        let (moe, shared, dense_ffn) = if cfg.is_dense_ffn() {
            (None, None, Some(DenseFfnPrefillScratch::new(cfg, device, max_tokens)?))
        } else {
            let moe = Some(MoePrefillScratch::new(cfg, device, max_tokens)?);
            let shared = if cfg.shared_expert_intermediate_size.is_some() {
                Some(SharedExpertPrefillScratch::new(cfg, device, max_tokens)?)
            } else {
                None
            };
            (moe, shared, None)
        };

        let mid_f16 = device.alloc(hidden_bytes)?;
        let mid_norm_f16 = device.alloc(hidden_bytes)?;
        let shared_delta_f16 = device.alloc(hidden_bytes)?;
        let moe_residual_f16 = device.alloc(hidden_bytes)?;

        Ok(Self {
            max_tokens,
            full_attn,
            gdn,
            moe,
            shared,
            dense_ffn,
            mid_f16,
            mid_norm_f16,
            shared_delta_f16,
            moe_residual_f16,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.mid_f16, self.hidden_bytes)?;
            device.dealloc(self.mid_norm_f16, self.hidden_bytes)?;
            device.dealloc(self.shared_delta_f16, self.hidden_bytes)?;
            device.dealloc(self.moe_residual_f16, self.hidden_bytes)?;
        }
        if let Some(s) = self.full_attn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.moe.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.shared.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.dense_ffn.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for LayerPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "LayerPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// One prefill chunk through one full layer. Mirrors
/// `forward_layer_decode`'s math but over `[L, hidden]` tensors.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    layer_cache: &mut LayerCache,
    scratch: &mut LayerPrefillScratch,
    x_in: DevicePtr,
    x_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
) -> Result<()> {
    let hidden = cfg.hidden_size;
    let il = layer_weights.layer_idx;

    // 1. Attention (full-attn or GDN) → mid_f16 (attn delta).
    if cfg.is_recurrent(il) {
        let LayerCache::Gdn(state) = layer_cache else {
            bail!("layer {il} expected GDN cache");
        };
        let gdn = scratch.gdn.as_mut().context("LayerPrefillScratch.gdn missing")?;
        let crate::weights::AttnWeights::Gdn(g) = &layer_weights.attn else {
            bail!("layer {il} expected GDN weights");
        };
        forward_gdn_prefill(
            ops,
            stream,
            device,
            cfg,
            &layer_weights.attn_norm,
            g,
            state,
            gdn,
            x_in,
            scratch.mid_f16,
            n_tokens,
        )?;
    } else {
        let LayerCache::FullAttn(kv) = layer_cache else {
            bail!("layer {il} expected FullAttn cache");
        };
        let full_attn = scratch
            .full_attn
            .as_mut()
            .context("LayerPrefillScratch.full_attn missing")?;
        let crate::weights::AttnWeights::FullAttn(fa) = &layer_weights.attn else {
            bail!("layer {il} expected FullAttn weights");
        };
        forward_full_attn_prefill(
            ops,
            stream,
            device,
            cfg,
            &layer_weights.attn_norm,
            fa,
            kv,
            full_attn,
            x_in,
            scratch.mid_f16,
            n_tokens,
            start_position,
        )?;
    }

    // 2. Residual: mid = x_in + attn_delta (in-place on mid_f16).
    add_f16(
        ops,
        stream,
        x_in,
        scratch.mid_f16,
        scratch.mid_f16,
        n_tokens * hidden,
    )
    .context("prefill layer residual: x_in + attn_delta")?;

    // 3. post-attention / ffn norm.
    let post_norm = layer_weights
        .post_attention_norm
        .as_ref()
        .or(layer_weights.ffn_norm.as_ref())
        .context("layer missing both post_attention_norm and ffn_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.mid_f16,
        post_norm.ptr,
        scratch.mid_norm_f16,
        n_tokens,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("prefill post-attn rmsnorm")?;

    // 4. FFN. Dense (qwen35) or MoE + optional shared expert.
    if cfg.is_dense_ffn() {
        let dense_w = layer_weights
            .ffn
            .dense
            .as_ref()
            .context("dense FFN prefill: layer.ffn.dense missing")?;
        let dense_scratch = scratch
            .dense_ffn
            .as_mut()
            .context("LayerPrefillScratch.dense_ffn missing")?;
        forward_dense_ffn_prefill(
            ops,
            stream,
            cfg,
            dense_w,
            dense_scratch,
            scratch.mid_norm_f16,
            scratch.mid_f16,
            x_out,
            n_tokens,
        )?;
        return Ok(());
    }

    // MoE path.
    let moe_residual = if let (Some(shared_w), Some(shared_scratch)) =
        (layer_weights.ffn.shared.as_ref(), scratch.shared.as_mut())
    {
        forward_shared_expert_prefill(
            ops,
            stream,
            cfg,
            shared_w,
            shared_scratch,
            scratch.mid_norm_f16,
            scratch.shared_delta_f16,
            n_tokens,
        )?;
        add_f16(
            ops,
            stream,
            scratch.mid_f16,
            scratch.shared_delta_f16,
            scratch.moe_residual_f16,
            n_tokens * hidden,
        )
        .context("prefill moe residual: mid + shared_delta")?;
        scratch.moe_residual_f16
    } else {
        scratch.mid_f16
    };

    // 5. Router (dense F32 GEMV × L + topk).
    let moe = scratch
        .moe
        .as_mut()
        .context("LayerPrefillScratch.moe missing")?;
    forward_router_prefill(
        ops,
        stream,
        cfg,
        layer_weights.ffn.ffn_gate_inp.as_ref().expect("MoE forward: ffn_gate_inp missing"),
        moe,
        scratch.mid_norm_f16,
        n_tokens,
    )?;

    // 6. Routed MoE FFN — fuses the residual add in moe_combine.
    forward_moe_ffn_prefill(
        ops,
        stream,
        cfg,
        &layer_weights.ffn,
        moe,
        scratch.mid_norm_f16,
        moe_residual,
        x_out,
        n_tokens,
    )?;

    Ok(())
}

/// Complete scratch for a prefill chunk of L tokens: two hidden ping-pong
/// buffers + per-layer prefill scratch + output-head scratch.
pub struct ForwardPrefillScratch {
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl ForwardPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        let hidden_bytes = max_tokens * cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = Some(LayerPrefillScratch::new(cfg, device, max_tokens)?);
        let output_head = Some(OutputHeadScratch::new(cfg, device)?);
        Ok(Self {
            max_tokens,
            hidden_a,
            hidden_b,
            layer,
            output_head,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for ForwardPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "ForwardPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Prefill a chunk of L tokens end-to-end. Feeds the whole chunk through
/// every layer, then runs the output head on the LAST token's hidden and
/// returns the argmax-sampled next token id.
///
/// Semantics:
/// - Each token's embedding is gathered on host and uploaded into
///   `hidden_a[t]` (one F16 row per token).
/// - `forward_layer_prefill` runs over all L tokens per layer.
/// - KV cache / GDN state / conv history are updated with L tokens of
///   history before returning.
/// - Output head runs on the last token's final hidden (F16 `[hidden]`
///   slice at offset `(L-1) * hidden`). Argmax on host.
///
/// Caller is responsible for chunking a long prompt if `L > scratch.max_tokens`.
pub fn forward_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    weights: &crate::weights::ModelWeights,
    session: &mut crate::session::Qwen3MoESession,
    scratch: &mut ForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill called with empty tokens");
    }
    if l > scratch.max_tokens {
        bail!(
            "forward_prefill: L={l} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    // 1. Gather embeddings for all L tokens into hidden_a, row-by-row.
    for (t, &token_id) in tokens.iter().enumerate() {
        forward_embed_decode_host(
            device,
            stream,
            &weights.token_embd,
            token_id,
            scratch.hidden_a.offset_bytes(t * row_bytes),
            hidden,
        )?;
    }

    // 2. Per-layer loop with ping-pong hidden state.
    let layer_scratch = scratch
        .layer
        .as_mut()
        .context("ForwardPrefillScratch.layer missing")?;
    let (mut x_in, mut x_out) = (scratch.hidden_a, scratch.hidden_b);
    for (il, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut session.layers_mut()[il];
        forward_layer_prefill(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            l,
            start_position,
        )?;
        std::mem::swap(&mut x_in, &mut x_out);
    }
    // `x_in` now holds the final hidden `[L, hidden]` F16.

    // 3. Output head on the LAST token only — argmax logits for that token
    //    are what the sampler needs. Upstream prefill users also typically
    //    care only about the final position; if a use case wants
    //    per-position logits, a variant could return them all.
    let last_token_hidden = x_in.offset_bytes((l - 1) * row_bytes);
    let lm_head = weights
        .output
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let output_head_scratch = scratch
        .output_head
        .as_mut()
        .context("ForwardPrefillScratch.output_head missing")?;
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &weights.output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(device, stream, output_head_scratch.logits_f32, cfg.vocab_size)
}

