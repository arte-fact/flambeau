//! Full-attention forward: decode (one token) + prefill (L tokens).
//!
//! Composes the attention block — RMSNorm + fused Q|gate projection + K/V
//! projection + partial NeoX RoPE + KV append + softmax attention + output
//! projection. The KV cache is `F16Contig`; Q8-KV is a separate code path
//! in V2+.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{kv_cache_append_hip_slot, MemcpySlot, ScalarSlot};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    attention::{
        attention_decode_f16_slots, attention_decode_q8_kv, attention_prefill_f16_slots,
        split_q_gate_f16,
    },
    cast::cast_f32_to_f16,
    mlp::sigmoid_mul_f16,
    norm::{quantize_f16_q8_0, quantize_f16_q8_1, quantize_f16_q8_1_mmq, rmsnorm_f16, rmsnorm_quant_q8_1},
    pe::rope_neox_partial_f16,
    qmatmul::{mmvq, mmvq_q8_0_gate_up, qmatmul},
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::BlockQ8_1;
use flambeau_runtime::{CacheLayout, F16Contig, KvCache, Q8Contig};

use super::common::{mat_shape, qdtype_of};
use crate::config::Qwen3MoEConfig;
use crate::session::LayerCache;
use crate::weights::{DenseAttnWeights, DeviceTensor, FullAttnWeights};

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
    /// V1-BENCH-#116a — Q8_0 staging for KvCache<Q8Contig>. Sized for
    /// `n_kv_heads * head_dim / 32` Q8_0 blocks (18 B each). Unused on the
    /// F16-KV path; tiny relative to the F16 buffers (16× smaller per-row
    /// since 18 B vs 32 B per block).
    pub k_q8_0: DevicePtr,         // Q8_0 blocks [n_kv_heads * head_dim / 32]
    pub v_q8_0: DevicePtr,         // Q8_0 blocks [n_kv_heads * head_dim / 32]
    pub attn_out_f16: DevicePtr,   // F16 [n_heads * head_dim]
    pub gated_out_f16: DevicePtr,  // F16 [n_heads * head_dim]
    pub positions: DevicePtr,      // i32 [1] — position for the current token
    /// V2.27.a-i2b — persistent host-side 1-slot position backing. Same
    /// motivation as V2.26.a-i5a's `positions_host` for prefill: the
    /// per-layer `upload_position` HtoD memcpy's source was a stack-local
    /// `[i32; 1]` requiring an internal `stream.synchronize()` to keep
    /// it alive across the copy — a per-layer per-token barrier of ~50 µs
    /// (~5 % of decode wall at 53 tok/s on Mesh<4>). The persistent Vec
    /// lets us drop the sync and keeps the memcpy source stable for
    /// V2.27.a-i3's graph-capture path.
    pub(crate) positions_host: Vec<i32>,
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
    kv_q8_0_bytes: usize,
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
        // V1-BENCH-#116a — Q8_0 staging for the q8_contig KV path. Each
        // 32-element block is 34 B (2-byte fp16 scale + 32 int8 quants =
        // `sizeof(flambeau_block_q8_0)`). Allocated unconditionally so
        // dispatch on L::NAME can pick the right buffer without touching
        // scratch construction.
        //
        // V1-BENCH-#117 fix: was 18 B/block (wrong arithmetic — assumed
        // 16 int8 quants instead of QK8_0=32). Q8 KV path was OOB-writing
        // 1088 B into a 576 B staging slab, corrupting the next allocation
        // and writing only ~17/32 blocks worth of data into the cache.
        // First-token logits looked plausible because attn_out_f16 is
        // overwritten by the attention kernel after; from token 1 onward
        // the cache held mismatched data and logits collapsed to ~0.
        assert!(kv_width % 32 == 0, "kv_width must be a multiple of QK8_0=32 for Q8 KV staging");
        let kv_q8_0_bytes = (kv_width / 32) * flambeau_runtime::Q8_0_BLOCK_BYTES;
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
        let k_q8_0 = device.alloc(kv_q8_0_bytes)?;
        let v_q8_0 = device.alloc(kv_q8_0_bytes)?;
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
            k_q8_0,
            v_q8_0,
            attn_out_f16,
            gated_out_f16,
            positions,
            positions_host: vec![0i32; 1],
            splitk_partials_m,
            splitk_partials_s,
            splitk_partials_o,
            x_norm_bytes,
            x_q8_1_bytes,
            mmvq_f32_bytes,
            q_fused_bytes,
            qk_bytes,
            kv_bytes,
            kv_q8_0_bytes,
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
            device.dealloc(self.k_q8_0, self.kv_q8_0_bytes)?;
            device.dealloc(self.v_q8_0, self.kv_q8_0_bytes)?;
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

// `upload_position`, `qdtype_of`, `mat_shape` moved to `forward::common`.

/// Decode step for one full-attention layer. Consumes `x_in` (F16 `[H]`)
/// and writes the pre-residual output to `delta_out` (F16 `[H]`). The
/// caller is expected to do the residual sum (`out = x_in + delta_out`)
/// outside this function — V1.7.3-e adds the fused residual-add kernel
/// for the top-level compose.
///
/// Appends to `kv_cache` at the current tail. `position` is the 0-based
/// token index used by RoPE and also the `n_tokens_kv` for the attention
/// kernel after the append bumps the cache size by 1.
/// V2.27.a-i3 — per-layer slot bundle for graph-captureable decode.
/// Present only when the caller is building a decode graph capture;
/// non-capture callers pass `None` to `forward_full_attn_decode`.
#[derive(Clone, Copy, Debug)]
pub struct AttnDecodeSlots {
    /// Tags `n_tokens_kv` at `attention_decode_f16`. Value per replay
    /// = `position + 1` (cache tail after the current token's append).
    pub n_tokens_kv_slot: ScalarSlot,
    /// Tags the K-tensor dst of `kv_cache.append`.
    pub k_append_slot: MemcpySlot,
    /// Tags the V-tensor dst of `kv_cache.append`.
    pub v_append_slot: MemcpySlot,
}

pub fn forward_full_attn_decode<L: CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    post_attn_norm: Option<&DeviceTensor>,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
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
    let fuse_kv = weights.attn_k.dtype == flambeau_quant::GgmlDType::Q8_0
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
    //
    // V2.27.a-i2b — upload the position via the scratch's persistent
    // positions_host Vec so the HtoD source is stable across graph
    // replays AND we can drop the internal sync the stack-local
    // [position] variant needed. Saves ~50 µs/layer/token on decode.
    scratch.positions_host[0] = position as i32;
    // SAFETY: scratch.positions has 4 valid bytes; positions_host is a
    // persistent Vec on the scratch whose lifetime exceeds the copy.
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
    //
    // V1-BENCH-#116 — generic over `L: CacheLayout`. F16Contig appends
    // F16 K/V directly. Q8Contig quantises K/V (F16) → Q8_0 staging
    // first, then appends 18 B/block. Graph-capture slots are F16-only
    // (no Q8 capture path yet); Q8 + slots is rejected.
    let kv_layout = L::NAME;
    if let Some(AttnDecodeSlots { k_append_slot, v_append_slot, .. }) = slots {
        if kv_layout != F16Contig::NAME {
            bail!(
                "forward_full_attn_decode: graph-capture slots are F16-only; \
                 KV layout {kv_layout} not supported under capture"
            );
        }
        // SAFETY: scratch.k_f16 and v_f16 are contiguous F16 `[n_kv_heads, head_dim]`.
        // V2.27.a-i3 — capture-tagged append so dst can be retargeted
        // per replay via HipGraphExec::set_memcpy_slot.
        unsafe {
            kv_cache_append_hip_slot(
                kv_cache,
                device,
                stream,
                scratch.k_f16,
                scratch.v_f16,
                1,
                k_append_slot,
                v_append_slot,
            )?;
        }
    } else if kv_layout == Q8Contig::NAME {
        let kv_elems = n_kv_heads * head_dim;
        quantize_f16_q8_0(ops, stream, scratch.k_f16, scratch.k_q8_0, kv_elems)
            .context("quantize attn_k → q8_0")?;
        quantize_f16_q8_0(ops, stream, scratch.v_f16, scratch.v_q8_0, kv_elems)
            .context("quantize attn_v → q8_0")?;
        // SAFETY: k_q8_0/v_q8_0 hold (n_kv_heads * head_dim / 32) Q8_0
        // blocks (18 B each). KvCache<Q8Contig>::append expects exactly
        // `n_new * n_heads * Q8Contig::bytes_per_row(head_dim)` bytes,
        // which equals `n_new * n_heads * (head_dim/32) * 18` —
        // matches `kv_q8_0_bytes` for n_new=1.
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_q8_0, scratch.v_q8_0, 1)
                .map_err(|e| anyhow::anyhow!("kv_cache.append (q8): {e}"))?;
        }
    } else {
        // SAFETY: scratch.k_f16 and v_f16 are contiguous F16 `[n_kv_heads, head_dim]`.
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
                .map_err(|e| anyhow::anyhow!("kv_cache.append: {e}"))?;
        }
    }

    // 9. Attention decode against the full cache (includes the token we
    //    just appended — `current_tokens = position + 1`).
    //
    // V2.19.b — split-K (flash-decoding) for long contexts. The single-pass
    // kernel hits 27 % CU occupancy (16 heads × 1 block on 60 CUs) and
    // serialises over n_tokens_kv per block; at n_tokens=2048 that's 2647 µs
    // vs split-K's 340 µs (7.78×). FLAMBEAU_VARIANT=baseline opts out.
    //
    // V1-BENCH-#116 follow-up — split-K now exists for both F16 and Q8 KV.
    // The Q8 variant dequants on the fly during the chunk pass; combine
    // pass is layout-agnostic (operates on f32 partials).
    let n_tokens_kv = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    let use_splitk = slots.is_none() && n_tokens_kv > 256;
    if use_splitk {
        let chunk_size = flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv);
        let n_chunks = n_tokens_kv.div_ceil(chunk_size);
        debug_assert!(
            n_chunks <= MAX_SPLITK_CHUNKS,
            "split-K n_chunks={n_chunks} exceeds scratch budget MAX={MAX_SPLITK_CHUNKS}"
        );
        if kv_layout == Q8Contig::NAME {
            flambeau_ops::hip::attention::attention_decode_q8_kv_splitk(
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
            .context("attention_decode_q8_kv_splitk")?;
        } else {
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
        }
    } else if kv_layout == Q8Contig::NAME {
        attention_decode_q8_kv(
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
        .context("attention_decode_q8_kv")?;
    } else {
        let n_tokens_kv_slot = slots.map(|s| s.n_tokens_kv_slot);
        attention_decode_f16_slots(
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
            n_tokens_kv_slot,
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
///
/// V1-BENCH-#116 — dispatches on the cache variant (F16Contig or Q8Contig)
/// to the same generic `forward_full_attn_decode<L>` body; the compiler
/// monomorphises and the runtime branch in the body picks the right
/// quantise + attention kernels.
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
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    let crate::weights::AttnWeights::FullAttn(fa) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not FullAttn variant",
            layer_weights.layer_idx
        );
    };
    match layer_cache {
        LayerCache::FullAttn(kv) => forward_full_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm, layer_weights.post_attention_norm.as_ref(),
            fa, kv, scratch, x_in, delta_out, position, slots,
        ),
        LayerCache::FullAttnQ8(kv) => forward_full_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm, layer_weights.post_attention_norm.as_ref(),
            fa, kv, scratch, x_in, delta_out, position, slots,
        ),
        LayerCache::Gdn(_) => bail!(
            "layer {} is not a full-attn layer (cache variant mismatch)",
            layer_weights.layer_idx
        ),
    }
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
    /// **#266c** — per-slot K-cache base pointers, uploaded HtoD once
    /// per `forward_full_attn_layer_decode_batched_*` call so the
    /// batched-attention kernel can address each slot's KV. u64 [max_L].
    pub slot_k_ptrs: DevicePtr,
    /// **#266c** — per-slot V-cache base pointers. u64 [max_L].
    pub slot_v_ptrs: DevicePtr,
    /// **#266c** — per-slot KV-tail length post-append. i32 [max_L].
    pub slot_n_tokens_kv: DevicePtr,
    /// Persistent host-side staging for the slot tables. Same lifetime
    /// rationale as `positions_host`: stable address for HtoD memcpy.
    pub(crate) slot_k_ptrs_host: Vec<u64>,
    pub(crate) slot_v_ptrs_host: Vec<u64>,
    pub(crate) slot_n_tokens_kv_host: Vec<i32>,
    /// V2.26.a-i5a — persistent host-side position buffer. `positions`
    /// on the device is filled each prefill call via a HtoD memcpy
    /// whose *source* is this Vec's stable address. Keeping it on the
    /// scratch (and therefore alive for the scratch's lifetime) is
    /// what makes the memcpy safe to capture into a `HipGraphExec` —
    /// the previous path used a transient `Vec<i32>` created inside
    /// `upload_positions_range`, whose address becomes invalid once
    /// that function returns and breaks graph replay.
    ///
    /// Sized `max_tokens`; writes are in-place via `[..n].copy_from_slice`.
    pub(crate) positions_host: Vec<i32>,
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
    slot_k_ptrs_bytes: usize,
    slot_v_ptrs_bytes: usize,
    slot_n_tokens_kv_bytes: usize,
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
        let slot_k_ptrs_bytes = max_tokens * 8;
        let slot_v_ptrs_bytes = max_tokens * 8;
        let slot_n_tokens_kv_bytes = max_tokens * 4;

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
        let slot_k_ptrs = device.alloc(slot_k_ptrs_bytes)?;
        let slot_v_ptrs = device.alloc(slot_v_ptrs_bytes)?;
        let slot_n_tokens_kv = device.alloc(slot_n_tokens_kv_bytes)?;

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
            slot_k_ptrs,
            slot_v_ptrs,
            slot_n_tokens_kv,
            slot_k_ptrs_host: vec![0u64; max_tokens],
            slot_v_ptrs_host: vec![0u64; max_tokens],
            slot_n_tokens_kv_host: vec![0i32; max_tokens],
            positions_host: vec![0i32; max_tokens],
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
            slot_k_ptrs_bytes,
            slot_v_ptrs_bytes,
            slot_n_tokens_kv_bytes,
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
            device.dealloc(self.slot_k_ptrs, self.slot_k_ptrs_bytes)?;
            device.dealloc(self.slot_v_ptrs, self.slot_v_ptrs_bytes)?;
            device.dealloc(self.slot_n_tokens_kv, self.slot_n_tokens_kv_bytes)?;
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

/// Upload `L` i32 positions `[start_position, start_position + L)` into
/// the device-side `positions` scratch slot.
///
/// V2.26.a-i5a — the host-side source is `scratch.positions_host`, a
/// persistent `Vec<i32>` owned by the scratch. Previously this fn built
/// a transient `Vec<i32>` on the stack, uploaded, and synced to keep
/// the Vec alive across the copy. That works for direct dispatch but
/// breaks graph capture: the memcpy node records the source POINTER;
/// the driver re-reads from it at replay time; if the Vec is gone,
/// replay reads freed memory. Using `positions_host` keeps the source
/// stable for the scratch's lifetime, so the same memcpy replays safely
/// and — critically — picks up new values when we overwrite the host
/// Vec between replays.
///
/// The internal `stream.synchronize()` is dropped: the memcpy is
/// ordered in-stream with any subsequent RoPE launch, and the host
/// storage (`positions_host`) outlives both the copy and the stream
/// work that reads the device target.
pub(crate) fn upload_positions_range(
    device: &HipDevice,
    stream: &HipStream,
    scratch: &mut FullAttnPrefillScratch,
    start_position: usize,
    n: usize,
) -> Result<()> {
    if n > scratch.max_tokens {
        anyhow::bail!(
            "upload_positions_range: n={n} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    for i in 0..n {
        scratch.positions_host[i] = (start_position + i) as i32;
    }
    // SAFETY: `scratch.positions` has at least `n * 4` valid bytes;
    // `positions_host` has at least `n` valid i32 slots and outlives
    // every in-stream reader of the device dst (bounded by the scratch
    // lifetime, which the caller is responsible for keeping alive
    // until the stream is synced).
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            n * 4,
        )?;
    }
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
/// V2.26.a-i5b — optional slot bundle for graph-captureable prefill.
///
/// When a caller under [`flambeau_backend_hip::HipGraphExec::capture`]
/// passes `Some(AttnPrefillSlots { .. })`, `forward_full_attn_prefill`
/// tags the pos-varying kernel args + the K/V append memcpys so the
/// resulting exec can be replayed across ubatches with just
/// `set_slot` + `set_memcpy_slot` calls.
///
/// `None` preserves the original non-captureable behaviour — existing
/// callers are unaffected.
#[derive(Clone, Copy, Debug)]
pub struct AttnPrefillSlots {
    /// Tags `n_k_tokens` at `attention_prefill_f16`. Value per ubatch
    /// = `start_position + n_tokens` (cache tail after append).
    pub n_k_slot: ScalarSlot,
    /// Tags `q_offset` at `attention_prefill_f16`. Value per ubatch
    /// = `start_position`.
    pub q_off_slot: ScalarSlot,
    /// Tags the K-tensor dst of `kv_cache.append`. Value per ubatch
    /// = `cache.k_buffer + start_position * per_token_bytes`.
    pub k_append_slot: MemcpySlot,
    /// Tags the V-tensor dst of `kv_cache.append`.
    pub v_append_slot: MemcpySlot,
}

pub fn forward_full_attn_prefill<L: flambeau_runtime::CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &FullAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    slots: Option<AttnPrefillSlots>,
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
    upload_positions_range(device, stream, scratch, start_position, n_tokens)?;
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
    let kv_layout = L::NAME;
    if kv_layout == Q8Contig::NAME {
        // Q8 path: quantise directly into the cache slot. Slots variant
        // (graph-captured kv_cache_append_hip_slot) is F16-only — Q8
        // path bypasses graph capture for now (would need a slot-aware
        // quantize launch).
        let kv_elems_per_token = n_kv_heads * head_dim;
        let total_elems = n_tokens * kv_elems_per_token;
        let (k_dst, v_dst, _) = kv_cache
            .compute_append_dsts(n_tokens)
            .map_err(|e| anyhow::anyhow!("kv_cache.compute_append_dsts q8 (PP prefill): {e}"))?;
        flambeau_ops::hip::norm::quantize_f16_q8_0(ops, stream, scratch.k_f16, k_dst, total_elems)
            .context("quantize prefill K → q8_0 in-place (PP)")?;
        flambeau_ops::hip::norm::quantize_f16_q8_0(ops, stream, scratch.v_f16, v_dst, total_elems)
            .context("quantize prefill V → q8_0 in-place (PP)")?;
        kv_cache
            .bump_tail(n_tokens)
            .map_err(|e| anyhow::anyhow!("kv_cache.bump_tail q8 (PP prefill): {e}"))?;
    } else if let Some(AttnPrefillSlots { k_append_slot, v_append_slot, .. }) = slots {
        // V2.26.a-i5b — captureable variant: tag each memcpy so dst can
        // be retargeted per-replay via HipGraphExec::set_memcpy_slot.
        unsafe {
            kv_cache_append_hip_slot(
                kv_cache,
                device,
                stream,
                scratch.k_f16,
                scratch.v_f16,
                n_tokens,
                k_append_slot,
                v_append_slot,
            )?;
        }
    } else {
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_f16, scratch.v_f16, n_tokens)
                .map_err(|e| anyhow::anyhow!("kv_cache.append(L={n_tokens}): {e}"))?;
        }
    }

    // 8. Causal prefill attention. `n_k_tokens = start_position + L`
    // (after append); `q_offset = start_position` so Q row i attends to
    // K rows `0..start_position + i + 1`.
    let n_k_tokens = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    if kv_layout == Q8Contig::NAME {
        flambeau_ops::hip::attention::attention_prefill_q8_kv(
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
        .context("attention_prefill_q8_kv (PP)")?;
    } else {
        let (n_k_slot_opt, q_off_slot_opt) = match slots {
            Some(s) => (Some(s.n_k_slot), Some(s.q_off_slot)),
            None => (None, None),
        };
        attention_prefill_f16_slots(
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
            n_k_slot_opt,
            q_off_slot_opt,
        )
        .context("attention_prefill_f16")?;
    }

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

/// **P2.9b-i2-A1** — upload arbitrary per-slot positions into
/// `scratch.positions`. Sibling of `upload_positions_range` for the
/// continuous-batching path where each batched row decodes at its own
/// cache offset.
pub(crate) fn upload_positions_arbitrary(
    device: &HipDevice,
    stream: &HipStream,
    scratch: &mut FullAttnPrefillScratch,
    slot_positions: &[usize],
) -> Result<()> {
    let n = slot_positions.len();
    if n > scratch.max_tokens {
        anyhow::bail!(
            "upload_positions_arbitrary: n={n} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    for (i, &p) in slot_positions.iter().enumerate() {
        scratch.positions_host[i] = p as i32;
    }
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.positions,
            DevicePtr(scratch.positions_host.as_ptr() as usize),
            n * 4,
        )?;
    }
    Ok(())
}

/// **P2.9b-i2-A1** — batched decode for one full-attention layer.
///
/// Mirrors [`forward_full_attn_prefill`] for the input-side ops
/// (rmsnorm, Q|gate / K / V projections, per-head Q/K rmsnorm, RoPE)
/// at `n_tokens = slot_positions.len()`, so those kernel launches are
/// shared across slots. The KV-append (step 7) and attention (step 8)
/// are split per slot because each slot owns its own KV cache and
/// query history.
///
/// Layout:
///   - `x_in` / `delta_out` are F16 `[N, hidden]`, row `s` belongs to
///     slot `s` whose index in the per-rank session/cache arrays is
///     also `s`.
///   - `slot_caches[s]` is the layer-local KV cache for slot `s`. All
///     must be `LayerCache::FullAttn` with identical `n_kv_heads`
///     and `head_dim`.
///   - `slot_positions[s]` is the cache tail for slot `s` *before* this
///     token is appended (i.e. the position the new K/V row writes to).
///   - `scratch` is a single shared per-rank `FullAttnPrefillScratch`
///     sized for `max_tokens >= N` — the same buffers that prefill uses,
///     reused as the [N, *] batched workspace.
pub fn forward_full_attn_layer_decode_batched(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    slot_caches: &mut [&mut LayerCache],
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    slot_positions: &[usize],
) -> Result<()> {
    let n_tokens = slot_positions.len();
    if n_tokens == 0 {
        bail!("forward_full_attn_layer_decode_batched called with 0 slots");
    }
    if n_tokens != slot_caches.len() {
        bail!(
            "slot count mismatch: positions={n_tokens}, caches={}",
            slot_caches.len()
        );
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "n_tokens={n_tokens} > scratch.max_tokens={}; size scratch for ≥N",
            scratch.max_tokens
        );
    }

    let crate::weights::AttnWeights::FullAttn(weights) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not FullAttn (i2-A1 only handles gated full-attn)",
            layer_weights.layer_idx
        );
    };

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let rope = &cfg.rope;
    let attn_norm = &layer_weights.attn_norm;

    // Steps 1-6: identical to forward_full_attn_prefill body. These all
    // operate on [N, hidden] / [N, q_width] / [N, kv_width] and don't
    // depend on a single contiguous KV cache, so the existing kernels
    // batch across slots naturally.

    // 1. rmsnorm + Q8_1 quantise (both layouts).
    rmsnorm_f16(
        ops, stream, x_in, attn_norm.ptr, scratch.x_norm_f16,
        n_tokens, hidden, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("batched-decode x_norm → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("batched-decode x_norm → Q8_1 (MMQ DS4)")?;

    // 2. Q|gate fused projection.
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != 2 * n_heads * head_dim || q_k != hidden {
        bail!(
            "attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            2 * n_heads * head_dim, hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_q.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, q_k, q_rows, dtype_q,
    )
    .context("batched-decode qmatmul attn_q")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.q_fused_f16, n_tokens * q_rows,
    )
    .context("batched-decode cast attn_q → f16")?;

    // 3. Split Q | gate.
    split_q_gate_f16(
        ops, stream, scratch.q_fused_f16,
        scratch.q_f16, scratch.gate_f16,
        n_tokens, n_heads, head_dim,
    )
    .context("batched-decode split_q_gate")?;

    // 4. K / V projections.
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    if k_rows != kv_width || k_k != hidden {
        bail!(
            "attn_k shape [{k_rows}, {k_k}] != expected [{kv_width}, {hidden}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_k.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, k_k, k_rows, dtype_k,
    )
    .context("batched-decode qmatmul attn_k")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.k_f16, n_tokens * k_rows,
    )
    .context("batched-decode cast attn_k → f16")?;

    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if v_rows != kv_width || v_k != hidden {
        bail!(
            "attn_v shape [{v_rows}, {v_k}] != expected [{kv_width}, {hidden}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_v.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, v_k, v_rows, dtype_v,
    )
    .context("batched-decode qmatmul attn_v")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.v_f16, n_tokens * v_rows,
    )
    .context("batched-decode cast attn_v → f16")?;

    // 5. Per-head Q/K rmsnorm.
    let q_norm_dim = weights
        .attn_q_norm.dims.first().copied()
        .context("attn_q_norm missing dim")? as usize;
    if q_norm_dim != head_dim {
        bail!("attn_q_norm dim {q_norm_dim} != head_dim {head_dim}");
    }
    rmsnorm_f16(
        ops, stream, scratch.q_f16, weights.attn_q_norm.ptr, scratch.q_f16,
        n_tokens * n_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_q_norm")?;
    rmsnorm_f16(
        ops, stream, scratch.k_f16, weights.attn_k_norm.ptr, scratch.k_f16,
        n_tokens * n_kv_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("batched-decode attn_k_norm")?;

    // 6. RoPE on Q / K with per-slot positions.
    upload_positions_arbitrary(device, stream, scratch, slot_positions)?;
    rope_neox_partial_f16(
        ops, stream, scratch.q_f16, scratch.positions, rope.freq_base,
        n_tokens, n_heads, head_dim, rope.rotated_dims,
    )
    .context("batched-decode rope Q")?;
    rope_neox_partial_f16(
        ops, stream, scratch.k_f16, scratch.positions, rope.freq_base,
        n_tokens, n_kv_heads, head_dim, rope.rotated_dims,
    )
    .context("batched-decode rope K")?;

    // 7. Per-slot KV append. **#275 fix**: write at `current_tokens`
    //    (the cache tail) rather than `slot_positions[s]` (which is
    //    `prompt_ids.len() + step` = off by 1). Matches legacy
    //    `kv_cache.append()` semantics. F16-only (FullAttn cache).
    //
    //    **#266c**: in the same pass, populate the per-slot pointer/
    //    length tables consumed by `attention_decode_f16_batched`.
    let kv_per_token_bytes = kv_width * 2;
    for (s, cache) in slot_caches.iter_mut().enumerate() {
        let LayerCache::FullAttn(kv) = cache else {
            bail!(
                "i2-A1 batched decode: slot {s} cache is not FullAttn \
                 (Q8 KV path uses sequential single-slot decode)"
            );
        };
        let write_pos = kv.current_tokens();
        let k_src = scratch.k_f16.offset_bytes(s * kv_per_token_bytes);
        let v_src = scratch.v_f16.offset_bytes(s * kv_per_token_bytes);
        let k_dst = kv.k_buffer().offset_bytes(write_pos * kv_per_token_bytes);
        let v_dst = kv.v_buffer().offset_bytes(write_pos * kv_per_token_bytes);
        // SAFETY: src/dst are valid; write_pos < max_seq_len is bounded
        // by the cache.
        unsafe {
            device.memcpy_async(
                stream, CopyDirection::DeviceToDevice,
                k_dst, k_src, kv_per_token_bytes,
            )?;
            device.memcpy_async(
                stream, CopyDirection::DeviceToDevice,
                v_dst, v_src, kv_per_token_bytes,
            )?;
        }
        kv.bump_tail(1)
            .map_err(|e| anyhow::anyhow!("slot {s} bump_tail: {e}"))?;
        // Populate the slot tables for the batched-attention launch
        // below. n_tokens_kv reads post-bump (= includes just-written row).
        scratch.slot_k_ptrs_host[s] = kv.k_buffer().as_usize() as u64;
        scratch.slot_v_ptrs_host[s] = kv.v_buffer().as_usize() as u64;
        scratch.slot_n_tokens_kv_host[s] = kv.current_tokens() as i32;
    }

    // 8. Single-launch batched attention over all N slots
    //    (**#266c** — replaces the per-slot loop).
    let scale = (head_dim as f32).sqrt().recip();
    // SAFETY: each `slot_*_host[..n_tokens]` is a Vec<u64|i32> with
    // stable address; the corresponding device buffer is sized to
    // `max_tokens` ≥ n_tokens; HtoD bytes are bounded by the slice.
    unsafe {
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_k_ptrs,
            DevicePtr(scratch.slot_k_ptrs_host.as_ptr() as usize),
            n_tokens * 8,
        )?;
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_v_ptrs,
            DevicePtr(scratch.slot_v_ptrs_host.as_ptr() as usize),
            n_tokens * 8,
        )?;
        device.memcpy_async(
            stream, CopyDirection::HostToDevice,
            scratch.slot_n_tokens_kv,
            DevicePtr(scratch.slot_n_tokens_kv_host.as_ptr() as usize),
            n_tokens * 4,
        )?;
    }
    flambeau_ops::hip::attention::attention_decode_f16_batched(
        ops,
        stream,
        scratch.q_f16,
        scratch.slot_k_ptrs,
        scratch.slot_v_ptrs,
        scratch.attn_out_f16,
        scratch.slot_n_tokens_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        n_tokens,
        scale,
    )
    .context("batched-decode attention (i2-E #266c)")?;

    // 9. Sigmoid-gate over [N, q_width].
    sigmoid_mul_f16(
        ops, stream,
        scratch.gate_f16, scratch.attn_out_f16, scratch.gated_out_f16,
        n_tokens * q_width,
    )
    .context("batched-decode post-attn sigmoid-gate")?;

    // 10. Quantise gated_out to both Q8_1 layouts for output proj.
    quantize_f16_q8_1(
        ops, stream,
        scratch.gated_out_f16, scratch.gated_q8_1, n_tokens * q_width,
    )
    .context("batched-decode quantise gated → Q8_1 (std)")?;
    quantize_f16_q8_1_mmq(
        ops, stream,
        scratch.gated_out_f16, scratch.gated_q8_1_mmq, q_width, n_tokens,
    )
    .context("batched-decode quantise gated → Q8_1 (MMQ DS4)")?;

    // 11. Output projection over [N, hidden].
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != q_width {
        bail!(
            "attn_output shape [{o_rows}, {o_k}] != expected [{hidden}, {q_width}]"
        );
    }
    qmatmul(
        ops, stream, weights.attn_output.ptr,
        scratch.gated_q8_1, scratch.gated_q8_1_mmq, scratch.mmvq_f32,
        n_tokens, o_k, o_rows, dtype_o,
    )
    .context("batched-decode qmatmul attn_output")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, delta_out, n_tokens * hidden,
    )
    .context("batched-decode cast attn_output → f16")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// V2.28.b-i1 — dense attention (qwen3moe family).
//
// Differs from `forward_full_attn_*` above:
//   - Q projection is plain `[n_heads*head_dim, hidden]` (not 2× fused with
//     an input gate).
//   - No `split_q_gate`.
//   - No post-attn `sigmoid_mul` — `attn_out_f16` is quantised and fed
//     directly into the output projection.
//   - Q/K/V biases can be present (added after each matmul); Qwen3-Coder-30B
//     has none, but the path supports optional biases via a runtime bail if
//     the caller passes them (not yet implemented — asserts None).
//
// Reuses `FullAttnScratch` / `FullAttnPrefillScratch` — `q_fused_f16` and
// `gate_f16` inside them go unused on this path (~128 KiB dead per layer,
// fine at 4.11 GiB/rank for Qwen3-Coder).
// ---------------------------------------------------------------------------

pub fn forward_dense_attn_decode<L: CacheLayout>(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &DenseAttnWeights,
    kv_cache: &mut KvCache<L, HipDevice>,
    scratch: &mut FullAttnScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    position: usize,
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    if weights.attn_q_bias.is_some()
        || weights.attn_k_bias.is_some()
        || weights.attn_v_bias.is_some()
    {
        bail!(
            "forward_dense_attn_decode: Q/K/V biases not yet supported \
             (Qwen3-Coder-30B has none; add a bias_add step to support \
             older Qwen3 variants)"
        );
    }

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
    .context("dense attn_norm + quant")?;

    // 2. Plain Q projection → q_f16 directly (no fused gate split).
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != n_heads * head_dim || q_k != hidden {
        bail!(
            "dense attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            n_heads * head_dim,
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
    .context("dense mmvq attn_q")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.q_f16, q_rows)
        .context("dense cast attn_q → f16")?;

    // 3. K and V projections (optionally fused for Q8_0 pair).
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if k_rows != n_kv_heads * head_dim || k_k != hidden {
        bail!(
            "dense attn_k shape [{k_rows}, {k_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    if v_rows != n_kv_heads * head_dim || v_k != hidden {
        bail!(
            "dense attn_v shape [{v_rows}, {v_k}] != expected [{}, {}]",
            n_kv_heads * head_dim,
            hidden
        );
    }
    let fuse_kv = weights.attn_k.dtype == flambeau_quant::GgmlDType::Q8_0
        && weights.attn_v.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_kv {
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
        .context("dense attn_k + attn_v fused mmvq_q8_0")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.k_f16, k_rows)
            .context("dense cast attn_k → f16")?;
        cast_f32_to_f16(ops, stream, v_f32_offset, scratch.v_f16, v_rows)
            .context("dense cast attn_v → f16")?;
    } else {
        mmvq(
            ops, stream, weights.attn_k.ptr, scratch.x_q8_1,
            scratch.mmvq_f32, k_rows, k_k, dtype_k,
        ).context("dense mmvq attn_k")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.k_f16, k_rows)
            .context("dense cast attn_k → f16")?;
        mmvq(
            ops, stream, weights.attn_v.ptr, scratch.x_q8_1,
            scratch.mmvq_f32, v_rows, v_k, dtype_v,
        ).context("dense mmvq attn_v")?;
        cast_f32_to_f16(ops, stream, scratch.mmvq_f32, scratch.v_f16, v_rows)
            .context("dense cast attn_v → f16")?;
    }

    // 4. Per-head RMSNorm on Q and K.
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
    .context("dense attn_q_norm")?;
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
    .context("dense attn_k_norm")?;

    // 5. RoPE. qwen3moe uses standard NeoX RoPE over `head_dim` (no
    //    multi-freq sections — `rope.rotated_dims` equals head_dim).
    scratch.positions_host[0] = position as i32;
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
        ops, stream, scratch.q_f16, scratch.positions,
        rope.freq_base, 1, n_heads, head_dim, rope.rotated_dims,
    )
    .context("dense rope Q")?;
    rope_neox_partial_f16(
        ops, stream, scratch.k_f16, scratch.positions,
        rope.freq_base, 1, n_kv_heads, head_dim, rope.rotated_dims,
    )
    .context("dense rope K")?;

    // 6. KV append. V1-BENCH-#116 — same dispatch as gated full-attn.
    let kv_layout = L::NAME;
    if let Some(AttnDecodeSlots { k_append_slot, v_append_slot, .. }) = slots {
        if kv_layout != F16Contig::NAME {
            bail!(
                "forward_dense_attn_decode: graph-capture slots are F16-only; \
                 KV layout {kv_layout} not supported under capture"
            );
        }
        unsafe {
            kv_cache_append_hip_slot(
                kv_cache, device, stream,
                scratch.k_f16, scratch.v_f16, 1,
                k_append_slot, v_append_slot,
            )?;
        }
    } else if kv_layout == Q8Contig::NAME {
        let kv_elems = n_kv_heads * head_dim;
        quantize_f16_q8_0(ops, stream, scratch.k_f16, scratch.k_q8_0, kv_elems)
            .context("dense quantize attn_k → q8_0")?;
        quantize_f16_q8_0(ops, stream, scratch.v_f16, scratch.v_q8_0, kv_elems)
            .context("dense quantize attn_v → q8_0")?;
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_q8_0, scratch.v_q8_0, 1)
                .map_err(|e| anyhow::anyhow!("dense kv_cache.append (q8): {e}"))?;
        }
    } else {
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_f16, scratch.v_f16, 1)
                .map_err(|e| anyhow::anyhow!("dense kv_cache.append: {e}"))?;
        }
    }

    // 7. Attention decode. V1-BENCH-#116 follow-up — both F16 and Q8 KV
    // layouts have a split-K (flash-decoding) variant; combine pass is
    // f32 partials, layout-agnostic.
    let n_tokens_kv = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    let use_splitk = slots.is_none() && n_tokens_kv > 256;
    if use_splitk {
        let chunk_size = flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv);
        let n_chunks = n_tokens_kv.div_ceil(chunk_size);
        debug_assert!(
            n_chunks <= MAX_SPLITK_CHUNKS,
            "split-K n_chunks={n_chunks} exceeds scratch budget MAX={MAX_SPLITK_CHUNKS}"
        );
        if kv_layout == Q8Contig::NAME {
            flambeau_ops::hip::attention::attention_decode_q8_kv_splitk(
                ops, stream,
                scratch.q_f16, kv_cache.k_buffer(), kv_cache.v_buffer(),
                scratch.attn_out_f16,
                scratch.splitk_partials_m, scratch.splitk_partials_s, scratch.splitk_partials_o,
                n_heads, n_kv_heads, head_dim, n_tokens_kv, chunk_size, scale,
            )
            .context("dense attention_decode_q8_kv_splitk")?;
        } else {
            flambeau_ops::hip::attention::attention_decode_f16_splitk(
                ops, stream,
                scratch.q_f16, kv_cache.k_buffer(), kv_cache.v_buffer(),
                scratch.attn_out_f16,
                scratch.splitk_partials_m, scratch.splitk_partials_s, scratch.splitk_partials_o,
                n_heads, n_kv_heads, head_dim, n_tokens_kv, chunk_size, scale,
            )
            .context("dense attention_decode_f16_splitk")?;
        }
    } else if kv_layout == Q8Contig::NAME {
        attention_decode_q8_kv(
            ops, stream,
            scratch.q_f16, kv_cache.k_buffer(), kv_cache.v_buffer(),
            scratch.attn_out_f16,
            n_heads, n_kv_heads, head_dim, n_tokens_kv, scale,
        )
        .context("dense attention_decode_q8_kv")?;
    } else {
        let n_tokens_kv_slot = slots.map(|s| s.n_tokens_kv_slot);
        attention_decode_f16_slots(
            ops, stream,
            scratch.q_f16, kv_cache.k_buffer(), kv_cache.v_buffer(),
            scratch.attn_out_f16,
            n_heads, n_kv_heads, head_dim, n_tokens_kv, scale,
            n_tokens_kv_slot,
        )
        .context("dense attention_decode_f16")?;
    }

    // 8. Quantise attn_out directly (NO sigmoid gate in qwen3moe).
    let o_width = n_heads * head_dim;
    quantize_f16_q8_1(ops, stream, scratch.attn_out_f16, scratch.x_q8_1, o_width)
        .context("dense quantize attn_out → Q8_1")?;

    // 9. Output projection.
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != o_width {
        bail!(
            "dense attn_output shape [{o_rows}, {o_k}] != expected [{}, {}]",
            hidden,
            o_width
        );
    }
    mmvq(
        ops, stream, weights.attn_output.ptr, scratch.x_q8_1,
        scratch.mmvq_f32, o_rows, o_k, dtype_o,
    )
    .context("dense mmvq attn_output")?;
    cast_f32_to_f16(ops, stream, scratch.mmvq_f32, delta_out, o_rows)
        .context("dense cast attn_output → f16")?;

    Ok(())
}

/// Route a `LayerCache` entry through the dense-attn forward, pulling the
/// correct `KvCache` out of the enum. Fails if the layer is GDN.
pub fn forward_dense_attn_layer_decode(
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
    slots: Option<AttnDecodeSlots>,
) -> Result<()> {
    let crate::weights::AttnWeights::Dense(d) = &layer_weights.attn else {
        bail!(
            "layer {} weights are not Dense variant (dense attn path)",
            layer_weights.layer_idx
        );
    };
    match layer_cache {
        LayerCache::FullAttn(kv) => forward_dense_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm,
            d, kv, scratch, x_in, delta_out,
            position, slots,
        ),
        LayerCache::FullAttnQ8(kv) => forward_dense_attn_decode(
            ops, stream, device, cfg,
            &layer_weights.attn_norm,
            d, kv, scratch, x_in, delta_out,
            position, slots,
        ),
        LayerCache::Gdn(_) => bail!(
            "layer {} is not a full-attn layer (cache variant mismatch)",
            layer_weights.layer_idx
        ),
    }
}

pub fn forward_dense_attn_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    attn_norm: &DeviceTensor,
    weights: &DenseAttnWeights,
    kv_cache: &mut KvCache<flambeau_runtime::F16Contig, HipDevice>,
    scratch: &mut FullAttnPrefillScratch,
    x_in: DevicePtr,
    delta_out: DevicePtr,
    n_tokens: usize,
    start_position: usize,
    slots: Option<AttnPrefillSlots>,
) -> Result<()> {
    if weights.attn_q_bias.is_some()
        || weights.attn_k_bias.is_some()
        || weights.attn_v_bias.is_some()
    {
        bail!("forward_dense_attn_prefill: Q/K/V biases not yet supported");
    }
    if n_tokens == 0 {
        bail!("forward_dense_attn_prefill called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_dense_attn_prefill: n_tokens={n_tokens} > max_tokens={}",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let q_width = n_heads * head_dim;
    let rope = &cfg.rope;

    // 1. rmsnorm + dual Q8_1 quant (std + MMQ DS4).
    rmsnorm_f16(
        ops, stream, x_in, attn_norm.ptr, scratch.x_norm_f16,
        n_tokens, hidden, cfg.rms_norm_eps,
    )
    .context("dense prefill attn_norm")?;
    quantize_f16_q8_1(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden,
    )
    .context("dense prefill x_norm → Q8_1 std")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens,
    )
    .context("dense prefill x_norm → Q8_1 MMQ")?;

    // 2. Plain Q projection → q_f16 directly.
    let dtype_q = qdtype_of(weights.attn_q.dtype)?;
    let (q_rows, q_k) = mat_shape(&weights.attn_q)?;
    if q_rows != q_width || q_k != hidden {
        bail!(
            "dense prefill attn_q shape [{q_rows}, {q_k}] != expected [{}, {}]",
            q_width, hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_q.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32, n_tokens, q_k, q_rows, dtype_q,
    )
    .context("dense prefill qmatmul attn_q")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.q_f16, n_tokens * q_rows,
    )
    .context("dense prefill cast attn_q → f16")?;

    // 3. K / V projections.
    let dtype_k = qdtype_of(weights.attn_k.dtype)?;
    let (k_rows, k_k) = mat_shape(&weights.attn_k)?;
    if k_rows != n_kv_heads * head_dim || k_k != hidden {
        bail!(
            "dense prefill attn_k shape [{k_rows}, {k_k}] != expected [{}, {}]",
            n_kv_heads * head_dim, hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_k.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32, n_tokens, k_k, k_rows, dtype_k,
    )
    .context("dense prefill qmatmul attn_k")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.k_f16, n_tokens * k_rows,
    )
    .context("dense prefill cast attn_k → f16")?;

    let dtype_v = qdtype_of(weights.attn_v.dtype)?;
    let (v_rows, v_k) = mat_shape(&weights.attn_v)?;
    if v_rows != n_kv_heads * head_dim || v_k != hidden {
        bail!(
            "dense prefill attn_v shape [{v_rows}, {v_k}] != expected [{}, {}]",
            n_kv_heads * head_dim, hidden
        );
    }
    qmatmul(
        ops, stream, weights.attn_v.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.mmvq_f32, n_tokens, v_k, v_rows, dtype_v,
    )
    .context("dense prefill qmatmul attn_v")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, scratch.v_f16, n_tokens * v_rows,
    )
    .context("dense prefill cast attn_v → f16")?;

    // 4. Per-head Q/K rmsnorm.
    let q_norm_dim = weights
        .attn_q_norm.dims.first().copied()
        .context("attn_q_norm missing dim")? as usize;
    if q_norm_dim != head_dim {
        bail!("attn_q_norm dim {q_norm_dim} != head_dim {head_dim}");
    }
    rmsnorm_f16(
        ops, stream, scratch.q_f16, weights.attn_q_norm.ptr,
        scratch.q_f16, n_tokens * n_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("dense prefill attn_q_norm")?;
    rmsnorm_f16(
        ops, stream, scratch.k_f16, weights.attn_k_norm.ptr,
        scratch.k_f16, n_tokens * n_kv_heads, head_dim, cfg.rms_norm_eps,
    )
    .context("dense prefill attn_k_norm")?;

    // 5. RoPE (uniform — qwen3moe has no multi-freq sections).
    upload_positions_range(device, stream, scratch, start_position, n_tokens)?;
    rope_neox_partial_f16(
        ops, stream, scratch.q_f16, scratch.positions,
        rope.freq_base, n_tokens, n_heads, head_dim, rope.rotated_dims,
    )
    .context("dense prefill rope Q")?;
    rope_neox_partial_f16(
        ops, stream, scratch.k_f16, scratch.positions,
        rope.freq_base, n_tokens, n_kv_heads, head_dim, rope.rotated_dims,
    )
    .context("dense prefill rope K")?;

    // 6. KV append.
    if let Some(AttnPrefillSlots { k_append_slot, v_append_slot, .. }) = slots {
        unsafe {
            kv_cache_append_hip_slot(
                kv_cache, device, stream,
                scratch.k_f16, scratch.v_f16, n_tokens,
                k_append_slot, v_append_slot,
            )?;
        }
    } else {
        unsafe {
            kv_cache
                .append(device, stream, scratch.k_f16, scratch.v_f16, n_tokens)
                .map_err(|e| anyhow::anyhow!("dense kv_cache.append(L={n_tokens}): {e}"))?;
        }
    }

    // 7. Causal attention.
    let n_k_tokens = kv_cache.current_tokens();
    let scale = (head_dim as f32).sqrt().recip();
    let (n_k_slot_opt, q_off_slot_opt) = match slots {
        Some(s) => (Some(s.n_k_slot), Some(s.q_off_slot)),
        None => (None, None),
    };
    attention_prefill_f16_slots(
        ops, stream,
        scratch.q_f16, kv_cache.k_buffer(), kv_cache.v_buffer(),
        scratch.attn_out_f16,
        n_tokens, n_heads, n_kv_heads, head_dim,
        n_k_tokens, start_position, scale,
        n_k_slot_opt, q_off_slot_opt,
    )
    .context("dense attention_prefill_f16")?;

    // 8. Quantise attn_out to BOTH Q8_1 layouts for the output projection
    //    (no sigmoid gate).
    quantize_f16_q8_1(
        ops, stream, scratch.attn_out_f16, scratch.gated_q8_1, n_tokens * q_width,
    )
    .context("dense prefill quantise attn_out → Q8_1 std")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.attn_out_f16, scratch.gated_q8_1_mmq, q_width, n_tokens,
    )
    .context("dense prefill quantise attn_out → Q8_1 MMQ")?;

    // 9. Output projection.
    let dtype_o = qdtype_of(weights.attn_output.dtype)?;
    let (o_rows, o_k) = mat_shape(&weights.attn_output)?;
    if o_rows != hidden || o_k != q_width {
        bail!(
            "dense prefill attn_output shape [{o_rows}, {o_k}] != expected [{}, {}]",
            hidden, q_width
        );
    }
    qmatmul(
        ops, stream, weights.attn_output.ptr,
        scratch.gated_q8_1, scratch.gated_q8_1_mmq,
        scratch.mmvq_f32, n_tokens, o_k, o_rows, dtype_o,
    )
    .context("dense prefill qmatmul attn_output")?;
    cast_f32_to_f16(
        ops, stream, scratch.mmvq_f32, delta_out, n_tokens * hidden,
    )
    .context("dense prefill cast attn_output → f16")?;

    Ok(())
}
