//! `DenseMlp` — Llama / Qwen3-style dense FFN block.
//!
//! Pipeline (decode + prefill share the shape; prefill scales by L
//! tokens and uses both Q8_1 layouts for MMQ):
//!
//! 1. `quantize_f16_q8_1(x_norm)` (caller normalised x_in)
//! 2. fused gate+up projection — `mmvq_q8_0_gate_up` when both Q8_0,
//!    else `qmatmul` per branch (auto-dispatches MMVQ/MMQ by m)
//! 3. `swiglu_f32_to_f16(gate, up) → activated_f16`
//! 4. `quantize_f16_q8_1(activated)`
//! 5. down projection `qmatmul`
//! 6. `cast_f32_to_f16(down) → down_f16`, then `add_f16(residual, down) → x_out`
//!
//! The block has no host→device memcpy and no KV-cache plumbing —
//! every step is a kernel launch through `ops: &O`. Scratch fields are
//! all `DevicePtr` (Copy), so the view passes by value.

use anyhow::{bail, Context, Result};
use flambeau_core::device::DevicePtr;
use flambeau_ops::Ops;

use crate::moe_experts::Activation;
use crate::WeightHandle;

/// Borrowed-by-value view of a caller-owned dense FFN decode scratch.
/// All fields are `Copy`; caller (typically `qwen3-moe`) keeps owning
/// the underlying allocations.
#[derive(Copy, Clone)]
pub struct DenseMlpDecodeScratch {
    pub x_q8_1: DevicePtr,        // Q8_1 [hidden / 32]
    pub gate_f32: DevicePtr,      // F32 [intermediate]
    pub up_f32: DevicePtr,        // F32 [intermediate]
    pub activated_f16: DevicePtr, // F16 [intermediate]
    pub activated_q8_1: DevicePtr,// Q8_1 [intermediate / 32]
    pub down_f32: DevicePtr,      // F32 [hidden]
    pub down_f16: DevicePtr,      // F16 [hidden]
}

/// Borrowed-by-value view of a caller-owned dense FFN prefill scratch.
#[derive(Copy, Clone)]
pub struct DenseMlpPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub activated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,
    pub down_f16: DevicePtr,
}

/// Llama / Qwen3-style dense FFN block. Holds three matmul weights
/// (`gate`, `up`, `down`); the input rmsnorm + residual are owned by
/// the caller (since the block also serves as the per-expert path
/// inside `MoeExperts`, where the input norm is upstream).
pub struct DenseMlp {
    pub ffn_gate: WeightHandle,  // [intermediate, hidden]
    pub ffn_up: WeightHandle,    // [intermediate, hidden]
    pub ffn_down: WeightHandle,  // [hidden, intermediate]
    pub hidden: usize,
    pub intermediate: usize,
}

impl DenseMlp {
    pub fn new(
        ffn_gate: WeightHandle,
        ffn_up: WeightHandle,
        ffn_down: WeightHandle,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        if ffn_gate.dims != [intermediate, hidden] {
            bail!(
                "ffn_gate dims {:?} != expected [{}, {}]",
                ffn_gate.dims,
                intermediate,
                hidden
            );
        }
        if ffn_up.dims != [intermediate, hidden] {
            bail!(
                "ffn_up dims {:?} != expected [{}, {}]",
                ffn_up.dims,
                intermediate,
                hidden
            );
        }
        if ffn_down.dims != [hidden, intermediate] {
            bail!(
                "ffn_down dims {:?} != expected [{}, {}]",
                ffn_down.dims,
                hidden,
                intermediate,
            );
        }
        Ok(Self { ffn_gate, ffn_up, ffn_down, hidden, intermediate })
    }

    /// Single-token decode through the dense FFN. The caller is
    /// responsible for the input rmsnorm; `x_norm` is the F16 normed
    /// activation. Writes `x_out = residual + down(swiglu(gate, up))`.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        x_out: DevicePtr,
        scratch: DenseMlpDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;

        // 1. Quantise x_norm → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("dense ffn x_norm → Q8_1")?;

        // 2+3. gate + up. Fuse when both Q8_0 (Qwen3.6 dense path).
        let fuse_gate_up = self.ffn_gate.dtype == flambeau_core::op::QDtype::Q8_0
            && self.ffn_up.dtype == flambeau_core::op::QDtype::Q8_0;
        if fuse_gate_up {
            ops.mmvq_q8_0_gate_up(
                self.ffn_gate.ptr,
                self.ffn_up.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("dense ffn gate+up fused mmvq_q8_0")?;
        } else {
            ops.qmatmul(
                self.ffn_gate.ptr,
                scratch.x_q8_1,
                DevicePtr(0),
                scratch.gate_f32,
                1,
                hidden,
                inter,
                self.ffn_gate.dtype,
            )
            .context("dense ffn gate qmatmul")?;
            ops.qmatmul(
                self.ffn_up.ptr,
                scratch.x_q8_1,
                DevicePtr(0),
                scratch.up_f32,
                1,
                hidden,
                inter,
                self.ffn_up.dtype,
            )
            .context("dense ffn up qmatmul")?;
        }

        // 4+5. Fused SwiGLU → F16 + Q8_1 quantise.
        ops.swiglu_f32_to_f16(scratch.gate_f32, scratch.up_f32, scratch.activated_f16, inter)
            .context("dense ffn swiglu_f32_to_f16")?;
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, inter)
            .context("dense ffn quantise activated → Q8_1")?;

        // 6. down matmul.
        ops.qmatmul(
            self.ffn_down.ptr,
            scratch.activated_q8_1,
            DevicePtr(0),
            scratch.down_f32,
            1,
            inter,
            hidden,
            self.ffn_down.dtype,
        )
        .context("dense ffn down qmatmul")?;

        // 7. Cast + residual add.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, hidden)
            .context("dense ffn cast down → f16")?;
        ops.add_f16(residual, scratch.down_f16, x_out, hidden)
            .context("dense ffn residual: residual + down")?;
        Ok(())
    }

    /// Multi-token prefill. Same pipeline at L tokens; uses both Q8_1
    /// layouts for the gate/up/down qmatmul auto-dispatch into
    /// MMQ-turbo at m ≥ 128.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        residual: DevicePtr,
        x_out: DevicePtr,
        scratch: DenseMlpPrefillScratch,
        n_tokens: usize,
    ) -> Result<()> {
        if n_tokens == 0 {
            bail!("DenseMlp::forward_prefill called with n_tokens = 0");
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "DenseMlp::forward_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
                scratch.max_tokens
            );
        }
        let hidden = self.hidden;
        let inter = self.intermediate;

        // 1. Quantise x_norm → BOTH Q8_1 layouts.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, n_tokens * hidden)
            .context("dense ffn prefill x_norm → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("dense ffn prefill x_norm → Q8_1 (MMQ DS4)")?;

        // 2+3. gate + up qmatmul (auto MMVQ/MMQ).
        ops.qmatmul(
            self.ffn_gate.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.gate_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_gate.dtype,
        )
        .context("dense ffn prefill gate qmatmul")?;
        ops.qmatmul(
            self.ffn_up.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.up_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_up.dtype,
        )
        .context("dense ffn prefill up qmatmul")?;

        // 4+5. Fused SwiGLU → F16 + dual Q8_1 quantise.
        ops.swiglu_f32_to_f16(
            scratch.gate_f32,
            scratch.up_f32,
            scratch.activated_f16,
            n_tokens * inter,
        )
        .context("dense ffn prefill swiglu_f32_to_f16")?;
        ops.quantize_f16_q8_1(
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_tokens * inter,
        )
        .context("dense ffn prefill quantise activated → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(
            scratch.activated_f16,
            scratch.activated_q8_1_mmq,
            inter,
            n_tokens,
        )
        .context("dense ffn prefill quantise activated → Q8_1 (MMQ DS4)")?;

        // 6. down matmul.
        ops.qmatmul(
            self.ffn_down.ptr,
            scratch.activated_q8_1,
            scratch.activated_q8_1_mmq,
            scratch.down_f32,
            n_tokens,
            inter,
            hidden,
            self.ffn_down.dtype,
        )
        .context("dense ffn prefill down qmatmul")?;

        // 7. Cast + residual add.
        ops.cast_f32_to_f16(scratch.down_f32, scratch.down_f16, n_tokens * hidden)
            .context("dense ffn prefill cast down → f16")?;
        ops.add_f16(residual, scratch.down_f16, x_out, n_tokens * hidden)
            .context("dense ffn prefill residual: residual + down")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DenseMlpTp — tensor-parallel dense FFN
// ---------------------------------------------------------------------------

/// Tensor-parallel dense FFN, per-rank slice. Each rank owns:
/// - `ffn_gate: WeightHandle` shape `[intermediate_local, hidden]` (column-parallel)
/// - `ffn_up:   WeightHandle` shape `[intermediate_local, hidden]` (column-parallel)
/// - `ffn_down: WeightHandle` shape `[hidden, intermediate_local]` (row-parallel)
///
/// `intermediate_local = intermediate_full / tp_world`. The caller is
/// responsible for the slice and for the AR-sum that folds per-rank
/// `partial_ffn_out` contributions across ranks (and typically also
/// adds the residual stream as part of that fold).
///
/// Pipeline mirrors [`DenseMlp::forward_decode`] except:
/// - works on the `_local` intermediate dimension everywhere
/// - emits a per-rank partial (no residual add)
/// - activation is parameterised (qwen3.x → SwiGLU, gemma4 → Gelu)
/// - supports a `pre_quantized` skip for upstream fused AR+RMSNorm+Q8_1
pub struct DenseMlpTp {
    pub ffn_gate: WeightHandle,
    pub ffn_up: WeightHandle,
    pub ffn_down: WeightHandle,
    pub hidden: usize,
    pub intermediate_local: usize,
    pub activation: Activation,
}

impl DenseMlpTp {
    pub fn new(
        ffn_gate: WeightHandle,
        ffn_up: WeightHandle,
        ffn_down: WeightHandle,
        hidden: usize,
        intermediate_local: usize,
        activation: Activation,
    ) -> Result<Self> {
        if ffn_gate.dims != [intermediate_local, hidden] {
            bail!(
                "DenseMlpTp: ffn_gate dims {:?} != [{}, {}]",
                ffn_gate.dims, intermediate_local, hidden
            );
        }
        if ffn_up.dims != [intermediate_local, hidden] {
            bail!(
                "DenseMlpTp: ffn_up dims {:?} != [{}, {}]",
                ffn_up.dims, intermediate_local, hidden
            );
        }
        if ffn_down.dims != [hidden, intermediate_local] {
            bail!(
                "DenseMlpTp: ffn_down dims {:?} != [{}, {}]",
                ffn_down.dims, hidden, intermediate_local
            );
        }
        Ok(Self {
            ffn_gate,
            ffn_up,
            ffn_down,
            hidden,
            intermediate_local,
            activation,
        })
    }

    /// Per-rank decode. `x_norm` is the AR'd post-attn hidden state
    /// (replicated across ranks); when `pre_quantized` is true, the
    /// caller has already populated `scratch.x_q8_1` via the upstream
    /// fused AR+RMSNorm+Q8_1 kernel and `x_norm` is unused.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_ffn_out: DevicePtr,
        scratch: DenseMlpDecodeScratch,
        pre_quantized: bool,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate_local;

        // 1. Quantise x_norm → Q8_1 (skip when upstream already did it).
        if !pre_quantized {
            ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
                .context("DenseMlpTp x_norm → Q8_1")?;
        } else {
            let _ = x_norm;
        }

        // 2+3. gate + up — fused on Q8_0 / Q4_0 / Q4_1, unfused otherwise.
        let gate_dt = self.ffn_gate.dtype;
        let up_dt = self.ffn_up.dtype;
        let fuse_q8 = gate_dt == flambeau_core::op::QDtype::Q8_0
            && up_dt == flambeau_core::op::QDtype::Q8_0;
        let fuse_q4_0 = gate_dt == flambeau_core::op::QDtype::Q4_0
            && up_dt == flambeau_core::op::QDtype::Q4_0;
        let fuse_q4_1 = gate_dt == flambeau_core::op::QDtype::Q4_1
            && up_dt == flambeau_core::op::QDtype::Q4_1;
        if fuse_q8 {
            ops.mmvq_q8_0_gate_up(
                self.ffn_gate.ptr,
                self.ffn_up.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("DenseMlpTp gate+up fused mmvq_q8_0")?;
        } else if fuse_q4_0 {
            ops.mmvq_q4_0_gate_up_t128(
                self.ffn_gate.ptr,
                self.ffn_up.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("DenseMlpTp gate+up fused mmvq_q4_0_t128")?;
        } else if fuse_q4_1 {
            ops.mmvq_q4_1_gate_up(
                self.ffn_gate.ptr,
                self.ffn_up.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("DenseMlpTp gate+up fused mmvq_q4_1")?;
        } else {
            ops.qmatmul(
                self.ffn_gate.ptr,
                scratch.x_q8_1,
                DevicePtr(0),
                scratch.gate_f32,
                1,
                hidden,
                inter,
                gate_dt,
            )
            .context("DenseMlpTp gate qmatmul")?;
            ops.qmatmul(
                self.ffn_up.ptr,
                scratch.x_q8_1,
                DevicePtr(0),
                scratch.up_f32,
                1,
                hidden,
                inter,
                up_dt,
            )
            .context("DenseMlpTp up qmatmul")?;
        }

        // 4. Activation: SwiGLU(gate, up) or Gelu(gate)*up → F16.
        match self.activation {
            Activation::SwiGLU => ops
                .swiglu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    inter,
                )
                .context("DenseMlpTp swiglu_f32_to_f16")?,
            Activation::Gelu => ops
                .gelu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    inter,
                )
                .context("DenseMlpTp gelu_f32_to_f16")?,
        }

        // 5. Quantise activated → Q8_1.
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, inter)
            .context("DenseMlpTp activated → Q8_1")?;

        // 6. Row-parallel down matmul.
        ops.qmatmul(
            self.ffn_down.ptr,
            scratch.activated_q8_1,
            DevicePtr(0),
            scratch.down_f32,
            1,
            inter,
            hidden,
            self.ffn_down.dtype,
        )
        .context("DenseMlpTp down qmatmul")?;

        // 7. Cast F32→F16 directly into partial_ffn_out. The caller's
        // AR fold sums the per-rank partials (and adds the residual).
        ops.cast_f32_to_f16(scratch.down_f32, partial_ffn_out, hidden)
            .context("DenseMlpTp cast down → partial_ffn_out")?;
        Ok(())
    }

    /// Multi-token TP prefill. Mirrors [`DenseMlp::forward_prefill`]
    /// but on the local intermediate slab; emits to `partial_ffn_out`.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        partial_ffn_out: DevicePtr,
        scratch: DenseMlpPrefillScratch,
        n_tokens: usize,
    ) -> Result<()> {
        if n_tokens == 0 {
            bail!("DenseMlpTp::forward_prefill called with n_tokens = 0");
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "DenseMlpTp::forward_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
                scratch.max_tokens
            );
        }
        let hidden = self.hidden;
        let inter = self.intermediate_local;

        // 1. Quantise x_norm → BOTH Q8_1 layouts.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, n_tokens * hidden)
            .context("DenseMlpTp prefill x_norm → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("DenseMlpTp prefill x_norm → Q8_1 (MMQ DS4)")?;

        // 2+3. gate + up qmatmul (auto MMVQ/MMQ).
        ops.qmatmul(
            self.ffn_gate.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.gate_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_gate.dtype,
        )
        .context("DenseMlpTp prefill gate qmatmul")?;
        ops.qmatmul(
            self.ffn_up.ptr,
            scratch.x_q8_1,
            scratch.x_q8_1_mmq,
            scratch.up_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_up.dtype,
        )
        .context("DenseMlpTp prefill up qmatmul")?;

        // 4. Activation.
        let act_elems = n_tokens * inter;
        match self.activation {
            Activation::SwiGLU => ops
                .swiglu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    act_elems,
                )
                .context("DenseMlpTp prefill swiglu_f32_to_f16")?,
            Activation::Gelu => ops
                .gelu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    act_elems,
                )
                .context("DenseMlpTp prefill gelu_f32_to_f16")?,
        }

        // 5. Quantise activated → BOTH Q8_1 layouts.
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, act_elems)
            .context("DenseMlpTp prefill activated → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(scratch.activated_f16, scratch.activated_q8_1_mmq, inter, n_tokens)
            .context("DenseMlpTp prefill activated → Q8_1 (MMQ DS4)")?;

        // 6. Row-parallel down matmul.
        ops.qmatmul(
            self.ffn_down.ptr,
            scratch.activated_q8_1,
            scratch.activated_q8_1_mmq,
            scratch.down_f32,
            n_tokens,
            inter,
            hidden,
            self.ffn_down.dtype,
        )
        .context("DenseMlpTp prefill down qmatmul")?;

        // 7. Cast F32→F16 directly into partial_ffn_out.
        ops.cast_f32_to_f16(scratch.down_f32, partial_ffn_out, n_tokens * hidden)
            .context("DenseMlpTp prefill cast down → partial_ffn_out")?;
        Ok(())
    }
}
