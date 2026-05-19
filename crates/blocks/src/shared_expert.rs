//! `SharedExpert` — the per-token-sigmoid-gated dense FFN that
//! Qwen3-MoE arches add alongside the routed experts. Output is the
//! scaled FFN result; the caller folds it into the residual stream
//! via `moe_combine_two_residuals_f16` (or a separate `add_f16`).
//!
//! Pipeline:
//!
//! 1. `quantize_f16_q8_1(x_norm)`
//! 2-3. gate + up (fused `mmvq_q8_0_gate_up` when both Q8_0, else two `mmvq`)
//! 4-5. fused `swiglu_f32_to_q8_1` when `intermediate % 32 == 0`,
//!      else `swiglu_f32_to_f16` + `quantize_f16_q8_1`
//! 6. down `mmvq` → F32
//! 7. `cast_f16_to_f32(x_norm)` + `shared_expert_scale_f32` (in place)
//! 8. `cast_f32_to_f16` → `shared_out`

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;
use flambeau_ops::Ops;

use crate::driver_utils::RawAllocTracker;
use crate::moe_experts::Activation;
use crate::WeightHandle;

#[derive(Copy, Clone)]
pub struct SharedExpertDecodeScratch {
    pub x_q8_1: DevicePtr,        // Q8_1 [hidden / 32]
    pub gate_f32: DevicePtr,      // F32 [intermediate]
    pub up_f32: DevicePtr,        // F32 [intermediate]
    pub activated_f16: DevicePtr, // F16 [intermediate] (only used when fused-swiglu path is off)
    pub activated_q8_1: DevicePtr, // Q8_1 [intermediate / 32]
    pub down_f32: DevicePtr,      // F32 [hidden] — scaled in place
    pub x_norm_f32: DevicePtr,    // F32 [hidden] — gate-scale uses this
}

#[derive(Copy, Clone)]
pub struct SharedExpertPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,         // Q8_1 [max_tokens * hidden / 32]
    pub gate_f32: DevicePtr,       // F32 [max_tokens, intermediate]
    pub up_f32: DevicePtr,         // F32 [max_tokens, intermediate]
    pub activated_f16: DevicePtr,  // F16 [max_tokens, intermediate]
    pub activated_q8_1: DevicePtr, // Q8_1 [max_tokens * intermediate / 32]
    pub down_f32: DevicePtr,       // F32 [max_tokens, hidden] — scaled in place
    pub x_norm_f32: DevicePtr,     // F32 [max_tokens, hidden]
}

/// Shape inputs needed to size a `SharedExpert` decode scratch.
#[derive(Copy, Clone, Debug)]
pub struct SharedExpertScratchDims {
    pub hidden: usize,
    pub intermediate: usize,
}

/// Owned SharedExpert decode scratch.
pub struct OwnedSharedExpertDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub x_norm_f32: DevicePtr,
}

impl OwnedSharedExpertDecodeScratch {
    pub fn view(&self) -> SharedExpertDecodeScratch {
        SharedExpertDecodeScratch {
            x_q8_1: self.x_q8_1,
            gate_f32: self.gate_f32,
            up_f32: self.up_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            down_f32: self.down_f32,
            x_norm_f32: self.x_norm_f32,
        }
    }
}

/// Owned SharedExpert prefill scratch.
pub struct OwnedSharedExpertPrefillScratch {
    pub max_tokens: usize,
    pub x_q8_1: DevicePtr,
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub activated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,
    pub x_norm_f32: DevicePtr,
}

impl OwnedSharedExpertPrefillScratch {
    pub fn view(&self) -> SharedExpertPrefillScratch {
        SharedExpertPrefillScratch {
            max_tokens: self.max_tokens,
            x_q8_1: self.x_q8_1,
            gate_f32: self.gate_f32,
            up_f32: self.up_f32,
            activated_f16: self.activated_f16,
            activated_q8_1: self.activated_q8_1,
            down_f32: self.down_f32,
            x_norm_f32: self.x_norm_f32,
        }
    }
}

pub struct SharedExpert {
    /// F16 `[hidden]` per-token gate weights. `Some` for qwen3next-style
    /// shared experts (sigmoid-scaled output); `None` for qwen35moe-style
    /// (plain down output, no per-token gate). When `None`, step 7 of
    /// the pipeline (`shared_expert_scale_f32`) is skipped and the
    /// `x_norm_f32` scratch field is unused.
    pub ffn_gate_inp_shexp: Option<DevicePtr>,
    pub ffn_gate_shexp: WeightHandle,  // [intermediate, hidden]
    pub ffn_up_shexp: WeightHandle,    // [intermediate, hidden]
    pub ffn_down_shexp: WeightHandle,  // [hidden, intermediate]
    pub hidden: usize,
    pub intermediate: usize,
    /// Activation between gate/up and down. Default `SwiGLU` (qwen3.x
    /// shared expert); gemma4 MoE shared MLP uses `Gelu`.
    pub activation: Activation,
}

impl SharedExpert {
    pub fn new(
        ffn_gate_inp_shexp: Option<DevicePtr>,
        ffn_gate_shexp: WeightHandle,
        ffn_up_shexp: WeightHandle,
        ffn_down_shexp: WeightHandle,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        if ffn_gate_shexp.dims != [intermediate, hidden] {
            bail!(
                "ffn_gate_shexp dims {:?} != [{}, {}]",
                ffn_gate_shexp.dims,
                intermediate,
                hidden
            );
        }
        if ffn_up_shexp.dims != [intermediate, hidden] {
            bail!(
                "ffn_up_shexp dims {:?} != [{}, {}]",
                ffn_up_shexp.dims,
                intermediate,
                hidden
            );
        }
        if ffn_down_shexp.dims != [hidden, intermediate] {
            bail!(
                "ffn_down_shexp dims {:?} != [{}, {}]",
                ffn_down_shexp.dims,
                hidden,
                intermediate
            );
        }
        Ok(Self {
            ffn_gate_inp_shexp,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
            hidden,
            intermediate,
            activation: Activation::SwiGLU,
        })
    }

    /// Override the activation. Default is `SwiGLU`; gemma4 sets `Gelu`.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    pub fn scratch_dims(&self) -> SharedExpertScratchDims {
        SharedExpertScratchDims { hidden: self.hidden, intermediate: self.intermediate }
    }

    /// Allocate an [`OwnedSharedExpertDecodeScratch`] sized for `dims`.
    pub fn alloc_decode_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: SharedExpertScratchDims,
    ) -> Result<OwnedSharedExpertDecodeScratch> {
        let SharedExpertScratchDims { hidden, intermediate } = dims;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, hidden)?;
        let (gate_f32, _) = tracker.alloc_f32(device, intermediate)?;
        let (up_f32, _) = tracker.alloc_f32(device, intermediate)?;
        let (activated_f16, _) = tracker.alloc_f16(device, intermediate)?;
        let (activated_q8_1, _) = tracker.alloc_q8_1(device, intermediate)?;
        let (down_f32, _) = tracker.alloc_f32(device, hidden)?;
        let (x_norm_f32, _) = tracker.alloc_f32(device, hidden)?;
        Ok(OwnedSharedExpertDecodeScratch {
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
        })
    }

    /// Allocate an [`OwnedSharedExpertPrefillScratch`] sized for
    /// `dims` × `max_tokens`.
    pub fn alloc_prefill_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: SharedExpertScratchDims,
        max_tokens: usize,
    ) -> Result<OwnedSharedExpertPrefillScratch> {
        if max_tokens == 0 {
            bail!("alloc_prefill_scratch: max_tokens must be >= 1");
        }
        let SharedExpertScratchDims { hidden, intermediate } = dims;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * hidden)?;
        let (gate_f32, _) = tracker.alloc_f32(device, max_tokens * intermediate)?;
        let (up_f32, _) = tracker.alloc_f32(device, max_tokens * intermediate)?;
        let (activated_f16, _) = tracker.alloc_f16(device, max_tokens * intermediate)?;
        let (activated_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * intermediate)?;
        let (down_f32, _) = tracker.alloc_f32(device, max_tokens * hidden)?;
        let (x_norm_f32, _) = tracker.alloc_f32(device, max_tokens * hidden)?;
        Ok(OwnedSharedExpertPrefillScratch {
            max_tokens,
            x_q8_1,
            gate_f32,
            up_f32,
            activated_f16,
            activated_q8_1,
            down_f32,
            x_norm_f32,
        })
    }

    /// One decode step. Reads `x_norm` (F16 `[hidden]`); writes the
    /// per-token-scaled FFN output to `shared_out` (F16 `[hidden]`).
    /// Caller composes `shared_out` with the residual + routed-expert
    /// output via `moe_combine_two_residuals_f16`.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        shared_out: DevicePtr,
        scratch: SharedExpertDecodeScratch,
    ) -> Result<()> {
        let hidden = self.hidden;
        let inter = self.intermediate;

        // 1. Quantise x_norm → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, hidden)
            .context("shexp x_norm → Q8_1")?;

        // 2+3. gate + up. Fuse when both Q8_0 (mmvq_q8_0_gate_up) or
        // both Q4_0 (mmvq_q4_0_gate_up_t128, t128 schedule matches
        // legacy's fast path); else two plain mmvq launches.
        let fuse_q8_0 = self.ffn_gate_shexp.dtype == QDtype::Q8_0
            && self.ffn_up_shexp.dtype == QDtype::Q8_0;
        let fuse_q4_0 = self.ffn_gate_shexp.dtype == QDtype::Q4_0
            && self.ffn_up_shexp.dtype == QDtype::Q4_0;
        if fuse_q8_0 {
            ops.mmvq_q8_0_gate_up(
                self.ffn_gate_shexp.ptr,
                self.ffn_up_shexp.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("shexp gate+up fused mmvq_q8_0")?;
        } else if fuse_q4_0 {
            ops.mmvq_q4_0_gate_up_t128(
                self.ffn_gate_shexp.ptr,
                self.ffn_up_shexp.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                scratch.up_f32,
                inter,
                inter,
                hidden,
            )
            .context("shexp gate+up fused mmvq_q4_0_t128")?;
        } else {
            ops.mmvq(
                self.ffn_gate_shexp.ptr,
                scratch.x_q8_1,
                scratch.gate_f32,
                inter,
                hidden,
                self.ffn_gate_shexp.dtype,
            )
            .context("shexp gate mmvq")?;
            ops.mmvq(
                self.ffn_up_shexp.ptr,
                scratch.x_q8_1,
                scratch.up_f32,
                inter,
                hidden,
                self.ffn_up_shexp.dtype,
            )
            .context("shexp up mmvq")?;
        }

        // 4+5. activation(gate, up) → F16, then quantise to Q8_1.
        // SwiGLU has a fused `swiglu_f32_to_q8_1` when intermediate is
        // a multiple of QK8_1 (32); Gelu has no fused variant.
        match self.activation {
            Activation::SwiGLU if inter % 32 == 0 => {
                ops.swiglu_f32_to_q8_1(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_q8_1,
                    inter,
                )
                .context("shexp swiglu_f32_to_q8_1")?;
            }
            Activation::SwiGLU => {
                ops.swiglu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    inter,
                )
                .context("shexp swiglu_f32_to_f16")?;
                ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, inter)
                    .context("shexp quantise activated → Q8_1")?;
            }
            Activation::Gelu => {
                ops.gelu_f32_to_f16(
                    scratch.gate_f32,
                    scratch.up_f32,
                    scratch.activated_f16,
                    inter,
                )
                .context("shexp gelu_f32_to_f16")?;
                ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, inter)
                    .context("shexp quantise activated → Q8_1")?;
            }
        }

        // 6. down matmul → F32.
        ops.mmvq(
            self.ffn_down_shexp.ptr,
            scratch.activated_q8_1,
            scratch.down_f32,
            hidden,
            inter,
            self.ffn_down_shexp.dtype,
        )
        .context("shexp down mmvq")?;

        // 7. Optional per-token sigmoid gate scaling on `down_f32`
        // (qwen3next pattern). qwen35moe leaves this off — the down
        // output passes straight to the F16 cast.
        if let Some(gate_w) = self.ffn_gate_inp_shexp {
            ops.cast_f16_to_f32(x_norm, scratch.x_norm_f32, hidden)
                .context("shexp cast x_norm → f32")?;
            ops.shared_expert_scale_f32(
                scratch.down_f32,
                scratch.x_norm_f32,
                gate_w,
                1,
                hidden,
            )
            .context("shexp shared_expert_scale_f32")?;
        }

        // 8. Cast (scaled) output back to F16.
        ops.cast_f32_to_f16(scratch.down_f32, shared_out, hidden)
            .context("shexp cast → f16")
    }

    /// Multi-token prefill (also serves the per-rank TP shared-expert
    /// path — `self.intermediate` set to `local_inter`). Same pipeline
    /// as `forward_decode` extended over `n_tokens`. Uses single-layout
    /// Q8_1 + auto-dispatching `qmatmul` (the dense FFN's dual Q8_1 +
    /// fused gate_up isn't useful here at typical L × shared-inter
    /// sizes; the qwen3-moe shared-expert TP cert never engaged the
    /// MMQ DS4 path). Bails if any quant-only weight isn't supported.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        x_norm: DevicePtr,
        shared_out: DevicePtr,
        n_tokens: usize,
        scratch: SharedExpertPrefillScratch,
    ) -> Result<()> {
        if n_tokens == 0 {
            bail!("SharedExpert::forward_prefill called with n_tokens = 0");
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "SharedExpert::forward_prefill: n_tokens={n_tokens} > scratch.max_tokens={}",
                scratch.max_tokens
            );
        }
        let hidden = self.hidden;
        let inter = self.intermediate;

        // 1. Quantise x_norm[L, hidden] → Q8_1.
        ops.quantize_f16_q8_1(x_norm, scratch.x_q8_1, n_tokens * hidden)
            .context("shexp prefill x_norm → Q8_1")?;

        // 2-3. gate + up qmatmul (auto MMVQ / MMQ dispatch by m=n_tokens).
        ops.qmatmul(
            self.ffn_gate_shexp.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.gate_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_gate_shexp.dtype,
        )
        .context("shexp prefill gate qmatmul")?;
        ops.qmatmul(
            self.ffn_up_shexp.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.up_f32,
            n_tokens,
            hidden,
            inter,
            self.ffn_up_shexp.dtype,
        )
        .context("shexp prefill up qmatmul")?;

        // 4-5. activation → F16, then quantise to Q8_1.
        let n_total = n_tokens * inter;
        match self.activation {
            Activation::SwiGLU => ops
                .swiglu_f32_to_f16(scratch.gate_f32, scratch.up_f32, scratch.activated_f16, n_total)
                .context("shexp prefill swiglu_f32_to_f16")?,
            Activation::Gelu => ops
                .gelu_f32_to_f16(scratch.gate_f32, scratch.up_f32, scratch.activated_f16, n_total)
                .context("shexp prefill gelu_f32_to_f16")?,
        }
        ops.quantize_f16_q8_1(scratch.activated_f16, scratch.activated_q8_1, n_total)
            .context("shexp prefill activated → Q8_1")?;

        // 6. down qmatmul.
        ops.qmatmul(
            self.ffn_down_shexp.ptr,
            scratch.activated_q8_1,
            DevicePtr(0),
            scratch.down_f32,
            n_tokens,
            inter,
            hidden,
            self.ffn_down_shexp.dtype,
        )
        .context("shexp prefill down qmatmul")?;

        // 7. Optional per-token sigmoid gate scaling on the F32 down
        // output. Skipped for qwen35moe; applied for qwen3next.
        if let Some(gate_w) = self.ffn_gate_inp_shexp {
            ops.cast_f16_to_f32(x_norm, scratch.x_norm_f32, n_tokens * hidden)
                .context("shexp prefill cast x_norm → f32 (gate)")?;
            ops.shared_expert_scale_f32(
                scratch.down_f32,
                scratch.x_norm_f32,
                gate_w,
                n_tokens,
                hidden,
            )
            .context("shexp prefill shared_expert_scale_f32")?;
        }

        // 8. Cast scaled output back to F16.
        ops.cast_f32_to_f16(scratch.down_f32, shared_out, n_tokens * hidden)
            .context("shexp prefill cast → f16")
    }
}
