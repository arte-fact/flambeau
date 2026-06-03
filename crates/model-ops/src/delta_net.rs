//! `DeltaNetLayer` — Qwen3.5-Next / Qwen3.6 hybrid-recurrent (GDN) block.
//!
//! Decode + prefill share the same 17-step shape; prefill scales L
//! tokens through. The recurrent state and conv1d history are passed
//! in by pointer per call; the block mutates them in-place. F32
//! precision end-to-end from the post-MMVQ cast through the state
//! update, ssm_norm, and gated output (matches candle's
//! `delta_net.rs` precision).
//!
//! Decode pipeline (single token):
//! - fused rmsnorm(x_in) + Q8_1 quant
//! - `attn_qkv` + `attn_gate` (fused mmvq_q8_0 / mmvq_q4_0_gate_up when
//!   both same dtype, else separate mmvq per branch)
//! - `ssm_alpha` + `ssm_beta` (fused mmvq_q8_0 if both Q8_0, else separate)
//! - assemble conv input + causal conv1d + shift conv history
//! - silu(conv_out)
//! - slice silu_out into Q|K|V (pointer arithmetic, no kernel)
//! - l2_norm Q + l2_norm K (per-head)
//! - scale Q by 1 / sqrt(head_k_dim)
//! - fused gdn_state_step_alphabeta
//! - ssm_norm (rmsnorm_f32 on state_out)
//! - fused swiglu(z, out_normed) → Q8_1 (or unfused pair if d_inner is
//!   not a multiple of 32)
//! - ssm_out projection (mmvq)
//! - cast F32 → F16 → delta_out
//!
//! Prefill differs in shape, not pipeline:
//!  * `qmatmul` (auto-dispatching MMVQ/MMQ) replaces `mmvq` for the
//!    four projections; both Q8_1 layouts are quantised so M ≥ 128
//!    routes to the 4-warp MMQ kernels.
//!  * `assemble_conv_input_prefill` packs `[history (K-1) + qkv (L)]`
//!    rows; `causal_conv1d_f32(n_tokens=L)` consumes them; the
//!    history shift copies the last K-1 rows back.
//!  * `gdn_split_qkv_f32` replaces the Q|K|V pointer slice (the slice
//!    is too sparse to vectorise at L tokens).
//!  * The fused `swiglu_f32_to_q8_1` is replaced by `swiglu_f32` +
//!    dual `quantize_q8_1` / `quantize_q8_1_mmq` — the ssm_out matmul
//!    auto-dispatches MMVQ/MMQ, so both Q8_1 layouts must be live.
//!  * `ssm_out` goes through `qmatmul` (MMVQ at L=1 falls through to
//!    MMQ at L≥128).

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipEvent, HipStream};
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::Ops;

use crate::driver_utils::RawAllocTracker;
use crate::weight_handle::WeightHandle;

/// Borrowed-by-value view over a caller-owned GDN decode scratch.
#[derive(Copy, Clone)]
pub struct DeltaNetLayerDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub qkv_mixed_f32: DevicePtr, // [conv_channels]
    pub z_f32: DevicePtr,         // [d_inner]
    pub alpha_f32: DevicePtr,     // [num_v_heads]
    pub beta_f32: DevicePtr,      // [num_v_heads]
    pub conv_input: DevicePtr,    // [conv_kernel, conv_channels]
    pub conv_out: DevicePtr,      // [conv_channels]
    pub silu_out: DevicePtr,      // [conv_channels]
    pub q_norm_f32: DevicePtr,    // [num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,    // [num_k_heads, head_k_dim]
    pub state_out: DevicePtr,     // [num_v_heads, head_v_dim]
    pub out_normed: DevicePtr,    // [num_v_heads, head_v_dim]
    pub gated_f32: DevicePtr,     // [d_inner]
    pub gated_q8_1: DevicePtr,    // Q8_1 [d_inner / 32]
    pub ssm_out_f32: DevicePtr,   // [hidden]
}

/// Shape inputs needed to size a `DeltaNetLayer` decode scratch.
#[derive(Copy, Clone, Debug)]
pub struct DeltaNetScratchDims {
    pub hidden: usize,
    pub d_inner: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

/// Owned GDN decode scratch.
pub struct OwnedDeltaNetLayerDecodeScratch {
    pub x_q8_1: DevicePtr,
    pub qkv_mixed_f32: DevicePtr,
    pub z_f32: DevicePtr,
    pub alpha_f32: DevicePtr,
    pub beta_f32: DevicePtr,
    pub conv_input: DevicePtr,
    pub conv_out: DevicePtr,
    pub silu_out: DevicePtr,
    pub q_norm_f32: DevicePtr,
    pub k_norm_f32: DevicePtr,
    pub state_out: DevicePtr,
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub ssm_out_f32: DevicePtr,
}

/// Borrowed view over a caller-owned GDN batched-slots decode scratch.
/// All buffers sized `max_slots ×` the per-slot decode shape; slot-major
/// `[N, ...]` layout. The conv trio + state-step batched-slots kernels
/// consume q/k/v/alpha/beta/qkv_mixed/conv_out/state_out in this layout.
#[derive(Copy, Clone)]
pub struct DeltaNetLayerDecodeBatchedScratch {
    pub max_slots: usize,
    pub x_q8_1: DevicePtr,        // [N, hidden] Q8_1
    pub qkv_mixed_f32: DevicePtr, // [N, conv_channels]
    pub z_f32: DevicePtr,         // [N, d_inner]
    pub alpha_f32: DevicePtr,     // [N, num_v_heads]
    pub beta_f32: DevicePtr,      // [N, num_v_heads]
    pub conv_out: DevicePtr,      // [N, conv_channels]
    pub silu_out: DevicePtr,      // [N, conv_channels]
    pub q_norm_f32: DevicePtr,    // [N, num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,    // [N, num_k_heads, head_k_dim]
    pub state_out: DevicePtr,     // [N, num_v_heads, head_v_dim]
    pub out_normed: DevicePtr,    // [N, num_v_heads, head_v_dim]
    pub gated_f32: DevicePtr,     // [N, d_inner]
    pub gated_q8_1: DevicePtr,    // [N] Q8_1 d_inner
    pub ssm_out_f32: DevicePtr,   // [N, hidden]
}

/// Owned GDN batched-slots decode scratch. Allocated once per session
/// at `max_slots` capacity; the call passes `n_slots ≤ max_slots`.
pub struct OwnedDeltaNetLayerDecodeBatchedScratch {
    pub max_slots: usize,
    pub x_q8_1: DevicePtr,
    pub qkv_mixed_f32: DevicePtr,
    pub z_f32: DevicePtr,
    pub alpha_f32: DevicePtr,
    pub beta_f32: DevicePtr,
    pub conv_out: DevicePtr,
    pub silu_out: DevicePtr,
    pub q_norm_f32: DevicePtr,
    pub k_norm_f32: DevicePtr,
    pub state_out: DevicePtr,
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub ssm_out_f32: DevicePtr,
}

impl OwnedDeltaNetLayerDecodeBatchedScratch {
    pub fn view(&self) -> DeltaNetLayerDecodeBatchedScratch {
        DeltaNetLayerDecodeBatchedScratch {
            max_slots: self.max_slots,
            x_q8_1: self.x_q8_1,
            qkv_mixed_f32: self.qkv_mixed_f32,
            z_f32: self.z_f32,
            alpha_f32: self.alpha_f32,
            beta_f32: self.beta_f32,
            conv_out: self.conv_out,
            silu_out: self.silu_out,
            q_norm_f32: self.q_norm_f32,
            k_norm_f32: self.k_norm_f32,
            state_out: self.state_out,
            out_normed: self.out_normed,
            gated_f32: self.gated_f32,
            gated_q8_1: self.gated_q8_1,
            ssm_out_f32: self.ssm_out_f32,
        }
    }
}

impl OwnedDeltaNetLayerDecodeScratch {
    pub fn view(&self) -> DeltaNetLayerDecodeScratch {
        DeltaNetLayerDecodeScratch {
            x_q8_1: self.x_q8_1,
            qkv_mixed_f32: self.qkv_mixed_f32,
            z_f32: self.z_f32,
            alpha_f32: self.alpha_f32,
            beta_f32: self.beta_f32,
            conv_input: self.conv_input,
            conv_out: self.conv_out,
            silu_out: self.silu_out,
            q_norm_f32: self.q_norm_f32,
            k_norm_f32: self.k_norm_f32,
            state_out: self.state_out,
            out_normed: self.out_normed,
            gated_f32: self.gated_f32,
            gated_q8_1: self.gated_q8_1,
            ssm_out_f32: self.ssm_out_f32,
        }
    }
}

/// Owned GDN prefill scratch.
pub struct OwnedDeltaNetLayerPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr,
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,
    pub qkv_mixed_f32: DevicePtr,
    pub z_f32: DevicePtr,
    pub alpha_f32: DevicePtr,
    pub beta_f32: DevicePtr,
    pub conv_input: DevicePtr,
    pub conv_out: DevicePtr,
    pub silu_out: DevicePtr,
    pub q_norm_f32: DevicePtr,
    pub k_norm_f32: DevicePtr,
    pub v_f32: DevicePtr,
    pub state_out: DevicePtr,
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,
    pub ssm_out_f32: DevicePtr,
}

impl OwnedDeltaNetLayerPrefillScratch {
    pub fn view(&self) -> DeltaNetLayerPrefillScratch {
        DeltaNetLayerPrefillScratch {
            max_tokens: self.max_tokens,
            x_norm_f16: self.x_norm_f16,
            x_q8_1: self.x_q8_1,
            x_q8_1_mmq: self.x_q8_1_mmq,
            qkv_mixed_f32: self.qkv_mixed_f32,
            z_f32: self.z_f32,
            alpha_f32: self.alpha_f32,
            beta_f32: self.beta_f32,
            conv_input: self.conv_input,
            conv_out: self.conv_out,
            silu_out: self.silu_out,
            q_norm_f32: self.q_norm_f32,
            k_norm_f32: self.k_norm_f32,
            v_f32: self.v_f32,
            state_out: self.state_out,
            out_normed: self.out_normed,
            gated_f32: self.gated_f32,
            gated_q8_1: self.gated_q8_1,
            gated_q8_1_mmq: self.gated_q8_1_mmq,
            ssm_out_f32: self.ssm_out_f32,
        }
    }
}

/// Borrowed-by-value view over a caller-owned GDN prefill scratch.
/// All buffers scale with `max_tokens`.
#[derive(Copy, Clone)]
pub struct DeltaNetLayerPrefillScratch {
    pub max_tokens: usize,
    pub x_norm_f16: DevicePtr, // F16 [L, hidden]
    pub x_q8_1: DevicePtr,
    pub x_q8_1_mmq: DevicePtr,    // DS4 layout sibling for MMQ at L≥128
    pub qkv_mixed_f32: DevicePtr, // [L, conv_channels]
    pub z_f32: DevicePtr,         // [L, d_inner]
    pub alpha_f32: DevicePtr,     // [L, num_v_heads]
    pub beta_f32: DevicePtr,      // [L, num_v_heads]
    pub conv_input: DevicePtr,    // [(K-1) + L, conv_channels]
    pub conv_out: DevicePtr,      // [L, conv_channels]
    pub silu_out: DevicePtr,      // [L, conv_channels]
    pub q_norm_f32: DevicePtr,    // [L, num_k_heads, head_k_dim]
    pub k_norm_f32: DevicePtr,    // [L, num_k_heads, head_k_dim]
    pub v_f32: DevicePtr,         // [L, num_v_heads, head_v_dim]
    pub state_out: DevicePtr,     // [L, num_v_heads, head_v_dim]
    pub out_normed: DevicePtr,
    pub gated_f32: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,
    pub ssm_out_f32: DevicePtr, // [L, hidden]
}

/// Qwen3.5-Next / Qwen3.6 GDN (gated delta-net) block.
pub struct DeltaNetLayer {
    pub attn_qkv: WeightHandle,  // [conv_channels, hidden]
    pub attn_gate: WeightHandle, // [d_inner, hidden]
    pub ssm_alpha: WeightHandle, // [num_v_heads, hidden]
    pub ssm_beta: WeightHandle,  // [num_v_heads, hidden]
    pub ssm_out: WeightHandle,   // [hidden, d_inner]
    pub ssm_dt_bias: DevicePtr,  // 1-D [num_v_heads] F32
    pub ssm_a: DevicePtr,        // 1-D [num_v_heads] F32
    pub ssm_conv1d: DevicePtr,   // [conv_kernel, conv_channels] F32
    pub ssm_norm_w: DevicePtr,   // 1-D [head_v_dim] F16
    pub attn_norm_w: DevicePtr,  // 1-D [hidden] F16
    pub hidden: usize,
    pub d_inner: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub rms_norm_eps: f32,
    /// Q/K head-repeat layout. `false` (default) for cyclic
    /// `ggml_repeat_4d` (qwen35moe / Qwen3.6-35B-A3B). `true` for
    /// reshape-interleave (qwen3next / Coder-Next-80B). Wrong choice
    /// produces a degenerate logit attractor.
    pub rep_inner_layout: bool,
}

/// All weight pointers for one GDN block. Construction parameter bundle
/// for [`DeltaNetLayer::new`].
#[derive(Copy, Clone)]
pub struct DeltaNetWeights {
    pub attn_qkv: WeightHandle,
    pub attn_gate: WeightHandle,
    pub ssm_alpha: WeightHandle,
    pub ssm_beta: WeightHandle,
    pub ssm_out: WeightHandle,
    pub ssm_dt_bias: DevicePtr,
    pub ssm_a: DevicePtr,
    pub ssm_conv1d: DevicePtr,
    pub ssm_norm_w: DevicePtr,
    pub attn_norm_w: DevicePtr,
}

/// All dimension scalars for one GDN block.
#[derive(Copy, Clone, Debug)]
pub struct DeltaNetDims {
    pub hidden: usize,
    pub d_inner: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

/// Per-call HIP context (device + stream) shared by every GDN driver
/// method. Lifetime-borrowed; not Copy because `&HipDevice` references
/// the cluster-owned device handle.
#[derive(Copy, Clone)]
pub struct BackendCtx<'a> {
    pub device: &'a HipDevice,
    pub stream: &'a HipStream,
}

/// Single-slot decode I/O pointers consumed by
/// [`DeltaNetLayer::forward_decode`] and `_with_ar_hook`.
#[derive(Copy, Clone, Debug)]
pub struct GdnDecodeBuffers {
    pub x_in: DevicePtr,
    pub delta_out: DevicePtr,
    pub state: DevicePtr,
    pub conv_history: DevicePtr,
}

/// Batched-slots decode I/O pointers consumed by
/// [`DeltaNetLayer::forward_decode_with_ar_hook_batched_slots`]. The
/// `*_ptrs_dev` fields are `[n_slots] u64` device arrays.
#[derive(Copy, Clone, Debug)]
pub struct GdnDecodeBatchedBuffers {
    pub x_in_base: DevicePtr,
    pub delta_out_base: DevicePtr,
    pub state_in_ptrs_dev: DevicePtr,
    pub state_out_ptrs_dev: DevicePtr,
    pub conv_history_ptrs_dev: DevicePtr,
}

/// Prefill-only sequence handle: token count + optional state event
/// the state-step kernel waits on before reading `state`.
#[derive(Copy, Clone)]
pub struct GdnPrefillSeq<'a> {
    pub n_tokens: usize,
    pub state_event: Option<&'a HipEvent>,
}

impl DeltaNetLayer {
    pub fn new(
        weights: DeltaNetWeights,
        dims: DeltaNetDims,
        rms_norm_eps: f32,
        rep_inner_layout: bool,
    ) -> Result<Self> {
        if dims.head_k_dim != 128 || dims.head_v_dim != 128 {
            bail!(
                "DeltaNetLayer: gdn_state_step kernel only instantiated at S_v=128 (head_k_dim={}, head_v_dim={})",
                dims.head_k_dim,
                dims.head_v_dim
            );
        }
        if dims.num_v_heads * dims.head_v_dim != dims.d_inner {
            bail!(
                "DeltaNetLayer: num_v_heads * head_v_dim ({}) != d_inner ({})",
                dims.num_v_heads * dims.head_v_dim,
                dims.d_inner
            );
        }
        Ok(Self {
            attn_qkv: weights.attn_qkv,
            attn_gate: weights.attn_gate,
            ssm_alpha: weights.ssm_alpha,
            ssm_beta: weights.ssm_beta,
            ssm_out: weights.ssm_out,
            ssm_dt_bias: weights.ssm_dt_bias,
            ssm_a: weights.ssm_a,
            ssm_conv1d: weights.ssm_conv1d,
            ssm_norm_w: weights.ssm_norm_w,
            attn_norm_w: weights.attn_norm_w,
            hidden: dims.hidden,
            d_inner: dims.d_inner,
            num_v_heads: dims.num_v_heads,
            num_k_heads: dims.num_k_heads,
            head_k_dim: dims.head_k_dim,
            head_v_dim: dims.head_v_dim,
            conv_channels: dims.conv_channels,
            conv_kernel: dims.conv_kernel,
            rms_norm_eps,
            rep_inner_layout,
        })
    }

    pub fn scratch_dims(&self) -> DeltaNetScratchDims {
        DeltaNetScratchDims {
            hidden: self.hidden,
            d_inner: self.d_inner,
            num_v_heads: self.num_v_heads,
            num_k_heads: self.num_k_heads,
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            conv_channels: self.conv_channels,
            conv_kernel: self.conv_kernel,
        }
    }

    /// Allocate an [`OwnedDeltaNetLayerDecodeScratch`] sized for `dims`.
    pub fn alloc_decode_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: DeltaNetScratchDims,
    ) -> Result<OwnedDeltaNetLayerDecodeScratch> {
        let DeltaNetScratchDims {
            hidden,
            d_inner,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_channels,
            conv_kernel,
        } = dims;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, hidden)?;
        let (qkv_mixed_f32, _) = tracker.alloc_f32(device, conv_channels)?;
        let (z_f32, _) = tracker.alloc_f32(device, d_inner)?;
        let (alpha_f32, _) = tracker.alloc_f32(device, num_v_heads)?;
        let (beta_f32, _) = tracker.alloc_f32(device, num_v_heads)?;
        let (conv_input, _) = tracker.alloc_f32(device, conv_kernel * conv_channels)?;
        let (conv_out, _) = tracker.alloc_f32(device, conv_channels)?;
        let (silu_out, _) = tracker.alloc_f32(device, conv_channels)?;
        let (q_norm_f32, _) = tracker.alloc_f32(device, qk_size)?;
        let (k_norm_f32, _) = tracker.alloc_f32(device, qk_size)?;
        let (state_out, _) = tracker.alloc_f32(device, v_size)?;
        let (out_normed, _) = tracker.alloc_f32(device, v_size)?;
        let (gated_f32, _) = tracker.alloc_f32(device, d_inner)?;
        let (gated_q8_1, _) = tracker.alloc_q8_1(device, d_inner)?;
        let (ssm_out_f32, _) = tracker.alloc_f32(device, hidden)?;
        Ok(OwnedDeltaNetLayerDecodeScratch {
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
        })
    }

    /// Allocate an [`OwnedDeltaNetLayerDecodeBatchedScratch`] sized
    /// for `dims` × `max_slots`. The slot-pointer arrays themselves
    /// (per-slot state and conv-history base ptrs) live separately —
    /// owned by the caller's pool because they index per-layer state.
    pub fn alloc_decode_batched_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: DeltaNetScratchDims,
        max_slots: usize,
    ) -> Result<OwnedDeltaNetLayerDecodeBatchedScratch> {
        if max_slots == 0 {
            bail!("alloc_decode_batched_scratch: max_slots must be >= 1");
        }
        let DeltaNetScratchDims {
            hidden,
            d_inner,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_channels,
            ..
        } = dims;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;
        let n = max_slots;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, n * hidden)?;
        let (qkv_mixed_f32, _) = tracker.alloc_f32(device, n * conv_channels)?;
        let (z_f32, _) = tracker.alloc_f32(device, n * d_inner)?;
        let (alpha_f32, _) = tracker.alloc_f32(device, n * num_v_heads)?;
        let (beta_f32, _) = tracker.alloc_f32(device, n * num_v_heads)?;
        let (conv_out, _) = tracker.alloc_f32(device, n * conv_channels)?;
        let (silu_out, _) = tracker.alloc_f32(device, n * conv_channels)?;
        let (q_norm_f32, _) = tracker.alloc_f32(device, n * qk_size)?;
        let (k_norm_f32, _) = tracker.alloc_f32(device, n * qk_size)?;
        let (state_out, _) = tracker.alloc_f32(device, n * v_size)?;
        let (out_normed, _) = tracker.alloc_f32(device, n * v_size)?;
        let (gated_f32, _) = tracker.alloc_f32(device, n * d_inner)?;
        let (gated_q8_1, _) = tracker.alloc_q8_1(device, n * d_inner)?;
        let (ssm_out_f32, _) = tracker.alloc_f32(device, n * hidden)?;
        Ok(OwnedDeltaNetLayerDecodeBatchedScratch {
            max_slots,
            x_q8_1,
            qkv_mixed_f32,
            z_f32,
            alpha_f32,
            beta_f32,
            conv_out,
            silu_out,
            q_norm_f32,
            k_norm_f32,
            state_out,
            out_normed,
            gated_f32,
            gated_q8_1,
            ssm_out_f32,
        })
    }

    /// Allocate an [`OwnedDeltaNetLayerPrefillScratch`] sized for
    /// `dims` × `max_tokens`. The conv history is sized as
    /// `(K-1) + max_tokens` rows so that one prefill chunk plus the
    /// trailing history fits in `conv_input`.
    pub fn alloc_prefill_scratch(
        device: &HipDevice,
        tracker: &mut RawAllocTracker,
        dims: DeltaNetScratchDims,
        max_tokens: usize,
    ) -> Result<OwnedDeltaNetLayerPrefillScratch> {
        if max_tokens == 0 {
            bail!("alloc_prefill_scratch: max_tokens must be >= 1");
        }
        let DeltaNetScratchDims {
            hidden,
            d_inner,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_channels,
            conv_kernel,
        } = dims;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;
        let (x_norm_f16, _) = tracker.alloc_f16(device, max_tokens * hidden)?;
        let (x_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * hidden)?;
        let (x_q8_1_mmq, _) = tracker.alloc_q8_1_mmq(device, max_tokens * hidden)?;
        let (qkv_mixed_f32, _) = tracker.alloc_f32(device, max_tokens * conv_channels)?;
        let (z_f32, _) = tracker.alloc_f32(device, max_tokens * d_inner)?;
        let (alpha_f32, _) = tracker.alloc_f32(device, max_tokens * num_v_heads)?;
        let (beta_f32, _) = tracker.alloc_f32(device, max_tokens * num_v_heads)?;
        let (conv_input, _) =
            tracker.alloc_f32(device, ((conv_kernel - 1) + max_tokens) * conv_channels)?;
        let (conv_out, _) = tracker.alloc_f32(device, max_tokens * conv_channels)?;
        let (silu_out, _) = tracker.alloc_f32(device, max_tokens * conv_channels)?;
        let (q_norm_f32, _) = tracker.alloc_f32(device, max_tokens * qk_size)?;
        let (k_norm_f32, _) = tracker.alloc_f32(device, max_tokens * qk_size)?;
        let (v_f32, _) = tracker.alloc_f32(device, max_tokens * v_size)?;
        let (state_out, _) = tracker.alloc_f32(device, max_tokens * v_size)?;
        let (out_normed, _) = tracker.alloc_f32(device, max_tokens * v_size)?;
        let (gated_f32, _) = tracker.alloc_f32(device, max_tokens * d_inner)?;
        let (gated_q8_1, _) = tracker.alloc_q8_1(device, max_tokens * d_inner)?;
        let (gated_q8_1_mmq, _) = tracker.alloc_q8_1_mmq(device, max_tokens * d_inner)?;
        let (ssm_out_f32, _) = tracker.alloc_f32(device, max_tokens * hidden)?;
        Ok(OwnedDeltaNetLayerPrefillScratch {
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
        })
    }

    /// Single-token decode through the GDN block. Mutates `state`
    /// (the recurrent state) and `conv_history` (the conv1d history)
    /// in-place. Writes pre-residual output to `delta_out`.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        ctx: BackendCtx<'_>,
        buf: GdnDecodeBuffers,
        scratch: DeltaNetLayerDecodeScratch,
    ) -> Result<()> {
        self.forward_decode_with_ar_hook(ops, ctx, buf, scratch, None)
    }

    /// Like `forward_decode`, but with an AR hook fired on the
    /// `ssm_out_f32` partial-hidden buffer between the row-parallel
    /// ssm_out mmvq and the F16 cast. Under TP the v2 composite
    /// supplies a callback that AR-sums across ranks; under non-TP
    /// topologies pass `None` (equivalent to `forward_decode`).
    pub fn forward_decode_with_ar_hook<O: Ops>(
        &self,
        ops: &O,
        ctx: BackendCtx<'_>,
        buf: GdnDecodeBuffers,
        scratch: DeltaNetLayerDecodeScratch,
        ar_partial_callback: Option<
            &mut dyn FnMut(DevicePtr, usize, &HipDevice, &HipStream) -> Result<()>,
        >,
    ) -> Result<()> {
        let BackendCtx { device, stream } = ctx;
        let GdnDecodeBuffers {
            x_in,
            delta_out,
            state,
            conv_history,
        } = buf;
        let hidden = self.hidden;
        let d_inner = self.d_inner;
        let num_v_heads = self.num_v_heads;
        let num_k_heads = self.num_k_heads;
        let head_k_dim = self.head_k_dim;
        let head_v_dim = self.head_v_dim;
        let conv_channels = self.conv_channels;
        let conv_kernel = self.conv_kernel;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;

        // 1. fused rmsnorm + Q8_1.
        ops.rmsnorm_quant_q8_1(
            flambeau_ops::NormBuffers {
                input: x_in,
                weight: self.attn_norm_w,
                output: scratch.x_q8_1,
            },
            flambeau_ops::NormShape { m: 1, k: hidden },
            self.rms_norm_eps,
        )
        .context("gdn attn_norm + quant")?;

        // 2+3. attn_qkv + attn_gate. Fuse when both same Q-dtype.
        let qkv_dt = self.attn_qkv.dtype;
        let gate_dt = self.attn_gate.dtype;
        let fuse_q8_0 = qkv_dt == QDtype::Q8_0 && gate_dt == QDtype::Q8_0;
        let fuse_q4_0 = qkv_dt == QDtype::Q4_0 && gate_dt == QDtype::Q4_0;
        if fuse_q8_0 {
            ops.mmvq_q8_0_gate_up(
                flambeau_ops::MmvqGateUpBuffers { gate_w: self.attn_qkv.ptr, up_w: self.attn_gate.ptr, act_q8_1: scratch.x_q8_1, gate_out: scratch.qkv_mixed_f32, up_out: scratch.z_f32 },
                flambeau_ops::MmvqGateUpShape { n_rows_gate: conv_channels, n_rows_up: d_inner, k: hidden },
            )
            .context("attn_qkv + attn_gate fused mmvq_q8_0")?;
        } else if fuse_q4_0 {
            ops.mmvq_q4_0_gate_up_t128(
                flambeau_ops::MmvqGateUpBuffers { gate_w: self.attn_qkv.ptr, up_w: self.attn_gate.ptr, act_q8_1: scratch.x_q8_1, gate_out: scratch.qkv_mixed_f32, up_out: scratch.z_f32 },
                flambeau_ops::MmvqGateUpShape { n_rows_gate: conv_channels, n_rows_up: d_inner, k: hidden },
            )
            .context("attn_qkv + attn_gate fused mmvq_q4_0_t128")?;
        } else {
            ops.mmvq(
                self.attn_qkv.ptr,
                scratch.x_q8_1,
                scratch.qkv_mixed_f32,
                conv_channels,
                hidden,
                qkv_dt,
            )
            .context("attn_qkv mmvq")?;
            ops.mmvq(
                self.attn_gate.ptr,
                scratch.x_q8_1,
                scratch.z_f32,
                d_inner,
                hidden,
                gate_dt,
            )
            .context("attn_gate mmvq")?;
        }

        // 4+5. ssm_alpha + ssm_beta. Fuse when both Q8_0.
        let alpha_dt = self.ssm_alpha.dtype;
        let beta_dt = self.ssm_beta.dtype;
        let fuse_alpha_beta = alpha_dt == QDtype::Q8_0 && beta_dt == QDtype::Q8_0;
        if fuse_alpha_beta {
            ops.mmvq_q8_0_gate_up(
                flambeau_ops::MmvqGateUpBuffers { gate_w: self.ssm_alpha.ptr, up_w: self.ssm_beta.ptr, act_q8_1: scratch.x_q8_1, gate_out: scratch.alpha_f32, up_out: scratch.beta_f32 },
                flambeau_ops::MmvqGateUpShape { n_rows_gate: num_v_heads, n_rows_up: num_v_heads, k: hidden },
            )
            .context("ssm alpha+beta fused mmvq_q8_0")?;
        } else {
            ops.mmvq(
                self.ssm_alpha.ptr,
                scratch.x_q8_1,
                scratch.alpha_f32,
                num_v_heads,
                hidden,
                alpha_dt,
            )
            .context("ssm_alpha mmvq")?;
            ops.mmvq(
                self.ssm_beta.ptr,
                scratch.x_q8_1,
                scratch.beta_f32,
                num_v_heads,
                hidden,
                beta_dt,
            )
            .context("ssm_beta mmvq")?;
        }

        // 6. Conv1d step. assemble [history_{k-1}, qkv_mixed] → conv_input,
        // run causal conv, then shift history forward.
        ops.gdn_assemble_conv_input_f32(
            conv_history,
            scratch.qkv_mixed_f32,
            scratch.conv_input,
            conv_channels,
            conv_kernel,
        )
        .context("gdn_assemble_conv_input_f32")?;
        ops.causal_conv1d_f32(
            scratch.conv_input,
            self.ssm_conv1d,
            scratch.conv_out,
            1,
            conv_channels,
            conv_kernel,
        )
        .context("causal_conv1d_f32")?;
        // Shift conv_history forward by one row. `conv_input` now
        // holds [old_history, current]; copying `conv_input[1..k]`
        // back to `history` is a single DtoD memcpy.
        let row_bytes = conv_channels * 4;
        let hist_rows = conv_kernel - 1;
        // SAFETY: conv_input has at least `conv_kernel * row_bytes`
        // valid bytes, history has at least `hist_rows * row_bytes`.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                conv_history,
                scratch.conv_input.offset_bytes(row_bytes),
                hist_rows * row_bytes,
            )?;
        }

        // 7. silu.
        ops.silu_f32(scratch.conv_out, scratch.silu_out, conv_channels)
            .context("silu_f32(conv_out)")?;

        // 8. Slice silu_out into Q | K | V via pointer offsets.
        let q_src = scratch.silu_out;
        let k_src = scratch.silu_out.offset_bytes(qk_size * 4);
        let v_src = scratch.silu_out.offset_bytes(2 * qk_size * 4);

        // 9. L2-normalise Q and K per head.
        ops.l2_norm_f32(
            q_src,
            scratch.q_norm_f32,
            num_k_heads,
            head_k_dim,
            self.rms_norm_eps,
        )
        .context("l2_norm Q")?;
        ops.l2_norm_f32(
            k_src,
            scratch.k_norm_f32,
            num_k_heads,
            head_k_dim,
            self.rms_norm_eps,
        )
        .context("l2_norm K")?;

        // 10. Scale Q by 1 / sqrt(head_k_dim) (in-place).
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        ops.scale_f32(scratch.q_norm_f32, scratch.q_norm_f32, qk_size, q_scale)
            .context("scale_f32 Q")?;

        // 11. Fused state-step (absorbs α/β/gate compute).
        let n_rep = num_v_heads / num_k_heads;
        ops.gdn_state_step_alphabeta_f32_s128(
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            v_src,
            scratch.alpha_f32,
            scratch.beta_f32,
            self.ssm_dt_bias,
            self.ssm_a,
            state,
            state,
            scratch.state_out,
            1,
            num_v_heads,
            1,
            n_rep,
            self.rep_inner_layout,
        )
        .context("gdn_state_step_alphabeta_f32_s128")?;

        // 12. ssm_norm per-head on the state-step output.
        ops.rmsnorm_f32(
            flambeau_ops::NormBuffers {
                input: scratch.state_out,
                weight: self.ssm_norm_w,
                output: scratch.out_normed,
            },
            flambeau_ops::NormShape {
                m: num_v_heads,
                k: head_v_dim,
            },
            self.rms_norm_eps,
        )
        .context("ssm_norm (rmsnorm_f32)")?;

        // 13+14. fused swiglu(z, out_normed) → Q8_1, or unfused pair
        // when d_inner is not a multiple of 32.
        if v_size != d_inner {
            bail!("GDN layout: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})");
        }
        let fuse_tail = d_inner % 32 == 0;
        if fuse_tail {
            ops.swiglu_f32_to_q8_1(
                scratch.z_f32,
                scratch.out_normed,
                scratch.gated_q8_1,
                d_inner,
            )
            .context("swiglu_f32_to_q8_1(z, out_normed)")?;
        } else {
            ops.swiglu_f32(
                scratch.z_f32,
                scratch.out_normed,
                scratch.gated_f32,
                d_inner,
            )
            .context("swiglu_f32(z, out_normed)")?;
            ops.quantize_q8_1(scratch.gated_f32, scratch.gated_q8_1, d_inner)
                .context("quantize gated → Q8_1")?;
        }

        // 15. ssm_out projection. Output is rank-local partial under
        // TP (d_inner is per-rank); `ar_partial_callback` AR-sums it.
        ops.mmvq(
            self.ssm_out.ptr,
            scratch.gated_q8_1,
            scratch.ssm_out_f32,
            hidden,
            d_inner,
            self.ssm_out.dtype,
        )
        .context("ssm_out mmvq")?;

        if let Some(cb) = ar_partial_callback {
            cb(scratch.ssm_out_f32, hidden, device, stream)
                .context("gdn ar_partial_callback (ssm_out F32)")?;
        }

        // 16. Cast F32 → F16.
        ops.cast_f32_to_f16(scratch.ssm_out_f32, delta_out, hidden)
            .context("cast ssm_out → f16")?;

        Ok(())
    }

    /// Batched-slots single-token decode through the GDN block. Same
    /// per-token compute as `forward_decode_with_ar_hook` but issues
    /// one launch of the state-step + conv-trio kernels across all
    /// `n_slots` instead of `N` per-slot launches. Projections + per-
    /// head norms + swiglu + ssm_out are still per-slot looped (Phase
    /// 2 introduces batched MMVQ).
    ///
    /// `state_in_ptrs_dev` / `state_out_ptrs_dev` / `conv_history_ptrs_dev`
    /// are `[n_slots] u64` device arrays of per-slot base pointers.
    /// In-place use sets `state_out_ptrs_dev == state_in_ptrs_dev`.
    /// Activation inputs/outputs `x_in_base` and `delta_out_base` are
    /// slot-major `[n_slots, hidden]` F16. The AR callback fires once
    /// per slot on its `ssm_out_f32[slot]` partial.
    pub fn forward_decode_with_ar_hook_batched_slots<O: Ops>(
        &self,
        ops: &O,
        ctx: BackendCtx<'_>,
        buf: GdnDecodeBatchedBuffers,
        scratch: DeltaNetLayerDecodeBatchedScratch,
        n_slots: usize,
        ar_partial_callback: Option<
            &mut dyn FnMut(DevicePtr, usize, &HipDevice, &HipStream) -> Result<()>,
        >,
    ) -> Result<()> {
        let BackendCtx { device, stream } = ctx;
        let GdnDecodeBatchedBuffers {
            x_in_base,
            delta_out_base,
            state_in_ptrs_dev,
            state_out_ptrs_dev,
            conv_history_ptrs_dev,
        } = buf;
        if n_slots == 0 {
            bail!("DeltaNetLayer::forward_decode_with_ar_hook_batched_slots: n_slots = 0");
        }
        if n_slots > scratch.max_slots {
            bail!(
                "DeltaNetLayer::forward_decode_with_ar_hook_batched_slots: \
                 n_slots={n_slots} > scratch.max_slots={}; caller must chunk",
                scratch.max_slots
            );
        }

        let hidden = self.hidden;
        let d_inner = self.d_inner;
        let num_v_heads = self.num_v_heads;
        let num_k_heads = self.num_k_heads;
        let head_k_dim = self.head_k_dim;
        let head_v_dim = self.head_v_dim;
        let conv_channels = self.conv_channels;
        let conv_kernel = self.conv_kernel;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;

        if v_size != d_inner {
            bail!("GDN layout: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})");
        }

        let x_row_f16_bytes = hidden * 2;
        let x_q8_1_row_bytes = (hidden / 32) * 36;
        let qkv_mixed_row_bytes = conv_channels * 4;
        let z_row_bytes = d_inner * 4;
        let alphabeta_row_bytes = num_v_heads * 4;
        let qk_row_bytes = qk_size * 4;
        let v_row_bytes = v_size * 4;
        let gated_q8_1_row_bytes = (d_inner / 32) * 36;
        let gated_f32_row_bytes = d_inner * 4;
        let conv_out_row_bytes = conv_channels * 4;
        let silu_out_row_bytes = conv_channels * 4;
        let ssm_out_row_bytes = hidden * 4;

        let qkv_dt = self.attn_qkv.dtype;
        let gate_dt = self.attn_gate.dtype;
        let fuse_q8_0 = qkv_dt == QDtype::Q8_0 && gate_dt == QDtype::Q8_0;
        let fuse_q4_0 = qkv_dt == QDtype::Q4_0 && gate_dt == QDtype::Q4_0;
        let alpha_dt = self.ssm_alpha.dtype;
        let beta_dt = self.ssm_beta.dtype;
        let fuse_alpha_beta = alpha_dt == QDtype::Q8_0 && beta_dt == QDtype::Q8_0;
        let fuse_tail = d_inner % 32 == 0;
        // Phase 2 Slice A: Q4_0 gate+up row-tile batched MMVQ shares one
        // LDS-resident activation strip across N decode slots, replacing
        // N per-slot `mmvq_q4_0_gate_up_t128` launches with one batched
        // launch. Only firing for n_slots ∈ [2, 4] — the kernel only
        // specialises N up to 4.
        let use_q4_0_gate_up_row_tile =
            fuse_q4_0 && matches!(n_slots, 2..=4);

        // 1. per-slot rmsnorm+quant. Writes slot-major into scratch.x_q8_1.
        for i in 0..n_slots {
            let x_in_i = x_in_base.offset_bytes(i * x_row_f16_bytes);
            let x_q8_1_i = scratch.x_q8_1.offset_bytes(i * x_q8_1_row_bytes);
            ops.rmsnorm_quant_q8_1(
                flambeau_ops::NormBuffers {
                    input: x_in_i,
                    weight: self.attn_norm_w,
                    output: x_q8_1_i,
                },
                flambeau_ops::NormShape { m: 1, k: hidden },
                self.rms_norm_eps,
            )
            .context("gdn batched: attn_norm + quant")?;
        }

        // 2+3. attn_qkv + attn_gate. Row-tile batched (Q4_0 N∈[2,4]) is
        // one launch for all N slots; otherwise per-slot loop.
        if use_q4_0_gate_up_row_tile {
            ops.mmvq_q4_0_gate_up_row_tile_batched(
                flambeau_ops::MmvqGateUpBuffers {
                    gate_w: self.attn_qkv.ptr,
                    up_w: self.attn_gate.ptr,
                    act_q8_1: scratch.x_q8_1,
                    gate_out: scratch.qkv_mixed_f32,
                    up_out: scratch.z_f32,
                },
                flambeau_ops::MmvqGateUpBatchShape {
                    n_rows_gate: conv_channels,
                    n_rows_up: d_inner,
                    k: hidden,
                    n_slots,
                },
            )
            .context("gdn batched: attn_qkv+attn_gate row-tile batched Q4_0")?;
        } else {
            for i in 0..n_slots {
                let x_q8_1_i = scratch.x_q8_1.offset_bytes(i * x_q8_1_row_bytes);
                let qkv_mixed_i = scratch.qkv_mixed_f32.offset_bytes(i * qkv_mixed_row_bytes);
                let z_i = scratch.z_f32.offset_bytes(i * z_row_bytes);
                if fuse_q8_0 {
                    ops.mmvq_q8_0_gate_up(
                        flambeau_ops::MmvqGateUpBuffers { gate_w: self.attn_qkv.ptr, up_w: self.attn_gate.ptr, act_q8_1: x_q8_1_i, gate_out: qkv_mixed_i, up_out: z_i },
                        flambeau_ops::MmvqGateUpShape { n_rows_gate: conv_channels, n_rows_up: d_inner, k: hidden },
                    )
                    .context("gdn batched: attn_qkv + attn_gate fused mmvq_q8_0")?;
                } else if fuse_q4_0 {
                    ops.mmvq_q4_0_gate_up_t128(
                        flambeau_ops::MmvqGateUpBuffers { gate_w: self.attn_qkv.ptr, up_w: self.attn_gate.ptr, act_q8_1: x_q8_1_i, gate_out: qkv_mixed_i, up_out: z_i },
                        flambeau_ops::MmvqGateUpShape { n_rows_gate: conv_channels, n_rows_up: d_inner, k: hidden },
                    )
                    .context("gdn batched: attn_qkv + attn_gate fused mmvq_q4_0_t128")?;
                } else {
                    ops.mmvq(self.attn_qkv.ptr, x_q8_1_i, qkv_mixed_i, conv_channels, hidden, qkv_dt)
                        .context("gdn batched: attn_qkv mmvq")?;
                    ops.mmvq(self.attn_gate.ptr, x_q8_1_i, z_i, d_inner, hidden, gate_dt)
                        .context("gdn batched: attn_gate mmvq")?;
                }
            }
        }

        // 4+5. ssm_alpha + ssm_beta. Phase 2 Slice B: for n_slots ∈
        // [2, 4], drop the Q8_0 gate+up fusion and split into two
        // batched qmatmul calls — `qmatmul` auto-dispatches Q8_0
        // m∈{2,3,4} to `mmvq_q8_0_batched` (one launch per matrix,
        // weight read once across all N slots vs. N per-slot reads
        // in the fused path). Fused path retained for n_slots == 1
        // and for dtypes the slot-batched dispatch doesn't cover.
        let use_alphabeta_batched = matches!(n_slots, 2..=4);
        if use_alphabeta_batched {
            ops.qmatmul(
                flambeau_ops::QmatmulBuffers {
                    weights: self.ssm_alpha.ptr,
                    act_q8_1: scratch.x_q8_1,
                    act_q8_1_mmq: scratch.x_q8_1,
                    dst: scratch.alpha_f32,
                },
                flambeau_ops::MatmulShape {
                    m: n_slots,
                    k: hidden,
                    n: num_v_heads,
                },
                alpha_dt,
            )
            .context("gdn batched: ssm_alpha qmatmul (m=N batched)")?;
            ops.qmatmul(
                flambeau_ops::QmatmulBuffers {
                    weights: self.ssm_beta.ptr,
                    act_q8_1: scratch.x_q8_1,
                    act_q8_1_mmq: scratch.x_q8_1,
                    dst: scratch.beta_f32,
                },
                flambeau_ops::MatmulShape {
                    m: n_slots,
                    k: hidden,
                    n: num_v_heads,
                },
                beta_dt,
            )
            .context("gdn batched: ssm_beta qmatmul (m=N batched)")?;
        } else {
            for i in 0..n_slots {
                let x_q8_1_i = scratch.x_q8_1.offset_bytes(i * x_q8_1_row_bytes);
                let alpha_i = scratch.alpha_f32.offset_bytes(i * alphabeta_row_bytes);
                let beta_i = scratch.beta_f32.offset_bytes(i * alphabeta_row_bytes);
                if fuse_alpha_beta {
                    ops.mmvq_q8_0_gate_up(
                        flambeau_ops::MmvqGateUpBuffers { gate_w: self.ssm_alpha.ptr, up_w: self.ssm_beta.ptr, act_q8_1: x_q8_1_i, gate_out: alpha_i, up_out: beta_i },
                        flambeau_ops::MmvqGateUpShape { n_rows_gate: num_v_heads, n_rows_up: num_v_heads, k: hidden },
                    )
                    .context("gdn batched: ssm alpha+beta fused mmvq_q8_0")?;
                } else {
                    ops.mmvq(self.ssm_alpha.ptr, x_q8_1_i, alpha_i, num_v_heads, hidden, alpha_dt)
                        .context("gdn batched: ssm_alpha mmvq")?;
                    ops.mmvq(self.ssm_beta.ptr, x_q8_1_i, beta_i, num_v_heads, hidden, beta_dt)
                        .context("gdn batched: ssm_beta mmvq")?;
                }
            }
        }

        // 6. Fused conv trio across N slots in one launch.
        ops.gdn_conv_trio_decode_f32_batched_slots(
            conv_history_ptrs_dev,
            scratch.qkv_mixed_f32,
            self.ssm_conv1d,
            scratch.conv_out,
            n_slots,
            conv_channels,
            conv_kernel,
        )
        .context("gdn batched: conv_trio_decode")?;

        // 7-10: per-slot silu, slice, L2 norm Q+K, scale Q.
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        for i in 0..n_slots {
            let conv_out_i = scratch.conv_out.offset_bytes(i * conv_out_row_bytes);
            let silu_out_i = scratch.silu_out.offset_bytes(i * silu_out_row_bytes);
            ops.silu_f32(conv_out_i, silu_out_i, conv_channels)
                .context("gdn batched: silu_f32(conv_out)")?;

            let q_src = silu_out_i;
            let k_src = silu_out_i.offset_bytes(qk_size * 4);
            // v_src = silu_out_i + 2*qk_size*4, but the state-step reads
            // V from a slot-major contiguous buffer; we need v in the
            // batched-slots [N, num_v_heads, head_v_dim] layout. Pack
            // v_src directly out of silu_out by a memcpy_async into the
            // appropriate state_out region? No — state_out is the
            // recurrent state output. Use a dedicated v-pack: stride is
            // already correct because silu_out_i + 2*qk_size*4 holds the
            // V slice for slot i, but the state-step kernel expects all
            // V slots adjacent in `[N, L=1, H, S_v]` slot-major layout.
            // Solution: write the per-slot V slice into `state_out`
            // staging? No — state_out is the output. The cleanest fix is
            // a per-slot DtoD copy of the V slice into a dedicated
            // contiguous buffer; we reuse `out_normed` as that buffer
            // (it's sized [N, num_v_heads, head_v_dim] and is consumed
            // only AFTER the state-step).
            let v_src = silu_out_i.offset_bytes(2 * qk_size * 4);
            let v_dst_i = scratch.out_normed.offset_bytes(i * v_row_bytes);
            // SAFETY: silu_out has v_size*4 valid bytes past the QK
            // slice; out_normed has v_row_bytes per slot.
            unsafe {
                device.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    v_dst_i,
                    v_src,
                    v_row_bytes,
                )?;
            }

            let q_norm_i = scratch.q_norm_f32.offset_bytes(i * qk_row_bytes);
            let k_norm_i = scratch.k_norm_f32.offset_bytes(i * qk_row_bytes);
            ops.l2_norm_f32(q_src, q_norm_i, num_k_heads, head_k_dim, self.rms_norm_eps)
                .context("gdn batched: l2_norm Q")?;
            ops.l2_norm_f32(k_src, k_norm_i, num_k_heads, head_k_dim, self.rms_norm_eps)
                .context("gdn batched: l2_norm K")?;
            ops.scale_f32(q_norm_i, q_norm_i, qk_size, q_scale)
                .context("gdn batched: scale_f32 Q")?;
        }

        // 11. Fused state-step across N slots. V lives in `out_normed`
        // (slot-major contiguous); state-step writes attn output into
        // `state_out`. After this call we discard the V staging in
        // `out_normed` and re-use it for ssm_norm output.
        let n_rep = num_v_heads / num_k_heads;
        ops.gdn_state_step_alphabeta_f32_s128_batched_slots(
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.out_normed, // V staging (re-used as ssm_norm output below)
            scratch.alpha_f32,
            scratch.beta_f32,
            self.ssm_dt_bias,
            self.ssm_a,
            state_in_ptrs_dev,
            state_out_ptrs_dev,
            scratch.state_out,
            n_slots,
            num_v_heads,
            1,
            n_rep,
            self.rep_inner_layout,
        )
        .context("gdn batched: gdn_state_step_alphabeta_f32_s128_batched_slots")?;

        // 12+13+14: per-slot ssm_norm + swiglu (produces slot-major
        // gated_q8_1 ready for the batched ssm_out matmul below).
        for i in 0..n_slots {
            let state_out_i = scratch.state_out.offset_bytes(i * v_row_bytes);
            let out_normed_i = scratch.out_normed.offset_bytes(i * v_row_bytes);
            let z_i = scratch.z_f32.offset_bytes(i * z_row_bytes);
            let gated_q8_1_i = scratch.gated_q8_1.offset_bytes(i * gated_q8_1_row_bytes);
            let gated_f32_i = scratch.gated_f32.offset_bytes(i * gated_f32_row_bytes);

            ops.rmsnorm_f32(
                flambeau_ops::NormBuffers {
                    input: state_out_i,
                    weight: self.ssm_norm_w,
                    output: out_normed_i,
                },
                flambeau_ops::NormShape {
                    m: num_v_heads,
                    k: head_v_dim,
                },
                self.rms_norm_eps,
            )
            .context("gdn batched: ssm_norm (rmsnorm_f32)")?;

            if fuse_tail {
                ops.swiglu_f32_to_q8_1(z_i, out_normed_i, gated_q8_1_i, d_inner)
                    .context("gdn batched: swiglu_f32_to_q8_1(z, out_normed)")?;
            } else {
                ops.swiglu_f32(z_i, out_normed_i, gated_f32_i, d_inner)
                    .context("gdn batched: swiglu_f32(z, out_normed)")?;
                ops.quantize_q8_1(gated_f32_i, gated_q8_1_i, d_inner)
                    .context("gdn batched: quantize gated → Q8_1")?;
            }
        }

        // 15. ssm_out projection. Phase 2 Slice C: for n_slots ∈
        // [2, 4], one batched qmatmul replaces N per-slot mmvq launches.
        // The dispatch at `qmatmul.rs:71-105` routes Q4_0/Q8_0/Q4_1/Q4_K
        // /Q6_K m∈{2,3,4} to the corresponding slot-batched MMVQ variant
        // (1 weight read per N slots vs N reads in the per-slot fallback).
        // ssm_out is the largest weight in the GDN block — biggest absolute
        // weight-HBM saving in Phase 2 outside the gate+up row-tile.
        let use_ssm_out_batched = matches!(n_slots, 2..=4);
        if use_ssm_out_batched {
            ops.qmatmul(
                flambeau_ops::QmatmulBuffers {
                    weights: self.ssm_out.ptr,
                    act_q8_1: scratch.gated_q8_1,
                    act_q8_1_mmq: scratch.gated_q8_1,
                    dst: scratch.ssm_out_f32,
                },
                flambeau_ops::MatmulShape {
                    m: n_slots,
                    k: d_inner,
                    n: hidden,
                },
                self.ssm_out.dtype,
            )
            .context("gdn batched: ssm_out qmatmul (m=N batched)")?;
        } else {
            for i in 0..n_slots {
                let gated_q8_1_i = scratch.gated_q8_1.offset_bytes(i * gated_q8_1_row_bytes);
                let ssm_out_i = scratch.ssm_out_f32.offset_bytes(i * ssm_out_row_bytes);
                ops.mmvq(
                    self.ssm_out.ptr,
                    gated_q8_1_i,
                    ssm_out_i,
                    hidden,
                    d_inner,
                    self.ssm_out.dtype,
                )
                .context("gdn batched: ssm_out mmvq (per-slot fallback)")?;
            }
        }

        if let Some(cb) = ar_partial_callback {
            for i in 0..n_slots {
                let ssm_out_i = scratch.ssm_out_f32.offset_bytes(i * ssm_out_row_bytes);
                cb(ssm_out_i, hidden, device, stream)
                    .context("gdn batched: ar_partial_callback (ssm_out F32 per slot)")?;
            }
        }

        for i in 0..n_slots {
            let ssm_out_i = scratch.ssm_out_f32.offset_bytes(i * ssm_out_row_bytes);
            let delta_out_i = delta_out_base.offset_bytes(i * x_row_f16_bytes);
            ops.cast_f32_to_f16(ssm_out_i, delta_out_i, hidden)
                .context("gdn batched: cast ssm_out → f16")?;
        }

        Ok(())
    }

    /// L-token prefill through the GDN block. Mutates `state` and
    /// `conv_history` in place; writes pre-residual `delta_out`
    /// (F16 `[L, hidden]`). The recurrent state-step kernel is
    /// L-aware and keeps the per-head `[S_v, S_v]` state register-
    /// resident across the whole L recurrence in a single launch.
    pub fn forward_prefill<O: Ops>(
        &self,
        ops: &O,
        ctx: BackendCtx<'_>,
        buf: GdnDecodeBuffers,
        scratch: DeltaNetLayerPrefillScratch,
        seq: GdnPrefillSeq<'_>,
    ) -> Result<()> {
        self.forward_prefill_with_ar_hook(ops, ctx, buf, scratch, seq, None)
    }

    /// Like `forward_prefill` but with an AR hook on the `ssm_out_f32`
    /// partial buffer between the row-parallel ssm_out qmatmul and the
    /// F32→F16 cast. Mirrors [`Self::forward_decode_with_ar_hook`].
    pub fn forward_prefill_with_ar_hook<O: Ops>(
        &self,
        ops: &O,
        ctx: BackendCtx<'_>,
        buf: GdnDecodeBuffers,
        scratch: DeltaNetLayerPrefillScratch,
        seq: GdnPrefillSeq<'_>,
        ar_partial_callback: Option<
            &mut dyn FnMut(DevicePtr, usize, &HipDevice, &HipStream) -> Result<()>,
        >,
    ) -> Result<()> {
        let BackendCtx { device, stream } = ctx;
        let GdnDecodeBuffers {
            x_in,
            delta_out,
            state,
            conv_history,
        } = buf;
        let GdnPrefillSeq {
            n_tokens,
            state_event,
        } = seq;
        if n_tokens == 0 {
            bail!("DeltaNetLayer::forward_prefill called with n_tokens = 0");
        }
        if n_tokens > scratch.max_tokens {
            bail!(
                "DeltaNetLayer::forward_prefill: n_tokens={n_tokens} > scratch.max_tokens={}; caller must chunk",
                scratch.max_tokens
            );
        }

        let hidden = self.hidden;
        let d_inner = self.d_inner;
        let num_v_heads = self.num_v_heads;
        let num_k_heads = self.num_k_heads;
        let head_k_dim = self.head_k_dim;
        let head_v_dim = self.head_v_dim;
        let conv_channels = self.conv_channels;
        let conv_kernel = self.conv_kernel;
        let qk_size = num_k_heads * head_k_dim;
        let v_size = num_v_heads * head_v_dim;

        // 1. rmsnorm(x_in) → F16 scratch, then quantise to BOTH Q8_1
        // layouts (std + DS4) so qmatmul can dispatch MMQ at L≥128.
        ops.rmsnorm_f16(
            flambeau_ops::NormBuffers {
                input: x_in,
                weight: self.attn_norm_w,
                output: scratch.x_norm_f16,
            },
            flambeau_ops::NormShape {
                m: n_tokens,
                k: hidden,
            },
            self.rms_norm_eps,
        )
        .context("gdn prefill attn_norm")?;
        ops.quantize_f16_q8_1(scratch.x_norm_f16, scratch.x_q8_1, n_tokens * hidden)
            .context("gdn prefill x_norm → Q8_1 (std)")?;
        ops.quantize_f16_q8_1_mmq(scratch.x_norm_f16, scratch.x_q8_1_mmq, hidden, n_tokens)
            .context("gdn prefill x_norm → Q8_1 (MMQ DS4)")?;

        // 2..5. Hidden-input projections at M = L. qmatmul auto-
        // dispatches MMVQ vs MMQ based on M and weight dtype.
        ops.qmatmul(
            flambeau_ops::QmatmulBuffers {
                weights: self.attn_qkv.ptr,
                act_q8_1: scratch.x_q8_1,
                act_q8_1_mmq: scratch.x_q8_1_mmq,
                dst: scratch.qkv_mixed_f32,
            },
            flambeau_ops::MatmulShape {
                m: n_tokens,
                k: hidden,
                n: conv_channels,
            },
            self.attn_qkv.dtype,
        )
        .context("gdn prefill attn_qkv qmatmul")?;
        ops.qmatmul(
            flambeau_ops::QmatmulBuffers {
                weights: self.attn_gate.ptr,
                act_q8_1: scratch.x_q8_1,
                act_q8_1_mmq: scratch.x_q8_1_mmq,
                dst: scratch.z_f32,
            },
            flambeau_ops::MatmulShape {
                m: n_tokens,
                k: hidden,
                n: d_inner,
            },
            self.attn_gate.dtype,
        )
        .context("gdn prefill attn_gate qmatmul")?;
        ops.qmatmul(
            flambeau_ops::QmatmulBuffers {
                weights: self.ssm_alpha.ptr,
                act_q8_1: scratch.x_q8_1,
                act_q8_1_mmq: scratch.x_q8_1_mmq,
                dst: scratch.alpha_f32,
            },
            flambeau_ops::MatmulShape {
                m: n_tokens,
                k: hidden,
                n: num_v_heads,
            },
            self.ssm_alpha.dtype,
        )
        .context("gdn prefill ssm_alpha qmatmul")?;
        ops.qmatmul(
            flambeau_ops::QmatmulBuffers {
                weights: self.ssm_beta.ptr,
                act_q8_1: scratch.x_q8_1,
                act_q8_1_mmq: scratch.x_q8_1_mmq,
                dst: scratch.beta_f32,
            },
            flambeau_ops::MatmulShape {
                m: n_tokens,
                k: hidden,
                n: num_v_heads,
            },
            self.ssm_beta.dtype,
        )
        .context("gdn prefill ssm_beta qmatmul")?;

        // 6. Conv1d across L tokens. Assemble `[history, qkv_mixed]`
        // → conv_input via two DtoD memcpys; run conv; shift the
        // last (K-1) rows back into history.
        let row_bytes = conv_channels * 4;
        let hist_rows = conv_kernel - 1;
        // SAFETY: conv_input has at least (K-1+L) rows, history has
        // K-1 rows, qkv_mixed_f32 has L rows; all of size row_bytes.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                scratch.conv_input,
                conv_history,
                hist_rows * row_bytes,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                scratch.conv_input.offset_bytes(hist_rows * row_bytes),
                scratch.qkv_mixed_f32,
                n_tokens * row_bytes,
            )?;
        }
        ops.causal_conv1d_f32(
            scratch.conv_input,
            self.ssm_conv1d,
            scratch.conv_out,
            n_tokens,
            conv_channels,
            conv_kernel,
        )
        .context("gdn prefill causal_conv1d_f32")?;
        // SAFETY: conv_input has (K-1+L) valid rows; history has K-1.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                conv_history,
                scratch.conv_input.offset_bytes(n_tokens * row_bytes),
                hist_rows * row_bytes,
            )?;
        }

        // 7. silu(conv_out).
        ops.silu_f32(scratch.conv_out, scratch.silu_out, n_tokens * conv_channels)
            .context("gdn prefill silu_f32(conv_out)")?;

        // 8. Split silu_out into Q | K | V contiguous buffers via the
        // fused kernel (memcpy slice loop is ~1500 driver calls/layer
        // at L=512).
        ops.gdn_split_qkv_f32(
            scratch.silu_out,
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            n_tokens,
            qk_size,
            v_size,
        )
        .context("gdn prefill gdn_split_qkv_f32")?;

        // 9. L2-normalise Q and K per head.
        ops.l2_norm_f32(
            scratch.q_norm_f32,
            scratch.q_norm_f32,
            n_tokens * num_k_heads,
            head_k_dim,
            self.rms_norm_eps,
        )
        .context("gdn prefill l2_norm Q")?;
        ops.l2_norm_f32(
            scratch.k_norm_f32,
            scratch.k_norm_f32,
            n_tokens * num_k_heads,
            head_k_dim,
            self.rms_norm_eps,
        )
        .context("gdn prefill l2_norm K")?;

        // 10. Scale Q by 1 / sqrt(head_k_dim) (in-place).
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        ops.scale_f32(
            scratch.q_norm_f32,
            scratch.q_norm_f32,
            n_tokens * qk_size,
            q_scale,
        )
        .context("gdn prefill scale_f32 Q")?;

        // 11. Fused state-step (absorbs α/β/gate compute). Optional
        // cross-stream event ordering: the caller may provide a per-
        // (rank, layer) `HipEvent` so that residual streams downstream
        // wait on the state-step before reading `state`.
        let n_rep = num_v_heads / num_k_heads;
        if let Some(ev) = state_event {
            ev.stream_wait(stream)
                .context("gdn prefill state_step stream_wait")?;
        }
        ops.gdn_state_step_alphabeta_f32_s128(
            scratch.q_norm_f32,
            scratch.k_norm_f32,
            scratch.v_f32,
            scratch.alpha_f32,
            scratch.beta_f32,
            self.ssm_dt_bias,
            self.ssm_a,
            state,
            state,
            scratch.state_out,
            1,
            num_v_heads,
            n_tokens,
            n_rep,
            self.rep_inner_layout,
        )
        .context("gdn prefill gdn_state_step_alphabeta_f32_s128")?;
        if let Some(ev) = state_event {
            ev.record(stream).context("gdn prefill state_step record")?;
        }

        // 12. ssm_norm per-head over L × num_v_heads rows.
        ops.rmsnorm_f32(
            flambeau_ops::NormBuffers {
                input: scratch.state_out,
                weight: self.ssm_norm_w,
                output: scratch.out_normed,
            },
            flambeau_ops::NormShape {
                m: n_tokens * num_v_heads,
                k: head_v_dim,
            },
            self.rms_norm_eps,
        )
        .context("gdn prefill ssm_norm (rmsnorm_f32)")?;

        // 13+14. swiglu(z, out_normed) → F32, then dual-quantise to
        // both Q8_1 layouts so the ssm_out matmul auto-dispatches.
        if v_size != d_inner {
            bail!("GDN layout: num_v_heads * head_v_dim ({v_size}) != d_inner ({d_inner})");
        }
        ops.swiglu_f32(
            scratch.z_f32,
            scratch.out_normed,
            scratch.gated_f32,
            n_tokens * d_inner,
        )
        .context("gdn prefill swiglu_f32(z, out_normed)")?;
        ops.quantize_q8_1(scratch.gated_f32, scratch.gated_q8_1, n_tokens * d_inner)
            .context("gdn prefill quantise gated → Q8_1 (std)")?;
        ops.quantize_q8_1_mmq(scratch.gated_f32, scratch.gated_q8_1_mmq, d_inner, n_tokens)
            .context("gdn prefill quantise gated → Q8_1 (MMQ DS4)")?;

        // 15. ssm_out projection at M = L.
        ops.qmatmul(
            flambeau_ops::QmatmulBuffers {
                weights: self.ssm_out.ptr,
                act_q8_1: scratch.gated_q8_1,
                act_q8_1_mmq: scratch.gated_q8_1_mmq,
                dst: scratch.ssm_out_f32,
            },
            flambeau_ops::MatmulShape {
                m: n_tokens,
                k: d_inner,
                n: hidden,
            },
            self.ssm_out.dtype,
        )
        .context("gdn prefill ssm_out qmatmul")?;

        // ssm_out output is rank-local partial under TP. AR-sum it
        // before casting to F16. Caller passes `None` on SD/PP.
        if let Some(cb) = ar_partial_callback {
            cb(scratch.ssm_out_f32, n_tokens * hidden, device, stream)
                .context("gdn prefill ar_partial_callback (ssm_out F32)")?;
        }

        // 16. Cast back to F16 for the outer residual path.
        ops.cast_f32_to_f16(scratch.ssm_out_f32, delta_out, n_tokens * hidden)
            .context("gdn prefill cast ssm_out → f16")?;

        Ok(())
    }
}
