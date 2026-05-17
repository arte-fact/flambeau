//! `SingleDeviceForwardCtx` — single-GPU forward context.
//!
//! The simplest topology: one device, one stream, no AR, no peer-copy.
//! `layer_range` yields `0..num_layers`; every composite runs locally.
//! Validates the trait surface against real device work without any
//! topology machinery.
//!
//! State owned per request:
//! - HipDevice / HipStream / OpsRegistry references — borrowed from
//!   the model crate that constructed the ctx.
//! - `ScratchPool` — pre-allocated F16 / F32 / Q8_1 device slots, sized
//!   from a `ScratchConfig` at construction.
//! - `KvCache` per layer — F16 K + V buffers `[max_seq_len, kv_width]`
//!   per layer. The ctx owns these (allocated at construction) and
//!   tracks the per-layer write head separately via the `position`
//!   parameter passed to `standard_attn`.
//! - Host-side logits buffer populated by `output_head`.
//!
//! Aliasing safety: composites return `Tensor<F16>` handles that point
//! into the scratch pool. `&mut self` enforces sequential calls. The
//! caller-side convention is "consume the returned tensor before the
//! next composite call" — composites overwrite their dedicated slots.
//! For the residual stream specifically, two slots (a/b) ping-pong so
//! `residual_add(prev_resid, delta)` can read both without aliasing.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, I32, Q8_1};
use flambeau_ops::OpsRegistry;

use crate::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights,
    ModelLayout, MoeWeights,
};

/// Shape parameters needed to size the ctx's scratch + KV pools at
/// construction time. Carried separately from `ModelLayout` because
/// the model may set runtime maxes (e.g. cap `max_seq_len` per request)
/// that aren't part of the static layout.
#[derive(Clone, Copy, Debug)]
pub struct ScratchConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub q_width: usize,
    pub kv_width: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
}

/// Pre-allocated device scratch + per-layer KV cache for a single
/// inflight forward sequence.
pub struct ScratchPool {
    config: ScratchConfig,

    resid_a: DevicePtr,
    resid_b: DevicePtr,
    norm: DevicePtr,
    delta: DevicePtr,

    norm_q8_1: DevicePtr,
    q_f16: DevicePtr,
    k_f16: DevicePtr,
    v_f16: DevicePtr,
    attn_out_f16: DevicePtr,
    attn_out_q8_1: DevicePtr,
    attn_proj_f32: DevicePtr,

    gate_f32: DevicePtr,
    up_f32: DevicePtr,
    gated_f16: DevicePtr,
    gated_q8_1: DevicePtr,
    down_f32: DevicePtr,

    logits_f32_dev: DevicePtr,
    position_i32: DevicePtr,

    kv_caches: Vec<KvCache>,

    allocs: Vec<(DevicePtr, usize)>,
}

/// One layer's KV cache: F16 `[max_seq_len, kv_width]` each.
#[derive(Clone, Copy)]
struct KvCache {
    k: DevicePtr,
    v: DevicePtr,
}

impl ScratchPool {
    /// Allocate every device buffer the ctx will ever need for this
    /// shape. Caller must `dispose(device)` before drop to release HBM.
    pub fn new(device: &HipDevice, config: ScratchConfig) -> Result<Self> {
        let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
        let mut alloc_bytes = |bytes: usize| -> Result<DevicePtr> {
            let p = device.alloc(bytes).context("alloc")?;
            allocs.push((p, bytes));
            Ok(p)
        };

        let f16 = 2;
        let f32 = 4;
        let i32_b = 4;
        let q8_1 = |n: usize| n.div_ceil(32) * 36;

        let h = config.hidden;
        let m = config.intermediate;
        let qw = config.q_width;
        let kvw = config.kv_width;

        let resid_a = alloc_bytes(h * f16)?;
        let resid_b = alloc_bytes(h * f16)?;
        let norm = alloc_bytes(h * f16)?;
        let delta = alloc_bytes(h * f16)?;

        let norm_q8_1 = alloc_bytes(q8_1(h))?;
        let q_f16 = alloc_bytes(qw * f16)?;
        let k_f16 = alloc_bytes(kvw * f16)?;
        let v_f16 = alloc_bytes(kvw * f16)?;
        let attn_out_f16 = alloc_bytes(qw * f16)?;
        let attn_out_q8_1 = alloc_bytes(q8_1(qw))?;
        // Reused as Q (qw F32), K/V (kvw F32), and output-proj (h F32) target — size to the max.
        let attn_proj_f32 = alloc_bytes(qw.max(kvw).max(h) * f32)?;

        let gate_f32 = alloc_bytes(m * f32)?;
        let up_f32 = alloc_bytes(m * f32)?;
        let gated_f16 = alloc_bytes(m * f16)?;
        let gated_q8_1 = alloc_bytes(q8_1(m))?;
        let down_f32 = alloc_bytes(h * f32)?;

        let logits_f32_dev = alloc_bytes(config.vocab * f32)?;
        let position_i32 = alloc_bytes(i32_b)?;

        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let k = alloc_bytes(config.max_seq_len * kvw * f16)?;
            let v = alloc_bytes(config.max_seq_len * kvw * f16)?;
            kv_caches.push(KvCache { k, v });
        }

        Ok(Self {
            config,
            resid_a,
            resid_b,
            norm,
            delta,
            norm_q8_1,
            q_f16,
            k_f16,
            v_f16,
            attn_out_f16,
            attn_out_q8_1,
            attn_proj_f32,
            gate_f32,
            up_f32,
            gated_f16,
            gated_q8_1,
            down_f32,
            logits_f32_dev,
            position_i32,
            kv_caches,
            allocs,
        })
    }

    /// Free every device buffer this pool allocated. Safe to call once;
    /// repeated calls are no-ops because `allocs` is drained.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: `ptr` came from `device.alloc(bytes)` above; not
            // freed elsewhere, not aliased.
            unsafe { device.dealloc(ptr, bytes) }.context("dealloc")?;
        }
        Ok(())
    }
}

/// Single-GPU forward context.
pub struct SingleDeviceForwardCtx<'a> {
    pub device: &'a HipDevice,
    pub stream: &'a HipStream,
    pub reg: &'a OpsRegistry,
    pub pool: &'a mut ScratchPool,

    /// Which residual slot (`a` or `b`) currently holds the live
    /// residual stream. `embed` and `residual_add` flip it.
    current_residual_is_a: bool,

    /// Host-side logits buffer populated by `output_head`. Sized to
    /// `vocab` at first `output_head` call (or pre-sized if the model
    /// constructs the ctx with `with_logits_capacity`).
    logits_host: Vec<f32>,
}

impl<'a> SingleDeviceForwardCtx<'a> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
    ) -> Self {
        Self {
            device,
            stream,
            reg,
            pool,
            current_residual_is_a: true,
            logits_host: Vec::new(),
        }
    }

    /// Reset KV write heads + residual slot selection between forward
    /// passes. Does NOT zero the K/V buffers — caller's invariant is
    /// that the per-layer `position` arg passed to `standard_attn`
    /// matches a freshly-empty cache.
    pub fn reset(&mut self) {
        self.current_residual_is_a = true;
    }

    fn flambeau_ops(&self) -> flambeau_ops::HipOps<'a> {
        flambeau_ops::HipOps::new(self.reg, self.stream)
    }

    fn h(&self) -> usize {
        self.pool.config.hidden
    }

    fn next_residual_slot(&mut self) -> DevicePtr {
        self.current_residual_is_a = !self.current_residual_is_a;
        if self.current_residual_is_a {
            self.pool.resid_a
        } else {
            self.pool.resid_b
        }
    }

    fn slot_f16(&self, ptr: DevicePtr, n_elems: usize) -> Tensor<F16> {
        // SAFETY: `ptr` is one of the pool's pre-allocated F16 slots,
        // sized for at least `n_elems` F16 elements.
        unsafe { Tensor::<F16>::from_raw(ptr, n_elems) }
    }
}

impl ForwardCtx for SingleDeviceForwardCtx<'_> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        let hidden = self.h();
        if hidden != weights.hidden {
            bail!(
                "embed: ctx hidden {hidden} != weights.hidden {}",
                weights.hidden
            );
        }
        if (token_id as usize) >= weights.vocab_size {
            bail!(
                "embed: token_id {token_id} >= vocab_size {}",
                weights.vocab_size
            );
        }
        if weights.token_embd.n_elems < weights.vocab_size * hidden {
            bail!(
                "embed: token_embd has {} F16 elems, need >= {}",
                weights.token_embd.n_elems,
                weights.vocab_size * hidden
            );
        }
        // Embedding is F16 on device; one F16 row DtoD memcpy.
        let row_bytes = hidden * 2;
        let src = weights
            .token_embd
            .ptr
            .offset_bytes((token_id as usize) * row_bytes);
        let dst = self.next_residual_slot();
        // SAFETY: src points at >= row_bytes valid F16 weight bytes;
        // dst is a pool slot sized for hidden F16 elems; stream is live.
        unsafe {
            self.device.memcpy_async(
                self.stream,
                CopyDirection::DeviceToDevice,
                dst,
                src,
                row_bytes,
            ).context("embed: DtoD row memcpy")?;
        }
        Ok(self.slot_f16(dst, hidden))
    }

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>> {
        let hidden = self.h();
        let mut out = self.slot_f16(self.pool.norm, hidden);
        let ops = self.flambeau_ops();
        flambeau_model_ops::rmsnorm_f16(input, weight, &mut out, 1, hidden, eps, &ops)?;
        Ok(out)
    }

    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>> {
        let hidden = self.h();
        let out_ptr = self.next_residual_slot();
        let mut out = self.slot_f16(out_ptr, hidden);
        let ops = self.flambeau_ops();
        flambeau_model_ops::add_f16(&a, &b, &mut out, hidden, &ops)?;
        Ok(out)
    }

    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>> {
        let hidden = self.h();
        if layer_idx >= self.pool.kv_caches.len() {
            bail!(
                "standard_attn: layer_idx {layer_idx} >= num_layers {}",
                self.pool.kv_caches.len()
            );
        }
        if position >= self.pool.config.max_seq_len {
            bail!(
                "standard_attn: position {position} >= max_seq_len {}",
                self.pool.config.max_seq_len
            );
        }
        let q_width = weights.n_heads * weights.head_dim;
        let kv_width = weights.n_kv_heads * weights.head_dim;
        if q_width != self.pool.config.q_width {
            bail!(
                "standard_attn: weights q_width {q_width} != ctx.q_width {}",
                self.pool.config.q_width
            );
        }
        if kv_width != self.pool.config.kv_width {
            bail!(
                "standard_attn: weights kv_width {kv_width} != ctx.kv_width {}",
                self.pool.config.kv_width
            );
        }

        let ops = self.flambeau_ops();

        // 1. rmsnorm-quant: input → Q8_1 row.
        let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(self.pool.norm_q8_1, hidden) };
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &weights.attn_norm,
            &mut norm_q8_1,
            1,
            hidden,
            weights.rms_eps,
            &ops,
        )?;

        // 2. Q/K/V projections — F32 output, then cast back to F16.
        let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
        let q_f32_buf = self.pool.attn_proj_f32;
        let mut q_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, q_width) };
        weights
            .attn_q
            .qmatmul(&norm_q8_1, &act_mmq_null, &mut q_f32, 1, hidden, q_width, &ops)?;
        let mut q_f16 = unsafe { Tensor::<F16>::from_raw(self.pool.q_f16, q_width) };
        flambeau_model_ops::cast_f32_to_f16(&q_f32, &mut q_f16, q_width, &ops)?;

        // Reuse the F32 buffer for K — sized at construction to max(q_width, kv_width, hidden) F32.
        let mut k_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, kv_width) };
        weights
            .attn_k
            .qmatmul(&norm_q8_1, &act_mmq_null, &mut k_f32, 1, hidden, kv_width, &ops)?;
        let mut k_f16 = unsafe { Tensor::<F16>::from_raw(self.pool.k_f16, kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&k_f32, &mut k_f16, kv_width, &ops)?;

        let mut v_f32 = unsafe { Tensor::<F32>::from_raw(q_f32_buf, kv_width) };
        weights
            .attn_v
            .qmatmul(&norm_q8_1, &act_mmq_null, &mut v_f32, 1, hidden, kv_width, &ops)?;
        let mut v_f16 = unsafe { Tensor::<F16>::from_raw(self.pool.v_f16, kv_width) };
        flambeau_model_ops::cast_f32_to_f16(&v_f32, &mut v_f16, kv_width, &ops)?;
        let _ = q_f32_buf; // silence unused after final reuse

        // 3. Optional Q/K norm (qwen3.x; gemma4 attn-norm path).
        if let Some(q_norm_w) = weights.attn_q_norm.as_ref() {
            // Per-head rmsnorm: treat q_f16 as n_heads rows of head_dim.
            let q_normed = unsafe { Tensor::<F16>::from_raw(self.pool.q_f16, q_width) };
            // Re-read and write through self.pool.q_f16 in place. The
            // existing rmsnorm_f16 doesn't support in-place, so route
            // through `norm` scratch as a temporary.
            let mut tmp = unsafe { Tensor::<F16>::from_raw(self.pool.attn_out_f16, q_width) };
            flambeau_model_ops::rmsnorm_f16(
                &q_normed,
                q_norm_w,
                &mut tmp,
                weights.n_heads,
                weights.head_dim,
                weights.rms_eps,
                &ops,
            )?;
            // DtoD memcpy tmp → q_f16.
            let bytes = q_width * 2;
            unsafe {
                self.device
                    .memcpy_async(
                        self.stream,
                        CopyDirection::DeviceToDevice,
                        self.pool.q_f16,
                        tmp.ptr,
                        bytes,
                    )
                    .context("standard_attn: q_norm DtoD copy back")?;
            }
            let _ = q_normed;
        }
        if let Some(k_norm_w) = weights.attn_k_norm.as_ref() {
            let k_normed = unsafe { Tensor::<F16>::from_raw(self.pool.k_f16, kv_width) };
            let mut tmp = unsafe { Tensor::<F16>::from_raw(self.pool.attn_out_f16, kv_width) };
            flambeau_model_ops::rmsnorm_f16(
                &k_normed,
                k_norm_w,
                &mut tmp,
                weights.n_kv_heads,
                weights.head_dim,
                weights.rms_eps,
                &ops,
            )?;
            let bytes = kv_width * 2;
            unsafe {
                self.device
                    .memcpy_async(
                        self.stream,
                        CopyDirection::DeviceToDevice,
                        self.pool.k_f16,
                        tmp.ptr,
                        bytes,
                    )
                    .context("standard_attn: k_norm DtoD copy back")?;
            }
            let _ = k_normed;
        }

        // 4. RoPE on Q and K. Position tensor lives in position_i32 (1 elem).
        let pos_val = [position as i32];
        // SAFETY: `position_i32` is a 4-byte device slot; pos_val is one i32.
        unsafe {
            self.device.memcpy_async(
                self.stream,
                CopyDirection::HostToDevice,
                self.pool.position_i32,
                DevicePtr(pos_val.as_ptr() as usize),
                4,
            ).context("standard_attn: positions HtoD")?;
        }
        let positions = unsafe { Tensor::<I32>::from_raw(self.pool.position_i32, 1) };
        let mut q_f16_rope = unsafe { Tensor::<F16>::from_raw(self.pool.q_f16, q_width) };
        let mut k_f16_rope = unsafe { Tensor::<F16>::from_raw(self.pool.k_f16, kv_width) };
        if weights.rotated_dims == weights.head_dim {
            flambeau_model_ops::rope_f16(
                &mut q_f16_rope,
                &positions,
                weights.rope_theta,
                1,
                weights.n_heads,
                weights.head_dim,
                &ops,
            )?;
            flambeau_model_ops::rope_f16(
                &mut k_f16_rope,
                &positions,
                weights.rope_theta,
                1,
                weights.n_kv_heads,
                weights.head_dim,
                &ops,
            )?;
        } else {
            flambeau_model_ops::rope_neox_partial_f16(
                &mut q_f16_rope,
                &positions,
                weights.rope_theta,
                1,
                weights.n_heads,
                weights.head_dim,
                weights.rotated_dims,
                &ops,
            )?;
            flambeau_model_ops::rope_neox_partial_f16(
                &mut k_f16_rope,
                &positions,
                weights.rope_theta,
                1,
                weights.n_kv_heads,
                weights.head_dim,
                weights.rotated_dims,
                &ops,
            )?;
        }

        // 5. KV append at row `position`.
        let kv = self.pool.kv_caches[layer_idx];
        let mut k_cache = unsafe {
            Tensor::<F16>::from_raw(kv.k, self.pool.config.max_seq_len * kv_width)
        };
        let mut v_cache = unsafe {
            Tensor::<F16>::from_raw(kv.v, self.pool.config.max_seq_len * kv_width)
        };
        flambeau_model_ops::kv_append_f16(
            &k_f16_rope,
            &v_f16,
            &mut k_cache,
            &mut v_cache,
            1,
            kv_width,
            position,
            self.pool.config.max_seq_len,
            self.device,
            self.stream,
        )?;

        // 6. Attention decode against the populated cache (rows [0, position+1)).
        let n_tokens_kv = position + 1;
        let mut attn_out =
            unsafe { Tensor::<F16>::from_raw(self.pool.attn_out_f16, q_width) };
        let scale = weights
            .softmax_scale
            .unwrap_or_else(|| (weights.head_dim as f32).sqrt().recip());
        flambeau_model_ops::attn_decode_f16(
            &q_f16_rope,
            &k_cache,
            &v_cache,
            &mut attn_out,
            weights.n_heads,
            weights.n_kv_heads,
            weights.head_dim,
            n_tokens_kv,
            scale,
            weights.window_size,
            &ops,
        )?;

        // 7. Quantise attn_out for output projection.
        let mut attn_out_q8_1 = unsafe {
            Tensor::<Q8_1>::from_raw(self.pool.attn_out_q8_1, q_width)
        };
        flambeau_model_ops::quantize_f16_to_q8_1(&attn_out, &mut attn_out_q8_1, q_width, &ops)?;

        // 8. Output projection: F32 result, cast back to F16 in `delta`.
        let mut proj_f32 =
            unsafe { Tensor::<F32>::from_raw(self.pool.attn_proj_f32, hidden) };
        weights.attn_output.qmatmul(
            &attn_out_q8_1,
            &act_mmq_null,
            &mut proj_f32,
            1,
            q_width,
            hidden,
            &ops,
        )?;
        let mut delta = unsafe { Tensor::<F16>::from_raw(self.pool.delta, hidden) };
        flambeau_model_ops::cast_f32_to_f16(&proj_f32, &mut delta, hidden, &ops)?;
        Ok(delta)
    }

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>> {
        let hidden = self.h();
        let m = self.pool.config.intermediate;
        let ops = self.flambeau_ops();

        // 1. rmsnorm-quant.
        let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(self.pool.norm_q8_1, hidden) };
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &weights.ffn_norm,
            &mut norm_q8_1,
            1,
            hidden,
            weights.rms_eps,
            &ops,
        )?;
        let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };

        // 2. gate, up projections → F32.
        let mut gate_f32 = unsafe { Tensor::<F32>::from_raw(self.pool.gate_f32, m) };
        weights
            .ffn_gate
            .qmatmul(&norm_q8_1, &act_mmq_null, &mut gate_f32, 1, hidden, m, &ops)?;
        let mut up_f32 = unsafe { Tensor::<F32>::from_raw(self.pool.up_f32, m) };
        weights
            .ffn_up
            .qmatmul(&norm_q8_1, &act_mmq_null, &mut up_f32, 1, hidden, m, &ops)?;

        // 3. Activate (gate, up) → F16.
        let mut gated_f16 = unsafe { Tensor::<F16>::from_raw(self.pool.gated_f16, m) };
        match weights.activation {
            Activation::SwiGLU => {
                flambeau_model_ops::swiglu_f32_to_f16(&gate_f32, &up_f32, &mut gated_f16, m, &ops)?;
            }
            Activation::GeluTanh => {
                flambeau_model_ops::gelu_mul_f32_to_f16(
                    &gate_f32,
                    &up_f32,
                    &mut gated_f16,
                    m,
                    &ops,
                )?;
            }
        }

        // 4. Quantise activated → Q8_1, down projection → F32, cast to delta.
        let mut gated_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(self.pool.gated_q8_1, m) };
        flambeau_model_ops::quantize_f16_to_q8_1(&gated_f16, &mut gated_q8_1, m, &ops)?;
        let mut down_f32 = unsafe { Tensor::<F32>::from_raw(self.pool.down_f32, hidden) };
        weights
            .ffn_down
            .qmatmul(&gated_q8_1, &act_mmq_null, &mut down_f32, 1, m, hidden, &ops)?;
        let mut delta = unsafe { Tensor::<F16>::from_raw(self.pool.delta, hidden) };
        flambeau_model_ops::cast_f32_to_f16(&down_f32, &mut delta, hidden, &ops)?;
        Ok(delta)
    }

    fn moe_ffn(&mut self, _input: &Tensor<F16>, _weights: &MoeWeights) -> Result<Tensor<F16>> {
        bail!("SingleDeviceForwardCtx::moe_ffn — not implemented for P2; lands with qwen3.6-v2 in P7")
    }

    fn output_head(
        &mut self,
        input: &Tensor<F16>,
        lm_head: &LmHeadWeights,
    ) -> Result<()> {
        let hidden = self.h();
        if hidden != lm_head.hidden {
            bail!(
                "output_head: ctx hidden {hidden} != lm_head.hidden {}",
                lm_head.hidden
            );
        }
        let vocab = lm_head.vocab_size;
        let ops = self.flambeau_ops();

        // 1. rmsnorm-quant of final residual.
        let mut norm_q8_1 = unsafe { Tensor::<Q8_1>::from_raw(self.pool.norm_q8_1, hidden) };
        flambeau_model_ops::rmsnorm_quant_q8_1(
            input,
            &lm_head.output_norm,
            &mut norm_q8_1,
            1,
            hidden,
            lm_head.rms_eps,
            &ops,
        )?;

        // 2. LM-head matmul: hidden → vocab, F32 output.
        let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };
        let mut logits_f32 =
            unsafe { Tensor::<F32>::from_raw(self.pool.logits_f32_dev, vocab) };
        lm_head.lm_head.qmatmul(
            &norm_q8_1,
            &act_mmq_null,
            &mut logits_f32,
            1,
            hidden,
            vocab,
            &ops,
        )?;

        // 3. Optional softcap (gemma4 — left as TODO for P8; qwen35 won't hit this).
        if let Some(_cap) = lm_head.final_logit_softcap {
            bail!(
                "output_head: final_logit_softcap not implemented in P2; needed by gemma4 in P8"
            );
        }

        // 4. DtoH download into the host logits buffer.
        if self.logits_host.len() != vocab {
            self.logits_host = vec![0.0_f32; vocab];
        }
        let bytes = vocab * 4;
        // SAFETY: logits_f32_dev has `vocab*4` bytes; host buffer sized to match.
        unsafe {
            self.device.memcpy_async(
                self.stream,
                CopyDirection::DeviceToHost,
                DevicePtr(self.logits_host.as_mut_ptr() as usize),
                self.pool.logits_f32_dev,
                bytes,
            ).context("output_head: logits DtoH")?;
        }
        // Sync so the host slice is observable on return.
        flambeau_core::Stream::synchronize(self.stream)?;
        Ok(())
    }

    fn layer_range<'b>(
        &'b mut self,
        layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        Box::new(0..layout.num_layers)
    }

    fn logits(&self) -> &[f32] {
        &self.logits_host
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Pod;
    use flambeau_core::Stream;
    use flambeau_quant::quantize_k::quantize_row_q8_0;
    use half::f16;

    struct DeviceAllocs {
        device: HipDevice,
        allocs: Vec<(DevicePtr, usize)>,
    }

    impl DeviceAllocs {
        fn new(device: HipDevice) -> Self {
            Self {
                device,
                allocs: Vec::new(),
            }
        }

        fn upload<P: Pod>(&mut self, host: &[P]) -> (DevicePtr, usize) {
            let bytes = std::mem::size_of_val(host);
            let ptr = self.device.alloc(bytes).expect("alloc");
            let stream = self.device.default_stream();
            unsafe {
                self.device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(host.as_ptr() as usize),
                        bytes,
                    )
                    .expect("memcpy HtoD");
            }
            stream.synchronize().expect("sync");
            self.allocs.push((ptr, bytes));
            (ptr, bytes)
        }

        fn upload_f16(&mut self, host_f32: &[f32]) -> Tensor<F16> {
            let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let (ptr, _) = self.upload(&host_f16);
            unsafe { Tensor::<F16>::from_raw(ptr, host_f16.len()) }
        }

        fn upload_q8_0(
            &mut self,
            host_f32: &[f32],
            rows: usize,
            cols: usize,
        ) -> crate::ctx::QuantWeight {
            assert_eq!(host_f32.len(), rows * cols);
            assert!(cols % 32 == 0, "Q8_0 needs cols % 32 == 0");
            let mut bytes: Vec<u8> = Vec::with_capacity(rows * cols / 32 * 34);
            for r in 0..rows {
                quantize_row_q8_0(&host_f32[r * cols..(r + 1) * cols], &mut bytes);
            }
            let (ptr, _) = self.upload(&bytes);
            let tensor =
                unsafe { Tensor::<flambeau_model_ops::Q8_0>::from_raw(ptr, rows * cols) };
            crate::ctx::QuantWeight::Q8_0(tensor)
        }
    }

    impl Drop for DeviceAllocs {
        fn drop(&mut self) {
            for (ptr, bytes) in self.allocs.drain(..) {
                unsafe {
                    let _ = self.device.dealloc(ptr, bytes);
                }
            }
        }
    }

    fn det_signal(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let s = (seed as f32) * 0.013 + (i as f32) * 0.027;
                (s.sin() + (s * 1.7).cos()) * 0.1
            })
            .collect()
    }

    #[test]
    fn single_device_synthetic_qwen_dense_one_token_forward() {
        // Tiny qwen-shaped dense model.
        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const INTERMEDIATE: usize = 256;
        const NUM_LAYERS: usize = 2;
        const N_HEADS: usize = 4;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const MAX_SEQ_LEN: usize = 16;
        const RMS_EPS: f32 = 1e-5;

        let q_width = N_HEADS * HEAD_DIM;
        let kv_width = N_KV_HEADS * HEAD_DIM;

        let device = HipDevice::new(0).expect("HipDevice 0");
        device.bind().expect("bind");
        let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
        let stream = device.default_stream();
        let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

        let embd = EmbeddingWeights {
            token_embd: allocs.upload_f16(&det_signal(VOCAB * HIDDEN, 1)),
            vocab_size: VOCAB,
            hidden: HIDDEN,
        };
        let lm_head_weights = LmHeadWeights {
            output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
            final_logit_softcap: None,
            vocab_size: VOCAB,
            hidden: HIDDEN,
            rms_eps: RMS_EPS,
        };

        let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(NUM_LAYERS);
        let mut ffn_weights: Vec<FfnWeights> = Vec::with_capacity(NUM_LAYERS);
        for li in 0..NUM_LAYERS {
            let seed = 100 + (li as u32) * 10;
            attn_weights.push(AttnWeights {
                attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
                attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 1), q_width, HIDDEN),
                attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 2), kv_width, HIDDEN),
                attn_v: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 3), kv_width, HIDDEN),
                attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, seed + 4), HIDDEN, q_width),
                attn_q_norm: None,
                attn_k_norm: None,
                n_heads: N_HEADS,
                n_kv_heads: N_KV_HEADS,
                head_dim: HEAD_DIM,
                rotated_dims: HEAD_DIM,
                rope_theta: 10000.0,
                window_size: 0,
                rms_eps: RMS_EPS,
                softmax_scale: None,
            });
            ffn_weights.push(FfnWeights {
                ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
                ffn_gate: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 5), INTERMEDIATE, HIDDEN),
                ffn_up: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 6), INTERMEDIATE, HIDDEN),
                ffn_down: allocs.upload_q8_0(&det_signal(HIDDEN * INTERMEDIATE, seed + 7), HIDDEN, INTERMEDIATE),
                activation: Activation::SwiGLU,
                rms_eps: RMS_EPS,
            });
        }

        let cfg = ScratchConfig {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            q_width,
            kv_width,
            vocab: VOCAB,
            max_seq_len: MAX_SEQ_LEN,
            num_layers: NUM_LAYERS,
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
        let layout = ModelLayout {
            num_layers: NUM_LAYERS,
            hidden: HIDDEN,
            kv_max_seq_len: MAX_SEQ_LEN,
        };

        {
            let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

            let token_id: u32 = 7;
            let position: usize = 0;

            let mut resid = ctx.embed(&embd, token_id).expect("embed");
            let layers: Vec<usize> = ctx.layer_range(&layout).collect();
            assert_eq!(layers, vec![0, 1]);

            for li in layers {
                // Attention block.
                let normed = ctx.rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS)
                    .expect("attn rmsnorm");
                let delta = ctx
                    .standard_attn(&normed, &attn_weights[li], li, position)
                    .expect("standard_attn");
                resid = ctx.residual_add(resid, delta).expect("attn residual_add");

                // FFN block.
                let normed = ctx.rmsnorm(&resid, &ffn_weights[li].ffn_norm, RMS_EPS)
                    .expect("ffn rmsnorm");
                let delta = ctx.dense_ffn(&normed, &ffn_weights[li]).expect("dense_ffn");
                resid = ctx.residual_add(resid, delta).expect("ffn residual_add");
            }

            ctx.output_head(&resid, &lm_head_weights).expect("output_head");
            let logits = ctx.logits();
            assert_eq!(logits.len(), VOCAB, "logits len mismatch");
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "logits[{i}] = {l} is not finite (NaN/Inf in forward)"
                );
            }
            // Sanity: not all identical.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(
                max - min > 1e-3,
                "logits collapsed to a constant: min={min} max={max}"
            );
        }

        pool.dispose(&device).expect("pool dispose");
        // `allocs` Drop frees the weight buffers.
    }
}

