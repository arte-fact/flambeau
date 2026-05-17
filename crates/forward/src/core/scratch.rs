//! Pre-allocated device scratch + per-layer KV cache for a single
//! inflight forward sequence.

use anyhow::{Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, DevicePtr};

/// Shape parameters needed to size the ctx's scratch + KV pools at
/// construction time.
///
/// `q_width` / `kv_width` are the per-rank widths under TP (model
/// width divided by tp_size); under SingleDevice / PP they are the
/// model widths. The pool doesn't know which — the caller computes
/// them appropriately.
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

/// One layer's KV cache: F16 `[max_seq_len, kv_width]` for K and V.
#[derive(Clone, Copy)]
pub struct KvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
}

/// Pre-allocated device buffers used by every composite. Caller must
/// invoke `dispose(device)` before drop to release HBM.
///
/// Holds the ping-pong residual selector (`current_residual_is_a`)
/// because that's shared across composites — `embed` flips it, every
/// `residual_add` flips it.
pub struct ScratchPool {
    pub config: ScratchConfig,

    pub resid_a: DevicePtr,
    pub resid_b: DevicePtr,
    pub norm: DevicePtr,
    pub delta: DevicePtr,

    pub norm_q8_1: DevicePtr,
    pub q_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub attn_out_q8_1: DevicePtr,
    pub attn_proj_f32: DevicePtr,

    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub gated_f16: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub down_f32: DevicePtr,

    pub logits_f32_dev: DevicePtr,
    pub position_i32: DevicePtr,

    pub kv_caches: Vec<KvCache>,

    pub current_residual_is_a: bool,

    allocs: Vec<(DevicePtr, usize)>,
}

impl ScratchPool {
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
            current_residual_is_a: true,
            allocs,
        })
    }

    /// Free every device buffer this pool allocated. Idempotent.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: `ptr` came from `device.alloc(bytes)`; not freed
            // elsewhere, not aliased (caller's invariant for dispose is
            // "no further forward calls").
            unsafe { device.dealloc(ptr, bytes) }.context("dealloc")?;
        }
        Ok(())
    }

    /// Flip the residual-slot selector and return the new live slot's
    /// device pointer. Composites that write a fresh residual call this
    /// (embed, residual_add).
    pub fn next_residual_slot(&mut self) -> DevicePtr {
        self.current_residual_is_a = !self.current_residual_is_a;
        if self.current_residual_is_a {
            self.resid_a
        } else {
            self.resid_b
        }
    }
}
