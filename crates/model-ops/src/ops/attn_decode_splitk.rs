//! Split-K (flash-decoding) decode attention, F16. Partitions the KV
//! context across `grid.y = n_chunks` to attack single-pass
//! occupancy starvation at long contexts. `chunk_size` controls the
//! partition; pick via `splitk_chunk_size(n_tokens_kv)`. F16 KV only.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32};
use crate::error::Result;
use crate::tensor::Tensor;

/// Output + split-K partial buffers for a splitk attention call. F16
/// output, F32 partials (m, s, o) shared across both F16-KV and Q8-KV
/// variants since splitk merges in F32.
pub struct SplitkOutputs<'a> {
    pub out: &'a mut Tensor<F16>,
    pub m: &'a mut Tensor<F32>,
    pub s: &'a mut Tensor<F32>,
    pub o: &'a mut Tensor<F32>,
}

/// Caller must size partials at `n_heads_q * n_chunks` (m, s) and
/// `n_heads_q * n_chunks * head_dim` (o), where `n_chunks =
/// ceil(n_tokens_kv / chunk_size)`.
pub fn attn_decode_f16_splitk(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    outputs: SplitkOutputs<'_>,
    shape: flambeau_ops::AttnSplitkShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &HipOps<'_>,
) -> Result<()> {
    let SplitkOutputs { out, m: partials_m, s: partials_s, o: partials_o } = outputs;
    let flambeau_ops::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
    } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_splitk: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || chunk_size == 0 {
        bail!(
            "attn_decode_f16_splitk: zero counts (n_heads_q={n_heads_q}, \
             n_heads_kv={n_heads_kv}, chunk_size={chunk_size})"
        );
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_splitk: n_heads_q ({n_heads_q}) must be divisible by \
             n_heads_kv ({n_heads_kv}) for GQA"
        );
    }
    let n_chunks = n_tokens_kv.div_ceil(chunk_size);
    let q_need = n_heads_q * head_dim;
    let cache_need = n_tokens_kv * n_heads_kv * head_dim;
    let ms_need = n_heads_q * n_chunks;
    let o_need = n_heads_q * n_chunks * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_decode_f16_splitk: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_decode_f16_splitk: k_cache has {} F16 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_decode_f16_splitk: v_cache has {} F16 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < q_need {
        bail!(
            "attn_decode_f16_splitk: out has {} F16 elems, need >= {q_need}",
            out.n_elems
        );
    }
    if partials_m.n_elems < ms_need {
        bail!(
            "attn_decode_f16_splitk: partials_m has {} F32 elems, need >= {ms_need}",
            partials_m.n_elems
        );
    }
    if partials_s.n_elems < ms_need {
        bail!(
            "attn_decode_f16_splitk: partials_s has {} F32 elems, need >= {ms_need}",
            partials_s.n_elems
        );
    }
    if partials_o.n_elems < o_need {
        bail!(
            "attn_decode_f16_splitk: partials_o has {} F32 elems, need >= {o_need}",
            partials_o.n_elems
        );
    }
    ops.attention_decode_f16_splitk(
        flambeau_ops::AttnBuffers {
            q: q.ptr,
            k: k_cache.ptr,
            v: v_cache.ptr,
            out: out.ptr,
        },
        flambeau_ops::AttnSplitkPartials {
            partials_m: partials_m.ptr,
            partials_s: partials_s.ptr,
            partials_o: partials_o.ptr,
        },
        shape,
        knobs,
    )
}

/// Same threshold as `flambeau_ops::hip::attention::splitk_chunk_size`
/// — re-exported for callers that need to size scratch.
pub fn splitk_chunk_size(n_tokens_kv: usize) -> usize {
    flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv)
}

/// S5a microbench — times `attn_decode_f16_splitk` at gemma4-31B-shaped
/// inputs (per-rank dims under TP=2). Compares head_dim ∈ {256, 512}
/// to characterise whether the head_dim=512 path is bandwidth-bound
/// (no S5b lever) or undertuned (lever exists). Run with:
///
/// ```sh
/// cargo test --release -p flambeau-model-ops --lib --features hip \
///     attn_microbench -- --ignored --nocapture
/// ```
#[cfg(test)]
mod attn_microbench {
    use super::*;
    use crate::testing::{test_device, test_ops_registry};
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
    use half::f16;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn attn_microbench_head_dim_256_vs_512() {
        const N_HEADS_Q: usize = 16;
        const N_HEADS_KV: usize = 8;
        const N_TOKENS_KV: usize = 1500;
        const N_ITERS: usize = 1000;
        const N_WARMUP: usize = 50;

        let device = test_device();
        device.bind().expect("bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        println!(
            "\n=== S5a microbench: attn_decode_f16_splitk @ gemma4-31B-Q4_0 / TP=2 / ctx {N_TOKENS_KV} ===\n  n_heads_q={N_HEADS_Q} n_heads_kv={N_HEADS_KV} n_iters={N_ITERS}\n"
        );

        for &head_dim in &[256usize, 512usize] {
            let q_n = N_HEADS_Q * head_dim;
            let cache_n = N_TOKENS_KV * N_HEADS_KV * head_dim;

            let q_host: Vec<f16> = (0..q_n)
                .map(|i| f16::from_f32(0.01 * ((i as f32) - (q_n as f32) / 2.0) / (q_n as f32)))
                .collect();
            let k_host: Vec<f16> = (0..cache_n)
                .map(|i| f16::from_f32(0.005 * ((i % 31) as f32)))
                .collect();
            let v_host: Vec<f16> = (0..cache_n)
                .map(|i| f16::from_f32(0.003 * ((i % 17) as f32)))
                .collect();

            let q_ptr = device.alloc(q_n * 2).unwrap();
            let k_ptr = device.alloc(cache_n * 2).unwrap();
            let v_ptr = device.alloc(cache_n * 2).unwrap();
            let out_ptr = device.alloc(q_n * 2).unwrap();
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        q_ptr,
                        DevicePtr(q_host.as_ptr() as usize),
                        q_n * 2,
                    )
                    .unwrap();
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        k_ptr,
                        DevicePtr(k_host.as_ptr() as usize),
                        cache_n * 2,
                    )
                    .unwrap();
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        v_ptr,
                        DevicePtr(v_host.as_ptr() as usize),
                        cache_n * 2,
                    )
                    .unwrap();
            }
            Stream::synchronize(stream).unwrap();

            let chunk_size = splitk_chunk_size(N_TOKENS_KV);
            let n_chunks = N_TOKENS_KV.div_ceil(chunk_size);
            let pm_n = N_HEADS_Q * n_chunks;
            let po_n = pm_n * head_dim;
            let pm_ptr = device.alloc(pm_n * 4).unwrap();
            let ps_ptr = device.alloc(pm_n * 4).unwrap();
            let po_ptr = device.alloc(po_n * 4).unwrap();

            let q_t = unsafe { Tensor::<F16>::from_raw(q_ptr, q_n) };
            let k_t = unsafe { Tensor::<F16>::from_raw(k_ptr, cache_n) };
            let v_t = unsafe { Tensor::<F16>::from_raw(v_ptr, cache_n) };
            let mut out_t = unsafe { Tensor::<F16>::from_raw(out_ptr, q_n) };
            let mut pm_t = unsafe { Tensor::<F32>::from_raw(pm_ptr, pm_n) };
            let mut ps_t = unsafe { Tensor::<F32>::from_raw(ps_ptr, pm_n) };
            let mut po_t = unsafe { Tensor::<F32>::from_raw(po_ptr, po_n) };

            let scale = 1.0 / (head_dim as f32).sqrt();

            for _ in 0..N_WARMUP {
                attn_decode_f16_splitk(
                    &q_t, &k_t, &v_t,
                    SplitkOutputs {
                        out: &mut out_t,
                        m: &mut pm_t,
                        s: &mut ps_t,
                        o: &mut po_t,
                    },
                    flambeau_ops::AttnSplitkShape {
                        n_heads_q: N_HEADS_Q,
                        n_heads_kv: N_HEADS_KV,
                        head_dim,
                        n_tokens_kv: N_TOKENS_KV,
                        chunk_size,
                    },
                    flambeau_ops::AttnKnobs { scale, window_size: 0, ring_depth: 0 },
                    &ops,
                )
                .unwrap();
            }
            Stream::synchronize(stream).unwrap();

            let t0 = Instant::now();
            for _ in 0..N_ITERS {
                attn_decode_f16_splitk(
                    &q_t, &k_t, &v_t,
                    SplitkOutputs {
                        out: &mut out_t,
                        m: &mut pm_t,
                        s: &mut ps_t,
                        o: &mut po_t,
                    },
                    flambeau_ops::AttnSplitkShape {
                        n_heads_q: N_HEADS_Q,
                        n_heads_kv: N_HEADS_KV,
                        head_dim,
                        n_tokens_kv: N_TOKENS_KV,
                        chunk_size,
                    },
                    flambeau_ops::AttnKnobs { scale, window_size: 0, ring_depth: 0 },
                    &ops,
                )
                .unwrap();
            }
            Stream::synchronize(stream).unwrap();
            let elapsed = t0.elapsed();
            let per_call_us = elapsed.as_micros() as f64 / N_ITERS as f64;

            // Per-call HBM traffic: K+V read = n_tokens × n_heads_kv × head_dim × 2 B × 2.
            let kv_bytes = (N_TOKENS_KV * N_HEADS_KV * head_dim * 2 * 2) as f64;
            let bandwidth_gb_s = kv_bytes / 1024.0 / 1024.0 / 1024.0 / (per_call_us / 1e6);

            println!(
                "head_dim={head_dim:>3}: per-call {per_call_us:>7.1} µs | KV read {:>5.1} MB | effective HBM BW {:>5.0} GB/s",
                kv_bytes / 1024.0 / 1024.0,
                bandwidth_gb_s,
            );

            unsafe {
                Device::dealloc(&device, q_ptr, q_n * 2).unwrap();
                Device::dealloc(&device, k_ptr, cache_n * 2).unwrap();
                Device::dealloc(&device, v_ptr, cache_n * 2).unwrap();
                Device::dealloc(&device, out_ptr, q_n * 2).unwrap();
                Device::dealloc(&device, pm_ptr, pm_n * 4).unwrap();
                Device::dealloc(&device, ps_ptr, pm_n * 4).unwrap();
                Device::dealloc(&device, po_ptr, po_n * 4).unwrap();
            }
        }
        println!(
            "\n  gfx906 peak HBM BW = ~1024 GB/s (HBM2 1.0 TB/s nominal).\n  Effective BW well below ceiling → kernel undertuned → S5b lever exists.\n  Effective BW near ceiling → bandwidth-bound → S5 is null.\n"
        );
    }
}
