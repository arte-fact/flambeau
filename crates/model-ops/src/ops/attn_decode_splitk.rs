//! Split-K (flash-decoding) decode attention, F16. Partitions the KV
//! context across `grid.y = n_chunks` to attack single-pass
//! occupancy starvation at long contexts. `chunk_size` controls the
//! partition; pick via `splitk_chunk_size(n_tokens_kv)`. F16 KV only.

use anyhow::bail;
use flambeau_ops::Ops;

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
    ops: &impl Ops,
) -> Result<()> {
    attn_decode_f16_splitk_slots(q, k_cache, v_cache, outputs, shape, knobs, ops, None)
}

/// Graph-captureable variant of [`attn_decode_f16_splitk`]. `slots=None` is
/// byte-identical to the wrapper; `Some` tags the chunk kernel's `n_tokens_kv`
/// + `chunk_base` scalars so a captured graph rebinds them per replay.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_f16_splitk_slots(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    outputs: SplitkOutputs<'_>,
    shape: flambeau_ops::AttnSplitkShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
    slots: Option<flambeau_ops::AttnSplitkSlots>,
) -> Result<()> {
    let SplitkOutputs { out, m: partials_m, s: partials_s, o: partials_o } = outputs;
    let flambeau_ops::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
        chunk_base,
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
    let n_chunks = (n_tokens_kv - chunk_base).div_ceil(chunk_size);
    let q_need = n_heads_q * head_dim;
    let cache_rows = crate::ops::kv_append::ring_cache_rows(n_tokens_kv, knobs.ring_depth as usize);
    let cache_need = cache_rows * n_heads_kv * head_dim;
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
    ops.attention_decode_f16_splitk_slots(
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
        slots,
    )
}

/// Device-position variant of [`attn_decode_f16_splitk`]. Reads `position` from
/// `pos_ptr` on-device (the decode-graph `[position, token_id]` context) and
/// derives the per-token `n_tokens_kv` / `chunk_base` / slid cache pointer from
/// the baked [`flambeau_ops::AttnDposKnobs`] regime, so a captured decode graph
/// rebinds nothing per token. `k_cache` / `v_cache` are the UNSLID slot base
/// pointers (the kernel slides in the SWA in-band regime).
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_f16_splitk_dpos(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    outputs: SplitkOutputs<'_>,
    shape: flambeau_ops::AttnSplitkDposShape,
    knobs: flambeau_ops::AttnDposKnobs,
    ops: &impl Ops,
    pos_ptr: flambeau_core::DevicePtr,
) -> Result<()> {
    let SplitkOutputs { out, m: partials_m, s: partials_s, o: partials_o } = outputs;
    let flambeau_ops::AttnSplitkDposShape { n_heads_q, n_heads_kv, head_dim, n_chunks, chunk_size } =
        shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_splitk_dpos: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || chunk_size == 0 || n_chunks == 0 {
        bail!(
            "attn_decode_f16_splitk_dpos: zero counts (n_heads_q={n_heads_q}, \
             n_heads_kv={n_heads_kv}, chunk_size={chunk_size}, n_chunks={n_chunks})"
        );
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_splitk_dpos: n_heads_q ({n_heads_q}) must be divisible by \
             n_heads_kv ({n_heads_kv}) for GQA"
        );
    }
    let q_need = n_heads_q * head_dim;
    let ms_need = n_heads_q * n_chunks;
    let o_need = ms_need * head_dim;
    if q.n_elems < q_need {
        bail!("attn_decode_f16_splitk_dpos: q has {} F16 elems, need >= {q_need}", q.n_elems);
    }
    if out.n_elems < q_need {
        bail!("attn_decode_f16_splitk_dpos: out has {} F16 elems, need >= {q_need}", out.n_elems);
    }
    if partials_m.n_elems < ms_need || partials_s.n_elems < ms_need {
        bail!("attn_decode_f16_splitk_dpos: partials_m/s too small, need >= {ms_need}");
    }
    if partials_o.n_elems < o_need {
        bail!(
            "attn_decode_f16_splitk_dpos: partials_o has {} F32 elems, need >= {o_need}",
            partials_o.n_elems
        );
    }
    let _ = (k_cache, v_cache);
    ops.attention_decode_f16_splitk_dpos(
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
        pos_ptr,
    )
}

/// Same threshold as `flambeau_ops::hip::attention::splitk_chunk_size`
/// — re-exported for callers that need to size scratch.
pub fn splitk_chunk_size(n_tokens_kv: usize, max_chunks: usize) -> usize {
    flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv, max_chunks)
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
    use flambeau_ops::HipOps;
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

            let chunk_size = splitk_chunk_size(N_TOKENS_KV, 32);
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
                        chunk_base: 0,
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
                        chunk_base: 0,
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

#[cfg(test)]
mod slots_parity {
    use super::*;
    use crate::testing::{alloc, download, free, test_device, test_ops_registry, upload};
    use flambeau_core::{Device, Stream};
    use flambeau_ops::HipOps;
    use half::f16;

    /// `attn_decode_f16_splitk_slots(None)` must produce the exact same
    /// device output as `attn_decode_f16_splitk` — the transparency contract
    /// for the graph-capture variant outside capture.
    #[test]
    fn splitk_slots_none_bit_identical() {
        const N_HEADS_Q: usize = 4;
        const N_HEADS_KV: usize = 2;
        const HEAD_DIM: usize = 128;
        const N_TOKENS_KV: usize = 12;
        const CHUNK_SIZE: usize = 4;
        const CHUNK_BASE: usize = 0;
        let scale = (HEAD_DIM as f32).sqrt().recip();

        let device = test_device();
        device.bind().expect("bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let n_chunks = (N_TOKENS_KV - CHUNK_BASE).div_ceil(CHUNK_SIZE);
        let q_n = N_HEADS_Q * HEAD_DIM;
        let cache_n = N_TOKENS_KV * N_HEADS_KV * HEAD_DIM;
        let ms_n = N_HEADS_Q * n_chunks;
        let o_n = ms_n * HEAD_DIM;

        let q_host: Vec<f16> =
            (0..q_n).map(|i| f16::from_f32(((i as f32) * 0.017 + 0.3).sin() * 0.5)).collect();
        let k_host: Vec<f16> =
            (0..cache_n).map(|i| f16::from_f32(((i as f32) * 0.023 - 0.4).cos() * 0.5)).collect();
        let v_host: Vec<f16> =
            (0..cache_n).map(|i| f16::from_f32(((i as f32) * 0.031 + 0.1).sin() * 0.5)).collect();

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host, q_n);
        let (k_t, k_ptr) = upload::<F16, f16>(&device, &k_host, cache_n);
        let (v_t, v_ptr) = upload::<F16, f16>(&device, &v_host, cache_n);

        let shape = flambeau_ops::AttnSplitkShape {
            n_heads_q: N_HEADS_Q,
            n_heads_kv: N_HEADS_KV,
            head_dim: HEAD_DIM,
            n_tokens_kv: N_TOKENS_KV,
            chunk_size: CHUNK_SIZE,
            chunk_base: CHUNK_BASE,
        };
        let knobs = flambeau_ops::AttnKnobs { scale, window_size: 0, ring_depth: 0 };

        let (mut out_a, out_a_ptr) = alloc::<F16>(&device, q_n);
        let (mut m_a, m_a_ptr) = alloc::<F32>(&device, ms_n);
        let (mut s_a, s_a_ptr) = alloc::<F32>(&device, ms_n);
        let (mut o_a, o_a_ptr) = alloc::<F32>(&device, o_n);
        attn_decode_f16_splitk(
            &q_t,
            &k_t,
            &v_t,
            SplitkOutputs { out: &mut out_a, m: &mut m_a, s: &mut s_a, o: &mut o_a },
            shape,
            knobs,
            &ops,
        )
        .expect("plain split-K");

        let (mut out_b, out_b_ptr) = alloc::<F16>(&device, q_n);
        let (mut m_b, m_b_ptr) = alloc::<F32>(&device, ms_n);
        let (mut s_b, s_b_ptr) = alloc::<F32>(&device, ms_n);
        let (mut o_b, o_b_ptr) = alloc::<F32>(&device, o_n);
        attn_decode_f16_splitk_slots(
            &q_t,
            &k_t,
            &v_t,
            SplitkOutputs { out: &mut out_b, m: &mut m_b, s: &mut s_b, o: &mut o_b },
            shape,
            knobs,
            &ops,
            None,
        )
        .expect("split-K slots=None");

        stream.synchronize().expect("sync");
        let a: Vec<f16> = download::<F16, f16>(&device, &out_a);
        let b: Vec<f16> = download::<F16, f16>(&device, &out_b);
        let a_bits: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
        let b_bits: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
        assert_eq!(a_bits, b_bits, "slots=None output must be bit-identical to plain");

        free(&device, q_ptr, q_t.bytes());
        free(&device, k_ptr, k_t.bytes());
        free(&device, v_ptr, v_t.bytes());
        free(&device, out_a_ptr, out_a.bytes());
        free(&device, m_a_ptr, m_a.bytes());
        free(&device, s_a_ptr, s_a.bytes());
        free(&device, o_a_ptr, o_a.bytes());
        free(&device, out_b_ptr, out_b.bytes());
        free(&device, m_b_ptr, m_b.bytes());
        free(&device, s_b_ptr, s_b.bytes());
        free(&device, o_b_ptr, o_b.bytes());
    }
}
