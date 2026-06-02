//! attention prefill with Q8_0 KV cache — correctness cert.
//!
//! Covers both impl paths in `attention_prefill_q8_kv`'s launcher:
//!   * Flash-tile fast path (BR=4/8 LDS-tiled) for head_dim ∈ {64, 128, 256}
//!     when `n_q_tokens >= 4`.
//!   * Oracle path for head_dim=512 and the short-prompt fallback
//!     (`n_q_tokens < 4`).
//!
//! Critically includes `window_size > 0` shapes that straddle the SWA
//! boundary — the case the deterministic-decode rig surfaced as the
//! gemma `<pad>` regression. The reference F32 attention applies the
//! same SWA mask, so any error is kernel arithmetic, not quant noise.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a kernel launch or a memcpy_async \
              over buffers allocated locally in the same function and freed before \
              return; invariant is uniform across all sites."
)]

use std::path::Path;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{
    device_count, FuncAttributes, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

use crate::cert::{now_utc_iso8601, Cert, PmcSnapshot, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig, seeded_f32_range};

const QK: usize = QK8_0;

/// Dispatch label — selects which kernel symbol + launch shape to use.
#[derive(Clone, Copy)]
enum Impl {
    OracleD64,
    OracleD128,
    OracleD256,
    OracleD512,
    FlashTileD64Br4,
    FlashTileD128Br4,
    FlashTileD256Br8,
}

impl Impl {
    fn label(self) -> &'static str {
        match self {
            Impl::OracleD64 => "oracle_d64",
            Impl::OracleD128 => "oracle_d128",
            Impl::OracleD256 => "oracle_d256",
            Impl::OracleD512 => "oracle_d512",
            Impl::FlashTileD64Br4 => "flash_tile_d64_br4",
            Impl::FlashTileD128Br4 => "flash_tile_d128_br4",
            Impl::FlashTileD256Br8 => "flash_tile_d256_br8",
        }
    }

    fn is_flash_tile(self) -> bool {
        matches!(
            self,
            Impl::FlashTileD64Br4 | Impl::FlashTileD128Br4 | Impl::FlashTileD256Br8
        )
    }

    fn br(self) -> u32 {
        match self {
            Impl::FlashTileD256Br8 => 8,
            Impl::FlashTileD64Br4 | Impl::FlashTileD128Br4 => 4,
            _ => 1, // unused for oracle
        }
    }
}

struct Case {
    head_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    n_q_tokens: usize,
    n_k_tokens: usize,
    q_offset: usize,
    window_size: i32,
    impl_kind: Impl,
}

pub fn run_sweep(repo_root: &Path) -> Result<Cert> {
    let n = device_count().context("hipGetDeviceCount")?;
    if n < 1 {
        bail!("no HIP devices");
    }
    let dev = HipDevice::new(0)?;
    dev.bind()?;

    let kb_oracle = kernels::hsaco("attention_prefill_q8_kv").unwrap();
    let mod_oracle = HipModule::load(dev.id(), kb_oracle)?;
    let k_oracle = mod_oracle.kernel("flambeau_attention_prefill_q8_kv")?;
    let attrs_oracle: FuncAttributes = k_oracle.attributes()?;

    let kb_tile = kernels::hsaco("attention_prefill_flash_tile_q8_kv").unwrap();
    let mod_tile = HipModule::load(dev.id(), kb_tile)?;
    let k_tile_d64 = mod_tile.kernel("flambeau_attention_prefill_flash_tile_d64_q8_kv")?;
    let k_tile_d128 = mod_tile.kernel("flambeau_attention_prefill_flash_tile_d128_q8_kv")?;
    let k_tile_d256 = mod_tile.kernel("flambeau_attention_prefill_flash_tile_d256_br8_q8_kv")?;

    // (head_dim, n_heads_q, n_heads_kv, n_q, n_k, q_offset, window, impl).
    // SWA-crossing cases are the ones that would have caught the
    // attention_prefill_flash_tile_q8_kv NaN-init regression: any case
    // where the first row of the first active chunk is masked
    // (`row < t_start`) triggers the `(-inf) - (-inf) = NaN` path
    // unless the kernel skips the online-softmax update.
    let cases: &[Case] = &[
        // No SWA baseline — flash-tile path on every head_dim.
        Case {
            head_dim: 64,
            n_heads_q: 32,
            n_heads_kv: 8,
            n_q_tokens: 8,
            n_k_tokens: 8,
            q_offset: 0,
            window_size: 0,
            impl_kind: Impl::FlashTileD64Br4,
        },
        Case {
            head_dim: 128,
            n_heads_q: 32,
            n_heads_kv: 4,
            n_q_tokens: 8,
            n_k_tokens: 8,
            q_offset: 0,
            window_size: 0,
            impl_kind: Impl::FlashTileD128Br4,
        },
        Case {
            head_dim: 256,
            n_heads_q: 16,
            n_heads_kv: 2,
            n_q_tokens: 8,
            n_k_tokens: 8,
            q_offset: 0,
            window_size: 0,
            impl_kind: Impl::FlashTileD256Br8,
        },
        // SWA within window — boundary not crossed, mask should be no-op.
        Case {
            head_dim: 128,
            n_heads_q: 32,
            n_heads_kv: 4,
            n_q_tokens: 32,
            n_k_tokens: 32,
            q_offset: 0,
            window_size: 64,
            impl_kind: Impl::FlashTileD128Br4,
        },
        // SWA crossing — q_offset > window in the second prefill batch.
        Case {
            head_dim: 64,
            n_heads_q: 32,
            n_heads_kv: 8,
            n_q_tokens: 32,
            n_k_tokens: 96,
            q_offset: 64,
            window_size: 32,
            impl_kind: Impl::FlashTileD64Br4,
        },
        Case {
            head_dim: 128,
            n_heads_q: 32,
            n_heads_kv: 4,
            n_q_tokens: 32,
            n_k_tokens: 96,
            q_offset: 64,
            window_size: 32,
            impl_kind: Impl::FlashTileD128Br4,
        },
        // The exact regression shape: gemma SWA layers (head_dim=256,
        // window=1024) at prompt > 1024. Test the analogous shape at
        // a smaller window so the cert finishes quickly.
        Case {
            head_dim: 256,
            n_heads_q: 16,
            n_heads_kv: 2,
            n_q_tokens: 32,
            n_k_tokens: 96,
            q_offset: 64,
            window_size: 32,
            impl_kind: Impl::FlashTileD256Br8,
        },
        // Q at boundary edge — first q_token has qpos == window_size,
        // so t_start = qpos - window_size + 1 = 1 and row=0 is masked.
        // This is the exact lane the NaN-init bug hit.
        Case {
            head_dim: 256,
            n_heads_q: 16,
            n_heads_kv: 2,
            n_q_tokens: 8,
            n_k_tokens: 40,
            q_offset: 32,
            window_size: 32,
            impl_kind: Impl::FlashTileD256Br8,
        },
        // Oracle path: head_dim=512 (always oracle, no flash-tile template).
        Case {
            head_dim: 512,
            n_heads_q: 8,
            n_heads_kv: 1,
            n_q_tokens: 8,
            n_k_tokens: 8,
            q_offset: 0,
            window_size: 0,
            impl_kind: Impl::OracleD512,
        },
        Case {
            head_dim: 512,
            n_heads_q: 8,
            n_heads_kv: 1,
            n_q_tokens: 16,
            n_k_tokens: 80,
            q_offset: 64,
            window_size: 32,
            impl_kind: Impl::OracleD512,
        },
        // Oracle fallback: head_dim != 512 but n_q < 4 forces oracle.
        Case {
            head_dim: 128,
            n_heads_q: 32,
            n_heads_kv: 4,
            n_q_tokens: 2,
            n_k_tokens: 64,
            q_offset: 32,
            window_size: 16,
            impl_kind: Impl::OracleD128,
        },
        Case {
            head_dim: 64,
            n_heads_q: 32,
            n_heads_kv: 8,
            n_q_tokens: 1,
            n_k_tokens: 64,
            q_offset: 32,
            window_size: 16,
            impl_kind: Impl::OracleD64,
        },
        Case {
            head_dim: 256,
            n_heads_q: 16,
            n_heads_kv: 2,
            n_q_tokens: 2,
            n_k_tokens: 64,
            q_offset: 32,
            window_size: 16,
            impl_kind: Impl::OracleD256,
        },
    ];

    let mut results = Vec::new();
    for case in cases {
        let kernel = match case.impl_kind {
            Impl::OracleD64 | Impl::OracleD128 | Impl::OracleD256 | Impl::OracleD512 => &k_oracle,
            Impl::FlashTileD64Br4 => &k_tile_d64,
            Impl::FlashTileD128Br4 => &k_tile_d128,
            Impl::FlashTileD256Br8 => &k_tile_d256,
        };
        let seed = 0xDECADE
            ^ (case.head_dim as u64 * 7919)
            ^ (case.n_q_tokens as u64 * 101)
            ^ (case.n_k_tokens as u64 * 31)
            ^ (case.q_offset as u64 * 17)
            ^ (case.window_size as u64);
        let (got, reference) = run_shape(&dev, kernel, case, seed)?;
        let has_nonfinite = got.iter().any(|v| !v.is_finite());
        let max_rel = if has_nonfinite {
            f32::INFINITY
        } else {
            max_rel_err_with_floor(&got, &reference, (case.head_dim as f32).sqrt() * 0.01)
        };
        let tol = 5e-2;
        results.push(ShapeResult {
            m: case.n_q_tokens,
            k: case.n_k_tokens,
            n: case.head_dim,
            seed,
            max_rel_err: max_rel,
            tolerance: tol,
            pass: max_rel <= tol,
        });
        tracing::info!(
            target: "flambeau_bench::sweep_attention_prefill_q8_kv",
            head_dim = case.head_dim,
            n_heads_q = case.n_heads_q,
            n_heads_kv = case.n_heads_kv,
            n_q = case.n_q_tokens,
            n_k = case.n_k_tokens,
            q_off = case.q_offset,
            window = case.window_size,
            impl_label = case.impl_kind.label(),
            max_rel,
            tol,
            "q8 kv prefill shape"
        );
    }

    let pass = results.iter().all(|r| r.pass);
    let rig = rig();
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "attention_prefill_q8_kv_gfx906".to_string(),
        backend: "hip".to_string(),
        arch: "gfx906".to_string(),
        op: "attention_prefill_q8_kv".to_string(),
        dtype_weight: "Q8_0".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula:
            "|err| <= 5e-2 * max(|ref|, sqrt(head_dim))  (covers flash-tile + oracle paths, \
             window_size ∈ {0, >0} with SWA boundary crossings)"
                .to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig,
        pmc: Some(PmcSnapshot {
            vgpr_count: Some(attrs_oracle.num_regs),
            sgpr_count: None,
            waves_per_simd: Some(attrs_oracle.gfx906_waves_per_simd()),
            mem_busy_pct: None,
            valu_busy_pct: None,
        }),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn quantize_row_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK, 0);
    let nb = xs.len() / QK;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK..(i + 1) * QK];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK];
        for (j, &v) in block.iter().enumerate() {
            qs[j] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
        out.push(BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        });
    }
    out
}

fn dequantize_row_q8_0(xs: &[BlockQ8_0]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len() * QK];
    for (i, b) in xs.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..QK {
            out[i * QK + j] = (b.qs[j] as f32) * d;
        }
    }
    out
}

fn run_shape(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    case: &Case,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let head_dim = case.head_dim;
    let n_heads_q = case.n_heads_q;
    let n_heads_kv = case.n_heads_kv;
    let n_q_tokens = case.n_q_tokens;
    let n_k_tokens = case.n_k_tokens;
    let q_offset = case.q_offset;
    let window_size = case.window_size;

    let q_len = n_q_tokens * n_heads_q * head_dim;
    let kv_len = n_k_tokens * n_heads_kv * head_dim;

    let q_f32 = seeded_f32_range(seed, q_len, -0.5, 0.5);
    let k_f32 = seeded_f32_range(seed.wrapping_add(0xA1), kv_len, -0.5, 0.5);
    let v_f32 = seeded_f32_range(seed.wrapping_add(0xA2), kv_len, -0.5, 0.5);

    // K and V quantized per row (head_dim elements) — same layout
    // `[n_k_tokens, n_heads_kv, head_dim/32]` as KvCache<Q8Contig>.
    let nb_per_row = head_dim / QK;
    let n_rows = n_k_tokens * n_heads_kv;
    let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
    let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
    for row in 0..n_rows {
        let k_row = &k_f32[row * head_dim..(row + 1) * head_dim];
        let v_row = &v_f32[row * head_dim..(row + 1) * head_dim];
        k_blocks.extend(quantize_row_q8_0(k_row));
        v_blocks.extend(quantize_row_q8_0(v_row));
    }

    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = alloc_and_upload(dev, &q_f16);
    let d_k = alloc_and_upload(dev, &k_blocks);
    let d_v = alloc_and_upload(dev, &v_blocks);
    let out_bytes = q_len * 2;
    let d_out = dev.alloc(out_bytes)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    {
        let stream = dev.default_stream();
        let n_q_i = n_q_tokens as i32;
        let n_heads_q_i = n_heads_q as i32;
        let n_heads_kv_i = n_heads_kv as i32;
        let n_k_i = n_k_tokens as i32;
        let q_off_i = q_offset as i32;
        let scale_f = scale;
        let window_i = window_size;
        let d_q_ptr: u64 = d_q.as_usize() as u64;
        let d_k_ptr: u64 = d_k.as_usize() as u64;
        let d_v_ptr: u64 = d_v.as_usize() as u64;
        let d_out_ptr: u64 = d_out.as_usize() as u64;

        let mut args = KernelArgs::new();
        args.push(&d_q_ptr);
        args.push(&d_k_ptr);
        args.push(&d_v_ptr);
        args.push(&d_out_ptr);
        args.push(&n_q_i);
        args.push(&n_heads_q_i);
        args.push(&n_heads_kv_i);
        // Oracle has an explicit head_dim arg; flash-tile templates encode it.
        let head_dim_i = head_dim as i32;
        if !case.impl_kind.is_flash_tile() {
            args.push(&head_dim_i);
        }
        args.push(&n_k_i);
        args.push(&q_off_i);
        args.push(&scale_f);
        args.push(&window_i);

        let cfg = if case.impl_kind.is_flash_tile() {
            let br = case.impl_kind.br();
            LaunchCfg {
                grid: ((n_q_tokens as u32).div_ceil(br), n_heads_q as u32, 1),
                block: (64, br, 1),
                shared_bytes: 0,
            }
        } else {
            LaunchCfg {
                grid: (n_q_tokens as u32, n_heads_q as u32, 1),
                block: ((head_dim / 4) as u32, 1, 1),
                shared_bytes: 0,
            }
        };
        unsafe { kernel.launch(stream, cfg, args)? };
        stream.synchronize()?;
    }

    let mut out_f16: Vec<f16> = vec![f16::from_f32(0.0); q_len];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_f16.as_mut_ptr() as usize),
            d_out,
            out_bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2)?;
        dev.dealloc(d_k, k_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_v, v_blocks.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_out, out_bytes)?;
    }
    let got: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();

    // Reference: dequantize K/V back to F32 and run the F32 attention
    // math with the same SWA mask as the kernel. Any delta we observe
    // is kernel arithmetic, not quant noise.
    let mut k_rt = vec![0.0f32; kv_len];
    let mut v_rt = vec![0.0f32; kv_len];
    for row in 0..n_rows {
        let blocks_k = &k_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        let blocks_v = &v_blocks[row * nb_per_row..(row + 1) * nb_per_row];
        k_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(blocks_k));
        v_rt[row * head_dim..(row + 1) * head_dim].copy_from_slice(&dequantize_row_q8_0(blocks_v));
    }
    let q_in: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let group = n_heads_q / n_heads_kv;
    let mut reference = vec![0.0f32; q_len];
    for qt in 0..n_q_tokens {
        let qpos = q_offset + qt;
        let limit = usize::min(qpos + 1, n_k_tokens);
        let t_start = if window_size > 0 {
            let lower = (qpos as i32) - window_size + 1;
            lower.max(0) as usize
        } else {
            0
        };
        if t_start >= limit {
            for qh in 0..n_heads_q {
                for d in 0..head_dim {
                    reference[(qt * n_heads_q + qh) * head_dim + d] = 0.0;
                }
            }
            continue;
        }
        for qh in 0..n_heads_q {
            let kvh = qh / group;
            let mut scores = vec![0.0f32; limit - t_start];
            for (i, t) in (t_start..limit).enumerate() {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    let qv = q_in[(qt * n_heads_q + qh) * head_dim + d];
                    let kv = k_rt[(t * n_heads_kv + kvh) * head_dim + d];
                    dot += (qv * kv) as f64;
                }
                scores[i] = (dot as f32) * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s as f64;
            }
            let inv = 1.0f32 / sum as f32;
            for s in scores.iter_mut() {
                *s *= inv;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for (i, t) in (t_start..limit).enumerate() {
                    let vv = v_rt[(t * n_heads_kv + kvh) * head_dim + d];
                    acc += (scores[i] * vv) as f64;
                }
                reference[(qt * n_heads_q + qh) * head_dim + d] =
                    f16::from_f32(acc as f32).to_f32();
            }
        }
    }
    Ok((got, reference))
}
