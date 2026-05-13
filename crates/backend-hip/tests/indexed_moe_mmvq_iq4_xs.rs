//! Structural parity for `flambeau_indexed_moe_mmvq_iq4_xs_q8_1` on MI50.
//!
//!! Kernels are byte-identical to the dense MMVQ kernels
//! in the per-element decode body — the only difference is the
//! `expert_ids[]` indirection in pointer math and the dst write offset.
//! This test exercises that indirection: given `n_experts` distinct
//! expert weight slabs, a random `[n_tokens, top_k]` expert routing, and
//! a Q8_1 activation per token, the indexed kernel's `dst[token, slot,
//! row]` must match the dense MMVQ result for the equivalent
//! `(weights = experts[expert_ids[token, slot]], activation = act[token])`
//! pair.
//!
//! Since the inner arithmetic is already validated by `mmvq_iq4_xs`, a
//! single dtype (IQ4_XS — the most-used in UD-XL builds) suffices as the
//! structural gate for the other 8 IQ indexed-MoE kernels.

#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture")]
#![expect(clippy::cast_possible_wrap, reason = "test fixture")]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockIq4Xs, BlockQ8_1, QK8_0, QK_K};
use half::f16;

const QK8: usize = QK8_0;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 24) as u8
        })
        .collect()
}
fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn random_iq4_xs_block(seed: u64, idx: usize) -> BlockIq4Xs {
    let bytes = seeded_bytes(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)), 136);
    let d = f16::from_f32((bytes[0] as f32 / 255.0) * 0.02 + 0.002);
    let scales_h = u16::from_le_bytes([bytes[2], bytes[3]]);
    let mut scales_l = [0u8; QK_K / 64];
    scales_l.copy_from_slice(&bytes[4..4 + QK_K / 64]);
    let mut qs = [0u8; QK_K / 2];
    qs.copy_from_slice(&bytes[8..8 + QK_K / 2]);
    BlockIq4Xs { d, scales_h, scales_l, qs }
}

fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len() % QK8, 0);
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

#[test]
fn indexed_moe_mmvq_iq4_xs_parity() {
    if !maybe_skip() {
        return;
    }

    // Shape: 2 experts × 8 rows × 256 K, 4 tokens × top_k=2.
    let n_experts = 2usize;
    let n_rows = 8usize;
    let k = 256usize;
    let n_tokens = 4usize;
    let top_k = 2usize;
    let n_sb_per_row = k / QK_K;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_module = HipModule::load(0, kernels::hsaco("quantize_q8_1").unwrap()).unwrap();
    let m_module =
        HipModule::load(0, kernels::hsaco("indexed_moe_mmvq_iq4_xs").unwrap()).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k_mmvq: HipKernel<'_> = m_module
        .kernel("flambeau_indexed_moe_mmvq_iq4_xs_q8_1")
        .unwrap();

    // Per-expert weight slab.
    let n_blocks_per_expert = n_rows * n_sb_per_row;
    let mut all_weights: Vec<BlockIq4Xs> = Vec::with_capacity(n_experts * n_blocks_per_expert);
    for e in 0..n_experts {
        for i in 0..n_blocks_per_expert {
            all_weights.push(random_iq4_xs_block(0xC0FFEE ^ (e as u64), i));
        }
    }

    // Dequant each expert's weights to f32 for the CPU reference.
    let elems_per_expert = n_rows * k;
    let mut expert_f32 = vec![0.0f32; n_experts * elems_per_expert];
    {
        let raw: &[u8] = bytemuck::cast_slice(&all_weights);
        let block_bytes = std::mem::size_of::<BlockIq4Xs>();
        for e in 0..n_experts {
            flambeau_quant::dequantize_into(
                flambeau_quant::GgmlDType::Iq4Xs,
                &raw[e * n_blocks_per_expert * block_bytes
                    ..(e + 1) * n_blocks_per_expert * block_bytes],
                &mut expert_f32[e * elems_per_expert..(e + 1) * elems_per_expert],
            )
            .unwrap();
        }
    }

    // Per-token activation rows (f32 → device, then quantize on GPU).
    let act_f32 = seeded_f32(0xFEEDFACE, n_tokens * k);

    // Expert routing: token t goes to experts [t % n_experts, (t+1) % n_experts].
    let mut expert_ids: Vec<i32> = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            expert_ids.push(((t + s) % n_experts) as i32);
        }
    }

    let d_x = alloc_and_upload(&dev, &all_weights);
    let d_y_f32 = alloc_and_upload(&dev, &act_f32);
    let d_eids = alloc_and_upload(&dev, &expert_ids);
    let y_blocks = n_tokens * (k / QK8);
    let d_y_q8_1 = dev.alloc(y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
    let d_dst = dev.alloc(n_tokens * top_k * n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        let n_elems = (n_tokens * k) as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_y_f32_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks as u32, QK8 as u32);
        unsafe { k_quantize.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_tokens_i = n_tokens as i32;
        let top_k_i = top_k as i32;
        let nb_i = n_sb_per_row as i32;
        let w_ptr: u64 = d_x.as_usize() as u64;
        let y_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let e_ptr: u64 = d_eids.as_usize() as u64;
        let d_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&w_ptr);
        args.push(&y_ptr);
        args.push(&e_ptr);
        args.push(&d_ptr);
        args.push(&n_rows_i);
        args.push(&n_tokens_i);
        args.push(&top_k_i);
        args.push(&nb_i);
        let cfg = LaunchCfg {
            grid: (n_rows as u32, (n_tokens * top_k) as u32, 1),
            block: (64, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_mmvq.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let mut got = vec![0.0f32; n_tokens * top_k * n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(got.as_mut_ptr() as usize),
            d_dst,
            n_tokens * top_k * n_rows * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    // CPU reference: for each (token, slot), find expert, take that
    // expert's weight slice, matmul with the Q8_1-roundtripped activation.
    let mut reference = vec![0.0f32; n_tokens * top_k * n_rows];
    for t in 0..n_tokens {
        let act_rt = quantize_q8_1_roundtrip(&act_f32[t * k..(t + 1) * k]);
        for s in 0..top_k {
            let expert = expert_ids[t * top_k + s] as usize;
            for row in 0..n_rows {
                let w_row =
                    &expert_f32[expert * elems_per_expert + row * k..expert * elems_per_expert + (row + 1) * k];
                let mut acc = 0.0f64;
                for j in 0..k {
                    acc += (w_row[j] * act_rt[j]) as f64;
                }
                reference[(t * top_k + s) * n_rows + row] = acc as f32;
            }
        }
    }

    // Use the same envelope as the cert sweeps: |err| <= tol * max(|ref|, sqrt(k)).
    // The `sqrt(k)` floor handles outputs that happen to be near zero, where a
    // bounded absolute kernel diff looks like a large relative error.
    let sqrt_k = (k as f32).sqrt();
    let err: f32 = got
        .iter()
        .zip(&reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(sqrt_k))
        .fold(0.0f32, f32::max);
    let tol = 3e-2;
    eprintln!(
        "[indexed_moe_mmvq_iq4_xs n_experts={n_experts} n_tokens={n_tokens} top_k={top_k} \
         n_rows={n_rows} k={k}] err={err:.3e}, tol={tol:.3e}"
    );

    unsafe {
        dev.dealloc(d_x, all_weights.len() * std::mem::size_of::<BlockIq4Xs>()).unwrap();
        dev.dealloc(d_y_f32, act_f32.len() * 4).unwrap();
        dev.dealloc(d_eids, expert_ids.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
        dev.dealloc(d_dst, n_tokens * top_k * n_rows * 4).unwrap();
    }

    assert!(err <= tol, "err {err:.3e} > {tol:.3e}");
}
