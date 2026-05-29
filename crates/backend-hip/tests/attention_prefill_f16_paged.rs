//! Parity for `flambeau_attention_prefill_f16_paged` vs the
//! contiguous baseline `flambeau_attention_prefill_f16`. With the
//! block table identity-mapped (`block_table[p] = p`) and the K/V
//! pool filled byte-for-byte with the same data as the contiguous
//! slab, the paged kernel reads from the SAME logical KV memory and
//! must produce **bit-equal** output. The kernel difference is the
//! extra `t / page_size` indirection, which yields the same address
//! arithmetic under the identity map.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture; same shape rationale as siblings"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "kernel-shape math bounded by GGUF dims"
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

fn maybe_skip() -> bool {
    matches!(device_count(), Ok(n) if n >= 1)
}

fn seeded_f16(seed: u64, n: usize) -> Vec<f16> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            f16::from_f32((u as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
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

fn copy_back_f16(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    page_size: usize,
    seed: u64,
) -> (Vec<f16>, Vec<f16>) {
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let baseline_module =
        HipModule::load(0, kernels::hsaco("attention_prefill_f16").unwrap()).unwrap();
    let paged_module =
        HipModule::load(0, kernels::hsaco("attention_prefill_f16_paged").unwrap()).unwrap();
    let baseline_kernel: HipKernel<'_> = baseline_module
        .kernel("flambeau_attention_prefill_f16")
        .unwrap();
    let paged_kernel: HipKernel<'_> = paged_module
        .kernel("flambeau_attention_prefill_f16_paged")
        .unwrap();

    let kv_width = n_heads_kv * head_dim;
    // For the parity check the pool is sized exactly n_k_tokens rows
    // with identity block_table[p] = p. `n_pages = n_k_tokens /
    // page_size` (n_k_tokens must be a multiple of page_size).
    assert_eq!(n_k_tokens % page_size, 0, "n_k_tokens must be a multiple of page_size");
    let n_pages = n_k_tokens / page_size;
    let total_kv = n_k_tokens * kv_width;

    let q_data = seeded_f16(seed, n_q_tokens * n_heads_q * head_dim);
    let k_data = seeded_f16(seed.wrapping_add(13), total_kv);
    let v_data = seeded_f16(seed.wrapping_add(29), total_kv);

    // -------- baseline contiguous --------
    let baseline_q = alloc_and_upload(&dev, &q_data);
    let baseline_k = alloc_and_upload(&dev, &k_data);
    let baseline_v = alloc_and_upload(&dev, &v_data);
    let zero_out = vec![f16::from_f32(0.0); n_q_tokens * n_heads_q * head_dim];
    let baseline_out = alloc_and_upload(&dev, &zero_out);

    let n_q_i = n_q_tokens as i32;
    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_k_i = n_k_tokens as i32;
    let q_off_i = q_offset as i32;
    let window_size_i: i32 = 0;
    let scale: f32 = 1.0_f32 / (head_dim as f32).sqrt();
    let q_ptr: u64 = baseline_q.as_usize() as u64;
    let k_ptr: u64 = baseline_k.as_usize() as u64;
    let v_ptr: u64 = baseline_v.as_usize() as u64;
    let bo_ptr: u64 = baseline_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&k_ptr);
    args.push(&v_ptr);
    args.push(&bo_ptr);
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_k_i);
    args.push(&q_off_i);
    args.push(&scale);
    args.push(&window_size_i);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        baseline_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let baseline_out_host =
        copy_back_f16(&dev, baseline_out, n_q_tokens * n_heads_q * head_dim);

    // -------- paged path --------
    let paged_q = alloc_and_upload(&dev, &q_data);
    let paged_k = alloc_and_upload(&dev, &k_data);
    let paged_v = alloc_and_upload(&dev, &v_data);
    let paged_out = alloc_and_upload(&dev, &zero_out);
    let block_table: Vec<u32> = (0..n_pages as u32).collect();
    let bt_dev = alloc_and_upload(&dev, &block_table);

    let page_size_i = page_size as i32;
    let pq_ptr: u64 = paged_q.as_usize() as u64;
    let pk_ptr: u64 = paged_k.as_usize() as u64;
    let pv_ptr: u64 = paged_v.as_usize() as u64;
    let bt_ptr: u64 = bt_dev.as_usize() as u64;
    let po_ptr: u64 = paged_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&pq_ptr);
    args.push(&pk_ptr);
    args.push(&pv_ptr);
    args.push(&bt_ptr);
    args.push(&po_ptr);
    args.push(&n_q_i);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_k_i);
    args.push(&q_off_i);
    args.push(&page_size_i);
    args.push(&scale);
    args.push(&window_size_i);
    let cfg = LaunchCfg {
        grid: (n_q_tokens as u32, n_heads_q as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        paged_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let paged_out_host = copy_back_f16(&dev, paged_out, n_q_tokens * n_heads_q * head_dim);

    (baseline_out_host, paged_out_host)
}

#[test]
fn paged_prefill_attn_matches_contiguous_q4_qh8_kvh2_hd64_page16_nk32() {
    if !maybe_skip() {
        return;
    }
    // n_q = 4, n_k = 32 spanning 2 pages at page_size 16.
    let (b, p) = run_case(4, 8, 2, 64, 32, 0, 16, 0xCAFEBABE);
    assert_eq!(b, p, "prefill attn mismatch (Q=4, n_k=32, page=16)");
}

#[test]
fn paged_prefill_attn_matches_contiguous_q16_qh16_kvh4_hd128_page16_nk128() {
    if !maybe_skip() {
        return;
    }
    // n_q = 16 (full causal prefill), n_k = 128 spanning 8 pages.
    let (b, p) = run_case(16, 16, 4, 128, 128, 0, 16, 0x1234567890ABCDEF);
    assert_eq!(b, p, "prefill attn mismatch (Q=16, n_k=128, page=16)");
}

#[test]
fn paged_prefill_attn_matches_contiguous_q8_qh16_kvh2_hd256_page32_nk64() {
    if !maybe_skip() {
        return;
    }
    // head_dim=256 with page_size=32.
    let (b, p) = run_case(8, 16, 2, 256, 64, 0, 32, 0xABCDEF01);
    assert_eq!(b, p, "prefill attn mismatch (Q=8, n_k=64, page=32, hd=256)");
}
