//! Parity for `flambeau_attention_decode_f16_paged` vs
//! `flambeau_attention_decode_f16_batched`. Identity-mapped block
//! tables (slot s holds pages [s * max_pages_per_slot, ..]) make the
//! paged kernel read from the SAME logical KV memory as the
//! contiguous batched kernel. With the same per-slot K/V data the
//! outputs must be **bit-equal** — the only kernel difference is the
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
    n_slots: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    page_size: usize,
    max_pages_per_slot: usize,
    n_tokens_kv: &[usize],
    seed: u64,
) -> (Vec<f16>, Vec<f16>) {
    assert_eq!(n_tokens_kv.len(), n_slots);
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let batched_module =
        HipModule::load(0, kernels::hsaco("attention_decode_f16_batched").unwrap()).unwrap();
    let paged_module =
        HipModule::load(0, kernels::hsaco("attention_decode_f16_paged").unwrap()).unwrap();
    let batched_kernel: HipKernel<'_> = batched_module
        .kernel("flambeau_attention_decode_f16_batched")
        .unwrap();
    let paged_kernel: HipKernel<'_> = paged_module
        .kernel("flambeau_attention_decode_f16_paged")
        .unwrap();

    let kv_width = n_heads_kv * head_dim;
    let max_seq_len = page_size * max_pages_per_slot;

    // Same K/V data for both layouts. `[n_slots, max_seq_len, kv_width]`
    // (= contiguous per-slot) equals `[n_pages = n_slots *
    // max_pages_per_slot, page_size, kv_width]` byte-for-byte when the
    // block table is identity-mapped (slot s → pages [s*mpps,
    // s*mpps+mpps)).
    let total_kv = n_slots * max_seq_len * kv_width;
    let k_data = seeded_f16(seed, total_kv);
    let v_data = seeded_f16(seed.wrapping_add(13), total_kv);
    // Q is shared across paths.
    let q_data = seeded_f16(seed.wrapping_add(29), n_slots * n_heads_q * head_dim);

    // -------- batched (contiguous) path --------
    let batched_k = alloc_and_upload(&dev, &k_data);
    let batched_v = alloc_and_upload(&dev, &v_data);
    let batched_q = alloc_and_upload(&dev, &q_data);
    let zero_out = vec![f16::from_f32(0.0); n_slots * n_heads_q * head_dim];
    let batched_out = alloc_and_upload(&dev, &zero_out);

    let batched_k_ptrs: Vec<u64> = (0..n_slots)
        .map(|s| {
            batched_k
                .offset_bytes(s * max_seq_len * kv_width * 2)
                .as_usize() as u64
        })
        .collect();
    let batched_v_ptrs: Vec<u64> = (0..n_slots)
        .map(|s| {
            batched_v
                .offset_bytes(s * max_seq_len * kv_width * 2)
                .as_usize() as u64
        })
        .collect();
    let batched_k_arr = alloc_and_upload(&dev, &batched_k_ptrs);
    let batched_v_arr = alloc_and_upload(&dev, &batched_v_ptrs);
    let n_tokens_i32: Vec<i32> = n_tokens_kv.iter().map(|&n| n as i32).collect();
    let n_tokens_dev = alloc_and_upload(&dev, &n_tokens_i32);

    let n_heads_q_i = n_heads_q as i32;
    let n_heads_kv_i = n_heads_kv as i32;
    let head_dim_i = head_dim as i32;
    let n_slots_i = n_slots as i32;
    let scale: f32 = 1.0_f32 / (head_dim as f32).sqrt();
    let q_ptr: u64 = batched_q.as_usize() as u64;
    let bk_arr_ptr: u64 = batched_k_arr.as_usize() as u64;
    let bv_arr_ptr: u64 = batched_v_arr.as_usize() as u64;
    let bo_ptr: u64 = batched_out.as_usize() as u64;
    let nkv_ptr: u64 = n_tokens_dev.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&q_ptr);
    args.push(&bk_arr_ptr);
    args.push(&bv_arr_ptr);
    args.push(&bo_ptr);
    args.push(&nkv_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_slots_i);
    args.push(&scale);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        batched_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let batched_out_host = copy_back_f16(&dev, batched_out, n_slots * n_heads_q * head_dim);

    // -------- paged path --------
    // Layout: `[n_pages, page_size, kv_width]`. Use the same flat
    // bytes as the contiguous path → same K/V data → identity map.
    let paged_k = alloc_and_upload(&dev, &k_data);
    let paged_v = alloc_and_upload(&dev, &v_data);
    let paged_q = alloc_and_upload(&dev, &q_data);
    let paged_out = alloc_and_upload(&dev, &zero_out);
    let mut block_tables: Vec<u32> = vec![0; n_slots * max_pages_per_slot];
    for s in 0..n_slots {
        for p in 0..max_pages_per_slot {
            block_tables[s * max_pages_per_slot + p] = (s * max_pages_per_slot + p) as u32;
        }
    }
    let bt_dev = alloc_and_upload(&dev, &block_tables);

    let page_size_i = page_size as i32;
    let max_pps_i = max_pages_per_slot as i32;
    let pq_ptr: u64 = paged_q.as_usize() as u64;
    let pk_pool_ptr: u64 = paged_k.as_usize() as u64;
    let pv_pool_ptr: u64 = paged_v.as_usize() as u64;
    let bt_ptr: u64 = bt_dev.as_usize() as u64;
    let po_ptr: u64 = paged_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&pq_ptr);
    args.push(&pk_pool_ptr);
    args.push(&pv_pool_ptr);
    args.push(&bt_ptr);
    args.push(&po_ptr);
    args.push(&nkv_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_slots_i);
    args.push(&page_size_i);
    args.push(&max_pps_i);
    args.push(&scale);
    let cfg = LaunchCfg {
        grid: (n_heads_q as u32, n_slots as u32, 1),
        block: (head_dim as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        paged_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let paged_out_host = copy_back_f16(&dev, paged_out, n_slots * n_heads_q * head_dim);

    (batched_out_host, paged_out_host)
}

#[test]
fn attention_paged_matches_batched_n2_qh8_kvh2_hd64_page16() {
    if !maybe_skip() {
        return;
    }
    let (b, p) = run_case(2, 8, 2, 64, 16, 4, &[40, 60], 0xCAFEBABE);
    assert_eq!(b, p, "attention output mismatch (N=2, head_dim=64, page=16)");
}

#[test]
fn attention_paged_matches_batched_n4_qh16_kvh4_hd128_page16() {
    if !maybe_skip() {
        return;
    }
    let (b, p) = run_case(4, 16, 4, 128, 16, 8, &[20, 80, 100, 127], 0x1234567890ABCDEF);
    assert_eq!(b, p, "attention output mismatch (N=4, head_dim=128, page=16)");
}

#[test]
fn attention_paged_matches_batched_n2_qh16_kvh2_hd256_page32() {
    if !maybe_skip() {
        return;
    }
    let (b, p) = run_case(2, 16, 2, 256, 32, 4, &[31, 127], 0xABCDEF01);
    assert_eq!(b, p, "attention output mismatch (N=2, head_dim=256, page=32)");
}
