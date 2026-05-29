//! Parity for `flambeau_kv_append_f16_paged_slots` vs
//! `flambeau_kv_append_f16_batched_slots`. With the block-table
//! identity-mapped (slot s holds pages [s * max_pages, s * max_pages +
//! max_pages)), the paged kernel writes to the same logical KV rows
//! as the contiguous batched-slots kernel — and must produce
//! **bit-equal** pool contents.

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

fn run_case(
    n_slots: usize,
    kv_width: usize,
    page_size: usize,
    max_pages_per_slot: usize,
    write_pos: &[usize],
    seed: u64,
) -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>) {
    assert_eq!(write_pos.len(), n_slots);
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let paged_module =
        HipModule::load(0, kernels::hsaco("kv_append_f16_paged_slots").unwrap()).unwrap();
    let batched_module =
        HipModule::load(0, kernels::hsaco("kv_append_f16_batched_slots").unwrap()).unwrap();
    let paged_kernel: HipKernel<'_> = paged_module
        .kernel("flambeau_kv_append_f16_paged_slots")
        .unwrap();
    let batched_kernel: HipKernel<'_> = batched_module
        .kernel("flambeau_kv_append_f16_batched_slots")
        .unwrap();

    let max_seq_len = page_size * max_pages_per_slot;
    let n_pages = n_slots * max_pages_per_slot;

    // K/V src: [N, kv_width]
    let k_src = seeded_f16(seed, n_slots * kv_width);
    let v_src = seeded_f16(seed.wrapping_add(7), n_slots * kv_width);

    // Initialise BOTH paths' KV with the same deterministic garbage so
    // unchanged rows can be compared bit-for-bit.
    let total_kv = n_pages * page_size * kv_width;
    let cache_init_k = seeded_f16(seed.wrapping_add(101), total_kv);
    let cache_init_v = seeded_f16(seed.wrapping_add(211), total_kv);

    // -------- batched (contiguous) path --------
    // Treat the cache as `[n_slots, max_seq_len, kv_width]` contiguous.
    let batched_k_cache = alloc_and_upload(&dev, &cache_init_k);
    let batched_v_cache = alloc_and_upload(&dev, &cache_init_v);
    let batched_k_dst_ptrs: Vec<u64> = (0..n_slots)
        .map(|s| {
            batched_k_cache
                .offset_bytes(s * max_seq_len * kv_width * 2)
                .as_usize() as u64
        })
        .collect();
    let batched_v_dst_ptrs: Vec<u64> = (0..n_slots)
        .map(|s| {
            batched_v_cache
                .offset_bytes(s * max_seq_len * kv_width * 2)
                .as_usize() as u64
        })
        .collect();
    let batched_k_dst_arr = alloc_and_upload(&dev, &batched_k_dst_ptrs);
    let batched_v_dst_arr = alloc_and_upload(&dev, &batched_v_dst_ptrs);

    let host_write_pos: Vec<i32> = write_pos.iter().map(|&w| w as i32).collect();
    let batched_wpos = alloc_and_upload(&dev, &host_write_pos);
    let k_src_dev = alloc_and_upload(&dev, &k_src);
    let v_src_dev = alloc_and_upload(&dev, &v_src);

    let n_slots_i = n_slots as i32;
    let kv_width_i = kv_width as i32;
    let k_src_ptr: u64 = k_src_dev.as_usize() as u64;
    let v_src_ptr: u64 = v_src_dev.as_usize() as u64;
    let bk_dst_arr_ptr: u64 = batched_k_dst_arr.as_usize() as u64;
    let bv_dst_arr_ptr: u64 = batched_v_dst_arr.as_usize() as u64;
    let bwpos_ptr: u64 = batched_wpos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&bk_dst_arr_ptr);
    args.push(&bv_dst_arr_ptr);
    args.push(&bwpos_ptr);
    args.push(&n_slots_i);
    args.push(&kv_width_i);
    let block_threads: u32 = (kv_width as u32).min(128).max(1);
    let cfg = LaunchCfg {
        grid: (n_slots as u32, 1, 1),
        block: (block_threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        batched_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    let batched_k_out = copy_back_f16(&dev, batched_k_cache, total_kv);
    let batched_v_out = copy_back_f16(&dev, batched_v_cache, total_kv);

    // -------- paged path --------
    // Identity-mapped block table: slot s holds pages [s * mpps, s *
    // mpps + mpps). Then `(page * page_size + offset) * kv_width` is
    // the same byte offset as the contiguous path's
    // `slot * max_seq_len * kv_width + write_pos * kv_width`.
    let paged_k_pool = alloc_and_upload(&dev, &cache_init_k);
    let paged_v_pool = alloc_and_upload(&dev, &cache_init_v);
    let mut block_tables: Vec<u32> = vec![0; n_slots * max_pages_per_slot];
    for s in 0..n_slots {
        for p in 0..max_pages_per_slot {
            block_tables[s * max_pages_per_slot + p] = (s * max_pages_per_slot + p) as u32;
        }
    }
    let bt_dev = alloc_and_upload(&dev, &block_tables);

    let page_size_i = page_size as i32;
    let max_pps_i = max_pages_per_slot as i32;
    let paged_k_pool_ptr: u64 = paged_k_pool.as_usize() as u64;
    let paged_v_pool_ptr: u64 = paged_v_pool.as_usize() as u64;
    let bt_ptr: u64 = bt_dev.as_usize() as u64;
    let pwpos_ptr: u64 = batched_wpos.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&paged_k_pool_ptr);
    args.push(&paged_v_pool_ptr);
    args.push(&bt_ptr);
    args.push(&pwpos_ptr);
    args.push(&n_slots_i);
    args.push(&kv_width_i);
    args.push(&page_size_i);
    args.push(&max_pps_i);
    let cfg = LaunchCfg {
        grid: (n_slots as u32, 1, 1),
        block: (block_threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        paged_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    let paged_k_out = copy_back_f16(&dev, paged_k_pool, total_kv);
    let paged_v_out = copy_back_f16(&dev, paged_v_pool, total_kv);

    (batched_k_out, batched_v_out, paged_k_out, paged_v_out)
}

#[test]
fn kv_append_paged_matches_batched_n2_kv64() {
    if !maybe_skip() {
        return;
    }
    // 2 slots × 4 pages × 16 tokens = 128 token rows total. write_pos
    // [3, 17] hits slot 0 page 0 and slot 1 page 1, exercising both
    // page_offset within the same page AND a page-boundary crossing.
    let (b_k, b_v, p_k, p_v) = run_case(2, 64, 16, 4, &[3, 17], 0xCAFEBABE);
    assert_eq!(b_k, p_k, "K cache mismatch (paged vs batched at write_pos[3, 17])");
    assert_eq!(b_v, p_v, "V cache mismatch (paged vs batched at write_pos[3, 17])");
}

#[test]
fn kv_append_paged_matches_batched_n4_kv128_page16() {
    if !maybe_skip() {
        return;
    }
    let (b_k, b_v, p_k, p_v) = run_case(4, 128, 16, 8, &[0, 31, 64, 127], 0x1234567890ABCDEF);
    assert_eq!(b_k, p_k, "K cache mismatch (N=4 kv_width=128 page_size=16)");
    assert_eq!(b_v, p_v, "V cache mismatch (N=4 kv_width=128 page_size=16)");
}

#[test]
fn kv_append_paged_matches_batched_n2_kv256_page32() {
    if !maybe_skip() {
        return;
    }
    let (b_k, b_v, p_k, p_v) = run_case(2, 256, 32, 4, &[15, 95], 0xABCDEF01);
    assert_eq!(b_k, p_k, "K cache mismatch (N=2 kv_width=256 page_size=32)");
    assert_eq!(b_v, p_v, "V cache mismatch (N=2 kv_width=256 page_size=32)");
}
