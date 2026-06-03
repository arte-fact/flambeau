//! Parity for `flambeau_kv_append_f16_paged_prefill` vs the
//! contiguous-DtoD baseline (what `kv_append_f16` does — two
//! stream-ordered DtoD memcpys of `[n_tokens, kv_width]` F16). With
//! an identity-mapped block table the paged write addresses align
//! byte-for-byte with the contiguous slab layout, so the K + V
//! pools must end up **bit-equal** to the contiguous caches.

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
    kv_width: usize,
    page_size: usize,
    max_pages_per_slot: usize,
    n_tokens: usize,
    start_pos: usize,
    seed: u64,
) -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>) {
    assert!(start_pos + n_tokens <= page_size * max_pages_per_slot);
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let paged_module =
        HipModule::load(0, kernels::hsaco("kv_append_f16_paged_prefill").unwrap()).unwrap();
    let paged_kernel: HipKernel<'_> = paged_module
        .kernel("flambeau_kv_append_f16_paged_prefill")
        .unwrap();

    let max_seq_len = page_size * max_pages_per_slot;
    let n_pages = max_pages_per_slot;
    let total_pool = n_pages * page_size * kv_width;

    let k_src = seeded_f16(seed, n_tokens * kv_width);
    let v_src = seeded_f16(seed.wrapping_add(7), n_tokens * kv_width);
    let pool_init_k = seeded_f16(seed.wrapping_add(101), total_pool);
    let pool_init_v = seeded_f16(seed.wrapping_add(211), total_pool);

    // -------- contiguous baseline --------
    // The reference is two stream-ordered DtoD memcpys (what
    // kv_append_f16 does). We mirror that with one device.memcpy_async
    // per K / V into a `[max_seq_len, kv_width]` slab.
    let baseline_k = alloc_and_upload(&dev, &pool_init_k);
    let baseline_v = alloc_and_upload(&dev, &pool_init_v);
    let k_src_dev = alloc_and_upload(&dev, &k_src);
    let v_src_dev = alloc_and_upload(&dev, &v_src);
    let row_bytes = kv_width * 2;
    let copy_bytes = n_tokens * row_bytes;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToDevice,
            baseline_k.offset_bytes(start_pos * row_bytes),
            k_src_dev,
            copy_bytes,
        )
        .unwrap();
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToDevice,
            baseline_v.offset_bytes(start_pos * row_bytes),
            v_src_dev,
            copy_bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let baseline_k_out = copy_back_f16(&dev, baseline_k, total_pool);
    let baseline_v_out = copy_back_f16(&dev, baseline_v, total_pool);

    // -------- paged path --------
    let paged_k_pool = alloc_and_upload(&dev, &pool_init_k);
    let paged_v_pool = alloc_and_upload(&dev, &pool_init_v);
    // Identity-mapped block table for a single slot: page p of the
    // slot is global page p. With one slot and `n_pages =
    // max_pages_per_slot`, this matches the contiguous slab.
    let block_table: Vec<u32> = (0..max_pages_per_slot as u32).collect();
    let bt_dev = alloc_and_upload(&dev, &block_table);

    let n_tokens_i = n_tokens as i32;
    let kv_width_i = kv_width as i32;
    let start_pos_i = start_pos as i32;
    let page_size_i = page_size as i32;
    let k_src_ptr: u64 = k_src_dev.as_usize() as u64;
    let v_src_ptr: u64 = v_src_dev.as_usize() as u64;
    let k_pool_ptr: u64 = paged_k_pool.as_usize() as u64;
    let v_pool_ptr: u64 = paged_v_pool.as_usize() as u64;
    let bt_ptr: u64 = bt_dev.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&k_src_ptr);
    args.push(&v_src_ptr);
    args.push(&k_pool_ptr);
    args.push(&v_pool_ptr);
    args.push(&bt_ptr);
    args.push(&n_tokens_i);
    args.push(&kv_width_i);
    args.push(&start_pos_i);
    args.push(&page_size_i);
    let block_threads: u32 = (kv_width as u32).clamp(1, 128);
    let cfg = LaunchCfg {
        grid: (n_tokens as u32, 1, 1),
        block: (block_threads, 1, 1),
        shared_bytes: 0,
    };
    unsafe {
        paged_kernel
            .launch(dev.default_stream(), cfg, args)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    let paged_k_out = copy_back_f16(&dev, paged_k_pool, total_pool);
    let paged_v_out = copy_back_f16(&dev, paged_v_pool, total_pool);

    let _ = max_seq_len;
    (baseline_k_out, baseline_v_out, paged_k_out, paged_v_out)
}

#[test]
fn paged_prefill_matches_contiguous_kv64_page16_l4_start0() {
    if !maybe_skip() {
        return;
    }
    let (b_k, b_v, p_k, p_v) = run_case(64, 16, 4, 4, 0, 0xCAFEBABE);
    assert_eq!(b_k, p_k, "K mismatch (kv=64 page=16 L=4 start=0)");
    assert_eq!(b_v, p_v, "V mismatch (kv=64 page=16 L=4 start=0)");
}

#[test]
fn paged_prefill_matches_contiguous_kv128_page16_l31_crosses_boundary() {
    if !maybe_skip() {
        return;
    }
    // L=31 starting at pos 1 → writes positions [1..32). Spans two
    // pages (page 0 takes positions 1..16, page 1 takes 16..32),
    // exercising the per-token block-table walk.
    let (b_k, b_v, p_k, p_v) = run_case(128, 16, 4, 31, 1, 0xDEADBEEF);
    assert_eq!(b_k, p_k, "K mismatch (kv=128 page=16 L=31 start=1)");
    assert_eq!(b_v, p_v, "V mismatch (kv=128 page=16 L=31 start=1)");
}

#[test]
fn paged_prefill_matches_contiguous_kv256_page32_l64_crosses_two_boundaries() {
    if !maybe_skip() {
        return;
    }
    // start=16, L=64 → positions [16, 80). Spans 3 pages (page 0
    // takes 16..32, page 1 takes 32..64, page 2 takes 64..80) at
    // page_size=32.
    let (b_k, b_v, p_k, p_v) = run_case(256, 32, 4, 64, 16, 0xABCDEF01);
    assert_eq!(b_k, p_k, "K mismatch (kv=256 page=32 L=64 start=16)");
    assert_eq!(b_v, p_v, "V mismatch (kv=256 page=32 L=64 start=16)");
}
