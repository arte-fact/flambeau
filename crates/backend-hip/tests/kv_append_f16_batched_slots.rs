//! Parity for `flambeau_kv_append_f16_batched_slots` vs the per-slot
//! `hipMemcpyAsync(DtoD)` baseline. Both must produce **bit-equal** KV
//! cache contents — the kernel just executes the same copies in one
//! launch.

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
    max_seq_len: usize,
    seed: u64,
) -> (Vec<Vec<f16>>, Vec<Vec<f16>>, Vec<Vec<f16>>, Vec<Vec<f16>>) {
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let kv_append_module =
        HipModule::load(0, kernels::hsaco("kv_append_f16_batched_slots").unwrap()).unwrap();
    let k_kernel: HipKernel<'_> = kv_append_module
        .kernel("flambeau_kv_append_f16_batched_slots")
        .unwrap();

    // K/V src: [N, kv_width]
    let k_src = seeded_f16(seed, n_slots * kv_width);
    let v_src = seeded_f16(seed.wrapping_add(7), n_slots * kv_width);

    // Per-slot cache buffers, each [max_seq_len, kv_width]. Pre-fill with
    // deterministic garbage so we can verify only the target row got
    // overwritten.
    let cache_init: Vec<Vec<f16>> = (0..n_slots)
        .map(|i| {
            seeded_f16(
                seed.wrapping_add(101).wrapping_add(i as u64),
                max_seq_len * kv_width,
            )
        })
        .collect();
    let cache_init_v: Vec<Vec<f16>> = (0..n_slots)
        .map(|i| {
            seeded_f16(
                seed.wrapping_add(211).wrapping_add(i as u64),
                max_seq_len * kv_width,
            )
        })
        .collect();

    // REF caches.
    let ref_k_caches: Vec<DevicePtr> = cache_init
        .iter()
        .map(|c| alloc_and_upload(&dev, c))
        .collect();
    let ref_v_caches: Vec<DevicePtr> = cache_init_v
        .iter()
        .map(|c| alloc_and_upload(&dev, c))
        .collect();
    // KER (kernel) caches with same init.
    let ker_k_caches: Vec<DevicePtr> = cache_init
        .iter()
        .map(|c| alloc_and_upload(&dev, c))
        .collect();
    let ker_v_caches: Vec<DevicePtr> = cache_init_v
        .iter()
        .map(|c| alloc_and_upload(&dev, c))
        .collect();

    let d_k_src = alloc_and_upload(&dev, &k_src);
    let d_v_src = alloc_and_upload(&dev, &v_src);

    // Write positions: spread across the cache.
    let write_pos: Vec<usize> = (0..n_slots)
        .map(|i| (i * (max_seq_len - 1) / n_slots.max(1)).min(max_seq_len - 1))
        .collect();

    // REF: per-slot DtoD memcpys (mirrors the legacy step-8 loop).
    {
        let stream = dev.default_stream();
        let row_bytes = kv_width * 2;
        for s in 0..n_slots {
            let k_dst = DevicePtr(ref_k_caches[s].as_usize() + write_pos[s] * row_bytes);
            let v_dst = DevicePtr(ref_v_caches[s].as_usize() + write_pos[s] * row_bytes);
            let k_src_off = DevicePtr(d_k_src.as_usize() + s * row_bytes);
            let v_src_off = DevicePtr(d_v_src.as_usize() + s * row_bytes);
            unsafe {
                dev.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    k_dst,
                    k_src_off,
                    row_bytes,
                )
                .unwrap();
                dev.memcpy_async(
                    stream,
                    CopyDirection::DeviceToDevice,
                    v_dst,
                    v_src_off,
                    row_bytes,
                )
                .unwrap();
            }
        }
        stream.synchronize().unwrap();
    }

    // KER: build slot pointer tables + write_pos table; one launch.
    let ker_k_ptrs_host: Vec<u64> = ker_k_caches.iter().map(|p| p.as_usize() as u64).collect();
    let ker_v_ptrs_host: Vec<u64> = ker_v_caches.iter().map(|p| p.as_usize() as u64).collect();
    let write_pos_host: Vec<i32> = write_pos.iter().map(|p| *p as i32).collect();
    let d_k_ptrs = alloc_and_upload(&dev, &ker_k_ptrs_host);
    let d_v_ptrs = alloc_and_upload(&dev, &ker_v_ptrs_host);
    let d_write_pos = alloc_and_upload(&dev, &write_pos_host);
    {
        let stream = dev.default_stream();
        let n_slots_i = n_slots as i32;
        let kv_width_i = kv_width as i32;
        let k_src_ptr: u64 = d_k_src.as_usize() as u64;
        let v_src_ptr: u64 = d_v_src.as_usize() as u64;
        let k_dst_arr: u64 = d_k_ptrs.as_usize() as u64;
        let v_dst_arr: u64 = d_v_ptrs.as_usize() as u64;
        let wpos_ptr: u64 = d_write_pos.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&k_src_ptr);
        args.push(&v_src_ptr);
        args.push(&k_dst_arr);
        args.push(&v_dst_arr);
        args.push(&wpos_ptr);
        args.push(&n_slots_i);
        args.push(&kv_width_i);
        let block_threads = (kv_width.min(128) as u32).max(1);
        let cfg = LaunchCfg {
            grid: (n_slots as u32, 1, 1),
            block: (block_threads, 1, 1),
            shared_bytes: 0,
        };
        unsafe { k_kernel.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let ref_k_out: Vec<Vec<f16>> = ref_k_caches
        .iter()
        .map(|p| copy_back_f16(&dev, *p, max_seq_len * kv_width))
        .collect();
    let ref_v_out: Vec<Vec<f16>> = ref_v_caches
        .iter()
        .map(|p| copy_back_f16(&dev, *p, max_seq_len * kv_width))
        .collect();
    let ker_k_out: Vec<Vec<f16>> = ker_k_caches
        .iter()
        .map(|p| copy_back_f16(&dev, *p, max_seq_len * kv_width))
        .collect();
    let ker_v_out: Vec<Vec<f16>> = ker_v_caches
        .iter()
        .map(|p| copy_back_f16(&dev, *p, max_seq_len * kv_width))
        .collect();

    unsafe {
        for p in &ref_k_caches {
            dev.dealloc(*p, cache_init[0].len() * 2).unwrap();
        }
        for p in &ref_v_caches {
            dev.dealloc(*p, cache_init_v[0].len() * 2).unwrap();
        }
        for p in &ker_k_caches {
            dev.dealloc(*p, cache_init[0].len() * 2).unwrap();
        }
        for p in &ker_v_caches {
            dev.dealloc(*p, cache_init_v[0].len() * 2).unwrap();
        }
        dev.dealloc(d_k_src, k_src.len() * 2).unwrap();
        dev.dealloc(d_v_src, v_src.len() * 2).unwrap();
        dev.dealloc(d_k_ptrs, ker_k_ptrs_host.len() * 8).unwrap();
        dev.dealloc(d_v_ptrs, ker_v_ptrs_host.len() * 8).unwrap();
        dev.dealloc(d_write_pos, write_pos_host.len() * 4).unwrap();
    }

    (ref_k_out, ref_v_out, ker_k_out, ker_v_out)
}

fn assert_bit_equal(label: &str, a: &[f16], b: &[f16]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: idx={i} ref={} ker={}",
            x.to_f32(),
            y.to_f32()
        );
    }
}

#[test]
fn parity_n2_kv256_seq16() {
    if !maybe_skip() {
        eprintln!("[skip] no HIP device");
        return;
    }
    let (rk, rv, kk, kv) = run_case(2, 256, 16, 0xC0FFEE);
    for s in 0..2 {
        assert_bit_equal(&format!("K slot {s}"), &rk[s], &kk[s]);
        assert_bit_equal(&format!("V slot {s}"), &rv[s], &kv[s]);
    }
}

#[test]
fn parity_n4_kv512_seq64() {
    if !maybe_skip() {
        return;
    }
    let (rk, rv, kk, kv) = run_case(4, 512, 64, 0xFEED_FACE);
    for s in 0..4 {
        assert_bit_equal(&format!("K slot {s}"), &rk[s], &kk[s]);
        assert_bit_equal(&format!("V slot {s}"), &rv[s], &kv[s]);
    }
}

#[test]
fn parity_n3_kv256_seq8_qwen36_35b_shape() {
    if !maybe_skip() {
        return;
    }
    // Qwen3.6-35B-A3B TP2 local_kv_width: head_dim=128 × n_kv_heads=2 = 256.
    let (rk, rv, kk, kv) = run_case(3, 256, 8, 0x1234_5678);
    for s in 0..3 {
        assert_bit_equal(&format!("K slot {s}"), &rk[s], &kk[s]);
        assert_bit_equal(&format!("V slot {s}"), &rv[s], &kv[s]);
    }
}

#[test]
fn parity_n8_kv256_seq32() {
    if !maybe_skip() {
        return;
    }
    let (rk, rv, kk, kv) = run_case(8, 256, 32, 0xABCDEF01);
    for s in 0..8 {
        assert_bit_equal(&format!("K slot {s}"), &rk[s], &kk[s]);
        assert_bit_equal(&format!("V slot {s}"), &rv[s], &kv[s]);
    }
}
