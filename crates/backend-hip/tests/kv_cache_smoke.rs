//! KvCache<F16Contig, HipDevice> smoke test — alloc, append, read-back.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` below is a kernel launch or `memcpy_async` \
              whose invariant is uniform: host/device buffers live for the bounded \
              `synchronize()` that follows, pointers are freshly allocated above, kernel \
              ABIs match kernels-hip. Per-site SAFETY comments would just repeat this."
)]

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_runtime::{F16Contig, KvCache};
use half::f16;

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

#[test]
fn f16_contig_new_append_readback() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let n_heads = 4;
    let head_dim = 128;
    let max_tokens = 64;

    let mut cache: KvCache<F16Contig, HipDevice> =
        KvCache::new(&dev, n_heads, head_dim, max_tokens).unwrap();

    assert_eq!(cache.current_tokens(), 0);
    assert_eq!(cache.max_tokens(), max_tokens);
    assert_eq!(cache.layout_name(), "f16_contig");
    assert_eq!(cache.bytes_per_tensor(), max_tokens * n_heads * head_dim * 2);

    // Construct 8 tokens of K and V on host: values carry the token index
    // so readback verifies positional ordering after append.
    let n_new = 8;
    let elems = n_new * n_heads * head_dim;
    let k_host: Vec<f16> = (0..elems)
        .map(|i| f16::from_f32(i as f32 * 0.001))
        .collect();
    let v_host: Vec<f16> = (0..elems)
        .map(|i| f16::from_f32((i as f32 + 10_000.0) * 0.001))
        .collect();

    // Upload to device (caller's contract — matches typical call-site where
    // K/V come out of a projection matmul).
    let d_k_src = dev.alloc(elems * 2).unwrap();
    let d_v_src = dev.alloc(elems * 2).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_k_src,
            DevicePtr(k_host.as_ptr() as usize),
            elems * 2,
        )
        .unwrap();
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_v_src,
            DevicePtr(v_host.as_ptr() as usize),
            elems * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    // Append.
    unsafe {
        cache
            .append(&dev, dev.default_stream(), d_k_src, d_v_src, n_new)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    assert_eq!(cache.current_tokens(), n_new);

    // Read back the appended range from cache.k_buffer() / v_buffer().
    let mut k_back = vec![f16::from_f32(0.0); elems];
    let mut v_back = vec![f16::from_f32(0.0); elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(k_back.as_mut_ptr() as usize),
            cache.k_buffer(),
            elems * 2,
        )
        .unwrap();
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(v_back.as_mut_ptr() as usize),
            cache.v_buffer(),
            elems * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    for i in 0..elems {
        assert_eq!(k_back[i].to_bits(), k_host[i].to_bits(), "K[{i}] mismatch");
        assert_eq!(v_back[i].to_bits(), v_host[i].to_bits(), "V[{i}] mismatch");
    }

    // Append a second batch — should land at offset 8, not 0.
    let second = vec![f16::from_f32(-1.0); elems];
    let d_k2 = dev.alloc(elems * 2).unwrap();
    let d_v2 = dev.alloc(elems * 2).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_k2,
            DevicePtr(second.as_ptr() as usize),
            elems * 2,
        )
        .unwrap();
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_v2,
            DevicePtr(second.as_ptr() as usize),
            elems * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    unsafe {
        cache
            .append(&dev, dev.default_stream(), d_k2, d_v2, n_new)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    assert_eq!(cache.current_tokens(), 2 * n_new);

    // First 8 tokens still carry the original payload; next 8 carry -1.
    let total_elems = 2 * n_new * n_heads * head_dim;
    let mut all_k = vec![f16::from_f32(0.0); total_elems];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(all_k.as_mut_ptr() as usize),
            cache.k_buffer(),
            total_elems * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    assert_eq!(all_k[0].to_bits(), k_host[0].to_bits());
    assert_eq!(all_k[elems].to_bits(), f16::from_f32(-1.0).to_bits());

    unsafe {
        dev.dealloc(d_k_src, elems * 2).unwrap();
        dev.dealloc(d_v_src, elems * 2).unwrap();
        dev.dealloc(d_k2, elems * 2).unwrap();
        dev.dealloc(d_v2, elems * 2).unwrap();
    }
    cache.dispose(&dev).unwrap();
}

#[test]
fn capacity_exceeded_is_reported() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();
    let mut cache: KvCache<F16Contig, HipDevice> = KvCache::new(&dev, 1, 4, 4).unwrap();
    let dummy = dev.alloc(32).unwrap();
    // 8 tokens into a 4-token cap.
    let err = unsafe {
        cache.append(&dev, dev.default_stream(), dummy, dummy, 8)
    }
    .unwrap_err();
    assert!(format!("{err}").contains("capacity"));
    unsafe { dev.dealloc(dummy, 32).unwrap() };
    cache.dispose(&dev).unwrap();
}
