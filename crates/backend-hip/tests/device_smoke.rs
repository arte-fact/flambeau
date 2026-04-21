//! HIP device smoke test — runs end-to-end against real hardware.
//!
//! The whole test file is gated on `#[cfg(hip_hardware_present)]` which is
//! never set by cargo unless you go out of your way. Instead we check at
//! runtime: if `hipGetDeviceCount` reports zero devices or fails, we skip.
//! That keeps `cargo test -p flambeau-backend-hip` green on hosts that link
//! `libamdhip64` but have no GPUs (containers, CI lint boxes).

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, Stream};

fn maybe_skip() -> Option<i32> {
    match device_count() {
        Ok(n) if n > 0 => Some(n),
        Ok(_) => {
            eprintln!("[skip] 0 HIP devices on this host");
            None
        }
        Err(e) => {
            eprintln!("[skip] HIP runtime not available: {e}");
            None
        }
    }
}

#[test]
fn enumerate_devices() {
    let Some(n) = maybe_skip() else { return };
    println!("HIP device count: {n}");
    assert!(n >= 1);
}

#[test]
fn create_device_and_stream() {
    let Some(_) = maybe_skip() else { return };
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");
    assert_eq!(dev.id(), 0);
    assert_eq!(dev.backend(), "hip");
    dev.default_stream().synchronize().unwrap();
    let extra = dev.new_stream().unwrap();
    extra.synchronize().unwrap();
}

#[test]
fn alloc_copy_roundtrip_256_floats() {
    let Some(_) = maybe_skip() else { return };
    let dev = HipDevice::new(0).unwrap();
    let src: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    let bytes = src.len() * 4;

    let d_ptr = dev.alloc(bytes).unwrap();
    assert!(!d_ptr.is_null());

    // SAFETY: src buffer lives for the full sync cycle below.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_ptr,
            flambeau_core::DevicePtr(src.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    let mut back = vec![0.0f32; 256];
    // SAFETY: back lives for the full sync cycle below.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            flambeau_core::DevicePtr(back.as_mut_ptr() as usize),
            d_ptr,
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    // SAFETY: pointer returned by our alloc; no outstanding work after sync.
    unsafe {
        dev.dealloc(d_ptr, bytes).unwrap();
    }
    assert_eq!(src, back);
}

#[test]
fn multi_device_bind_is_safe() {
    let Some(n) = maybe_skip() else { return };
    if n < 2 {
        eprintln!("[skip] need >= 2 HIP devices, got {n}");
        return;
    }
    let d0 = HipDevice::new(0).unwrap();
    let d1 = HipDevice::new(1).unwrap();
    // Each alloc must bind to the right device internally.
    let p0 = d0.alloc(64).unwrap();
    let p1 = d1.alloc(64).unwrap();
    unsafe {
        d0.dealloc(p0, 64).unwrap();
        d1.dealloc(p1, 64).unwrap();
    }
}
