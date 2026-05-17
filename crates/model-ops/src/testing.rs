//! Test-only helpers shared across op tests.
//!
//! Each op file defines its own CPU reference function. This module
//! provides the device-side scaffolding: alloc / upload / download /
//! assert_close. All helpers panic on HIP errors — they are tests, not
//! production code, and a HIP failure during a test is a test failure.

#![cfg(test)]

use bytemuck::Pod;
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};

use crate::dtype::ElemType;
use crate::tensor::Tensor;

/// Borrow rank 0's `HipDevice`. Every test in this crate uses a single
/// device (multi-rank ops still parameterise over a slice of buffers,
/// but the buffers can all live on device 0 for testing — we're not
/// exercising P2P here, the parity checks are kernel-level).
pub fn test_device() -> HipDevice {
    HipDevice::new(0).expect("HIP device 0 required for model-ops tests")
}

/// Alloc `n_elems * T::bytes_per_elem()` zeroed bytes on the device,
/// wrap as a `Tensor<T>`. Returns the tensor and the raw pointer so
/// the test can free at the end.
pub fn alloc<T: ElemType>(device: &HipDevice, n_elems: usize) -> (Tensor<T>, DevicePtr) {
    let bytes = n_elems * T::bytes_per_elem();
    let ptr = device.alloc(bytes).expect("device alloc");
    // SAFETY: ptr is a fresh device allocation of the expected size.
    let t = unsafe { Tensor::<T>::from_raw(ptr, n_elems) };
    (t, ptr)
}

/// Upload a host slice into a fresh device buffer wrapped as `Tensor<T>`.
pub fn upload<T: ElemType, P: Pod>(device: &HipDevice, host: &[P]) -> (Tensor<T>, DevicePtr) {
    let bytes = std::mem::size_of_val(host);
    let ptr = device.alloc(bytes).expect("device alloc");
    let stream = device.default_stream();
    // SAFETY: ptr owns `bytes`; `host.as_ptr()` valid for `bytes`.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host.as_ptr() as usize),
                bytes,
            )
            .expect("memcpy HtoD");
    }
    stream.synchronize().expect("stream sync");
    let n_elems = bytes / T::bytes_per_elem();
    // SAFETY: ptr is a fresh allocation of `bytes`, matched n_elems.
    let t = unsafe { Tensor::<T>::from_raw(ptr, n_elems) };
    (t, ptr)
}

/// Download a `Tensor<T>` to a fresh host `Vec<P>`. `P` must have a
/// byte-compatible layout with `T` (e.g. `T = F16`, `P = half::f16`).
pub fn download<T: ElemType, P: Pod + Default + Clone>(
    device: &HipDevice,
    tensor: &Tensor<T>,
) -> Vec<P> {
    let bytes = tensor.bytes();
    let n_elems_p = bytes / std::mem::size_of::<P>();
    let mut host = vec![P::default(); n_elems_p];
    let stream = device.default_stream();
    // SAFETY: tensor.ptr owns `bytes`; host has `bytes` writeable bytes.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                tensor.ptr,
                bytes,
            )
            .expect("memcpy DtoH");
    }
    stream.synchronize().expect("stream sync");
    host
}

/// Free a tensor's backing allocation. Tests should call this for
/// every `(_, ptr)` they got back from `alloc` / `upload`.
pub fn free(device: &HipDevice, ptr: DevicePtr, bytes: usize) {
    if !ptr.is_null() && bytes > 0 {
        // SAFETY: ptr was produced by `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(ptr, bytes).expect("dealloc");
        }
    }
}

/// Assert that two equal-length `f32` slices match elementwise within
/// `abs_tol` OR `rel_tol`. Element-by-element check with a useful
/// failure message (index, expected, got, abs/rel error). Used by
/// every op test in the crate.
pub fn assert_close_f32(got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(
        got.len(),
        expected.len(),
        "length mismatch: got {}, expected {}",
        got.len(),
        expected.len()
    );
    let mut worst: Option<(usize, f32, f32, f32, f32)> = None;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs = (g - e).abs();
        let rel = if e.abs() > 1e-9 { abs / e.abs() } else { abs };
        let ok = abs <= abs_tol || rel <= rel_tol;
        if !ok {
            match worst {
                Some((_, _, _, a, _)) if a >= abs => {}
                _ => worst = Some((i, g, e, abs, rel)),
            }
        }
    }
    if let Some((i, g, e, abs, rel)) = worst {
        panic!(
            "tensor mismatch at idx {i}: got={g} expected={e} abs={abs} rel={rel} \
             (abs_tol={abs_tol} rel_tol={rel_tol})"
        );
    }
}

/// Convenience: `assert_close_f32` after converting an `f16` slice to
/// `f32`. Op tests usually upload `f32` host data, run a kernel that
/// produces `f16`, download, convert to `f32` for the assert.
pub fn assert_close_f16(got: &[half::f16], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    let got_f32: Vec<f32> = got.iter().map(|h| h.to_f32()).collect();
    assert_close_f32(&got_f32, expected, abs_tol, rel_tol);
}
