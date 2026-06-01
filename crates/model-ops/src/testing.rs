//! Device-side test scaffolding (alloc / upload / download /
//! assert_close). Panics on HIP errors — these are tests.

#![cfg(test)]

use bytemuck::Pod;
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::OpsRegistry;

use crate::dtype::ElemType;
use crate::tensor::Tensor;

pub fn test_device() -> HipDevice {
    HipDevice::new(0).expect("HIP device 0 required for model-ops tests")
}

pub fn test_ops_registry(device: &HipDevice) -> OpsRegistry {
    OpsRegistry::new(device).expect("OpsRegistry::new")
}

pub fn alloc<T: ElemType>(device: &HipDevice, n_elems: usize) -> (Tensor<T>, DevicePtr) {
    let bytes = T::bytes_for_n_elems(n_elems);
    let ptr = device.alloc(bytes).expect("device alloc");
    // SAFETY: ptr is a fresh device allocation of the expected size.
    let t = unsafe { Tensor::<T>::from_raw(ptr, n_elems) };
    (t, ptr)
}

/// `n_elems` is the LOGICAL element count; allocated bytes come from
/// `T::bytes_for_n_elems(n_elems)`. For quant dtypes the host slice
/// must already be pre-packed in the GGUF block layout.
pub fn upload<T: ElemType, P: Pod>(
    device: &HipDevice,
    host: &[P],
    n_elems: usize,
) -> (Tensor<T>, DevicePtr) {
    let bytes_host = std::mem::size_of_val(host);
    let bytes_alloc = T::bytes_for_n_elems(n_elems);
    assert!(
        bytes_host == bytes_alloc,
        "upload: host bytes {bytes_host} != T::bytes_for_n_elems({n_elems})={bytes_alloc} for dtype {}",
        T::name()
    );
    let ptr = device.alloc(bytes_alloc).expect("device alloc");
    let stream = device.default_stream();
    // SAFETY: ptr owns `bytes_alloc`; `host.as_ptr()` valid for `bytes_host == bytes_alloc`.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host.as_ptr() as usize),
                bytes_alloc,
            )
            .expect("memcpy HtoD");
    }
    stream.synchronize().expect("stream sync");
    // SAFETY: ptr is a fresh allocation of `bytes_alloc`, matched n_elems.
    let t = unsafe { Tensor::<T>::from_raw(ptr, n_elems) };
    (t, ptr)
}

/// `P` must be byte-compatible with `T` (e.g. `T=F16`, `P=half::f16`).
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

pub fn free(device: &HipDevice, ptr: DevicePtr, bytes: usize) {
    if !ptr.is_null() && bytes > 0 {
        // SAFETY: ptr was produced by `device.alloc(bytes)` above.
        unsafe {
            device.dealloc(ptr, bytes).expect("dealloc");
        }
    }
}

/// Element-wise within `abs_tol` OR `rel_tol`; panics on the worst-
/// abs mismatch.
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

pub fn assert_close_f16(got: &[half::f16], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    let got_f32: Vec<f32> = got.iter().map(|h| h.to_f32()).collect();
    assert_close_f32(&got_f32, expected, abs_tol, rel_tol);
}
