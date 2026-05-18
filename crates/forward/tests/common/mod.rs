//! Shared helpers for the synth_*.rs integration tests. Each test
//! binary runs in its own process so the multi-Session-per-process
//! state leakage on qwen35-9B-style workloads doesn't interfere.

#![cfg(feature = "hip")]
#![allow(dead_code)] // each binary only uses a subset

use bytemuck::Pod;
use flambeau_backend_hip::HipDevice;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::ctx::QuantWeight;
use flambeau_model_ops::{Tensor, F16};
use flambeau_quant::quantize_k::quantize_row_q8_0;
use half::f16;

pub struct DeviceAllocs {
    pub device: HipDevice,
    pub allocs: Vec<(DevicePtr, usize)>,
}

impl DeviceAllocs {
    pub fn new(device: HipDevice) -> Self {
        Self {
            device,
            allocs: Vec::new(),
        }
    }

    pub fn upload<P: Pod>(&mut self, host: &[P]) -> (DevicePtr, usize) {
        let bytes = std::mem::size_of_val(host);
        let ptr = self.device.alloc(bytes).expect("alloc");
        let stream = self.device.default_stream();
        unsafe {
            self.device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    ptr,
                    DevicePtr(host.as_ptr() as usize),
                    bytes,
                )
                .expect("memcpy HtoD");
        }
        stream.synchronize().expect("sync");
        self.allocs.push((ptr, bytes));
        (ptr, bytes)
    }

    pub fn upload_f16(&mut self, host_f32: &[f32]) -> Tensor<F16> {
        let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let (ptr, _) = self.upload(&host_f16);
        unsafe { Tensor::<F16>::from_raw(ptr, host_f16.len()) }
    }

    pub fn upload_f32(&mut self, host_f32: &[f32]) -> Tensor<flambeau_model_ops::F32> {
        let (ptr, _) = self.upload(host_f32);
        unsafe { Tensor::<flambeau_model_ops::F32>::from_raw(ptr, host_f32.len()) }
    }

    pub fn upload_q8_0(&mut self, host_f32: &[f32], rows: usize, cols: usize) -> QuantWeight {
        assert_eq!(host_f32.len(), rows * cols);
        assert!(cols % 32 == 0, "Q8_0 needs cols % 32 == 0");
        let mut bytes: Vec<u8> = Vec::with_capacity(rows * cols / 32 * 34);
        for r in 0..rows {
            quantize_row_q8_0(&host_f32[r * cols..(r + 1) * cols], &mut bytes);
        }
        let (ptr, _) = self.upload(&bytes);
        QuantWeight {
            ptr,
            dtype: QDtype::Q8_0,
            n_elems: rows * cols,
        }
    }
}

impl Drop for DeviceAllocs {
    fn drop(&mut self) {
        for (ptr, bytes) in self.allocs.drain(..) {
            unsafe {
                let _ = self.device.dealloc(ptr, bytes);
            }
        }
    }
}

/// Deterministic small-magnitude signal for synth weights.
pub fn det_signal(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let s = (seed as f32) * 0.013 + (i as f32) * 0.027;
            (s.sin() + (s * 1.7).cos()) * 0.1
        })
        .collect()
}
