//! End-to-end smoke for `flambeau_add_f32`: load the cubin, launch, assert
//! `y = a + b` bitwise. Skips when the cubin is absent (`CUDA_SKIP_BUILD`) or
//! no CUDA device is present.

use flambeau_backend_cuda::{device_count, CudaDevice, CudaModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

#[test]
fn add_f32_round_trip() {
    let Some(cubin) = flambeau_kernels_cuda::cubin("add_f32") else {
        eprintln!("add_f32 cubin absent (CUDA_SKIP_BUILD / no nvcc) — skipping");
        return;
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("no CUDA device present — skipping");
        return;
    }

    let dev = CudaDevice::new(0).expect("CudaDevice::new(0)");
    let stream = dev.default_stream();
    let module = CudaModule::load(0, cubin).expect("load add_f32 cubin");
    let kernel = module.kernel("flambeau_add_f32").expect("resolve flambeau_add_f32");

    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
    let bytes = N * std::mem::size_of::<f32>();

    let da = dev.alloc(bytes).expect("alloc a");
    let db = dev.alloc(bytes).expect("alloc b");
    let dy = dev.alloc(bytes).expect("alloc y");

    // SAFETY: host buffers outlive the synchronize below; device allocs are
    // `bytes` each; nothing else is in flight on `stream`.
    unsafe {
        let ha = DevicePtr(a.as_ptr() as usize);
        let hb = DevicePtr(b.as_ptr() as usize);
        dev.memcpy_async(stream, CopyDirection::HostToDevice, da, ha, bytes).expect("H2D a");
        dev.memcpy_async(stream, CopyDirection::HostToDevice, db, hb, bytes).expect("H2D b");
    }

    let a_ptr = da.as_usize() as u64;
    let b_ptr = db.as_usize() as u64;
    let y_ptr = dy.as_usize() as u64;
    let n_i = N as i32;
    let mut args = KernelArgs::new();
    args.push(&a_ptr);
    args.push(&b_ptr);
    args.push(&y_ptr);
    args.push(&n_i);

    let blocks = ((N + 255) / 256) as u32;
    kernel
        .launch(stream, LaunchCfg::one_d(blocks, 256), args)
        .expect("launch flambeau_add_f32");

    let mut y = vec![0f32; N];
    // SAFETY: `y` outlives the synchronize; `dy` is `bytes`.
    unsafe {
        let hy = DevicePtr(y.as_mut_ptr() as usize);
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, hy, dy, bytes).expect("D2H y");
    }
    stream.synchronize().expect("stream sync");

    for i in 0..N {
        assert_eq!(y[i], a[i] + b[i], "mismatch at {i}");
    }

    // SAFETY: each ptr came from `alloc` above; stream is synced so no op is
    // in flight against them.
    unsafe {
        dev.dealloc(da, bytes).ok();
        dev.dealloc(db, bytes).ok();
        dev.dealloc(dy, bytes).ok();
    }
}
