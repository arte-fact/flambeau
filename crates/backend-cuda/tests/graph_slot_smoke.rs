//! `CudaGraphExec` slot-map smoke — the hard ABI gate for Stage A.
//!
//! `capture_with_slots` captures a real `flambeau_scale_f32` launch with its
//! `scale` arg tagged via `KernelArgs::push_slot`; after instantiate we replay
//! (y = x·scale0), then `set_slot` a NEW scale and replay again, asserting the
//! device output equals x·scale1 bit-for-bit. A green run proves the v1
//! `CUDA_KERNEL_NODE_PARAMS` struct layout + the base-symbol
//! `cuGraphExecKernelNodeSetParams` binding are ABI-correct — a wrong struct
//! would write garbage into the launch arg frame and the values would diverge.
//!
//! A second test drives two launches with distinct slots (proving the launch
//! recorder lines up 1:1 with `cuGraphGetNodes` dispatch order), and a third
//! exercises the memcpy source-swap path (`set_memcpy_slot`).

use flambeau_backend_cuda::module::{CudaModule, KernelArgs, LaunchCfg};
use flambeau_backend_cuda::{device_count, CudaDevice, CudaGraphExec, CudaStream, MemcpySlot, ScalarSlot};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

fn maybe_skip() -> bool {
    if device_count().unwrap_or(0) < 1 {
        eprintln!("no CUDA device present — skipping");
        return false;
    }
    true
}

/// Capture `flambeau_scale_f32` with scale tagged as a `ScalarSlot`, replay
/// (y = x·2), `set_slot` scale → 5, replay (y = x·5). The hard ABI gate.
#[test]
fn slot_scalar_update_scale_f32() {
    if !maybe_skip() {
        return;
    }
    let Some(cubin) = flambeau_kernels_cuda::cubin("scale_f32") else {
        eprintln!("scale_f32 cubin absent (CUDA_SKIP_BUILD / no nvcc) — skipping");
        return;
    };
    let dev = CudaDevice::new(0).expect("CudaDevice::new(0)");
    let module = CudaModule::load(0, cubin).expect("load scale_f32 cubin");
    let kernel = module.kernel("flambeau_scale_f32").expect("resolve flambeau_scale_f32");

    let n: usize = 1024;
    let bytes = n * 4;
    let x_dev = dev.alloc(bytes).unwrap();
    let y_dev = dev.alloc(bytes).unwrap();

    let x_host: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let stream = dev.new_stream().unwrap();
    // SAFETY: x_host outlives the sync below; x_dev holds `bytes`.
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::HostToDevice,
            x_dev,
            DevicePtr(x_host.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Stable arg storage — kernelParams pointers reference these across both
    // the capture+launch and the later set_slot+launch.
    let x_ptr_u64: u64 = x_dev.as_usize() as u64;
    let y_ptr_u64: u64 = y_dev.as_usize() as u64;
    let n_i: i32 = n as i32;
    let captured_scale: f32 = 2.0;

    let slot = ScalarSlot::new();
    let cap_stream = dev.new_stream().unwrap();
    let exec = CudaGraphExec::capture_with_slots(&cap_stream, |s| {
        let mut args = KernelArgs::new();
        args.push(&x_ptr_u64);
        args.push(&y_ptr_u64);
        args.push(&n_i);
        args.push_slot(&captured_scale, slot);
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        // SAFETY: arg storage outlives the launch (function-scoped above);
        // x_dev/y_dev are live buffers on this device.
        kernel.launch(s, cfg, args)
    })
    .expect("capture_with_slots scale_f32");

    assert_eq!(exec.num_kernel_nodes(), 1, "expected exactly 1 kernel node");
    let binding = exec.slot_map().get(slot).expect("scale slot bound");
    assert_eq!(binding.kernel_node_idx, 0);
    assert_eq!(binding.arg_index, 3);
    assert_eq!(binding.arity, 4);

    // First replay — y = x·2.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let y0 = download(&dev, &stream, y_dev, n);
    for (i, &got) in y0.iter().enumerate() {
        let want = (i + 1) as f32 * 2.0;
        assert!((got - want).abs() < 1e-4, "pre-update y[{i}]={got} want {want}");
    }

    // Rebind scale → 5.0, replay — y = x·5. Proves the v1 struct ABI.
    let new_scale: f32 = 5.0;
    // SAFETY: new_scale is an f32 matching scale_f32's 4th arg and outlives
    // this call (function-scoped).
    unsafe {
        exec.set_slot(slot, &new_scale).expect("set_slot(scale)");
    }
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let y1 = download(&dev, &stream, y_dev, n);
    for (i, &got) in y1.iter().enumerate() {
        let want = (i + 1) as f32 * 5.0;
        assert_eq!(got, want, "post-update y[{i}]={got} want {want} (bit-identical)");
    }

    // SAFETY: all work synced above.
    unsafe {
        dev.dealloc(x_dev, bytes).unwrap();
        dev.dealloc(y_dev, bytes).unwrap();
    }
    let _ = (captured_scale, new_scale);
}

/// Two `flambeau_scale_f32` launches, each tagging its scale with a distinct
/// slot. Bindings must land on kernel nodes 0 and 1 in dispatch order; update
/// both and verify both outputs.
#[test]
fn slot_map_two_launches_round_trip() {
    if !maybe_skip() {
        return;
    }
    let Some(cubin) = flambeau_kernels_cuda::cubin("scale_f32") else {
        eprintln!("scale_f32 cubin absent — skipping");
        return;
    };
    let dev = CudaDevice::new(0).expect("CudaDevice::new(0)");
    let module = CudaModule::load(0, cubin).unwrap();
    let kernel = module.kernel("flambeau_scale_f32").unwrap();

    let n: usize = 256;
    let bytes = n * 4;
    let x_dev = dev.alloc(bytes).unwrap();
    let y0_dev = dev.alloc(bytes).unwrap();
    let y1_dev = dev.alloc(bytes).unwrap();

    let x_host: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let stream = dev.new_stream().unwrap();
    // SAFETY: x_host outlives the sync; x_dev holds `bytes`.
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::HostToDevice,
            x_dev,
            DevicePtr(x_host.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    let x_ptr_u64: u64 = x_dev.as_usize() as u64;
    let y0_ptr_u64: u64 = y0_dev.as_usize() as u64;
    let y1_ptr_u64: u64 = y1_dev.as_usize() as u64;
    let n_i: i32 = n as i32;
    let init_scale_0: f32 = 2.0;
    let init_scale_1: f32 = 3.0;

    let slot_0 = ScalarSlot::new();
    let slot_1 = ScalarSlot::new();

    let cap_stream = dev.new_stream().unwrap();
    let exec = CudaGraphExec::capture_with_slots(&cap_stream, |s| {
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        let mut a0 = KernelArgs::new();
        a0.push(&x_ptr_u64);
        a0.push(&y0_ptr_u64);
        a0.push(&n_i);
        a0.push_slot(&init_scale_0, slot_0);
        // SAFETY: arg storage function-scoped; buffers live.
        kernel.launch(s, cfg, a0)?;

        let mut a1 = KernelArgs::new();
        a1.push(&x_ptr_u64);
        a1.push(&y1_ptr_u64);
        a1.push(&n_i);
        a1.push_slot(&init_scale_1, slot_1);
        kernel.launch(s, cfg, a1)
    })
    .expect("capture");

    assert_eq!(exec.num_kernel_nodes(), 2);
    let b0 = *exec.slot_map().get(slot_0).expect("slot_0 bound");
    let b1 = *exec.slot_map().get(slot_1).expect("slot_1 bound");
    assert_eq!((b0.kernel_node_idx, b0.arg_index, b0.arity), (0, 3, 4));
    assert_eq!((b1.kernel_node_idx, b1.arg_index, b1.arity), (1, 3, 4));

    // Rebind both, replay, verify y0 = x·7, y1 = x·11.
    let new_scale_0: f32 = 7.0;
    let new_scale_1: f32 = 11.0;
    // SAFETY: both f32, function-scoped, matching arg type.
    unsafe {
        exec.set_slot(slot_0, &new_scale_0).unwrap();
        exec.set_slot(slot_1, &new_scale_1).unwrap();
    }
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let y0 = download(&dev, &stream, y0_dev, n);
    let y1 = download(&dev, &stream, y1_dev, n);
    for i in 0..n {
        assert_eq!(y0[i], (i + 1) as f32 * 7.0, "y0[{i}]");
        assert_eq!(y1[i], (i + 1) as f32 * 11.0, "y1[{i}]");
    }

    // SAFETY: synced above.
    unsafe {
        dev.dealloc(x_dev, bytes).unwrap();
        dev.dealloc(y0_dev, bytes).unwrap();
        dev.dealloc(y1_dev, bytes).unwrap();
    }
    let _ = (init_scale_0, init_scale_1, new_scale_0, new_scale_1);
}

/// Capture a D→D memcpy tagged with a `MemcpySlot`, replay (dst gets src_a),
/// then `set_memcpy_slot` to re-source from src_b, replay, verify dst gets
/// src_b. Proves `cuGraphExecMemcpyNodeSetParams` + the `CUDA_MEMCPY3D` layout.
#[test]
fn memcpy_slot_src_swap_round_trip() {
    if !maybe_skip() {
        return;
    }
    let dev = CudaDevice::new(0).expect("CudaDevice::new(0)");
    let n = 256usize;
    let bytes = n * 4;
    let src_a = dev.alloc(bytes).unwrap();
    let src_b = dev.alloc(bytes).unwrap();
    let dst = dev.alloc(bytes).unwrap();

    let a_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..n).map(|i| 1000.0 + i as f32).collect();
    let stream = dev.new_stream().unwrap();
    // SAFETY: host buffers outlive the sync; device buffers hold `bytes`.
    unsafe {
        dev.memcpy_async(&stream, CopyDirection::HostToDevice, src_a, DevicePtr(a_host.as_ptr() as usize), bytes).unwrap();
        dev.memcpy_async(&stream, CopyDirection::HostToDevice, src_b, DevicePtr(b_host.as_ptr() as usize), bytes).unwrap();
        let zeros = vec![0f32; n];
        dev.memcpy_async(&stream, CopyDirection::HostToDevice, dst, DevicePtr(zeros.as_ptr() as usize), bytes).unwrap();
        stream.synchronize().unwrap();
    }

    let slot = MemcpySlot::new();
    let cap_stream = dev.new_stream().unwrap();
    let exec = CudaGraphExec::capture_with_slots(&cap_stream, |s| {
        // SAFETY: all buffers live for the full capture + replay cycle.
        unsafe { dev.memcpy_async_slot(s, CopyDirection::DeviceToDevice, dst, src_a, bytes, slot) }
    })
    .expect("capture tagged memcpy");

    let binding = *exec.slot_map().get_memcpy(slot).expect("memcpy slot bound");
    assert_eq!(binding.memcpy_node_idx, 0);
    assert_eq!(binding.count, bytes);
    assert_eq!(binding.src, src_a.as_usize());
    assert_eq!(binding.dst, dst.as_usize());

    // Pre-update replay — dst gets src_a.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(download(&dev, &stream, dst, n), a_host, "pre-update dst != src_a");

    // Re-source from src_b, replay — dst gets src_b.
    // SAFETY: src_b is live for `bytes` device reads.
    unsafe {
        exec.set_memcpy_slot(slot, src_b).expect("set_memcpy_slot");
    }
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(download(&dev, &stream, dst, n), b_host, "post-update dst != src_b");

    // SAFETY: synced above.
    unsafe {
        dev.dealloc(src_a, bytes).unwrap();
        dev.dealloc(src_b, bytes).unwrap();
        dev.dealloc(dst, bytes).unwrap();
    }
}

fn download(dev: &CudaDevice, stream: &CudaStream, ptr: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    // SAFETY: out holds n f32; ptr holds n f32; synced before read.
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            ptr,
            n * 4,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    out
}
