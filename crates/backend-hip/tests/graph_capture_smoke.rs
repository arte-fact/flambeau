//! 6.a — HipGraphExec smoke test.
//! Captures a 2-step `memcpy_async` subgraph into an exec, replays it
//! N times, and verifies the final device buffer equals the last input.
//! Goal is to prove the FFI + wrapper cycle (begin/end capture,
//! instantiate, launch, destroy) is correct on gfx906; perf measurement
//! lives in the forward-path integration commit.
//! 6.a-i2 adds the kernel-node param-update POC: capture a real
//! `flambeau_scale_f32` launch with scale=2.0, replay (y = x·2.0), then
//! call `hipGraphExecKernelNodeSetParams` to swap scale → 5.0, replay
//! again, verify y = x·5.0. Proves the per-node param update path on
//! gfx906 — the enabling mechanism for 6.a-i3 through -i7.

use flambeau_backend_hip::module::{HipModule, KernelArgs, LaunchCfg};
use flambeau_backend_hip::sys::{hipDim3, hipKernelNodeParams};
use flambeau_backend_hip::{
    device_count, HipDevice, HipGraphExec, HipStream, MemcpySlot, ScalarSlot,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use std::ffi::c_void;

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
fn capture_replay_memcpy() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");

    let n = 1024usize;
    let bytes = n * 4;

    let src_staging = dev.alloc(bytes).unwrap();
    let dst = dev.alloc(bytes).unwrap();

    // Non-blocking stream so the captured graph doesn't touch the null
    // stream (capture on null stream is a documented error on HIP).
    let cap_stream = HipStream::new_non_blocking(0).unwrap();

    // Capture: a single device-to-device copy from `src_staging` to `dst`.
    // At replay we rewrite `src_staging` beforehand, so the graph copies
    // whatever is currently there. That's the pattern the forward path
    // uses: record once, feed different scratch contents per replay.
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        // SAFETY: both pointers are owned by `dev` and live for the full
        // capture+launch cycle. Capture mode records the call; no actual
        // copy happens here.
        unsafe {
            dev.memcpy_async(s, CopyDirection::DeviceToDevice, dst, src_staging, bytes)?;
        }
        Ok(())
    })
    .expect("HipGraphExec::capture");

    assert_eq!(exec.device_id(), 0);

    // Replay three times with different inputs. After each launch+sync,
    // `dst` should equal the pre-launch contents of `src_staging`.
    let launch_stream = HipStream::new_non_blocking(0).unwrap();
    for iter in 0..3u32 {
        let input: Vec<f32> = (0..n).map(|i| (iter as f32) * 100.0 + i as f32).collect();
        // SAFETY: input lives for the full sync below.
        unsafe {
            dev.memcpy_async(
                &launch_stream,
                CopyDirection::HostToDevice,
                src_staging,
                DevicePtr(input.as_ptr() as usize),
                bytes,
            )
            .unwrap();
        }
        launch_stream.synchronize().unwrap();

        exec.launch(&launch_stream).unwrap();
        launch_stream.synchronize().unwrap();

        let mut back = vec![0f32; n];
        // SAFETY: back lives for the sync below.
        unsafe {
            dev.memcpy_async(
                &launch_stream,
                CopyDirection::DeviceToHost,
                DevicePtr(back.as_mut_ptr() as usize),
                dst,
                bytes,
            )
            .unwrap();
        }
        launch_stream.synchronize().unwrap();

        assert_eq!(back, input, "graph replay iter {iter} mismatch");
    }

    // SAFETY: no outstanding work — all syncs above.
    unsafe {
        dev.dealloc(src_staging, bytes).unwrap();
        dev.dealloc(dst, bytes).unwrap();
    }
}

#[test]
fn capture_empty_graph_is_noop() {
    if !maybe_skip() {
        return;
    }
    let _dev = HipDevice::new(0).expect("HipDevice::new(0)");
    let s = HipStream::new_non_blocking(0).unwrap();

    // Capture an empty closure — should still produce a valid exec that
    // launches as a noop. Verifies end-capture tolerates no recorded work.
    let exec = HipGraphExec::capture(&s, |_s| Ok(())).expect("empty capture");
    exec.launch(&s).unwrap();
    s.synchronize().unwrap();
}

/// 6.a-i2 POC: capture `flambeau_scale_f32` with scale=2.0, instantiate
/// the exec, replay and verify y = x·2.0, then use
/// `hipGraphExecKernelNodeSetParams` to swap scale → 5.0, replay, and
/// verify y = x·5.0.
/// Proves we can update scalar kernel params on an instantiated graph
/// exec — the foundation 6.a-i3+ needs to make `forward_layer_prefill`
/// captureable across different pos values.
#[test]
fn kernel_param_update_scale_f32() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");

    // Load the scale_f32 kernel from flambeau_kernels_hip's hsaco catalogue.
    let hsaco = flambeau_kernels_hip::hsaco("scale_f32")
        .expect("scale_f32.hsaco present in this build (else HIP_SKIP_BUILD set)");
    let module = HipModule::load(0, hsaco).expect("load scale_f32 module");
    let kernel = module.kernel("flambeau_scale_f32").expect("kernel fn");

    let n: usize = 1024;
    let bytes = n * 4;
    let x_dev = dev.alloc(bytes).unwrap();
    let y_dev = dev.alloc(bytes).unwrap();

    // Host-side x = [1.0, 2.0, ..., N] upload.
    let x_host: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let stream = HipStream::new_non_blocking(0).unwrap();
    // SAFETY: x_host lives for the sync below.
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

    // Stable argument storage — kernelParams[i] pointers reference these
    // slots at capture time and must remain live across both the initial
    // capture + launch and the later param-update + launch.
    let x_ptr_u64: u64 = x_dev.as_usize() as u64;
    let y_ptr_u64: u64 = y_dev.as_usize() as u64;
    let n_i: i32 = n as i32;
    let captured_scale: f32 = 2.0;

    // Capture the scale_f32 launch with scale=2.0.
    let cap_stream = HipStream::new_non_blocking(0).unwrap();
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        let mut args = KernelArgs::new();
        args.push(&x_ptr_u64);
        args.push(&y_ptr_u64);
        args.push(&n_i);
        args.push(&captured_scale);
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        // SAFETY: all arg-storage references outlive the launch (lexically
        // bound above). x_dev / y_dev are live device buffers on this dev.
        unsafe { kernel.launch(s, cfg, args)? };
        Ok(())
    })
    .expect("HipGraphExec::capture scale_f32");

    assert_eq!(
        exec.num_kernel_nodes(),
        1,
        "expected exactly 1 captured kernel node, got {}",
        exec.num_kernel_nodes()
    );

    // First replay — should compute y = x · 2.0.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut y_host = vec![0.0f32; n];
    // SAFETY: y_host lives for the sync below.
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(y_host.as_mut_ptr() as usize),
            y_dev,
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    for i in 0..n {
        let expect = (i + 1) as f32 * 2.0;
        assert!(
            (y_host[i] - expect).abs() < 1e-4,
            "post-capture launch y[{i}]={} want {expect}",
            y_host[i]
        );
    }

    // Build new kernel params with scale = 5.0 via
    // hipGraphExecKernelNodeSetParams. kernel_params[3] points to a fresh
    // host f32 holding 5.0; the other 3 slots re-use the existing
    // x_ptr_u64 / y_ptr_u64 / n_i storage.
    let new_scale: f32 = 5.0;
    let mut new_kernel_params: [*mut c_void; 4] = [
        std::ptr::from_ref(&x_ptr_u64) as *mut c_void,
        std::ptr::from_ref(&y_ptr_u64) as *mut c_void,
        std::ptr::from_ref(&n_i) as *mut c_void,
        std::ptr::from_ref(&new_scale) as *mut c_void,
    ];
    let current = exec
        .get_kernel_node_params(0)
        .expect("get_kernel_node_params");
    assert!(
        !current.func.is_null(),
        "captured node func must be non-null"
    );
    assert_eq!(current.grid_dim.x, (n as u32).div_ceil(256));
    assert_eq!(current.block_dim.x, 256);

    let new_params = hipKernelNodeParams {
        block_dim: current.block_dim,
        extra: std::ptr::null_mut(),
        func: current.func,
        grid_dim: current.grid_dim,
        kernel_params: new_kernel_params.as_mut_ptr(),
        shared_mem_bytes: current.shared_mem_bytes,
    };
    // SAFETY: new_kernel_params covers the 4 args scale_f32 expects, with
    // correct sizes / types at each slot. The driver copies param values
    // from the pointed-to storage during this call; the arrays may be
    // freed afterwards.
    unsafe {
        exec.set_kernel_node_params(0, &new_params)
            .expect("set_kernel_node_params");
    }

    // Second replay — should compute y = x · 5.0.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    // SAFETY: y_host lives for the sync below.
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(y_host.as_mut_ptr() as usize),
            y_dev,
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    for i in 0..n {
        let expect = (i + 1) as f32 * 5.0;
        assert!(
            (y_host[i] - expect).abs() < 1e-3,
            "post-param-update launch y[{i}]={} want {expect}",
            y_host[i]
        );
    }

    // SAFETY: all work on x_dev / y_dev has synced above.
    unsafe {
        dev.dealloc(x_dev, bytes).unwrap();
        dev.dealloc(y_dev, bytes).unwrap();
    }

    // Keep these alive until here (arg storage lifetime).
    let _ = (captured_scale, new_scale, new_kernel_params);
    let _ = hipDim3::default();
}

/// 6.a-i3 end-to-end: capture TWO `flambeau_scale_f32` launches in
/// sequence, each tagging its `scale` arg with a distinct ScalarSlot.
/// The resulting exec's SlotMap should bind the first slot to kernel
/// node 0 and the second to kernel node 1. Update both slots via
/// `set_slot`, replay, verify both output buffers reflect the updated
/// scales.
/// This proves the thread-local launch-recorder lines up with
/// hipGraphGetNodes's dispatch-order node enumeration on gfx906 — the
/// 6.a-i4 pos-rewiring depends on this 1:1 mapping.
#[test]
fn slot_map_two_launches_round_trip() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");
    let hsaco = flambeau_kernels_hip::hsaco("scale_f32").expect("scale_f32.hsaco present");
    let module = HipModule::load(0, hsaco).unwrap();
    let kernel = module.kernel("flambeau_scale_f32").unwrap();

    let n: usize = 256;
    let bytes = n * 4;
    // Two independent (x, y) pairs; each captured launch targets its own y.
    let x_dev = dev.alloc(bytes).unwrap();
    let y0_dev = dev.alloc(bytes).unwrap();
    let y1_dev = dev.alloc(bytes).unwrap();

    let x_host: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let stream = HipStream::new_non_blocking(0).unwrap();
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

    // Stable storage for arg pointers during and after capture.
    let x_ptr_u64: u64 = x_dev.as_usize() as u64;
    let y0_ptr_u64: u64 = y0_dev.as_usize() as u64;
    let y1_ptr_u64: u64 = y1_dev.as_usize() as u64;
    let n_i: i32 = n as i32;
    let init_scale_0: f32 = 2.0;
    let init_scale_1: f32 = 3.0;

    // Allocate slots BEFORE capture.
    let slot_0 = ScalarSlot::new();
    let slot_1 = ScalarSlot::new();

    let cap_stream = HipStream::new_non_blocking(0).unwrap();
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        // Launch 1: y0 = x * 2.0; tag scale at arg index 3.
        let mut args0 = KernelArgs::new();
        args0.push(&x_ptr_u64);
        args0.push(&y0_ptr_u64);
        args0.push(&n_i);
        args0.push_slot(&init_scale_0, slot_0);
        let cfg = LaunchCfg::one_d((n as u32).div_ceil(256), 256);
        // SAFETY: arg storage lives for the full capture + replay cycle.
        unsafe { kernel.launch(s, cfg, args0)? };

        // Launch 2: y1 = x * 3.0; tag scale at arg index 3.
        let mut args1 = KernelArgs::new();
        args1.push(&x_ptr_u64);
        args1.push(&y1_ptr_u64);
        args1.push(&n_i);
        args1.push_slot(&init_scale_1, slot_1);
        // SAFETY: see above.
        unsafe { kernel.launch(s, cfg, args1)? };
        Ok(())
    })
    .expect("capture");

    assert_eq!(exec.num_kernel_nodes(), 2);
    let map = exec.slot_map();
    let b0 = map.get(slot_0).expect("slot_0 bound");
    let b1 = map.get(slot_1).expect("slot_1 bound");
    assert_eq!(b0.kernel_node_idx, 0);
    assert_eq!(b0.arg_index, 3);
    assert_eq!(b0.arity, 4);
    assert_eq!(b1.kernel_node_idx, 1);
    assert_eq!(b1.arg_index, 3);
    assert_eq!(b1.arity, 4);

    // First replay — outputs = x·2, x·3.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    // Update both slots: scale_0 = 7.0, scale_1 = 11.0.
    let new_scale_0: f32 = 7.0;
    let new_scale_1: f32 = 11.0;
    // SAFETY: new_scale_{0,1} are f32 matching scale_f32's 4th arg;
    // each outlives the set_slot call (stack-bound in this function).
    unsafe {
        exec.set_slot(slot_0, &new_scale_0)
            .expect("set_slot(slot_0)");
        exec.set_slot(slot_1, &new_scale_1)
            .expect("set_slot(slot_1)");
    }

    // Second replay — outputs = x·7, x·11.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut y0_host = vec![0f32; n];
    let mut y1_host = vec![0f32; n];
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(y0_host.as_mut_ptr() as usize),
            y0_dev,
            bytes,
        )
        .unwrap();
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(y1_host.as_mut_ptr() as usize),
            y1_dev,
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    for i in 0..n {
        let want_0 = (i + 1) as f32 * 7.0;
        let want_1 = (i + 1) as f32 * 11.0;
        assert!(
            (y0_host[i] - want_0).abs() < 1e-3,
            "y0[{i}]={} want {want_0}",
            y0_host[i]
        );
        assert!(
            (y1_host[i] - want_1).abs() < 1e-3,
            "y1[{i}]={} want {want_1}",
            y1_host[i]
        );
    }

    // SAFETY: all work has synced above.
    unsafe {
        dev.dealloc(x_dev, bytes).unwrap();
        dev.dealloc(y0_dev, bytes).unwrap();
        dev.dealloc(y1_dev, bytes).unwrap();
    }

    // Storage-lifetime anchors.
    let _ = (init_scale_0, init_scale_1, new_scale_0, new_scale_1);
}

/// 6.a-i5b POC: capture a D→D memcpy tagged with a `MemcpySlot`,
/// replay to verify the dst landed, then use `set_memcpy_slot` to
/// retarget dst to a different device buffer, replay, verify the new
/// dst got the same data (and the old dst is unchanged since the
/// second replay).
/// Proves the memcpy-node update path (`hipGraphExecMemcpyNodeSetParams1D`)
/// on gfx906 — the foundation 6.a-i5b's KvCache::append wiring needs.
#[test]
fn memcpy_slot_update_round_trip() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).expect("HipDevice::new(0)");

    let n = 256usize;
    let bytes = n * 4;
    let src_dev = dev.alloc(bytes).unwrap();
    let dst_a_dev = dev.alloc(bytes).unwrap();
    let dst_b_dev = dev.alloc(bytes).unwrap();

    // Seed src with a known pattern; zero both dsts so we can tell
    // unambiguously which one the replay wrote into.
    let src_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
    let stream = HipStream::new_non_blocking(0).unwrap();
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::HostToDevice,
            src_dev,
            DevicePtr(src_host.as_ptr() as usize),
            bytes,
        )
        .unwrap();
        let zeros = vec![0f32; n];
        dev.memcpy_async(
            &stream,
            CopyDirection::HostToDevice,
            dst_a_dev,
            DevicePtr(zeros.as_ptr() as usize),
            bytes,
        )
        .unwrap();
        dev.memcpy_async(
            &stream,
            CopyDirection::HostToDevice,
            dst_b_dev,
            DevicePtr(zeros.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Capture a tagged D→D memcpy (src_dev → dst_a_dev).
    let slot = MemcpySlot::new();
    let cap_stream = HipStream::new_non_blocking(0).unwrap();
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        // SAFETY: all buffers live for the full capture + replay cycle.
        unsafe {
            dev.memcpy_async_slot(
                s,
                CopyDirection::DeviceToDevice,
                dst_a_dev,
                src_dev,
                bytes,
                slot,
            )?;
        }
        Ok(())
    })
    .expect("capture tagged memcpy");

    // Slot must be bound to a memcpy node.
    let binding = exec.slot_map().get_memcpy(slot).expect("memcpy slot bound");
    assert_eq!(binding.memcpy_node_idx, 0);
    assert_eq!(binding.count, bytes);
    assert_eq!(binding.dst, dst_a_dev.as_usize());
    assert_eq!(binding.src, src_dev.as_usize());

    // Pre-update replay — dst_a should get the data, dst_b stays zero.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut a_back = vec![0f32; n];
    let mut b_back = vec![0f32; n];
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(a_back.as_mut_ptr() as usize),
            dst_a_dev,
            bytes,
        )
        .unwrap();
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(b_back.as_mut_ptr() as usize),
            dst_b_dev,
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(a_back, src_host, "pre-update: dst_a mismatch");
    assert!(
        b_back.iter().all(|v| *v == 0.0),
        "pre-update: dst_b should still be zero"
    );

    // Retarget dst → dst_b_dev via set_memcpy_slot.
    // SAFETY: dst_b_dev is live for `bytes` device writes.
    unsafe {
        exec.set_memcpy_slot(slot, dst_b_dev)
            .expect("set_memcpy_slot");
    }
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    let mut b_back2 = vec![0f32; n];
    unsafe {
        dev.memcpy_async(
            &stream,
            CopyDirection::DeviceToHost,
            DevicePtr(b_back2.as_mut_ptr() as usize),
            dst_b_dev,
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(b_back2, src_host, "post-update: dst_b didn't receive src");

    // SAFETY: syncs above.
    unsafe {
        dev.dealloc(src_dev, bytes).unwrap();
        dev.dealloc(dst_a_dev, bytes).unwrap();
        dev.dealloc(dst_b_dev, bytes).unwrap();
    }
}
