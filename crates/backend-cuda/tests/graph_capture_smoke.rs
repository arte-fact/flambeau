//! `CudaGraphExec` core smoke: capture a device-to-device memcpy, then replay.
//! Each replay re-runs the captured copy, so after mutating the source on the
//! device a second `launch` propagates the new bytes to the destination —
//! proving capture → instantiate → launch works without any per-replay rebind.
//! Single GPU; no NCCL.

use flambeau_backend_cuda::{CudaDevice, CudaGraphExec};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

fn upload(dev: &CudaDevice, data: &[f32]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    // SAFETY: host slice outlives the synchronous HtoD copy; d holds `bytes`.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::HostToDevice, d, DevicePtr(data.as_ptr() as usize), bytes)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn download(dev: &CudaDevice, ptr: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    // SAFETY: out holds n f32; ptr holds n f32.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(out.as_mut_ptr() as usize), ptr, n * 4)
            .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

#[test]
fn cuda_graph_capture_replay_dtod_memcpy() {
    let dev = CudaDevice::new(0).expect("CudaDevice");
    let n = 8usize;
    let bytes = n * 4;
    let src = upload(&dev, &[1.0f32; 8]);
    let dst = dev.alloc(bytes).unwrap();
    let cap_stream = dev.new_stream().unwrap();

    // Capture: dst <- src (device-to-device, async → recorded as a graph node).
    let graph = CudaGraphExec::capture(&cap_stream, |s| {
        // SAFETY: src/dst are live device buffers of `bytes`.
        unsafe { dev.memcpy_async(s, CopyDirection::DeviceToDevice, dst, src, bytes) }
    })
    .expect("capture");
    assert_eq!(graph.device_id(), 0);

    // First replay copies the [1.0; 8] source into dst.
    graph.launch(&cap_stream).unwrap();
    cap_stream.synchronize().unwrap();
    assert_eq!(download(&dev, dst, n), vec![1.0f32; n]);

    // Mutate the source on-device, replay → dst tracks the new bytes (the graph
    // re-runs the same DtoD copy, no rebind).
    let src2 = upload(&dev, &[7.0f32; 8]);
    // SAFETY: both live, `bytes` each.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToDevice, src, src2, bytes).unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    graph.launch(&cap_stream).unwrap();
    cap_stream.synchronize().unwrap();
    assert_eq!(download(&dev, dst, n), vec![7.0f32; n]);

    // SAFETY: each buffer alloc'd with `bytes`.
    unsafe {
        dev.dealloc(src, bytes).unwrap();
        dev.dealloc(src2, bytes).unwrap();
        dev.dealloc(dst, bytes).unwrap();
    }
}
