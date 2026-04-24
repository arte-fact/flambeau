//! V2.26.a — HipGraphExec smoke test.
//!
//! Captures a 2-step `memcpy_async` subgraph into an exec, replays it
//! N times, and verifies the final device buffer equals the last input.
//! Goal is to prove the FFI + wrapper cycle (begin/end capture,
//! instantiate, launch, destroy) is correct on gfx906; perf measurement
//! lives in the forward-path integration commit.

use flambeau_backend_hip::{device_count, HipDevice, HipGraphExec, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};

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
