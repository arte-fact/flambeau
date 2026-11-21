//! Parity test for `kv_append_v_unit_norm_f16_slots`: `slots=None` must write
//! a byte-identical K/V cache to the plain `kv_append_v_unit_norm_f16`. This is
//! the transparency contract for the graph-capture variant outside capture —
//! the `write_pos` scalar arg is only tagged when a real slot is supplied.

#![cfg(feature = "hip")]

use flambeau_backend_hip::{device_count, HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::{HipOps, KvAppendBuffers, KvAppendVUnitShape, Ops, OpsRegistry};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping kv_append_v_unit_norm_slots_parity");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload_f16(dev: &HipDevice, stream: &HipStream, data: &[f16]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    // SAFETY: fresh allocation of `bytes`; host slice valid for the copy that
    // completes at the synchronize below.
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    d
}

fn readback_f16(dev: &HipDevice, stream: &HipStream, src: DevicePtr, n: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; n];
    // SAFETY: `src` owns `n * 2` bytes; `out` is `n * 2` writeable bytes; the
    // copy completes at the synchronize below.
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 2,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    out
}

#[test]
fn kv_append_v_unit_norm_slots_none_bit_identical() {
    let Some(dev) = dev_or_skip() else { return };
    let stream = dev.default_stream();
    let reg = OpsRegistry::new(&dev).expect("OpsRegistry::new");
    let ops = HipOps::new(&reg, stream);

    const N_TOKENS: usize = 1;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const KV_WIDTH: usize = N_KV_HEADS * HEAD_DIM;
    const MAX_SEQ: usize = 8;
    const WRITE_POS: usize = 3;
    const EPS: f32 = 1e-6;

    let k_src: Vec<f16> =
        (0..N_TOKENS * KV_WIDTH).map(|i| f16::from_f32(((i as f32) * 0.013 + 0.2).sin() * 0.5)).collect();
    let v_src: Vec<f16> =
        (0..N_TOKENS * KV_WIDTH).map(|i| f16::from_f32(((i as f32) * 0.021 - 0.1).cos() * 0.5)).collect();
    let cache_init: Vec<f16> = vec![f16::from_f32(-7.0); MAX_SEQ * KV_WIDTH];

    let k_src_p = upload_f16(&dev, stream, &k_src);
    let v_src_p = upload_f16(&dev, stream, &v_src);

    let k_dst_a = upload_f16(&dev, stream, &cache_init);
    let v_dst_a = upload_f16(&dev, stream, &cache_init);
    let k_dst_b = upload_f16(&dev, stream, &cache_init);
    let v_dst_b = upload_f16(&dev, stream, &cache_init);

    let shape = KvAppendVUnitShape {
        n_tokens: N_TOKENS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        ring_depth: 0,
    };

    ops.kv_append_v_unit_norm_f16(
        KvAppendBuffers { k_src: k_src_p, v_src: v_src_p, k_dst: k_dst_a, v_dst: v_dst_a },
        shape,
        WRITE_POS,
        EPS,
    )
    .expect("plain kv_append_v_unit_norm_f16");

    ops.kv_append_v_unit_norm_f16_slots(
        KvAppendBuffers { k_src: k_src_p, v_src: v_src_p, k_dst: k_dst_b, v_dst: v_dst_b },
        shape,
        WRITE_POS,
        EPS,
        None,
    )
    .expect("kv_append_v_unit_norm_f16_slots slots=None");

    stream.synchronize().unwrap();

    let ka = readback_f16(&dev, stream, k_dst_a, MAX_SEQ * KV_WIDTH);
    let kb = readback_f16(&dev, stream, k_dst_b, MAX_SEQ * KV_WIDTH);
    let va = readback_f16(&dev, stream, v_dst_a, MAX_SEQ * KV_WIDTH);
    let vb = readback_f16(&dev, stream, v_dst_b, MAX_SEQ * KV_WIDTH);

    let bits = |xs: &[f16]| xs.iter().map(|x| x.to_bits()).collect::<Vec<u16>>();
    assert_eq!(bits(&ka), bits(&kb), "K cache: slots=None must be bit-identical");
    assert_eq!(bits(&va), bits(&vb), "V cache: slots=None must be bit-identical");

    for (p, bytes) in [
        (k_src_p, k_src.len() * 2),
        (v_src_p, v_src.len() * 2),
        (k_dst_a, cache_init.len() * 2),
        (v_dst_a, cache_init.len() * 2),
        (k_dst_b, cache_init.len() * 2),
        (v_dst_b, cache_init.len() * 2),
    ] {
        // SAFETY: each ptr is a live allocation of `bytes` from this test.
        unsafe { Device::dealloc(&dev, p, bytes).unwrap() };
    }
}
