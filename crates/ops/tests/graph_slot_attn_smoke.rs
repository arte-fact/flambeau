//! 6.a-i4 — parity test for `attention_prefill_f16_slots` under
//! `HipGraphExec` capture + `set_slot` update.
//! Plan:
//! 1. Run `attention_prefill_f16` uncaptured at (n_k_tokens=N_K_FINAL,
//! q_offset=Q_OFF_FINAL) → reference output.
//! 2. Capture `attention_prefill_f16_slots` at (N_K_INIT, Q_OFF_INIT),
//! tagging both pos-varying scalars. Discard the first launch's
//! output (captured at the wrong pos).
//! 3. Update both slots via `HipGraphExec::set_slot`, replay.
//! 4. Assert the post-update output bit-matches the reference.
//! Q/K/V buffers are deterministic small F16 values so the kernel
//! output stays in a narrow range. We rely on F16 bit-equality (via
//! `assert_eq` on the raw `u16` bits after cast) — the same kernel
//! launched with the same inputs on gfx906 should produce identical
//! outputs regardless of whether it came through a graph replay or
//! direct dispatch.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async \
              over host/device buffers that live for the bounded synchronize() that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice, HipGraphExec, HipStream, ScalarSlot};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::attention::{attention_prefill_f16, attention_prefill_f16_slots};
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping graph_slot_attn_smoke");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload_f16(dev: &HipDevice, stream: &HipStream, data: &[f16]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
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
fn attention_prefill_slot_update_parity() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };

    let reg = OpsRegistry::new(&dev)?;

    // Shape: small, flash-tile path (n_q >= 4). Single head, head_dim=64.
    const N_Q: usize = 4;
    const N_HEADS_Q: usize = 1;
    const N_HEADS_KV: usize = 1;
    const HEAD_DIM: usize = 64;
    const K_CAPACITY: usize = 16; // max n_k we'll exercise
    const N_K_INIT: usize = 6;
    const Q_OFF_INIT: usize = 2;
    const N_K_FINAL: usize = 10;
    const Q_OFF_FINAL: usize = 6;
    // Scale: 1/sqrt(head_dim). HIP F16 reciprocal-sqrt approximates; any
    // consistent value is fine — we compare two paths that use the
    // identical scalar.
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();

    let stream = HipStream::new_non_blocking(0).unwrap();

    // Deterministic data — small to avoid F16 under/overflow after the
    // scale multiplication. K/V buffer is sized for the max n_k we
    // ever use; Q shared across both paths.
    let q_host: Vec<f16> = (0..N_Q * N_HEADS_Q * HEAD_DIM)
        .map(|i| f16::from_f32(0.01 * (i % 17) as f32))
        .collect();
    let k_host: Vec<f16> = (0..K_CAPACITY * N_HEADS_KV * HEAD_DIM)
        .map(|i| f16::from_f32(0.013 * (i % 13) as f32 + 0.001 * (i % 7) as f32))
        .collect();
    let v_host: Vec<f16> = (0..K_CAPACITY * N_HEADS_KV * HEAD_DIM)
        .map(|i| f16::from_f32(0.007 * (i % 11) as f32 - 0.004 * (i % 5) as f32))
        .collect();

    let q_dev = upload_f16(&dev, &stream, &q_host);
    let k_dev = upload_f16(&dev, &stream, &k_host);
    let v_dev = upload_f16(&dev, &stream, &v_host);
    let out_bytes = N_Q * N_HEADS_Q * HEAD_DIM * 2;
    let out_ref_dev = dev.alloc(out_bytes).unwrap();
    let out_capt_dev = dev.alloc(out_bytes).unwrap();

    // === 1. Reference: uncaptured attn at (N_K_FINAL, Q_OFF_FINAL). ===
    attention_prefill_f16(
        &reg,
        &stream,
        q_dev,
        k_dev,
        v_dev,
        out_ref_dev,
        N_Q,
        N_HEADS_Q,
        N_HEADS_KV,
        HEAD_DIM,
        N_K_FINAL,
        Q_OFF_FINAL,
        scale,
    )?;
    stream.synchronize().unwrap();
    let y_ref = readback_f16(&dev, &stream, out_ref_dev, N_Q * N_HEADS_Q * HEAD_DIM);

    // === 2. Capture at (N_K_INIT, Q_OFF_INIT) with two slots. ===
    let slot_n_k = ScalarSlot::new();
    let slot_q_off = ScalarSlot::new();
    let cap_stream = HipStream::new_non_blocking(0).unwrap();
    let exec = HipGraphExec::capture(&cap_stream, |s| {
        attention_prefill_f16_slots(
            &reg,
            s,
            q_dev,
            k_dev,
            v_dev,
            out_capt_dev,
            N_Q,
            N_HEADS_Q,
            N_HEADS_KV,
            HEAD_DIM,
            N_K_INIT,
            Q_OFF_INIT,
            scale,
            Some(slot_n_k),
            Some(slot_q_off),
        )
        .map_err(|e| flambeau_core::DeviceError::Backend {
            backend: "hip",
            code: -1,
            message: format!("attn capture: {e}"),
        })
    })
    .expect("capture attn_prefill");

    assert_eq!(exec.num_kernel_nodes(), 1, "expected exactly 1 attn kernel");
    let b_n_k = exec.slot_map().get(slot_n_k).expect("slot_n_k bound");
    let b_q_off = exec.slot_map().get(slot_q_off).expect("slot_q_off bound");
    eprintln!(
        "slot_n_k  binding: node={}, arg={}, arity={}",
        b_n_k.kernel_node_idx, b_n_k.arg_index, b_n_k.arity
    );
    eprintln!(
        "slot_q_off binding: node={}, arg={}, arity={}",
        b_q_off.kernel_node_idx, b_q_off.arg_index, b_q_off.arity
    );
    assert_eq!(b_n_k.kernel_node_idx, 0);
    assert_eq!(b_q_off.kernel_node_idx, 0);
    // Flash-tile path: args are [q, k, v, out, n_q, n_heads_q, n_heads_kv,
    // n_k, q_off, scale] → n_k at idx 7, q_off at idx 8.
    assert_eq!(b_n_k.arg_index, 7, "n_k expected at arg idx 7");
    assert_eq!(b_q_off.arg_index, 8, "q_off expected at arg idx 8");
    assert_eq!(b_n_k.arity, 10);

    // Sanity check: pre-update replay should match uncaptured at
    // (N_K_INIT, Q_OFF_INIT). If this fails, the capture itself is
    // wrong, not the slot update.
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let y_capt_init = readback_f16(&dev, &stream, out_capt_dev, N_Q * N_HEADS_Q * HEAD_DIM);

    let out_init_ref_dev = dev.alloc(out_bytes).unwrap();
    attention_prefill_f16(
        &reg,
        &stream,
        q_dev,
        k_dev,
        v_dev,
        out_init_ref_dev,
        N_Q,
        N_HEADS_Q,
        N_HEADS_KV,
        HEAD_DIM,
        N_K_INIT,
        Q_OFF_INIT,
        scale,
    )?;
    stream.synchronize().unwrap();
    let y_init_ref = readback_f16(&dev, &stream, out_init_ref_dev, N_Q * N_HEADS_Q * HEAD_DIM);
    let init_match = y_capt_init
        .iter()
        .zip(y_init_ref.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("pre-update capture vs uncaptured @ init params: match = {init_match}");
    unsafe {
        dev.dealloc(out_init_ref_dev, out_bytes).unwrap();
    }

    // === 3. Update both slots to (N_K_FINAL, Q_OFF_FINAL). ===
    let n_k_final_i: i32 = N_K_FINAL as i32;
    let q_off_final_i: i32 = Q_OFF_FINAL as i32;
    unsafe {
        exec.set_slot(slot_n_k, &n_k_final_i).expect("set_slot n_k");
        exec.set_slot(slot_q_off, &q_off_final_i)
            .expect("set_slot q_off");
    }

    // === 4. Replay + readback. ===
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let y_capt = readback_f16(&dev, &stream, out_capt_dev, N_Q * N_HEADS_Q * HEAD_DIM);

    // === Parity. ===
    let mut mismatches = 0usize;
    for (i, (a, b)) in y_ref.iter().zip(y_capt.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "  mismatch @ {i}: ref={:.6} (bits {:#06x})  capt={:.6} (bits {:#06x})",
                    a.to_f32(),
                    a.to_bits(),
                    b.to_f32(),
                    b.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{} attn outputs differ between uncaptured (N_K={N_K_FINAL}, Q_OFF={Q_OFF_FINAL}) \
         and captured-then-slot-updated paths",
        y_ref.len()
    );

    unsafe {
        dev.dealloc(q_dev, q_host.len() * 2).unwrap();
        dev.dealloc(k_dev, k_host.len() * 2).unwrap();
        dev.dealloc(v_dev, v_host.len() * 2).unwrap();
        dev.dealloc(out_ref_dev, out_bytes).unwrap();
        dev.dealloc(out_capt_dev, out_bytes).unwrap();
    }

    // Keep scalar args alive.
    let _ = (n_k_final_i, q_off_final_i);
    Ok(())
}
