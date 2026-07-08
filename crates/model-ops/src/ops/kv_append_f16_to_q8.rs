//! Quantize-then-append F16 K/V into Q8_0 cache slabs. Each row of
//! `kv_width` F16 elements is quantized into `kv_width / 32` Q8_0
//! blocks of 34 bytes each, and written into the cache at the row
//! offset specified by `write_pos`. Positions are contiguous over
//! `n_tokens` (decode = 1 row, prefill chunk = chunk rows starting at
//! the chunk's first position).
//!
//! Caller invariants:
//! - `kv_width % 32 == 0` (Q8_0 block size).
//! - `write_pos + n_tokens <= max_seq_len` (cache cap).
//! - `k_cache.n_elems` and `v_cache.n_elems` count Q8_0 blocks, sized
//!   for `max_seq_len * (kv_width / 32)`.

use anyhow::bail;
use flambeau_core::{DevicePtr, ScalarSlot};
use flambeau_ops::Ops;

use crate::dtype::{F16, Q8_0};
use crate::error::Result;
use crate::tensor::Tensor;

const QK8_0: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

/// Graph-capture destination slots for [`kv_append_f16_to_q8_slots`]. Tags the
/// two Q8_0 cache-write destination pointers — one per quantize launch — so a
/// captured graph rebinds each per replay. Valid only for the single-segment
/// (`ring_depth == 0`) append: a ring wrap splits the write into two segments
/// (four launches) whose topology a stable graph can't express.
#[derive(Copy, Clone, Debug)]
pub struct KvAppendQ8DstSlots {
    pub k_dst: ScalarSlot,
    pub v_dst: ScalarSlot,
}

pub fn kv_append_f16_to_q8(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_cache: &mut Tensor<Q8_0>,
    v_cache: &mut Tensor<Q8_0>,
    spec: crate::ops::kv_append::KvAppendSpec,
    ops: &impl Ops,
) -> Result<()> {
    kv_append_f16_to_q8_slots(k_src, v_src, k_cache, v_cache, spec, None, ops)
}

/// Graph-captureable variant of [`kv_append_f16_to_q8`]. With `slots=None` the
/// behaviour is byte-identical to the wrapper; with `Some`, the K and V Q8_0
/// cache-write destination pointers are tagged so a captured graph rebinds them
/// per replay. Requires `ring_depth == 0` (single segment) when `slots` is
/// `Some` — see [`KvAppendQ8DstSlots`].
#[allow(clippy::too_many_arguments)]
pub fn kv_append_f16_to_q8_slots(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_cache: &mut Tensor<Q8_0>,
    v_cache: &mut Tensor<Q8_0>,
    spec: crate::ops::kv_append::KvAppendSpec,
    slots: Option<KvAppendQ8DstSlots>,
    ops: &impl Ops,
) -> Result<()> {
    let crate::ops::kv_append::KvAppendSpec { n_tokens, kv_width, write_pos, max_seq_len, ring_depth } =
        spec;
    if n_tokens == 0 {
        return Ok(());
    }
    if kv_width % QK8_0 != 0 {
        bail!(
            "kv_append_f16_to_q8: kv_width {kv_width} must be a multiple of {QK8_0}"
        );
    }
    let src_need = n_tokens * kv_width;
    if k_src.n_elems < src_need {
        bail!(
            "kv_append_f16_to_q8: k_src has {} F16 elems, need >= {src_need}",
            k_src.n_elems
        );
    }
    if v_src.n_elems < src_need {
        bail!(
            "kv_append_f16_to_q8: v_src has {} F16 elems, need >= {src_need}",
            v_src.n_elems
        );
    }
    if ring_depth == 0 {
        if write_pos + n_tokens > max_seq_len {
            bail!(
                "kv_append_f16_to_q8: write_pos {write_pos} + n_tokens {n_tokens} > \
                 max_seq_len {max_seq_len}"
            );
        }
    } else if n_tokens > ring_depth {
        bail!("kv_append_f16_to_q8: n_tokens {n_tokens} > ring_depth {ring_depth}");
    }
    // `Tensor<Q8_0>::n_elems` is the LOGICAL F16-equivalent element
    // count (per the model-ops Tensor contract on block dtypes), so
    // the cache must hold `max_seq_len * kv_width` logical elements.
    let cache_elems_need = max_seq_len * kv_width;
    if k_cache.n_elems < cache_elems_need {
        bail!(
            "kv_append_f16_to_q8: k_cache has {} logical Q8_0 elems, need >= {cache_elems_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_elems_need {
        bail!(
            "kv_append_f16_to_q8: v_cache has {} logical Q8_0 elems, need >= {cache_elems_need}",
            v_cache.n_elems
        );
    }
    // Slot tagging is valid only for the single-segment append: a ring wrap
    // (ring_depth != 0) can split into two segments = four quantize launches,
    // which no stable graph topology can express. Surface it rather than
    // silently emitting extra launches that would break the captured graph.
    if slots.is_some() && ring_depth != 0 {
        bail!(
            "kv_append_f16_to_q8_slots: dst slots require ring_depth == 0 (single \
             segment); got ring_depth {ring_depth}"
        );
    }
    let (k_slot, v_slot) = match slots {
        Some(s) => (Some(s.k_dst), Some(s.v_dst)),
        None => (None, None),
    };
    let blocks_per_row = kv_width / QK8_0;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    // A ring write that straddles the slab boundary splits into a head
    // segment ending at the last physical row and a wrap segment from row 0.
    // The contiguous quantize can't wrap in one call, so each segment is a
    // separate launch; `ring_depth == 0` collapses to the single absolute
    // segment (bit-identical to the pre-ring path).
    for (phys_row, src_row, rows) in
        crate::ops::kv_append::ring_append_segments(write_pos, n_tokens, ring_depth)
    {
        if rows == 0 {
            continue;
        }
        let elems = rows * kv_width;
        let k_dst: DevicePtr = k_cache.ptr.offset_bytes(phys_row * row_bytes);
        let v_dst: DevicePtr = v_cache.ptr.offset_bytes(phys_row * row_bytes);
        let k_src_seg = k_src.ptr.offset_bytes(src_row * kv_width * 2);
        let v_src_seg = v_src.ptr.offset_bytes(src_row * kv_width * 2);
        ops.quantize_f16_q8_0_slots(k_src_seg, k_dst, elems, k_slot)?;
        ops.quantize_f16_q8_0_slots(v_src_seg, v_dst, elems, v_slot)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flambeau_ops::HipOps;
    use crate::testing::{alloc, download, free, test_device, test_ops_registry, upload};
    use flambeau_core::{Device, Stream};
    use half::f16;

    #[test]
    fn kv_append_f16_to_q8_writes_at_correct_block_offset() {
        const KV_WIDTH: usize = 64;
        const MAX_SEQ_LEN: usize = 8;
        const N_TOKENS: usize = 2;
        const WRITE_POS: usize = 3;
        const BLOCKS_PER_ROW: usize = KV_WIDTH / QK8_0;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let k_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(0.1 + (i as f32) * 0.01))
            .collect();
        let v_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(-0.2 + (i as f32) * 0.013))
            .collect();

        const CACHE_LOGICAL_ELEMS: usize = MAX_SEQ_LEN * KV_WIDTH;
        const CACHE_BYTES: usize = MAX_SEQ_LEN * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
        let cache_init: Vec<u8> = vec![0xAB; CACHE_BYTES];

        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_src_host, k_src_host.len());
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_src_host, v_src_host.len());
        let (mut k_cache_t, k_cache_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);
        let (mut v_cache_t, v_cache_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);

        kv_append_f16_to_q8(
            &k_src_t,
            &v_src_t,
            &mut k_cache_t,
            &mut v_cache_t,
            crate::ops::kv_append::KvAppendSpec {
                n_tokens: N_TOKENS,
                kv_width: KV_WIDTH,
                write_pos: WRITE_POS,
                max_seq_len: MAX_SEQ_LEN,
                ring_depth: 0,
            },
            &ops,
        )
        .expect("kv_append_f16_to_q8");
        stream.synchronize().expect("stream sync");

        let k_raw: Vec<u8> = download::<Q8_0, u8>(&device, &k_cache_t);
        let v_raw: Vec<u8> = download::<Q8_0, u8>(&device, &v_cache_t);

        // Rows < WRITE_POS unchanged (still 0xAB).
        for row in 0..WRITE_POS {
            for block in 0..BLOCKS_PER_ROW {
                let byte_off = (row * BLOCKS_PER_ROW + block) * Q8_0_BLOCK_BYTES;
                for b in &k_raw[byte_off..byte_off + Q8_0_BLOCK_BYTES] {
                    assert_eq!(*b, 0xAB, "k_cache row {row} block {block}: was overwritten");
                }
                for b in &v_raw[byte_off..byte_off + Q8_0_BLOCK_BYTES] {
                    assert_eq!(*b, 0xAB);
                }
            }
        }

        // Rows [WRITE_POS, WRITE_POS+N_TOKENS) — dequant back and check
        // within Q8_0 tolerance vs source.
        let written_blocks = N_TOKENS * BLOCKS_PER_ROW;
        let written_byte_off = WRITE_POS * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
        let k_written = &k_raw[written_byte_off..written_byte_off + written_blocks * Q8_0_BLOCK_BYTES];
        let v_written = &v_raw[written_byte_off..written_byte_off + written_blocks * Q8_0_BLOCK_BYTES];
        let k_got = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            k_written,
            N_TOKENS * KV_WIDTH,
        )
        .expect("dequant K");
        let v_got = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            v_written,
            N_TOKENS * KV_WIDTH,
        )
        .expect("dequant V");
        let k_expected_f32: Vec<f32> = k_src_host.iter().map(|x| x.to_f32()).collect();
        let v_expected_f32: Vec<f32> = v_src_host.iter().map(|x| x.to_f32()).collect();
        for i in 0..N_TOKENS * KV_WIDTH {
            assert!(
                (k_got[i] - k_expected_f32[i]).abs() < 0.05,
                "K[{i}]: got {} expected {}",
                k_got[i],
                k_expected_f32[i]
            );
            assert!(
                (v_got[i] - v_expected_f32[i]).abs() < 0.05,
                "V[{i}]: got {} expected {}",
                v_got[i],
                v_expected_f32[i]
            );
        }

        // Rows after the written range unchanged.
        for row in (WRITE_POS + N_TOKENS)..MAX_SEQ_LEN {
            for block in 0..BLOCKS_PER_ROW {
                let byte_off = (row * BLOCKS_PER_ROW + block) * Q8_0_BLOCK_BYTES;
                for b in &k_raw[byte_off..byte_off + Q8_0_BLOCK_BYTES] {
                    assert_eq!(*b, 0xAB);
                }
            }
        }

        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_ptr, k_cache_t.bytes());
        free(&device, v_cache_ptr, v_cache_t.bytes());
        let _ = alloc::<Q8_0>;
    }

    /// `kv_append_f16_to_q8_slots(None)` must write byte-identical Q8_0 cache
    /// contents to `kv_append_f16_to_q8` on the same inputs — the transparency
    /// contract for the graph-capture variant outside capture.
    #[test]
    fn kv_append_f16_to_q8_slots_none_bit_identical() {
        const KV_WIDTH: usize = 64;
        const MAX_SEQ_LEN: usize = 8;
        const N_TOKENS: usize = 2;
        const WRITE_POS: usize = 3;
        const BLOCKS_PER_ROW: usize = KV_WIDTH / QK8_0;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let k_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(0.1 + (i as f32) * 0.01))
            .collect();
        let v_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(-0.2 + (i as f32) * 0.013))
            .collect();

        const CACHE_LOGICAL_ELEMS: usize = MAX_SEQ_LEN * KV_WIDTH;
        const CACHE_BYTES: usize = MAX_SEQ_LEN * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
        let cache_init: Vec<u8> = vec![0xAB; CACHE_BYTES];

        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_src_host, k_src_host.len());
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_src_host, v_src_host.len());
        let (mut k_cache_a, k_cache_a_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);
        let (mut v_cache_a, v_cache_a_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);
        let (mut k_cache_b, k_cache_b_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);
        let (mut v_cache_b, v_cache_b_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);

        let spec = crate::ops::kv_append::KvAppendSpec {
            n_tokens: N_TOKENS,
            kv_width: KV_WIDTH,
            write_pos: WRITE_POS,
            max_seq_len: MAX_SEQ_LEN,
            ring_depth: 0,
        };

        kv_append_f16_to_q8(&k_src_t, &v_src_t, &mut k_cache_a, &mut v_cache_a, spec, &ops)
            .expect("kv_append_f16_to_q8");
        kv_append_f16_to_q8_slots(
            &k_src_t,
            &v_src_t,
            &mut k_cache_b,
            &mut v_cache_b,
            spec,
            None,
            &ops,
        )
        .expect("kv_append_f16_to_q8_slots(None)");
        stream.synchronize().expect("stream sync");

        let k_a: Vec<u8> = download::<Q8_0, u8>(&device, &k_cache_a);
        let v_a: Vec<u8> = download::<Q8_0, u8>(&device, &v_cache_a);
        let k_b: Vec<u8> = download::<Q8_0, u8>(&device, &k_cache_b);
        let v_b: Vec<u8> = download::<Q8_0, u8>(&device, &v_cache_b);

        assert_eq!(k_a, k_b, "slots=None K cache must be byte-identical to plain");
        assert_eq!(v_a, v_b, "slots=None V cache must be byte-identical to plain");

        let written_blocks = N_TOKENS * BLOCKS_PER_ROW;
        let off = WRITE_POS * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
        let k_a_deq = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            &k_a[off..off + written_blocks * Q8_0_BLOCK_BYTES],
            N_TOKENS * KV_WIDTH,
        )
        .expect("dequant K a");
        let k_b_deq = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            &k_b[off..off + written_blocks * Q8_0_BLOCK_BYTES],
            N_TOKENS * KV_WIDTH,
        )
        .expect("dequant K b");
        assert_eq!(k_a_deq, k_b_deq, "dequantized K parity");

        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_a_ptr, k_cache_a.bytes());
        free(&device, v_cache_a_ptr, v_cache_a.bytes());
        free(&device, k_cache_b_ptr, k_cache_b.bytes());
        free(&device, v_cache_b_ptr, v_cache_b.bytes());
    }

    #[test]
    fn kv_append_f16_to_q8_ring_wrap_splits_quantize() {
        // Ring slab DEPTH rows; a 4-row write at logical pos DEPTH-2 lands
        // physical rows [DEPTH-2, DEPTH-1] then wraps to [0, 1]. The Q8
        // quantize can't wrap in one call, so this exercises the split.
        const KV_WIDTH: usize = 64;
        const DEPTH: usize = 6;
        const N_TOKENS: usize = 4;
        const WRITE_POS: usize = 10; // 10 % 6 = 4 → straddles (4,5,0,1)
        const BLOCKS_PER_ROW: usize = KV_WIDTH / QK8_0;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let k_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(0.1 + (i as f32) * 0.01))
            .collect();
        let v_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(-0.2 + (i as f32) * 0.013))
            .collect();

        const CACHE_LOGICAL_ELEMS: usize = DEPTH * KV_WIDTH;
        const CACHE_BYTES: usize = DEPTH * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
        let cache_init: Vec<u8> = vec![0xAB; CACHE_BYTES];

        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_src_host, k_src_host.len());
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_src_host, v_src_host.len());
        let (mut k_cache_t, k_cache_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);
        let (mut v_cache_t, v_cache_ptr) =
            upload::<Q8_0, u8>(&device, &cache_init, CACHE_LOGICAL_ELEMS);

        kv_append_f16_to_q8(
            &k_src_t,
            &v_src_t,
            &mut k_cache_t,
            &mut v_cache_t,
            crate::ops::kv_append::KvAppendSpec {
                n_tokens: N_TOKENS,
                kv_width: KV_WIDTH,
                write_pos: WRITE_POS,
                max_seq_len: DEPTH,
                ring_depth: DEPTH,
            },
            &ops,
        )
        .expect("kv_append_f16_to_q8 ring");
        stream.synchronize().expect("stream sync");

        let k_raw: Vec<u8> = download::<Q8_0, u8>(&device, &k_cache_t);
        let v_raw: Vec<u8> = download::<Q8_0, u8>(&device, &v_cache_t);

        // Token t → physical row (WRITE_POS + t) % DEPTH; dequant matches src.
        for tok in 0..N_TOKENS {
            let phys = (WRITE_POS + tok) % DEPTH;
            let byte_off = phys * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
            let k_row = &k_raw[byte_off..byte_off + BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES];
            let v_row = &v_raw[byte_off..byte_off + BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES];
            let k_got = flambeau_quant::dequantize_to_vec(
                flambeau_quant::GgmlDType::Q8_0,
                k_row,
                KV_WIDTH,
            )
            .expect("dequant K");
            let v_got = flambeau_quant::dequantize_to_vec(
                flambeau_quant::GgmlDType::Q8_0,
                v_row,
                KV_WIDTH,
            )
            .expect("dequant V");
            for col in 0..KV_WIDTH {
                let src_idx = tok * KV_WIDTH + col;
                assert!(
                    (k_got[col] - k_src_host[src_idx].to_f32()).abs() < 0.05,
                    "K phys row {phys} (tok {tok}) col {col}: got {} expected {}",
                    k_got[col],
                    k_src_host[src_idx].to_f32()
                );
                assert!(
                    (v_got[col] - v_src_host[src_idx].to_f32()).abs() < 0.05,
                    "V phys row {phys} col {col}"
                );
            }
        }
        // Physical rows 2 and 3 untouched (still 0xAB).
        for phys in [2usize, 3] {
            let byte_off = phys * BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES;
            for b in &k_raw[byte_off..byte_off + BLOCKS_PER_ROW * Q8_0_BLOCK_BYTES] {
                assert_eq!(*b, 0xAB, "k_cache phys row {phys} overwritten");
            }
        }

        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_ptr, k_cache_t.bytes());
        free(&device, v_cache_ptr, v_cache_t.bytes());
    }
}
