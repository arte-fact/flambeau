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
use flambeau_core::DevicePtr;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, Q8_0};
use crate::error::Result;
use crate::tensor::Tensor;

const QK8_0: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

pub fn kv_append_f16_to_q8(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_cache: &mut Tensor<Q8_0>,
    v_cache: &mut Tensor<Q8_0>,
    n_tokens: usize,
    kv_width: usize,
    write_pos: usize,
    max_seq_len: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
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
    if write_pos + n_tokens > max_seq_len {
        bail!(
            "kv_append_f16_to_q8: write_pos {write_pos} + n_tokens {n_tokens} > \
             max_seq_len {max_seq_len}"
        );
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
    let blocks_per_row = kv_width / QK8_0;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    let k_dst: DevicePtr = k_cache.ptr.offset_bytes(write_pos * row_bytes);
    let v_dst: DevicePtr = v_cache.ptr.offset_bytes(write_pos * row_bytes);
    ops.quantize_f16_q8_0(k_src.ptr, k_dst, src_need)?;
    ops.quantize_f16_q8_0(v_src.ptr, v_dst, src_need)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
            N_TOKENS,
            KV_WIDTH,
            WRITE_POS,
            MAX_SEQ_LEN,
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
}
