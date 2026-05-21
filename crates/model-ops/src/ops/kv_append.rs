//! DtoD memcpy of `[n_tokens, kv_width]` K + V into the per-layer
//! cache at row `write_pos`. Takes `&HipDevice + &HipStream` (not
//! `&HipOps`) because it's a memcpy, not a kernel launch.

use anyhow::bail;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// Two stream-ordered DtoD memcpys (K and V) into per-layer caches.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_f16(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_cache: &mut Tensor<F16>,
    v_cache: &mut Tensor<F16>,
    n_tokens: usize,
    kv_width: usize,
    write_pos: usize,
    max_seq_len: usize,
    device: &HipDevice,
    stream: &HipStream,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }
    let src_need = n_tokens * kv_width;
    if k_src.n_elems < src_need {
        bail!(
            "kv_append_f16: k_src has {} F16 elems, need >= {src_need}",
            k_src.n_elems
        );
    }
    if v_src.n_elems < src_need {
        bail!(
            "kv_append_f16: v_src has {} F16 elems, need >= {src_need}",
            v_src.n_elems
        );
    }
    if write_pos + n_tokens > max_seq_len {
        bail!(
            "kv_append_f16: write_pos {write_pos} + n_tokens {n_tokens} > max_seq_len {max_seq_len}"
        );
    }
    let cache_need = max_seq_len * kv_width;
    if k_cache.n_elems < cache_need {
        bail!(
            "kv_append_f16: k_cache has {} F16 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "kv_append_f16: v_cache has {} F16 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    let bytes = n_tokens * kv_width * 2;
    let row_bytes = kv_width * 2;
    let k_dst = k_cache.ptr.offset_bytes(write_pos * row_bytes);
    let v_dst = v_cache.ptr.offset_bytes(write_pos * row_bytes);
    // SAFETY: bounds checks above guarantee `bytes` of valid storage at
    // both src and dst on the same device; stream is live.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            k_dst,
            k_src.ptr,
            bytes,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToDevice,
            v_dst,
            v_src.ptr,
            bytes,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{alloc, download, free, test_device, upload};
    use flambeau_core::Stream;
    use half::f16;

    #[test]
    fn kv_append_f16_writes_at_correct_row_offset() {
        const KV_WIDTH: usize = 16;
        const MAX_SEQ_LEN: usize = 8;
        const N_TOKENS: usize = 2;
        const WRITE_POS: usize = 3;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();

        let k_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(0.1 + (i as f32) * 0.01))
            .collect();
        let v_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(-0.2 + (i as f32) * 0.013))
            .collect();
        let cache_init: Vec<f16> = vec![f16::from_f32(-9.0); MAX_SEQ_LEN * KV_WIDTH];

        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_src_host, k_src_host.len());
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_src_host, v_src_host.len());
        let (mut k_cache_t, k_cache_ptr) =
            upload::<F16, f16>(&device, &cache_init, cache_init.len());
        let (mut v_cache_t, v_cache_ptr) =
            upload::<F16, f16>(&device, &cache_init, cache_init.len());

        kv_append_f16(
            &k_src_t,
            &v_src_t,
            &mut k_cache_t,
            &mut v_cache_t,
            N_TOKENS,
            KV_WIDTH,
            WRITE_POS,
            MAX_SEQ_LEN,
            &device,
            stream,
        )
        .expect("kv_append_f16");
        stream.synchronize().expect("stream sync");

        let k_got: Vec<f16> = download::<F16, f16>(&device, &k_cache_t);
        let v_got: Vec<f16> = download::<F16, f16>(&device, &v_cache_t);

        // Rows < WRITE_POS unchanged.
        for row in 0..WRITE_POS {
            for col in 0..KV_WIDTH {
                let idx = row * KV_WIDTH + col;
                assert_eq!(
                    k_got[idx].to_f32(),
                    -9.0,
                    "k_cache row {row} col {col}: was overwritten"
                );
                assert_eq!(v_got[idx].to_f32(), -9.0);
            }
        }
        // Rows [WRITE_POS, WRITE_POS+N_TOKENS) match the source.
        for tok in 0..N_TOKENS {
            for col in 0..KV_WIDTH {
                let cache_idx = (WRITE_POS + tok) * KV_WIDTH + col;
                let src_idx = tok * KV_WIDTH + col;
                assert_eq!(
                    k_got[cache_idx],
                    k_src_host[src_idx],
                    "k_cache row {} col {col}: expected src",
                    WRITE_POS + tok
                );
                assert_eq!(v_got[cache_idx], v_src_host[src_idx]);
            }
        }
        // Rows after the written range unchanged.
        for row in (WRITE_POS + N_TOKENS)..MAX_SEQ_LEN {
            for col in 0..KV_WIDTH {
                let idx = row * KV_WIDTH + col;
                assert_eq!(k_got[idx].to_f32(), -9.0);
                assert_eq!(v_got[idx].to_f32(), -9.0);
            }
        }

        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_ptr, k_cache_t.bytes());
        free(&device, v_cache_ptr, v_cache_t.bytes());
        // suppress unused: alloc is not used in this test.
        let _ = alloc::<F16>;
    }
}
