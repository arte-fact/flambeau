//! DtoD memcpy of `[n_tokens, kv_width]` K + V into the per-layer
//! cache at row `write_pos`. Takes `&D + &D::Stream` (not an `Ops`)
//! because it's a memcpy, not a kernel launch.

use anyhow::bail;
use flambeau_core::{CopyDirection, Device};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// Placement of an append into the per-layer cache.
#[derive(Copy, Clone, Debug)]
pub struct KvAppendSpec {
    pub n_tokens: usize,
    pub kv_width: usize,
    pub write_pos: usize,
    pub max_seq_len: usize,
    /// Ring-buffer slab depth in rows. `0` = absolute addressing (the
    /// strict `write_pos + n_tokens <= max_seq_len` bound applies). When
    /// non-zero the slab is window-sized and rows wrap at `ring_depth`
    /// (which equals the slab's row capacity `max_seq_len`); a write that
    /// straddles the boundary is split into two contiguous segments.
    pub ring_depth: usize,
}

/// Physical row layout for a ring-buffer KV append. Returns up to two
/// `(phys_row, src_row, n_rows)` segments; the second is the wrap tail and
/// carries `n_rows == 0` when the write does not straddle the boundary.
/// `ring_depth == 0` yields a single absolute segment at `write_pos`.
pub(crate) fn ring_append_segments(
    write_pos: usize,
    n_tokens: usize,
    ring_depth: usize,
) -> [(usize, usize, usize); 2] {
    if ring_depth == 0 {
        return [(write_pos, 0, n_tokens), (0, 0, 0)];
    }
    let phys_start = write_pos % ring_depth;
    let first = core::cmp::min(ring_depth - phys_start, n_tokens);
    [(phys_start, 0, first), (0, first, n_tokens - first)]
}

/// Distinct physical KV rows a ring-buffer attention read can touch:
/// `min(n_tokens, ring_depth)` when ring-addressed (rows wrap at
/// `ring_depth`), else `n_tokens`. The attention read ops size their
/// cache-tensor bounds check with this so a window-depth slab (which holds
/// fewer rows than the logical `n_k_tokens`) is not rejected.
pub(crate) fn ring_cache_rows(n_tokens: usize, ring_depth: usize) -> usize {
    if ring_depth > 0 {
        n_tokens.min(ring_depth)
    } else {
        n_tokens
    }
}

/// Two stream-ordered DtoD memcpys (K and V) into per-layer caches.
pub fn kv_append_f16<D: Device>(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_cache: &mut Tensor<F16>,
    v_cache: &mut Tensor<F16>,
    spec: KvAppendSpec,
    device: &D,
    stream: &D::Stream,
) -> Result<()> {
    let KvAppendSpec { n_tokens, kv_width, write_pos, max_seq_len, ring_depth } = spec;
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
    if ring_depth == 0 {
        if write_pos + n_tokens > max_seq_len {
            bail!(
                "kv_append_f16: write_pos {write_pos} + n_tokens {n_tokens} > max_seq_len {max_seq_len}"
            );
        }
    } else if n_tokens > ring_depth {
        bail!("kv_append_f16: n_tokens {n_tokens} > ring_depth {ring_depth}");
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
    let row_bytes = kv_width * 2;
    for (phys_row, src_row, rows) in ring_append_segments(write_pos, n_tokens, ring_depth) {
        if rows == 0 {
            continue;
        }
        let bytes = rows * row_bytes;
        let k_dst = k_cache.ptr.offset_bytes(phys_row * row_bytes);
        let v_dst = v_cache.ptr.offset_bytes(phys_row * row_bytes);
        let k_src_seg = k_src.ptr.offset_bytes(src_row * row_bytes);
        let v_src_seg = v_src.ptr.offset_bytes(src_row * row_bytes);
        // SAFETY: bounds checks above guarantee `bytes` of valid storage at
        // both src and dst on the same device; stream is live. Ring segments
        // partition `[0, n_tokens)` so writes never overlap.
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                k_dst,
                k_src_seg,
                bytes,
            )?;
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToDevice,
                v_dst,
                v_src_seg,
                bytes,
            )?;
        }
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
            KvAppendSpec {
                n_tokens: N_TOKENS,
                kv_width: KV_WIDTH,
                write_pos: WRITE_POS,
                max_seq_len: MAX_SEQ_LEN,
                ring_depth: 0,
            },
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

    #[test]
    fn ring_append_segments_absolute_is_single_segment() {
        // ring_depth == 0 → one segment at write_pos, no wrap.
        let segs = ring_append_segments(3, 2, 0);
        assert_eq!(segs, [(3, 0, 2), (0, 0, 0)]);
    }

    #[test]
    fn ring_append_segments_no_wrap_below_boundary() {
        // Fits before the boundary → identical to absolute.
        let segs = ring_append_segments(2, 3, 8);
        assert_eq!(segs, [(2, 0, 3), (0, 3, 0)]);
    }

    #[test]
    fn ring_append_segments_straddles_boundary() {
        // depth 8, write at logical 6, 4 rows → [6,7] then wrap [0,1].
        let segs = ring_append_segments(6, 4, 8);
        assert_eq!(segs, [(6, 0, 2), (0, 2, 2)]);
    }

    #[test]
    fn ring_append_segments_wrapped_start_no_straddle() {
        // Logical pos past one revolution; phys start = 10 % 8 = 2.
        let segs = ring_append_segments(10, 3, 8);
        assert_eq!(segs, [(2, 0, 3), (0, 3, 0)]);
    }

    #[test]
    fn kv_append_f16_ring_wrap_writes_split_rows() {
        // Ring slab of DEPTH rows; a 4-row write at logical pos DEPTH-2
        // must land rows [DEPTH-2, DEPTH-1] then wrap to [0, 1].
        const KV_WIDTH: usize = 16;
        const DEPTH: usize = 6;
        const N_TOKENS: usize = 4;
        const WRITE_POS: usize = 10; // 10 % 6 = 4 → straddles (4,5,0,1)

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();

        let k_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(0.1 + (i as f32) * 0.01))
            .collect();
        let v_src_host: Vec<f16> = (0..N_TOKENS * KV_WIDTH)
            .map(|i| f16::from_f32(-0.2 + (i as f32) * 0.013))
            .collect();
        let cache_init: Vec<f16> = vec![f16::from_f32(-9.0); DEPTH * KV_WIDTH];

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
            KvAppendSpec {
                n_tokens: N_TOKENS,
                kv_width: KV_WIDTH,
                write_pos: WRITE_POS,
                max_seq_len: DEPTH,
                ring_depth: DEPTH,
            },
            &device,
            stream,
        )
        .expect("kv_append_f16 ring");
        stream.synchronize().expect("stream sync");

        let k_got: Vec<f16> = download::<F16, f16>(&device, &k_cache_t);
        let v_got: Vec<f16> = download::<F16, f16>(&device, &v_cache_t);

        // Logical token t writes physical row (WRITE_POS + t) % DEPTH.
        for tok in 0..N_TOKENS {
            let phys = (WRITE_POS + tok) % DEPTH;
            for col in 0..KV_WIDTH {
                let cache_idx = phys * KV_WIDTH + col;
                let src_idx = tok * KV_WIDTH + col;
                assert_eq!(
                    k_got[cache_idx], k_src_host[src_idx],
                    "k_cache phys row {phys} (tok {tok}) col {col}"
                );
                assert_eq!(v_got[cache_idx], v_src_host[src_idx]);
            }
        }
        // Physical rows not in {4,5,0,1} → still 2 and 3 → untouched.
        for phys in [2usize, 3] {
            for col in 0..KV_WIDTH {
                let idx = phys * KV_WIDTH + col;
                assert_eq!(k_got[idx].to_f32(), -9.0, "k_cache phys row {phys} overwritten");
                assert_eq!(v_got[idx].to_f32(), -9.0);
            }
        }

        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_ptr, k_cache_t.bytes());
        free(&device, v_cache_ptr, v_cache_t.bytes());
    }
}
