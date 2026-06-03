//! F32 → Q2_K / Q3_K / Q4_K host quantizers.
//!
//! Ports of llama.cpp `quantize_row_q{2,3,4}_K_ref` from `ggml-quants.c`.
//! The arithmetic mirrors the upstream reference exactly so round-trip
//! through `dequantize_into` reproduces llama.cpp's quant output on the
//! same input. Used by the loader's at-load convert path
//! (`upload_via_dequant_to`) to target a K-quant size instead of Q8_0
//! when the source dtype is itself a 2–4 bpw family (IQ2_*, IQ3_*,
//! IQ4_*). Targeting a same-bpw destination keeps post-convert tensor
//! size close to source size — Q8_0 would blow up IQ3 by ~2.7×.
//!
//! Performance is not a concern here — these run once per tensor at
//! model-load time. Each fn writes the canonical GGUF byte layout (the
//! flambeau_block_q{2,3,4}_K struct layout) into the output buffer.
//!
//! NOTE: these are the *reference* (scalar, deterministic) quantizers.
//! No AVX2 / Neon / GPU paths. The make_qkx2_quants iterative search
//! (Q2_K / Q4_K) and the make_q3_quants RMSE-fixpoint (Q3_K) match
//! their C counterparts step-for-step; bit-identical output on
//! IEEE754-friendly hardware.

use half::f16;
use rayon::prelude::*;

use crate::dtype::QK_K;

const Q4_K_BLOCK_BYTES: usize = 144;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q2_K_BLOCK_BYTES: usize = 84;

/// Round to nearest int, ties to even (matches C's `lrintf` default).
fn nearest_int(x: f32) -> i32 {
    // round_ties_even is `f32` → `f32` in Rust; combined with `as i32`
    // gives the same truncation behaviour as C's `lrintf` under
    // round-to-nearest-even FE rounding mode.
    x.round_ties_even() as i32
}

const GROUP_MAX_EPS: f32 = 1e-15;

/// Borrowed slice quartet driving [`make_qkx2_quants`].
struct QkxBuffers<'a> {
    x: &'a [f32],
    weights: &'a [f32],
    out_l: &'a mut [u8],
    aux_l: &'a mut [u8],
}

/// Iterative-refinement knobs for [`make_qkx2_quants`].
#[derive(Copy, Clone, Debug)]
struct QkxKnobs {
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
}

/// `make_qkx2_quants` — joint scale + min fit for an affine quant
/// (`y = scale*l + min` with `l ∈ [0, nmax]`). Initial guess from
/// max/min, then `nstep` refinements that solve the weighted least-
/// squares (scale, min) for the current quantised values. Returns the
/// best scale found; writes the chosen `l` codes into `out_l` and the
/// negated min into `*the_min`. Used by Q2_K (nmax=3) and Q4_K (nmax=15).
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    bufs: QkxBuffers<'_>,
    the_min: &mut f32,
    knobs: QkxKnobs,
) -> f32 {
    let QkxBuffers { x, weights, out_l, aux_l } = bufs;
    let QkxKnobs { rmin, rdelta, nstep, use_mad } = knobs;
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for slot in out_l.iter_mut().take(n) {
            *slot = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        out_l[i] = l.max(0).min(nmax) as u8;
        let diff = scale * out_l[i] as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        let iscale_try = (rmin + rdelta * (is as f32) + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale_try * (x[i] - min));
            let l = l.max(0).min(nmax) as u8;
            aux_l[i] = l;
            let w = weights[i];
            sum_l += w * (l as f32);
            sum_l2 += w * (l as f32) * (l as f32);
            sum_xl += w * (l as f32) * x[i];
        }
        let d_det = sum_w * sum_l2 - sum_l * sum_l;
        if d_det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * (aux_l[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                out_l[..n].copy_from_slice(&aux_l[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
            let _ = iscale; // silence warning; the C code mirrors this
        }
        iscale = iscale_try;
    }
    *the_min = -min;
    scale
}

/// `make_q3_quants` — pure-scale signed quantiser with RMSE fixpoint.
/// Returns the best scale; writes per-element codes into `out_l` as
/// `l + nmax` (so the caller sees an unsigned 0..2*nmax index).
fn make_q3_quants(n: usize, nmax: i32, x: &[f32], out_l: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in &x[..n] {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        for slot in out_l.iter_mut().take(n) {
            *slot = 0;
        }
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
            out_l[i] = l as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * (l as f32);
            suml2 += w * (l as f32) * (l as f32);
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let slx_base = sumlx - w * x[i] * (out_l[i] as f32);
                if slx_base > 0.0 {
                    let sl2_base = suml2 - w * (out_l[i] as f32) * (out_l[i] as f32);
                    let new_l = nearest_int(x[i] * sl2_base / slx_base)
                        .max(-nmax)
                        .min(nmax - 1);
                    if new_l != out_l[i] as i32 {
                        let slx = slx_base + w * x[i] * (new_l as f32);
                        let sl2 = sl2_base + w * (new_l as f32) * (new_l as f32);
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            out_l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for slot in out_l.iter_mut().take(n) {
            *slot = (*slot as i32 + nmax) as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).max(-nmax).min(nmax - 1);
        out_l[i] = (l + nmax) as i8;
    }
    1.0 / iscale
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Pack a single Q4_K super-block (256 F32 inputs → 144 output bytes).
/// Per-super-block work is independent, so this is what the parallel
/// outer loop in `quantize_row_q4_k` schedules across rayon threads.
fn pack_block_q4_k(xi: &[f32], out: &mut [u8]) {
    debug_assert_eq!(xi.len(), QK_K);
    debug_assert_eq!(out.len(), Q4_K_BLOCK_BYTES);
    let mut local_l = [0u8; QK_K];
    let mut aux_l = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];
    let mut block_scales = [0u8; 12];

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..QK_K / 32 {
        let xj = &xi[32 * j..32 * (j + 1)];
        let mut sum_x2 = 0.0f32;
        for &v in xj {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for l in 0..32 {
            weights[l] = av_x + xj[l].abs();
        }
        let mut the_min = 0.0f32;
        scales[j] = make_qkx2_quants(
            32,
            15,
            QkxBuffers {
                x: xj,
                weights: &weights,
                out_l: &mut local_l[32 * j..32 * (j + 1)],
                aux_l: &mut aux_l,
            },
            &mut the_min,
            QkxKnobs { rmin: -1.0, rdelta: 0.1, nstep: 20, use_mad: false },
        );
        mins[j] = the_min;
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    for j in 0..QK_K / 32 {
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            block_scales[j] = ls;
            block_scales[j + 4] = lm;
        } else {
            block_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            block_scales[j - 4] |= (ls >> 4) << 6;
            block_scales[j] |= (lm >> 4) << 6;
        }
    }
    let d_f16 = f16::from_f32(max_scale / 63.0);
    let dmin_f16 = f16::from_f32(max_min / 63.0);

    for j in 0..QK_K / 32 {
        let (sc, m) = get_scale_min_k4(j, &block_scales);
        let d = d_f16.to_f32() * sc as f32;
        if d == 0.0 {
            continue;
        }
        let dm = dmin_f16.to_f32() * m as f32;
        for ii in 0..32 {
            let l = nearest_int((xi[32 * j + ii] + dm) / d).clamp(0, 15);
            local_l[32 * j + ii] = l as u8;
        }
    }

    out[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    out[2..4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());
    out[4..16].copy_from_slice(&block_scales);
    let qs = &mut out[16..144];
    let mut qi = 0;
    for j_off in (0..QK_K).step_by(64) {
        for l in 0..32 {
            qs[qi] = local_l[j_off + l] | (local_l[j_off + l + 32] << 4);
            qi += 1;
        }
    }
}

/// F32 → Q4_K (144 B/block, 256 elems). Mirrors `quantize_row_q4_K_ref`.
/// Output bytes match the on-disk layout: `d (f16) + dmin (f16) +
/// scales[12] + qs[128]`. Block-level parallel via rayon; the per-block
/// state is purely local so threads never contend.
pub fn quantize_row_q4_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let start = out.len();
    out.resize(start + nb * Q4_K_BLOCK_BYTES, 0);
    out[start..]
        .par_chunks_exact_mut(Q4_K_BLOCK_BYTES)
        .zip(x.par_chunks_exact(QK_K))
        .for_each(|(block_out, xi)| pack_block_q4_k(xi, block_out));
}

/// Pack a single Q3_K super-block (256 F32 inputs → 110 output bytes).
fn pack_block_q3_k(xi: &[f32], out: &mut [u8]) {
    debug_assert_eq!(xi.len(), QK_K);
    debug_assert_eq!(out.len(), Q3_K_BLOCK_BYTES);
    let mut local_l = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    let mut block_scales = [0u8; 12];
    let mut hmask = [0u8; QK_K / 8];
    let mut qs = [0u8; QK_K / 4];

    let mut max_scale = 0.0f32;
    let mut amax = 0.0f32;
    for j in 0..QK_K / 16 {
        scales[j] = make_q3_quants(
            16,
            4,
            &xi[16 * j..16 * (j + 1)],
            &mut local_l[16 * j..16 * (j + 1)],
            true,
        );
        let sa = scales[j].abs();
        if sa > amax {
            amax = sa;
            max_scale = scales[j];
        }
    }

    let d_f16 = if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for j in 0..QK_K / 16 {
            let l = nearest_int(iscale * scales[j]).clamp(-32, 31) + 32;
            let l = l as u8;
            if j < 8 {
                block_scales[j] = l & 0xF;
            } else {
                block_scales[j - 8] |= (l & 0xF) << 4;
            }
            let lh = l >> 4;
            block_scales[(j % 4) + 8] |= lh << (2 * (j / 4));
        }
        f16::from_f32(1.0 / iscale)
    } else {
        f16::from_f32(0.0)
    };

    for j in 0..QK_K / 16 {
        let sc_low = if j < 8 {
            block_scales[j] & 0xF
        } else {
            block_scales[j - 8] >> 4
        };
        let sc = sc_low | ((block_scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4;
        let sc = (sc as i32) - 32;
        let d = d_f16.to_f32() * sc as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let l = nearest_int(xi[16 * j + ii] / d).clamp(-4, 3);
            local_l[16 * j + ii] = (l + 4) as i8;
        }
    }

    let mut m_idx = 0usize;
    let mut hm: u8 = 1;
    for slot in local_l.iter_mut().take(QK_K) {
        if *slot > 3 {
            hmask[m_idx] |= hm;
            *slot -= 4;
        }
        m_idx += 1;
        if m_idx == QK_K / 8 {
            m_idx = 0;
            hm <<= 1;
        }
    }
    for j_off in (0..QK_K).step_by(128) {
        for l in 0..32 {
            qs[j_off / 4 + l] = (local_l[j_off + l] as u8)
                | ((local_l[j_off + l + 32] as u8) << 2)
                | ((local_l[j_off + l + 64] as u8) << 4)
                | ((local_l[j_off + l + 96] as u8) << 6);
        }
    }

    out[0..32].copy_from_slice(&hmask);
    out[32..96].copy_from_slice(&qs);
    out[96..108].copy_from_slice(&block_scales);
    out[108..110].copy_from_slice(&d_f16.to_bits().to_le_bytes());
}

/// F32 → Q3_K (110 B/block, 256 elems). Mirrors `quantize_row_q3_K_ref`.
/// Output bytes: `hmask[32] + qs[64] + scales[12] + d (f16)`. Block-level
/// parallel via rayon.
pub fn quantize_row_q3_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let start = out.len();
    out.resize(start + nb * Q3_K_BLOCK_BYTES, 0);
    out[start..]
        .par_chunks_exact_mut(Q3_K_BLOCK_BYTES)
        .zip(x.par_chunks_exact(QK_K))
        .for_each(|(block_out, xi)| pack_block_q3_k(xi, block_out));
}

/// Pack a single Q2_K super-block (256 F32 inputs → 84 output bytes).
fn pack_block_q2_k(xi: &[f32], out: &mut [u8]) {
    debug_assert_eq!(xi.len(), QK_K);
    debug_assert_eq!(out.len(), Q2_K_BLOCK_BYTES);
    const Q4SCALE: f32 = 15.0;
    let mut local_l = [0u8; QK_K];
    let mut aux_l = [0u8; 16];
    let mut weights = [0.0f32; 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut scales = [0.0f32; QK_K / 16];
    let mut block_scales = [0u8; QK_K / 16];

    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..QK_K / 16 {
        let xj = &xi[16 * j..16 * (j + 1)];
        for l in 0..16 {
            weights[l] = xj[l].abs();
        }
        let mut the_min = 0.0f32;
        scales[j] = make_qkx2_quants(
            16,
            3,
            QkxBuffers {
                x: xj,
                weights: &weights,
                out_l: &mut local_l[16 * j..16 * (j + 1)],
                aux_l: &mut aux_l,
            },
            &mut the_min,
            QkxKnobs { rmin: -0.5, rdelta: 0.1, nstep: 15, use_mad: true },
        );
        mins[j] = the_min;
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }

    let d_f16 = if max_scale > 0.0 {
        let iscale = Q4SCALE / max_scale;
        for j in 0..QK_K / 16 {
            block_scales[j] = nearest_int(iscale * scales[j]) as u8;
        }
        f16::from_f32(max_scale / Q4SCALE)
    } else {
        f16::from_f32(0.0)
    };
    let dmin_f16 = if max_min > 0.0 {
        let iscale = Q4SCALE / max_min;
        for j in 0..QK_K / 16 {
            let m = nearest_int(iscale * mins[j]) as u8;
            block_scales[j] |= m << 4;
        }
        f16::from_f32(max_min / Q4SCALE)
    } else {
        f16::from_f32(0.0)
    };

    for j in 0..QK_K / 16 {
        let d = d_f16.to_f32() * (block_scales[j] & 0xF) as f32;
        if d == 0.0 {
            continue;
        }
        let dm = dmin_f16.to_f32() * (block_scales[j] >> 4) as f32;
        for ii in 0..16 {
            let l = nearest_int((xi[16 * j + ii] + dm) / d).clamp(0, 3);
            local_l[16 * j + ii] = l as u8;
        }
    }

    let mut qs = [0u8; QK_K / 4];
    for j_off in (0..QK_K).step_by(128) {
        for l in 0..32 {
            qs[j_off / 4 + l] = local_l[j_off + l]
                | (local_l[j_off + l + 32] << 2)
                | (local_l[j_off + l + 64] << 4)
                | (local_l[j_off + l + 96] << 6);
        }
    }

    out[0..16].copy_from_slice(&block_scales);
    out[16..80].copy_from_slice(&qs);
    out[80..82].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    out[82..84].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());
}

/// F32 → Q2_K (84 B/block, 256 elems). Mirrors `quantize_row_q2_K_ref`.
/// Output bytes: `scales[16] + qs[64] + d (f16) + dmin (f16)`. Block-level
/// parallel via rayon.
pub fn quantize_row_q2_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let start = out.len();
    out.resize(start + nb * Q2_K_BLOCK_BYTES, 0);
    out[start..]
        .par_chunks_exact_mut(Q2_K_BLOCK_BYTES)
        .zip(x.par_chunks_exact(QK_K))
        .for_each(|(block_out, xi)| pack_block_q2_k(xi, block_out));
}

/// F32 → Q8_0 (34 B/block, 32 elems). Pure linear-absmax scalar quant —
/// no iterative search, so the win from parallelisation is smaller than
/// for the K-quant family. Block-level parallel via rayon. Shared by the
/// dense and TP loaders (the inline Q8_0 encoders previously duplicated
/// in both `upload_via_dequant_to` paths).
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    use crate::dtype::QK8_0;
    assert!(x.len() % QK8_0 == 0);
    let nb = x.len() / QK8_0;
    let start = out.len();
    out.resize(start + nb * 34, 0);
    out[start..]
        .par_chunks_exact_mut(34)
        .zip(x.par_chunks_exact(QK8_0))
        .for_each(|(block_out, block_in)| {
            let absmax = block_in.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = absmax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d_f16 = f16::from_f32(d);
            block_out[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
            for (i, &v) in block_in.iter().enumerate() {
                let q = (v * id).round_ties_even() as i32;
                let q = q.clamp(-127, 127) as i8;
                block_out[2 + i] = q as u8;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_f32_buf(n_blocks: usize) -> Vec<f32> {
        let n = n_blocks * QK_K;
        let mut v = Vec::with_capacity(n);
        // Mix of large positive, large negative, zero, tiny — exercises the
        // full quant range and the zero-d shortcut.
        for i in 0..n {
            let phase = (i as f32) * 0.017;
            let amp = if i % 73 == 0 {
                0.0
            } else {
                1.5 + 0.5 * (i as f32 / 1024.0).sin()
            };
            v.push(amp * phase.sin() - 0.3 * phase.cos());
        }
        v
    }

    fn assert_quantize_deterministic<F: Fn(&[f32], &mut Vec<u8>)>(f: F, n_blocks: usize) {
        let x = deterministic_f32_buf(n_blocks);
        let mut a = Vec::new();
        let mut b = Vec::new();
        f(&x, &mut a);
        f(&x, &mut b);
        assert_eq!(
            a, b,
            "rayon-parallel quantize should be deterministic across runs"
        );
        // Output prefix-appends: a second call appends to the same Vec.
        let mut c = b.clone();
        f(&x, &mut c);
        assert_eq!(
            &c[a.len()..],
            &a[..],
            "second call must produce identical bytes to the first call"
        );
    }

    #[test]
    fn q4_k_parallel_is_deterministic() {
        assert_quantize_deterministic(quantize_row_q4_k, 64);
    }

    #[test]
    fn q3_k_parallel_is_deterministic() {
        assert_quantize_deterministic(quantize_row_q3_k, 64);
    }

    #[test]
    fn q2_k_parallel_is_deterministic() {
        assert_quantize_deterministic(quantize_row_q2_k, 64);
    }

    #[test]
    fn q8_0_parallel_is_deterministic() {
        use crate::dtype::QK8_0;
        let n = 1024 * QK8_0;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 1.7).collect();
        let mut a = Vec::new();
        let mut b = Vec::new();
        quantize_row_q8_0(&x, &mut a);
        quantize_row_q8_0(&x, &mut b);
        assert_eq!(a, b);
        assert_eq!(a.len(), (n / QK8_0) * 34);
    }
}
