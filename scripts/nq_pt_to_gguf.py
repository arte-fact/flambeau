#!/usr/bin/env python3
"""Convert a SamsungLabs/NanoQuant checkpoint (.pt) into Flambeau GGUF files.

Outputs:
  <out>-NQ2.gguf            linear weights as NQ2 (GGUF wire type id 40) blobs,
                            everything else copied verbatim from the source GGUF
  <out>-NQ2-dequant-f16.gguf  same file but every NQ2 weight dequantized to F16
                            (for llama-perplexity / flambeau cross-checks)

NQ2 blob layout per linear layer (little-endian), matching
crates/quant/src/gguf.rs::TensorInfo::size_in_bytes and the mmvq_nq_bin kernels:
  [ U_packed  dout x wpr u32 ][ V_packed  din x wpr u32 ][ s1 dout f16 ][ s2 din f16 ]
  wpr = ceil(r/32), r = nq2_rank(dout, din) (see below).
  Bit convention = repo's pack_binary_gemv (lsb_first): bit set = -1, clear = +1.
  U is stored (dout, wpr) exactly as in the .pt; V is stored TRANSPOSED from the
  .pt layout (rank x din/32) into (din, wpr) so each input row's rank bits are
  contiguous — that is the layout the stage-1 kernel indexes.

Source: SamsungLabs/NanoQuant @ a9e0a430 (Apache-2.0), checkpoint keys:
  <hf_name>.U_packed (dout, wpr) int32 | .V_packed (r, din/32) int32
  <hf_name>.U_shape (dout, r) | .V_shape (r, din)
  <hf_name>.scale_post (1, dout) bf16 = s1 | .scale_pre (1, din) bf16 = s2
"""

import argparse
import os
import re
import struct

import numpy as np
import torch

NQ2_TYPE_ID = 40
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
# ggml type id -> type size in bytes (only what a F16/F32 GGUF contains)
TYPE_SIZE = {0: 4, 1: 2, 30: 2}  # F32, F16, BF16
VT_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

ATTN_MAP = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "output"}
MLP_MAP = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def nq2_rank(dout: int, din: int) -> int:
    """Mirror of crates/quant/src/gguf.rs::nq2_rank — SamsungLabs/NanoQuant
    utils/utils.py::calculate_ranks for bits=2.0, num_scales=2:
    r0 = a*b*2/(a+b) - 16; int(); floor to 32; clamp to [32, min(a,b)]."""
    a, b = max(dout, din), min(dout, din)
    r0 = a * b * 2 / (a + b) - 16.0
    r = int(r0) if r0 > 0 else 0
    r = (r // 32) * 32
    if r == 0:
        r = 32
    return min(r, b)


def hf_to_gguf(hf_base: str) -> str:
    m = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)", hf_base)
    assert m, hf_base
    i, grp, proj = int(m.group(1)), m.group(2), m.group(3)
    if grp == "self_attn":
        return f"blk.{i}.attn_{ATTN_MAP[proj]}.weight"
    return f"blk.{i}.ffn_{MLP_MAP[proj]}.weight"


# ---------------- GGUF header parsing (read-only) ----------------

def _rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode()


def _skip_val(f, t):
    if t == 8:
        _rd_str(f)
        return
    if t == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        if et == 8:
            for _ in range(cnt):
                _rd_str(f)
        else:
            f.read(VT_FIXED[et] * cnt)
        return
    f.read(VT_FIXED[t])


def parse_gguf_header(path):
    f = open(path, "rb")
    magic, ver = struct.unpack("<II", f.read(8))
    assert magic == GGUF_MAGIC and ver == GGUF_VERSION, (hex(magic), ver)
    ntok, nkv = struct.unpack("<QQ", f.read(16))
    kv_start = f.tell()
    align = 32
    for _ in range(nkv):
        k = _rd_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        if k == "general.alignment":
            (align,) = struct.unpack("<I", f.read(4))
        else:
            _skip_val(f, t)
    kv_end = f.tell()
    kv_bytes = open(path, "rb").read()[kv_start:kv_end]
    tensors = []
    for _ in range(ntok):
        name = _rd_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        tt, off = struct.unpack("<IQ", f.read(12))
        tensors.append({"name": name, "ne": dims, "type": tt, "off": off})
    infos_end = f.tell()
    f.close()
    data_start = (infos_end + align - 1) // align * align
    return kv_bytes, nkv, tensors, align, data_start


def gguf_payload_size(ne, tt):
    n = 1
    for d in ne:
        n *= d
    assert tt in TYPE_SIZE, f"unsupported source type {tt}"
    return n * TYPE_SIZE[tt]


# ---------------- bit packing ----------------

def unpack_bits(packed_u32: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(rows, wpr) u32 lsb_first -> (rows, cols) uint8 bits (padding dropped)."""
    wpr = (cols + 31) // 32
    p = packed_u32.reshape(rows, wpr, 1).astype(np.uint32)
    shifts = np.arange(32, dtype=np.uint32)
    bits = ((p >> shifts) & np.uint32(1)).astype(np.uint8)
    return bits.reshape(rows, wpr * 32)[:, :cols]


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """(rows, cols) uint8 bits -> (rows, ceil(cols/32)) u32 lsb_first, zero pad."""
    rows, cols = bits.shape
    wpr = (cols + 31) // 32
    pad = np.zeros((rows, wpr * 32 - cols), dtype=np.uint8)
    b = np.concatenate([bits, pad], axis=1).reshape(rows, wpr, 32)
    shifts = np.arange(32, dtype=np.uint32)
    return (b.astype(np.uint32) << shifts).sum(axis=2, dtype=np.uint32)


# ---------------- NQ2 blob ----------------

def build_nq2(sd, hf_base):
    up = sd[hf_base + ".U_packed"].detach().cpu().numpy().astype(np.uint32)  # (dout, wpr)
    vp = sd[hf_base + ".V_packed"].detach().cpu().numpy().astype(np.uint32)  # (r, din/32)
    ushape = sd[hf_base + ".U_shape"].detach().cpu().numpy()
    vshape = sd[hf_base + ".V_shape"].detach().cpu().numpy()
    dout, r = int(ushape[0]), int(ushape[1])
    r2, din = int(vshape[0]), int(vshape[1])
    assert r == r2, (hf_base, r, r2)
    assert nq2_rank(dout, din) == r, (hf_base, r, nq2_rank(dout, din))
    wpr = (r + 31) // 32
    assert up.shape == (dout, wpr), (hf_base, up.shape, dout, wpr)
    # V: (r, din) bits -> transpose (din, r) -> pack -> (din, wpr)
    v_bits = unpack_bits(vp, r, din)  # (r, din)
    v_rep = pack_bits(np.ascontiguousarray(v_bits.T))  # (din, wpr)
    # round-trip check: unpacking the repacked V must give the same bits
    rt = unpack_bits(v_rep, din, r)
    assert np.array_equal(rt, np.ascontiguousarray(v_bits.T)), hf_base
    s1 = sd[hf_base + ".scale_post"].detach().cpu().float().half().numpy().reshape(-1)  # (dout,) f16
    s2 = sd[hf_base + ".scale_pre"].detach().cpu().float().half().numpy().reshape(-1)  # (din,) f16
    assert s1.shape == (dout,) and s2.shape == (din,), hf_base
    blob = up.tobytes() + v_rep.tobytes() + s1.tobytes() + s2.tobytes()
    expected = (dout + din) * wpr * 4 + (dout + din) * 2
    assert len(blob) == expected, (hf_base, len(blob), expected)
    # dequantized weight for the cross-check GGUF: W = diag(s1) U V^T diag(s2)
    u_sign = (1 - 2 * unpack_bits(up, dout, r).astype(np.int32)).astype(np.float32)  # (dout, r)
    v_sign = (1 - 2 * v_bits.astype(np.int32)).astype(np.float32)  # (r, din)
    w = (
        s1.astype(np.float32)[:, None]
        * (u_sign @ v_sign)
        * s2.astype(np.float32)[None, :]
    )
    return blob, w.astype(np.float16), (dout, din, r, wpr)


# ---------------- GGUF writer ----------------

def write_gguf(out_path, kv_bytes, nkv, entries, align):
    """entries: list of (name, ne, type_id, payload_bytes) in file order."""
    n = len(entries)
    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, n, nkv) + kv_bytes
    # layout data with per-tensor alignment; offsets relative to data section start
    offsets = []
    cur = 0
    for (_, _, _, payload) in entries:
        offsets.append(cur)
        cur += len(payload)
        cur = (cur + align - 1) // align * align
    info_blob = b""
    for (name, ne, tt, _), off in zip(entries, offsets):
        nb = name.encode()
        info_blob += struct.pack("<Q", len(nb)) + nb
        info_blob += struct.pack("<I", len(ne))
        info_blob += struct.pack("<" + "Q" * len(ne), *ne)
        info_blob += struct.pack("<IQ", tt, off)
    head_len = len(header) + len(info_blob)
    pad = (align - head_len % align) % align
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(info_blob)
        f.write(b"\0" * pad)
        for (name, _, _, payload), off in zip(entries, offsets):
            cur = f.tell()
            data_start = head_len + pad
            assert cur - data_start == off, (name, cur, data_start, off)
            f.write(payload)
            nxt = data_start + off + len(payload)
            nxt = (nxt + align - 1) // align * align
            f.write(b"\0" * (nxt - cur - len(payload)))
    return os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-gguf", required=True, help="source F16 GGUF (KVs + non-linear tensors copied from it)")
    ap.add_argument("--ckpt", required=True, help="NanoQuant .pt checkpoint")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>-NQ2.gguf and <out>-NQ2-dequant-f16.gguf")
    args = ap.parse_args()

    kv_bytes, nkv, tensors, align, data_start = parse_gguf_header(args.src_gguf)
    print(f"src: {len(tensors)} tensors, {nkv} kvs, align={align}, data_start={data_start}")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    q_bases = sorted({k[: -len(".U_packed")] for k in sd if k.endswith(".U_packed")})
    print(f"ckpt: {len(q_bases)} quantized linears")
    q_map = {hf_to_gguf(b): b for b in q_bases}

    src = open(args.src_gguf, "rb")
    entries_nq2 = []
    entries_deq = []
    src_payload = {}
    ranks = []
    for t in tensors:
        name, ne, tt = t["name"], t["ne"], t["type"]
        size = gguf_payload_size(ne, tt)
        src.seek(data_start + t["off"])
        raw = src.read(size)
        assert len(raw) == size
        if name in q_map:
            blob, w_f16, (dout, din, r, wpr) = build_nq2(sd, q_map[name])
            assert len(ne) == 2 and ne[0] == din and ne[1] == dout, (name, ne, din, dout)
            entries_nq2.append((name, ne, NQ2_TYPE_ID, blob))
            entries_deq.append((name, ne, 1, w_f16.tobytes()))  # F16
            ranks.append((name, dout, din, r, wpr))
        else:
            entries_nq2.append((name, ne, tt, raw))
            entries_deq.append((name, ne, tt, raw))
        src_payload[name] = raw
    src.close()

    for name, dout, din, r, wpr in ranks[:8]:
        print(f"  {name}: dout={dout} din={din} r={r} wpr={wpr}")
    n_nq = sum(1 for e in entries_nq2 if e[2] == NQ2_TYPE_ID)
    print(f"NQ2 tensors: {n_nq}")

    # sanity: dequantized W vs source F16 weights on one layer
    name0 = ranks[0][0]
    w_ref = np.frombuffer(src_payload[name0], dtype=np.float16).reshape(ranks[0][1], ranks[0][2]).astype(np.float32)
    _, w_f16_0, _ = build_nq2(sd, q_map[name0])
    w_hat = w_f16_0.astype(np.float32)
    denom = np.maximum(np.abs(w_ref), 1e-3)
    print(f"sanity {name0}: max|W_ref|={np.abs(w_ref).max():.4f} max|W_hat-W_ref|/|W_ref|={np.max(np.abs(w_hat-w_ref)/denom):.3f} (2bpw, expected O(0.3-1))")

    s1 = write_gguf(args.out + "-NQ2.gguf", kv_bytes, nkv, entries_nq2, align)
    s2 = write_gguf(args.out + "-NQ2-dequant-f16.gguf", kv_bytes, nkv, entries_deq, align)
    print(f"wrote {args.out}-NQ2.gguf ({s1/1e6:.1f} MB) and {args.out}-NQ2-dequant-f16.gguf ({s2/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
