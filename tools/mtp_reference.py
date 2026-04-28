#!/usr/bin/env python3
"""
MTP-3 reference — PyTorch implementation of one Qwen3.6-27B MTP step,
loading weights directly from `Qwen3.6-27B-mtp.gguf`.

Saves test vectors as raw F32 binary files for the Rust integration
test:
    tests/data/mtp_ref/h_t.bin              — base hidden state input  [5120]
    tests/data/mtp_ref/e_token.bin          — next-token embedding      [5120]
    tests/data/mtp_ref/expected_h_final.bin — final pre-LM-head hidden  [5120]
    tests/data/mtp_ref/inputs.json          — config + intermediate norms

Math (matches HF Transformers `Qwen3_5MTP` block):
    norm_h  = RMSNorm(h_t,    pre_fc_norm_hidden,    eps)
    norm_e  = RMSNorm(e_tok,  pre_fc_norm_embedding, eps)
    h0      = concat([norm_h, norm_e]) @ fc.T           # [10240] → [5120]
    h0n     = RMSNorm(h0, input_layernorm)
    [Q | gate] = h0n @ q_proj.T                         # 24*256 + 24*256 (gated)
    K, V    = h0n @ {k,v}_proj.T                        # 4*256 each
    Q       = RMSNorm-per-head(Q, q_norm)
    K       = RMSNorm-per-head(K, k_norm)
    Q, K    = MROPE(Q, K, position)                     # partial 0.25 (first 64 of 256)
    attn    = softmax(Q · K_repeated / sqrt(256)) · V_repeated   # single token, no history
    attn    = attn * silu(gate)                         # output_gate_type="swish"
    h1      = h0 + attn @ o_proj.T
    h1n     = RMSNorm(h1, post_attention_layernorm)
    mlp     = (silu(h1n @ gate_proj.T) * (h1n @ up_proj.T)) @ down_proj.T
    h2      = h1 + mlp
    h_final = RMSNorm(h2, mtp.norm)

Setting position=0 keeps MROPE as identity (cos=1, sin=0) — simplifies
the first round of validation. We exercise the position!=0 path in MTP-4
once basic forward parity is established.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf import GGMLQuantizationType


MTP_GGUF = Path("/artefact/models/Qwen3.6-27B-mtp.gguf")
OUT_DIR = Path("/artefact/flambeau/crates/models/qwen3-moe/tests/data/mtp_ref")

# Config derived from Qwen/Qwen3.6-27B/config.json (text_config).
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
RMS_EPS = 1e-6
ROPE_DIM_TOTAL = HEAD_DIM
ROPE_PARTIAL_FACTOR = 0.25  # → first 64 of head_dim get rotated
ROPE_PARTIAL_DIM = int(HEAD_DIM * ROPE_PARTIAL_FACTOR)  # 64

# Single-MTP-step: position=0 makes MROPE identity, isolating the math.
TEST_POSITION = 0
SEED = 42


def load_mtp_weights() -> dict[str, torch.Tensor]:
    """
    Read 15 mtp.* tensors from the GGUF; dequantize Q8_0 → F32. The
    returned tensors are in PyTorch nn.Linear convention `[out_dim,
    in_dim]` (verified empirically against the safetensors source —
    GGUF stores shape REVERSED in metadata, but `gguf.quants.dequantize`
    and the F32 byte layout produce arrays in the PyTorch convention
    natively when `t.data` is taken as-is).
    """
    print(f"loading: {MTP_GGUF}")
    r = gguf.GGUFReader(str(MTP_GGUF))
    out: dict[str, torch.Tensor] = {}
    for t in r.tensors:
        if not t.name.startswith("mtp."):
            continue
        if t.tensor_type == GGMLQuantizationType.F32:
            # F32 1D norm — single dim, no reshape needed.
            arr = np.array(t.data, dtype=np.float32, copy=True)
        elif t.tensor_type == GGMLQuantizationType.Q8_0:
            # Dequantize Q8_0 — output is already in PyTorch
            # [out_dim, in_dim] order (does NOT need reshape).
            arr = gguf.quants.dequantize(t.data, GGMLQuantizationType.Q8_0).astype(np.float32, copy=False)
        else:
            raise SystemExit(f"unexpected dtype for {t.name}: {t.tensor_type}")
        out[t.name] = torch.from_numpy(arr)
        print(f"  {t.name}: shape={tuple(out[t.name].shape)} dtype={out[t.name].dtype}")
    if len(out) != 15:
        raise SystemExit(f"expected 15 mtp.* tensors, got {len(out)}")
    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight, computed in F32."""
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


def rmsnorm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm — `x` is [n_heads, head_dim], weight is [head_dim]."""
    return rmsnorm(x, weight, eps)


def mrope_partial(
    q: torch.Tensor,  # [n_heads, head_dim]
    k: torch.Tensor,  # [n_kv_heads, head_dim]
    position: int,
    rope_theta: float = 1e7,
    partial_dim: int = ROPE_PARTIAL_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Partial rotary embedding (factor 0.25 → first 64 of 256). At
    position=0 this is identity — useful for the first reference test.
    The rest of the head_dim passes through.
    """
    half = partial_dim // 2  # 32
    if position == 0:
        return q, k  # identity at position 0

    # Frequencies for the first `partial_dim` dimensions
    freq_idx = torch.arange(half, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (freq_idx * 2 / partial_dim))
    angle = inv_freq * position
    cos = torch.cos(angle)  # [half]
    sin = torch.sin(angle)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :partial_dim].clone()
        x_pass = x[..., partial_dim:]
        # Standard RoPE: split first half / second half within partial_dim
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos
        return torch.cat([rotated_1, rotated_2, x_pass], dim=-1)

    return rotate(q), rotate(k)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def forward_mtp_step(
    h_t: torch.Tensor,   # [hidden] base hidden state (post final norm)
    e_tok: torch.Tensor, # [hidden] embedding of token t+1
    position: int,
    w: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Run one MTP step. Returns (h_final, intermediates). h_final is
    [hidden] post mtp.norm; the LM head matmul is the caller's job
    (shared with base, not in the MTP file).
    """
    interm: dict[str, torch.Tensor] = {}

    norm_h = rmsnorm(h_t, w["mtp.pre_fc_norm_hidden.weight"], RMS_EPS)
    norm_e = rmsnorm(e_tok, w["mtp.pre_fc_norm_embedding.weight"], RMS_EPS)
    fc_in = torch.cat([norm_h, norm_e], dim=-1)  # [10240]
    interm["fc_in"] = fc_in.clone()

    # fc.weight is stored as [out=5120, in=10240] (PyTorch nn.Linear convention
    # confirmed by safetensors shape (5120, 10240) earlier).
    h0 = fc_in @ w["mtp.fc.weight"].T
    interm["h0"] = h0.clone()

    # ── transformer block ──
    h0n = rmsnorm(h0, w["mtp.layers.0.input_layernorm.weight"], RMS_EPS)

    # q_proj: [12288, 5120] = (Q heads × 256) + (gate × 256) concatenated
    # along the output dim. Q is first half, gate is second half.
    q_full = h0n @ w["mtp.layers.0.self_attn.q_proj.weight"].T  # [12288]
    q_flat = q_full[:NUM_Q_HEADS * HEAD_DIM]                    # [6144]
    gate_flat = q_full[NUM_Q_HEADS * HEAD_DIM:]                 # [6144]

    k_flat = h0n @ w["mtp.layers.0.self_attn.k_proj.weight"].T  # [1024]
    v_flat = h0n @ w["mtp.layers.0.self_attn.v_proj.weight"].T  # [1024]

    q = q_flat.reshape(NUM_Q_HEADS, HEAD_DIM)
    k = k_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
    v = v_flat.reshape(NUM_KV_HEADS, HEAD_DIM)

    # Per-head RMSNorm on Q and K (Qwen3 convention).
    q = rmsnorm_per_head(q, w["mtp.layers.0.self_attn.q_norm.weight"], RMS_EPS)
    k = rmsnorm_per_head(k, w["mtp.layers.0.self_attn.k_norm.weight"], RMS_EPS)

    # Partial MROPE (identity at position=0).
    q, k = mrope_partial(q, k, position)

    # GQA expansion: each KV head serves N_REP query heads.
    n_rep = NUM_Q_HEADS // NUM_KV_HEADS
    k_full = k.repeat_interleave(n_rep, dim=0)  # [24, 256]
    v_full = v.repeat_interleave(n_rep, dim=0)

    # Single-token attention (no KV history; this is the simplest MTP-step
    # validation case — full prefix-aware attention is exercised in MTP-4).
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_logits = (q.float() * k_full.float()).sum(dim=-1, keepdim=True) * scale  # [24, 1]
    attn_w = torch.softmax(attn_logits, dim=-1)  # trivially [24, 1] = 1.0
    attn = (attn_w * v_full.float()).to(h_t.dtype)  # [24, 256]
    attn = attn.reshape(NUM_Q_HEADS * HEAD_DIM)    # [6144]
    interm["attn_pre_gate"] = attn.clone()

    # Output gate: SIGMOID elementwise multiply (NOT silu). This matches the
    # llama.cpp qwen35moe graph and the V1.7.4.b finding for Qwen3.5/3.6 base
    # full-attn layers — `output_gate_type="swish"` in config is a misnomer;
    # actual op is `sigmoid(gate) * attn`. Empirically validated bit-exact
    # vs llama.cpp for the 35B-A3B base in V1.7.4.b.
    attn = attn * torch.sigmoid(gate_flat)
    interm["attn_post_gate"] = attn.clone()

    # o_proj: [5120, 6144] → projects 6144 down to hidden 5120.
    attn_out = attn @ w["mtp.layers.0.self_attn.o_proj.weight"].T  # [5120]
    h1 = h0 + attn_out
    interm["h1"] = h1.clone()

    # ── MLP ──
    h1n = rmsnorm(h1, w["mtp.layers.0.post_attention_layernorm.weight"], RMS_EPS)
    g = h1n @ w["mtp.layers.0.mlp.gate_proj.weight"].T  # [17408]
    u = h1n @ w["mtp.layers.0.mlp.up_proj.weight"].T    # [17408]
    mlp_act = silu(g) * u
    mlp_out = mlp_act @ w["mtp.layers.0.mlp.down_proj.weight"].T  # [5120]
    h2 = h1 + mlp_out
    interm["h2"] = h2.clone()

    h_final = rmsnorm(h2, w["mtp.norm.weight"], RMS_EPS)
    interm["h_final"] = h_final.clone()
    return h_final, interm


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = load_mtp_weights()

    # Generate a fixed-seed input (F32; flambeau runs F16 hidden through but
    # the Python reference stays F32 for clarity — quantize at compare time).
    torch.manual_seed(SEED)
    h_t = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.02
    e_tok = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.02

    print(f"\nrunning MTP step at position={TEST_POSITION}...")
    h_final, interm = forward_mtp_step(h_t, e_tok, TEST_POSITION, weights)

    print(f"\noutput summary:")
    print(f"  h_final.shape = {tuple(h_final.shape)}")
    print(f"  h_final.mean = {h_final.mean().item():+.6f}")
    print(f"  h_final.std  = {h_final.std().item():+.6f}")
    print(f"  h_final.min  = {h_final.min().item():+.6f}")
    print(f"  h_final.max  = {h_final.max().item():+.6f}")
    print(f"  h_final[:8]  = {h_final[:8].tolist()}")

    # Dump test vectors as raw F32 little-endian for the Rust test.
    def dump(name: str, t: torch.Tensor) -> None:
        path = OUT_DIR / f"{name}.bin"
        path.write_bytes(t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes())
        print(f"  wrote {path} ({path.stat().st_size} bytes)")

    print(f"\nwriting test vectors → {OUT_DIR}")
    dump("h_t", h_t)
    dump("e_token", e_tok)
    dump("expected_fc_in", interm["fc_in"])
    dump("expected_h0", interm["h0"])
    dump("expected_attn_pre_gate", interm["attn_pre_gate"])
    dump("expected_attn_post_gate", interm["attn_post_gate"])
    dump("expected_h1", interm["h1"])
    dump("expected_h2", interm["h2"])
    dump("expected_h_final", interm["h_final"])

    meta = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_eps": RMS_EPS,
        "rope_partial_dim": ROPE_PARTIAL_DIM,
        "rope_theta": 1e7,
        "test_position": TEST_POSITION,
        "seed": SEED,
        "mtp_gguf": str(MTP_GGUF),
        "h_final_mean": float(h_final.mean().item()),
        "h_final_std": float(h_final.std().item()),
    }
    (OUT_DIR / "inputs.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {OUT_DIR / 'inputs.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
