#!/usr/bin/env bash
# Per-step logit parity: v2 SD vs legacy qwen3-moe PP=1 on the same
# model + prompt, run in SEPARATE PROCESSES so the multi-Session HIP
# address-reuse bug can't contaminate the comparison. Each test dumps
# its per-step F32 logit vectors to a binary file; this script reads
# both back and prints argmax + max|diff| per step.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V2_OUT="${V2_OUT:-/tmp/parity_v2.bin}"
LEG_OUT="${LEG_OUT:-/tmp/parity_legacy.bin}"

rm -f "$V2_OUT" "$LEG_OUT"

echo "=== v2 dump → $V2_OUT ==="
FLAMBEAU_PARITY_OUT="$V2_OUT" cargo test \
  --manifest-path "$ROOT/Cargo.toml" \
  -p flambeau-qwen35-v2 --features hip --release \
  --test parity_dump_v2 \
  -- --ignored --test-threads=1 --nocapture

echo "=== legacy dump → $LEG_OUT ==="
FLAMBEAU_PARITY_OUT="$LEG_OUT" cargo test \
  --manifest-path "$ROOT/Cargo.toml" \
  -p flambeau-qwen3-moe --features hip --release \
  --test parity_dump_legacy \
  -- --ignored --test-threads=1 --nocapture

echo "=== diff ==="
python3 - "$V2_OUT" "$LEG_OUT" <<'PY'
import struct, sys
def load(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(4)
            if not hdr: break
            n = struct.unpack('<I', hdr)[0]
            buf = f.read(n * 4)
            out.append(struct.unpack(f'<{n}f', buf))
    return out

a = load(sys.argv[1])
b = load(sys.argv[2])
assert len(a) == len(b), f"step count mismatch: v2={len(a)} legacy={len(b)}"

for i, (la, lb) in enumerate(zip(a, b)):
    assert len(la) == len(lb), f"step {i} vocab mismatch"
    argmax_a = max(range(len(la)), key=lambda k: la[k])
    argmax_b = max(range(len(lb)), key=lambda k: lb[k])
    diffs = [abs(x - y) for x, y in zip(la, lb)]
    mx = max(diffs)
    mx_idx = diffs.index(mx)
    mean = sum(diffs) / len(diffs)
    # cosine-ish similarity on the top-K to flag tail-vs-head divergence
    topk = sorted(range(len(la)), key=lambda k: la[k], reverse=True)[:8]
    pairs = [(k, la[k], lb[k], la[k] - lb[k]) for k in topk]
    print(f"step {i} argmax v2={argmax_a} legacy={argmax_b} "
          f"max|diff|={mx:.4f} @idx{mx_idx} mean|diff|={mean:.4f}")
    print(f"  v2 top8:     {pairs}")
PY
