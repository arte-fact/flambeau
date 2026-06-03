# scripts/migrate_moe_freefn.awk
#
# Migrate single-weight MoE MMVQ free functions in hip/moe.rs to the
# (ctx, MoeMmvqBuffers, MoeMmvqShape) aggregate signature.
#
# Targets functions with the exact 10-param shape:
#   pub fn indexed_moe_mmvq_<X>(
#       reg: &OpsRegistry,
#       stream: &HipStream,
#       w: DevicePtr,
#       y: DevicePtr,
#       expert_ids: DevicePtr,
#       dst: DevicePtr,
#       n_rows: usize,
#       n_tokens: usize,
#       top_k: usize,
#       n_(sb|blocks)_per_row: usize,
#   ) -> Result<()> {
#
# State machine: when the function-header line is seen, read the next
# ~11 lines, validate the pattern, and emit the rewritten sig +
# destructure prelude. Inside the rewritten function body, rewrite
# `reg.expect_module` -> `ctx.reg.expect_module` and
# `kernel.launch(stream,` -> `kernel.launch(ctx.stream,`. Track the
# closing `}` (column 1) to know when we leave the function.
#
# Usage:
#   awk -f scripts/migrate_moe_freefn.awk crates/ops/src/hip/moe.rs > tmp \
#     && mv tmp crates/ops/src/hip/moe.rs

/^pub fn indexed_moe_mmvq_[a-zA-Z0-9_]+\($/ {
  fname = $0
  sub(/^pub fn /, "", fname)
  sub(/\($/, "", fname)
  buf[0] = $0
  lcnt = 1
  ok = 1
  for (i = 1; i <= 11; i++) {
    if ((getline line) <= 0) { ok = 0; break }
    buf[lcnt++] = line
  }
  expected[1] = "    reg: &OpsRegistry,"
  expected[2] = "    stream: &HipStream,"
  expected[3] = "    w: DevicePtr,"
  expected[4] = "    y: DevicePtr,"
  expected[5] = "    expert_ids: DevicePtr,"
  expected[6] = "    dst: DevicePtr,"
  expected[7] = "    n_rows: usize,"
  expected[8] = "    n_tokens: usize,"
  expected[9] = "    top_k: usize,"
  if (ok) {
    for (i = 1; i <= 9; i++) {
      if (buf[i] != expected[i]) { ok = 0; break }
    }
  }
  slot = ""
  if (ok && buf[10] == "    n_sb_per_row: usize,") slot = "n_sb_per_row"
  else if (ok && buf[10] == "    n_blocks_per_row: usize,") slot = "n_blocks_per_row"
  else ok = 0
  if (ok && buf[11] != ") -> Result<()> {") ok = 0

  if (ok) {
    print "pub fn " fname "("
    print "    ctx: crate::OpCtx<'_>,"
    print "    buffers: crate::MoeMmvqBuffers,"
    print "    shape: crate::MoeMmvqShape,"
    print ") -> Result<()> {"
    print "    let crate::MoeMmvqBuffers { weights: w, act: y, expert_ids, dst } = buffers;"
    if (slot == "n_blocks_per_row") {
      print "    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row: n_blocks_per_row } = shape;"
    } else {
      print "    let crate::MoeMmvqShape { n_rows, n_tokens, top_k, n_sb_per_row } = shape;"
    }
    in_migrated = 1
    next
  }
  for (i = 0; i < lcnt; i++) print buf[i]
  next
}

in_migrated {
  if ($0 == "}") {
    print
    in_migrated = 0
    next
  }
  line = $0
  gsub(/reg\.expect_module/, "ctx.reg.expect_module", line)
  gsub(/kernel\.launch\(stream,/, "kernel.launch(ctx.stream,", line)
  print line
  next
}

{ print }
