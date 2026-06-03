# scripts/migrate_moe_impl.awk
#
# Migrate single-weight MoE MMVQ HipOps impl methods in ops_impl.rs to
# the (buffers, shape) aggregate signature + forward via OpCtx.
#
# Targets:
#       fn indexed_moe_mmvq_<X>(
#           &self,
#           w: DevicePtr,
#           y: DevicePtr,
#           expert_ids: DevicePtr,
#           dst: DevicePtr,
#           n_rows: usize,
#           n_tokens: usize,
#           top_k: usize,
#           n_(sb|blocks)_per_row: usize,
#       ) -> Result<()> {
#           super::moe::<X>(
#               self.reg,
#               self.stream,
#               w, y, expert_ids, dst, n_rows, n_tokens, top_k, <slot>,
#           )
#       }

/^    fn indexed_moe_mmvq_[a-zA-Z0-9_]+\($/ {
  fname = $0
  sub(/^    fn /, "", fname)
  sub(/\($/, "", fname)
  buf[0] = $0
  lcnt = 1
  ok = 1
  # Read the sig + body: 9 sig lines + open + 11 body + close = ~23 lines
  for (i = 1; i <= 23; i++) {
    if ((getline line) <= 0) { ok = 0; break }
    buf[lcnt++] = line
  }
  expected[1] = "        &self,"
  expected[2] = "        w: DevicePtr,"
  expected[3] = "        y: DevicePtr,"
  expected[4] = "        expert_ids: DevicePtr,"
  expected[5] = "        dst: DevicePtr,"
  expected[6] = "        n_rows: usize,"
  expected[7] = "        n_tokens: usize,"
  expected[8] = "        top_k: usize,"
  if (ok) {
    for (i = 1; i <= 8; i++) {
      if (buf[i] != expected[i]) { ok = 0; break }
    }
  }
  if (ok && buf[9] != "        n_sb_per_row: usize," && buf[9] != "        n_blocks_per_row: usize,") ok = 0
  if (ok && buf[10] != "    ) -> Result<()> {") ok = 0
  # Body shape: super::moe::<fname>(self.reg, self.stream, w, y, expert_ids,
  # dst, n_rows, n_tokens, top_k, <slot>,)
  expected_body_start = "        super::moe::" fname "("
  if (ok && buf[11] != expected_body_start) ok = 0
  # The next 11 body lines are mostly identifier args + closing `}`.

  if (ok) {
    print "    fn " fname "("
    print "        &self,"
    print "        buffers: crate::MoeMmvqBuffers,"
    print "        shape: crate::MoeMmvqShape,"
    print "    ) -> Result<()> {"
    print "        super::moe::" fname "("
    print "            crate::OpCtx {"
    print "                reg: self.reg,"
    print "                stream: self.stream,"
    print "            },"
    print "            buffers,"
    print "            shape,"
    print "        )"
    print "    }"
    next
  }
  for (i = 0; i < lcnt; i++) print buf[i]
  next
}

{ print }
