# scripts/migrate_moe_callers.awk
#
# Migrate ops.indexed_moe_mmvq_<X>(...) callers (8-arg flat form) to
# the (MoeMmvqBuffers, MoeMmvqShape) aggregate form.
#
# Targets:
#   <indent>ops.indexed_moe_mmvq_<X>(
#   <indent>    <weights_expr>,
#   <indent>    <act_expr>,
#   <indent>    <expert_ids_expr>,
#   <indent>    <dst_expr>,
#   <indent>    <n_rows_expr>,
#   <indent>    <n_tokens_expr>,
#   <indent>    <top_k_expr>,
#   <indent>    <n_sb_per_row_expr>,
#   <indent>)
#
# Each expr line may carry an arbitrary expression like
# `self.ffn_gate_exps.ptr` or `scratch.x_q8_1` or `inter / QK_K`.
# The indent of the closing `)` matches the indent of the
# `ops.indexed_moe_mmvq_<X>(` line; arg lines are indented one
# rust-step deeper.

/^[ \t]*ops\.indexed_moe_mmvq_[a-zA-Z0-9_]+\($/ {
  head = $0
  # Capture the indent of the head line
  match(head, /^[ \t]*/)
  outer_indent = substr(head, 1, RLENGTH)
  # Function name
  fname = head
  sub(/^[ \t]*ops\./, "ops.", fname)
  # Buffer the next 9 lines (8 args + closing line).
  buf[0] = head
  lcnt = 1
  ok = 1
  for (i = 1; i <= 9; i++) {
    if ((getline line) <= 0) { ok = 0; break }
    buf[lcnt++] = line
  }
  # Validate close paren line: indent + `)`
  close_expected = outer_indent ")"
  if (ok && buf[9] != close_expected) ok = 0

  if (ok) {
    # Inner indent = outer + 4 spaces
    inner_indent = outer_indent "    "
    # Each arg line should look like: <inner_indent><expr>,
    # Strip leading whitespace + trailing comma from each.
    for (i = 1; i <= 8; i++) {
      a = buf[i]
      # remove leading whitespace
      sub(/^[ \t]+/, "", a)
      # remove trailing comma + spaces
      sub(/,[ \t]*$/, "", a)
      arg[i] = a
    }
    print head
    print inner_indent "flambeau_ops::MoeMmvqBuffers {"
    print inner_indent "    weights: " arg[1] ","
    print inner_indent "    act: " arg[2] ","
    print inner_indent "    expert_ids: " arg[3] ","
    print inner_indent "    dst: " arg[4] ","
    print inner_indent "},"
    print inner_indent "flambeau_ops::MoeMmvqShape {"
    print inner_indent "    n_rows: " arg[5] ","
    print inner_indent "    n_tokens: " arg[6] ","
    print inner_indent "    top_k: " arg[7] ","
    print inner_indent "    n_sb_per_row: " arg[8] ","
    print inner_indent "},"
    print buf[9]
    next
  }
  for (i = 0; i < lcnt; i++) print buf[i]
  next
}

{ print }
