# scripts/strip_allow.awk
#
# Strip all `#[allow(...)]` and `#![allow(...)]` attributes from Rust
# source. Handles single-line and multi-line forms (paren-tracked).
#
# Usage:
#   find crates -name '*.rs' -print0 | while IFS= read -r -d '' f; do
#     awk -f scripts/strip_allow.awk "$f" > "$f.tmp" && mv "$f.tmp" "$f"
#   done
#
# Or via a single helper invocation (see scripts/strip_allow.sh).
#
# Re-running on a stripped tree is a no-op.

function paren_delta(s,   _o, _c) {
  _o = gsub(/\(/, "&", s)
  _c = gsub(/\)/, "&", s)
  return _o - _c
}

{
  if (in_allow) {
    depth += paren_delta($0)
    if (depth <= 0) in_allow = 0
    next
  }
  if ($0 ~ /^[[:space:]]*#!?\[[[:space:]]*allow[[:space:]]*\(/) {
    depth = paren_delta($0)
    if (depth <= 0) next       # single-line attribute, drop
    in_allow = 1
    next                       # first line of multi-line, drop
  }
  print
}
