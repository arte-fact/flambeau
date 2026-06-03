# scripts/fix_doc_lazy.awk
#
# Usage:
#   cargo clippy --release --features hip_serve --workspace --all-targets \
#     2>&1 | awk -f scripts/fix_doc_lazy.awk
#
# Reads clippy's `doc_lazy_continuation` warnings on stdin and inserts
# two spaces after the `///` / `//!` marker on each offending line so
# clippy 1.96+ stops treating the wrap line as a lazy list-item
# continuation. `//! foo` becomes `//!   foo`; `/// bar` becomes
# `///   bar`. Indented wrap lines are left untouched.
#
# Idempotent: re-running on a clean tree is a no-op (no warnings ->
# no sites). Re-running on a tree where some sites are already fixed
# only touches the remaining ones (clippy doesn't report fixed sites).

/^warning: doc list item without indentation/ { armed = 1; next }
armed && /^[[:space:]]*--> / {
  loc = $0
  sub(/^[[:space:]]*--> /, "", loc)
  n = split(loc, p, ":")
  if (n >= 2) {
    f = p[1]
    sites[f] = sites[f] " " (p[2] + 0)
  }
  armed = 0
  next
}
{ armed = 0 }

END {
  total_files = 0
  total_lines = 0
  for (f in sites) {
    delete want
    n = split(sites[f], lns, " ")
    for (i = 1; i <= n; i++) if (lns[i] != "") want[lns[i] + 0] = 1

    tmp = f ".lazy_fix.tmp"
    fnr = 0
    cnt = 0
    while ((getline ln < f) > 0) {
      fnr++
      if ((fnr in want) && \
          sub(/^([[:space:]]*\/\/[!\/])[[:space:]]/, "&  ", ln) > 0) {
        cnt++
      }
      print ln > tmp
    }
    close(f)
    close(tmp)
    system("mv " tmp " " f)
    if (cnt > 0) {
      printf "  %s: %d line(s) patched\n", f, cnt > "/dev/stderr"
      total_files++
      total_lines += cnt
    } else {
      system("rm -f " tmp)
    }
  }
  if (total_files > 0) {
    printf "fix_doc_lazy.awk: %d line(s) across %d file(s)\n", \
      total_lines, total_files > "/dev/stderr"
  } else {
    printf "fix_doc_lazy.awk: no doc_lazy_continuation warnings on stdin\n" \
      > "/dev/stderr"
  }
}
