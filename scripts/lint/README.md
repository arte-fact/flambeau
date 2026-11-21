# Comment / naming lint

`comment_lint.py` flags the comment debt the root `CLAUDE.md` bans: task/PR
refs (`#229`), commit shas, ROI/timing claims (`saves 200us`, `3x faster`),
dotted phase markers (`C4.1`, `5.g`, `step 2`), narrative prose (`Mirror of`),
and HIP/`gfx906` references inside `crates/{backend,kernels}-cuda`. It inspects
comment text only.

**Enforcement (new edits):** `.claude/settings.json` runs it as a `PostToolUse`
hook on `Write`/`Edit`/`MultiEdit`. It lints just the inserted text and blocks
(exit 2, findings fed back) so violations never land — editing files that still
carry old debt is not blocked.

**Sweep / CI (existing files):**

```
python3 scripts/lint/comment_lint.py $(git ls-files 'crates/**/*.rs' 'crates/**/*.cu' 'crates/**/*.cuh')
```

Exit 1 if any file carries a violation. To ratchet the backlog down, gate CI on
a per-file allowlist or a shrinking total.
