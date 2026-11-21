#!/usr/bin/env python3
"""Flag comment/naming debt the project CLAUDE.md bans.

Two modes:
  hook  — no args, reads a Claude Code tool-call JSON on stdin and lints only
          the inserted text (Write `content` / Edit `new_string`). Exit 2 with
          findings on stderr so the harness feeds them back before the edit
          "sticks". Exits 0 on anything it can't parse (never blocks spuriously).
  file  — paths as args, lints whole files. Exit 1 if any findings (CI / sweep).

Patterns are deliberately high-confidence (task refs, commit shas, ROI claims,
dotted phase markers, cross-backend leakage). It only inspects comment text.
"""
import json
import re
import sys

CODE_EXT = {".rs", ".cu", ".cuh", ".toml", ".sh", ".py"}
HASH_COMMENT_EXT = {".toml", ".sh", ".py"}

PATTERNS = [
    ("task-ref", re.compile(r"#\d{2,}\b")),
    ("commit-sha", re.compile(r"[0-9a-f]{7,40}(?:…|\.\.\.)|\bcommit\s+[0-9a-f]{7,}|(?<![0-9a-fx])[0-9a-f]{40}(?![0-9a-f])")),
    ("roi-claim", re.compile(r"\b(?:saves?|saving|~)\s*\d+\s*(?:ms|µs|us|ns)\b|\b\d+\s*[x×]\s*(?:faster|slower|speed-?up)|\b\d+\s*%\s*(?:faster|slower|regression|speed-?up|gain)", re.I)),
    ("phase-marker", re.compile(r"\b(?:step|phase)\s+\d+\b|\b[A-Z]\d+\.\d+[a-z]?\b|\b\d+\.[a-z]\d*\b")),
    ("narrative", re.compile(r"\bmirror(?:s|ed)? of\b|Mirrors llama|\blater phase\b", re.I)),
]
# Only in the CUDA backend/kernels trees: HIP / gfx906 references are leakage.
CUDA_CROSS = re.compile(r"\bgfx906\b|__builtin_amdgcn|hipSetDevice|backend-hip|\bHip[A-Z]\w*")


def comment_text(line: str, ext: str) -> str:
    """Best-effort extraction of the comment portion of a source line."""
    if ext in HASH_COMMENT_EXT:
        i = line.find("#")
        return line[i:] if i != -1 else ""
    i = line.find("//")
    if i != -1:
        return line[i:]
    s = line.lstrip()
    return line if s.startswith(("*", "/*")) else ""


def scan(text: str, path: str):
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    if ext not in CODE_EXT:
        return []
    pats = list(PATTERNS)
    if "/backend-cuda/" in path or "/kernels-cuda/" in path:
        pats = pats + [("cross-backend", CUDA_CROSS)]
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        c = comment_text(line, ext)
        if not c:
            continue
        for label, pat in pats:
            m = pat.search(c)
            if m:
                out.append((n, label, m.group(0).strip()))
    return out


def report(path, findings):
    for n, label, hit in findings:
        print(f"{path}:{n}: [{label}] {hit}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) > 1:  # file mode
        total = 0
        for path in sys.argv[1:]:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            f = scan(text, path)
            report(path, f)
            total += len(f)
        return 1 if total else 0

    # hook mode
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    ti = data.get("tool_input") or {}
    path = ti.get("file_path") or ""
    if not path:
        return 0
    if "content" in ti:
        text = ti.get("content") or ""
    elif "new_string" in ti:
        text = ti.get("new_string") or ""
    elif "edits" in ti:
        text = "\n".join((e or {}).get("new_string", "") for e in ti.get("edits") or [])
    else:
        return 0
    findings = scan(text, path)
    if not findings:
        return 0
    print("CLAUDE.md comment/naming rule — fix before continuing:", file=sys.stderr)
    report(path, findings)
    print("Comments: invariants / safety / non-obvious why only — no task refs, "
          "phase markers, commit shas, ROI claims, or HIP/gfx906 refs in CUDA code.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
