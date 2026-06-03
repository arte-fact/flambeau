"""Pass-criterion checks for each scenario.

Every check returns a list of dicts:
    {"name": str, "pass": bool, "detail": str}

Aggregate verdict per scenario is `all(a["pass"] for a in checks)`.
"""
from __future__ import annotations
import json
import re
from typing import Any

LEAK_PATTERNS = [
    r"<\|tool_call>",
    r"<tool_call\|>",
    r"<tool_call>",
    r"</tool_call>",
    r"<function=",
    r"<parameter=",
    r"<\|channel>thought",
    r"<channel\|>",
    r"<\|im_sep_user\|>",
]


def _has_leak(text: str | None) -> str | None:
    if not text:
        return None
    for pat in LEAK_PATTERNS:
        if re.search(pat, text):
            return pat
    return None


def _arg_dict(tool_call: Any) -> tuple[bool, dict | None, str]:
    """Validate `arguments` is a JSON string (NOT object) and parse."""
    args_raw = getattr(tool_call.function, "arguments", None)
    if not isinstance(args_raw, str):
        return False, None, f"arguments is {type(args_raw).__name__}, not str"
    try:
        return True, json.loads(args_raw), ""
    except Exception as e:
        return False, None, f"arguments not valid JSON: {e!r}"


def check_s1(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    out.append({
        "name": "finish_reason=tool_calls",
        "pass": choice.finish_reason == "tool_calls",
        "detail": f"got {choice.finish_reason!r}",
    })
    tcs = msg.tool_calls or []
    out.append({
        "name": "tool_calls non-empty",
        "pass": len(tcs) >= 1,
        "detail": f"len={len(tcs)}",
    })
    if not tcs:
        return out
    tc = tcs[0]
    out.append({
        "name": "tool_call.id present",
        "pass": bool(getattr(tc, "id", None)),
        "detail": f"id={getattr(tc, 'id', None)!r}",
    })
    out.append({
        "name": "tool_call.type=function",
        "pass": tc.type == "function",
        "detail": f"type={tc.type!r}",
    })
    out.append({
        "name": "tool_call.function.name=get_current_weather",
        "pass": tc.function.name == "get_current_weather",
        "detail": f"name={tc.function.name!r}",
    })
    ok, args, detail = _arg_dict(tc)
    out.append({
        "name": "arguments parses as JSON string",
        "pass": ok,
        "detail": detail,
    })
    if ok:
        loc = (args or {}).get("location") or ""
        out.append({
            "name": "arguments.location mentions Paris",
            "pass": "paris" in str(loc).lower(),
            "detail": f"location={loc!r}",
        })
    content = msg.content or ""
    out.append({
        "name": "content empty or whitespace",
        "pass": not content.strip(),
        "detail": f"content={content[:80]!r}",
    })
    leak = _has_leak(content)
    out.append({
        "name": "no raw tool/channel tokens in content",
        "pass": leak is None,
        "detail": f"leaked pattern: {leak!r}",
    })
    return out


def check_s2(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    out.append({
        "name": "finish_reason=tool_calls",
        "pass": choice.finish_reason == "tool_calls",
        "detail": f"got {choice.finish_reason!r}",
    })
    tcs = msg.tool_calls or []
    out.append({
        "name": "exactly one tool_call",
        "pass": len(tcs) == 1,
        "detail": f"len={len(tcs)}",
    })
    if not tcs:
        return out
    tc = tcs[0]
    out.append({
        "name": "routed to search_web",
        "pass": tc.function.name == "search_web",
        "detail": f"name={tc.function.name!r}",
    })
    ok, args, detail = _arg_dict(tc)
    out.append({"name": "arguments parses", "pass": ok, "detail": detail})
    if ok:
        q = (args or {}).get("query") or ""
        out.append({
            "name": "arguments.query mentions ROCm or 7.2",
            "pass": ("rocm" in str(q).lower()) or ("7.2" in str(q)),
            "detail": f"query={q!r}",
        })
    return out


def check_s3(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    out.append({
        "name": "finish_reason=stop",
        "pass": choice.finish_reason == "stop",
        "detail": f"got {choice.finish_reason!r}",
    })
    out.append({
        "name": "content non-empty",
        "pass": bool(content.strip()),
        "detail": f"len={len(content)}",
    })
    out.append({
        "name": "content mentions Paris",
        "pass": "paris" in content.lower(),
        "detail": "",
    })
    out.append({
        "name": "content mentions 18",
        "pass": "18" in content,
        "detail": "",
    })
    leak = _has_leak(content)
    out.append({
        "name": "no raw markers leak",
        "pass": leak is None,
        "detail": f"leaked: {leak!r}",
    })
    return out


def check_s4(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    tcs = msg.tool_calls or []
    out.append({
        "name": "finish_reason=tool_calls",
        "pass": choice.finish_reason == "tool_calls",
        "detail": f"got {choice.finish_reason!r}",
    })
    if len(tcs) >= 2:
        locs = []
        ids = set()
        for tc in tcs:
            ok, args, _ = _arg_dict(tc)
            if ok:
                locs.append(str((args or {}).get("location", "")).lower())
            ids.add(getattr(tc, "id", None))
        has_paris = any("paris" in l for l in locs)
        has_tokyo = any("tokyo" in l for l in locs)
        out.append({
            "name": "parallel calls cover Paris + Tokyo",
            "pass": has_paris and has_tokyo,
            "detail": f"locs={locs}",
        })
        out.append({
            "name": "tool_call ids unique",
            "pass": len(ids) == len(tcs) and None not in ids,
            "detail": f"ids={ids}",
        })
    else:
        # Sequential is acceptable; record but don't fail.
        out.append({
            "name": "parallel-not-supported (recorded, not failed)",
            "pass": True,
            "detail": f"len(tool_calls)={len(tcs)}",
        })
    return out


def check_s5(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    tcs = msg.tool_calls or []
    out.append({
        "name": "finish_reason=stop",
        "pass": choice.finish_reason == "stop",
        "detail": f"got {choice.finish_reason!r}",
    })
    out.append({
        "name": "no tool_calls emitted",
        "pass": len(tcs) == 0,
        "detail": f"len={len(tcs)}",
    })
    out.append({
        "name": "content non-empty",
        "pass": bool(content.strip()),
        "detail": "",
    })
    leak = _has_leak(content)
    out.append({
        "name": "no markers leak",
        "pass": leak is None,
        "detail": f"leaked: {leak!r}",
    })
    return out


def check_s6(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    out.append({
        "name": "finish_reason=stop",
        "pass": choice.finish_reason == "stop",
        "detail": f"got {choice.finish_reason!r}",
    })
    out.append({
        "name": "content mentions Paris",
        "pass": "paris" in content.lower(),
        "detail": f"content[:80]={content[:80]!r}",
    })
    leak = _has_leak(content)
    out.append({
        "name": "no tool/channel markers leak",
        "pass": leak is None,
        "detail": f"leaked: {leak!r}",
    })
    return out


CHECKS = {
    "S1": check_s1, "S2": check_s2, "S3": check_s3,
    "S4": check_s4, "S5": check_s5, "S6": check_s6,
}
