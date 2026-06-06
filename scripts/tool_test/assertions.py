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
    # A verbose-but-correct model can legitimately hit the token cap; the
    # signal here is "answers without tools, cleanly", not early EOS.
    out.append({
        "name": "finish_reason in {stop,length}",
        "pass": choice.finish_reason in ("stop", "length"),
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


def _reasoning_of(msg: Any) -> str | None:
    """`reasoning_content` is a flambeau extension field; the OpenAI SDK
    parks unknown fields in `model_extra`."""
    direct = getattr(msg, "reasoning_content", None)
    if direct:
        return direct
    extra = getattr(msg, "model_extra", None) or {}
    return extra.get("reasoning_content")


def check_s7(resp: Any) -> list[dict]:
    out = []
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    reasoning = _reasoning_of(msg)
    truncated = choice.finish_reason == "length"
    out.append({
        "name": "finish_reason in {stop,length}",
        "pass": choice.finish_reason in ("stop", "length"),
        "detail": f"got {choice.finish_reason!r}",
    })
    # The split is the actual unit under test: reasoning must land in its
    # own field, never inline.
    out.append({
        "name": "reasoning_content present",
        "pass": bool(reasoning and reasoning.strip()),
        "detail": f"len={len(reasoning or '')}",
    })
    # If the model was cut off mid-thought (length), an empty answer is
    # expected and not a failure; only require a non-empty answer when it
    # actually finished (stop).
    out.append({
        "name": "answer content non-empty (when not truncated)",
        "pass": bool(content.strip()) or truncated,
        "detail": f"truncated={truncated} content[:80]={content[:80]!r}",
    })
    # The split must be clean: no reasoning markers in either field.
    c_leak = _has_leak(content)
    r_leak = _has_leak(reasoning)
    for marker in ("<think>", "</think>"):
        if marker in content:
            c_leak = marker
    out.append({
        "name": "no think/channel markers leak into content",
        "pass": c_leak is None and "<think>" not in content and "</think>" not in content,
        "detail": f"content leak: {c_leak!r}",
    })
    out.append({
        "name": "reasoning field free of channel scaffolding",
        "pass": r_leak is None,
        "detail": f"reasoning leak: {r_leak!r}",
    })
    return out


def check_s8(resp: Any) -> list[dict]:
    """`resp` here is the accumulated-stream dict from run.py
    (`{reasoning, content, order, finish_reason}`), not an SDK object."""
    out = []
    reasoning = resp.get("reasoning", "")
    content = resp.get("content", "")
    order = resp.get("order", [])
    truncated = resp.get("finish_reason") == "length"
    out.append({
        "name": "reasoning_content streamed as deltas",
        "pass": bool(reasoning.strip()),
        "detail": f"reasoning_len={len(reasoning)}",
    })
    out.append({
        "name": "answer content streamed (or truncated mid-think)",
        "pass": bool(content.strip()) or truncated,
        "detail": f"content_len={len(content)} truncated={truncated}",
    })
    leaked = "<think>" in content or "</think>" in content or _has_leak(content)
    out.append({
        "name": "no think/channel markers in content deltas",
        "pass": not leaked,
        "detail": f"content[:80]={content[:80]!r}",
    })
    # Reasoning must lead: the first reasoning delta precedes the first
    # content delta.
    first_r = order.index("r") if "r" in order else len(order)
    first_c = order.index("c") if "c" in order else len(order)
    out.append({
        "name": "reasoning deltas precede content deltas",
        "pass": first_r <= first_c,
        "detail": f"first_reasoning={first_r} first_content={first_c}",
    })
    return out


CHECKS = {
    "S1": check_s1, "S2": check_s2, "S3": check_s3,
    "S4": check_s4, "S5": check_s5, "S6": check_s6,
    "S7": check_s7, "S8": check_s8,
}
