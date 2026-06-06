"""S1–S6 scenarios from doc/TOOL_CALLING_TEST_PROTOCOL.md.

Each scenario is a pure function that returns the kwargs we pass to
`client.chat.completions.create(...)`. The harness orchestrator drives
them, captures the response, and runs `assertions.check_<scenario>`.
"""
from __future__ import annotations
import json

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, e.g. 'Paris, France'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the public web for a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

COMMON = dict(temperature=0.0, top_p=1.0, seed=12345, max_tokens=256)


def s1(model: str) -> dict:
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "What's the weather in Paris right now?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        **COMMON,
    )


def s2(model: str) -> dict:
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "Find me the latest news about ROCm 7.2."}],
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        tool_choice="auto",
        **COMMON,
    )


def s3_followup(model: str, assistant_msg: dict, tool_call_id: str) -> dict:
    """Round-trip: feed the simulated tool result back."""
    return dict(
        model=model,
        messages=[
            {"role": "user",
             "content": "What's the weather in Paris right now?"},
            assistant_msg,
            {"role": "tool",
             "tool_call_id": tool_call_id,
             "name": "get_current_weather",
             "content": json.dumps(
                 {"temperature_c": 18, "sky": "overcast"})},
        ],
        tools=[WEATHER_TOOL],
        **{**COMMON, "max_tokens": 128},
    )


def s4(model: str) -> dict:
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "Compare the weather in Paris and Tokyo."}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        parallel_tool_calls=True,
        **COMMON,
    )


def s5(model: str) -> dict:
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "What's the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        **COMMON,
    )


def s6(model: str) -> dict:
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "What's the capital of France?"}],
        **{**COMMON, "max_tokens": 128},
    )


def s7(model: str) -> dict:
    """Thinking: request reasoning, expect a split `reasoning_content`
    with a clean `content` (no leaked `<think>` / channel markers).

    `enable_thinking` is a flambeau request extension, passed via the
    OpenAI SDK's `extra_body` escape hatch."""
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "A bat and ball cost $1.10. The bat costs $1 more "
                              "than the ball. How much is the ball? Think it through."}],
        extra_body={"enable_thinking": True},
        **{**COMMON, "temperature": 1.0, "max_tokens": 1024},
    )


def s8(model: str) -> dict:
    """Streaming thinking: reasoning must arrive as distinct
    `delta.reasoning_content` chunks, ahead of and separate from the
    answer `delta.content`, with no `<think>` markers leaking into either."""
    return dict(
        model=model,
        messages=[{"role": "user",
                   "content": "A bat and ball cost $1.10. The bat costs $1 more "
                              "than the ball. How much is the ball? Think it through."}],
        stream=True,
        extra_body={"enable_thinking": True},
        **{**COMMON, "temperature": 1.0, "max_tokens": 1024},
    )
