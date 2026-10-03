"""Opt-in tool-output filtering without altering message roles or tool-call structure."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from .core import build_report
from .filters import compact_json, trim_log
from .model import serialize
from .tokens import TokenCounter


@dataclass(frozen=True)
class RequestResult:
    request: dict
    report: dict


def optimize_request(
    request: dict, *, tool_kinds: dict[str, str], counter: TokenCounter | None = None
) -> RequestResult:
    """Return a copy. Only string outputs of explicitly allowlisted tools may change."""
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise ValueError("Request requires a messages array")
    if not isinstance(tool_kinds, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(kind, str)
        or kind not in {"log", "json"}
        for name, kind in tool_kinds.items()
    ):
        raise ValueError("tool_kinds must map tool names to log or json")
    counter = counter or TokenCounter()
    start = perf_counter()
    before = serialize(request)
    output = deepcopy(request)
    changes = []
    calls: dict[str, str | None] = {}
    for i, message in enumerate(output["messages"]):
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            for call in tool_calls if isinstance(tool_calls, list) else []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    call_id = call.get("id")
                    name = call["function"].get("name")
                    if isinstance(call_id, str):
                        # Reused IDs are ambiguous; do not guess which output is being returned.
                        calls[call_id] = (
                            name if call_id not in calls and isinstance(name, str) else None
                        )
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        call_id = message.get("tool_call_id")
        if call_id is not None and not isinstance(call_id, str):
            continue
        name = calls.get(call_id, message.get("name"))
        if not isinstance(name, str) or name not in tool_kinds:
            continue
        kind = tool_kinds[name]
        original = message["content"]
        try:
            text = trim_log(original).text if kind == "log" else compact_json(original)
        except (ValueError, RecursionError):
            changes.append(
                {
                    "message_index": i,
                    "tool": name,
                    "action": "invalid_json_unchanged",
                    "lossy": False,
                }
            )
            continue
        if counter.count(text) < counter.count(original):
            message["content"] = text
            changes.append(
                {
                    "message_index": i,
                    "tool": name,
                    "action": f"filter_{kind}",
                    "lossy": kind == "log",
                }
            )
    after = serialize(output)
    if counter.count(after) > counter.count(before):
        output, after, changes = deepcopy(request), before, []
    report = build_report(before, after, counter, changes, (perf_counter() - start) * 1000)
    return RequestResult(output, report)
