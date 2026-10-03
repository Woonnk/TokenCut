from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

ERROR = re.compile(
    r"\b(?:error|fatal|exception|traceback|failed|failure|panic)\b|\b\w*(?:Error|Exception)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"[a-zA-Z0-9_]{2,}")
STOP_WORDS = {"the", "and", "for", "with", "this", "that", "from", "what", "why", "how"}


def terms(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if word not in STOP_WORDS}


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def compact_json(text: str) -> str:
    # Validate structurally, then preserve numeric lexemes, duplicate keys, and key order.
    json.loads(text, parse_float=Decimal, parse_constant=_invalid_constant)
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            output.append(char)
        elif char not in " \t\r\n":
            output.append(char)
    return "".join(output)


@dataclass(frozen=True)
class LogResult:
    text: str
    removed_lines: int
    critical_lines: int


def trim_log(text: str, query: str = "", context: int = 2, tail: int = 6) -> LogResult:
    if type(context) is not int or context < 0 or type(tail) is not int or tail < 0:
        raise ValueError("context and tail must be nonnegative integers")
    lines = text.splitlines(keepends=True)
    if not lines:
        return LogResult(text, 0, 0)
    critical = {i for i, line in enumerate(lines) if ERROR.search(line)}
    # Keep complete Python traceback bodies, not just the exception headline.
    for i, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            for j in range(i + 1, len(lines)):
                critical.add(j)
                if lines[j].strip() and not lines[j][0].isspace():
                    break
    keep = set(range(min(2, len(lines))))
    if tail:
        keep.update(range(max(0, len(lines) - tail), len(lines)))
    query_terms = terms(query)
    matches = sorted(
        ((len(terms(line) & query_terms), i) for i, line in enumerate(lines)),
        key=lambda item: (-item[0], item[1]),
    )
    anchors = critical | {i for score, i in matches[:8] if score}
    for i in anchors:
        keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    if len(keep) == len(lines):
        return LogResult(text, 0, len(critical))
    output: list[str] = []
    i = 0
    while i < len(lines):
        if i in keep:
            output.append(lines[i])
            i += 1
        else:
            start = i
            while i < len(lines) and i not in keep:
                i += 1
            output.append(f"[TokenCut omitted {i - start} log lines]\n")
    return LogResult("".join(output), len(lines) - len(keep), len(critical))
