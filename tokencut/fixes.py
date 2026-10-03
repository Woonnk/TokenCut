"""Conservative, review-only source patches for selected audit findings."""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FixResult:
    source: str
    diff: str
    changed: bool
    skipped: int


def compact_json_preview(source: str, path: str = "<string>") -> FixResult:
    """Preview compact JSON calls; never write or execute the source."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise ValueError("Python source could not be parsed; no patch was generated") from exc
    edits: list[tuple[int, int, str]] = []
    skipped = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr == "dumps"
        ):
            continue
        indent = next((item for item in node.keywords if item.arg == "indent"), None)
        has_separators = any(item.arg == "separators" for item in node.keywords)
        if indent is None or has_separators or node.lineno != node.end_lineno:
            skipped += int(indent is not None)
            continue
        line_start = sum(len(line) for line in source.splitlines(keepends=True)[: node.lineno - 1])
        start, end = line_start + node.col_offset, line_start + node.end_col_offset
        call = source[start:end]
        if "#" in call:
            skipped += 1
            continue
        compact, replacements = re.subn(
            r",\s*indent\s*=\s*(?:[0-9]+|None)\s*\)$",
            ", separators=(',', ':'))",
            call,
        )
        if replacements != 1:
            skipped += 1
            continue
        edits.append((start, end, compact))
    output = source
    for start, end, replacement in sorted(edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            output.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (proposed)",
        )
    )
    return FixResult(output, diff, bool(edits), skipped)
