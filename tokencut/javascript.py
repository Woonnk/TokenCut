"""Conservative, dependency-free JS/TS source inspection; no code execution."""

from __future__ import annotations

import re

EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
IDENT = r"[A-Za-z_$][\w$]*"
IMPORT = re.compile(r"\b(?:import|export)\b[^\n;]*?\bfrom\s*(['\"])([^'\"\n]+)\1")
SIDE_EFFECT_IMPORT = re.compile(r"\bimport\s*(['\"])([^'\"\n]+)\1")
REQUIRE = re.compile(r"\brequire\s*\(\s*(['\"])([^'\"\n]+)\1\s*\)")
FUNCTION = re.compile(rf"\bfunction\s+({IDENT})\s*\(")
CLASS = re.compile(rf"\bclass\s+({IDENT})\b")
ARROW = re.compile(
    rf"\b(?:const|let|var)\s+({IDENT})\s*=\s*(?:async\s*)?(?:\([^;]*?\)|{IDENT})\s*=>"
)
METHOD = re.compile(rf"^\s*(?:async\s+)?({IDENT})\s*\([^;]*?\)\s*(?::[^{{;]+)?\s*{{")
CALL = re.compile(
    r"\b(?:client|openai|model|llm|chat)\b[\w.$]*\.(?:create|invoke|generate|complete)\s*\("
)
HISTORY = re.compile(r"\b(?:messages|input)\s*:\s*(?:history|messages|chatHistory|conversation)\b")
FILE_READ = re.compile(r"\b(?:readFileSync|readFile|readTextFile|readFileAsString)\s*\(")
VERBOSE_JSON = re.compile(r"\bJSON\s*\.\s*stringify\s*\([^\n;]*?,\s*null\s*,\s*[1-9]\d*\s*\)")
SIMPLE_VERBOSE_JSON = re.compile(
    rf"\bJSON\s*\.\s*stringify\s*\(\s*{IDENT}(?:\.{IDENT})*\s*,\s*null\s*,\s*[1-9]\d*\s*\)"
)
LOOP = re.compile(r"\b(?:for|while)\s*\(")


def mask_literals(source: str) -> str:
    """Keep offsets/newlines; hide strings, templates, comments and simple regex literals."""
    result = list(source)
    i = 0
    state = "code"
    start = 0
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if char in "'\"`":
                state = char
                start = i
            elif char == "/" and nxt in {"/", "*"}:
                state = "line" if nxt == "/" else "block"
                start = i
            elif char == "/" and source[:i].rstrip().endswith(("=", "(", "[", ":", ",")):
                state = "regex"
                start = i
        elif state in {"'", '"', "`", "regex"}:
            if char == "\\":
                i += 1
            elif char == state or (state == "regex" and char == "/"):
                for pos in range(start, min(i + 1, len(source))):
                    if result[pos] != "\n":
                        result[pos] = " "
                state = "code"
        elif state == "line" and char == "\n":
            for pos in range(start, i):
                result[pos] = " "
            state = "code"
        elif state == "block" and char == "*" and nxt == "/":
            i += 1
            for pos in range(start, i + 1):
                if result[pos] != "\n":
                    result[pos] = " "
            state = "code"
        i += 1
    if state != "code":
        for pos in range(start, len(source)):
            if result[pos] != "\n":
                result[pos] = " "
    return "".join(result)


def inspect_module(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    masked = mask_literals(source)
    imports: set[str] = set()
    symbols: list[str] = []
    depth = 0
    active_class: tuple[str, int] | None = None
    for original, line in zip(source.splitlines(), masked.splitlines(), strict=True):
        if depth == 0:
            for pattern in (IMPORT, SIDE_EFFECT_IMPORT, REQUIRE):
                for match in pattern.finditer(original):
                    if line[match.start() : match.start() + 6].strip() in {
                        "import", "export", "requir"
                    }:
                        imports.add(match.group(2))
            for pattern in (FUNCTION, CLASS, ARROW):
                for match in pattern.finditer(line):
                    symbols.append(match.group(1))
                    if pattern is CLASS:
                        active_class = (match.group(1), depth + 1)
        elif active_class and depth == active_class[1]:
            match = METHOD.match(line)
            if match and match.group(1) not in {"if", "for", "while", "switch"}:
                symbols.append(f"{active_class[0]}.{match.group(1)}")
        depth += line.count("{") - line.count("}")
        if active_class and depth < active_class[1]:
            active_class = None
    return tuple(sorted(imports)), tuple(symbols)


def inspect_waste(source: str) -> list[tuple[int, str]]:
    masked = mask_literals(source)
    findings: list[tuple[int, str]] = []
    depth = 0
    loop_depths: list[int] = []
    for lineno, (_original, line) in enumerate(
        zip(source.splitlines(), masked.splitlines(), strict=True), 1
    ):
        if LOOP.search(line) and "{" in line:
            loop_depths.append(depth + 1)
        for call in CALL.finditer(line):
            close = line.find(")", call.end())
            tail = line[call.end() : close if close >= 0 else len(line)]
            if HISTORY.search(tail):
                findings.append((lineno, "TC001"))
            if loop_depths:
                findings.append((lineno, "TC002"))
        if FILE_READ.search(line):
            findings.append((lineno, "TC003"))
        if VERBOSE_JSON.search(line):
            findings.append((lineno, "TC004"))
        depth += line.count("{") - line.count("}")
        loop_depths = [level for level in loop_depths if depth >= level]
    return findings
