"""Heuristic Python source audit. Never imports or executes the audited project."""

from __future__ import annotations

import ast
import fnmatch
import io
import os
import re
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
MODEL_METHODS = {"create", "invoke", "ainvoke", "generate", "generate_content", "complete"}
HISTORY_NAMES = {"history", "messages", "chat_history", "conversation", "conversation_history"}
IGNORE = re.compile(r"tokencut:\s*ignore\[([A-Z0-9, ]+)\]")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    confidence: str
    message: str
    suggestion: str


@dataclass(frozen=True)
class AuditResult:
    files_scanned: int
    findings: tuple[Finding, ...]
    skipped: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "findings": [asdict(item) for item in self.findings],
            "skipped": list(self.skipped),
            "note": "Heuristics, not proven waste. No code was executed or changed.",
        }


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def scan_python(source: str, path: str = "<string>") -> list[Finding]:
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, RecursionError) as exc:
        return [
            Finding(
                path,
                getattr(exc, "lineno", None) or 1,
                "TC000",
                "warning",
                "high",
                "Python source could not be parsed; file was not audited.",
                "Check syntax and use a compatible Python interpreter.",
            )
        ]
    ignores: dict[int, set[str]] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and (match := IGNORE.search(token.string)):
                ignores[token.start[0]] = {rule.strip() for rule in match[1].split(",")}
    except tokenize.TokenError:
        pass
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    json_names = {"json.dumps"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            json_names.update(
                f"{alias.asname or 'json'}.dumps" for alias in node.names if alias.name == "json"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "json":
            json_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "dumps"
            )
    findings = []

    def add(
        node: ast.AST,
        rule: str,
        message: str,
        suggestion: str,
        severity: str = "info",
        confidence: str = "medium",
    ) -> None:
        line = node.lineno
        if rule not in ignores.get(line, set()):
            findings.append(Finding(path, line, rule, severity, confidence, message, suggestion))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) >= 2000
        ):
            parent = parents.get(node)
            if isinstance(parent, ast.Expr):
                continue
            add(
                node,
                "TC006",
                "Large embedded text may be resent on every request.",
                "Profile its usage; retrieve relevant sections and keep required instructions.",
            )
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        method = name.rsplit(".", 1)[-1]
        kwargs = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        is_model = method in MODEL_METHODS and any(
            keyword in kwargs for keyword in {"messages", "prompt", "input", "contents"}
        )
        if is_model:
            for keyword in ("messages", "input"):
                value = kwargs.get(keyword)
                if value is not None and _name(value).rsplit(".", 1)[-1] in HISTORY_NAMES:
                    add(
                        node,
                        "TC001",
                        "A history-like variable is passed directly to a model call.",
                        "Check whether history is bounded upstream; profile and compact old turns "
                        "without removing instructions or separating tool calls from results.",
                        "warning",
                    )
                    break
            parent = parents.get(node)
            while parent and not isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                if isinstance(
                    parent,
                    (
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                ):
                    add(
                        node,
                        "TC002",
                        "A model call appears inside a loop.",
                        "Add an iteration/token budget and stop when repeated calls "
                        "make no progress.",
                        "warning",
                        "high",
                    )
                    break
                parent = parents.get(parent)
        if method in {"read", "read_text"} and isinstance(node.func, ast.Attribute):
            size = node.args[0] if node.args else kwargs.get("size")
            bounded = method == "read" and size is not None
            if isinstance(size, ast.Constant) and size.value is None:
                bounded = False
            if isinstance(size, ast.UnaryOp) and isinstance(size.op, ast.USub):
                bounded = False
            if not bounded:
                add(
                    node,
                    "TC003",
                    "An entire file or stream is read into memory.",
                    "If this reaches a prompt, retrieve relevant functions or line ranges first.",
                )
        if name in json_names and "indent" in kwargs:
            value = kwargs["indent"]
            if not isinstance(value, ast.Constant) or value.value is not None:
                add(
                    node,
                    "TC004",
                    "JSON is pretty-printed; prompt whitespace may be avoidable.",
                    "For model-facing JSON, use json.dumps(data, separators=(',', ':')). "
                    "Keep human-facing output readable.",
                    confidence="high",
                )
        if name.endswith(("subprocess.run", "subprocess.check_output", "subprocess.Popen")):
            capture = kwargs.get("capture_output")
            if (isinstance(capture, ast.Constant) and capture.value is True) or (
                "stdout" in kwargs or name.endswith("check_output")
            ):
                add(
                    node,
                    "TC005",
                    "Subprocess output is captured without an evident prompt filter.",
                    "If sent to a model, retain errors and useful context; "
                    "filter routine log lines.",
                )
    return sorted(findings, key=lambda item: (item.line, item.rule))


def audit(
    path: str | Path, exclude: tuple[str, ...] = (), max_bytes: int = 1_048_576
) -> AuditResult:
    root = Path(path)
    if root.is_symlink():
        raise ValueError("Refusing to audit a symlink root")
    root = root.resolve()
    if not root.exists():
        raise ValueError(f"Audit path does not exist: {path}")
    candidates: list[Path] = []
    skipped: list[str] = []
    if root.is_file():
        if root.suffix != ".py":
            raise ValueError("This version audits Python (.py) source only")
        candidates.append(root)
        base = root.parent
    else:
        base = root
        for directory, dirs, files in os.walk(
            root, followlinks=False, onerror=lambda error: skipped.append(str(error))
        ):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in SKIP_DIRS
                and not (Path(directory) / d).is_symlink()
                and not any(
                    fnmatch.fnmatch(str((Path(directory) / d).relative_to(base)), pattern)
                    for pattern in exclude
                )
            )
            for filename in sorted(files):
                file = Path(directory) / filename
                if file.suffix == ".py":
                    candidates.append(file)
    findings = []
    scanned = 0
    for file in candidates:
        relative = file.relative_to(base).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        try:
            if file.is_symlink() or not file.is_file() or file.stat().st_size > max_bytes:
                skipped.append(
                    f"{relative}: not a regular file, symlink, or larger than {max_bytes} bytes"
                )
                continue
            with tokenize.open(file) as handle:
                source = handle.read()
            findings.extend(scan_python(source, relative))
            scanned += 1
        except (OSError, UnicodeError, SyntaxError) as exc:
            skipped.append(f"{relative}: {type(exc).__name__}")
    return AuditResult(scanned, tuple(findings), tuple(skipped))
