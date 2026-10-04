"""Compact, static maps of Python and JS/TS repositories for code retrieval."""

from __future__ import annotations

import ast
import os
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path

from .audit import SKIP_DIRS
from .javascript import EXTENSIONS, inspect_module


@dataclass(frozen=True)
class ModuleMap:
    path: str
    imports: tuple[str, ...]
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryMap:
    modules: tuple[ModuleMap, ...]
    skipped: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "modules": [asdict(module) for module in self.modules],
            "skipped": list(self.skipped),
            "note": "Static Python AST and heuristic JS/TS map. Source was not executed.",
        }

    def render(self) -> str:
        lines: list[str] = []
        for module in self.modules:
            lines.append(f"# {module.path}")
            if module.imports:
                lines.append("imports: " + ", ".join(module.imports))
            lines.extend(f"- {symbol}" for symbol in module.symbols)
        lines.extend(f"# skipped: {item}" for item in self.skipped)
        return "\n".join(lines) + ("\n" if lines else "")


def _imports(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            names.append(prefix + (node.module or ""))
    return tuple(sorted(set(names)))


def _symbols(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            names.append(node.name)
            names.extend(
                f"{node.name}.{item.name}"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return tuple(names)


def build_repository_map(
    path: str | Path, *, max_files: int = 500, max_bytes: int = 1_048_576
) -> RepositoryMap:
    if type(max_files) is not int or max_files < 1:
        raise ValueError("max_files must be a positive integer")
    root = Path(path)
    if root.is_symlink():
        raise ValueError("Refusing to map a symlink root")
    root = root.resolve()
    if not root.exists():
        raise ValueError(f"Map path does not exist: {path}")
    if root.is_file() and root.suffix not in EXTENSIONS | {".py"}:
        raise ValueError("Map supports Python, JavaScript, and TypeScript source")
    base = root.parent if root.is_file() else root
    candidates = [root] if root.is_file() else []
    skipped: list[str] = []
    if root.is_dir():
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
            )
            candidates.extend(
                Path(directory) / name
                for name in sorted(files)
                if Path(name).suffix in EXTENSIONS | {".py"}
            )
    modules: list[ModuleMap] = []
    for file in candidates:
        relative = file.relative_to(base).as_posix()
        if len(modules) >= max_files:
            skipped.append(f"more than {max_files} Python files")
            break
        try:
            if file.is_symlink() or not file.is_file() or file.stat().st_size > max_bytes:
                skipped.append(
                    f"{relative}: not a regular file, symlink, or larger than {max_bytes} bytes"
                )
                continue
            if file.suffix == ".py":
                with tokenize.open(file) as handle:
                    tree = ast.parse(handle.read(), filename=relative)
                imports, symbols = _imports(tree), _symbols(tree)
            else:
                imports, symbols = inspect_module(file.read_text(encoding="utf-8"))
            modules.append(ModuleMap(relative, imports, symbols))
        except (OSError, SyntaxError, UnicodeError, RecursionError) as exc:
            skipped.append(f"{relative}: {type(exc).__name__}")
    return RepositoryMap(tuple(modules), tuple(skipped))
