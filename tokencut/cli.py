from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import __version__
from .audit import audit
from .bench import benchmark
from .core import BudgetExceeded, build_report, optimize
from .filters import trim_log
from .model import ContextPacket, serialize
from .request import optimize_request
from .tokens import TokenCounter

MAX_INPUT = 16 * 1024 * 1024


def read_text(path: str) -> str:
    if path == "-":
        text = sys.stdin.read(MAX_INPUT + 1)
    else:
        if Path(path).stat().st_size > MAX_INPUT:
            raise ValueError("Input exceeds the 16 MiB limit")
        with open(path, encoding="utf-8") as handle:
            text = handle.read(MAX_INPUT + 1)
    if len(text.encode("utf-8")) > MAX_INPUT:
        raise ValueError("Input exceeds the 16 MiB limit")
    return text


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str) -> object:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-JSON numeric constant: {value}")

    return json.loads(read_text(path), object_pairs_hook=_unique_object, parse_constant=invalid)


def preflight(paths: list[str | None], input_path: str | None, force: bool) -> None:
    resolved: set[Path] = set()
    source = Path(input_path).resolve() if input_path and input_path != "-" else None
    for path in paths:
        if not path or path == "-":
            continue
        target = Path(path).resolve()
        if target == source or target in resolved:
            raise ValueError("Output paths must be distinct and must not overwrite the input")
        if Path(path).is_symlink():
            raise ValueError(f"Refusing to write through a symlink: {path}")
        if target.exists() and not force:
            raise ValueError(f"Output exists: {path}; pass --force to replace it")
        if not target.parent.is_dir() or target.is_dir():
            raise ValueError(f"Output must name a file in an existing directory: {path}")
        resolved.add(target)


def write_text(path: str | None, text: str, force: bool = False) -> None:
    if not path or path == "-":
        sys.stdout.write(text)
        return
    target = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=".tokencut-", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
        if force:
            os.replace(temporary, target)
        else:
            # Exclusive creation prevents a file appearing after preflight from being overwritten.
            os.link(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def summary(report: dict) -> None:
    qualifier = "estimated " if report["counter"]["approximate"] else ""
    print(
        f"TokenCut: {report['before_tokens']:,} -> {report['after_tokens']:,} {qualifier}tokens "
        f"({report['savings_percent']:.1f}% saved). Risk: {report['risk']}.",
        file=sys.stderr,
    )
    for warning in report["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="tokencut", description="Inspectable agent token optimization"
    )
    root.add_argument("--version", action="version", version=f"TokenCut {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--counter", choices=["auto", "estimate", "tiktoken"], default="auto")
    common.add_argument("--encoding", default="cl100k_base")
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("-o", "--output", help="Output path; stdout when omitted")
    output.add_argument("--force", action="store_true", help="Replace existing output files")
    profile = commands.add_parser(
        "profile", parents=[common], help="Count a file or context packet"
    )
    profile.add_argument("input", help="File path or - for stdin")
    profile.add_argument("--packet", action="store_true", help="Count canonical context JSON")
    profile.add_argument("--format", choices=["text", "json"], default="text")
    opt = commands.add_parser(
        "optimize", parents=[common, output], help="Optimize a context packet"
    )
    opt.add_argument("input")
    opt.add_argument("--budget", type=int)
    opt.add_argument("--report", help="Write a separate JSON report")
    opt.add_argument("--diff", help="Write a unified context diff; may contain sensitive input")
    opt.add_argument("--no-log-filter", action="store_true")
    opt.add_argument("--no-dedupe", action="store_true")
    req = commands.add_parser(
        "request", parents=[common, output], help="Filter allowlisted tool outputs"
    )
    req.add_argument("input")
    req.add_argument("--tool", action="append", default=[], metavar="NAME=log|json", required=True)
    req.add_argument("--report")
    log = commands.add_parser(
        "trim-log", parents=[common, output], help="Keep errors and nearby log lines"
    )
    log.add_argument("input")
    log.add_argument("--query", default="")
    log.add_argument("--context", type=int, default=2)
    log.add_argument("--tail", type=int, default=6)
    scan = commands.add_parser(
        "audit", parents=[output], help="Audit Python agent source without running it"
    )
    scan.add_argument("path")
    scan.add_argument("--exclude", action="append", default=[], help="Relative path glob to skip")
    scan.add_argument("--format", choices=["text", "json"], default="text")
    scan.add_argument("--fail-on", choices=["never", "warning", "info"], default="never")
    bench = commands.add_parser(
        "bench", parents=[common, output], help="Run synthetic retention benchmarks"
    )
    bench.add_argument("--suite", help="Custom JSON array of name/packet/budget/must_keep cases")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--format", choices=["text", "json"], default="text")
    evaluation = commands.add_parser(
        "eval",
        parents=[common, output],
        help="Compare a tool-using agent before and after optimization",
    )
    evaluation.add_argument("--suite", help="JSON task suite; included demo cases when omitted")
    evaluation.add_argument("--model", help="Your model name; required for live runs")
    evaluation.add_argument(
        "--dry-run", action="store_true", help="Show plan without making model calls"
    )
    evaluation.add_argument("--repeats", type=int, default=1)
    evaluation.add_argument("--max-steps", type=int, default=4)
    evaluation.add_argument("--max-completion-tokens", type=int, default=1024)
    evaluation.add_argument("--target-savings", type=float, default=50)
    evaluation.add_argument("--timeout", type=float, default=30)
    evaluation.add_argument(
        "--include-content", action="store_true", help="Include sensitive answers and messages"
    )
    evaluation.add_argument("--format", choices=["text", "json"], default="text")
    return root


def run(args: argparse.Namespace) -> int:
    if args.command == "audit":
        preflight([args.output], args.path, args.force)
        result = audit(args.path, exclude=tuple(args.exclude))
        if args.format == "json":
            text = json_text(result.to_dict())
        else:
            lines = [
                f"Audited {result.files_scanned} Python files; {len(result.findings)} findings.",
                "Heuristics only. No code was executed or changed.",
            ]
            for item in result.findings:
                lines.extend(
                    [
                        f"\n{item.path}:{item.line} {item.rule} [{item.severity}] {item.message}",
                        f"  Suggestion: {item.suggestion}",
                    ]
                )
            lines.extend(f"Skipped: {item}" for item in result.skipped)
            text = "\n".join(lines) + "\n"
        write_text(args.output, text, args.force)
        threshold = {"warning"} if args.fail_on == "warning" else {"warning", "info"}
        return int(
            args.fail_on != "never"
            and (bool(result.skipped) or any(f.severity in threshold for f in result.findings))
        )
    counter = TokenCounter(
        "estimate" if args.command == "eval" and args.dry_run else args.counter,
        args.encoding,
    )
    if args.command == "profile":
        if args.packet:
            packet = ContextPacket.from_dict(read_json(args.input))
            count = counter.count(packet.render())
            chunks = [
                {
                    "id": c.id,
                    "kind": c.kind,
                    "pinned": c.pinned,
                    "text_tokens": counter.count(c.text),
                }
                for c in packet.chunks
            ]
        else:
            count, chunks = counter.count(read_text(args.input)), []
        report = {
            "tokens": count,
            "counter": counter.metadata(),
            "chunks": chunks,
            "note": "Chunk text counts exclude framing and are not additive.",
        }
        if args.format == "json":
            sys.stdout.write(json_text(report))
        else:
            print(f"{count:,} {'estimated ' if counter.approximate else ''}tokens ({counter.name})")
            for chunk in chunks:
                print(f"  {chunk['id']}: {chunk['text_tokens']:,} text tokens [{chunk['kind']}]")
        return 0
    paths = [args.output, getattr(args, "report", None), getattr(args, "diff", None)]
    if any(path == "-" for path in paths[1:]):
        raise ValueError("Report and diff require file paths, not stdout")
    preflight(paths, getattr(args, "input", None) or getattr(args, "suite", None), args.force)
    if args.command == "eval":
        from .demo_agent import demo_agent_cases
        from .evaluation import evaluate, load_cases, plan_evaluation
        from .providers import OpenAIChatClient

        cases = load_cases(read_json(args.suite)) if args.suite else demo_agent_cases()
        options = {
            "model": args.model,
            "repeats": args.repeats,
            "max_steps": args.max_steps,
            "max_completion_tokens": args.max_completion_tokens,
            "target_savings": args.target_savings,
        }
        planned = plan_evaluation(cases, **options)
        if args.dry_run:
            result = planned
        else:
            if not args.model or not args.model.strip():
                raise ValueError("Live evaluation requires --model with your model's name")
            client = OpenAIChatClient(timeout=args.timeout)
            try:
                result = evaluate(
                    cases, client, counter=counter, include_content=args.include_content, **options
                )
            finally:
                client.close()
        if args.format == "json":
            text = json_text(result)
        elif args.dry_run:
            text = (
                f"Dry run: {len(cases)} cases, {args.repeats} repeat(s), baseline + optimized.\n"
                f"Maximum model calls: {planned['maximum_model_calls']}. "
                f"Completion limit: {args.max_completion_tokens} tokens per call.\n"
                f"Savings target: {args.target_savings:g}%. No model calls were made.\n"
            )
        else:
            rows = [
                "Paired agent evaluation (provider-reported usage)",
                "Variant      Passed     Calls     Total tokens     Tokens/success",
            ]
            for variant, totals in result["summary"].items():
                tokens = str(totals["usage"]["total_tokens"]) if totals["usage"] else "unknown"
                per_success = totals["tokens_per_successful_task"]
                value = f"{per_success:.1f}" if per_success is not None else "unknown"
                rows.append(
                    f"{variant:<12} {totals['successful_runs']}/{totals['runs']:<8} "
                    f"{totals['model_calls']:<9} {tokens:<16} {value}"
                )
            savings = result["total_token_savings_percent"]
            rows.append(
                f"Total token savings: {savings}%"
                if savings is not None
                else "Total token savings: unavailable (incomplete usage)"
            )
            rows.append(f"Target gate: {'PASS' if result['gate_passed'] else 'FAIL'}")
            rows.append(result["note"])
            text = "\n".join(rows) + "\n"
        write_text(args.output, text, args.force)
        return 0 if args.dry_run else int(not result["gate_passed"])
    if args.command == "optimize":
        packet = ContextPacket.from_dict(read_json(args.input))
        result = optimize(
            packet,
            budget=args.budget,
            counter=counter,
            filter_logs=not args.no_log_filter,
            deduplicate=not args.no_dedupe,
        )
        if args.report:
            write_text(args.report, json_text(result.report), args.force)
        if args.diff:
            write_text(args.diff, result.diff(), args.force)
        write_text(args.output, result.render(), args.force)
        summary(result.report)
    elif args.command == "request":
        kinds = {}
        for item in args.tool:
            name, separator, kind = item.partition("=")
            if not separator or not name or kind not in {"log", "json"} or name in kinds:
                raise ValueError("Each --tool must be a unique NAME=log or NAME=json")
            kinds[name] = kind
        result = optimize_request(read_json(args.input), tool_kinds=kinds, counter=counter)
        if args.report:
            write_text(args.report, json_text(result.report), args.force)
        write_text(args.output, serialize(result.request), args.force)
        summary(result.report)
    elif args.command == "trim-log":
        before = read_text(args.input)
        filtered = trim_log(before, args.query, args.context, args.tail)
        after = filtered.text if counter.count(filtered.text) < counter.count(before) else before
        changes = [{"action": "filter_log", "lossy": True}] if after != before else []
        write_text(args.output, after, args.force)
        summary(build_report(before, after, counter, changes, 0))
    elif args.command == "bench":
        result = benchmark(counter, read_json(args.suite) if args.suite else None, args.runs)
        if args.format == "json":
            text = json_text(result)
        else:
            rows = [
                "Synthetic retention benchmark (not model-quality evaluation)",
                f"Counter: {counter.name}",
                "Case                       Before    After    Saved    Checks",
            ]
            for case in result["cases"]:
                rows.append(
                    f"{case['name']:<26} {case['before_tokens']:>7}  {case['after_tokens']:>7}  "
                    f"{case['savings_percent']:>6.1f}%   {'PASS' if case['passed'] else 'FAIL'}"
                )
            rows.append(f"Aggregate: {result['savings_percent']:.1f}% fewer context tokens.")
            rows.append(result["note"])
            text = "\n".join(rows) + "\n"
        write_text(args.output, text, args.force)
        return int(not result["all_checks_passed"])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except BudgetExceeded as exc:
        print(f"TokenCut budget blocked: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f"TokenCut error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
