"""Reproducible synthetic retention checks, not a model-quality benchmark."""

from __future__ import annotations

import json
from statistics import median

from .core import optimize
from .model import Chunk, ContextPacket
from .tokens import TokenCounter


def demo_cases() -> list[dict]:
    logs = "\n".join(f"INFO worker heartbeat tick={i:04d} status=healthy" for i in range(120))
    logs += "\nERROR payment retry failed: request_id=abc123 timeout=30s\n"
    logs += "\n".join(f"INFO shutdown step={i:03d} complete" for i in range(40))
    verbose = json.dumps(
        {"items": [{"id": i, "status": "ready", "active": True} for i in range(35)]}, indent=4
    )
    guide = "Retries must use exponential backoff and preserve the idempotency key. " * 12
    return [
        {
            "name": "noisy-log",
            "packet": ContextPacket(
                "Diagnose the payment timeout",
                (Chunk("run-log", logs, "log", source="test-run-1"),),
                ("Never expose credentials.",),
            ).to_dict(),
            "must_keep": ["request_id=abc123", "timeout=30s", "Never expose credentials."],
        },
        {
            "name": "pretty-json",
            "packet": ContextPacket(
                "List ready items", (Chunk("items", verbose, "json", source="items-response"),)
            ).to_dict(),
            "must_keep": ['"id":34', '"active":true'],
        },
        {
            "name": "duplicate-context",
            "packet": ContextPacket(
                "How should retries work?",
                tuple(Chunk(f"guide-{i}", guide, source="docs/retries.md") for i in range(5)),
            ).to_dict(),
            "must_keep": ["idempotency key"],
        },
        {
            "name": "budgeted-retrieval",
            "packet": ContextPacket(
                "Fix payment retry idempotency",
                (
                    Chunk("constraint", "Do not change the public API.", pinned=True),
                    Chunk(
                        "payment",
                        "Payment retry idempotency requires a stable request key.",
                        source="payment.py",
                    ),
                    *(
                        Chunk(
                            f"unrelated-{i}",
                            "UI font size and page color settings. " * 40,
                            source=f"ui-{i}.md",
                        )
                        for i in range(5)
                    ),
                ),
            ).to_dict(),
            "budget": 250,
            "must_keep": ["stable request key", "Do not change the public API."],
        },
        {
            "name": "already-lean",
            "packet": ContextPacket(
                "Fix one typo", (Chunk("source", 'name = "correct"\n', "code", pinned=True),)
            ).to_dict(),
            "must_keep": ['name = "correct"'],
        },
    ]


def benchmark(counter: TokenCounter, cases: list[dict] | None = None, runs: int = 3) -> dict:
    if type(runs) is not int or not 1 <= runs <= 100:
        raise ValueError("runs must be between 1 and 100")
    cases = demo_cases() if cases is None else cases
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark requires a nonempty array of cases")
    results = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise ValueError("Each benchmark case requires a name and packet")
        expected = case.get("must_keep", [])
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise ValueError("must_keep must be a list of strings")
        packet = ContextPacket.from_dict(case.get("packet"))
        timings = []
        for _ in range(runs):
            result = optimize(packet, budget=case.get("budget"), counter=counter)
            timings.append(result.report["elapsed_ms"])
        retained = "\n".join(
            [
                result.packet.query,
                *result.packet.instructions,
                *(chunk.text for chunk in result.packet.chunks),
            ]
        )
        checks = [{"text": value, "passed": value in retained} for value in expected]
        original_by_id = {chunk.id: chunk for chunk in result.packet.chunks}
        pinned_ok = all(
            original_by_id.get(chunk.id) == chunk for chunk in packet.chunks if chunk.pinned
        )
        protected_ok = (
            result.packet.query == packet.query
            and result.packet.instructions == packet.instructions
        )
        results.append(
            {
                "name": case["name"],
                "before_tokens": result.report["before_tokens"],
                "after_tokens": result.report["after_tokens"],
                "savings_percent": result.report["savings_percent"],
                "median_ms": round(median(timings), 3),
                "retention_checks": checks,
                "protected_context_unchanged": pinned_ok and protected_ok,
                "passed": all(check["passed"] for check in checks) and pinned_ok and protected_ok,
            }
        )
    before = sum(case["before_tokens"] for case in results)
    after = sum(case["after_tokens"] for case in results)
    return {
        "kind": "synthetic-context-retention",
        "counter": counter.metadata(),
        "runs_per_case": runs,
        "before_tokens": before,
        "after_tokens": after,
        "savings_percent": round(100 * (before - after) / before, 2) if before else 0.0,
        "all_checks_passed": all(case["passed"] for case in results),
        "quality_verified": False,
        "note": "Synthetic fixtures and substring checks only. No model calls, task-success "
        "evaluation, output-token savings, or billing claims.",
        "cases": results,
    }
