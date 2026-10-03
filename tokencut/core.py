from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, replace
from time import perf_counter

from .filters import ERROR, compact_json, terms, trim_log
from .model import Chunk, ContextPacket
from .tokens import TokenCounter


class BudgetExceeded(ValueError):
    def __init__(self, required: int, budget: int, approximate: bool) -> None:
        self.required = required
        self.budget = budget
        self.approximate = approximate
        qualifier = "estimated " if approximate else ""
        super().__init__(
            f"Protected context needs {required} {qualifier}tokens, exceeding budget {budget}. "
            "Raise the budget or explicitly revise protected inputs. Nothing was emitted."
        )


def build_report(
    before: str,
    after: str,
    counter: TokenCounter,
    changes: list[dict],
    elapsed_ms: float,
    budget: int | None = None,
) -> dict:
    before_count, after_count = counter.count(before), counter.count(after)
    lossy = any(change.get("lossy") for change in changes)
    warnings = []
    if counter.approximate:
        warnings.append("Counts and budget checks are estimates, not a hard model-token limit.")
    if counter.warning:
        warnings.append(counter.warning)
    if lossy:
        warnings.append("Content was omitted. Review the diff and run task-level evaluations.")
    if any(change.get("action") == "invalid_json_unchanged" for change in changes):
        warnings.append("An invalid JSON chunk/output was left unchanged.")
    return {
        "before_tokens": before_count,
        "after_tokens": after_count,
        "saved_tokens": before_count - after_count,
        "savings_percent": round(100 * (before_count - after_count) / before_count, 2)
        if before_count
        else 0.0,
        "counter": counter.metadata(),
        "budget": budget,
        "budget_met": after_count <= budget if budget is not None else None,
        "elapsed_ms": round(elapsed_ms, 3),
        "risk": "lossy" if lossy else "structural" if before != after else "none",
        "quality_verified": False,
        "changes": changes,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class OptimizationResult:
    packet: ContextPacket
    report: dict
    original: ContextPacket

    def render(self) -> str:
        return self.packet.render()

    def diff(self) -> str:
        before = json.dumps(self.original.to_dict(), ensure_ascii=False, indent=2) + "\n"
        after = json.dumps(self.packet.to_dict(), ensure_ascii=False, indent=2) + "\n"
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="before.json",
                tofile="after.json",
            )
        )


def rank_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    query_terms = terms(query)
    documents = [terms(chunk.text + " " + (chunk.source or "")) for chunk in chunks]
    frequency = {word: sum(word in doc for doc in documents) for word in query_terms}
    weights = {
        word: 1 + math.log((1 + len(chunks)) / (1 + count)) for word, count in frequency.items()
    }
    scores = [sum(weights[word] for word in query_terms & doc) for doc in documents]
    return [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], i))]


def optimize(
    packet: ContextPacket,
    budget: int | None = None,
    counter: TokenCounter | None = None,
    filter_logs: bool = True,
    deduplicate: bool = True,
) -> OptimizationResult:
    """Optimize explicit context chunks; query, instructions, and pinned chunks are immutable."""
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("budget must be a positive integer")
    counter = counter or TokenCounter()
    start = perf_counter()
    changes: list[dict] = []
    chunks: list[Chunk] = []
    seen = {
        (chunk.kind, chunk.source, chunk.text): chunk.id for chunk in packet.chunks if chunk.pinned
    }
    essential: set[str] = {chunk.id for chunk in packet.chunks if chunk.pinned}
    for chunk in packet.chunks:
        if chunk.pinned:
            chunks.append(chunk)
            continue
        key = (chunk.kind, chunk.source, chunk.text)
        if deduplicate and key in seen:
            changes.append(
                {"id": chunk.id, "action": "deduplicate", "kept": seen[key], "lossy": False}
            )
            continue
        seen[key] = chunk.id
        text = chunk.text
        action = None
        detail: dict = {}
        if chunk.kind == "log":
            if ERROR.search(text):
                essential.add(chunk.id)
            if filter_logs:
                filtered = trim_log(text, packet.query)
                text = filtered.text
                action = "filter_log"
                detail = {"removed_lines": filtered.removed_lines, "lossy": True}
        elif chunk.kind == "json":
            try:
                text = compact_json(text)
                action = "compact_json"
                detail = {"lossy": False}
            except (ValueError, RecursionError):
                changes.append({"id": chunk.id, "action": "invalid_json_unchanged", "lossy": False})
        if action and counter.count(text) < counter.count(chunk.text):
            chunks.append(replace(chunk, text=text))
            changes.append({"id": chunk.id, "action": action, **detail})
        else:
            chunks.append(chunk)
    candidate = replace(packet, chunks=tuple(chunks))
    if budget is not None and counter.count(candidate.render()) > budget:
        selected = [chunk for chunk in chunks if chunk.id in essential]
        required = counter.count(replace(packet, chunks=tuple(selected)).render())
        if required > budget:
            raise BudgetExceeded(required, budget, counter.approximate)
        selected_ids = {chunk.id for chunk in selected}
        for chunk in rank_chunks(packet.query, [c for c in chunks if c.id not in essential]):
            trial_ids = selected_ids | {chunk.id}
            trial = replace(packet, chunks=tuple(c for c in chunks if c.id in trial_ids))
            if counter.count(trial.render()) <= budget:
                selected_ids.add(chunk.id)
            else:
                changes.append({"id": chunk.id, "action": "drop_for_budget", "lossy": True})
        candidate = replace(packet, chunks=tuple(c for c in chunks if c.id in selected_ids))
    # Recount the complete rendered payload; token counts are not generally additive.
    if budget is not None and counter.count(candidate.render()) > budget:
        raise BudgetExceeded(counter.count(candidate.render()), budget, counter.approximate)
    if counter.count(candidate.render()) > counter.count(packet.render()):
        candidate = packet
        changes = []
    report = build_report(
        packet.render(),
        candidate.render(),
        counter,
        changes,
        (perf_counter() - start) * 1000,
        budget,
    )
    return OptimizationResult(candidate, report, packet)
