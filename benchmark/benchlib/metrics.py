"""Dependency-free metrics for the record/message retrieval benchmark.

Ranking metrics operate on distinct record IDs, deduplicated before the top-k
cutoff. Evidence alternatives match the joint (record_id, message_id) identity.
Latency percentiles use nearest rank: sorted_values[ceil(p * n) - 1].
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


_QUALITY_KEYS = (
    "hit_at_k",
    "recall_at_k",
    "rr_at_k",
    "evidence_coverage",
    "evidence_complete",
)


def _valid_latency(value: Any) -> float | None:
    """Reject missing, boolean, nonnumeric, negative and nonfinite timings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def score_query(
    query: dict[str, Any], prediction: dict[str, Any], k: int = 5
) -> dict[str, Any]:
    """Score one prediction without modifying either input.

    Unanswerable queries have None quality scores; they still contribute to
    resource, latency and runtime-error statistics. An errored answerable query
    receives zero for every quality score, even if partial results were returned.
    Recall and evidence scores are undefined (None) when their gold denominator
    is empty on a successful answerable query. Such missing gold annotations
    should normally be rejected by dataset validation before an evaluation run.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    answerable = bool(query.get("answerable", False))
    error = prediction.get("runtime_status", "ok") != "ok"
    messages = prediction.get("returned_messages") or []
    returned_characters = sum(
        len(message["content"])
        for message in messages
        if isinstance(message.get("content"), str)
    )
    result: dict[str, Any] = {
        "answerable": answerable,
        "category": query.get("category") or "uncategorized",
        **{key: None for key in _QUALITY_KEYS},
        "returned_characters": returned_characters,
        "returned_messages": len(messages),
        "latency_ms": _valid_latency(prediction.get("latency_ms")),
        "error": error,
    }
    if not answerable:
        return result
    if error:
        result.update({key: 0.0 for key in _QUALITY_KEYS})
        return result

    ranked_ids = list(dict.fromkeys(prediction.get("ranked_record_ids") or []))[:k]
    relevant_ids = set(query.get("relevant_record_ids") or [])
    matched_ids = relevant_ids.intersection(ranked_ids)
    result["hit_at_k"] = float(bool(matched_ids))
    result["recall_at_k"] = (
        len(matched_ids) / len(relevant_ids) if relevant_ids else None
    )
    result["rr_at_k"] = next(
        (1.0 / rank for rank, record_id in enumerate(ranked_ids, 1)
         if record_id in relevant_ids),
        0.0,
    )

    returned_identities = {
        (message.get("record_id"), message.get("message_id"))
        for message in messages
        if message.get("record_id") is not None
        and message.get("message_id") is not None
    }
    evidence_groups = query.get("evidence_groups") or []
    if evidence_groups:
        covered_groups = sum(
            any(
                (alternative.get("record_id"), alternative.get("message_id"))
                in returned_identities
                for alternative in group.get("alternatives", [])
            )
            for group in evidence_groups
        )
        result["evidence_coverage"] = covered_groups / len(evidence_groups)
        result["evidence_complete"] = covered_groups == len(evidence_groups)
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(row[key])
    ]
    return sum(values) / len(values) if values else None


def _is_answerable(row: dict[str, Any]) -> bool:
    if "answerable" in row:
        return bool(row["answerable"])
    # Permit aggregation of externally saved scored rows using only the public
    # metric fields: unanswerable rows have None for all quality metrics.
    return any(row.get(key) is not None for key in _QUALITY_KEYS)


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return values[math.ceil(percentile * len(values)) - 1]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in rows if _is_answerable(row)]
    error_count = sum(bool(row.get("error", False)) for row in rows)
    latencies = sorted(
        latency
        for row in rows
        if (latency := _valid_latency(row.get("latency_ms"))) is not None
    )
    return {
        "query_count": len(rows),
        "answerable_count": len(answerable_rows),
        "error_count": error_count,
        "error_rate": error_count / len(rows) if rows else None,
        "hit_at_k": _mean(answerable_rows, "hit_at_k"),
        "recall_at_k": _mean(answerable_rows, "recall_at_k"),
        "mrr_at_k": _mean(answerable_rows, "rr_at_k"),
        "evidence_coverage": _mean(answerable_rows, "evidence_coverage"),
        "evidence_complete": _mean(answerable_rows, "evidence_complete"),
        "returned_characters_mean": _mean(rows, "returned_characters"),
        "returned_messages_mean": _mean(rows, "returned_messages"),
        "latency_p50_ms": _nearest_rank(latencies, 0.50),
        "latency_p95_ms": _nearest_rank(latencies, 0.95),
        "invalid_latency_count": len(rows) - len(latencies),
    }


def aggregate(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute query-macro averages and independent per-category summaries.

    Quality means include answerable runtime errors as zero, and exclude
    unanswerable queries. Resource means and valid latency percentiles include
    every query, including runtime errors. A percentile's finite nonnegative
    timing sample count is query_count - invalid_latency_count.
    """
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        categories[str(row.get("category") or "uncategorized")].append(row)
    summary = _summarize(scored_rows)
    summary["by_category"] = {
        category: _summarize(rows)
        for category, rows in sorted(categories.items())
    }
    return summary
