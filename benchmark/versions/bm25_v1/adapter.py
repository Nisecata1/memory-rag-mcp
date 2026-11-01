"""BM25 v1: whole-message lexical retrieval, preserved from the first baseline."""
from __future__ import annotations
from collections import Counter
import math
import re
import time
from pathlib import Path
from typing import Any
from benchlib.contracts import normalize_records

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['\u2019][a-z0-9]+)?|[\u3400-\u9fff]", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).casefold())


def _expand_messages(record: dict[str, Any], anchors: list[str], radius: int) -> list[dict[str, str]]:
    messages = record["messages"]
    positions = {message["message_id"]: index for index, message in enumerate(messages)}
    valid = [positions[message_id] for message_id in anchors if message_id in positions]
    selected: set[int] = set()
    if valid:
        for center in valid:
            selected.update(range(max(0, center - radius), min(len(messages), center + radius + 1)))
    else:
        selected.update(range(max(0, len(messages) - (2 * radius + 1)), len(messages)))
    return [
        {"record_id": record["record_id"], "message_id": messages[index]["message_id"], "content": messages[index]["content"]}
        for index in sorted(selected)
    ]


class BM25Retriever:
    """Okapi BM25 over titles and individual messages, max-pooled by record.

    Message anchors use positive-score rank and the project's position spacing.
    The cosine score gap is deliberately NOT reused on the BM25 score scale.
    Whole-message lexical documents differ from production's BGE-token chunks;
    this difference is recorded in build_stats and no text is truncated.
    """

    def __init__(self, records: list[dict[str, Any]], *, window_radius: int = 10,
                 message_top_k: int = 3, message_score_gap: float = 0.2) -> None:
        if window_radius < 0 or message_top_k < 1:
            raise ValueError("window_radius must be non-negative and message_top_k positive.")
        started = time.perf_counter()
        self.records = normalize_records(records)
        self.window_radius = int(window_radius)
        self.message_top_k = int(message_top_k)
        self.documents: list[dict[str, Any]] = []
        for record in self.records:
            self.documents.append({"record_id": record["record_id"], "message_id": None,
                                   "position": -1, "terms": Counter(_tokens(record["title"]))})
            for position, message in enumerate(record["messages"]):
                self.documents.append({
                    "record_id": record["record_id"], "message_id": message["message_id"],
                    "position": position,
                    "terms": Counter(_tokens(f"{message['role']}: {message['content']}")),
                })
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(document["terms"].keys())
            document["length"] = sum(document["terms"].values())
        self.average_length = sum(document["length"] for document in self.documents) / max(1, len(self.documents))
        self.build_stats = {
            "backend": "stdlib_bm25_okapi", "k1": 1.5, "b": 0.75,
            "record_count": len(self.records), "document_count": len(self.documents),
            "message_count": sum(len(record["messages"]) for record in self.records),
            "max_message_lexical_tokens": max((document["length"] for document in self.documents if document["message_id"]), default=0),
            "corpus_fields": "title and original message role/content; no QA, answers, generated summaries or tags",
            "tokenizer": "casefolded alphanumeric words and individual CJK characters; no stemming/stopword removal",
            "chunk_policy": "one lexical document per whole message; differs from BGE tokenizer-based overlapping chunks; no truncation",
            "record_aggregation": "maximum title/message score",
            "anchor_policy": "positive-score rank, at most message_top_k; positions must differ by more than window_radius",
            "cosine_message_score_gap_applied": False,
            "ignored_cosine_message_score_gap": float(message_score_gap),
            "window_radius": self.window_radius, "message_top_k": self.message_top_k,
            "build_ms": (time.perf_counter() - started) * 1000,
        }

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not str(query).strip():
            raise ValueError("query is required")
        started = time.perf_counter()
        query_terms = set(_tokens(query))
        record_scores = {record["record_id"]: 0.0 for record in self.records}
        message_scores: dict[str, list[tuple[float, int, str]]] = {record["record_id"]: [] for record in self.records}
        count = len(self.documents)
        for document in self.documents:
            score = 0.0
            for term in query_terms:
                frequency = document["terms"].get(term, 0)
                if not frequency:
                    continue
                idf = math.log1p((count - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * document["length"] / (self.average_length or 1))
                score += idf * frequency * 2.5 / denominator
            record_id = document["record_id"]
            record_scores[record_id] = max(record_scores[record_id], score)
            if document["message_id"] is not None and score > 0:
                message_scores[record_id].append((score, document["position"], document["message_id"]))
        # Preserve fixture order for zero-score ties, rather than using labels.
        ordered_ids = sorted(record_scores, key=lambda record_id: -record_scores[record_id])[:max(1, min(int(top_k), 20))]
        by_id = {record["record_id"]: record for record in self.records}
        selected: dict[str, list[str]] = {}
        returned: list[dict[str, str]] = []
        for record_id in ordered_ids:
            positions: list[int] = []
            anchors: list[str] = []
            for score, position, message_id in sorted(message_scores[record_id], key=lambda item: (-item[0], item[1], item[2])):
                if any(abs(position - selected_position) <= self.window_radius for selected_position in positions):
                    continue
                positions.append(position)
                anchors.append(message_id)
                if len(anchors) == self.message_top_k:
                    break
            selected[record_id] = anchors
            returned.extend(_expand_messages(by_id[record_id], anchors, self.window_radius))
        return {
            "ranked_record_ids": ordered_ids, "scores": [record_scores[record_id] for record_id in ordered_ids],
            "selected_message_ids": selected, "returned_messages": returned,
            "latency_ms": (time.perf_counter() - started) * 1000, "runtime_status": "ok",
        }

    def close(self) -> None:
        pass



def create_retriever(records, *, project_dir, runtime_dir, window_radius, message_top_k, options):
    if options:
        raise ValueError("BM25 v1 has no additional adapter options")
    return BM25Retriever(records, window_radius=window_radius, message_top_k=message_top_k)
