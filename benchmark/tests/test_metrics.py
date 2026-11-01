"""Boundary tests for record ranking, joint evidence identity and aggregation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib.metrics import aggregate, score_query  # noqa: E402


def make_query(**overrides):
    query = {
        "answerable": True,
        "category": "single_evidence",
        "relevant_record_ids": ["record-a"],
        "evidence_groups": [
            {"group_id": "fact-a", "alternatives": [
                {"record_id": "record-a", "message_id": "message-1"},
            ]},
        ],
    }
    query.update(overrides)
    return query


def make_prediction(**overrides):
    prediction = {
        "ranked_record_ids": ["record-a"],
        "returned_messages": [
            {"record_id": "record-a", "message_id": "message-1", "content": "证据"},
        ],
        "runtime_status": "ok",
        "latency_ms": 10.0,
    }
    prediction.update(overrides)
    return prediction


class ScoreQueryTests(unittest.TestCase):
    def test_distinct_records_are_ranked_before_cutoff(self):
        result = score_query(
            make_query(relevant_record_ids=["record-a", "record-b", "record-c"]),
            make_prediction(ranked_record_ids=[
                "noise", "noise", "record-a", "record-b", "record-a", "record-c",
            ]),
            k=3,
        )
        self.assertEqual(result["hit_at_k"], 1.0)
        self.assertAlmostEqual(result["recall_at_k"], 2 / 3)
        self.assertEqual(result["rr_at_k"], 0.5)

    def test_relevant_result_outside_cutoff_has_zero_rank_scores(self):
        result = score_query(
            make_query(),
            make_prediction(ranked_record_ids=["noise-1", "noise-1", "noise-2", "record-a"]),
            k=2,
        )
        self.assertEqual(result["hit_at_k"], 0.0)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["rr_at_k"], 0.0)

    def test_relevant_gold_duplicates_do_not_change_denominator(self):
        result = score_query(
            make_query(relevant_record_ids=["record-a", "record-a", "record-b"]),
            make_prediction(),
        )
        self.assertEqual(result["recall_at_k"], 0.5)

    def test_all_required_groups_need_evidence_but_alternatives_are_equivalent(self):
        query = make_query(evidence_groups=[
            {"group_id": "fact-a", "alternatives": [
                {"record_id": "record-a", "message_id": "message-1"},
                {"record_id": "record-b", "message_id": "message-2"},
            ]},
            {"group_id": "fact-c", "alternatives": [
                {"record_id": "record-c", "message_id": "message-3"},
            ]},
        ])
        partial = score_query(query, make_prediction())
        self.assertEqual(partial["evidence_coverage"], 0.5)
        self.assertFalse(partial["evidence_complete"])
        complete = score_query(query, make_prediction(returned_messages=[
            {"record_id": "record-b", "message_id": "message-2", "content": "备选"},
            {"record_id": "record-c", "message_id": "message-3", "content": "必要事实"},
        ]))
        self.assertEqual(complete["evidence_coverage"], 1.0)
        self.assertTrue(complete["evidence_complete"])

    def test_message_id_on_wrong_record_is_not_evidence(self):
        result = score_query(make_query(), make_prediction(returned_messages=[
            {"record_id": "wrong-record", "message_id": "message-1", "content": "无关"},
        ]))
        self.assertEqual(result["hit_at_k"], 1.0)
        self.assertEqual(result["evidence_coverage"], 0.0)
        self.assertFalse(result["evidence_complete"])

    def test_duplicate_messages_cannot_inflate_evidence_coverage(self):
        message = make_prediction()["returned_messages"][0]
        result = score_query(make_query(), make_prediction(returned_messages=[message, message]))
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["returned_messages"], 2)
        self.assertEqual(result["returned_characters"], 4)

    def test_runtime_error_zeroes_quality_but_keeps_partial_resources(self):
        result = score_query(make_query(), make_prediction(runtime_status="error"))
        for key in ("hit_at_k", "recall_at_k", "rr_at_k", "evidence_coverage", "evidence_complete"):
            self.assertEqual(result[key], 0.0)
        self.assertTrue(result["error"])
        self.assertEqual(result["returned_messages"], 1)
        self.assertEqual(result["returned_characters"], 2)
        self.assertEqual(result["latency_ms"], 10.0)

    def test_unanswerable_quality_is_undefined_including_runtime_errors(self):
        for status in ("ok", "error"):
            with self.subTest(status=status):
                result = score_query(
                    make_query(answerable=False, relevant_record_ids=[], evidence_groups=[]),
                    make_prediction(runtime_status=status),
                )
                for key in ("hit_at_k", "recall_at_k", "rr_at_k", "evidence_coverage", "evidence_complete"):
                    self.assertIsNone(result[key])
                self.assertEqual(result["error"], status == "error")

    def test_empty_prediction_scores_zero_when_gold_exists(self):
        result = score_query(
            make_query(), make_prediction(ranked_record_ids=[], returned_messages=[])
        )
        for key in ("hit_at_k", "recall_at_k", "rr_at_k", "evidence_coverage", "evidence_complete"):
            self.assertEqual(result[key], 0)
        self.assertEqual(result["returned_characters"], 0)

    def test_empty_gold_denominators_are_undefined(self):
        result = score_query(
            make_query(relevant_record_ids=[], evidence_groups=[]), make_prediction()
        )
        self.assertIsNone(result["recall_at_k"])
        self.assertIsNone(result["evidence_coverage"])
        self.assertIsNone(result["evidence_complete"])

    def test_invalid_cutoffs_are_rejected(self):
        for cutoff in (0, -1, True, 1.5):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                score_query(make_query(), make_prediction(), k=cutoff)

    def test_scoring_does_not_modify_inputs(self):
        query, prediction = make_query(), make_prediction()
        expected_query, expected_prediction = copy.deepcopy(query), copy.deepcopy(prediction)
        score_query(query, prediction)
        self.assertEqual(query, expected_query)
        self.assertEqual(prediction, expected_prediction)


class AggregateTests(unittest.TestCase):
    def test_empty_run_has_none_means_and_no_nan(self):
        result = aggregate([])
        self.assertEqual(result["query_count"], 0)
        self.assertEqual(result["answerable_count"], 0)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["invalid_latency_count"], 0)
        self.assertEqual(result["by_category"], {})
        for key in ("error_rate", "hit_at_k", "recall_at_k", "mrr_at_k", "evidence_coverage",
                    "evidence_complete", "returned_characters_mean", "returned_messages_mean",
                    "latency_p50_ms", "latency_p95_ms"):
            self.assertIsNone(result[key])
        json.dumps(result, allow_nan=False)

    def test_answerable_error_is_included_as_zero_and_unanswerable_is_excluded(self):
        rows = [
            score_query(make_query(), make_prediction()),
            score_query(make_query(), make_prediction(runtime_status="error")),
            score_query(make_query(answerable=False), make_prediction()),
        ]
        result = aggregate(rows)
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["answerable_count"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertAlmostEqual(result["error_rate"], 1 / 3)
        for key in ("hit_at_k", "recall_at_k", "mrr_at_k", "evidence_coverage", "evidence_complete"):
            self.assertEqual(result[key], 0.5)

    def test_all_unanswerable_run_still_reports_errors_and_resources(self):
        rows = [
            score_query(make_query(answerable=False), make_prediction()),
            score_query(make_query(answerable=False), make_prediction(runtime_status="error")),
        ]
        result = aggregate(rows)
        self.assertEqual(result["answerable_count"], 0)
        self.assertEqual(result["error_rate"], 0.5)
        self.assertEqual(result["returned_characters_mean"], 2.0)
        self.assertEqual(result["returned_messages_mean"], 1.0)
        for key in ("hit_at_k", "recall_at_k", "mrr_at_k", "evidence_coverage", "evidence_complete"):
            self.assertIsNone(result[key])

    def test_nearest_rank_percentiles_include_failed_requests(self):
        rows = [
            score_query(make_query(), make_prediction(latency_ms=value))
            for value in range(1, 10)
        ]
        rows.append(score_query(
            make_query(), make_prediction(latency_ms=100, runtime_status="error")
        ))
        result = aggregate(rows)
        self.assertEqual(result["latency_p50_ms"], 5.0)
        self.assertEqual(result["latency_p95_ms"], 100.0)
        self.assertEqual(result["invalid_latency_count"], 0)
        self.assertEqual(result["hit_at_k"], 0.9)

    def test_invalid_latencies_are_counted_without_entering_percentiles(self):
        invalid = [None, float("nan"), float("inf"), -float("inf"), -1, True, "10"]
        rows = [
            score_query(make_query(), make_prediction(latency_ms=value))
            for value in invalid
        ]
        rows.append(score_query(make_query(), make_prediction(latency_ms=0)))
        result = aggregate(rows)
        self.assertEqual(result["invalid_latency_count"], len(invalid))
        self.assertEqual(result["latency_p50_ms"], 0.0)
        self.assertEqual(result["latency_p95_ms"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_no_valid_latency_has_none_percentiles(self):
        result = aggregate([score_query(make_query(), make_prediction(latency_ms=None))])
        self.assertEqual(result["invalid_latency_count"], 1)
        self.assertIsNone(result["latency_p50_ms"])
        self.assertIsNone(result["latency_p95_ms"])

    def test_category_summaries_use_their_own_denominators(self):
        result = aggregate([
            score_query(make_query(category="positive"), make_prediction()),
            score_query(make_query(category="positive"), make_prediction(runtime_status="error")),
            score_query(make_query(category="negative", answerable=False), make_prediction()),
        ])
        positive = result["by_category"]["positive"]
        negative = result["by_category"]["negative"]
        self.assertEqual(positive["query_count"], 2)
        self.assertEqual(positive["hit_at_k"], 0.5)
        self.assertEqual(negative["query_count"], 1)
        self.assertEqual(negative["answerable_count"], 0)
        self.assertIsNone(negative["hit_at_k"])
        self.assertNotIn("by_category", positive)

    def test_metrics_only_saved_rows_can_be_aggregated(self):
        row = score_query(make_query(), make_prediction())
        del row["answerable"]
        result = aggregate([row])
        self.assertEqual(result["answerable_count"], 1)
        self.assertEqual(result["hit_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
