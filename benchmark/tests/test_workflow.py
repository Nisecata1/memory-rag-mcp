from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchlib.dataset import validate_dataset
from benchlib.reporting import compare_runs, write_json
from benchlib.retrievers import BM25Retriever, ProjectRetriever
from run_benchmark import apply_run_config, validate_prediction


def fixture():
    records = [
        {"scenario_id": "a", "record_id": "r1", "title": "Session 1", "messages": [{"message_id": "D1:1", "role": "user", "content": "Alice moved to Berlin."}]},
        {"scenario_id": "b", "record_id": "r2", "title": "Session 2", "messages": [{"message_id": "D1:1", "role": "user", "content": "Bob stayed in Oslo."}]},
    ]
    query = {"query_id": "q1", "scenario_id": "a", "split": "dev", "category": "single_hop", "query": "Where did Alice move?", "answerable": True, "relevant_record_ids": ["r1"], "evidence_groups": [{"group_id": "g", "alternatives": [{"record_id": "r1", "message_id": "D1:1"}]}]}
    return records, query


class WorkflowTests(unittest.TestCase):
    def test_scenario_split_cannot_leak_same_conversation(self):
        records, query = fixture()
        other = {**query, "query_id": "q2", "split": "test"}
        with self.assertRaises(ValueError):
            validate_dataset(records, [query, other])

    def test_evidence_cannot_cross_scenarios(self):
        records, query = fixture()
        query["evidence_groups"][0]["alternatives"][0]["record_id"] = "r2"
        query["relevant_record_ids"] = ["r2"]
        with self.assertRaises(ValueError):
            validate_dataset(records, [query])

    def test_bm25_preserves_original_evidence_content(self):
        records, _ = fixture()
        result = BM25Retriever(records, window_radius=0).search("Alice Berlin", top_k=1)
        self.assertEqual(result["ranked_record_ids"], ["r1"])
        validate_prediction(result, records, 1)

    def test_legal_id_without_actual_original_text_is_rejected(self):
        records, _ = fixture()
        result = {"ranked_record_ids": ["r1"], "returned_messages": [{"record_id": "r1", "message_id": "D1:1", "content": "invented evidence"}]}
        with self.assertRaises(ValueError):
            validate_prediction(result, records, 1)

    def test_returned_evidence_must_belong_to_ranked_records(self):
        records, _ = fixture()
        result = {"ranked_record_ids": ["r1"], "returned_messages": [{"record_id": "r2", **records[1]["messages"][0]}]}
        with self.assertRaises(ValueError):
            validate_prediction(result, records, 1)

    def test_project_adapter_rejects_production_directory_before_import(self):
        with self.assertRaises(ValueError):
            ProjectRetriever([], project_dir=ROOT.parent, runtime_dir=ROOT.parent.parent / "memory-rag-mcp-data")

    def test_explicit_cli_flags_override_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            write_json(path, {"schema_version": 1, "top_k": 5, "window_radius": 10, "retrievers": ["bm25"]})
            args = argparse.Namespace(config=str(path), top_k=1, window_radius=0, split="dev")
            apply_run_config(args, ["run", "--top-k=1"])
            self.assertEqual(args.top_k, 1)
            self.assertEqual(args.window_radius, 10)
            self.assertEqual(args.retrievers, "bm25")

    def test_config_does_not_accept_boolean_for_integer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            write_json(path, {"schema_version": 1, "top_k": True})
            with self.assertRaises(ValueError):
                apply_run_config(argparse.Namespace(config=str(path)), [])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.before = self.root / "before"
        self.after = self.root / "after"
        self.manifest = {"run_id": "before", "status": "completed", "selection": {"query_count": 1}, "dataset_fingerprint": "same-data", "query_set_fingerprint": "same-queries", "metric_fingerprint": "same-metrics", "evaluation": {"top_k": 5}}
        write_json(self.before / "run_manifest.json", self.manifest)
        write_json(self.after / "run_manifest.json", {**self.manifest, "run_id": "after"})
        write_json(self.before / "summary.json", {"bm25": {"query_count": 1, "hit_at_k": 0.5}})
        write_json(self.after / "summary.json", {"bm25": {"query_count": 1, "hit_at_k": 0.75}})

    def tearDown(self):
        self.temp.cleanup()

    def test_compatible_runs_produce_measured_delta(self):
        delta = compare_runs(self.before, self.after, self.root / "comparison.md")
        self.assertEqual(delta["bm25"]["hit_at_k"], 0.25)
        self.assertTrue((self.root / "comparison.json").is_file())

    def test_changed_inputs_or_scoring_cannot_silently_compare(self):
        for field in ("dataset_fingerprint", "query_set_fingerprint", "metric_fingerprint"):
            with self.subTest(field=field):
                write_json(self.after / "run_manifest.json", {**self.manifest, field: "changed"})
                with self.assertRaises(ValueError):
                    compare_runs(self.before, self.after, self.root / "comparison.md")

    def test_different_cutoffs_cannot_silently_compare(self):
        write_json(self.after / "run_manifest.json", {**self.manifest, "evaluation": {"top_k": 1}})
        with self.assertRaises(ValueError):
            compare_runs(self.before, self.after, self.root / "comparison.md")

    def test_incomplete_or_invalid_runs_cannot_silently_compare(self):
        for status in ("running", "interrupted_or_failed", "invalid_source_changed"):
            with self.subTest(status=status):
                write_json(self.after / "run_manifest.json", {**self.manifest, "status": status})
                with self.assertRaises(ValueError):
                    compare_runs(self.before, self.after, self.root / "comparison.md")

    def test_missing_query_scores_cannot_silently_compare(self):
        write_json(self.after / "summary.json", {"bm25": {"query_count": 0, "hit_at_k": 1.0}})
        with self.assertRaises(ValueError):
            compare_runs(self.before, self.after, self.root / "comparison.md")


if __name__ == "__main__":
    unittest.main()
