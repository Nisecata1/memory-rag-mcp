"""Version isolation, extensibility and comparable runs without real models."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchlib.contracts import normalize_records
from benchlib import runner
from benchlib.reporting import compare_runs, write_json
from benchlib.versions import PROTOCOL, create_retriever, discover_versions, select_versions


class VersionPackagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "versions").mkdir()

    def package(self, name="custom_v2", **overrides):
        directory = self.root / "versions" / name
        directory.mkdir()
        (directory / "adapter.py").write_text("raise RuntimeError('discovery must not import adapters')", encoding="utf-8")
        write_json(directory / "config.json", {"schema_version": 1})
        manifest = {"schema_version": 1, "id": name, "label": name, "protocol": PROTOCOL,
                    "adapter": "adapter.py", "config": "config.json", "aliases": [], "source_files": []}
        manifest.update(overrides)
        write_json(directory / "version.json", manifest)
        return directory

    def test_new_package_is_discovered_without_runner_edits_or_import_side_effects(self):
        self.package(default=True, aliases=["custom"])
        self.assertEqual(list(discover_versions(self.root)), ["custom_v2"])
        self.assertEqual(select_versions("custom,custom_v2", root=self.root)[0].id, "custom_v2")
        self.assertEqual(len(select_versions("custom,custom_v2", root=self.root)), 1)

    def test_disabled_and_template_packages_are_not_run_by_all(self):
        self.package()
        self.package("disabled_v1", enabled=False)
        (self.root / "versions/_template").mkdir()
        self.assertEqual([v.id for v in select_versions(all_versions=True, root=self.root)], ["custom_v2"])

    def test_duplicate_alias_cannot_silently_select_wrong_implementation(self):
        self.package(aliases=["project"])
        self.package("other_v2", aliases=["project"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            discover_versions(self.root)

    def test_adapter_path_cannot_escape_its_version_package(self):
        self.package(adapter="../outside.py")
        (self.root / "versions/outside.py").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out-of-package"):
            discover_versions(self.root)

    def test_version_name_and_protocol_must_be_explicit(self):
        path = self.package(protocol="different-score-protocol")
        with self.assertRaisesRegex(ValueError, "protocol"):
            discover_versions(self.root)
        item = json.loads((path / "version.json").read_text(encoding="utf-8"))
        item.update(protocol=PROTOCOL, id="other_v1")
        write_json(path / "version.json", item)
        with self.assertRaisesRegex(ValueError, "directory"):
            discover_versions(self.root)

    def test_separate_adapter_namespaces_support_local_helpers(self):
        for version, value in (("one_v1", 1), ("two_v1", 2)):
            path = self.package(version)
            (path / "helper.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
            (path / "adapter.py").write_text("from .helper import VALUE\ndef create_retriever(records, **kwargs): return VALUE\n", encoding="utf-8")
        values = [create_retriever(v, []) for v in discover_versions(self.root).values()]
        self.assertEqual(values, [1, 2])

    def test_normalization_strips_labels_from_corpus_input(self):
        record = {"record_id": "r1", "title": "session", "reference_answer": "hidden", "evidence_groups": [],
                  "messages": [{"message_id": "m1", "role": "user", "content": "raw", "is_gold": True}]}
        actual = normalize_records([record])[0]
        self.assertNotIn("reference_answer", actual)
        self.assertNotIn("evidence_groups", actual)
        self.assertNotIn("is_gold", actual["messages"][0])

    def test_source_manifest_can_track_another_backend_without_sqlite_modules(self):
        project = self.root / "project"
        project.mkdir()
        (project / "new_backend.py").write_text("# alternate backend\n", encoding="utf-8")
        result = runner.source_snapshot(project, self.root / "snapshot", files=("new_backend.py",))
        self.assertEqual(set(result["files_sha256"]), {"new_backend.py"})

    def test_explicit_selection_overrides_suite_config_and_local_parameters(self):
        path = self.root / "suite.json"
        write_json(path, {"schema_version": 1, "versions": ["project_v1"], "top_k": 5})
        args = argparse.Namespace(config=str(path), versions="bm25_v1", top_k=1, split="dev")
        runner.apply_run_config(args, ["run", "--versions=bm25_v1", "--top-k=1"])
        self.assertEqual(args.versions, "bm25_v1")
        self.assertEqual(args.top_k, 1)
        local = self.root / "local.json"
        write_json(local, {"schema_version": 1, "top_k": 9})
        runner.apply_run_config(args, ["run", "--top-k=1"], config_path=local)
        self.assertEqual(args.top_k, 1)

    def test_suite_config_wins_over_version_defaults(self):
        path = self.root / "suite.json"
        write_json(path, {"schema_version": 1, "top_k": 5})
        args = argparse.Namespace(config=str(path), top_k=1, split="dev")
        runner.apply_run_config(args, ["run"])
        local = self.root / "local.json"
        write_json(local, {"schema_version": 1, "top_k": 9})
        runner.apply_run_config(args, [], config_path=local, skip_fields=args._configured_fields)
        self.assertEqual(args.top_k, 5)

    def test_failure_in_one_version_does_not_suppress_other_versions(self):
        args = argparse.Namespace(versions="bm25_v1,project_v1", retrievers=None, all_versions=False,
                                  config=None, split="dev", adapter_options={})
        calls = []
        def fake_execute(current):
            calls.append(current.version_id)
            if current.version_id == "bm25_v1":
                raise RuntimeError("failed adapter")
            result = self.root / "runs" / current.version_id / "test-run"
            result.mkdir(parents=True)
            return result
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "execute", side_effect=fake_execute):
            with self.assertRaisesRegex(RuntimeError, "Failed versions"):
                runner.run_selected(args, [])
        self.assertEqual(calls, ["bm25_v1", "project_v1"])
        suite_path = next((self.root / "runs/suites").glob("*/suite_manifest.json"))
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        self.assertEqual(suite["status"], "completed_with_errors")


class CrossVersionComparisonTests(unittest.TestCase):
    def test_different_single_scheme_names_can_be_compared_without_renaming_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"run_id": "before", "status": "completed", "selection": {"query_count": 1},
                        "dataset_fingerprint": "data", "query_set_fingerprint": "query", "metric_fingerprint": "metrics",
                        "evaluation": {"top_k": 5}}
            write_json(root / "before/run_manifest.json", manifest)
            write_json(root / "after/run_manifest.json", {**manifest, "run_id": "after"})
            write_json(root / "before/summary.json", {"project": {"query_count": 1, "hit_at_k": .5}})
            write_json(root / "after/summary.json", {"fts": {"query_count": 1, "hit_at_k": 1.0}})
            delta = compare_runs(root / "before", root / "after", root / "comparison.md")
            self.assertEqual(delta["project -> fts"]["hit_at_k"], .5)
            write_json(root / "before/summary.json", {"bm25": {"query_count": 1, "hit_at_k": .75}, "project": {"query_count": 1, "hit_at_k": .5}})
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                compare_runs(root / "before", root / "after", root / "comparison.md")
            delta = compare_runs(root / "before", root / "after", root / "comparison.md", baseline_retriever="project")
            self.assertEqual(delta["project -> fts"]["hit_at_k"], .5)


if __name__ == "__main__":
    unittest.main()
