"""Source snapshots contain every service module without importing it."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_benchmark import PROJECT_SOURCE_FILES, hash_file, source_snapshot


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        for relative in PROJECT_SOURCE_FILES:
            target = self.project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"unimportable snapshot content: {relative}\n", encoding="utf-8")
        self.git = patch("run_benchmark.subprocess.run", return_value=SimpleNamespace(stdout="fixture-commit\n", returncode=0))
        self.git.start()
        self.addCleanup(self.git.stop)

    def test_snapshot_copies_and_hashes_new_service_modules(self) -> None:
        destination = self.root / "snapshot"
        manifest = source_snapshot(self.project, destination)
        self.assertEqual(set(manifest["files_sha256"]), set(PROJECT_SOURCE_FILES))
        for relative in PROJECT_SOURCE_FILES:
            with self.subTest(relative=relative):
                self.assertEqual((destination / relative).read_bytes(), (self.project / relative).read_bytes())
                self.assertEqual(manifest["files_sha256"][relative], hash_file(destination / relative))
        self.assertEqual(manifest["git_commit"], "fixture-commit")

    def test_change_in_each_new_module_changes_snapshot_fingerprint(self) -> None:
        baseline = source_snapshot(self.project, self.root / "baseline")
        for index, module in enumerate(("memory_domain.py", "memory_store.py", "memory_access.py")):
            with self.subTest(module=module):
                source = self.project / module
                original = source.read_text(encoding="utf-8")
                source.write_text(original + "changed behavior\n", encoding="utf-8")
                changed = source_snapshot(self.project, self.root / f"changed-{index}")
                self.assertNotEqual(changed["fingerprint"], baseline["fingerprint"])
                self.assertNotEqual(changed["files_sha256"][module], baseline["files_sha256"][module])
                source.write_text(original, encoding="utf-8")

    def test_missing_module_fails_before_creating_partial_snapshot(self) -> None:
        (self.project / "memory_store.py").unlink()
        destination = self.root / "snapshot"
        with self.assertRaisesRegex(FileNotFoundError, "memory_store.py"):
            source_snapshot(self.project, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
