"""Moving a raw-data cache must not invalidate unchanged benchmark inputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib.dataset import _write_dataset_manifest


class DatasetManifestRelocationTests(unittest.TestCase):
    def test_cache_move_preserves_bytes_but_real_input_change_updates_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            original = {
                "dataset_id": "same-dataset-v1",
                "corpus_sha256": "original-corpus",
                "queries_sha256": "original-questions",
                "raw_source_manifest": "previous-location/source_manifest.json",
            }
            # Even formatting is part of the full-file hash in historical runs.
            original_bytes = json.dumps(original, indent=4).encode("utf-8")
            path.write_bytes(original_bytes)
            relocated = {**original, "raw_source_manifest": "data/raw/locomo/source_manifest.json"}
            _write_dataset_manifest(path, relocated)
            self.assertEqual(path.read_bytes(), original_bytes)

            changed = {**relocated, "queries_sha256": "changed-questions"}
            _write_dataset_manifest(path, changed)
            self.assertNotEqual(path.read_bytes(), original_bytes)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), changed)


if __name__ == "__main__":
    unittest.main()
