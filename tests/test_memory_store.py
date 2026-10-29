from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import server_utils as utils


class MemoryStoreTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in {
            "SQLITE_PATH": self.root / "memory.db",
            "AI_MEMORY_DIR": self.root,
            "DATA_DIR": self.root / "data",
        }.items():
            self.stack.enter_context(patch.object(utils, name, value))
        self.store = utils.get_memory_store()

    def entries(self) -> list[dict]:
        common = {"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
        return [
            {
                **common,
                "id": "factMemId-transaction",
                "memory_kind": "fact",
                "title": "transaction title",
                "short_summary": "transaction summary",
                "detailed_summary": "original body",
                "tags": [],
                "source_paths": [],
                "reference_doc_path": "",
                "retrieval_fields": ["title"],
            },
            {
                **common,
                "id": "fmem-transaction",
                "memory_kind": "field_record",
                "source_memory_id": "factMemId-transaction",
                "source_field_name": "title",
                "source_field_value_text": "transaction title",
                "source_field_value_json": '"transaction title"',
            },
        ]

    def test_main_and_derived_records_commit_together(self) -> None:
        with self.store.transaction() as transaction:
            utils.upsert_store_entries(self.entries(), connection=transaction)
        found = utils.fetch_store_entries_by_ids([entry["id"] for entry in self.entries()])
        self.assertEqual([entry["id"] for entry in found], [entry["id"] for entry in self.entries()])

    def test_main_and_derived_records_rollback_together(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "abort after both writes"):
            with self.store.transaction() as transaction:
                utils.upsert_store_entries(self.entries(), connection=transaction)
                raise RuntimeError("abort after both writes")
        self.assertEqual(utils.fetch_store_entries_by_ids([entry["id"] for entry in self.entries()]), [])

    def test_factory_uses_current_explicit_database_path(self) -> None:
        first_path = self.store.path
        second_path = self.root / "another.db"
        with patch.object(utils, "SQLITE_PATH", second_path):
            other_store = utils.get_memory_store()
            with other_store.transaction() as transaction:
                utils.upsert_store_entries(self.entries(), connection=transaction)
        self.assertEqual(self.store.path, first_path)
        self.assertEqual(other_store.path, second_path)
        self.assertEqual(utils.fetch_store_entries_by_ids(["factMemId-transaction"]), [])

    def test_search_projection_preserves_project_attachment(self) -> None:
        entry = {
            **self.entries()[0],
            "id": "projMemId-attached",
            "memory_kind": "project_record",
            "project_id": "projRegId-owner",
        }
        with self.store.transaction() as transaction:
            utils.upsert_store_entries([entry], connection=transaction)
        found = utils.fetch_search_parents_by_ids([entry["id"]])
        self.assertEqual(found[0]["project_id"], entry["project_id"])
        self.assertEqual(
            found[0]["project_id"], utils.fetch_store_entries_by_ids([entry["id"]])[0]["project_id"]
        )


if __name__ == "__main__":
    unittest.main()
