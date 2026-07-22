import inspect
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

import server
import server_utils as utils


class RecordingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.texts.extend(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vectors.append([float(byte + 1) for byte in digest])
        return np.asarray(vectors, dtype="float32")


class ChatEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = root / "memory.db"
        self.data_dir = root / "data"
        self.artifact_paths = {
            "SQLITE_PATH": self.db_path,
            "DATA_DIR": self.data_dir,
            "AI_MEMORY_DIR": root,
            "TIMELINE_PATH": self.data_dir / "timeline.md",
            "EMBEDDINGS_PATH": self.data_dir / "memory_embeddings.npy",
            "INDEX_PATH": self.data_dir / "memory.faiss",
            "META_PATH": self.data_dir / "meta.json",
            "REBUILD_STATE_PATH": self.data_dir / "rebuild_state.json",
        }
        self.patches = [patch.object(utils, name, value) for name, value in self.artifact_paths.items()]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def _create_v3_chat_database(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                CREATE TABLE memory_registry (
                    id TEXT PRIMARY KEY,
                    memory_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE chat_events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    short_summary TEXT NOT NULL,
                    detailed_summary TEXT NOT NULL,
                    reference_doc_path TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_paths_json TEXT NOT NULL,
                    retrieval_fields_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO memory_registry VALUES (?, ?, ?, ?)",
                ("event-test", "chat_event", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
            )
            connection.execute(
                "INSERT INTO chat_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "event-test",
                    "legacy event",
                    "legacy summary",
                    "legacy detailed summary",
                    "",
                    '["legacy"]',
                    "[]",
                    '["title", "short_summary", "tags"]',
                ),
            )
            connection.execute("PRAGMA user_version = 3")
            connection.commit()

    def test_v3_migration_copies_legacy_summary_and_is_idempotent(self) -> None:
        self._create_v3_chat_database()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            utils.initialize_sqlite_schema(connection)
            utils.initialize_sqlite_schema(connection)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_events)").fetchall()
            }
            row = connection.execute(
                "SELECT raw_dialogue, detailed_summary FROM chat_events WHERE id = 'event-test'"
            ).fetchone()
            view_row = connection.execute(
                "SELECT detailed_summary, raw_dialogue FROM public_memory_view WHERE id = 'event-test'"
            ).fetchone()
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertIn("raw_dialogue", columns)
            self.assertEqual(row["raw_dialogue"], "legacy detailed summary")
            self.assertEqual(row["detailed_summary"], "legacy detailed summary")
            self.assertEqual(view_row["detailed_summary"], "")
            self.assertEqual(view_row["raw_dialogue"], "legacy detailed summary")

    def test_chat_payload_never_uses_raw_dialogue_for_retrieval(self) -> None:
        payload = utils.build_payload(
            memory_kind="chat_event",
            title="event title",
            short_summary="event summary",
            raw_dialogue="user: secret\n\nassistant: response",
            tags=["chat"],
            validate_reference_doc_path=False,
        )
        self.assertEqual(payload["detailed_summary"], "")
        self.assertEqual(payload["raw_dialogue"], "user: secret\n\nassistant: response")
        self.assertNotIn("raw_dialogue", payload["retrieval_fields"])
        self.assertNotIn("secret", utils.build_retrieval_text(payload))
        field_records = utils.build_field_record_entries(
            {
                "id": "event-test",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                **payload,
            }
        )
        self.assertNotIn("raw_dialogue", {entry["source_field_name"] for entry in field_records})

    def test_save_chat_event_schema_and_update_contract(self) -> None:
        parameter_names = list(inspect.signature(server.save_chatEvent).parameters)
        self.assertEqual(parameter_names[:3], ["title", "short_summary", "raw_dialogue"])
        self.assertNotIn("detailed_summary", parameter_names)

        self.assertNotIn("detailed_summary", utils.get_update_allowed_fields_for_memory_kind("chat_event"))
        self.assertIn("raw_dialogue", utils.get_update_allowed_fields_for_memory_kind("chat_event"))
        with self.assertRaises(ValueError):
            utils.validate_update_changes({"raw_dialogue": None})
        with self.assertRaises(ValueError):
            utils.build_payload(
                memory_kind="chat_event",
                title="event title",
                short_summary="event summary",
                raw_dialogue="",
                validate_reference_doc_path=False,
            )

    def test_detail_projection_preserves_type_specific_contracts(self) -> None:
        utils.ensure_sqlite_store_ready()
        chat_payload = utils.build_payload(
            memory_kind="chat_event",
            title="event title",
            short_summary="event summary",
            raw_dialogue="user: hello\nassistant: hi",
            tags=["chat"],
            validate_reference_doc_path=False,
        )
        fact_payload = utils.build_payload(
            memory_kind="fact",
            title="fact title",
            short_summary="fact summary",
            detailed_summary="fact details",
            tags=["fact"],
            validate_reference_doc_path=False,
        )
        project_payload = utils.build_payload(
            memory_kind="project_record",
            title="project title",
            short_summary="project summary",
            detailed_summary="project details",
            problem_background="background",
            analysis="analysis",
            action_steps="steps",
            validation_result="validated",
            project_id="projRegId-parent",
            tags=["project"],
            validate_reference_doc_path=False,
        )
        registry_payload = utils.build_payload(
            memory_kind="project_registry",
            title="registry title",
            overview_summary="registry overview",
            tags=["registry"],
            validate_reference_doc_path=False,
        )
        base_times = {"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
        entries = [
            {"id": "eventMemId-new", **base_times, **chat_payload},
            {"id": "pmem-legacy", **base_times, **fact_payload},
            {"id": "projMemId-new", **base_times, **project_payload},
            {"id": "projRegId-new", **base_times, **registry_payload},
            {"id": "projRegisterMemId-old", **base_times, **registry_payload},
            {"id": "proj-legacy", **base_times, **registry_payload},
        ]
        with closing(sqlite3.connect(self.db_path)) as connection:
            utils.initialize_sqlite_schema(connection)
            utils.upsert_store_entries(entries, connection=connection)
            connection.commit()

        with patch.object(utils, "fetch_public_memory_entries_by_ids", side_effect=AssertionError("detail used public view")):
            details = server.get_detail_records_by_ids(
                [
                    "pmem-legacy",
                    "eventMemId-new",
                    "projMemId-new",
                    "projRegId-new",
                    "projRegisterMemId-old",
                    "proj-legacy",
                    "missing-id",
                ]
            )
        by_id = {record["id"]: record for record in details["records"]}
        self.assertEqual(
            [record["id"] for record in details["records"]],
            [
                "pmem-legacy",
                "eventMemId-new",
                "projMemId-new",
                "projRegId-new",
                "projRegisterMemId-old",
                "proj-legacy",
            ],
        )
        self.assertEqual(details["missing_ids"], ["missing-id"])
        self.assertEqual(
            set(by_id["eventMemId-new"]),
            {"id", "memory_kind", "created_at", "updated_at", "title", "short_summary", "raw_dialogue", "reference_doc_path", "tags", "source_paths"},
        )
        self.assertIn("detailed_summary", by_id["pmem-legacy"])
        self.assertNotIn("raw_dialogue", by_id["pmem-legacy"])
        self.assertIn("problem_background", by_id["projMemId-new"])
        for registry_id in ("projRegId-new", "projRegisterMemId-old", "proj-legacy"):
            self.assertEqual(by_id[registry_id]["project_id"], registry_id)
            self.assertNotIn("retrieval_fields_json", by_id[registry_id])

    def test_detail_id_routing_supports_legacy_prefixes_and_rejects_field_records(self) -> None:
        self.assertEqual(utils.resolve_memory_kind_from_record_id("eventMemId-1"), "chat_event")
        self.assertEqual(utils.resolve_memory_kind_from_record_id("factMemId-1"), "fact")
        self.assertEqual(utils.resolve_memory_kind_from_record_id("projMemId-1"), "project_record")
        for registry_id in ("projRegId-1", "projRegisterMemId-1", "proj-legacy"):
            self.assertEqual(utils.resolve_memory_kind_from_record_id(registry_id), "project_registry")
        self.assertIsNone(utils.resolve_memory_kind_from_record_id("pmem-legacy"))
        with self.assertRaisesRegex(ValueError, "field_record is an internal index asset"):
            server.get_detail_records_by_ids(["fmem-internal"])

    def test_retrieval_signature_and_result_field_boundaries_remain_stable(self) -> None:
        self.assertEqual(
            utils.RETRIEVAL_FIELD_SIGNATURE,
            "title|short_summary|problem_background|analysis|action_steps|validation_result|tags|reference_doc_path",
        )
        self.assertNotIn("raw_dialogue", utils.SEARCH_RESULT_FIELDS)
        self.assertNotIn("detail_result_fields", utils.PROJECT_CONFIG)

    def test_chat_event_crud_smoke_keeps_raw_text_out_of_embeddings(self) -> None:
        root = self.db_path.parent
        model_dir = root / "fake-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        embedder = RecordingEmbedder()

        utils.ensure_sqlite_store_ready()
        seed_payload = utils.build_payload(
            memory_kind="fact",
            title="seed title",
            short_summary="seed summary",
            detailed_summary="seed details",
            tags=["seed"],
            validate_reference_doc_path=False,
        )
        seed_entry = {
            "id": "fact-seed",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            **seed_payload,
        }
        seed_entries = [seed_entry, *utils.build_field_record_entries(seed_entry)]
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries(seed_entries, connection=connection)
            connection.commit()
        utils.rebuild_vector_index(entries=seed_entries, embedder=embedder, model_path=str(model_dir))

        with patch.object(server, "INDEX_PATH", utils.INDEX_PATH), patch.object(server, "META_PATH", utils.META_PATH):
            saved = server.save_record(
                memory_kind="chat_event",
                title="chat title",
                short_summary="chat summary",
                raw_dialogue="user: raw-secret\nassistant: response",
                tags=["chat"],
                embedder=embedder,
                model_path=str(model_dir),
            )
            record_id = saved["id"]
            search_results = server.search_records(
                query="chat summary",
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertTrue(search_results)
            self.assertTrue(all("raw_dialogue" not in result for result in search_results))

            details = server.get_detail_records_by_ids([record_id])
            self.assertEqual(details["records"][0]["raw_dialogue"], "user: raw-secret\nassistant: response")
            self.assertNotIn("detailed_summary", details["records"][0])

            server.update_record(
                record_id=record_id,
                changes={"raw_dialogue": "user: updated-secret\nassistant: updated"},
                embedder=embedder,
                model_path=str(model_dir),
            )
            updated_details = server.get_detail_records_by_ids([record_id])
            self.assertEqual(
                updated_details["records"][0]["raw_dialogue"],
                "user: updated-secret\nassistant: updated",
            )

            deleted = server.delete_records_by_ids(
                [record_id],
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertEqual(deleted["deleted_ids"], [record_id])
            self.assertEqual(server.get_detail_records_by_ids([record_id])["missing_ids"], [record_id])

        embedded_text = "\n".join(embedder.texts)
        self.assertNotIn("raw-secret", embedded_text)
        self.assertNotIn("updated-secret", embedded_text)


if __name__ == "__main__":
    unittest.main()
