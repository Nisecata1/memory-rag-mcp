import inspect
import hashlib
import json
import os
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
        self.tokenizer = CharacterTokenizer(model_max_length=512)

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.texts.extend(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vectors.append([float(byte + 1) for byte in digest])
        return np.asarray(vectors, dtype="float32")


class CharacterTokenizer:
    def __init__(self, model_max_length: int) -> None:
        self.model_max_length = model_max_length

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 2

    def encode(self, text: str, add_special_tokens: bool = False, **_: object) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def build_dialogue(user_text: str, assistant_text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "messages": [
            {"message_id": "msg-user", "role": "user", "content": user_text},
            {"message_id": "msg-assistant", "role": "assistant", "content": assistant_text},
        ],
    }


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
            "WRITE_LOCK_PATH": self.data_dir / ".memory-rag-write.lock",
        }
        self.patches = [patch.object(utils, name, value) for name, value in self.artifact_paths.items()]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def _create_fake_model(self, name: str = "fake-model") -> Path:
        model_dir = self.db_path.parent / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return model_dir

    def _store_and_index(
        self,
        entries: list[dict[str, object]],
        embedder: RecordingEmbedder,
        model_dir: Path,
    ) -> None:
        utils.ensure_sqlite_store_ready()
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries(entries, connection=connection)
            connection.commit()
        utils.rebuild_vector_index(
            entries=entries,
            embedder=embedder,
            model_path=str(model_dir),
        )

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

    def test_chat_payload_indexes_structured_messages_only_through_field_records(self) -> None:
        dialogue = build_dialogue("secret question", "secret response")
        payload = utils.build_payload(
            memory_kind="chat_event",
            title="event title",
            short_summary="event summary",
            raw_dialogue=dialogue,
            tags=["chat"],
            validate_reference_doc_path=False,
        )
        self.assertEqual(payload["detailed_summary"], "")
        self.assertEqual(payload["raw_dialogue"], dialogue)
        self.assertNotIn("raw_dialogue", payload["retrieval_fields"])
        self.assertNotIn("secret", utils.build_retrieval_text(payload))
        field_records = utils.build_field_record_entries(
            {
                "id": "event-test",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                **payload,
            },
            dialogue_tokenizer=CharacterTokenizer(model_max_length=512),
            model_path=str(self.db_path.parent),
        )
        raw_records = [
            entry
            for entry in field_records
            if entry["source_field_name"] == "raw_dialogue"
        ]
        self.assertEqual(len(raw_records), 2)
        self.assertEqual(
            [entry["source_field_value_text"] for entry in raw_records],
            ["user: secret question", "assistant: secret response"],
        )
        self.assertEqual(
            utils.normalize_store_entry(raw_records[0]),
            raw_records[0],
        )

        legacy_payload = utils.build_payload(
            memory_kind="chat_event",
            title="legacy event",
            short_summary="legacy summary",
            raw_dialogue="user: legacy\nassistant: text",
            validate_reference_doc_path=False,
            allow_legacy_raw_dialogue=True,
        )
        legacy_records = utils.build_field_record_entries(
            {
                "id": "event-legacy",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                **legacy_payload,
            }
        )
        self.assertNotIn("raw_dialogue", {entry["source_field_name"] for entry in legacy_records})

    def test_save_chat_event_schema_and_update_contract(self) -> None:
        parameter_names = list(inspect.signature(server.save_chatEvent).parameters)
        self.assertEqual(parameter_names[:3], ["title", "short_summary", "raw_dialogue"])
        self.assertNotIn("detailed_summary", parameter_names)
        registered_tool = server.server._tool_manager._tools["save_chatEvent"]
        self.assertIs(registered_tool.fn, server.save_chatEvent)

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
        with self.assertRaisesRegex(ValueError, "structured object"):
            utils.build_payload(
                memory_kind="chat_event",
                title="event title",
                short_summary="event summary",
                raw_dialogue="user: old input",
                validate_reference_doc_path=False,
            )
        with self.assertRaisesRegex(ValueError, "role must be one of"):
            utils.build_payload(
                memory_kind="chat_event",
                title="event title",
                short_summary="event summary",
                raw_dialogue={"messages": [{"role": "tool", "content": "noise"}]},
                validate_reference_doc_path=False,
            )

        generated_once = utils.normalize_raw_dialogue(
            {"messages": [{"role": "user", "content": "stable id"}]}
        )
        generated_twice = utils.normalize_raw_dialogue(
            {"messages": [{"role": "user", "content": "stable id"}]}
        )
        self.assertEqual(
            generated_once["messages"][0]["message_id"],
            generated_twice["messages"][0]["message_id"],
        )

    def test_detail_projection_preserves_type_specific_contracts(self) -> None:
        utils.ensure_sqlite_store_ready()
        chat_payload = utils.build_payload(
            memory_kind="chat_event",
            title="event title",
            short_summary="event summary",
            raw_dialogue=build_dialogue("hello", "hi"),
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
        self.assertEqual(
            utils.FIELD_RECORD_DERIVATION_SIGNATURE,
            "chat_dialogue_v1|roles=user,assistant|chunk_max=1024|chunk_overlap=128",
        )
        self.assertIsNone(utils.SEARCH_MIN_SIMILARITY)

    def test_min_similarity_config_validation(self) -> None:
        self.assertIsNone(utils.parse_min_similarity(None))
        self.assertEqual(utils.parse_min_similarity(0), 0.0)
        self.assertEqual(utils.parse_min_similarity(1), 1.0)
        for invalid_value in (True, False, "0.5", -0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value), self.assertRaises(RuntimeError):
                utils.parse_min_similarity(invalid_value)

    def test_min_similarity_filters_after_entity_fold(self) -> None:
        class FakeIndex:
            def search(self, query_vector: np.ndarray, search_count: int) -> tuple[np.ndarray, np.ndarray]:
                del query_vector, search_count
                return (
                    np.asarray([[0.95, 0.60, 0.75]], dtype="float32"),
                    np.asarray([[0, 1, 2]], dtype="int64"),
                )

        parent_a = {
            "id": "eventMemId-parent-a",
            "memory_kind": "chat_event",
            "title": "parent a",
            "short_summary": "summary a",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "tags": [],
            "reference_doc_path": "",
        }
        parent_b = {
            "id": "eventMemId-parent-b",
            "memory_kind": "chat_event",
            "title": "parent b",
            "short_summary": "summary b",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "tags": [],
            "reference_doc_path": "",
        }
        field_a_high = {
            "id": "fmem-a-high",
            "memory_kind": "field_record",
            "source_memory_id": parent_a["id"],
            "source_field_name": "raw_dialogue",
        }
        field_a_low = {
            "id": "fmem-a-low",
            "memory_kind": "field_record",
            "source_memory_id": parent_a["id"],
            "source_field_name": "raw_dialogue",
        }
        field_b = {
            "id": "fmem-b",
            "memory_kind": "field_record",
            "source_memory_id": parent_b["id"],
            "source_field_name": "raw_dialogue",
        }
        direct_entries = [field_a_high, field_a_low, field_b]
        meta_path = self.data_dir / "meta.json"
        index_path = self.data_dir / "memory.faiss"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "row_to_id": [entry["id"] for entry in direct_entries],
                    "embedding_model_fingerprint": "fingerprint",
                    "retrieval_field_signature": utils.RETRIEVAL_FIELD_SIGNATURE,
                    "field_record_derivation_signature": utils.FIELD_RECORD_DERIVATION_SIGNATURE,
                    "dim": 2,
                }
            ),
            encoding="utf-8",
        )
        index_path.touch()

        with patch.object(server, "META_PATH", meta_path), patch.object(server, "INDEX_PATH", index_path), patch.object(
            server, "ensure_sqlite_store_ready"
        ), patch.object(
            server, "build_embedding_model_fingerprint", return_value="fingerprint"
        ), patch.object(server, "encode_texts", return_value=np.ones((1, 2), dtype="float32")), patch.object(
            server.faiss, "read_index", return_value=FakeIndex()
        ), patch.object(server, "fetch_search_entries_by_ids", return_value=direct_entries), patch.object(
            server, "fetch_search_parents_by_ids", return_value=[parent_a, parent_b]
        ):
            with patch.object(server, "SEARCH_MIN_SIMILARITY", 0.90):
                results = server.search_records("query", top_k=5, embedder=object(), model_path="model")
            with patch.object(server, "SEARCH_MIN_SIMILARITY", 0.75):
                boundary_results = server.search_records("query", top_k=5, embedder=object(), model_path="model")
            with patch.object(server, "SEARCH_MIN_SIMILARITY", None):
                disabled_results = server.search_records("query", top_k=5, embedder=object(), model_path="model")

        self.assertEqual([result["id"] for result in results], [parent_a["id"]])
        self.assertAlmostEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["matched_fields"], ["raw_dialogue"])
        self.assertEqual(
            [result["id"] for result in boundary_results],
            [parent_a["id"], parent_b["id"]],
        )
        self.assertEqual(
            [result["id"] for result in disabled_results],
            [parent_a["id"], parent_b["id"]],
        )

    def test_dialogue_chunking_uses_config_and_model_token_limits(self) -> None:
        model_dir = self.db_path.parent / "token-model"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text(
            '{"model_max_length": 12}',
            encoding="utf-8",
        )
        tokenizer = CharacterTokenizer(model_max_length=20)
        with patch.object(utils, "CHAT_EVENT_CHUNK_MAX_TOKENS", 16), patch.object(
            utils,
            "CHAT_EVENT_CHUNK_OVERLAP_TOKENS",
            0,
        ):
            chunks = utils.split_dialogue_message_content(
                "user",
                "abcdefghij",
                tokenizer,
                str(model_dir),
            )
        # 12 model tokens - 2 special tokens - 6 role-prefix tokens = 4 content tokens.
        self.assertEqual(chunks, ["abcd", "efgh", "ij"])

        source_entry = {
            "id": "event-chunked",
            "memory_kind": "chat_event",
            "raw_dialogue": {
                "schema_version": 1,
                "messages": [
                    {"message_id": "msg-long", "role": "user", "content": "abcdefghij"}
                ],
            },
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        with patch.object(utils, "CHAT_EVENT_CHUNK_MAX_TOKENS", 16), patch.object(
            utils,
            "CHAT_EVENT_CHUNK_OVERLAP_TOKENS",
            0,
        ):
            field_records = utils.build_chat_dialogue_field_record_entries(
                source_entry,
                tokenizer,
                str(model_dir),
            )
        self.assertEqual(len(field_records), 3)
        self.assertEqual(len({entry["id"] for entry in field_records}), 3)

    def test_chat_event_crud_smoke_indexes_messages_and_keeps_search_lightweight(self) -> None:
        root = self.db_path.parent
        model_dir = self._create_fake_model()
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
                raw_dialogue=build_dialogue("raw-secret question", "raw-secret response"),
                tags=["chat"],
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertNotIn("_".join(("fact", "extraction", "reminder")), saved)
            record_id = saved["id"]
            search_results = server.search_records(
                query="chat summary",
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertTrue(search_results)
            self.assertTrue(all("raw_dialogue" not in result for result in search_results))

            details = server.get_detail_records_by_ids([record_id], full_dialogue=True)
            self.assertEqual(
                details["records"][0]["raw_dialogue"],
                build_dialogue("raw-secret question", "raw-secret response"),
            )
            self.assertNotIn("detailed_summary", details["records"][0])

            embedder.texts.clear()
            title_updated = server.update_record(
                record_id=record_id,
                changes={"title": "updated title"},
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertNotIn("index_refresh_state", title_updated)
            self.assertEqual(len(embedder.texts), 2)
            self.assertTrue(any("updated title" in text for text in embedder.texts))

            embedder.texts.clear()
            summary_updated = server.update_record(
                record_id=record_id,
                changes={"short_summary": "updated summary"},
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertNotIn("index_refresh_state", summary_updated)
            self.assertEqual(len(embedder.texts), 2)
            self.assertTrue(any("updated summary" in text for text in embedder.texts))

            with closing(utils.get_sqlite_connection()) as connection:
                raw_rows = connection.execute(
                    "SELECT id, source_field_value_text FROM field_records WHERE source_memory_id = ? AND source_field_name = 'raw_dialogue' ORDER BY id",
                    (record_id,),
                ).fetchall()
            self.assertEqual(len(raw_rows), 2)
            original_raw_field_ids = {str(row["id"]) for row in raw_rows}

            embedder.texts.clear()
            updated = server.update_record(
                record_id=record_id,
                changes={"raw_dialogue": build_dialogue("updated-secret question", "updated-secret response")},
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertNotIn("_".join(("fact", "extraction", "reminder")), updated)
            self.assertNotIn("index_refresh_state", updated)
            self.assertEqual(len(embedder.texts), 2)
            self.assertTrue(all("updated title" not in text for text in embedder.texts))
            updated_details = server.get_detail_records_by_ids([record_id], full_dialogue=True)
            self.assertEqual(
                updated_details["records"][0]["raw_dialogue"],
                build_dialogue("updated-secret question", "updated-secret response"),
            )
            with closing(utils.get_sqlite_connection()) as connection:
                updated_raw_rows = connection.execute(
                    "SELECT id, source_field_value_text FROM field_records WHERE source_memory_id = ? AND source_field_name = 'raw_dialogue'",
                    (record_id,),
                ).fetchall()
            updated_raw_field_ids = {str(row["id"]) for row in updated_raw_rows}
            self.assertEqual(original_raw_field_ids, updated_raw_field_ids)
            self.assertTrue(
                all("updated-secret" in str(row["source_field_value_text"]) for row in updated_raw_rows)
            )

            deleted = server.delete_records_by_ids(
                [record_id],
                embedder=embedder,
                model_path=str(model_dir),
            )
            self.assertEqual(deleted["deleted_ids"], [record_id])
            self.assertEqual(server.get_detail_records_by_ids([record_id])["missing_ids"], [record_id])

        embedded_text = "\n".join(embedder.texts)
        self.assertIn("updated-secret", embedded_text)

    def test_update_repairs_unrelated_missing_field_record_without_full_rebuild(self) -> None:
        model_dir = self._create_fake_model()
        embedder = RecordingEmbedder()
        base_times = {"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
        target_payload = utils.build_payload(
            memory_kind="fact",
            title="target title",
            short_summary="target summary",
            detailed_summary="target details",
            tags=["target"],
            validate_reference_doc_path=False,
        )
        other_payload = utils.build_payload(
            memory_kind="chat_event",
            title="other title",
            short_summary="other summary",
            raw_dialogue=build_dialogue("unrelated raw input", "unrelated raw output"),
            tags=["other"],
            validate_reference_doc_path=False,
        )
        target = {"id": "fact-target", **base_times, **target_payload}
        other = {"id": "event-other", **base_times, **other_payload}
        entries = [
            target,
            *utils.build_field_record_entries(target),
            other,
            *utils.build_field_record_entries(
                other,
                dialogue_tokenizer=embedder.tokenizer,
                model_path=str(model_dir),
            ),
        ]
        self._store_and_index(entries, embedder, model_dir)

        with self.artifact_paths["META_PATH"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with self.artifact_paths["EMBEDDINGS_PATH"].open("rb") as handle:
            embeddings = np.load(handle, allow_pickle=False)
        missing_id = next(
            str(entry["id"])
            for entry in utils.fetch_store_entries_by_ids(meta["row_to_id"])
            if entry.get("source_field_name") == "raw_dialogue"
        )
        missing_index = meta["row_to_id"].index(missing_id)
        kept_ids = [entry_id for entry_id in meta["row_to_id"] if entry_id != missing_id]
        kept_embeddings = np.delete(embeddings, missing_index, axis=0)
        utils.rebuild_index_from_embeddings(
            kept_ids,
            kept_embeddings,
            str(model_dir),
            utils.build_embedding_model_fingerprint(str(model_dir)),
        )

        embedder.texts.clear()
        with patch.object(
            utils,
            "launch_background_full_rebuild",
            side_effect=AssertionError("unrelated cache drift must stay incremental"),
        ):
            result = server.update_record(
                record_id="fact-target",
                changes={"title": "target title changed"},
                embedder=embedder,
                model_path=str(model_dir),
            )
        self.assertNotIn("index_refresh_state", result)
        self.assertEqual(len(embedder.texts), 3, embedder.texts)
        self.assertTrue(any("unrelated" in text for text in embedder.texts), embedder.texts)
        with self.artifact_paths["META_PATH"].open("r", encoding="utf-8") as handle:
            repaired_meta = json.load(handle)
        self.assertEqual(
            repaired_meta["row_to_id"],
            utils.fetch_ordered_store_entry_ids(),
        )

    def test_update_fallback_reports_extra_meta_ids(self) -> None:
        model_dir = self._create_fake_model()
        embedder = RecordingEmbedder()
        payload = utils.build_payload(
            memory_kind="fact",
            title="stale meta title",
            short_summary="stale meta summary",
            detailed_summary="stale meta details",
            tags=[],
            validate_reference_doc_path=False,
        )
        entry = {"id": "fact-stale-meta", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00", **payload}
        self._store_and_index([entry, *utils.build_field_record_entries(entry)], embedder, model_dir)
        with self.artifact_paths["META_PATH"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with self.artifact_paths["EMBEDDINGS_PATH"].open("rb") as handle:
            embeddings = np.load(handle, allow_pickle=False)
        stale_ids = [*meta["row_to_id"], "fmem-stale-cache-row"]
        stale_embeddings = np.vstack(
            [embeddings, np.ones((1, embeddings.shape[1]), dtype="float32")]
        )
        utils.rebuild_index_from_embeddings(
            stale_ids,
            stale_embeddings,
            str(model_dir),
            utils.build_embedding_model_fingerprint(str(model_dir)),
        )

        def fake_launch(model_path: str, target_entry_count: int, reason: str, missing_entry_count: int = 0) -> dict[str, object]:
            del model_path, target_entry_count
            return utils.build_background_rebuild_notice(reason, missing_entry_count)

        with patch.object(utils, "launch_background_full_rebuild", side_effect=fake_launch):
            result = server.update_record(
                record_id="fact-stale-meta",
                changes={"title": "changed with stale meta"},
                embedder=embedder,
                model_path=str(model_dir),
            )
        self.assertEqual(result["index_refresh_reason"], "missing_ids")
        self.assertEqual(result["missing_entry_count"], 1)

    def test_update_fallback_reports_signature_mismatch(self) -> None:
        model_dir = self._create_fake_model()
        embedder = RecordingEmbedder()
        payload = utils.build_payload(
            memory_kind="fact",
            title="fallback title",
            short_summary="fallback summary",
            detailed_summary="fallback details",
            tags=[],
            validate_reference_doc_path=False,
        )
        entry = {"id": "fact-fallback", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00", **payload}
        self._store_and_index([entry, *utils.build_field_record_entries(entry)], embedder, model_dir)
        with self.artifact_paths["META_PATH"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["retrieval_field_signature"] = "stale-signature"
        self.artifact_paths["META_PATH"].write_text(json.dumps(meta), encoding="utf-8")

        def fake_launch(model_path: str, target_entry_count: int, reason: str, missing_entry_count: int = 0) -> dict[str, object]:
            del model_path, target_entry_count
            return utils.build_background_rebuild_notice(reason, missing_entry_count)

        with patch.object(utils, "launch_background_full_rebuild", side_effect=fake_launch):
            result = server.update_record(
                record_id="fact-fallback",
                changes={"title": "changed after signature drift"},
                embedder=embedder,
                model_path=str(model_dir),
            )
        self.assertEqual(result["index_refresh_reason"], "signature_mismatch")

    def test_update_fallback_reports_cache_unavailable(self) -> None:
        model_dir = self._create_fake_model()
        embedder = RecordingEmbedder()
        payload = utils.build_payload(
            memory_kind="fact",
            title="uncached title",
            short_summary="uncached summary",
            detailed_summary="uncached details",
            tags=[],
            validate_reference_doc_path=False,
        )
        entry = {"id": "fact-uncached", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00", **payload}
        utils.ensure_sqlite_store_ready()
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries([entry, *utils.build_field_record_entries(entry)], connection=connection)
            connection.commit()

        def fake_launch(model_path: str, target_entry_count: int, reason: str, missing_entry_count: int = 0) -> dict[str, object]:
            del model_path, target_entry_count
            return utils.build_background_rebuild_notice(reason, missing_entry_count)

        with patch.object(utils, "launch_background_full_rebuild", side_effect=fake_launch):
            result = server.update_record(
                record_id="fact-uncached",
                changes={"title": "changed without cache"},
                embedder=embedder,
                model_path=str(model_dir),
            )
        self.assertEqual(result["index_refresh_reason"], "cache_unavailable")

    def test_update_fallback_reports_invalid_cache_payload(self) -> None:
        model_dir = self._create_fake_model()
        embedder = RecordingEmbedder()
        payload = utils.build_payload(
            memory_kind="fact",
            title="invalid cache title",
            short_summary="invalid cache summary",
            detailed_summary="invalid cache details",
            tags=[],
            validate_reference_doc_path=False,
        )
        entry = {"id": "fact-invalid-cache", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00", **payload}
        self._store_and_index([entry, *utils.build_field_record_entries(entry)], embedder, model_dir)
        with self.artifact_paths["META_PATH"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["dim"] = int(meta["dim"]) + 1
        self.artifact_paths["META_PATH"].write_text(json.dumps(meta), encoding="utf-8")

        def fake_launch(model_path: str, target_entry_count: int, reason: str, missing_entry_count: int = 0) -> dict[str, object]:
            del model_path, target_entry_count
            return utils.build_background_rebuild_notice(reason, missing_entry_count)

        with patch.object(utils, "launch_background_full_rebuild", side_effect=fake_launch):
            result = server.update_record(
                record_id="fact-invalid-cache",
                changes={"title": "changed with invalid cache"},
                embedder=embedder,
                model_path=str(model_dir),
            )
        self.assertEqual(result["index_refresh_reason"], "payload_invalid")

    def test_search_reports_running_rebuild_reason(self) -> None:
        utils.write_rebuild_state(
            utils.build_rebuild_state_snapshot(
                state="running",
                mode=utils.BACKGROUND_REBUILD_MODE,
                index_refresh_reason="missing_ids",
                missing_entry_count=3,
                worker_pid=os.getpid(),
            )
        )
        response = server.search("anything")
        self.assertFalse(response["search_available"])
        self.assertEqual(response["index_refresh_reason"], "missing_ids")
        self.assertEqual(response["missing_entry_count"], 3)


if __name__ == "__main__":
    unittest.main()
