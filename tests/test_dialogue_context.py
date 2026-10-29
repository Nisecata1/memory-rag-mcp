from __future__ import annotations

import asyncio
import copy
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack, closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import server
import server_utils as utils


def dialogue(count: int) -> dict:
    return {
        "schema_version": 1,
        "metadata": {"session_id": "test-session"},
        "messages": [
            {
                "message_id": f"msg-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"verbatim message {i}\nsecond line",
                "timestamp": f"time-{i}",
                "metadata": {"original_index": i},
            }
            for i in range(count)
        ],
    }


def returned_ids(view: dict) -> list[str]:
    return [item["message_id"] for item in view["messages"] if "message_id" in item]


class DialogueContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(utils, "RETURN_MESSAGE_NUM", 10))
        self.stack.enter_context(patch.object(utils, "MESSAGE_TOP_K", 3))
        self.stack.enter_context(patch.object(utils, "MESSAGE_SCORE_GAP", 0.2))

    def test_config_defaults_and_validation(self) -> None:
        self.assertEqual(utils.parse_dialogue_return_config({}, {}), (0.2, 10, 3))
        for key, invalids in (
            ("message_score_gap", [True, None, "0.2", -0.1, float("nan"), float("inf")]),
            ("returnMessageNum", [False, None, "10", 1.5, -1]),
            ("message_top_k", [True, False, None, "3", 3.0, 0, -1]),
        ):
            for value in invalids:
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    utils.parse_dialogue_return_config(
                        {} if key == "returnMessageNum" else {key: value},
                        {key: value} if key == "returnMessageNum" else {},
                    )
        self.assertEqual(
            utils.parse_dialogue_return_config({"message_score_gap": 0, "message_top_k": 1}, {"returnMessageNum": 0}),
            (0.0, 0, 1),
        )
        for old_limit in (40, 1, None, False, "obsolete"):
            with self.subTest(old_limit=old_limit):
                self.assertEqual(utils.parse_dialogue_return_config({}, {"max_total_messages": old_limit}), (0.2, 10, 3))

    def test_message_limit_default_override_and_strict_validation(self) -> None:
        with patch.object(utils, "MESSAGE_TOP_K", 5):
            self.assertEqual(utils.resolve_message_top_k(), 5)
            self.assertEqual(utils.resolve_message_top_k(None), 5)
            self.assertEqual(utils.resolve_message_top_k(1), 1)
        for invalid in (True, False, "3", 3.0, 0, -1):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                utils.resolve_message_top_k(invalid)
            with patch.object(server, "load_rebuild_state") as load_state:
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    server.search("query", message_top_k=invalid)
                load_state.assert_not_called()

    def test_config_load_defaults_and_no_retrieval_signature_change(self) -> None:
        config = yaml.safe_load(utils.CONFIG_PATH.read_text(encoding="utf-8"))
        config["search"].pop("message_score_gap", None)
        config["search"].pop("message_top_k", None)
        config["results"].pop("returnMessageNum", None)
        config["results"].pop("max_total_messages", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.object(utils, "CONFIG_PATH", path):
                defaults = utils.load_project_config()
                config["search"]["message_score_gap"] = 0.05
                config["search"]["message_top_k"] = 5
                config["results"].update(returnMessageNum=3, max_total_messages=9)
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                changed = utils.load_project_config()
        self.assertEqual((defaults["message_score_gap"], defaults["return_message_num"], defaults["message_top_k"]), (0.2, 10, 3))
        self.assertEqual((changed["message_score_gap"], changed["return_message_num"], changed["message_top_k"]), (0.05, 3, 5))
        self.assertNotIn("max_total_messages", changed)
        for key in ("retrieval_field_candidates", "chat_event_chunk_max_tokens", "chat_event_chunk_overlap_tokens"):
            self.assertEqual(defaults[key], changed[key])
        for loaded in (defaults, changed):
            self.assertEqual("|".join(loaded["retrieval_field_candidates"]), utils.RETRIEVAL_FIELD_SIGNATURE)
            self.assertEqual(
                f"chat_dialogue_v1|roles=user,assistant|chunk_max={loaded['chat_event_chunk_max_tokens']}|chunk_overlap={loaded['chat_event_chunk_overlap_tokens']}",
                utils.FIELD_RECORD_DERIVATION_SIGNATURE,
            )

    def test_score_boundary_ties_and_limit(self) -> None:
        scores = {"late": (0.8, 11), "early": (0.8, 0), "boundary": (0.6, 22), "outside": (0.599, 33)}
        selected = utils.select_matched_messages(scores)
        self.assertEqual(selected, [
            {"message_id": "early", "score": 0.8},
            {"message_id": "late", "score": 0.8},
            {"message_id": "boundary", "score": 0.6},
        ])
        scores = {f"msg-{i}": (0.5, i * 11) for i in reversed(range(60))}
        self.assertEqual([item["message_id"] for item in utils.select_matched_messages(scores)], [f"msg-{i}" for i in range(3)])
        self.assertEqual(len(utils.select_matched_messages(scores, 50)), 50)
        with patch.object(utils, "MESSAGE_SCORE_GAP", 0):
            self.assertEqual(utils.select_matched_messages({"a": (0.8, 0), "b": (0.8 - 1e-8, 11)}), [{"message_id": "a", "score": 0.8}])

    def test_distance_boundary_and_rejected_candidates_do_not_suppress(self) -> None:
        scores = {"best": (0.95, 20), "inside": (0.94, 30), "next": (0.9, 31), "far": (0.8, 80)}
        self.assertEqual([item["message_id"] for item in utils.select_matched_messages(scores)], ["best", "next", "far"])
        self.assertEqual(len(utils.select_matched_messages(scores, 1)), 1)
        chain = {"a": (0.9, 0), "b": (0.89, 10), "c": (0.88, 20)}
        self.assertEqual([item["message_id"] for item in utils.select_matched_messages(chain)], ["a", "c"])
        self.assertEqual(utils.select_matched_messages({}), [])

    def test_nonfinite_scores_negative_scores_and_id_tiebreak(self) -> None:
        scores = {"nan": (float("nan"), 0), "inf": (float("inf"), 30), "negative-inf": (float("-inf"), 40), "b": (-0.1, 10), "a": (-0.1, 10), "later": (-0.2, 11)}
        with patch.object(utils, "RETURN_MESSAGE_NUM", 0):
            self.assertEqual(utils.select_matched_messages(scores), [
                {"message_id": "a", "score": -0.1}, {"message_id": "later", "score": -0.2},
            ])

    def test_single_window_preserves_original_messages_and_metadata(self) -> None:
        source = dialogue(100)
        before = copy.deepcopy(source)
        view = utils.build_dialogue_context(source, ["msg-50"])
        self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(40, 61)])
        self.assertEqual(view["messages"][1:-1], source["messages"][40:61])
        self.assertEqual(view["metadata"], source["metadata"])
        self.assertEqual([item["omitted_count"] for item in view["messages"] if "omitted_count" in item], [40, 39])
        self.assertEqual(view["context"]["returned_messages"], 21)
        self.assertTrue(view["context"]["truncated"])
        self.assertEqual(source, before)

    def test_edges_small_dialogue_and_zero_radius(self) -> None:
        for anchor, expected in (("msg-0", range(11)), ("msg-49", range(39, 50))):
            with self.subTest(anchor=anchor):
                self.assertEqual(returned_ids(utils.build_dialogue_context(dialogue(50), [anchor])), [f"msg-{i}" for i in expected])
        view = utils.build_dialogue_context(dialogue(3), ["msg-1"])
        self.assertFalse(view["context"]["truncated"])
        self.assertEqual(len(view["messages"]), 3)
        with patch.object(utils, "RETURN_MESSAGE_NUM", 0):
            self.assertEqual(returned_ids(utils.build_dialogue_context(dialogue(50), ["msg-20"])), ["msg-20"])

    def test_overlapping_and_adjacent_windows_have_no_internal_omission(self) -> None:
        with patch.object(utils, "RETURN_MESSAGE_NUM", 2):
            for anchors, expected in ((["msg-10", "msg-12"], range(8, 15)), (["msg-10", "msg-15"], range(8, 18))):
                with self.subTest(anchors=anchors):
                    view = utils.build_dialogue_context(dialogue(30), anchors)
                    self.assertEqual(returned_ids(view), [f"msg-{i}" for i in expected])
                    self.assertTrue(all("message_id" in item for item in view["messages"][1:-1]))

    def test_separate_windows_have_explicit_gaps(self) -> None:
        with patch.object(utils, "RETURN_MESSAGE_NUM", 2):
            view = utils.build_dialogue_context(dialogue(50), ["msg-10", "msg-40"])
        omissions = [item for item in view["messages"] if item.get("type") == "omission"]
        self.assertEqual([item["omitted_count"] for item in omissions], [8, 25, 7])
        self.assertTrue(all("role" not in item and "message_id" not in item for item in omissions))
        self.assertEqual(sum(item["omitted_count"] for item in omissions) + view["context"]["returned_messages"], 50)

    def test_three_windows_are_complete_without_40_message_cap(self) -> None:
        view = utils.build_dialogue_context(dialogue(120), ["msg-20", "msg-60", "msg-100"])
        self.assertEqual(view["context"]["returned_messages"], 63)
        self.assertEqual(view["context"]["omitted_anchor_count"], 0)
        self.assertEqual(returned_ids(view), [f"msg-{i}" for start in (10, 50, 90) for i in range(start, start + 21)])
        self.assertEqual(sum(item["omitted_count"] for item in view["messages"] if item.get("type") == "omission"), 57)

    def test_representative_windows_can_overlap(self) -> None:
        view = utils.build_dialogue_context(dialogue(60), ["msg-20", "msg-35"])
        self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(10, 46)])
        self.assertEqual(view["context"]["returned_messages"], 36)

    def test_tail_preview_is_not_capped_at_40(self) -> None:
        with patch.object(utils, "RETURN_MESSAGE_NUM", 25):
            view = utils.build_dialogue_context(dialogue(100))
        self.assertEqual(view["context"]["returned_messages"], 51)

    def test_excess_anchors_duplicates_and_unresolved_ids(self) -> None:
        anchors = [f"msg-{i}" for i in reversed(range(50))]
        view = utils.build_dialogue_context(dialogue(60), [anchors[0], *anchors, "gone"])
        self.assertEqual(view["context"]["anchor_message_ids"], anchors)
        self.assertEqual(view["context"]["omitted_anchor_count"], 0)
        self.assertEqual(view["context"]["unresolved_message_ids"], ["gone"])
        self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(60)])

    def test_no_anchor_or_stale_anchor_returns_labeled_tail(self) -> None:
        for anchors, reason in ((None, "no_anchors"), (["gone"], "anchors_not_found")):
            with self.subTest(anchors=anchors):
                view = utils.build_dialogue_context(dialogue(60), anchors)
                self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(39, 60)])
                self.assertEqual(view["context"]["mode"], "tail_preview")
                self.assertEqual(view["context"]["reason"], reason)
        view = utils.build_dialogue_context(dialogue(60), ["gone", "msg-20"])
        self.assertEqual(view["context"]["mode"], "matched_context")
        self.assertEqual(view["context"]["unresolved_message_ids"], ["gone"])

    def test_context_view_cannot_be_normalized_for_write(self) -> None:
        view = utils.build_dialogue_context(dialogue(4))
        with self.assertRaisesRegex(ValueError, "read-only context view"):
            utils.normalize_raw_dialogue(view)
        with self.assertRaisesRegex(ValueError, "full_dialogue=true"):
            utils.build_dialogue_context("legacy plain text")


class DialogueDetailIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name, path in {
            "AI_MEMORY_DIR": root, "DATA_DIR": root, "SQLITE_PATH": root / "memory.db",
            "WRITE_LOCK_PATH": root / "write.lock", "REBUILD_STATE_PATH": root / "rebuild_state.json",
        }.items():
            self.stack.enter_context(patch.object(utils, name, path))
        self.raw = dialogue(80)
        times = {"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
        self.entries = [
            {"id": "eventMemId-a", **times, **utils.build_payload(memory_kind="chat_event", title="a", short_summary="a", raw_dialogue=self.raw)},
            {"id": "eventMemId-b", **times, **utils.build_payload(memory_kind="chat_event", title="b", short_summary="b", raw_dialogue=dialogue(80))},
            {"id": "factMemId-a", **times, **utils.build_payload(memory_kind="fact", title="fact", short_summary="fact", detailed_summary="full fact")},
        ]
        with closing(sqlite3.connect(utils.SQLITE_PATH)) as connection:
            connection.row_factory = sqlite3.Row
            utils.initialize_sqlite_schema(connection)
            utils.upsert_store_entries(self.entries, connection=connection)
            connection.commit()

    def test_detail_grouping_full_override_and_ssot_unchanged(self) -> None:
        ids = ["eventMemId-b", "factMemId-a", "eventMemId-a", "missing-id"]
        result = server.get_details_by_ids(ids, {"eventMemId-a": ["msg-20"], "eventMemId-b": ["msg-60"]})
        self.assertEqual(result["missing_ids"], ["missing-id"])
        self.assertEqual([entry["id"] for entry in result["records"]], ids[:-1])
        self.assertEqual(returned_ids(result["records"][0]["raw_dialogue"]), [f"msg-{i}" for i in range(50, 71)])
        self.assertEqual(returned_ids(result["records"][2]["raw_dialogue"]), [f"msg-{i}" for i in range(10, 31)])
        full = server.get_details_by_ids(ids, {"unrequested": ["gone"]}, full_dialogue=True)
        self.assertEqual(full["records"][1], result["records"][1])
        self.assertEqual(full["records"][2]["raw_dialogue"], self.raw)
        preview = server.get_details_by_ids(["eventMemId-a"])["records"][0]["raw_dialogue"]
        self.assertEqual(returned_ids(preview), [f"msg-{i}" for i in range(59, 80)])
        self.assertEqual(utils.fetch_detail_records_by_ids(["eventMemId-a"])[0]["raw_dialogue"], self.raw)

    def test_update_rejects_view_before_any_write(self) -> None:
        preview = server.get_details_by_ids(["eventMemId-a"])["records"][0]["raw_dialogue"]
        with patch.object(server, "update_store_and_rebuild") as write:
            with self.assertRaisesRegex(ValueError, "read-only context view"):
                server.update_record("eventMemId-a", {"raw_dialogue": preview})
            write.assert_not_called()
        self.assertEqual(utils.fetch_detail_records_by_ids(["eventMemId-a"])[0]["raw_dialogue"], self.raw)

    def test_invalid_anchor_maps_are_rejected(self) -> None:
        for anchors in ([], "", {"unknown": []}, {"eventMemId-a": "msg-1"}, {"eventMemId-a": [""]}):
            with self.subTest(anchors=anchors), self.assertRaisesRegex(ValueError, "matched_message_ids"):
                server.get_detail_records_by_ids(["eventMemId-a"], anchors)

    def test_stdio_details_full_read_and_write_rejection(self) -> None:
        async def check_protocol() -> None:
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).with_name("concurrent_stdio_worker.py"))],
                cwd=str(Path(server.__file__).parent),
                env={**os.environ, "MEMORY_RAG_TEST_ROOT": str(utils.SQLITE_PATH.parent), "MEMORY_RAG_WORKER_MODE": "stdio"},
            )
            with open(os.devnull, "w", encoding="utf-8") as errors:
                async with stdio_client(params, errlog=errors) as streams, ClientSession(*streams) as session:
                    await session.initialize()

                    async def read(arguments):
                        response = await session.call_tool("get_details_by_ids", arguments, read_timeout_seconds=timedelta(seconds=10))
                        self.assertFalse(response.isError)
                        return json.loads(next(block.text for block in response.content if block.type == "text"))

                    context = await read({"ids": ["eventMemId-a"], "matched_message_ids": {"eventMemId-a": ["msg-20"]}})
                    view = context["records"][0]["raw_dialogue"]
                    self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(10, 31)])
                    windows = await read({"ids": ["eventMemId-a"], "matched_message_ids": {"eventMemId-a": ["msg-10", "msg-40", "msg-69"]}})
                    self.assertEqual(windows["records"][0]["raw_dialogue"]["context"]["returned_messages"], 63)
                    many = await read({"ids": ["eventMemId-a"], "matched_message_ids": {"eventMemId-a": [f"msg-{i}" for i in range(70)]}})
                    self.assertEqual(len(many["records"][0]["raw_dialogue"]["context"]["anchor_message_ids"]), 70)
                    self.assertEqual(many["records"][0]["raw_dialogue"]["context"]["returned_messages"], 80)
                    rejected = await session.call_tool("update", {"id": "eventMemId-a", "changes": {"raw_dialogue": view}}, read_timeout_seconds=timedelta(seconds=10))
                    self.assertTrue(rejected.isError)
                    self.assertIn("read-only context view", str(rejected.content))
                    full = await read({"ids": ["eventMemId-a"], "full_dialogue": True})
                    self.assertEqual(full["records"][0]["raw_dialogue"], self.raw)
        asyncio.run(asyncio.wait_for(check_protocol(), timeout=30))

    def test_mcp_schema_has_optional_anchor_map_and_full_switch(self) -> None:
        tools = asyncio.run(server.server.list_tools())
        schema = next(tool.inputSchema for tool in tools if tool.name == "get_details_by_ids")
        self.assertEqual(schema["required"], ["ids"])
        self.assertIn("matched_message_ids", schema["properties"])
        self.assertFalse(schema["properties"]["full_dialogue"]["default"])
        search_tool = next(tool for tool in tools if tool.name == "search")
        search_schema = search_tool.inputSchema
        self.assertEqual(search_schema["required"], ["query"])
        self.assertIsNone(search_schema["properties"]["message_top_k"]["default"])
        self.assertIn("message_top_k", search_tool.description)
        detail_tool = next(tool for tool in tools if tool.name == "get_details_by_ids")
        self.assertNotIn("最多40条", detail_tool.description)
        self.assertIn("默认最多21条", detail_tool.description)
        self.assertIn("不使用上一次 search", detail_tool.description)
        self.assertIn("context.mode=tail_preview", detail_tool.description)
        self.assertIn("主记忆", schema["properties"]["ids"]["description"])
        self.assertIn("末尾预览", schema["properties"]["matched_message_ids"]["description"])
        self.assertIn("不会自动沿用", search_tool.description)

    def test_search_envelope_config_override_and_rebuild_states(self) -> None:
        with patch.object(utils, "MESSAGE_TOP_K", 5):
            for state in ("idle", "running", "failed"):
                for override, expected in ((None, 5), (2, 2)):
                    with self.subTest(state=state, override=override):
                        with patch.object(server, "load_rebuild_state", return_value={"state": state}), patch.object(server, "search_records", return_value=[]) as search_records:
                            result = server.search("query", top_k=2, message_top_k=override)
                        self.assertEqual(result["message_top_k"], expected)
                        self.assertEqual(result["top_k"], 2)
                        if state == "idle":
                            search_records.assert_called_once_with(query="query", top_k=2, message_top_k=expected)
                        else:
                            search_records.assert_not_called()
                            self.assertFalse(result["search_available"])
                            self.assertEqual(result["rebuild_state"], state)

    def test_stdio_search_scores_limits_and_strict_validation(self) -> None:
        root = utils.SQLITE_PATH.parent
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        entries = list(self.entries)
        for parent in self.entries[:2]:
            for index in range(0, 80, 11):
                entries.append(utils.build_chat_dialogue_field_record_entry(
                    parent["id"], {"message_id": f"msg-{index}", "message_index": index, "role": "user", "content": "text", "chunk_index": 0, "chunk_count": 1},
                    parent["created_at"], parent["updated_at"],
                ))
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries(entries, connection=connection)
            connection.commit()
        with ExitStack() as stack:
            for name, path in {
                "META_PATH": root / "data" / "meta.json",
                "INDEX_PATH": root / "data" / "memory.faiss",
                "EMBEDDINGS_PATH": root / "data" / "memory_embeddings.npy",
            }.items():
                stack.enter_context(patch.object(utils, name, path))
            vectors = np.full((len(entries), 8), 1 / np.sqrt(8), dtype="float32")
            utils.rebuild_index_from_embeddings([entry["id"] for entry in entries], vectors, str(model), utils.build_embedding_model_fingerprint(str(model)))

        async def check_protocol() -> None:
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).with_name("concurrent_stdio_worker.py"))],
                cwd=str(Path(server.__file__).parent),
                env={**os.environ, "MEMORY_RAG_TEST_ROOT": str(root), "MEMORY_RAG_WORKER_MODE": "stdio"},
            )
            with open(os.devnull, "w", encoding="utf-8") as errors:
                async with stdio_client(params, errlog=errors) as streams, ClientSession(*streams) as session:
                    await session.initialize()
                    for extra, expected in (({}, 3), ({"message_top_k": None}, 3), ({"message_top_k": 1}, 1), ({"message_top_k": 5}, 5)):
                        response = await session.call_tool("search", {"query": "query", "top_k": 3, **extra}, read_timeout_seconds=timedelta(seconds=10))
                        self.assertFalse(response.isError, str(response.content))
                        result = json.loads(next(block.text for block in response.content if block.type == "text"))
                        self.assertEqual(result["message_top_k"], expected)
                        chats = [entry for entry in result["results"] if entry["memory_kind"] == "chat_event"]
                        self.assertEqual(len(chats), 2)
                        for entry in chats:
                            self.assertEqual(entry["matched_message_ids"], [f"msg-{i * 11}" for i in range(expected)])
                            self.assertEqual(entry["matched_message_ids"], [item["message_id"] for item in entry["matched_messages"]])
                            self.assertTrue(all(type(item["score"]) is float for item in entry["matched_messages"]))
                    for invalid in (True, False, "3", 3.0, 0, -1):
                        response = await session.call_tool("search", {"query": "query", "message_top_k": invalid}, read_timeout_seconds=timedelta(seconds=10))
                        self.assertTrue(response.isError, f"accepted invalid message_top_k: {invalid!r}")
        asyncio.run(asyncio.wait_for(check_protocol(), timeout=45))

    def test_search_passes_ranked_message_anchors_without_changing_parent_rank(self) -> None:
        parents = [*self.entries[:2], {**self.entries[0], "id": "eventMemId-no-messages"}, self.entries[2]]
        direct = [*parents]
        scores = [0.99, 0.95, 0.9, 0.85]
        for parent, message_id, index, score, chunk in (
            (parents[0], "msg-20", 20, 0.80, 0),
            (parents[0], "msg-20", 20, 0.79, 1),
            (parents[0], "msg-10", 10, 0.80, 0),
            (parents[0], "msg-30", 30, 0.60, 0),
            (parents[0], "msg-40", 40, 0.59, 0),
            (parents[0], "msg-61", 61, float("nan"), 0),
            (parents[0], "msg-61", 61, 0.70, 1),
            (parents[0], "msg-61", 61, 0.69, 2),
            (parents[1], "msg-50", 50, 0.30, 0),
            (parents[1], "msg-61", 61, 0.10, 0),
        ):
            direct.append(utils.build_chat_dialogue_field_record_entry(
                parent["id"], {"message_id": message_id, "message_index": index, "role": "user", "content": "text", "chunk_index": chunk, "chunk_count": 3},
                parent["created_at"], parent["updated_at"],
            ))
            scores.append(score)
        path = utils.DATA_DIR / "meta.json"
        path.write_text(json.dumps({"row_to_id": [entry["id"] for entry in direct], "embedding_model_fingerprint": "test", "retrieval_field_signature": utils.RETRIEVAL_FIELD_SIGNATURE, "field_record_derivation_signature": utils.FIELD_RECORD_DERIVATION_SIGNATURE, "dim": 2}), encoding="utf-8")
        index_path = utils.DATA_DIR / "index.faiss"
        index_path.touch()

        class FakeIndex:
            def search(self, query, count):
                return np.asarray([scores], dtype="float32"), np.asarray([list(range(len(direct)))])

        with ExitStack() as stack:
            for target, name, value in ((server, "META_PATH", path), (server, "INDEX_PATH", index_path), (server, "SEARCH_MIN_SIMILARITY", None), (utils, "MESSAGE_SCORE_GAP", 0.2), (utils, "MESSAGE_TOP_K", 3), (utils, "RETURN_MESSAGE_NUM", 10)):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.object(server, "build_embedding_model_fingerprint", return_value="test"))
            stack.enter_context(patch.object(server, "encode_texts", return_value=np.ones((1, 2), dtype="float32")))
            stack.enter_context(patch.object(server.faiss, "read_index", return_value=FakeIndex()))
            stack.enter_context(patch.object(server, "fetch_search_entries_by_ids", return_value=direct))
            stack.enter_context(patch.object(server, "fetch_search_parents_by_ids", return_value=parents))
            results = server.search_records("query", top_k=4, embedder=object(), model_path="test")
            limited = server.search_records("query", top_k=4, message_top_k=1, embedder=object(), model_path="test")
            top_two = server.search_records("query", top_k=2, embedder=object(), model_path="test")
        self.assertEqual([result["id"] for result in results], [entry["id"] for entry in parents])
        self.assertAlmostEqual(results[0]["score"], 0.99)
        self.assertEqual(results[0]["matched_fields"], ["raw_dialogue"])
        self.assertEqual(results[0]["matched_message_ids"], ["msg-10", "msg-61", "msg-30"])
        self.assertEqual(results[1]["matched_message_ids"], ["msg-50", "msg-61"])
        self.assertEqual(results[2]["matched_message_ids"], [])
        self.assertEqual(results[2]["matched_messages"], [])
        self.assertNotIn("matched_messages", results[3])
        self.assertNotIn("matched_message_ids", results[3])
        self.assertEqual([item["id"] for item in top_two], [parent["id"] for parent in parents[:2]])
        self.assertEqual(limited[0]["matched_message_ids"], ["msg-10"])
        for expected, result in zip(scores[:4], results):
            self.assertAlmostEqual(result["score"], expected)
        for result, changed_limit in zip(results, limited):
            self.assertEqual({key: value for key, value in result.items() if key not in ("matched_messages", "matched_message_ids")}, {key: value for key, value in changed_limit.items() if key not in ("matched_messages", "matched_message_ids")})
        for result in results[:3]:
            self.assertEqual(result["matched_message_ids"], [item["message_id"] for item in result["matched_messages"]])
        self.assertAlmostEqual(results[0]["matched_messages"][0]["score"], 0.8)
        self.assertAlmostEqual(results[0]["matched_messages"][1]["score"], 0.7)
        self.assertAlmostEqual(results[0]["matched_messages"][2]["score"], 0.6)
        self.assertTrue(all("raw_dialogue" not in result for result in results))
        detail = server.get_details_by_ids([item["id"] for item in results[:2]], {item["id"]: item["matched_message_ids"] for item in results[:2]})
        self.assertEqual(detail["records"][0]["raw_dialogue"]["context"]["anchor_message_ids"], results[0]["matched_message_ids"])


if __name__ == "__main__":
    unittest.main()
