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
        self.stack.enter_context(patch.object(utils, "MAX_TOTAL_MESSAGES", 40))
        self.stack.enter_context(patch.object(utils, "MESSAGE_SCORE_GAP", 0.2))

    def test_config_defaults_and_validation(self) -> None:
        self.assertEqual(utils.parse_dialogue_return_config({}, {}), (0.2, 10, 40))
        for key, invalids in (
            ("message_score_gap", [True, None, "0.2", -0.1, float("nan"), float("inf")]),
            ("returnMessageNum", [False, None, "10", 1.5, -1]),
            ("max_total_messages", [True, None, "40", 2.5, 0, -1]),
        ):
            for value in invalids:
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    utils.parse_dialogue_return_config(
                        {key: value} if key == "message_score_gap" else {},
                        {} if key == "message_score_gap" else {key: value},
                    )
        self.assertEqual(
            utils.parse_dialogue_return_config({"message_score_gap": 0}, {"returnMessageNum": 0, "max_total_messages": 1}),
            (0.0, 0, 1),
        )

    def test_config_load_defaults_and_no_retrieval_signature_change(self) -> None:
        config = yaml.safe_load(utils.CONFIG_PATH.read_text(encoding="utf-8"))
        config["search"].pop("message_score_gap", None)
        config["results"].pop("returnMessageNum", None)
        config["results"].pop("max_total_messages", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.object(utils, "CONFIG_PATH", path):
                defaults = utils.load_project_config()
                config["search"]["message_score_gap"] = 0.05
                config["results"].update(returnMessageNum=3, max_total_messages=9)
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                changed = utils.load_project_config()
        self.assertEqual((defaults["message_score_gap"], defaults["return_message_num"], defaults["max_total_messages"]), (0.2, 10, 40))
        self.assertEqual((changed["message_score_gap"], changed["return_message_num"], changed["max_total_messages"]), (0.05, 3, 9))
        for key in ("retrieval_field_candidates", "chat_event_chunk_max_tokens", "chat_event_chunk_overlap_tokens"):
            self.assertEqual(defaults[key], changed[key])

    def test_score_boundary_ties_and_limit(self) -> None:
        scores = {"late": (0.8, 7), "early": (0.8, 2), "boundary": (0.6, 8), "outside": (0.599, 9)}
        self.assertEqual(utils.select_matched_message_ids(scores), ["early", "late", "boundary"])
        scores = {f"msg-{i}": (0.5, i) for i in reversed(range(60))}
        self.assertEqual(utils.select_matched_message_ids(scores), [f"msg-{i}" for i in range(40)])
        with patch.object(utils, "MESSAGE_SCORE_GAP", 0):
            self.assertEqual(utils.select_matched_message_ids({"a": (0.8, 0), "b": (0.8 - 1e-8, 1)}), ["a"])

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

    def test_budget_keeps_anchors_then_nearest_neighbors(self) -> None:
        with patch.object(utils, "MAX_TOTAL_MESSAGES", 7):
            view = utils.build_dialogue_context(dialogue(80), ["msg-20", "msg-60"])
        self.assertEqual(returned_ids(view), ["msg-18", "msg-19", "msg-20", "msg-21", "msg-59", "msg-60", "msg-61"])
        view = utils.build_dialogue_context(dialogue(100), ["msg-20", "msg-70"])
        self.assertEqual(view["context"]["returned_messages"], 40)
        self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(10, 31)] + [f"msg-{i}" for i in range(61, 80)])

    def test_excess_anchors_duplicates_and_unresolved_ids(self) -> None:
        anchors = [f"msg-{i}" for i in reversed(range(50))]
        view = utils.build_dialogue_context(dialogue(60), [anchors[0], *anchors, "gone"])
        self.assertEqual(view["context"]["anchor_message_ids"], anchors[:40])
        self.assertEqual(view["context"]["omitted_anchor_count"], 10)
        self.assertEqual(view["context"]["unresolved_message_ids"], ["gone"])
        self.assertEqual(returned_ids(view), [f"msg-{i}" for i in range(10, 50)])

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

    def test_search_passes_ranked_message_anchors_without_changing_parent_rank(self) -> None:
        parents = self.entries[:2]
        direct = [*parents]
        scores = [0.99, 0.95]
        for parent, message_id, index, score, chunk in (
            (parents[0], "msg-20", 20, 0.80, 0),
            (parents[0], "msg-20", 20, 0.79, 1),
            (parents[0], "msg-10", 10, 0.80, 0),
            (parents[0], "msg-30", 30, 0.60, 0),
            (parents[0], "msg-40", 40, 0.59, 0),
            (parents[1], "msg-50", 50, 0.30, 0),
            (parents[1], "msg-60", 60, 0.10, 0),
        ):
            direct.append(utils.build_chat_dialogue_field_record_entry(
                parent["id"], {"message_id": message_id, "message_index": index, "role": "user", "content": "text", "chunk_index": chunk, "chunk_count": 2},
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
            for target, name, value in ((server, "META_PATH", path), (server, "INDEX_PATH", index_path), (server, "SEARCH_MIN_SIMILARITY", None), (utils, "MESSAGE_SCORE_GAP", 0.2)):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.object(server, "build_embedding_model_fingerprint", return_value="test"))
            stack.enter_context(patch.object(server, "encode_texts", return_value=np.ones((1, 2), dtype="float32")))
            stack.enter_context(patch.object(server.faiss, "read_index", return_value=FakeIndex()))
            stack.enter_context(patch.object(server, "fetch_store_entries_by_ids", return_value=direct))
            stack.enter_context(patch.object(server, "fetch_entry_family_by_source_ids", return_value=parents))
            results = server.search_records("query", top_k=2, embedder=object(), model_path="test")
        self.assertEqual([result["id"] for result in results], [entry["id"] for entry in parents])
        self.assertAlmostEqual(results[0]["score"], 0.99)
        self.assertEqual(results[0]["matched_fields"], ["raw_dialogue"])
        self.assertEqual(results[0]["matched_message_ids"], ["msg-10", "msg-20", "msg-30"])
        self.assertEqual(results[1]["matched_message_ids"], ["msg-50", "msg-60"])
        self.assertTrue(all("raw_dialogue" not in result for result in results))
        detail = server.get_details_by_ids([item["id"] for item in results], {item["id"]: item["matched_message_ids"] for item in results})
        self.assertEqual(detail["records"][0]["raw_dialogue"]["context"]["anchor_message_ids"], results[0]["matched_message_ids"])


if __name__ == "__main__":
    unittest.main()
