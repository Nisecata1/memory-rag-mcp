from __future__ import annotations

import copy
from functools import partial
import unittest
from unittest.mock import Mock

from memory_access import read_details, read_search_records
from memory_domain import build_dialogue_context


def normalize_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def chat(record_id: str, count: int = 8) -> dict:
    return {
        "id": record_id,
        "memory_kind": "chat_event",
        "title": record_id,
        "raw_dialogue": {
            "schema_version": 1,
            "metadata": {"source": {"session_id": record_id}},
            "messages": [
                {"message_id": f"m-{index}", "role": "user", "content": f"original {index}"}
                for index in range(count)
            ],
        },
    }


def field_record(record_id: str, parent_id: str) -> dict:
    return {
        "id": record_id,
        "memory_kind": "field_record",
        "source_memory_id": parent_id,
        "source_field_name": "raw_dialogue",
        "source_field_value_json": '{"message_id":"m-2","message_index":2}',
    }


class MemoryAccessDetailTests(unittest.TestCase):
    def test_details_are_loaded_in_one_batch_and_each_chat_is_selected_once(self) -> None:
        chat_a, chat_b = chat("chat-a"), chat("chat-b")
        fact = {"id": "fact-a", "memory_kind": "fact", "detailed_summary": "unchanged"}
        before = copy.deepcopy([chat_a, chat_b, fact])
        read_records = Mock(return_value=[chat_b, fact, chat_a])
        build_context = Mock(wraps=partial(build_dialogue_context, return_message_num=1))
        result = read_details(
            [" chat-a ", "missing", "chat-b", "chat-a", "fact-a"],
            {"chat-a": ["m-2", "m-3"], "chat-b": ["m-5"]},
            read_records=read_records,
            normalize_ids=normalize_ids,
            build_context=build_context,
        )
        read_records.assert_called_once_with(["chat-a", "missing", "chat-b", "fact-a"])
        self.assertEqual(build_context.call_count, 2)
        self.assertEqual(result["ids"], ["chat-a", "missing", "chat-b", "fact-a"])
        self.assertEqual([entry["id"] for entry in result["records"]], ["chat-a", "chat-b", "fact-a"])
        self.assertEqual(result["missing_ids"], ["missing"])
        first_ids = [message["message_id"] for message in result["records"][0]["raw_dialogue"]["messages"] if "message_id" in message]
        self.assertEqual(first_ids, ["m-1", "m-2", "m-3", "m-4"])
        self.assertEqual(result["records"][2], fact)
        self.assertEqual([chat_a, chat_b, fact], before)

    def test_full_dialogue_ignores_anchor_validation_and_context_processing(self) -> None:
        source = chat("chat-a", count=50)
        for ignored_anchors in [None, {"unrequested": ["missing"]}, ["invalid map"], {"chat-a": [None]}]:
            with self.subTest(ignored_anchors=ignored_anchors):
                read_records = Mock(return_value=[source])
                build_context = Mock(side_effect=AssertionError("Full reads must not select a context"))
                result = read_details(
                    ["chat-a"], ignored_anchors, full_dialogue=True,
                    read_records=read_records,
                    normalize_ids=normalize_ids,
                    build_context=build_context,
                )
                read_records.assert_called_once_with(["chat-a"])
                build_context.assert_not_called()
                self.assertEqual(result["records"], [source])
                self.assertNotIn("context", result["records"][0]["raw_dialogue"])
                self.assertEqual(len(result["records"][0]["raw_dialogue"]["messages"]), 50)

    def test_invalid_anchors_fail_before_storage_is_touched(self) -> None:
        invalid_maps = [
            ["not a map"],
            {"other": ["m-1"]},
            {"chat-a": "m-1"},
            {"chat-a": [""]},
            {"chat-a": ["   "]},
            {"chat-a": [None]},
            {"chat-a": [1]},
        ]
        for anchors in invalid_maps:
            with self.subTest(anchors=anchors):
                read_records = Mock(side_effect=AssertionError("Invalid input reached storage"))
                build_context = Mock()
                with self.assertRaisesRegex(ValueError, "matched_message_ids"):
                    read_details(
                        ["chat-a"], anchors,
                        read_records=read_records,
                        normalize_ids=normalize_ids,
                        build_context=build_context,
                    )
                read_records.assert_not_called()
                build_context.assert_not_called()

    def test_empty_anchor_list_keeps_the_existing_tail_preview_behavior(self) -> None:
        source = chat("chat-a", count=8)
        result = read_details(
            ["chat-a"], {"chat-a": []},
            read_records=lambda _: [source],
            normalize_ids=normalize_ids,
            build_context=partial(build_dialogue_context, return_message_num=1),
        )
        view = result["records"][0]["raw_dialogue"]
        self.assertEqual(view["context"]["mode"], "tail_preview")
        self.assertEqual(view["context"]["reason"], "no_anchors")
        self.assertEqual([item["message_id"] for item in view["messages"] if "message_id" in item], ["m-5", "m-6", "m-7"])


class MemoryAccessSearchTests(unittest.TestCase):
    def test_parent_already_in_hits_is_not_loaded_again(self) -> None:
        parent = {"id": "chat-a", "memory_kind": "chat_event", "title": "original"}
        child = field_record("field-a", "chat-a")
        read_entries = Mock(return_value=[child, parent])
        read_parents = Mock(side_effect=AssertionError("Already loaded parent was re-read"))
        before = copy.deepcopy([child, parent])
        result = read_search_records(
            ["field-a", "chat-a"], read_entries=read_entries, read_parents=read_parents,
        )
        read_entries.assert_called_once_with(["field-a", "chat-a"])
        read_parents.assert_not_called()
        self.assertEqual(result.direct_by_id, {"field-a": child, "chat-a": parent})
        self.assertEqual(result.public_by_id, {"chat-a": parent})
        self.assertEqual([child, parent], before)

    def test_all_missing_parents_are_fetched_in_one_deduplicated_batch(self) -> None:
        direct_parent = {"id": "chat-c", "memory_kind": "chat_event", "title": "already read"}
        children = [field_record("a1", "chat-a"), field_record("a2", "chat-a"), field_record("b1", "chat-b"), field_record("c1", "chat-c")]
        parent_a = {"id": "chat-a", "memory_kind": "chat_event", "title": "a"}
        parent_b = {"id": "chat-b", "memory_kind": "chat_event", "title": "b"}
        read_entries = Mock(return_value=[*children, direct_parent])
        read_parents = Mock(return_value=[parent_b, parent_a])
        result = read_search_records(
            ["a1", "a2", "b1", "c1", "chat-c"],
            read_entries=read_entries, read_parents=read_parents,
        )
        read_parents.assert_called_once_with(["chat-a", "chat-b"])
        self.assertEqual(set(result.direct_by_id), {"a1", "a2", "b1", "c1", "chat-c"})
        self.assertEqual(result.public_by_id, {"chat-a": parent_a, "chat-b": parent_b, "chat-c": direct_parent})

    def test_missing_and_internal_parents_do_not_create_public_results(self) -> None:
        orphan = field_record("orphan", "missing-parent")
        invalid_parent = field_record("missing-parent", "another-parent")
        unrelated = {"id": "unrelated", "memory_kind": "fact"}
        result = read_search_records(
            ["orphan"], read_entries=lambda _: [orphan],
            read_parents=lambda _: [invalid_parent, unrelated],
        )
        self.assertEqual(result.direct_by_id, {"orphan": orphan})
        self.assertEqual(result.public_by_id, {})

    def test_empty_hits_do_not_call_either_reader(self) -> None:
        read_entries, read_parents = Mock(), Mock()
        result = read_search_records([], read_entries=read_entries, read_parents=read_parents)
        read_entries.assert_not_called()
        read_parents.assert_not_called()
        self.assertEqual(result.direct_by_id, {})
        self.assertEqual(result.public_by_id, {})


if __name__ == "__main__":
    unittest.main()
