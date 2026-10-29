from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import memory_domain as domain


def dialogue(count: int) -> dict:
    return {
        "schema_version": 1,
        "metadata": {"session_id": "source", "nested": {"tags": ["original"]}},
        "messages": [
            {"message_id": f"m-{index}", "role": "user" if index % 2 == 0 else "assistant", "content": f"message {index}"}
            for index in range(count)
        ],
    }


class MemoryDomainTests(unittest.TestCase):
    def test_module_import_has_no_storage_or_runtime_dependency(self) -> None:
        original_import = builtins.__import__

        def isolated_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"server_utils", "sqlite3", "yaml", "numpy", "faiss", "memory_store"}:
                raise AssertionError(f"Domain imported runtime dependency: {name}")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("isolated_memory_domain", Path(domain.__file__))
        module = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", side_effect=isolated_import):
            spec.loader.exec_module(module)
        self.assertEqual(module.build_dialogue_context(dialogue(3))["context"]["returned_messages"], 3)

    def test_context_policy_is_explicit_and_does_not_leak_between_calls(self) -> None:
        source = dialogue(100)
        before = copy.deepcopy(source)
        narrow = domain.build_dialogue_context(source, ["m-50"], return_message_num=1)
        wide = domain.build_dialogue_context(source, ["m-50"], return_message_num=25)
        default = domain.build_dialogue_context(source, ["m-50"])
        self.assertEqual([narrow["context"]["returned_messages"], wide["context"]["returned_messages"], default["context"]["returned_messages"]], [3, 51, 21])
        self.assertEqual(source, before)
        self.assertEqual(wide["metadata"], source["metadata"])

    def test_unresolved_anchors_and_preview_preserve_read_only_contract(self) -> None:
        source = dialogue(12)
        view = domain.build_dialogue_context(source, ["missing", "missing"], return_message_num=0)
        self.assertEqual(view["context"]["reason"], "anchors_not_found")
        self.assertEqual(view["context"]["unresolved_message_ids"], ["missing"])
        self.assertEqual(view["messages"][-1], source["messages"][-1])
        with self.assertRaisesRegex(ValueError, "read-only context view"):
            domain.normalize_raw_dialogue(view)

    def test_anchor_ranking_uses_only_the_supplied_policy(self) -> None:
        scores = {"best": (0.9, 0), "near": (0.85, 1), "far": (0.7, 20), "nan": (float("nan"), 50)}
        spaced = domain.select_matched_messages(scores, default_message_top_k=9, return_message_num=1)
        self.assertEqual([item["message_id"] for item in spaced], ["best", "far"])
        tight = domain.select_matched_messages(scores, message_score_gap=0, return_message_num=0)
        self.assertEqual([item["message_id"] for item in tight], ["best"])
        all_nearby = domain.select_matched_messages(scores, default_message_top_k=2, return_message_num=0)
        self.assertEqual([item["message_id"] for item in all_nearby], ["best", "near"])

    def test_top_k_validation_keeps_boolean_and_invalid_defaults_out(self) -> None:
        for invalid in [True, False, 0, -1, 1.5, "3"]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                domain.resolve_message_top_k(default_message_top_k=invalid)
        self.assertEqual(domain.resolve_message_top_k(1, default_message_top_k=7), 1)

    def test_normalization_preserves_generated_identity_and_copies_metadata(self) -> None:
        source = {"metadata": {"nested": {"values": [1]}}, "messages": [{"role": " USER ", "content": "  line 1\r\nline 2  ", "metadata": {"tags": ["x"]}}]}
        normalized = domain.normalize_raw_dialogue(source)
        repeated = domain.normalize_raw_dialogue(copy.deepcopy(source))
        self.assertEqual(normalized, repeated)
        self.assertEqual(normalized["messages"][0]["content"], "line 1\nline 2")
        self.assertTrue(normalized["messages"][0]["message_id"].startswith("msg-"))
        normalized["metadata"]["nested"]["values"].append(2)
        normalized["messages"][0]["metadata"]["tags"].append("y")
        self.assertEqual(source["metadata"]["nested"]["values"], [1])
        self.assertEqual(source["messages"][0]["metadata"]["tags"], ["x"])

    def test_normalization_rejects_bad_metadata_and_duplicate_ids(self) -> None:
        for invalid in [float("nan"), object()]:
            with self.subTest(invalid=type(invalid)), self.assertRaisesRegex(ValueError, "JSON-serializable"):
                domain.normalize_raw_dialogue({"messages": [{"role": "user", "content": "x", "metadata": {"bad": invalid}}]})
        source = dialogue(2)
        source["messages"][1]["message_id"] = "m-0"
        with self.assertRaisesRegex(ValueError, "duplicate message_id"):
            domain.normalize_raw_dialogue(source)

    def test_schema_and_roles_can_be_supplied_without_configuration_import(self) -> None:
        normalized = domain.normalize_raw_dialogue({"schema_version": 2, "messages": [{"role": "tool", "content": "raw"}]}, schema_version=2, roles=("tool",))
        self.assertEqual(normalized["schema_version"], 2)
        self.assertEqual(normalized["messages"][0]["role"], "tool")
        self.assertEqual(domain.normalize_raw_dialogue(" old\r\ntext ", allow_legacy=True), "old\ntext")
        with self.assertRaisesRegex(ValueError, "structured object"):
            domain.normalize_raw_dialogue("old text")

    def test_summary_limit_and_json_serialization_are_independent_of_runtime(self) -> None:
        self.assertEqual(domain.build_short_summary("Title", "标题：Title\n问题：Detailed explanation", max_length=8), "Detailed...")
        self.assertEqual(domain.build_short_summary("Title", "unused", " a\n b ", max_length=20), "a b")
        self.assertEqual(domain.normalize_list([" one ", "", "one", "two"]), ["one", "two"])
        self.assertEqual(json.loads(json.dumps(domain.normalize_raw_dialogue(dialogue(2)), ensure_ascii=False)), dialogue(2))


if __name__ == "__main__":
    unittest.main()
