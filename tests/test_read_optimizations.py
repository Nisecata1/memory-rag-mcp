"""Read whole records only when callers actually need dialogue contents."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

import server
import server_utils as utils
from memory_store import SQLiteMemoryStore


class ReadOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.data = self.root / 'data'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in {
            'AI_MEMORY_DIR': self.root,
            'DATA_DIR': self.data,
            'SQLITE_PATH': self.root / 'memory.db',
            'TIMELINE_PATH': self.data / 'timeline.md',
            'EMBEDDINGS_PATH': self.data / 'memory_embeddings.npy',
            'INDEX_PATH': self.data / 'memory.faiss',
            'META_PATH': self.data / 'meta.json',
            'REBUILD_STATE_PATH': self.data / 'rebuild.json',
            'WRITE_LOCK_PATH': self.data / '.write.lock',
        }.items():
            self.stack.enter_context(patch.object(utils, name, value))
        utils.ensure_sqlite_store_ready()

    def seed_chat(self, message_count: int = 1000) -> tuple[dict, dict]:
        raw = {
            'schema_version': 1,
            'metadata': {'session_id': 'fixture-session', 'note': 'keep metadata'},
            'messages': [
                {'message_id': f'msg-{index}', 'role': 'user' if index % 2 == 0 else 'assistant',
                 'content': f'original {index}\n' + 'verbatim content ' * 12,
                 'metadata': {'position': index}}
                for index in range(message_count)
            ],
        }
        entry = {
            'id': 'eventMemId-read-fixture',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-02T00:00:00',
            **utils.build_payload(memory_kind='chat_event', title='fixture title',
                                  short_summary='fixture summary', raw_dialogue=raw,
                                  tags=['fixture'], validate_reference_doc_path=False),
        }
        field = utils.build_chat_dialogue_field_record_entry(
            entry['id'],
            {'message_id': 'msg-20', 'message_index': 20, 'role': 'user',
             'content': raw['messages'][20]['content'], 'chunk_index': 0, 'chunk_count': 1},
            entry['created_at'], entry['updated_at'],
        )
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries([entry, field], connection=connection)
            connection.commit()
        return entry, field

    def test_light_projections_match_complete_records_timeline_and_index_order(self) -> None:
        chat, dialogue_field = self.seed_chat(30)
        registry_id = 'projRegId-differential-parent'
        times = {'created_at': '2026-03-01T08:00:00', 'updated_at': '2026-03-04T09:00:00'}
        paths = ['C:/本地会话/会话甲/transcript.jsonl', 'C:/资料/数据库设计.md']
        registry = {
            'id': registry_id, **times,
            **utils.build_payload(
                memory_kind='project_registry', title='数据库整理项目',
                overview_summary='标题：数据库整理项目\n摘要：保留中文与项目关联的完整说明。',
                tags=['中文标签', '数据库'], source_paths=paths,
                reference_doc_path='C:/资料/项目说明.md', validate_reference_doc_path=False,
            ),
        }
        project = {
            'id': 'projMemId-differential-linked', **times,
            **utils.build_payload(
                memory_kind='project_record', title='读取边界改进',
                short_summary='按用途查询', detailed_summary='完整记录保留项目编号和中文字段。',
                problem_background='原文读取过多', analysis='按用途分离字段',
                action_steps='保留存储接口', validation_result='字段与排序一致',
                project_id=registry_id, tags=['优化', '中文标签'], source_paths=paths,
                reference_doc_path='C:/资料/读取验证.md', validate_reference_doc_path=False,
            ),
        }
        fact = {
            'id': 'factMemId-differential-known',
            'created_at': times['created_at'], 'updated_at': '2026-03-05T10:00:00',
            **utils.build_payload(
                memory_kind='fact', title='中文事实', short_summary='保留来源',
                detailed_summary='事实详情和原始路径均应保持。',
                tags=['事实', '中文标签'], source_paths=paths,
                reference_doc_path='C:/资料/事实依据.md', validate_reference_doc_path=False,
            ),
        }
        # A legacy empty row is valid at the database level but must not acquire
        # an index entry merely because the optimized query now reads fewer fields.
        empty = {
            **fact, 'id': 'factMemId-differential-empty', 'title': '', 'short_summary': '',
            'detailed_summary': '', 'tags': [], 'source_paths': [],
            'reference_doc_path': '', 'retrieval_fields': [],
        }
        tags_field = utils.build_one_hot_field_record_entry(
            source_memory_id=project['id'], source_field_name='tags',
            field_value=project['tags'], created_at=times['created_at'], updated_at=times['updated_at'],
        )
        with closing(utils.get_sqlite_connection()) as connection:
            utils.upsert_store_entries([registry, project, fact, empty, tags_field], connection=connection)
            connection.commit()

        ids = [project['id'], chat['id'], registry_id, fact['id'], dialogue_field['id'], tags_field['id'], empty['id']]
        requested = [*ids, project['id'], 'missing-differential-record']
        complete = utils.fetch_store_entries_by_ids(requested)
        lightweight = utils.fetch_search_entries_by_ids(requested)
        self.assertEqual([record['id'] for record in complete], ids)
        self.assertEqual([record['id'] for record in lightweight], ids)
        for original, projected in zip(complete, lightweight):
            with self.subTest(memory_kind=original['memory_kind'], record_id=original['id']):
                excluded = {'raw_dialogue'} if original['memory_kind'] == 'chat_event' else set()
                self.assertEqual(
                    {key: value for key, value in original.items() if key not in excluded},
                    {key: value for key, value in projected.items() if key not in excluded},
                )
        by_id = {record['id']: record for record in lightweight}
        self.assertEqual(by_id[project['id']]['project_id'], registry_id)
        self.assertEqual(by_id[registry_id]['project_id'], registry_id)
        self.assertEqual(
            by_id[registry_id]['short_summary'],
            utils.build_short_summary(registry['title'], registry['overview_summary']),
        )
        self.assertTrue(by_id[registry_id]['short_summary'])
        self.assertEqual(by_id[project['id']]['source_paths'], paths)
        self.assertEqual(by_id[tags_field['id']]['tags'], project['tags'])

        with patch.object(utils, 'now_iso', return_value='2026-10-29T12:34:56'):
            expected_timeline = utils.build_timeline_markdown(complete)
            actual_timeline = utils.build_timeline_markdown_from_batches(
                utils.iter_public_entries_for_timeline(batch_size=2)
            )
        self.assertEqual(actual_timeline, expected_timeline)
        self.assertIn('读取边界改进', actual_timeline)
        self.assertIn('C:/资料/数据库设计.md', actual_timeline)

        expected_index_ids = [
            entry_id for entry_id, _, _ in utils.build_indexable_entry_payloads(utils.sort_entries(complete))
        ]
        self.assertNotIn(empty['id'], expected_index_ids)
        self.assertEqual(utils.fetch_ordered_store_entry_ids(), expected_index_ids)

    def test_light_reads_do_not_decode_or_fetch_full_dialogue(self) -> None:
        entry, field = self.seed_chat()
        with patch.object(SQLiteMemoryStore, 'load_raw_dialogue', side_effect=AssertionError('full dialogue decoded')):
            direct = utils.fetch_search_entries_by_ids([field['id'], entry['id']])
            parents = utils.fetch_search_parents_by_ids([entry['id']])
            summaries = utils.fetch_public_timeline_summary_entries()
            timeline_batches = list(utils.iter_public_entries_for_timeline())
            timeline = utils.build_timeline_markdown_from_batches(iter(timeline_batches))
            ordered_ids = utils.fetch_ordered_store_entry_ids()
        self.assertEqual([item['id'] for item in direct], [field['id'], entry['id']])
        self.assertEqual(parents[0]['short_summary'], 'fixture summary')
        self.assertEqual(summaries[0]['id'], entry['id'])
        self.assertIn(field['id'], ordered_ids)
        self.assertIn('fixture title', timeline)
        self.assertNotIn('verbatim content', timeline)
        self.assertTrue(all(not item.get('raw_dialogue') for item in [*direct, *parents, *summaries]))

    def test_search_keeps_ranking_and_does_not_reload_present_parent(self) -> None:
        entry, field = self.seed_chat()
        self.data.mkdir(exist_ok=True)
        meta_path = self.data / 'meta.json'
        index_path = self.data / 'memory.faiss'
        index_path.touch()
        meta_path.write_text(json.dumps({
            'row_to_id': [field['id'], entry['id']], 'embedding_model_fingerprint': 'fixture',
            'retrieval_field_signature': utils.RETRIEVAL_FIELD_SIGNATURE,
            'field_record_derivation_signature': utils.FIELD_RECORD_DERIVATION_SIGNATURE, 'dim': 2,
        }), encoding='utf-8')

        class Index:
            def search(self, vector, count):
                return np.asarray([[0.9, 0.6]], dtype='float32'), np.asarray([[0, 1]], dtype='int64')

        with ExitStack() as stack:
            stack.enter_context(patch.object(server, 'META_PATH', meta_path))
            stack.enter_context(patch.object(server, 'INDEX_PATH', index_path))
            stack.enter_context(patch.object(server, 'build_embedding_model_fingerprint', return_value='fixture'))
            stack.enter_context(patch.object(server, 'encode_texts', return_value=np.ones((1, 2), dtype='float32')))
            stack.enter_context(patch.object(server.faiss, 'read_index', return_value=Index()))
            stack.enter_context(patch.object(SQLiteMemoryStore, 'load_raw_dialogue', side_effect=AssertionError('full dialogue decoded')))
            read_parents = stack.enter_context(patch.object(server, 'fetch_search_parents_by_ids', side_effect=AssertionError('parent loaded twice')))
            results = server.search_records('query', embedder=object(), model_path='fixture')
        read_parents.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], entry['id'])
        self.assertAlmostEqual(results[0]['score'], 0.9, places=6)
        self.assertEqual(results[0]['matched_message_ids'], ['msg-20'])
        self.assertNotIn('raw_dialogue', results[0])

    def test_details_decode_once_per_chat_for_multiple_anchors(self) -> None:
        entry, _ = self.seed_chat()
        original = SQLiteMemoryStore.load_raw_dialogue
        decoded: list[str] = []

        def track_decode(store, value):
            decoded.append(value)
            return original(store, value)

        with patch.object(SQLiteMemoryStore, 'load_raw_dialogue', new=track_decode):
            response = server.get_detail_records_by_ids(
                [entry['id'], entry['id']],
                matched_message_ids={entry['id']: ['msg-20', 'msg-500', 'msg-900']},
            )
        self.assertEqual(len(decoded), 1)
        context = response['records'][0]['raw_dialogue']
        self.assertEqual(context['context']['returned_messages'], 63)
        self.assertEqual(context['metadata'], entry['raw_dialogue']['metadata'])
        self.assertEqual(server.get_detail_records_by_ids([entry['id']], full_dialogue=True)['records'][0]['raw_dialogue'], entry['raw_dialogue'])

    def test_explicit_full_dialogue_search_configuration_is_preserved(self) -> None:
        entry, field = self.seed_chat(30)
        self.data.mkdir(exist_ok=True)
        meta_path = self.data / 'meta.json'
        index_path = self.data / 'memory.faiss'
        index_path.touch()

        for parent_already_loaded in (False, True):
            with self.subTest(parent_already_loaded=parent_already_loaded):
                ids = [field['id'], entry['id']] if parent_already_loaded else [field['id']]
                meta_path.write_text(json.dumps({
                    'row_to_id': ids, 'embedding_model_fingerprint': 'fixture',
                    'retrieval_field_signature': utils.RETRIEVAL_FIELD_SIGNATURE,
                    'field_record_derivation_signature': utils.FIELD_RECORD_DERIVATION_SIGNATURE, 'dim': 2,
                }), encoding='utf-8')

                class Index:
                    def search(self, vector, count):
                        return np.ones((1, count), dtype='float32'), np.arange(count).reshape(1, count)

                with ExitStack() as stack:
                    stack.enter_context(patch.object(server, 'META_PATH', meta_path))
                    stack.enter_context(patch.object(server, 'INDEX_PATH', index_path))
                    stack.enter_context(patch.object(server, 'SEARCH_RESULT_FIELDS', ['id', 'memory_kind', 'raw_dialogue']))
                    stack.enter_context(patch.object(server, 'build_embedding_model_fingerprint', return_value='fixture'))
                    stack.enter_context(patch.object(server, 'encode_texts', return_value=np.ones((1, 2), dtype='float32')))
                    stack.enter_context(patch.object(server.faiss, 'read_index', return_value=Index()))
                    stack.enter_context(patch.object(server, 'fetch_search_entries_by_ids', side_effect=AssertionError('full dialogue was explicitly requested')))
                    stack.enter_context(patch.object(server, 'fetch_search_parents_by_ids', side_effect=AssertionError('full dialogue was explicitly requested')))
                    results = server.search_records('query', embedder=object(), model_path='fixture')
                self.assertEqual(results[0]['raw_dialogue'], entry['raw_dialogue'])

    def test_light_reads_do_not_touch_dialogue_column(self) -> None:
        entry, field = self.seed_chat()
        original_connect = sqlite3.connect
        denied: list[tuple[str, str]] = []

        def connection_with_authorizer(*args, **kwargs):
            connection = original_connect(*args, **kwargs)

            def authorize(action, table, column, database, source):
                if action == sqlite3.SQLITE_READ and table == 'chat_events' and column == 'raw_dialogue':
                    denied.append((table, column))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)
            return connection

        with patch.object(sqlite3, 'connect', side_effect=connection_with_authorizer):
            utils.fetch_search_entries_by_ids([entry['id'], field['id']])
            utils.fetch_search_parents_by_ids([entry['id']])
            utils.fetch_public_timeline_summary_entries()
            list(utils.iter_public_entries_for_timeline())
            utils.fetch_ordered_store_entry_ids()
        self.assertEqual(denied, [])


if __name__ == '__main__':
    unittest.main()
