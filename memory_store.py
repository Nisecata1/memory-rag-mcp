"""SQLite persistence, encoding and purpose-specific projections.

This module never imports the service entry points. Paths and the pure record
policy are provided explicitly, so isolated callers cannot capture production
paths when importing the module.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Protocol

SQLITE_SCHEMA_VERSION = 4  # SQLite 主数据层自己的 schema 版本；和 memory.store_version 分开维护。


CHAT_EVENT_DIALOGUE_FIELD_NAME = "raw_dialogue"


SQLITE_REBUILD_BATCH_SIZE = 200  # SQLite 全量重建时每批读取的记录数，避免一次性堆太多检索文本。

SQLITE_ENTRY_ORDER_BY = (
    "updated_at DESC, created_at DESC, id DESC"  # SQLite 取当前稳定顺序时沿用现有 Python 排序口径。
)

SQLITE_ENTRY_COLUMNS = (
    "id",
    "memory_kind",
    "title",
    "short_summary",
    "overview_summary",
    "detailed_summary",
    "raw_dialogue",
    "problem_background",
    "analysis",
    "action_steps",
    "validation_result",
    "reference_doc_path",
    "project_id",
    "created_at",
    "updated_at",
    "source_memory_id",
    "source_field_name",
    "tags_json",
    "source_paths_json",
    "retrieval_fields_json",
)  # SQLite 主表完整列顺序；读写转换都统一沿用这一份定义。

SQLITE_REGISTRY_COLUMNS = (
    "id",
    "memory_kind",
    "created_at",
    "updated_at",
)

SQLITE_PROJECT_DETAIL_COLUMNS = (
    "id",
    "title",
    "short_summary",
    "detailed_summary",
    "problem_background",
    "analysis",
    "action_steps",
    "validation_result",
    "reference_doc_path",
    "tags_json",
    "source_paths_json",
    "retrieval_fields_json",
)

SQLITE_PROJECT_REGISTRY_COLUMNS = (
    "id",
    "title",
    "overview_summary",
    "reference_doc_path",
    "project_id",
    "tags_json",
    "source_paths_json",
    "retrieval_fields_json",
)

SQLITE_LIGHT_DETAIL_COLUMNS = (
    "id",
    "title",
    "short_summary",
    "detailed_summary",
    "reference_doc_path",
    "tags_json",
    "source_paths_json",
    "retrieval_fields_json",
)

SQLITE_FIELD_RECORD_COLUMNS = (
    "id",
    "source_memory_id",
    "source_field_name",
    "source_field_value_text",
    "source_field_value_json",
    "created_at",
    "updated_at",
)

PUBLIC_MEMORY_VIEW = "public_memory_view"

ALL_MEMORY_VIEW = "all_memory_view"

DETAIL_MEMORY_KIND_BY_ID_PREFIX = (
    ("projRegisterMemId-", "project_registry"),
    ("projRegId-", "project_registry"),
    ("projMemId-", "project_record"),
    ("eventMemId-", "chat_event"),
    ("factMemId-", "fact"),
    ("proj-", "project_registry"),
)

DETAIL_TABLE_BY_MEMORY_KIND = {
    "project_record": "project_records",
    "chat_event": "chat_events",
    "fact": "facts",
    "project_registry": "project_registry",
}

DETAIL_COLUMNS_BY_MEMORY_KIND = {
    "project_record": (
        "title",
        "short_summary",
        "detailed_summary",
        "problem_background",
        "analysis",
        "action_steps",
        "validation_result",
        "reference_doc_path",
        "project_id",
        "tags_json",
        "source_paths_json",
    ),
    "chat_event": (
        "title",
        "short_summary",
        "raw_dialogue",
        "reference_doc_path",
        "tags_json",
        "source_paths_json",
    ),
    "fact": (
        "title",
        "short_summary",
        "detailed_summary",
        "reference_doc_path",
        "tags_json",
        "source_paths_json",
    ),
    "project_registry": (
        "title",
        "overview_summary",
        "reference_doc_path",
        "tags_json",
        "source_paths_json",
    ),
}


@dataclass(frozen=True)
class StorePolicy:
    normalize_list: Callable[..., list[str]]
    normalize_raw_dialogue: Callable[..., Any]
    normalize_record_ids: Callable[..., list[str]]
    build_short_summary: Callable[..., str]
    collapse_text: Callable[..., str]
    sort_entries: Callable[..., list[dict[str, Any]]]
    build_indexable_entry_payloads: Callable[..., Any]
    is_field_record_entry: Callable[..., bool]
    supported_field_names: tuple[str, ...]


@dataclass(frozen=True)
class StoreTransaction:
    """Opaque handle; only the SQLite adapter unwraps its connection."""

    _connection: sqlite3.Connection


def unwrap_connection(connection: sqlite3.Connection | StoreTransaction | None) -> sqlite3.Connection | None:
    return connection._connection if isinstance(connection, StoreTransaction) else connection


class MemoryStore(Protocol):
    def fetch_detail_records_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]: ...
    def fetch_search_entries_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]: ...
    def fetch_search_parents_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]: ...
    def iter_public_entries_for_timeline(self, batch_size: int = 200) -> Iterator[list[dict[str, Any]]]: ...
    def transaction(self) -> ContextManager[StoreTransaction]: ...


class SQLiteMemoryStore:
    def __init__(
        self,
        path: Path,
        policy: StorePolicy,
        *,
        directories: tuple[Path, ...] = (),
        init_lock: Any = None,
        schema_version: int = SQLITE_SCHEMA_VERSION,
    ):
        self.path = Path(path)
        self.policy = policy
        self.directories = tuple(Path(directory) for directory in directories)
        self.init_lock = init_lock if init_lock is not None else threading.Lock()
        self.schema_version = schema_version

    def get_sqlite_connection(self) -> sqlite3.Connection:
        for directory in (*self.directories, self.path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[StoreTransaction]:
        with closing(self.get_sqlite_connection()) as connection:
            try:
                self.initialize_sqlite_schema(connection)
                yield StoreTransaction(connection)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def resolve_field_record_value(self, entry: dict[str, Any], source_field_name: str) -> Any:
        return self.load_field_record_value_json(
            source_field_name, entry.get("source_field_value_json"), entry.get("source_field_value_text")
        )

    def _light_public_query(self, columns: tuple[str, ...] | None = None) -> str:
        """Use physical tables, so even SQL authorization never touches raw JSON.

        Projecting a wide view is insufficient: its definition can still cause
        SQLite's authorizer to visit the original dialogue column. The chat
        compatibility detailed_summary column is also deliberately excluded.
        """
        columns = columns or tuple(name for name in SQLITE_ENTRY_COLUMNS if name != "raw_dialogue")
        projections = []
        detail_columns = {
            "project_record": set(SQLITE_PROJECT_DETAIL_COLUMNS) | {"project_id"},
            "project_registry": set(SQLITE_PROJECT_REGISTRY_COLUMNS) - {"project_id"},
            "chat_event": set(SQLITE_LIGHT_DETAIL_COLUMNS) - {"detailed_summary"},
            "fact": set(SQLITE_LIGHT_DETAIL_COLUMNS),
        }
        for kind, table in DETAIL_TABLE_BY_MEMORY_KIND.items():
            expressions = []
            for name in columns:
                if name in SQLITE_REGISTRY_COLUMNS:
                    expression = f"registry.{name}"
                elif name in detail_columns[kind]:
                    expression = f"detail.{name}"
                elif name == "project_id" and kind == "project_registry":
                    expression = "registry.id"
                elif name in {"tags_json", "source_paths_json", "retrieval_fields_json"}:
                    expression = "'[]'"
                else:
                    expression = "''"
                expressions.append(f"{expression} AS {name}")
            expressions.extend(["'' AS source_field_value_text", "'' AS source_field_value_json"])
            projections.append(
                f"SELECT {', '.join(expressions)} FROM memory_registry AS registry "
                f"INNER JOIN {table} AS detail ON detail.id = registry.id "
                f"WHERE registry.memory_kind = '{kind}'"
            )
        return " UNION ALL ".join(projections)

    def _light_field_query(self) -> str:
        columns = [name for name in SQLITE_ENTRY_COLUMNS if name != "raw_dialogue"]
        expressions = []
        for name in columns:
            if name == "memory_kind":
                expression = "'field_record'"
            elif name in SQLITE_FIELD_RECORD_COLUMNS:
                expression = name
            elif name in {"tags_json", "source_paths_json", "retrieval_fields_json"}:
                expression = "'[]'"
            else:
                expression = "''"
            expressions.append(f"{expression} AS {name}")
        expressions.extend(["source_field_value_text", "source_field_value_json"])
        return f"SELECT {', '.join(expressions)} FROM field_records"

    def fetch_search_parents_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]:
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        self.ensure_sqlite_store_ready()
        placeholders = ",".join("?" for _ in normalized_ids)
        with closing(self.get_sqlite_connection()) as connection:
            rows = connection.execute(
                f"SELECT * FROM ({self._light_public_query()}) WHERE id IN ({placeholders})",
                normalized_ids,
            ).fetchall()
        entries = [self.build_entry_from_sqlite_row(row, include_dialogue=False) for row in rows]
        by_id = {entry["id"]: entry for entry in entries}
        return [by_id[record_id] for record_id in normalized_ids if record_id in by_id]

    def fetch_search_entries_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]:
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        public_entries = self.fetch_search_parents_by_ids(normalized_ids)
        field_entries = self.fetch_field_record_entries_by_ids(normalized_ids)
        by_id = {entry["id"]: entry for entry in [*public_entries, *field_entries]}
        return [by_id[record_id] for record_id in normalized_ids if record_id in by_id]

    def iter_index_entries_in_batches(
        self, batch_size: int = SQLITE_REBUILD_BATCH_SIZE
    ) -> Iterator[list[dict[str, Any]]]:
        self.ensure_sqlite_store_ready()
        offset = 0
        size = max(1, int(batch_size or SQLITE_REBUILD_BATCH_SIZE))
        query = self._light_public_query() + " UNION ALL " + self._light_field_query()
        while True:
            with closing(self.get_sqlite_connection()) as connection:
                rows = connection.execute(
                    f"SELECT * FROM ({query}) ORDER BY {SQLITE_ENTRY_ORDER_BY} LIMIT ? OFFSET ?",
                    (size, offset),
                ).fetchall()
            if not rows:
                return
            yield [self.build_entry_from_sqlite_row(row, include_dialogue=False) for row in rows]
            offset += size

    def fetch_all_public_entries(
        self, connection: sqlite3.Connection | StoreTransaction
    ) -> list[dict[str, Any]]:
        db = unwrap_connection(connection)
        rows = db.execute(
            f"SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json "
            f"FROM {PUBLIC_MEMORY_VIEW} ORDER BY {SQLITE_ENTRY_ORDER_BY}"
        ).fetchall()
        return [self.build_entry_from_sqlite_row(row) for row in rows]

    def clear_field_records(self, connection: sqlite3.Connection | StoreTransaction) -> None:
        unwrap_connection(connection).execute("DELETE FROM field_records")

    def dump_json_string_list(self, values: list[str] | None) -> str:
        return json.dumps(self.policy.normalize_list(values), ensure_ascii=False)

    def load_json_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return self.policy.normalize_list([str(item or "").strip() for item in value])
        raw_text = str(value or "").strip()
        if not raw_text:
            return []
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return self.policy.normalize_list([str(item or "").strip() for item in parsed])

    def dump_raw_dialogue(self, value: str | dict[str, Any] | None) -> str:
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value or "")

    def load_raw_dialogue(self, value: Any) -> str | dict[str, Any]:
        raw_text = str(value or "")
        if not raw_text.strip():
            return ""
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
        if not isinstance(parsed, dict):
            return raw_text
        try:
            normalized = self.policy.normalize_raw_dialogue(parsed, allow_legacy=False)
        except ValueError:
            return raw_text
        return normalized if isinstance(normalized, dict) else raw_text

    def dump_field_record_value_json(self, field_value: str | list[str] | dict[str, Any]) -> str:
        if isinstance(field_value, dict):
            return json.dumps(field_value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(field_value, list):
            return json.dumps(self.policy.normalize_list(field_value), ensure_ascii=False)
        return json.dumps(str(field_value or ""), ensure_ascii=False)

    def render_field_record_value_text(self, field_value: str | list[str] | dict[str, Any]) -> str:
        if isinstance(field_value, dict):
            role = str(field_value.get("role") or "").strip()
            content = str(field_value.get("content") or "")
            return f"{role}: {content}" if role else content
        if isinstance(field_value, list):
            return "，".join(self.policy.normalize_list(field_value))
        return str(field_value or "")

    def load_field_record_value_json(
        self, source_field_name: str, json_text: Any, fallback_text: Any
    ) -> str | list[str] | dict[str, Any]:
        raw_json_text = str(json_text or "").strip()
        if raw_json_text:
            try:
                parsed = json.loads(raw_json_text)
            except json.JSONDecodeError:
                parsed = None
            if source_field_name == CHAT_EVENT_DIALOGUE_FIELD_NAME:
                if isinstance(parsed, dict):
                    return parsed
            elif source_field_name == "tags":
                if isinstance(parsed, list):
                    return self.policy.normalize_list([str(item or "").strip() for item in parsed])
            elif isinstance(parsed, str):
                return str(parsed or "")
        if source_field_name == CHAT_EVENT_DIALOGUE_FIELD_NAME:
            return {"role": "", "content": str(fallback_text or "")}
        if source_field_name == "tags":
            return self.policy.normalize_list(str(fallback_text or "").split("，"))
        return str(fallback_text or "")

    def build_sqlite_entry_row(self, entry: dict[str, Any]) -> dict[str, Any]:
        normalized_memory_kind = str(entry.get("memory_kind") or "").strip()
        base_row = {
            "id": str(entry.get("id") or "").strip(),
            "memory_kind": normalized_memory_kind,
            "created_at": str(entry.get("created_at") or "").strip(),
            "updated_at": str(entry.get("updated_at") or "").strip(),
        }
        if normalized_memory_kind == "project_record":
            return {
                **base_row,
                "title": str(entry.get("title") or ""),
                "short_summary": str(entry.get("short_summary") or ""),
                "detailed_summary": str(entry.get("detailed_summary") or ""),
                "problem_background": str(entry.get("problem_background") or ""),
                "analysis": str(entry.get("analysis") or ""),
                "action_steps": str(entry.get("action_steps") or ""),
                "validation_result": str(entry.get("validation_result") or ""),
                "reference_doc_path": str(entry.get("reference_doc_path") or ""),
                "project_id": str(entry.get("project_id") or "").strip() or None,
                "tags_json": self.dump_json_string_list(entry.get("tags")),
                "source_paths_json": self.dump_json_string_list(entry.get("source_paths")),
                "retrieval_fields_json": self.dump_json_string_list(entry.get("retrieval_fields")),
            }
        if normalized_memory_kind == "project_registry":
            return {
                **base_row,
                "title": str(entry.get("title") or ""),
                "overview_summary": str(entry.get("overview_summary") or ""),
                "reference_doc_path": str(entry.get("reference_doc_path") or ""),
                "tags_json": self.dump_json_string_list(entry.get("tags")),
                "source_paths_json": self.dump_json_string_list(entry.get("source_paths")),
                "retrieval_fields_json": self.dump_json_string_list(entry.get("retrieval_fields")),
            }
        if normalized_memory_kind == "chat_event":
            return {
                **base_row,
                "title": str(entry.get("title") or ""),
                "short_summary": str(entry.get("short_summary") or ""),
                "detailed_summary": "",
                "raw_dialogue": self.dump_raw_dialogue(entry.get("raw_dialogue")),
                "reference_doc_path": str(entry.get("reference_doc_path") or ""),
                "tags_json": self.dump_json_string_list(entry.get("tags")),
                "source_paths_json": self.dump_json_string_list(entry.get("source_paths")),
                "retrieval_fields_json": self.dump_json_string_list(entry.get("retrieval_fields")),
            }
        if normalized_memory_kind == "fact":
            return {
                **base_row,
                "title": str(entry.get("title") or ""),
                "short_summary": str(entry.get("short_summary") or ""),
                "detailed_summary": str(entry.get("detailed_summary") or ""),
                "reference_doc_path": str(entry.get("reference_doc_path") or ""),
                "tags_json": self.dump_json_string_list(entry.get("tags")),
                "source_paths_json": self.dump_json_string_list(entry.get("source_paths")),
                "retrieval_fields_json": self.dump_json_string_list(entry.get("retrieval_fields")),
            }
        if normalized_memory_kind == "field_record":
            normalized_source_field_name = str(entry.get("source_field_name") or "").strip()
            field_value = self.resolve_field_record_value(entry, normalized_source_field_name)
            if normalized_source_field_name == CHAT_EVENT_DIALOGUE_FIELD_NAME:
                source_field_value_text = str(entry.get("source_field_value_text") or "").strip()
            else:
                source_field_value_text = self.render_field_record_value_text(field_value)
            return {
                **base_row,
                "source_memory_id": str(entry.get("source_memory_id") or "").strip(),
                "source_field_name": normalized_source_field_name,
                "source_field_value_text": source_field_value_text,
                "source_field_value_json": self.dump_field_record_value_json(field_value),
            }
        raise RuntimeError(f"Unsupported memory_kind for SQLite row conversion: {normalized_memory_kind}")

    def build_entry_from_sqlite_row(
        self, row: sqlite3.Row | dict[str, Any], *, include_dialogue: bool = True
    ) -> dict[str, Any]:
        row_data = dict(row)
        normalized_memory_kind = str(row_data.get("memory_kind") or "").strip()
        normalized_title = str(row_data.get("title") or "")
        normalized_overview_summary = str(row_data.get("overview_summary") or "")
        normalized_short_summary = str(row_data.get("short_summary") or "")
        normalized_project_id = str(row_data.get("project_id") or "").strip()
        common_entry = {
            "id": str(row_data.get("id") or "").strip(),
            "memory_kind": normalized_memory_kind,
            "created_at": str(row_data.get("created_at") or "").strip(),
            "updated_at": str(row_data.get("updated_at") or "").strip(),
        }
        if normalized_memory_kind == "field_record":
            source_field_name = str(row_data.get("source_field_name") or "").strip()
            field_value = self.resolve_field_record_value(row_data, source_field_name)
            field_entry = {
                **common_entry,
                "title": "",
                "short_summary": "",
                "overview_summary": "",
                "detailed_summary": "",
                "raw_dialogue": "",
                "problem_background": "",
                "analysis": "",
                "action_steps": "",
                "validation_result": "",
                "reference_doc_path": field_value if source_field_name == "reference_doc_path" else "",
                "project_id": "",
                "source_memory_id": str(row_data.get("source_memory_id") or "").strip(),
                "source_field_name": source_field_name,
                "tags": (
                    list(field_value) if source_field_name == "tags" and isinstance(field_value, list) else []
                ),
                "source_paths": [],
                "retrieval_fields": [source_field_name] if source_field_name else [],
                "source_field_value_text": str(row_data.get("source_field_value_text") or ""),
                "source_field_value_json": str(row_data.get("source_field_value_json") or ""),
            }
            if source_field_name != CHAT_EVENT_DIALOGUE_FIELD_NAME:
                field_entry[source_field_name] = field_value
            return field_entry
        if normalized_memory_kind == "project_registry" and (not normalized_short_summary):
            normalized_short_summary = self.policy.build_short_summary(
                title=normalized_title, detailed_summary=normalized_overview_summary
            )
        normalized_detailed_summary = str(row_data.get("detailed_summary") or "")
        if normalized_memory_kind == "project_registry" and (not normalized_detailed_summary):
            normalized_detailed_summary = normalized_overview_summary
        if normalized_memory_kind == "chat_event":
            normalized_detailed_summary = ""
        return {
            **common_entry,
            "title": normalized_title,
            "short_summary": normalized_short_summary,
            "overview_summary": normalized_overview_summary,
            "detailed_summary": normalized_detailed_summary,
            "raw_dialogue": (
                self.load_raw_dialogue(row_data.get("raw_dialogue"))
                if normalized_memory_kind == "chat_event" and include_dialogue
                else ""
            ),
            "problem_background": str(row_data.get("problem_background") or ""),
            "analysis": str(row_data.get("analysis") or ""),
            "action_steps": str(row_data.get("action_steps") or ""),
            "validation_result": str(row_data.get("validation_result") or ""),
            "reference_doc_path": str(row_data.get("reference_doc_path") or ""),
            "project_id": normalized_project_id
            or (common_entry["id"] if normalized_memory_kind == "project_registry" else ""),
            "source_memory_id": "",
            "source_field_name": "",
            "tags": self.load_json_string_list(row_data.get("tags_json")),
            "source_paths": self.load_json_string_list(row_data.get("source_paths_json")),
            "retrieval_fields": self.load_json_string_list(row_data.get("retrieval_fields_json")),
        }

    def ensure_supported_sqlite_schema_version(
        self,
        connection: sqlite3.Connection,
        allowed_versions: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> int:
        resolved_allowed_versions = (
            {0, self.schema_version}
            if allowed_versions is None
            else {int(version) for version in allowed_versions}
        )
        current_schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        if current_schema_version not in resolved_allowed_versions:
            expected_versions = ", ".join((str(version) for version in sorted(resolved_allowed_versions)))
            raise RuntimeError(
                f"Unsupported SQLite schema version {current_schema_version}; expected one of: {expected_versions}."
            )
        return current_schema_version

    def create_split_sqlite_schema_objects(self, connection: sqlite3.Connection) -> None:
        connection.execute("""
        CREATE TABLE IF NOT EXISTS memory_registry (
            id TEXT PRIMARY KEY,
            memory_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        connection.execute("""
        CREATE TABLE IF NOT EXISTS project_records (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            short_summary TEXT NOT NULL,
            detailed_summary TEXT NOT NULL,
            problem_background TEXT NOT NULL,
            analysis TEXT NOT NULL,
            action_steps TEXT NOT NULL,
            validation_result TEXT NOT NULL,
            reference_doc_path TEXT NOT NULL,
            project_id TEXT,
            tags_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            retrieval_fields_json TEXT NOT NULL
        )
        """)
        connection.execute("""
        CREATE TABLE IF NOT EXISTS project_registry (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            overview_summary TEXT NOT NULL,
            reference_doc_path TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            retrieval_fields_json TEXT NOT NULL
        )
        """)
        connection.execute("""
        CREATE TABLE IF NOT EXISTS chat_events (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            short_summary TEXT NOT NULL,
            detailed_summary TEXT NOT NULL,
            raw_dialogue TEXT NOT NULL DEFAULT '',
            reference_doc_path TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            retrieval_fields_json TEXT NOT NULL
        )
        """)
        connection.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            short_summary TEXT NOT NULL,
            detailed_summary TEXT NOT NULL,
            reference_doc_path TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            retrieval_fields_json TEXT NOT NULL
        )
        """)
        connection.execute("""
        CREATE TABLE IF NOT EXISTS field_records (
            id TEXT PRIMARY KEY,
            source_memory_id TEXT NOT NULL,
            source_field_name TEXT NOT NULL,
            source_field_value_text TEXT NOT NULL,
            source_field_value_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_registry_memory_kind ON memory_registry(memory_kind)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_registry_updated_at ON memory_registry(updated_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_records_project_id ON project_records(project_id)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_project_registry_title ON project_registry(title)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_field_records_source_memory_id ON field_records(source_memory_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_field_records_source_field_name ON field_records(source_field_name)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_field_records_memory_and_field ON field_records(source_memory_id, source_field_name)"
        )
        connection.execute(f"DROP VIEW IF EXISTS {PUBLIC_MEMORY_VIEW}")
        connection.execute(f"""
        CREATE VIEW {PUBLIC_MEMORY_VIEW} AS
        SELECT
            registry.id AS id,
            registry.memory_kind AS memory_kind,
            detail.title AS title,
            detail.short_summary AS short_summary,
            detail.overview_summary AS overview_summary,
            detail.detailed_summary AS detailed_summary,
            detail.raw_dialogue AS raw_dialogue,
            detail.problem_background AS problem_background,
            detail.analysis AS analysis,
            detail.action_steps AS action_steps,
            detail.validation_result AS validation_result,
            detail.reference_doc_path AS reference_doc_path,
            detail.project_id AS project_id,
            registry.created_at AS created_at,
            registry.updated_at AS updated_at,
            '' AS source_memory_id,
            '' AS source_field_name,
            detail.tags_json AS tags_json,
            detail.source_paths_json AS source_paths_json,
            detail.retrieval_fields_json AS retrieval_fields_json,
            '' AS source_field_value_text,
            '' AS source_field_value_json
        FROM memory_registry AS registry
        JOIN (
            SELECT
                id,
                title,
                short_summary,
                '' AS overview_summary,
                detailed_summary,
                '' AS raw_dialogue,
                problem_background,
                analysis,
                action_steps,
                validation_result,
                reference_doc_path,
                project_id,
                tags_json,
                source_paths_json,
                retrieval_fields_json,
                'project_record' AS memory_kind
            FROM project_records
            UNION ALL
            SELECT
                id,
                title,
                short_summary,
                '' AS overview_summary,
                '' AS detailed_summary,
                raw_dialogue,
                '' AS problem_background,
                '' AS analysis,
                '' AS action_steps,
                '' AS validation_result,
                reference_doc_path,
                '' AS project_id,
                tags_json,
                source_paths_json,
                retrieval_fields_json,
                'chat_event' AS memory_kind
            FROM chat_events
            UNION ALL
            SELECT
                id,
                title,
                short_summary,
                '' AS overview_summary,
                detailed_summary,
                '' AS raw_dialogue,
                '' AS problem_background,
                '' AS analysis,
                '' AS action_steps,
                '' AS validation_result,
                reference_doc_path,
                '' AS project_id,
                tags_json,
                source_paths_json,
                retrieval_fields_json,
                'fact' AS memory_kind
            FROM facts
            UNION ALL
            SELECT
                id,
                title,
                '' AS short_summary,
                overview_summary,
                '' AS detailed_summary,
                '' AS raw_dialogue,
                '' AS problem_background,
                '' AS analysis,
                '' AS action_steps,
                '' AS validation_result,
                reference_doc_path,
                id AS project_id,
                tags_json,
                source_paths_json,
                retrieval_fields_json,
                'project_registry' AS memory_kind
            FROM project_registry
        ) AS detail
        ON registry.id = detail.id AND registry.memory_kind = detail.memory_kind
        """)
        connection.execute(f"DROP VIEW IF EXISTS {ALL_MEMORY_VIEW}")
        connection.execute(f"""
        CREATE VIEW {ALL_MEMORY_VIEW} AS
        SELECT
            id,
            memory_kind,
            title,
            short_summary,
            overview_summary,
            detailed_summary,
            raw_dialogue,
            problem_background,
            analysis,
            action_steps,
            validation_result,
            reference_doc_path,
            project_id,
            created_at,
            updated_at,
            source_memory_id,
            source_field_name,
            tags_json,
            source_paths_json,
            retrieval_fields_json,
            source_field_value_text,
            source_field_value_json
        FROM {PUBLIC_MEMORY_VIEW}
        UNION ALL
        SELECT
            field_records.id AS id,
            'field_record' AS memory_kind,
            '' AS title,
            '' AS short_summary,
            '' AS overview_summary,
            '' AS detailed_summary,
            '' AS raw_dialogue,
            '' AS problem_background,
            '' AS analysis,
            '' AS action_steps,
            '' AS validation_result,
            '' AS reference_doc_path,
            '' AS project_id,
            field_records.created_at AS created_at,
            field_records.updated_at AS updated_at,
            field_records.source_memory_id AS source_memory_id,
            field_records.source_field_name AS source_field_name,
            '[]' AS tags_json,
            '[]' AS source_paths_json,
            json_array(field_records.source_field_name) AS retrieval_fields_json,
            field_records.source_field_value_text AS source_field_value_text,
            field_records.source_field_value_json AS source_field_value_json
        FROM field_records
        """)

    def migrate_sqlite_schema_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        existing_project_record_columns = {
            str(row["name"] or "").strip()
            for row in connection.execute("PRAGMA table_info(project_records)").fetchall()
        }
        if "project_id" not in existing_project_record_columns:
            connection.execute("ALTER TABLE project_records ADD COLUMN project_id TEXT")

    def migrate_sqlite_schema_v3_to_v4(self, connection: sqlite3.Connection) -> None:
        existing_chat_event_columns = {
            str(row["name"] or "").strip()
            for row in connection.execute("PRAGMA table_info(chat_events)").fetchall()
        }
        if not existing_chat_event_columns:
            return
        if "raw_dialogue" not in existing_chat_event_columns:
            connection.execute("ALTER TABLE chat_events ADD COLUMN raw_dialogue TEXT NOT NULL DEFAULT ''")
        if "detailed_summary" in existing_chat_event_columns:
            connection.execute("""
            UPDATE chat_events
            SET raw_dialogue = detailed_summary
            WHERE raw_dialogue = ''
            """)

    def initialize_sqlite_schema(self, connection: sqlite3.Connection) -> None:
        current_schema_version = self.ensure_supported_sqlite_schema_version(
            connection, allowed_versions={0, 2, 3, self.schema_version}
        )
        if current_schema_version == 2:
            self.migrate_sqlite_schema_v2_to_v3(connection)
        if current_schema_version in {2, 3}:
            self.migrate_sqlite_schema_v3_to_v4(connection)
        self.create_split_sqlite_schema_objects(connection)
        if current_schema_version != self.schema_version:
            connection.execute(f"PRAGMA user_version = {self.schema_version}")

    def upsert_memory_registry(self, entries: list[dict[str, Any]], connection: sqlite3.Connection) -> None:
        if not entries:
            return
        connection.executemany(
            """
        INSERT OR REPLACE INTO memory_registry ( id, memory_kind, created_at, updated_at
        ) VALUES (
            :id, :memory_kind, :created_at, :updated_at
        )
        """,
            [
                {
                    "id": str(entry.get("id") or "").strip(),
                    "memory_kind": str(entry.get("memory_kind") or "").strip(),
                    "created_at": str(entry.get("created_at") or "").strip(),
                    "updated_at": str(entry.get("updated_at") or "").strip(),
                }
                for entry in entries
            ],
        )

    def upsert_public_memory_details(
        self, entries: list[dict[str, Any]], connection: sqlite3.Connection
    ) -> None:
        if not entries:
            return
        grouped_entries: dict[str, list[dict[str, Any]]] = {
            "project_record": [],
            "chat_event": [],
            "fact": [],
            "project_registry": [],
        }
        for entry in entries:
            memory_kind = str(entry.get("memory_kind") or "").strip()
            if memory_kind not in grouped_entries:
                raise RuntimeError(f"Unsupported public memory_kind for detail upsert: {memory_kind}")
            grouped_entries[memory_kind].append(self.build_sqlite_entry_row(entry))
        if grouped_entries["project_record"]:
            connection.executemany(
                """
            INSERT OR REPLACE INTO project_records (
                id, title, short_summary, detailed_summary, problem_background,
                analysis, action_steps, validation_result, reference_doc_path, project_id,
                tags_json, source_paths_json, retrieval_fields_json
            ) VALUES (
                :id, :title, :short_summary, :detailed_summary, :problem_background,
                :analysis, :action_steps, :validation_result, :reference_doc_path, :project_id,
                :tags_json, :source_paths_json, :retrieval_fields_json
            )
            """,
                grouped_entries["project_record"],
            )
        if grouped_entries["chat_event"]:
            connection.executemany(
                """
            INSERT OR REPLACE INTO chat_events (
                id, title, short_summary, detailed_summary, raw_dialogue, reference_doc_path,
                tags_json, source_paths_json, retrieval_fields_json
            ) VALUES (
                :id, :title, :short_summary, :detailed_summary, :raw_dialogue, :reference_doc_path,
                :tags_json, :source_paths_json, :retrieval_fields_json
            )
            """,
                grouped_entries["chat_event"],
            )
        if grouped_entries["fact"]:
            connection.executemany(
                """
            INSERT OR REPLACE INTO facts (
                id, title, short_summary, detailed_summary, reference_doc_path,
                tags_json, source_paths_json, retrieval_fields_json
            ) VALUES (
                :id, :title, :short_summary, :detailed_summary, :reference_doc_path,
                :tags_json, :source_paths_json, :retrieval_fields_json
            )
            """,
                grouped_entries["fact"],
            )
        if grouped_entries["project_registry"]:
            connection.executemany(
                """
            INSERT OR REPLACE INTO project_registry (
                id, title, overview_summary, reference_doc_path,
                tags_json, source_paths_json, retrieval_fields_json
            ) VALUES (
                :id, :title, :overview_summary, :reference_doc_path,
                :tags_json, :source_paths_json, :retrieval_fields_json
            )
            """,
                grouped_entries["project_registry"],
            )

    def upsert_field_record_rows(self, entries: list[dict[str, Any]], connection: sqlite3.Connection) -> None:
        if not entries:
            return
        connection.executemany(
            """
        INSERT OR REPLACE INTO field_records (
            id, source_memory_id, source_field_name, source_field_value_text, source_field_value_json, created_at, updated_at
        ) VALUES (
            :id, :source_memory_id, :source_field_name, :source_field_value_text,
            :source_field_value_json, :created_at, :updated_at
        )
        """,
            [self.build_sqlite_entry_row(entry) for entry in entries],
        )

    def fetch_public_memory_entries_by_ids(
        self, record_ids: list[str], connection: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        placeholders = ",".join(("?" for _ in normalized_ids))
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            rows = db.execute(
                f"SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json FROM {PUBLIC_MEMORY_VIEW} WHERE id IN ({placeholders})",
                normalized_ids,
            ).fetchall()
        finally:
            if owned_connection:
                db.close()
        entry_map = {
            str(entry.get("id") or "").strip(): entry
            for entry in (self.build_entry_from_sqlite_row(row) for row in rows)
        }
        return [entry_map[record_id] for record_id in normalized_ids if record_id in entry_map]

    def resolve_memory_kind_from_record_id(self, record_id: str) -> str | None:
        normalized_record_id = str(record_id or "").strip()
        for id_prefix, memory_kind in DETAIL_MEMORY_KIND_BY_ID_PREFIX:
            if normalized_record_id.startswith(id_prefix):
                return memory_kind
        return None

    def build_detail_record_from_sqlite_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        row_data = dict(row)
        record = {
            "id": str(row_data.pop("id", "") or "").strip(),
            "memory_kind": str(row_data.pop("memory_kind", "") or "").strip(),
            "created_at": str(row_data.pop("created_at", "") or "").strip(),
            "updated_at": str(row_data.pop("updated_at", "") or "").strip(),
        }
        for field_name, value in row_data.items():
            if field_name == "tags_json":
                record["tags"] = self.load_json_string_list(value)
            elif field_name == "source_paths_json":
                record["source_paths"] = self.load_json_string_list(value)
            elif field_name == "raw_dialogue" and record["memory_kind"] == "chat_event":
                record["raw_dialogue"] = self.load_raw_dialogue(value)
            else:
                record[field_name] = str(value or "")
        if record["memory_kind"] == "project_registry":
            record["project_id"] = record["id"]
        return record

    def fetch_detail_records_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]:
        self.ensure_sqlite_store_ready()
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        field_record_ids = [record_id for record_id in normalized_ids if record_id.startswith("fmem-")]
        if field_record_ids:
            raise ValueError(
                f"field_record is an internal index asset and cannot be read directly: {field_record_ids[0]}"
            )
        memory_kind_by_id = {
            record_id: self.resolve_memory_kind_from_record_id(record_id) for record_id in normalized_ids
        }
        unresolved_ids = [
            record_id for (record_id, memory_kind) in memory_kind_by_id.items() if memory_kind is None
        ]
        with closing(self.get_sqlite_connection()) as connection:
            if unresolved_ids:
                placeholders = ",".join(("?" for _ in unresolved_ids))
                rows = connection.execute(
                    f"SELECT id, memory_kind FROM memory_registry WHERE id IN ({placeholders})",
                    unresolved_ids,
                ).fetchall()
                for row in rows:
                    record_id = str(row["id"] or "").strip()
                    memory_kind = str(row["memory_kind"] or "").strip()
                    if memory_kind in DETAIL_TABLE_BY_MEMORY_KIND:
                        memory_kind_by_id[record_id] = memory_kind
            ids_by_memory_kind = {memory_kind: [] for memory_kind in DETAIL_TABLE_BY_MEMORY_KIND}
            for record_id in normalized_ids:
                memory_kind = memory_kind_by_id.get(record_id)
                if memory_kind in ids_by_memory_kind:
                    ids_by_memory_kind[memory_kind].append(record_id)
            detail_record_map: dict[str, dict[str, Any]] = {}
            for memory_kind, ids_for_kind in ids_by_memory_kind.items():
                if not ids_for_kind:
                    continue
                table_name = DETAIL_TABLE_BY_MEMORY_KIND[memory_kind]
                detail_columns = DETAIL_COLUMNS_BY_MEMORY_KIND[memory_kind]
                selected_detail_columns = ", ".join(
                    (f"detail.{column_name}" for column_name in detail_columns)
                )
                placeholders = ",".join(("?" for _ in ids_for_kind))
                rows = connection.execute(
                    f"""
                SELECT
                    registry.id AS id,
                    registry.memory_kind AS memory_kind,
                    registry.created_at AS created_at,
                    registry.updated_at AS updated_at,
                    {selected_detail_columns}
                FROM memory_registry AS registry
                INNER JOIN {table_name} AS detail ON detail.id = registry.id
                WHERE registry.memory_kind = ? AND registry.id IN ({placeholders})
                """,
                    [memory_kind, *ids_for_kind],
                ).fetchall()
                for row in rows:
                    detail_record = self.build_detail_record_from_sqlite_row(row)
                    detail_record_map[detail_record["id"]] = detail_record
        return [
            detail_record_map[record_id] for record_id in normalized_ids if record_id in detail_record_map
        ]

    def fetch_field_record_entries_by_ids(
        self, record_ids: list[str], connection: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        placeholders = ",".join(("?" for _ in normalized_ids))
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            rows = db.execute(
                f"""
            SELECT
                id,
                'field_record' AS memory_kind,
                '' AS title,
                '' AS short_summary,
                '' AS overview_summary,
                '' AS detailed_summary,
                '' AS raw_dialogue,
                '' AS problem_background,
                '' AS analysis,
                '' AS action_steps,
                '' AS validation_result,
                '' AS reference_doc_path,
                '' AS project_id,
                created_at,
                updated_at,
                source_memory_id,
                source_field_name,
                '[]' AS tags_json,
                '[]' AS source_paths_json,
                json_array(source_field_name) AS retrieval_fields_json,
                source_field_value_text,
                source_field_value_json
            FROM field_records
            WHERE id IN ({placeholders})
            """,
                normalized_ids,
            ).fetchall()
        finally:
            if owned_connection:
                db.close()
        entry_map = {
            str(entry.get("id") or "").strip(): entry
            for entry in (self.build_entry_from_sqlite_row(row) for row in rows)
        }
        return [entry_map[record_id] for record_id in normalized_ids if record_id in entry_map]

    def upsert_store_entries(
        self, entries: list[dict[str, Any]], connection: sqlite3.Connection | None = None
    ) -> None:
        if not entries:
            return
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            public_entries = [entry for entry in entries if not self.policy.is_field_record_entry(entry)]
            field_entries = [entry for entry in entries if self.policy.is_field_record_entry(entry)]
            if public_entries:
                self.upsert_memory_registry(public_entries, connection=db)
                self.upsert_public_memory_details(public_entries, connection=db)
            if field_entries:
                self.upsert_field_record_rows(field_entries, connection=db)
            if owned_connection:
                db.commit()
        finally:
            if owned_connection:
                db.close()

    def delete_store_entries_by_ids(
        self, record_ids: list[str], connection: sqlite3.Connection | None = None
    ) -> None:
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return
        placeholders = ",".join(("?" for _ in normalized_ids))
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            registry_rows = db.execute(
                f"SELECT id, memory_kind FROM memory_registry WHERE id IN ({placeholders})", normalized_ids
            ).fetchall()
            ids_by_kind: dict[str, list[str]] = {
                "project_record": [],
                "chat_event": [],
                "fact": [],
                "project_registry": [],
            }
            for row in registry_rows:
                memory_kind = str(row["memory_kind"] or "").strip()
                if memory_kind in ids_by_kind:
                    ids_by_kind[memory_kind].append(str(row["id"] or "").strip())
            for memory_kind, ids_for_kind in ids_by_kind.items():
                if not ids_for_kind:
                    continue
                detail_placeholders = ",".join(("?" for _ in ids_for_kind))
                target_table = {
                    "project_record": "project_records",
                    "chat_event": "chat_events",
                    "fact": "facts",
                    "project_registry": "project_registry",
                }[memory_kind]
                db.execute(f"DELETE FROM {target_table} WHERE id IN ({detail_placeholders})", ids_for_kind)
            db.execute(f"DELETE FROM memory_registry WHERE id IN ({placeholders})", normalized_ids)
            if owned_connection:
                db.commit()
        finally:
            if owned_connection:
                db.close()

    def delete_field_records_by_source_ids(
        self, source_ids: list[str], connection: sqlite3.Connection | None = None
    ) -> None:
        normalized_source_ids = self.policy.normalize_record_ids(source_ids)
        placeholders = ",".join(("?" for _ in normalized_source_ids))
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            db.execute(
                f"""
            DELETE FROM field_records
            WHERE source_memory_id IN ({placeholders})
            """,
                normalized_source_ids,
            )
            if owned_connection:
                db.commit()
        finally:
            if owned_connection:
                db.close()

    def delete_field_records_by_source_and_fields(
        self,
        source_memory_id: str,
        field_names: list[str] | tuple[str, ...] | set[str],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        normalized_source_memory_id = str(source_memory_id or "").strip()
        normalized_field_name_set = {
            str(field_name or "").strip()
            for field_name in field_names
            if str(field_name or "").strip() in self.policy.supported_field_names
        }
        normalized_field_names = [
            field_name
            for field_name in self.policy.supported_field_names
            if field_name in normalized_field_name_set
        ]
        if not normalized_source_memory_id or not normalized_field_names:
            return
        placeholders = ",".join(("?" for _ in normalized_field_names))
        owned_connection = connection is None
        if owned_connection:
            self.ensure_sqlite_store_ready()
        db = connection or self.get_sqlite_connection()
        try:
            db.execute(
                f"""
            DELETE FROM field_records
            WHERE source_memory_id = ?
              AND source_field_name IN ({placeholders})
            """,
                [normalized_source_memory_id, *normalized_field_names],
            )
            if owned_connection:
                db.commit()
        finally:
            if owned_connection:
                db.close()

    def fetch_field_record_ids_by_source_and_fields(
        self,
        source_memory_id: str,
        field_names: list[str] | tuple[str, ...] | set[str],
        connection: sqlite3.Connection,
    ) -> list[str]:
        normalized_source_memory_id = str(source_memory_id or "").strip()
        normalized_field_names = [
            field_name
            for field_name in self.policy.supported_field_names
            if field_name in {str(item or "").strip() for item in field_names}
        ]
        if not normalized_source_memory_id or not normalized_field_names:
            return []
        placeholders = ",".join(("?" for _ in normalized_field_names))
        rows = connection.execute(
            f"""
        SELECT id
        FROM field_records
        WHERE source_memory_id = ?
          AND source_field_name IN ({placeholders})
        ORDER BY id
        """,
            [normalized_source_memory_id, *normalized_field_names],
        ).fetchall()
        return [str(row["id"] or "").strip() for row in rows if str(row["id"] or "").strip()]

    def fetch_store_entries_by_ids(self, record_ids: list[str]) -> list[dict[str, Any]]:
        self.ensure_sqlite_store_ready()
        normalized_ids = self.policy.normalize_record_ids(record_ids)
        if not normalized_ids:
            return []
        public_entries = self.fetch_public_memory_entries_by_ids(normalized_ids)
        field_entries = self.fetch_field_record_entries_by_ids(normalized_ids)
        entry_map = {str(entry.get("id") or "").strip(): entry for entry in [*public_entries, *field_entries]}
        return [entry_map[record_id] for record_id in normalized_ids if record_id in entry_map]

    def fetch_store_entry_by_id(self, record_id: str) -> dict[str, Any] | None:
        self.ensure_sqlite_store_ready()
        normalized_record_id = str(record_id or "").strip()
        if not normalized_record_id:
            return None
        fetched_entries = self.fetch_store_entries_by_ids([normalized_record_id])
        return fetched_entries[0] if fetched_entries else None

    def fetch_project_registry_entry_by_title(self, title: str) -> dict[str, Any] | None:
        self.ensure_sqlite_store_ready()
        normalized_title = self.policy.collapse_text(title)
        if not normalized_title:
            return None
        with closing(self.get_sqlite_connection()) as connection:
            row = connection.execute(
                f"""
            SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json
            FROM {PUBLIC_MEMORY_VIEW}
            WHERE memory_kind = 'project_registry' AND title = ?
            ORDER BY {SQLITE_ENTRY_ORDER_BY}
            LIMIT 1
            """,
                (normalized_title,),
            ).fetchone()
        if row is None:
            return None
        return self.build_entry_from_sqlite_row(row)

    def project_registry_has_project_records(self, project_id: str) -> bool:
        self.ensure_sqlite_store_ready()
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            return False
        with closing(self.get_sqlite_connection()) as connection:
            row = connection.execute(
                """
            SELECT 1 AS exists_value
            FROM project_records
            WHERE project_id = ?
            LIMIT 1
            """,
                (normalized_project_id,),
            ).fetchone()
        return row is not None

    def fetch_entry_family_by_source_ids(self, source_ids: set[str]) -> list[dict[str, Any]]:
        self.ensure_sqlite_store_ready()
        normalized_source_ids = [
            str(source_id or "").strip() for source_id in sorted(source_ids) if str(source_id or "").strip()
        ]
        if not normalized_source_ids:
            return []
        placeholders = ",".join(("?" for _ in normalized_source_ids))
        public_entries = self.fetch_public_memory_entries_by_ids(normalized_source_ids)
        with closing(self.get_sqlite_connection()) as connection:
            field_rows = connection.execute(
                f"""
            SELECT
                id,
                'field_record' AS memory_kind,
                '' AS title,
                '' AS short_summary,
                '' AS overview_summary,
                '' AS detailed_summary,
                '' AS raw_dialogue,
                '' AS problem_background,
                '' AS analysis,
                '' AS action_steps,
                '' AS validation_result,
                '' AS reference_doc_path,
                '' AS project_id,
                created_at,
                updated_at,
                source_memory_id,
                source_field_name,
                '[]' AS tags_json,
                '[]' AS source_paths_json,
                json_array(source_field_name) AS retrieval_fields_json,
                source_field_value_text,
                source_field_value_json
            FROM field_records
            WHERE source_memory_id IN ({placeholders})
            ORDER BY {SQLITE_ENTRY_ORDER_BY}
            """,
                normalized_source_ids,
            ).fetchall()
        family_entries = [*public_entries, *(self.build_entry_from_sqlite_row(row) for row in field_rows)]
        return self.policy.sort_entries(family_entries)

    def fetch_ordered_store_entry_ids(self) -> list[str]:
        ordered_ids = []
        for batch in self.iter_index_entries_in_batches():
            ordered_ids.extend(
                entry_id for entry_id, _, _ in self.policy.build_indexable_entry_payloads(batch)
            )
        return ordered_ids

    def count_public_store_entries(self) -> int:
        self.ensure_sqlite_store_ready()
        with closing(self.get_sqlite_connection()) as connection:
            row = connection.execute("SELECT COUNT(1) AS count_value FROM memory_registry").fetchone()
        return int(row["count_value"] or 0) if row is not None else 0

    def fetch_public_timeline_summary_entries(self, limit: int = 3) -> list[dict[str, Any]]:
        self.ensure_sqlite_store_ready()
        columns = ("id", "memory_kind", "created_at", "updated_at", "title")
        with closing(self.get_sqlite_connection()) as connection:
            rows = connection.execute(
                f"SELECT * FROM ({self._light_public_query(columns)}) ORDER BY {SQLITE_ENTRY_ORDER_BY} LIMIT ?",
                (max(0, int(limit or 0)),),
            ).fetchall()
        return [self.build_entry_from_sqlite_row(row, include_dialogue=False) for row in rows]

    def iter_store_entries_in_batches(self, batch_size: int = SQLITE_REBUILD_BATCH_SIZE) -> Any:
        self.ensure_sqlite_store_ready()
        offset = 0
        resolved_batch_size = max(1, int(batch_size or SQLITE_REBUILD_BATCH_SIZE))
        while True:
            with closing(self.get_sqlite_connection()) as connection:
                rows = connection.execute(
                    f"""
                SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json
                FROM {ALL_MEMORY_VIEW}
                ORDER BY {SQLITE_ENTRY_ORDER_BY}
                LIMIT ? OFFSET ?
                """,
                    (resolved_batch_size, offset),
                ).fetchall()
            if not rows:
                return
            yield [self.build_entry_from_sqlite_row(row) for row in rows]
            offset += resolved_batch_size

    def iter_public_entries_for_timeline(
        self, batch_size: int = SQLITE_REBUILD_BATCH_SIZE
    ) -> Iterator[list[dict[str, Any]]]:
        self.ensure_sqlite_store_ready()
        offset = 0
        size = max(1, int(batch_size or SQLITE_REBUILD_BATCH_SIZE))
        columns = (
            "id",
            "memory_kind",
            "created_at",
            "updated_at",
            "title",
            "short_summary",
            "overview_summary",
            "detailed_summary",
            "validation_result",
            "reference_doc_path",
            "tags_json",
            "source_paths_json",
        )
        while True:
            with closing(self.get_sqlite_connection()) as connection:
                rows = connection.execute(
                    f"SELECT * FROM ({self._light_public_query(columns)}) ORDER BY {SQLITE_ENTRY_ORDER_BY} LIMIT ? OFFSET ?",
                    (size, offset),
                ).fetchall()
            if not rows:
                return
            yield [self.build_entry_from_sqlite_row(row, include_dialogue=False) for row in rows]
            offset += size

    def ensure_sqlite_store_ready(self) -> None:
        with self.init_lock:
            with closing(self.get_sqlite_connection()) as connection:
                self.initialize_sqlite_schema(connection)
                connection.commit()
