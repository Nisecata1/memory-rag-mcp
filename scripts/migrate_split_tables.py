"""把旧单表 memory_entries 拆成轻总表 + 类型分表 + field_records 的一次性迁移脚本。"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from server_utils import (  # noqa: E402
    SQLITE_SCHEMA_VERSION,
    SQLITE_PATH,
    build_entry_from_sqlite_row,
    create_split_sqlite_schema_objects,
    ensure_supported_sqlite_schema_version,
    get_sqlite_connection,
    is_field_record_entry,
    upsert_store_entries,
)


# 校验当前数据库是否仍然保留旧单表 memory_entries；这一步只服务本次一次性拆表迁移。
def ensure_legacy_single_table_exists(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT COUNT(1) AS count_value FROM sqlite_master WHERE type='table' AND name='memory_entries'"
    ).fetchone()
    if int(row["count_value"] or 0) <= 0:
        raise RuntimeError("Legacy table memory_entries was not found. Nothing to migrate.")


# 读取旧单表中的全部记录，并还原成当前统一 entry dict；后面再交给新分表写入函数路由落盘。
def load_legacy_memory_entries(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT
            id,
            memory_kind,
            title,
            short_summary,
            detailed_summary,
            problem_background,
            analysis,
            action_steps,
            validation_result,
            reference_doc_path,
            fingerprint,
            created_at,
            updated_at,
            source_memory_id,
            source_field_name,
            tags_json,
            source_paths_json,
            retrieval_fields_json,
            '' AS source_field_value_text,
            '' AS source_field_value_json
        FROM memory_entries
        ORDER BY updated_at DESC, created_at DESC, id DESC
        """
    ).fetchall()
    return [build_entry_from_sqlite_row(row) for row in rows]


# 删除新分表结构里的业务表旧数据；脚本允许重复执行，但每次都要求从备份库或旧单表重新导入。
def clear_split_tables(connection: sqlite3.Connection) -> None:
    for table_name in ("field_records", "project_records", "chat_events", "facts", "memory_registry"):
        connection.execute(f"DELETE FROM {table_name}")


# 对迁移结果做最小一致性校验，避免脚本执行完后留下缺父记录的 field_record。
def validate_split_result(connection: sqlite3.Connection, legacy_public_count: int, legacy_field_count: int) -> None:
    public_count = int(connection.execute("SELECT COUNT(1) AS count_value FROM memory_registry").fetchone()["count_value"] or 0)
    field_count = int(connection.execute("SELECT COUNT(1) AS count_value FROM field_records").fetchone()["count_value"] or 0)
    if public_count != legacy_public_count:
        raise RuntimeError(f"Public memory count mismatch after migration: expected {legacy_public_count}, got {public_count}")
    if field_count != legacy_field_count:
        raise RuntimeError(f"field_record count mismatch after migration: expected {legacy_field_count}, got {field_count}")
    missing_parent_count = int(
        connection.execute(
            """
            SELECT COUNT(1) AS count_value
            FROM field_records AS field_records
            LEFT JOIN memory_registry AS registry
              ON registry.id = field_records.source_memory_id
            WHERE registry.id IS NULL
            """
        ).fetchone()["count_value"] or 0
    )
    if missing_parent_count:
        raise RuntimeError(f"Detected {missing_parent_count} field_record rows whose source_memory_id has no parent memory.")


# 复制一份当前 memory.db 作为外部回滚备份；脚本不在无备份状态下直接改原库。
def backup_sqlite_file(sqlite_path: Path) -> Path:
    backup_path = sqlite_path.with_name("memory.pre_split_tables.backup.db")
    shutil.copy2(sqlite_path, backup_path)
    return backup_path


# 执行一次性拆表迁移：备份旧库、导入旧单表记录、校验结果、写迁移完成标记、最后删旧表。
def main() -> None:
    if not SQLITE_PATH.exists():
        raise RuntimeError(f"SQLite file does not exist: {SQLITE_PATH}")

    backup_path = backup_sqlite_file(SQLITE_PATH)
    print(f"Backup created: {backup_path}")

    with closing(get_sqlite_connection()) as connection:
        ensure_supported_sqlite_schema_version(
            connection,
            allowed_versions={0, 1, SQLITE_SCHEMA_VERSION},
        )
        ensure_legacy_single_table_exists(connection)
        create_split_sqlite_schema_objects(connection)
        legacy_entries = load_legacy_memory_entries(connection)
        legacy_public_count = sum(1 for entry in legacy_entries if not is_field_record_entry(entry))
        legacy_field_count = sum(1 for entry in legacy_entries if is_field_record_entry(entry))

        connection.execute("BEGIN IMMEDIATE")
        clear_split_tables(connection)
        upsert_store_entries(legacy_entries, connection=connection)
        validate_split_result(connection, legacy_public_count, legacy_field_count)
        connection.execute("DROP TABLE memory_entries")
        connection.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")
        connection.commit()

        remaining_legacy_table = connection.execute(
            "SELECT COUNT(1) AS count_value FROM sqlite_master WHERE type='table' AND name='memory_entries'"
        ).fetchone()
        if int(remaining_legacy_table["count_value"] or 0) != 0:
            raise RuntimeError("Legacy table memory_entries still exists after migration.")

    print("Split-table migration completed.")


if __name__ == "__main__":
    main()
