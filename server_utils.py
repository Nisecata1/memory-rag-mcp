"""memory-rag-mcp 的底层实现模块。

这个模块负责三件事：
1. 保存高信息密度的结构化项目经验、会话事件和事实类记录。
2. 生成按最近更新时间整理的人类可读时间线。
3. 构建并查询基于向量相似度的 RAG 检索索引。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import threading
import uuid
from datetime import datetime
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import faiss
import numpy as np
import yaml
from pydantic import BeforeValidator

# 先拿到当前脚本文件所在目录。
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from embedding_utils import l2_normalize, load_sentence_embedder


CONFIG_DIR = CURRENT_DIR / "config"  # 项目配置目录。
CONFIG_PATH = CONFIG_DIR / "settings.yaml"  # 项目 YAML 配置文件。
SQLITE_SCHEMA_VERSION = 3  # SQLite 主数据层自己的 schema 版本；和 memory.store_version 分开维护。


# 把 YAML 配置里的路径值解析成当前项目可直接使用的绝对路径；相对路径一律相对脚本文件所在目录。
# 上层函数：load_project_config() ,把 settings.yaml 里的各个路径字段统一收口成绝对路径。
def resolve_configured_path(path_value: str) -> Path:
    # 参数 path_value：从 settings.yaml 某个路径字段里读出来的原始字符串，例如 data_dir、store_path、model_path。
    # 返回值：当前代码后续可直接使用的绝对 Path 对象。
    # 先用 str(...) 把配置值统一转成字符串，再用 strip() 去掉首尾空白，避免误判路径。
    normalized_path = str(path_value or "").strip()
    if not normalized_path:  # 配置值为空时直接抛错
        raise RuntimeError(f"Path value in {CONFIG_PATH} must not be empty.")
    # 用 Path(...) 把普通字符串包装成 pathlib 对象，后面才能做路径拼接。
    candidate_path = Path(normalized_path)
    # 只有相对路径才需要补上项目根目录；绝对路径保持调用方原意不变。
    if not candidate_path.is_absolute():
        # 先把相对路径挂到 CURRENT_DIR 下面，再调用 resolve() 归一化成最终绝对路径。
        candidate_path = (CURRENT_DIR / candidate_path).resolve()
    return candidate_path


# 读取并规范 settings.yaml，返回一个 dict 对象。
def load_project_config() -> dict[str, Any]:
    # 第一阶段：读取并解析 settings.yaml，先拿到 YAML 顶层对象。
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Project config file does not exist: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise RuntimeError(f"{CONFIG_PATH.name} must contain a top-level object.")

    # 第二阶段：从顶层配置里取出各个一级分组，后面会分别校验和规范化。
    server_config = raw_config.get("server")  # server_config 对应的是 server 这个配置分组，不是一个单独字符串
    paths_config = raw_config.get("paths")
    embedding_config = raw_config.get("embedding")
    storage_config = raw_config.get("storage") or {}
    memory_config = raw_config.get("memory")
    summary_config = raw_config.get("summary")
    search_config = raw_config.get("search")
    retrieval_config = raw_config.get("retrieval")
    results_config = raw_config.get("results")

    # 第三阶段：统一校验这些必须存在的分组都必须是对象，避免后面按 dict 取字段时报错。
    required_sections = {
        "server": server_config,
        "paths": paths_config,
        "embedding": embedding_config,
        "memory": memory_config,
        "summary": summary_config,
        "search": search_config,
        "retrieval": retrieval_config,
        "results": results_config,
    }
    for section_name, section_value in required_sections.items():
        if not isinstance(section_value, dict):
            raise RuntimeError(f"Section '{section_name}' in {CONFIG_PATH.name} must be an object.")

    # 第四阶段：解析服务名、路径和 embedding 这类基础运行配置，把它们转成后续可直接使用的值。
    # 取出一个str
    server_name = str(server_config.get("name") or "").strip()
    if not server_name:
        raise RuntimeError(f"server.name in {CONFIG_PATH.name} must not be empty.")

    ai_memory_dir = resolve_configured_path(str(paths_config.get("ai_memory_dir") or ""))
    data_dir = resolve_configured_path(str(paths_config.get("data_dir") or ""))
    timeline_path = resolve_configured_path(str(paths_config.get("timeline_path") or ""))
    embeddings_path = resolve_configured_path(str(paths_config.get("embeddings_path") or ""))
    index_path = resolve_configured_path(str(paths_config.get("index_path") or ""))
    meta_path = resolve_configured_path(str(paths_config.get("meta_path") or ""))

    embed_model_path = resolve_configured_path(str(embedding_config.get("model_path") or ""))
    embed_device = str(embedding_config.get("device") or "").strip()
    if not embed_device:
        raise RuntimeError(f"embedding.device in {CONFIG_PATH.name} must not be empty.")

    if storage_config and not isinstance(storage_config, dict):
        raise RuntimeError(f"Section 'storage' in {CONFIG_PATH.name} must be an object when present.")
    storage_backend = str(storage_config.get("backend") or "sqlite").strip().lower()
    if storage_backend != "sqlite":
        raise RuntimeError(f"storage.backend in {CONFIG_PATH.name} must currently be 'sqlite'.")
    sqlite_path = resolve_configured_path(str(storage_config.get("sqlite_path") or "../memory-rag-mcp-data/memory.db"))

    # 第五阶段：解析主数据结构、摘要策略和搜索限制这类业务规则配置。
    kind_values = memory_config.get("kind_values")
    if not isinstance(kind_values, list) or not kind_values:
        raise RuntimeError(f"memory.kind_values in {CONFIG_PATH.name} must be a non-empty list.")
    normalized_kind_values = tuple(str(item or "").strip() for item in kind_values if str(item or "").strip())
    if not normalized_kind_values:
        raise RuntimeError(f"memory.kind_values in {CONFIG_PATH.name} must contain valid values.")

    tag_fallback_limit = int(summary_config.get("tag_fallback_limit") or 0)
    short_summary_max_length = int(summary_config.get("short_summary_max_length") or 0)
    if tag_fallback_limit <= 0 or short_summary_max_length <= 0:
        raise RuntimeError(f"summary section in {CONFIG_PATH.name} must contain positive integers.")

    max_top_k = int(search_config.get("max_top_k") or 0)
    if max_top_k <= 0:
        raise RuntimeError(f"search.max_top_k in {CONFIG_PATH.name} must be a positive integer.")

    # 第六阶段：解析检索文本拼接规则和结果裁剪规则，保证字段列表和标签映射彼此一致。
    field_candidates = retrieval_config.get("field_candidates")
    field_labels = retrieval_config.get("field_labels")
    if not isinstance(field_candidates, list) or not field_candidates:
        raise RuntimeError(f"retrieval.field_candidates in {CONFIG_PATH.name} must be a non-empty list.")
    if not isinstance(field_labels, dict) or not field_labels:
        raise RuntimeError(f"retrieval.field_labels in {CONFIG_PATH.name} must be an object.")
    normalized_field_candidates = [str(item or "").strip() for item in field_candidates if str(item or "").strip()]
    if not normalized_field_candidates:
        raise RuntimeError(f"retrieval.field_candidates in {CONFIG_PATH.name} must contain valid values.")
    normalized_field_labels = {
        str(key or "").strip(): str(value or "").strip()
        for key, value in field_labels.items()
        if str(key or "").strip() and str(value or "").strip()
    }
    for field_name in normalized_field_candidates:
        if field_name not in normalized_field_labels:
            raise RuntimeError(
                f"retrieval.field_labels in {CONFIG_PATH.name} must contain a label for '{field_name}'."
            )

    search_fields = results_config.get("search_fields")
    detail_fields = results_config.get("detail_fields")
    if not isinstance(search_fields, list) or not search_fields:
        raise RuntimeError(f"results.search_fields in {CONFIG_PATH.name} must be a non-empty list.")
    if not isinstance(detail_fields, list) or not detail_fields:
        raise RuntimeError(f"results.detail_fields in {CONFIG_PATH.name} must be a non-empty list.")

    normalized_search_fields = tuple(str(item or "").strip() for item in search_fields if str(item or "").strip())
    normalized_detail_fields = tuple(str(item or "").strip() for item in detail_fields if str(item or "").strip())
    if not normalized_search_fields or not normalized_detail_fields:
        raise RuntimeError(f"results section in {CONFIG_PATH.name} must contain valid field names.")

    return {
        "server_name": server_name,
        "ai_memory_dir": ai_memory_dir,
        "data_dir": data_dir,
        "timeline_path": timeline_path,
        "embeddings_path": embeddings_path,
        "index_path": index_path,
        "meta_path": meta_path,
        "embed_model_path": embed_model_path,
        "embed_device": embed_device,
        "storage_backend": storage_backend,
        "sqlite_path": sqlite_path,
        "memory_kind_values": set(normalized_kind_values),
        "tag_fallback_limit": tag_fallback_limit,
        "short_summary_max_length": short_summary_max_length,
        "max_top_k": max_top_k,
        "retrieval_field_candidates": normalized_field_candidates,
        "field_labels": normalized_field_labels,
        "search_result_fields": normalized_search_fields,
        "detail_result_fields": normalized_detail_fields,
    }


# 模块导入时先生成一次项目级配置快照。
# 后面这整段模块级常量，都是从这份快照里展开出来的，以供后续业务代码使用。
PROJECT_CONFIG = load_project_config()  # 项目级配置快照。

# 这一组是路径与运行时基础常量，后续主数据、时间线、模型和索引链路都会直接依赖它们。
SERVER_NAME = PROJECT_CONFIG["server_name"]  # MCP 服务名称。
AI_MEMORY_DIR = PROJECT_CONFIG["ai_memory_dir"]  # 共享的 AI-memory 根目录。
DATA_DIR = PROJECT_CONFIG["data_dir"]  # 项目记忆数据目录。
TIMELINE_PATH = PROJECT_CONFIG["timeline_path"]  # 人类可读的时间线文件。
EMBEDDINGS_PATH = PROJECT_CONFIG["embeddings_path"]  # 已计算向量的本地缓存矩阵文件。
INDEX_PATH = PROJECT_CONFIG["index_path"]  # FAISS 向量索引文件。
META_PATH = PROJECT_CONFIG["meta_path"]  # 向量行号到记录 id 的映射文件。
REBUILD_STATE_PATH = DATA_DIR / "rebuild_state.json"  # 后台全量重建状态文件。
EMBED_MODEL_PATH = PROJECT_CONFIG["embed_model_path"]  # YAML 配置中的本地嵌入模型目录。
EMBED_DEVICE = PROJECT_CONFIG["embed_device"]  # YAML 配置中的嵌入设备。
STORAGE_BACKEND = PROJECT_CONFIG["storage_backend"]  # 当前主数据存储后端；本轮固定为 sqlite。
SQLITE_PATH = PROJECT_CONFIG["sqlite_path"]  # SQLite 主数据文件。

# 这一组是业务规则与返回裁剪常量，负责约束搜索上限、检索字段以及接口默认返回字段。
MAX_TOP_K = PROJECT_CONFIG["max_top_k"]  # 搜索结果最大返回条数。
TAG_FALLBACK_LIMIT = PROJECT_CONFIG["tag_fallback_limit"]  # 服务端兜底标签的最大数量。
SHORT_SUMMARY_MAX_LENGTH = PROJECT_CONFIG["short_summary_max_length"]  # 自动生成简短摘要时的最大字符数。
RETRIEVAL_FIELD_CANDIDATES = PROJECT_CONFIG["retrieval_field_candidates"]  # 统一的检索文本候选字段顺序。
RETRIEVAL_FIELD_SIGNATURE = "|".join(RETRIEVAL_FIELD_CANDIDATES)  # 当前检索字段规则签名，用来判断旧 embedding 是否失效。
FIELD_LABELS = PROJECT_CONFIG["field_labels"]  # 检索文本拼接时使用的人类可读字段标签。
MEMORY_KIND_VALUES = PROJECT_CONFIG["memory_kind_values"]  # 允许的记忆类型枚举值。
SEARCH_RESULT_FIELDS = PROJECT_CONFIG["search_result_fields"]  # search 默认返回的轻量字段。
DETAIL_RESULT_FIELDS = PROJECT_CONFIG["detail_result_fields"]  # 详情接口默认返回的完整业务字段。

UPDATE_ALLOWED_FIELDS = (
    "title",
    "short_summary",
    "overview_summary",
    "detailed_summary",
    "problem_background",
    "analysis",
    "action_steps",
    "validation_result",
    "project_id",
    "tags",
    "source_paths",
    "reference_doc_path",
)  # update 补丁允许修改的业务字段。
UPDATE_SYSTEM_FIELDS = (
    "id",
    "created_at",
    "updated_at",
    "retrieval_fields",
)  # update 禁止直接修改的系统字段。
UPDATE_LIST_FIELDS = {"tags", "source_paths"}  # update 中只能用 [] 清空的列表字段。
UPDATE_NULL_CLEARABLE_FIELDS = {
    "problem_background",
    "analysis",
    "action_steps",
    "validation_result",
    "reference_doc_path",
}  # update 中允许用 null 清空的可空文本字段。
PUBLIC_MEMORY_KIND_VALUES = (
    "project_record",
    "chat_event",
    "fact",
    "project_registry",
)  # 对外工具允许创建或更新到的公共记忆类型。
INTERNAL_MEMORY_KIND_VALUES = ("field_record",)  # 只允许服务端内部派生的字段级记忆类型。
FIELD_RECORD_CANDIDATE_FIELDS = (
    "title",
    "short_summary",
    "overview_summary",
    "problem_background",
    "analysis",
    "action_steps",
    "validation_result",
    "detailed_summary",
    "tags",
    "reference_doc_path",
)  # 自动拆成 field_record 的字段范围；不含 source_paths，避免把来源证明字段变成主要语义噪音。
FACT_RETRIEVAL_FIELD_CANDIDATES = (
    "title",
    "short_summary",
    "detailed_summary",
    "tags",
    "reference_doc_path",
)  # fact 的检索文本优先保留稳定事实正文，不沿用项目经验的字段拼接偏好。
PROJECT_REGISTRY_RETRIEVAL_FIELD_CANDIDATES = (
    "title",
    "overview_summary",
    "tags",
)  # project_registry 只围绕项目名、项目整体概述和标签做召回，避免把时间和路径字段变成噪音。
PROJECT_REGISTRY_FIELD_RECORD_CANDIDATE_FIELDS = (
    "title",
    "overview_summary",
    "tags",
)  # project_registry 只把这 3 个字段拆成 field_record，避免生成无意义的 short_summary 或路径子记录。
MATCHED_FIELDS_LIMIT = 3  # search 返回的 matched_fields 最多保留 3 个，避免结果对象膨胀。
BACKGROUND_REBUILD_MODE = "background_full_rebuild"  # 后台全量重建的固定模式名。
_BACKGROUND_REBUILD_LOCK = threading.Lock()  # 同一进程内的后台重建串行锁。
_BACKGROUND_REBUILD_THREAD: threading.Thread | None = None  # 当前进程里正在运行的后台重建线程引用。
_SQLITE_INIT_LOCK = threading.Lock()  # 同一进程内串行初始化 SQLite，避免首次启动时重复建表或重复迁移。
SQLITE_REBUILD_BATCH_SIZE = 200  # SQLite 全量重建时每批读取的记录数，避免一次性堆太多检索文本。
SQLITE_ENTRY_ORDER_BY = "updated_at DESC, created_at DESC, id DESC"  # SQLite 取当前稳定顺序时沿用现有 Python 排序口径。
SQLITE_ENTRY_COLUMNS = (
    "id",
    "memory_kind",
    "title",
    "short_summary",
    "overview_summary",
    "detailed_summary",
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

# 类型声明上这是 Literal[...]，也就是“只允许固定几个字符串字面量”的类型；运行时拿到的数据类型仍然是 str，例如 "fact"。
# 对外 save / update 这类工具里的 memory_kind 只能传 "project_record"、"chat_event" 或 "fact"；最终进代码时就是这三个字符串之一，不是别的对象类型。
MemoryKind = Literal["project_record", "chat_event", "fact"]

# 类型声明上这是 Literal[...]，表示主数据实际允许保存的记忆类型全集；运行时仍然是 str，例如 "field_record"。
# 它和上面的 MemoryKind 不同：MemoryKind 面向工具层输入，StoredMemoryKind 面向主数据与内部派生记录，所以额外包含内部类型 "field_record"。
StoredMemoryKind = Literal["project_record", "chat_event", "fact", "project_registry", "field_record"]

# TagListInput 是 save.tags 的参数类型别名: 声明层写成 Annotated[list[str] | None, BeforeValidator(...)]，运行时最终拿到的还是 list[str] 或 None，
# BeforeValidator 会先把 "rag, mcp"、'["rag", "mcp"]' 这类字符串输入整理成 list[str]。
# 简单来说就是入口层帮你兜一下错误输入
TagListInput = Annotated[
    list[str] | None, 
    BeforeValidator(lambda value: coerce_tag_list_input(value))
]

# 类型声明上这是 Annotated[list[str] | None, ...]；运行时进入 save 前会被整理成 list[str] 或 None，例如 "C:/a.md" 会先转成 ["C:/a.md"]。
# save 工具里的 source_paths 最终类型是 list[str] | None；例如传 "C:/a.md" 会先变成 ["C:/a.md"]，传多个路径列表时最终仍然是 list[str]。
SourcePathListInput = Annotated[
    list[str] | None,
    BeforeValidator(lambda value: coerce_source_path_list_input(value)),
]


# 返回写入记录时使用的本地时间字符串。
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# 确保 AI-memory 目录和项目数据目录在读写前已经存在。
def ensure_memory_dir() -> None:
    AI_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)


# 打开一条 SQLite 连接，并统一设置本项目需要的基础 PRAGMA。
def get_sqlite_connection() -> sqlite3.Connection:
    ensure_memory_dir()
    connection = sqlite3.connect(str(SQLITE_PATH))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


# 把字符串列表编码成 SQLite 里存储用的 JSON 文本。
def dump_json_string_list(values: list[str] | None) -> str:
    return json.dumps(normalize_list(values), ensure_ascii=False)


# 把 SQLite 里的 JSON 文本还原成字符串列表，并在异常时安全退回空列表。
def load_json_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return normalize_list([str(item or "").strip() for item in value])

    raw_text = str(value or "").strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return normalize_list([str(item or "").strip() for item in parsed])


# 把 field_record 的单字段值编码成 SQLite 可直接保存的 JSON 文本。
def dump_field_record_value_json(field_value: str | list[str]) -> str:
    if isinstance(field_value, list):
        return json.dumps(normalize_list(field_value), ensure_ascii=False)
    return json.dumps(str(field_value or ""), ensure_ascii=False)


# 把 field_record 的单字段值渲染成检索文本，供 source_field_value_text 落盘。
def render_field_record_value_text(field_value: str | list[str]) -> str:
    if isinstance(field_value, list):
        return "，".join(normalize_list(field_value))
    return str(field_value or "")


# 从 SQLite 里的 JSON 文本恢复 field_record 的真实字段值。
def load_field_record_value_json(source_field_name: str, json_text: Any, fallback_text: Any) -> str | list[str]:
    raw_json_text = str(json_text or "").strip()
    if raw_json_text:
        try:
            parsed = json.loads(raw_json_text)
        except json.JSONDecodeError:
            parsed = None
        if source_field_name == "tags":
            if isinstance(parsed, list):
                return normalize_list([str(item or "").strip() for item in parsed])
        elif isinstance(parsed, str):
            return str(parsed or "")

    if source_field_name == "tags":
        return normalize_list(str(fallback_text or "").split("，"))
    return str(fallback_text or "")


# 把一条规范化后的主数据记录转换成 SQLite 参数字典。
def build_sqlite_entry_row(entry: dict[str, Any]) -> dict[str, Any]:
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
            "tags_json": dump_json_string_list(entry.get("tags")),
            "source_paths_json": dump_json_string_list(entry.get("source_paths")),
            "retrieval_fields_json": dump_json_string_list(entry.get("retrieval_fields")),
        }
    if normalized_memory_kind == "project_registry":
        return {
            **base_row,
            "title": str(entry.get("title") or ""),
            "overview_summary": str(entry.get("overview_summary") or ""),
            "reference_doc_path": str(entry.get("reference_doc_path") or ""),
            "tags_json": dump_json_string_list(entry.get("tags")),
            "source_paths_json": dump_json_string_list(entry.get("source_paths")),
            "retrieval_fields_json": dump_json_string_list(entry.get("retrieval_fields")),
        }
    if normalized_memory_kind in {"chat_event", "fact"}:
        return {
            **base_row,
            "title": str(entry.get("title") or ""),
            "short_summary": str(entry.get("short_summary") or ""),
            "detailed_summary": str(entry.get("detailed_summary") or ""),
            "reference_doc_path": str(entry.get("reference_doc_path") or ""),
            "tags_json": dump_json_string_list(entry.get("tags")),
            "source_paths_json": dump_json_string_list(entry.get("source_paths")),
            "retrieval_fields_json": dump_json_string_list(entry.get("retrieval_fields")),
        }
    if normalized_memory_kind == "field_record":
        normalized_source_field_name = str(entry.get("source_field_name") or "").strip()
        field_value = resolve_field_record_value(entry, normalized_source_field_name)
        return {
            **base_row,
            "source_memory_id": str(entry.get("source_memory_id") or "").strip(),
            "source_field_name": normalized_source_field_name,
            "source_field_value_text": render_field_record_value_text(field_value),
            "source_field_value_json": dump_field_record_value_json(field_value),
        }
    raise RuntimeError(f"Unsupported memory_kind for SQLite row conversion: {normalized_memory_kind}")


# 把 SQLite 读出来的一行转换回当前主数据记录 dict。
def build_entry_from_sqlite_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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
        field_value = resolve_field_record_value(row_data, source_field_name)
        return {
            **common_entry,
            "title": "",
            "short_summary": "",
            "overview_summary": "",
            "detailed_summary": "",
            "problem_background": "",
            "analysis": "",
            "action_steps": "",
            "validation_result": "",
            "reference_doc_path": field_value if source_field_name == "reference_doc_path" else "",
            "project_id": "",
            "source_memory_id": str(row_data.get("source_memory_id") or "").strip(),
            "source_field_name": source_field_name,
            "tags": list(field_value) if source_field_name == "tags" and isinstance(field_value, list) else [],
            "source_paths": [],
            "retrieval_fields": [source_field_name] if source_field_name else [],
            "source_field_value_text": str(row_data.get("source_field_value_text") or ""),
            "source_field_value_json": str(row_data.get("source_field_value_json") or ""),
            source_field_name: field_value,
        }
    if normalized_memory_kind == "project_registry" and not normalized_short_summary:
        normalized_short_summary = build_short_summary(
            title=normalized_title,
            detailed_summary=normalized_overview_summary,
        )
    normalized_detailed_summary = str(row_data.get("detailed_summary") or "")
    if normalized_memory_kind == "project_registry" and not normalized_detailed_summary:
        normalized_detailed_summary = normalized_overview_summary
    return {
        **common_entry,
        "title": normalized_title,
        "short_summary": normalized_short_summary,
        "overview_summary": normalized_overview_summary,
        "detailed_summary": normalized_detailed_summary,
        "problem_background": str(row_data.get("problem_background") or ""),
        "analysis": str(row_data.get("analysis") or ""),
        "action_steps": str(row_data.get("action_steps") or ""),
        "validation_result": str(row_data.get("validation_result") or ""),
        "reference_doc_path": str(row_data.get("reference_doc_path") or ""),
        "project_id": normalized_project_id or (common_entry["id"] if normalized_memory_kind == "project_registry" else ""),
        "source_memory_id": "",
        "source_field_name": "",
        "tags": load_json_string_list(row_data.get("tags_json")),
        "source_paths": load_json_string_list(row_data.get("source_paths_json")),
        "retrieval_fields": load_json_string_list(row_data.get("retrieval_fields_json")),
    }


# 读取并校验当前数据库的 user_version；运行时默认只接管空库或当前分表版本，迁移脚本可额外放行旧版本。
def ensure_supported_sqlite_schema_version(
    connection: sqlite3.Connection,
    allowed_versions: set[int] | list[int] | tuple[int, ...] | None = None,
) -> int:
    resolved_allowed_versions = (
        {0, SQLITE_SCHEMA_VERSION}
        if allowed_versions is None
        else {int(version) for version in allowed_versions}
    )
    current_schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    if current_schema_version not in resolved_allowed_versions:
        expected_versions = ", ".join(str(version) for version in sorted(resolved_allowed_versions))
        raise RuntimeError(
            f"Unsupported SQLite schema version {current_schema_version}; expected one of: {expected_versions}."
        )
    return current_schema_version


# 在当前 SQLite 连接里创建分表结构、只读视图和索引；运行时初始化和一次性迁移脚本都会复用这一步。
def create_split_sqlite_schema_objects(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_registry (
            id TEXT PRIMARY KEY,
            memory_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
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
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS project_registry (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            overview_summary TEXT NOT NULL,
            reference_doc_path TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            retrieval_fields_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_events (
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
        """
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
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS field_records (
            id TEXT PRIMARY KEY,
            source_memory_id TEXT NOT NULL,
            source_field_name TEXT NOT NULL,
            source_field_value_text TEXT NOT NULL,
            source_field_value_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_registry_memory_kind ON memory_registry(memory_kind)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_registry_updated_at ON memory_registry(updated_at)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_project_records_project_id ON project_records(project_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_project_registry_title ON project_registry(title)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_field_records_source_memory_id ON field_records(source_memory_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_field_records_source_field_name ON field_records(source_field_name)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_field_records_memory_and_field ON field_records(source_memory_id, source_field_name)"
    )
    connection.execute(f"DROP VIEW IF EXISTS {PUBLIC_MEMORY_VIEW}")
    connection.execute(
        f"""
        CREATE VIEW {PUBLIC_MEMORY_VIEW} AS
        SELECT
            registry.id AS id,
            registry.memory_kind AS memory_kind,
            detail.title AS title,
            detail.short_summary AS short_summary,
            detail.overview_summary AS overview_summary,
            detail.detailed_summary AS detailed_summary,
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
                detailed_summary,
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
        """
    )
    connection.execute(f"DROP VIEW IF EXISTS {ALL_MEMORY_VIEW}")
    connection.execute(
        f"""
        CREATE VIEW {ALL_MEMORY_VIEW} AS
        SELECT
            id,
            memory_kind,
            title,
            short_summary,
            overview_summary,
            detailed_summary,
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
        """
    )


# 把现有 schema v2 就地升级到 v3；这里只补 project_registry 物理表和 project_records.project_id，不动旧 project_record 内容。
def migrate_sqlite_schema_v2_to_v3(connection: sqlite3.Connection) -> None:
    existing_project_record_columns = {
        str(row["name"] or "").strip()
        for row in connection.execute("PRAGMA table_info(project_records)").fetchall()
    }
    if "project_id" not in existing_project_record_columns:
        connection.execute("ALTER TABLE project_records ADD COLUMN project_id TEXT")


# 在 SQLite 中初始化当前分表 schema；这里只接管空库或已经切到分表版本的数据库文件。
def initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    current_schema_version = ensure_supported_sqlite_schema_version(
        connection,
        allowed_versions={0, 2, SQLITE_SCHEMA_VERSION},
    )
    if current_schema_version == 2:
        migrate_sqlite_schema_v2_to_v3(connection)
    create_split_sqlite_schema_objects(connection)
    if current_schema_version != SQLITE_SCHEMA_VERSION:
        connection.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")


# 把主记忆写入轻总表；save/update 的公共父记录索引字段统一通过这里落盘。
def upsert_memory_registry(entries: list[dict[str, Any]], connection: sqlite3.Connection) -> None:
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


# 按类型把公共主记忆写到各自详情表；这层只负责物理落盘，不负责类型判断之外的业务规则。
def upsert_public_memory_details(entries: list[dict[str, Any]], connection: sqlite3.Connection) -> None:
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
        grouped_entries[memory_kind].append(build_sqlite_entry_row(entry))

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
                id, title, short_summary, detailed_summary, reference_doc_path,
                tags_json, source_paths_json, retrieval_fields_json
            ) VALUES (
                :id, :title, :short_summary, :detailed_summary, :reference_doc_path,
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


# 把 field_record 写进最小必要字段表；这里不再把一堆公共业务空列物理落盘。
def upsert_field_record_rows(entries: list[dict[str, Any]], connection: sqlite3.Connection) -> None:
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
        [build_sqlite_entry_row(entry) for entry in entries],
    )


# 从公共主记忆只读视图按 id 取一批记录；返回形状继续对齐旧的统一 entry dict。
def fetch_public_memory_entries_by_ids(record_ids: list[str], connection: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    normalized_ids = normalize_record_ids(record_ids)
    if not normalized_ids:
        return []
    placeholders = ",".join("?" for _ in normalized_ids)
    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
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
        for entry in (build_entry_from_sqlite_row(row) for row in rows)
    }
    return [entry_map[record_id] for record_id in normalized_ids if record_id in entry_map]


# 从 field_record 物理表按 id 取一批内部子记录；search 的子记录命中回源会用到这里。
def fetch_field_record_entries_by_ids(record_ids: list[str], connection: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    normalized_ids = normalize_record_ids(record_ids)
    if not normalized_ids:
        return []
    placeholders = ",".join("?" for _ in normalized_ids)
    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
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
        for entry in (build_entry_from_sqlite_row(row) for row in rows)
    }
    return [entry_map[record_id] for record_id in normalized_ids if record_id in entry_map]


# 把一批记录写进 SQLite；save/update 的主记忆和 field_record 家族都通过这里落盘。
def upsert_store_entries(entries: list[dict[str, Any]], connection: sqlite3.Connection | None = None) -> None:
    if not entries:
        return

    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
    try:
        public_entries = [entry for entry in entries if not is_field_record_entry(entry)]
        field_entries = [entry for entry in entries if is_field_record_entry(entry)]
        if public_entries:
            upsert_memory_registry(public_entries, connection=db)
            upsert_public_memory_details(public_entries, connection=db)
        if field_entries:
            upsert_field_record_rows(field_entries, connection=db)
        if owned_connection:
            db.commit()
    finally:
        if owned_connection:
            db.close()


# 按主记忆 id 删除记录；delete_by_ids 的主记忆删除和测试清理都通过这里执行。
def delete_store_entries_by_ids(record_ids: list[str], connection: sqlite3.Connection | None = None) -> None:
    normalized_ids = normalize_record_ids(record_ids)
    if not normalized_ids:
        return
    placeholders = ",".join("?" for _ in normalized_ids)
    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
    try:
        registry_rows = db.execute(
            f"SELECT id, memory_kind FROM memory_registry WHERE id IN ({placeholders})",
            normalized_ids,
        ).fetchall()
        ids_by_kind: dict[str, list[str]] = {"project_record": [], "chat_event": [], "fact": [], "project_registry": []}
        for row in registry_rows:
            memory_kind = str(row["memory_kind"] or "").strip()
            if memory_kind in ids_by_kind:
                ids_by_kind[memory_kind].append(str(row["id"] or "").strip())
        for memory_kind, ids_for_kind in ids_by_kind.items():
            if not ids_for_kind:
                continue
            detail_placeholders = ",".join("?" for _ in ids_for_kind)
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


# 按 source_memory_id 删除一批 field_record；update/delete 维护单个主记忆家族时通过这里清理旧子记录。
def delete_field_records_by_source_ids(source_ids: list[str], connection: sqlite3.Connection | None = None) -> None:
    normalized_source_ids = normalize_record_ids(source_ids)
    placeholders = ",".join("?" for _ in normalized_source_ids)
    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
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


# 按 source_memory_id + source_field_name 精确删除一批 field_record；update 的字段级同步只清理这次受影响的旧子记录。
def delete_field_records_by_source_and_fields(
    source_memory_id: str,
    field_names: list[str] | tuple[str, ...] | set[str],
    connection: sqlite3.Connection | None = None,
) -> None:
    normalized_source_memory_id = str(source_memory_id or "").strip()
    normalized_field_name_set = {
        str(field_name or "").strip()
        for field_name in field_names
        if str(field_name or "").strip() in FIELD_RECORD_CANDIDATE_FIELDS
    }
    normalized_field_names = [
        field_name
        for field_name in FIELD_RECORD_CANDIDATE_FIELDS
        if field_name in normalized_field_name_set
    ]
    if not normalized_source_memory_id or not normalized_field_names:
        return

    placeholders = ",".join("?" for _ in normalized_field_names)
    owned_connection = connection is None
    if owned_connection:
        ensure_sqlite_store_ready()
    db = connection or get_sqlite_connection()
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


# 从 SQLite 中读取一批指定 id 的记录，并保持调用方传入的 id 顺序。
def fetch_store_entries_by_ids(record_ids: list[str]) -> list[dict[str, Any]]:
    ensure_sqlite_store_ready()
    normalized_ids = normalize_record_ids(record_ids)
    if not normalized_ids:
        return []
    public_entries = fetch_public_memory_entries_by_ids(normalized_ids)
    field_entries = fetch_field_record_entries_by_ids(normalized_ids)
    entry_map = {str(entry.get("id") or "").strip(): entry for entry in [*public_entries, *field_entries]}
    return [
        entry_map[record_id]
        for record_id in normalized_ids
        if record_id in entry_map
    ]


# 从 db 里按 id 读取单条主数据记录；save/update 的精确回源和冲突检查都通过这里取旧记录。
def fetch_store_entry_by_id(record_id: str) -> dict[str, Any] | None:
    ensure_sqlite_store_ready()
    normalized_record_id = str(record_id or "").strip()
    if not normalized_record_id:
        return None
    fetched_entries = fetch_store_entries_by_ids([normalized_record_id])
    return fetched_entries[0] if fetched_entries else None


# 按项目标题读取 project_registry；create_project 用它挡住重复项目名。
def fetch_project_registry_entry_by_title(title: str) -> dict[str, Any] | None:
    ensure_sqlite_store_ready()
    normalized_title = collapse_text(title)
    if not normalized_title:
        return None
    with closing(get_sqlite_connection()) as connection:
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
    return build_entry_from_sqlite_row(row)


# 判断某个 project_registry 下面是否仍挂着项目问题；delete_by_ids 会先用它挡住误删。
def project_registry_has_project_records(project_id: str) -> bool:
    ensure_sqlite_store_ready()
    normalized_project_id = str(project_id or "").strip()
    if not normalized_project_id:
        return False
    with closing(get_sqlite_connection()) as connection:
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





# 读取一批 source_memory_id 相关的当前家族记录，包含主记忆本体和它当前仍存在的 field_record 子记录。
def fetch_entry_family_by_source_ids(source_ids: set[str]) -> list[dict[str, Any]]:
    ensure_sqlite_store_ready()
    normalized_source_ids = [
        str(source_id or "").strip()
        for source_id in sorted(source_ids)
        if str(source_id or "").strip()
    ]
    if not normalized_source_ids:
        return []
    placeholders = ",".join("?" for _ in normalized_source_ids)
    public_entries = fetch_public_memory_entries_by_ids(normalized_source_ids)
    with closing(get_sqlite_connection()) as connection:
        field_rows = connection.execute(
            f"""
            SELECT
                id,
                'field_record' AS memory_kind,
                '' AS title,
                '' AS short_summary,
                '' AS overview_summary,
                '' AS detailed_summary,
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
    family_entries = [*public_entries, *(build_entry_from_sqlite_row(row) for row in field_rows)]
    return sort_entries(family_entries)


# 读取当前整库里真正进入向量索引的稳定顺序 id 列表；空检索文本记录会在这里被排除。
def fetch_ordered_store_entry_ids() -> list[str]:
    ensure_sqlite_store_ready()
    with closing(get_sqlite_connection()) as connection:
        rows = connection.execute(
            f"""
            SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json
            FROM {ALL_MEMORY_VIEW}
            ORDER BY {SQLITE_ENTRY_ORDER_BY}
            """
        ).fetchall()
    ordered_entries = [build_entry_from_sqlite_row(row) for row in rows]
    return [entry_id for entry_id, _, _ in build_indexable_entry_payloads(ordered_entries)]


# 统计当前公开主记忆条数；对外 total_entries 不把 field_record 算进去。
def count_public_store_entries() -> int:
    ensure_sqlite_store_ready()
    with closing(get_sqlite_connection()) as connection:
        row = connection.execute(
            "SELECT COUNT(1) AS count_value FROM memory_registry"
        ).fetchone()
    return int(row["count_value"] or 0) if row is not None else 0


# 读取按当前稳定顺序排列的前若干条公开主记忆；save/update 的 timeline_summary 直接用这里，不再整库裁切。
def fetch_public_timeline_summary_entries(limit: int = 3) -> list[dict[str, Any]]:
    ensure_sqlite_store_ready()
    with closing(get_sqlite_connection()) as connection:
        rows = connection.execute(
            f"""
            SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json
            FROM {PUBLIC_MEMORY_VIEW}
            ORDER BY {SQLITE_ENTRY_ORDER_BY}
            LIMIT ?
            """,
            (max(0, int(limit or 0)),),
        ).fetchall()
    return [build_entry_from_sqlite_row(row) for row in rows]


# 逐批读取当前整库记录；后台全量 embedding 用这里分批拿 SQLite 数据，避免一次性堆完整检索文本。
def iter_store_entries_in_batches(batch_size: int = SQLITE_REBUILD_BATCH_SIZE) -> Any:
    ensure_sqlite_store_ready()
    offset = 0
    resolved_batch_size = max(1, int(batch_size or SQLITE_REBUILD_BATCH_SIZE))
    while True:
        with closing(get_sqlite_connection()) as connection:
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
        yield [build_entry_from_sqlite_row(row) for row in rows]
        offset += resolved_batch_size


# 逐批读取当前公开主记忆，用来生成时间线 Markdown，避免每次写入都先整库 load_store() 再裁切。
def iter_public_entries_for_timeline(batch_size: int = SQLITE_REBUILD_BATCH_SIZE) -> Any:
    ensure_sqlite_store_ready()
    offset = 0
    resolved_batch_size = max(1, int(batch_size or SQLITE_REBUILD_BATCH_SIZE))
    while True:
        with closing(get_sqlite_connection()) as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(SQLITE_ENTRY_COLUMNS)}, source_field_value_text, source_field_value_json
                FROM {PUBLIC_MEMORY_VIEW}
                ORDER BY {SQLITE_ENTRY_ORDER_BY}
                LIMIT ? OFFSET ?
                """,
                (resolved_batch_size, offset),
            ).fetchall()
        if not rows:
            return
        yield [build_entry_from_sqlite_row(row) for row in rows]
        offset += resolved_batch_size


# 确保 SQLite 主数据层已准备好；这里只接管已经切到分表结构的库或全新空库。
def ensure_sqlite_store_ready() -> None:
    with _SQLITE_INIT_LOCK:
        with get_sqlite_connection() as connection:
            initialize_sqlite_schema(connection)
            connection.commit()


# 把自由文本压缩成单行紧凑字符串，避免空白符干扰存储和字段比较。
# 比如 "A\nB" -> "A B"
def collapse_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


# 规范化长文本块，保留段落换行，但去掉多余空白和首尾空行。
def normalize_text_block(value: str | None) -> str:
    raw_text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw_text.split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


# 尝试把字符串形式的 JSON 数组解析成字符串列表。
def try_parse_json_string_list(value: str) -> list[str] | None:
    stripped = str(value or "").strip()
    if not stripped or not stripped.startswith("[") or not stripped.endswith("]"):
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    return [str(item).strip() for item in parsed if str(item).strip()]


# 把标签输入宽松整理成字符串列表，兼容数组、JSON 字符串和逗号分隔文本。
def coerce_tag_list_input(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        parsed = try_parse_json_string_list(value)
        if parsed is not None:
            return parsed

        stripped = value.strip()
        if not stripped:
            return []

        return [item.strip() for item in re.split(r"[\n,，；;]+", stripped) if item.strip()]

    return value


# 把来源路径输入宽松整理成字符串列表，兼容单个路径字符串和多路径文本。
def coerce_source_path_list_input(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        parsed = try_parse_json_string_list(value)
        if parsed is not None:
            return parsed

        stripped = value.strip()
        if not stripped:
            return []

        if re.search(r"[\n；;]+", stripped):
            return [item.strip() for item in re.split(r"[\n；;]+", stripped) if item.strip()]

        return [stripped]

    return value


# 规范化列表输入，去掉空值、重复值，并保持稳定顺序。
def normalize_list(values: list[str] | None) -> list[str]:
    if not values:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    return cleaned


# 解析记录时间戳，兼容旧的 YYYY-MM-DD 日期格式，并统一转成到秒的本地 ISO 字符串。
def resolve_record_timestamp(value: str | None, fallback: str | None = None) -> str:
    last_error: ValueError | None = None
    for raw_candidate in (value, fallback):
        candidate = collapse_text(raw_candidate)
        if not candidate:
            continue
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
                return datetime.strptime(candidate, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00")
            return datetime.fromisoformat(candidate).isoformat(timespec="seconds")
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise ValueError("timestamp must use YYYY-MM-DDTHH:MM:SS format") from last_error
    return now_iso()


# 规范化记忆类型，并限制在主数据当前允许的枚举值内。
def normalize_memory_kind(value: str | None) -> StoredMemoryKind:
    normalized = collapse_text(value)
    if normalized not in MEMORY_KIND_VALUES:
        raise ValueError(
            "memory_kind must be one of 'project_record', 'chat_event', 'fact', 'project_registry' or 'field_record'"
        )
    return normalized  # type: ignore[return-value]


# 判断一条主数据记录是不是内部字段级记忆；调用方包括 search、timeline、save/update/delete 的边界校验。
# 输入是主数据中的单条记录 dict；输出是 bool，True 表示它的 memory_kind 是内部类型 field_record。
def is_field_record_entry(entry: dict[str, Any]) -> bool:
    return str(entry.get("memory_kind") or "").strip() == "field_record"


# 按主记忆类型返回允许 update 的业务字段集合；server.py 会先拿它校验补丁，再决定是否继续重建 payload。
def get_update_allowed_fields_for_memory_kind(memory_kind: str) -> tuple[str, ...]:
    normalized_memory_kind = str(memory_kind or "").strip()
    if normalized_memory_kind == "project_record":
        return (
            "title",
            "short_summary",
            "detailed_summary",
            "problem_background",
            "analysis",
            "action_steps",
            "validation_result",
            "tags",
            "source_paths",
            "reference_doc_path",
        )
    if normalized_memory_kind == "project_registry":
        return (
            "title",
            "overview_summary",
            "tags",
            "source_paths",
            "reference_doc_path",
        )
    return (
        "title",
        "short_summary",
        "detailed_summary",
        "tags",
        "source_paths",
        "reference_doc_path",
    )


# 基于已经取到的源记录，进一步拦住“字段名虽然全局存在，但不属于该类型”的 update 补丁。
def validate_update_changes_for_entry(source_entry: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    resolved_memory_kind = str(source_entry.get("memory_kind") or "").strip()
    allowed_fields = set(get_update_allowed_fields_for_memory_kind(resolved_memory_kind))
    unsupported_fields = [
        field_name
        for field_name in changes
        if field_name not in allowed_fields
    ]
    if unsupported_fields:
        unsupported_fields_text = ", ".join(sorted(unsupported_fields))
        raise ValueError(
            f"changes contains fields unsupported for {resolved_memory_kind}: {unsupported_fields_text}"
        )
    return changes


# 按主记忆类型返回这一类记录真正应该拆成 field_record 的字段范围。
def get_field_record_candidate_fields_for_memory_kind(memory_kind: str) -> tuple[str, ...]:
    normalized_memory_kind = str(memory_kind or "").strip()
    if normalized_memory_kind == "project_registry":
        return PROJECT_REGISTRY_FIELD_RECORD_CANDIDATE_FIELDS
    return tuple(
        field_name
        for field_name in FIELD_RECORD_CANDIDATE_FIELDS
        if field_name != "overview_summary"
    )


# 按主记忆类型返回真正参与 embedding 文本拼接的字段范围。
def get_retrieval_field_candidates_for_memory_kind(memory_kind: str) -> tuple[str, ...]:
    normalized_memory_kind = str(memory_kind or "").strip()
    if normalized_memory_kind == "fact":
        return FACT_RETRIEVAL_FIELD_CANDIDATES
    if normalized_memory_kind == "project_registry":
        return PROJECT_REGISTRY_RETRIEVAL_FIELD_CANDIDATES
    return tuple(RETRIEVAL_FIELD_CANDIDATES)


# 生成项目注册表的稳定 id；create_project 会先调这里拿到 projRegId- 前缀 id，
# 再把它同时写进 project_registry.id、project_id 和 memory_registry，旧 proj-* / projRegisterMemId-* 继续只作为历史兼容格式保留。
def build_project_id() -> str:
    return f"projRegId-{hashlib.sha256(f'{now_iso()}|{time.time_ns()}'.encode('utf-8')).hexdigest()[:12]}"


# 从整库记录里筛出对外可见的主记忆；它服务于时间线、search 折叠回源、total_entries 统计等链路。
# 输入是 list[dict[str, Any]] 的完整主数据列表；输出是排除了内部 field_record 的主记忆列表。
def list_public_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if not is_field_record_entry(entry)
    ]


# 校验参考文档路径；新写入时要求文件存在，回读历史记录时允许保留已失效的原始路径。
# 输入：value 是调用方传入的参考文档路径字符串，允许为 None；require_exists=True 表示按“新写入”规则校验文件必须真实存在且可读，False 表示按“回读旧记录”规则只保留原始路径文本。
# 输出：返回整理后的路径字符串；如果 value 为空则返回空字符串，如果 require_exists=True 且路径不存在、不是文件或不可读，则抛出 ValueError。
def resolve_reference_doc_path(value: str | None, require_exists: bool = True) -> str:
    resolved = str(value or "").strip()  # 清洗成干净的 str
    if not resolved:
        return ""

    # 把已经清洗过的路径字符串包装成 pathlib.Path 对象，后面才能直接调用 exists()、is_file() 这些文件系统检查方法。
    candidate = Path(resolved)
    if not require_exists:
        return str(candidate)
    if not candidate.exists():
        raise ValueError(f"reference_doc_path does not exist: {resolved}")
    if not candidate.is_file():
        raise ValueError(f"reference_doc_path must be a file: {resolved}")
    if not os.access(candidate, os.R_OK):
        raise ValueError(f"reference_doc_path is not readable: {resolved}")
    # 转成 str 并 return
    return str(candidate)


# 从标题和总结中提取一批可读性还可以的标签候选词，作为服务端兜底。
def extract_tag_candidates(text: str) -> list[str]:
    pattern = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,31}|[\u4e00-\u9fff]{2,8}")
    candidates: list[str] = []
    for raw in pattern.findall(text):
        token = raw.strip("._- ")
        if not token:
            continue
        candidates.append(token)
    return candidates


# 当调用方没有提供标签时，按“AI 优先、服务端兜底”的原则生成基础标签。
def build_fallback_tags(memory_kind: StoredMemoryKind, title: str, detailed_summary: str) -> list[str]:
    generated: list[str] = []
    if memory_kind == "project_record":
        generated.append("project-record")
    elif memory_kind == "chat_event":
        generated.append("chat-event")
    elif memory_kind == "fact":
        generated.append("fact")
    elif memory_kind == "project_registry":
        generated.append("project-registry")
    else:
        generated.append("field-record")

    for token in extract_tag_candidates(f"{title}\n{detailed_summary}"):
        if token in generated:
            continue
        generated.append(token)
        if len(generated) >= TAG_FALLBACK_LIMIT:
            break

    return generated


# 为 save/update 的成功返回生成统一的 fact 提取提醒对象；调用方看到它后再自行判断是否值得继续提取 fact。
def build_fact_extraction_reminder() -> dict[str, Any]:
    return {
        "decision_by_caller": True,
        "message": "这条非 fact 记忆写入成功后，是否还值得继续提取长期可复用 fact，由调用方判断；优先关注个人长期事实，而不是把已经写清的项目排查和解决过程再重复沉淀成 fact。",
        "recommended_steps": [
            "先判断这条记忆里是否真的包含长期可复用、跨会话稳定、会持续影响回答的事实；如果只是项目问题的完整排查与解决链路，且 project_record 已经写清，通常不需要再额外提取 fact。",
            "若包含，先 search 相近 fact。",
            "若存在语义相近 fact，优先 update。",
            "若不存在相近 fact，再 save 新 fact。",
        ],
    }


# 按字段类型把 field_record 的单字段值规范化成最终存储形态。
# title / short_summary 收口成单行字符串，tags 收口成字符串列表，reference_doc_path 保留路径文本，其余长文本按块级正文规范化。
def normalize_field_record_field_value(field_name: str, raw_value: Any) -> str | list[str]:
    if field_name == "tags":
        return normalize_list(coerce_tag_list_input(raw_value))
    if field_name in {"title", "short_summary"}:
        return collapse_text(raw_value)
    if field_name == "reference_doc_path":
        return resolve_reference_doc_path(raw_value, require_exists=False)
    return normalize_text_block(raw_value)


# 从现有 field_record 中提取“真正属于 source_field_name 的值”。
# 当前 field_record 只有最小必要字段，source_field_value_json/source_field_value_text 就是唯一事实源。
def resolve_field_record_value(entry: dict[str, Any], source_field_name: str) -> str | list[str]:
    minimal_json_value = entry.get("source_field_value_json")
    return load_field_record_value_json(source_field_name, minimal_json_value, entry.get("source_field_value_text"))


# 构造 one-hot 形态的 field_record：只让 source_field_name 对应字段有值，其余业务字段全部清空。
# 这个函数服务于 save/update 的字段派生和旧 field_record 的重写规范化，确保内部索引资产始终维持单字段语义。
def build_one_hot_field_record_entry(
    source_memory_id: str,
    source_field_name: str,
    field_value: str | list[str],
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    if source_field_name not in FIELD_RECORD_CANDIDATE_FIELDS:
        raise ValueError(f"unsupported field_record source field: {source_field_name}")

    payload: dict[str, Any] = {
        "memory_kind": "field_record",
        "title": "",
        "short_summary": "",
        "problem_background": "",
        "analysis": "",
        "action_steps": "",
        "validation_result": "",
        "detailed_summary": "",
        "tags": [],
        "source_paths": [],
        "reference_doc_path": "",
        "retrieval_fields": [source_field_name],
        "source_field_value_text": render_field_record_value_text(field_value),
        "source_field_value_json": dump_field_record_value_json(field_value),
    }
    payload[source_field_name] = list(field_value) if isinstance(field_value, list) else field_value
    
    return {
        "id": build_field_record_id(source_memory_id, source_field_name),
        "created_at": created_at,
        "updated_at": updated_at,
        "source_memory_id": source_memory_id,
        "source_field_name": source_field_name,
        **payload,
    }


# 从一条已经规范化好的主记忆里提取某个字段，并把它收口成单条 field_record。
# 这个函数服务 save/update 的字段派生，让主流程只关心“哪个字段要写”，不用重复手拼 one-hot 子记录。
def build_field_record_entry_for_field(source_entry: dict[str, Any], field_name: str) -> dict[str, Any] | None:
    if is_field_record_entry(source_entry):
        raise ValueError("field_record source entry must be a public main memory")
    if field_name not in FIELD_RECORD_CANDIDATE_FIELDS:
        raise ValueError(f"unsupported field_record source field: {field_name}")

    source_memory_id = str(source_entry.get("id") or "").strip()
    if not source_memory_id:
        raise ValueError("field_record source entry must have a non-empty id")

    field_value = normalize_field_record_field_value(field_name, source_entry.get(field_name))
    if not field_value:
        return None

    return build_one_hot_field_record_entry(
        source_memory_id=source_memory_id,
        source_field_name=field_name,
        field_value=field_value,
        created_at=str(source_entry.get("created_at") or "").strip(),
        updated_at=str(source_entry.get("updated_at") or "").strip(),
    )


# 为当前记录生成简短摘要，优先复用调用方提供的内容，否则从详细总结中提炼首条有效信息。
def build_short_summary(
    title: str,
    detailed_summary: str,
    short_summary: str | None = None,
) -> str:
    resolved_short_summary = collapse_text(short_summary)
    if resolved_short_summary:
        if len(resolved_short_summary) <= SHORT_SUMMARY_MAX_LENGTH:
            return resolved_short_summary
        return f"{resolved_short_summary[:SHORT_SUMMARY_MAX_LENGTH].rstrip()}..."

    normalized_title = collapse_text(title)
    normalized_summary = normalize_text_block(detailed_summary)
    candidate_lines = [collapse_text(line) for line in normalized_summary.split("\n") if collapse_text(line)]
    skip_prefixes = ("标题：", "title:", "title：")
    preferred_lines = [
        line
        for line in candidate_lines
        if line and line != normalized_title and not line.lower().startswith(skip_prefixes)
    ]
    selected_line = preferred_lines[0] if preferred_lines else (candidate_lines[0] if candidate_lines else normalized_title)
    selected_line = re.sub(r"^(问题背景|问题|分析|步骤|验证结果|摘要|总结)[：:]\s*", "", selected_line).strip()
    if len(selected_line) <= SHORT_SUMMARY_MAX_LENGTH:
        return selected_line
    return f"{selected_line[:SHORT_SUMMARY_MAX_LENGTH].rstrip()}..."


# 按当前记忆类型计算真正参与向量拼接的字段名列表。
def build_retrieval_fields(entry: dict[str, Any]) -> list[str]:
    memory_kind = str(entry.get("memory_kind") or "").strip()
    if memory_kind == "field_record":
        source_field_name = str(entry.get("source_field_name") or "").strip()
        if source_field_name not in FIELD_RECORD_CANDIDATE_FIELDS:
            return []
        field_value = resolve_field_record_value(entry, source_field_name)
        return [source_field_name] if field_value else []
    field_candidates = get_retrieval_field_candidates_for_memory_kind(memory_kind)

    fields: list[str] = []
    for field_name in field_candidates:
        value = entry.get(field_name)
        if field_name == "tags":
            if isinstance(value, list) and value:
                fields.append(field_name)
            continue
        if normalize_text_block(value):
            fields.append(field_name)
    return fields


# 按统一规则把一条记录拼成最终用于向量化检索的文本。
def build_retrieval_text(entry: dict[str, Any]) -> str:
    if str(entry.get("memory_kind") or "").strip() == "field_record":
        source_field_name = str(entry.get("source_field_name") or "").strip()
        if source_field_name not in FIELD_RECORD_CANDIDATE_FIELDS:
            return ""
        field_value = resolve_field_record_value(entry, source_field_name)
        if isinstance(field_value, list):
            return "，".join(field_value)
        return str(field_value or "")

    parts: list[str] = []
    for field_name in build_retrieval_fields(entry):
        value = entry.get(field_name)
        if field_name == "tags":
            rendered = "，".join(value or [])
        else:
            rendered = normalize_text_block(value)
        if not rendered:
            continue
        parts.append(f"{FIELD_LABELS[field_name]}：{rendered}")
    return "\n".join(parts)


# 统一筛出真正允许进入向量索引的记录，并把 id / 原记录 / 检索文本绑定在一起。
# rebuild_vector_index 和 update_store_and_rebuild 都通过这里决定谁能写进 row_to_id，避免矩阵行数和 id 顺序错位。
def build_indexable_entry_payloads(entries: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    payloads: list[tuple[str, dict[str, Any], str]] = []
    for entry in entries:
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        retrieval_text = build_retrieval_text(entry)
        if not collapse_text(retrieval_text):
            continue
        payloads.append((entry_id, entry, retrieval_text))
    return payloads


# 根据结果或详细总结生成适合时间线显示的单行预览，避免 Markdown 视图过长。
def preview_text(value: str | None, limit: int = 160) -> str:
    single_line = collapse_text(value)
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit].rstrip()}..."


# 生成标准化后的记录载荷，作为写入、更新和向量拼接的统一输入。
def build_payload(
    memory_kind: str,
    title: str,
    detailed_summary: str,
    short_summary: str | None = None,
    overview_summary: str | None = None,
    problem_background: str | None = None,
    analysis: str | None = None,
    action_steps: str | None = None,
    validation_result: str | None = None,
    project_id: str | None = None,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    strict_project_requirements: bool = True,
    require_short_summary: bool = True,
    validate_reference_doc_path: bool = True,
) -> dict[str, Any]:
    resolved_memory_kind = normalize_memory_kind(memory_kind)
    resolved_title = collapse_text(title)
    resolved_overview_summary = normalize_text_block(overview_summary)
    resolved_problem_background = normalize_text_block(problem_background)
    resolved_analysis = normalize_text_block(analysis)
    resolved_action_steps = normalize_text_block(action_steps)
    resolved_validation_result = normalize_text_block(validation_result)
    resolved_detailed_summary = normalize_text_block(detailed_summary)
    resolved_project_id = collapse_text(project_id)
    resolved_source_paths = normalize_list(coerce_source_path_list_input(source_paths))
    resolved_reference_doc_path = resolve_reference_doc_path(
        reference_doc_path,
        require_exists=validate_reference_doc_path,
    )
    resolved_tags = normalize_list(coerce_tag_list_input(tags))

    if not resolved_title:
        raise ValueError("title is required")
    if resolved_memory_kind == "project_registry":
        if not resolved_overview_summary:
            raise ValueError("overview_summary is required for project_registry")
    else:
        if not resolved_detailed_summary:
            raise ValueError("detailed_summary is required")
        if require_short_summary and not collapse_text(short_summary):
            raise ValueError("short_summary is required")
    if resolved_memory_kind == "project_record" and strict_project_requirements:
        if not resolved_problem_background:
            raise ValueError("problem_background is required for project_record")
        if not resolved_analysis:
            raise ValueError("analysis is required for project_record")
        if not resolved_action_steps:
            raise ValueError("action_steps is required for project_record")
        if not resolved_validation_result:
            raise ValueError("validation_result is required for project_record")

    if not resolved_tags:
        resolved_tags = build_fallback_tags(
            memory_kind=resolved_memory_kind,
            title=resolved_title,
            detailed_summary=resolved_detailed_summary,
        )

    if resolved_memory_kind == "project_registry":
        resolved_short_summary = build_short_summary(
            title=resolved_title,
            detailed_summary=resolved_overview_summary,
        )
        resolved_detailed_summary = resolved_overview_summary
    else:
        resolved_short_summary = build_short_summary(
            title=resolved_title,
            detailed_summary=resolved_detailed_summary,
            short_summary=short_summary,
        )

    payload = {
        "memory_kind": resolved_memory_kind,
        "title": resolved_title,
        "short_summary": resolved_short_summary,
        "overview_summary": resolved_overview_summary if resolved_memory_kind == "project_registry" else "",
        "problem_background": resolved_problem_background,
        "analysis": resolved_analysis,
        "action_steps": resolved_action_steps,
        "validation_result": resolved_validation_result,
        "detailed_summary": resolved_detailed_summary,
        "project_id": resolved_project_id if resolved_memory_kind in {"project_record", "project_registry"} else "",
        "tags": resolved_tags,
        "source_paths": resolved_source_paths,
        "reference_doc_path": resolved_reference_doc_path,
    }
    payload["retrieval_fields"] = build_retrieval_fields(payload)
    return payload


# 基于类型生成新的带 UUID 后缀的随机 ID。
def generate_memory_id(memory_kind: str) -> str:
    normalized_kind = str(memory_kind or "").strip()
    uuid_suffix = uuid.uuid4().hex[:12]
    if normalized_kind == "project_registry":
        return f"projRegisterMemId-{uuid_suffix}"
    elif normalized_kind == "project_record":
        return f"projMemId-{uuid_suffix}"
    elif normalized_kind == "fact":
        return f"factMemId-{uuid_suffix}"
    elif normalized_kind == "chat_event":
        return f"eventMemId-{uuid_suffix}"
    return f"pmem-{uuid_suffix}"


# 为内部 field_record 生成稳定 id；它属于服务端派生规则，不直接暴露给 save/update 调用方。
# 输入包括源头主记忆 id 和原字段名；输出是稳定的 str 类型内部记录 id，例如 fmem-xxxx。
def build_field_record_id(source_memory_id: str, source_field_name: str) -> str:
    id_seed = f"{source_memory_id}|{source_field_name}"
    return f"fmem-{hashlib.sha256(id_seed.encode('utf-8')).hexdigest()[:12]}"


# 把主数据中的单条记录规范成当前正式 schema。
# 输入必须已经是当前字段口径；这个函数负责统一正文、检索字段、id 和时间字段。
def normalize_store_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(f"Every entry in {SQLITE_PATH.name} must be an object")

    raw_title = entry.get("title")
    raw_short_summary = entry.get("short_summary")
    raw_overview_summary = entry.get("overview_summary")
    raw_problem_background = entry.get("problem_background")
    raw_analysis = entry.get("analysis")
    raw_action_steps = entry.get("action_steps")
    raw_validation_result = entry.get("validation_result")
    raw_memory_kind = entry.get("memory_kind") or "project_record"
    raw_detailed_summary = entry.get("detailed_summary")
    raw_project_id = entry.get("project_id")
    raw_source_paths = entry.get("source_paths")
    raw_reference_doc_path = entry.get("reference_doc_path")
    raw_created_at = entry.get("created_at")
    raw_updated_at = entry.get("updated_at")
    raw_source_memory_id = entry.get("source_memory_id")
    raw_source_field_name = entry.get("source_field_name")
    normalized_created_at = resolve_record_timestamp(str(raw_created_at or ""))
    normalized_updated_at = resolve_record_timestamp(str(raw_updated_at or ""))

    resolved_memory_kind = normalize_memory_kind(str(raw_memory_kind))
    if resolved_memory_kind == "field_record":
        normalized_source_memory_id = str(raw_source_memory_id or "").strip()
        normalized_source_field_name = str(raw_source_field_name or "").strip()
        if not normalized_source_memory_id or not normalized_source_field_name:
            raise RuntimeError("field_record entries must contain source_memory_id and source_field_name")
        field_value = resolve_field_record_value(entry, normalized_source_field_name)
        if not field_value:
            raise RuntimeError("field_record entries must contain a non-empty source field value")
        return build_one_hot_field_record_entry(
            source_memory_id=normalized_source_memory_id,
            source_field_name=normalized_source_field_name,
            field_value=field_value,
            created_at=normalized_created_at,
            updated_at=normalized_updated_at,
        )

    payload = build_payload(
        memory_kind=str(raw_memory_kind),
        title=str(raw_title or ""),
        detailed_summary=str(raw_detailed_summary or ""),
        short_summary=str(raw_short_summary or ""),
        overview_summary=str(raw_overview_summary or ""),
        problem_background=str(raw_problem_background or ""),
        analysis=str(raw_analysis or ""),
        action_steps=str(raw_action_steps or ""),
        validation_result=str(raw_validation_result or ""),
        project_id=str(raw_project_id or ""),
        tags=entry.get("tags"),
        source_paths=raw_source_paths,
        reference_doc_path=str(raw_reference_doc_path or ""),
        strict_project_requirements=False,
        require_short_summary=False,
        validate_reference_doc_path=False,
    )

    
    normalized_id = str(entry.get("id") or "").strip() or generate_memory_id(str(raw_memory_kind))

    normalized_entry = {
        "id": normalized_id,
        "created_at": normalized_created_at,
        "updated_at": normalized_updated_at,
        **payload,
    }
    return normalized_entry


# 校验 update 的补丁对象；上层函数 update_record 会先经过这里，挡住未知字段和系统字段。
def validate_update_changes(changes: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    if not changes:
        raise ValueError("changes must not be empty")

    normalized_changes: dict[str, Any] = {}
    for raw_field_name, raw_field_value in changes.items():
        field_name = str(raw_field_name or "").strip()
        if not field_name:
            raise ValueError("changes contains an empty field name")
        if field_name == "memory_kind":
            raise ValueError(
                "changes.memory_kind is no longer supported; save a new record with the target type and delete the old one if needed"
            )
        if field_name == "project_id":
            raise ValueError("changes.project_id is not supported; project binding is immutable in update")
        if field_name in UPDATE_SYSTEM_FIELDS:
            raise ValueError(f"changes must not contain system field '{field_name}'")
        if field_name not in UPDATE_ALLOWED_FIELDS:
            raise ValueError(f"changes contains unsupported field '{field_name}'")
        if field_name in UPDATE_LIST_FIELDS and raw_field_value is None:
            raise ValueError(f"changes.{field_name} must use [] to clear the list")
        if raw_field_value is None and field_name not in UPDATE_NULL_CLEARABLE_FIELDS:
            raise ValueError(f"changes.{field_name} does not support null")
        normalized_changes[field_name] = raw_field_value

    if not normalized_changes:
        raise ValueError("changes must not be empty")
    return normalized_changes


# 用当前记录和已校验的 update 补丁合成一条候选新记录；上层函数 update_record 保留它，是为了让入口层继续看得见“补丁 -> 重建”主链路。
def build_update_candidate_record(source_entry: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    payload_inputs: dict[str, Any] = {}
    for field_name in get_update_allowed_fields_for_memory_kind(str(source_entry.get("memory_kind") or "")):
        if field_name in changes:
            payload_inputs[field_name] = changes[field_name]
        else:
            payload_inputs[field_name] = source_entry.get(field_name)

    payload = build_payload(
        memory_kind=str(source_entry.get("memory_kind") or ""),
        title=str(payload_inputs["title"] or ""),
        detailed_summary=str(payload_inputs.get("detailed_summary") or source_entry.get("detailed_summary") or ""),
        short_summary=payload_inputs.get("short_summary"),
        overview_summary=payload_inputs.get("overview_summary"),
        problem_background=payload_inputs.get("problem_background"),
        analysis=payload_inputs.get("analysis"),
        action_steps=payload_inputs.get("action_steps"),
        validation_result=payload_inputs.get("validation_result"),
        project_id=source_entry.get("project_id"),
        tags=payload_inputs.get("tags"),
        source_paths=payload_inputs.get("source_paths"),
        reference_doc_path=payload_inputs.get("reference_doc_path"),
    )
    

    return {
        "id": str(source_entry.get("id") or "").strip(),
        "created_at": str(source_entry.get("created_at") or "").strip(),
        "updated_at": str(source_entry.get("updated_at") or "").strip(),
        **payload,
    }


# 从一条主记忆派生一批 field_record；这批记录只进入主数据和索引层，不作为对外可直接维护的对象。
# 输入是单条主记忆 dict；输出是 list[dict[str, Any]]，每一项都带 source_memory_id/source_field_name，用于后续 search 折叠回源。
def build_field_record_entries(source_entry: dict[str, Any]) -> list[dict[str, Any]]:
    if is_field_record_entry(source_entry):
        return []

    source_memory_id = str(source_entry.get("id") or "").strip()
    if not source_memory_id:
        raise ValueError("field_record source entry must have a non-empty id")

    field_entries: list[dict[str, Any]] = []
    for field_name in get_field_record_candidate_fields_for_memory_kind(str(source_entry.get("memory_kind") or "")):
        field_entry = build_field_record_entry_for_field(source_entry, field_name)
        if field_entry is not None:
            field_entries.append(field_entry)
    return field_entries


# 只把一条主记忆里的某一个字段写成 field_record；update 的字段级增量会复用这里，避免每次整家重建。
def save_fieldRecord(
    source_entry: dict[str, Any],
    field_name: str,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    field_entry = build_field_record_entry_for_field(source_entry, field_name)
    if field_entry is None:
        return None
    upsert_store_entries([field_entry], connection=connection)
    return field_entry


# 比较旧主记忆和新主记忆在 field_record 候选字段上的差异，产出这次需要删哪些旧子记录、补哪些新子记录。
# 这个函数同时服务 SQLite 主数据精确删改和向量缓存精确复用，避免两边各写一套字段比较标准。
def plan_field_record_delta(
    previous_entry: dict[str, Any],
    updated_entry: dict[str, Any],
    touched_field_names: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    if is_field_record_entry(previous_entry) or is_field_record_entry(updated_entry):
        raise ValueError("plan_field_record_delta only accepts public main memories")

    normalized_touched_fields = {
        str(field_name or "").strip()
        for field_name in touched_field_names
        if str(field_name or "").strip()
    }
    candidate_field_names = {
        field_name
        for field_name in normalized_touched_fields
        if field_name in set(get_field_record_candidate_fields_for_memory_kind(str(updated_entry.get("memory_kind") or "")))
        or field_name in set(get_field_record_candidate_fields_for_memory_kind(str(previous_entry.get("memory_kind") or "")))
    }

    ordered_candidate_fields = [
        *[
            field_name
            for field_name in get_field_record_candidate_fields_for_memory_kind(str(previous_entry.get("memory_kind") or ""))
            if field_name in candidate_field_names
        ],
        *[
            field_name
            for field_name in get_field_record_candidate_fields_for_memory_kind(str(updated_entry.get("memory_kind") or ""))
            if field_name in candidate_field_names
            and field_name not in get_field_record_candidate_fields_for_memory_kind(str(previous_entry.get("memory_kind") or ""))
        ],
    ]
    delta_plan = {
        "unchanged_fields": set(),
        "created_fields": set(),
        "updated_fields": set(),
        "deleted_fields": set(),
        "fields_to_delete": [],
        "fields_to_upsert": [],
        "removed_field_record_ids": [],
    }
    if not ordered_candidate_fields:
        return delta_plan

    for field_name in ordered_candidate_fields:
        previous_field_entry = build_field_record_entry_for_field(previous_entry, field_name)
        updated_field_entry = build_field_record_entry_for_field(updated_entry, field_name)

        if previous_field_entry is None and updated_field_entry is None:
            delta_plan["unchanged_fields"].add(field_name)
            continue
        if previous_field_entry is not None and updated_field_entry is not None:
            if field_name not in normalized_touched_fields:
                delta_plan["unchanged_fields"].add(field_name)
                continue
        if previous_field_entry is None and updated_field_entry is not None:
            delta_plan["created_fields"].add(field_name)
            delta_plan["fields_to_upsert"].append(field_name)
            continue
        if previous_field_entry is not None and updated_field_entry is None:
            delta_plan["deleted_fields"].add(field_name)
            delta_plan["fields_to_delete"].append(field_name)
            delta_plan["removed_field_record_ids"].append(str(previous_field_entry.get("id") or "").strip())
            continue

        delta_plan["updated_fields"].add(field_name)
        delta_plan["fields_to_delete"].append(field_name)
        delta_plan["fields_to_upsert"].append(field_name)
        delta_plan["removed_field_record_ids"].append(str(previous_field_entry.get("id") or "").strip())

    return delta_plan


# 按主记忆生命周期同步 field_record；save 新增主记忆时只补新增源头，delete 时去掉对应子记录，完整重建时则整库重算。
# update 时不通过公共 delete 接口删除旧 field_record，而是直接按 source_memory_id 在内存里替换整家派生记录。
# 输入是已经规范化后的完整记录列表；输出仍是要写回主数据和索引的完整记录列表，其中会自动补齐或清理 field_record。
def synchronize_field_records(
    entries: list[dict[str, Any]],
    appended_source_ids: set[str] | None = None,
    deleted_source_ids: set[str] | None = None,
    updated_source_ids: set[str] | None = None,
    full_regenerate: bool = False,
) -> list[dict[str, Any]]:
    public_entries = list_public_entries(entries)
    public_entry_map = build_store_entry_map(public_entries)
    existing_field_entries = [
        entry
        for entry in entries
        if is_field_record_entry(entry)
        and str(entry.get("source_memory_id") or "").strip() in public_entry_map
    ]
    normalized_appended_source_ids = {
        str(source_id or "").strip()
        for source_id in set(appended_source_ids or set())
        if str(source_id or "").strip()
    }
    normalized_deleted_source_ids = {
        str(source_id or "").strip()
        for source_id in set(deleted_source_ids or set())
        if str(source_id or "").strip()
    }
    normalized_updated_source_ids = {
        str(source_id or "").strip()
        for source_id in set(updated_source_ids or set())
        if str(source_id or "").strip()
    }
    covered_source_ids = {
        str(entry.get("source_memory_id") or "").strip()
        for entry in existing_field_entries
        if str(entry.get("source_memory_id") or "").strip()
    }
    missing_covered_source_ids = [
        source_id
        for source_id in public_entry_map
        if source_id not in normalized_appended_source_ids
        and source_id not in normalized_updated_source_ids
        and source_id not in covered_source_ids
    ]
    if full_regenerate or missing_covered_source_ids:
        rebuilt_field_entries: list[dict[str, Any]] = []
        for entry in public_entries:
            rebuilt_field_entries.extend(build_field_record_entries(entry))
        return public_entries + rebuilt_field_entries

    retained_field_entries = [
        entry
        for entry in existing_field_entries
        if str(entry.get("source_memory_id") or "").strip() not in normalized_appended_source_ids
        and str(entry.get("source_memory_id") or "").strip() not in normalized_deleted_source_ids
        and str(entry.get("source_memory_id") or "").strip() not in normalized_updated_source_ids
    ]
    rebuilt_field_entries: list[dict[str, Any]] = []
    for entry in public_entries:
        if (
            str(entry.get("id") or "").strip() in normalized_appended_source_ids
            or str(entry.get("id") or "").strip() in normalized_updated_source_ids
        ):
            rebuilt_field_entries.extend(build_field_record_entries(entry))
    return public_entries + retained_field_entries + rebuilt_field_entries


# 读取当前主数据事实源；当前统一从 SQLite 读取。
def load_store() -> dict[str, Any]:
    ensure_sqlite_store_ready()
    normalized_entries: list[dict[str, Any]] = []
    for entry_batch in iter_store_entries_in_batches():
        normalized_entries.extend(entry_batch)
    return {
        "entries": normalized_entries,
    }


# 以原子方式写文本文件，避免中途写坏记忆文件。
def write_text_atomic(path: Path, text: str) -> None:
    ensure_memory_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")

    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2)

    path.write_text(text, encoding="utf-8")
    if tmp_path.exists():
        tmp_path.unlink()
    if last_error is not None:
        return


# 以原子方式写 JSON，保证结构化数据一致性。
def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    write_text_atomic(path, text + "\n")


# 生成统一的索引重建状态对象；这个对象会写进 rebuild_state.json，供 save、update、delete 和 search 共用。
def build_rebuild_state_snapshot(
    state: str,
    mode: str = "",
    started_at: str = "",
    finished_at: str = "",
    last_error: str = "",
    target_entry_count: int = 0,
    worker_pid: int = 0,
) -> dict[str, Any]:
    return {
        "state": str(state or "").strip() or "idle",
        "mode": str(mode or "").strip(),
        "started_at": str(started_at or "").strip(),
        "finished_at": str(finished_at or "").strip(),
        "last_error": str(last_error or "").strip(),
        "target_entry_count": int(target_entry_count or 0),
        "worker_pid": int(worker_pid or 0),
    }


# 以原子方式写 rebuild_state.json；前台请求和后台重建线程通过这份文件共享索引状态。
def write_rebuild_state(snapshot: dict[str, Any]) -> None:
    write_json_atomic(REBUILD_STATE_PATH, snapshot)


# 读取 rebuild_state.json，并在检测到“旧进程残留的 running 状态”时自动纠正为 failed。
# 这里直接用文件，而不是只靠进程内变量，是为了让后续请求能看到后台重建的最新状态。
def load_rebuild_state() -> dict[str, Any]:
    if not REBUILD_STATE_PATH.exists():
        return build_rebuild_state_snapshot(state="idle", finished_at=now_iso())

    try:
        with REBUILD_STATE_PATH.open("r", encoding="utf-8") as handle:
            raw_snapshot = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failed_snapshot = build_rebuild_state_snapshot(
            state="failed",
            mode=BACKGROUND_REBUILD_MODE,
            finished_at=now_iso(),
            last_error=f"rebuild_state.json is unreadable: {exc}",
            worker_pid=os.getpid(),
        )
        write_rebuild_state(failed_snapshot)
        return failed_snapshot

    if not isinstance(raw_snapshot, dict):
        failed_snapshot = build_rebuild_state_snapshot(
            state="failed",
            mode=BACKGROUND_REBUILD_MODE,
            finished_at=now_iso(),
            last_error="rebuild_state.json must contain an object.",
            worker_pid=os.getpid(),
        )
        write_rebuild_state(failed_snapshot)
        return failed_snapshot

    snapshot = build_rebuild_state_snapshot(
        state=str(raw_snapshot.get("state") or "idle"),
        mode=str(raw_snapshot.get("mode") or ""),
        started_at=str(raw_snapshot.get("started_at") or ""),
        finished_at=str(raw_snapshot.get("finished_at") or ""),
        last_error=str(raw_snapshot.get("last_error") or ""),
        target_entry_count=int(raw_snapshot.get("target_entry_count") or 0),
        worker_pid=int(raw_snapshot.get("worker_pid") or 0),
    )
    if snapshot["state"] == "running" and int(snapshot.get("worker_pid") or 0) != os.getpid():
        failed_snapshot = build_rebuild_state_snapshot(
            state="failed",
            mode=str(snapshot.get("mode") or BACKGROUND_REBUILD_MODE),
            started_at=str(snapshot.get("started_at") or ""),
            finished_at=now_iso(),
            last_error="Previous process exited while background index rebuild was still running.",
            target_entry_count=int(snapshot.get("target_entry_count") or 0),
            worker_pid=os.getpid(),
        )
        write_rebuild_state(failed_snapshot)
        return failed_snapshot
    return snapshot


# 在写入型工具入口统一拦住“后台全量重建仍在运行”的场景，避免新主数据和旧重建任务互相覆盖。
def ensure_write_operations_allowed() -> None:
    rebuild_state = load_rebuild_state()
    if str(rebuild_state.get("state") or "").strip() == "running":
        raise RuntimeError("Index rebuild is still running. Save, update, and delete are temporarily unavailable; retry later.")


# 读取当前 meta 里已落盘的索引统计；后台重建尚未完成时，上层返回会用它说明旧索引还停留在什么状态。
def load_current_index_stats() -> dict[str, int]:
    current_stats = {"count": 0, "dim": 0}
    if not META_PATH.exists():
        return current_stats
    try:
        with META_PATH.open("r", encoding="utf-8") as handle:
            current_meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return current_stats
    if not isinstance(current_meta, dict):
        return current_stats
    row_to_id = current_meta.get("row_to_id")
    if isinstance(row_to_id, list) and all(
        isinstance(row_item, str) and str(row_item or "").strip()
        for row_item in row_to_id
    ):
        current_stats["count"] = len(row_to_id)
    current_stats["dim"] = int(current_meta.get("dim") or 0)
    return current_stats


# 为 save/update 的异步慢路径生成统一的返回片段；它明确告诉调用方主数据已写入，但 search 需要等后台重建完成。
def build_background_rebuild_notice() -> dict[str, Any]:
    return {
        "index_refresh_state": "rebuilding",
        "index_refresh_mode": BACKGROUND_REBUILD_MODE,
        "search_available": False,
        "message": "主数据已写入，向量索引正在后台重建；重建完成前 search 不可用。",
    }


# 在当前 MCP 进程里后台执行一次完整索引重建；它只读最新 SQLite 主数据，并把索引侧文件补齐到一致状态。
def run_background_full_rebuild(started_at: str, model_path: str, target_entry_count: int) -> None:
    try:
        rebuild_stats = rebuild_vector_index(model_path=model_path)
        write_rebuild_state(
            build_rebuild_state_snapshot(
                state="idle",
                mode=BACKGROUND_REBUILD_MODE,
                started_at=started_at,
                finished_at=now_iso(),
                target_entry_count=int(rebuild_stats.get("count") or target_entry_count),
                worker_pid=os.getpid(),
            )
        )
    except Exception as exc:
        write_rebuild_state(
            build_rebuild_state_snapshot(
                state="failed",
                mode=BACKGROUND_REBUILD_MODE,
                started_at=started_at,
                finished_at=now_iso(),
                last_error=f"{type(exc).__name__}: {exc}",
                target_entry_count=target_entry_count,
                worker_pid=os.getpid(),
            )
        )


# 启动后台全量重建线程，并立刻返回给前台请求一个“主数据已写入、索引仍在补齐”的结构化提醒。
def launch_background_full_rebuild(model_path: str, target_entry_count: int) -> dict[str, Any]:
    global _BACKGROUND_REBUILD_THREAD

    with _BACKGROUND_REBUILD_LOCK:
        current_state = load_rebuild_state()
        if str(current_state.get("state") or "").strip() == "running":
            raise RuntimeError("Index rebuild is already running. Retry after it finishes.")

        started_at = now_iso()
        running_snapshot = build_rebuild_state_snapshot(
            state="running",
            mode=BACKGROUND_REBUILD_MODE,
            started_at=started_at,
            target_entry_count=target_entry_count,
            worker_pid=os.getpid(),
        )
        write_rebuild_state(running_snapshot)

        worker = threading.Thread(
            target=run_background_full_rebuild,
            args=(started_at, model_path, target_entry_count),
            name="memory-rag-mcp-background-rebuild",
            daemon=True,
        )
        try:
            worker.start()
        except Exception as exc:
            failed_snapshot = build_rebuild_state_snapshot(
                state="failed",
                mode=BACKGROUND_REBUILD_MODE,
                started_at=started_at,
                finished_at=now_iso(),
                last_error=f"Failed to start background rebuild thread: {exc}",
                target_entry_count=target_entry_count,
                worker_pid=os.getpid(),
            )
            write_rebuild_state(failed_snapshot)
            raise RuntimeError("Failed to start background index rebuild.") from exc

        _BACKGROUND_REBUILD_THREAD = worker
    return build_background_rebuild_notice()


# 以原子方式写本地 embedding 缓存矩阵，保证向量缓存不会在中途写坏。
def write_embeddings_atomic(path: Path, embeddings: np.ndarray) -> None:
    ensure_memory_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype="float32"), allow_pickle=False)

    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2)

    with path.open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype="float32"), allow_pickle=False)
    if tmp_path.exists():
        tmp_path.unlink()
    if last_error is not None:
        return


# 对传入的记录做稳定排序（按照 updated_at 字段），便于时间线输出和向量索引保持一致。
def sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda item: (
            item.get("updated_at", ""),
            item.get("created_at", ""),
            item.get("id", ""),
        ),
        reverse=True,
    )


# 渲染按最近更新时间整理的人类可读 Markdown 时间线。
def build_timeline_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG 记忆库时间线",
        "",
        "> 该文件由 MCP 的 save 接口自动生成和更新，请不要把它当作手工维护文档。",
        "> 这是 RAG 记忆库的按最近更新时间整理的时间线视图，用于人类回顾，不是主数据源。",
        f"> 主数据源：{SQLITE_PATH}",
        "",
        f"> 自动生成时间：{now_iso()}",
        "",
    ]

    sorted_entries = sort_entries(list_public_entries(entries))
    if not sorted_entries:
        lines.append("暂无记忆记录。")
        lines.append("")
        return "\n".join(lines)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in sorted_entries:
        updated_at = collapse_text(str(entry.get("updated_at") or ""))
        date_key = updated_at.split("T", 1)[0] if updated_at else "未知日期"
        grouped.setdefault(date_key, []).append(entry)

    for date_key in sorted(grouped.keys(), reverse=True):
        lines.append(f"## {date_key}")
        lines.append("")
        for entry in grouped[date_key]:
            summary_field_name = "short_summary" if entry.get("short_summary") else "detailed_summary"
            summary_label = "摘要（short_summary）" if summary_field_name == "short_summary" else "详细总结（detailed_summary）"
            summary_value = entry.get(summary_field_name) or ""
            lines.append(f"### {entry['title']}")
            lines.append(f"- ID（id）：{entry['id']}")
            lines.append(f"- 类型（memory_kind）：{entry['memory_kind']}")
            lines.append(f"- 更新时间（updated_at）：{entry['updated_at']}")
            lines.append(f"- 创建时间（created_at）：{entry['created_at']}")
            lines.append(f"- {summary_label}：{preview_text(summary_value)}")
            if entry.get("validation_result"):
                lines.append(f"- 验证结果（validation_result）：{preview_text(entry['validation_result'])}")
            if entry.get("tags"):
                lines.append(f"- 标签（tags）：{'，'.join(entry['tags'])}")
            if entry.get("reference_doc_path"):
                lines.append(f"- 参考文档（reference_doc_path）：{entry['reference_doc_path']}")
            if entry.get("source_paths"):
                lines.append(f"- 来源（source_paths）：{'；'.join(entry['source_paths'])}")
            lines.append("")

    return "\n".join(lines)


# 直接消费“已经按当前稳定顺序排好的公开主记忆批次”，生成时间线 Markdown。
# save/update/delete 写盘后用它按批读取 SQLite，避免为了时间线再把整库公开主记忆一次性搬进内存。
def build_timeline_markdown_from_batches(entry_batches: Any) -> str:
    lines = [
        "# RAG 记忆库时间线",
        "",
        "> 该文件由 MCP 的 save 接口自动生成和更新，请不要把它当作手工维护文档。",
        "> 这是 RAG 记忆库的按最近更新时间整理的时间线视图，用于人类回顾，不是主数据源。",
        f"> 主数据源：{SQLITE_PATH}",
        "",
        f"> 自动生成时间：{now_iso()}",
        "",
    ]
    current_date_key = ""
    has_entries = False

    for entry_batch in entry_batches:
        for entry in entry_batch:
            has_entries = True
            updated_at = collapse_text(str(entry.get("updated_at") or ""))
            date_key = updated_at.split("T", 1)[0] if updated_at else "未知日期"
            if date_key != current_date_key:
                current_date_key = date_key
                lines.append(f"## {date_key}")
                lines.append("")
            summary_field_name = "short_summary" if entry.get("short_summary") else "detailed_summary"
            summary_label = "摘要（short_summary）" if summary_field_name == "short_summary" else "详细总结（detailed_summary）"
            summary_value = entry.get(summary_field_name) or ""
            lines.append(f"### {entry['title']}")
            lines.append(f"- ID（id）：{entry['id']}")
            lines.append(f"- 类型（memory_kind）：{entry['memory_kind']}")
            lines.append(f"- 更新时间（updated_at）：{entry['updated_at']}")
            lines.append(f"- 创建时间（created_at）：{entry['created_at']}")
            lines.append(f"- {summary_label}：{preview_text(summary_value)}")
            if entry.get("validation_result"):
                lines.append(f"- 验证结果（validation_result）：{preview_text(entry['validation_result'])}")
            if entry.get("tags"):
                lines.append(f"- 标签（tags）：{'，'.join(entry['tags'])}")
            if entry.get("reference_doc_path"):
                lines.append(f"- 参考文档（reference_doc_path）：{entry['reference_doc_path']}")
            if entry.get("source_paths"):
                lines.append(f"- 来源（source_paths）：{'；'.join(entry['source_paths'])}")
            lines.append("")

    if not has_entries:
        lines.append("暂无记忆记录。")
        lines.append("")
    return "\n".join(lines)


# 解析 YAML 配置中的嵌入模型目录，并在真正使用前补一次目录存在性校验。
def resolve_embed_model_path() -> str:
    candidate_path = EMBED_MODEL_PATH
    if not candidate_path.is_dir():
        raise RuntimeError(
            "Embedding model directory configured in settings.yaml does not exist. "
            f"Checked path: {candidate_path}."
        )
    return str(candidate_path)


# 读取 YAML 配置中的嵌入设备；当前不再支持通过环境变量覆盖。
def resolve_embed_device() -> str:
    return EMBED_DEVICE


# 为当前 embedding 模型目录生成稳定的身份摘要；
# save/update 的缓存复用判断、search 的索引一致性校验和 meta 写盘都会用到它。
def build_embedding_model_fingerprint(model_path: str) -> str:
    normalized_model_path = str(model_path or "").strip()
    if not normalized_model_path:
        raise RuntimeError("Embedding model path must not be empty when building embedding_model_fingerprint.")

    model_dir = Path(normalized_model_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"Embedding model directory does not exist: {model_dir}")

    small_hash_targets = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "README.md",
    )
    large_hash_targets = (
        "pytorch_model.bin",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "colbert_linear.pt",
        "sparse_linear.pt",
    )
    sample_size = 64 * 1024
    digest = hashlib.sha256()
    digest.update(f"model_dir={model_dir.resolve()}".encode("utf-8"))

    for relative_name in small_hash_targets:
        target_path = model_dir / relative_name
        if not target_path.is_file():
            continue
        digest.update(f"small:{relative_name}:".encode("utf-8"))
        with target_path.open("rb") as handle:
            digest.update(handle.read())

    for relative_name in large_hash_targets:
        target_path = model_dir / relative_name
        if not target_path.is_file():
            continue
        target_size = target_path.stat().st_size
        digest.update(f"large:{relative_name}:{target_size}:".encode("utf-8"))
        with target_path.open("rb") as handle:
            digest.update(handle.read(sample_size))
            if target_size > sample_size:
                tail_offset = max(target_size - sample_size, 0)
                handle.seek(tail_offset)
                digest.update(handle.read(sample_size))

    return digest.hexdigest()


# 缓存 sentence-transformers 模型实例，避免每次工具调用都重复加载。
@lru_cache(maxsize=2)
def get_embedder(model_path: str, device: str) -> Any:
    return load_sentence_embedder(model_path, device=device)


# 把文本批量编码成归一化后的 float32 向量，供 FAISS 使用。
def encode_texts(texts: list[str], embedder: Any) -> np.ndarray:
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return l2_normalize(embeddings).astype("float32")


# 构建 id 到完整记录的内存映射，便于搜索命中后快速回源主数据。
def build_store_entry_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("id")): entry
        for entry in entries
        if str(entry.get("id") or "").strip()
    }


# 清洗详情接口传入的 ids，去掉空白项、按首次出现顺序折叠重复项，并保证最终列表非空。
def normalize_record_ids(record_ids: list[str] | None) -> list[str]:
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for record_id in list(record_ids or []):
        normalized_record_id = str(record_id or "").strip()
        if not normalized_record_id or normalized_record_id in seen_ids:
            continue
        normalized_ids.append(normalized_record_id)
        seen_ids.add(normalized_record_id)
    if not normalized_ids:
        raise ValueError("ids is required")
    return normalized_ids


# 这个函数就是根据传入的有序 id 列表写进 meta 文件的。
# entry_ids 参数是上游已经排好序、并且与 embedding 矩阵行顺序一一对应的当前整库记录 id 列表。
# update_store_and_rebuild / rebuild_vector_index 会先确定当前整库记录顺序，
# 再由 rebuild_index_from_embeddings 把这份顺序传进来。
# 这里的 model_path 只用于写入给人看的 meta["model"]，方便排查当前索引当时用了哪个模型目录，
# 它不再参与机器判断；真正给程序比对模型身份的是 embedding_model_fingerprint。
# dim 也不是配置常量，而是这次实际写盘的 embedding 矩阵列数，所以继续由上游按当前结果传进来。
# retrieval_field_signature 则继续直接使用模块级常量 RETRIEVAL_FIELD_SIGNATURE，
# 因为它来自启动时读取的 retrieval.field_candidates，表示当前代码这套检索字段拼接规则。
def write_vector_meta(
    entry_ids: list[str],
    model_path: str,
    embedding_model_fingerprint: str,
    dim: int,
) -> None:
    meta = {
        "model": model_path,
        "embedding_model_fingerprint": embedding_model_fingerprint,
        "dim": int(dim),
        "normalized": True,
        "retrieval_field_signature": RETRIEVAL_FIELD_SIGNATURE,
        "row_to_id": list(entry_ids),
    }
    write_json_atomic(META_PATH, meta)


# 校验并读取（若校验通过）当前仍可复用的本地向量缓存和 meta 索引。
# 上层函数 update_store_and_rebuild() 会在 save / delete 接口的快路径里优先走这里，来判断旧缓存还能不能继续复用。
# 这里要求调用方传入“本次请求最终选中的模型目录”对应的 embedding_model_fingerprint，
# 因为模型身份属于这次请求上下文，不应该在这个下层函数里再重复扫描模型目录生成一次。
# retrieval_field_signature 继续直接读取模块级常量 RETRIEVAL_FIELD_SIGNATURE，
# 因为它代表的是当前代码的字段拼接规则，不是每次请求单独生成的数据。
def load_reusable_embedding_cache(embedding_model_fingerprint: str) -> tuple[np.ndarray, list[str]] | None:
    if not EMBEDDINGS_PATH.exists() or not META_PATH.exists():
        return None

    try:
        with EMBEDDINGS_PATH.open("rb") as handle:
            embeddings = np.load(handle, allow_pickle=False)
    except (OSError, ValueError):
        return None

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        return None

    try:
        with META_PATH.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(meta, dict):
        return None
    if str(meta.get("embedding_model_fingerprint") or "").strip() != str(embedding_model_fingerprint or "").strip():
        return None
    if str(meta.get("retrieval_field_signature") or "") != RETRIEVAL_FIELD_SIGNATURE:
        return None

    row_to_id = meta.get("row_to_id")
    if not isinstance(row_to_id, list):
        return None
    if not all(isinstance(row_item, str) and str(row_item or "").strip() for row_item in row_to_id):
        return None
    row_ids = [str(row_item or "").strip() for row_item in row_to_id]

    normalized_embeddings = np.asarray(embeddings, dtype="float32")
    if normalized_embeddings.shape[0] != len(row_ids):
        return None

    dim = int(meta.get("dim") or 0)
    if not dim or normalized_embeddings.shape[1] != dim:
        return None

    return normalized_embeddings, row_ids


# update_store_and_rebuild、rebuild_vector_index 和增量缓存路径都通过这里把向量缓存与索引产物同步写盘。
# 这个函数只负责把“已经准备好的向量矩阵”和“已经确定好的模型身份信息”落成 .npy / .faiss / meta 三份产物。
# 它不会再自己生成 embedding_model_fingerprint；dim 也直接从当前 embeddings 的列数计算，
# 因为 dim 属于这次写盘结果，不适合提成服务级静态变量。
def rebuild_index_from_embeddings(
    entry_ids: list[str],
    embeddings: np.ndarray,
    model_path: str,
    embedding_model_fingerprint: str,
) -> dict[str, Any]:
    if not entry_ids:
        for artifact_path in (EMBEDDINGS_PATH, INDEX_PATH, META_PATH):
            if artifact_path.exists():
                artifact_path.unlink()
        return {"count": 0, "dim": 0}

    normalized_embeddings = l2_normalize(np.asarray(embeddings, dtype="float32")).astype("float32")
    if normalized_embeddings.ndim != 2:
        raise RuntimeError("Embedding cache must be a 2D float32 matrix.")
    if normalized_embeddings.shape[0] != len(entry_ids):
        raise RuntimeError("Embedding cache row count does not match current entry count.")

    dim = int(normalized_embeddings.shape[1])
    write_embeddings_atomic(EMBEDDINGS_PATH, normalized_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    write_vector_meta(entry_ids, model_path, embedding_model_fingerprint, dim)
    return {"count": len(entry_ids), "dim": dim}


# 首次建库和缓存自愈
# 上层函数 update_store_and_rebuild 通过这里执行“重新编码全部记录”的完整后门
def rebuild_vector_index(
    entries: list[dict[str, Any]] | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    resolved_model_path = model_path or resolve_embed_model_path()
    current_embedding_model_fingerprint = build_embedding_model_fingerprint(resolved_model_path)
    if entries is None:
        ordered_entry_ids: list[str] = []
        embedding_batches: list[np.ndarray] = []
        resolved_embedder = embedder
        for entry_batch in iter_store_entries_in_batches():
            indexable_payloads = build_indexable_entry_payloads(entry_batch)
            if not indexable_payloads:
                continue
            if resolved_embedder is None:
                resolved_embedder = get_embedder(resolved_model_path, resolve_embed_device())
            ordered_entry_ids.extend([entry_id for entry_id, _, _ in indexable_payloads])
            batch_embeddings = encode_texts(
                [retrieval_text for _, _, retrieval_text in indexable_payloads],
                resolved_embedder,
            )
            embedding_batches.append(batch_embeddings)
        if not ordered_entry_ids:
            return rebuild_index_from_embeddings(
                [],
                np.empty((0, 0), dtype="float32"),
                resolved_model_path,
                current_embedding_model_fingerprint
            )
        rebuilt_embeddings = np.vstack(embedding_batches).astype("float32")
        return rebuild_index_from_embeddings(
            ordered_entry_ids,
            rebuilt_embeddings,
            resolved_model_path,
            current_embedding_model_fingerprint
        )
    if not entries:
        return rebuild_index_from_embeddings(
            [],
            np.empty((0, 0), dtype="float32"),
            resolved_model_path,
            current_embedding_model_fingerprint
        )

    resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
    sorted_entries = sort_entries(entries)
    indexable_payloads = build_indexable_entry_payloads(sorted_entries)
    if not indexable_payloads:
        return rebuild_index_from_embeddings(
            [],
            np.empty((0, 0), dtype="float32"),
            resolved_model_path,
            current_embedding_model_fingerprint
        )
    embeddings = encode_texts(
        [retrieval_text for _, _, retrieval_text in indexable_payloads],
        resolved_embedder,
    )
    return rebuild_index_from_embeddings(
        [entry_id for entry_id, _, _ in indexable_payloads],
        embeddings,
        resolved_model_path,
        current_embedding_model_fingerprint
    )


# 顶层函数是 save_record、update_record 和 delete_records_by_ids。
# 这里不再要求上层先组一份完整整库列表，而是根据“这次新增了谁、删了谁、更新了谁”直接改 SQLite 主数据，
# 再基于新的主数据同步 field_record、重建时间线.md，并复用已有矩阵缓存重建索引。
# 没有矩阵缓存就走全量重建路线；update 时如果显式给出旧主记忆和变更字段集合，就把 field_record 收窄成字段级增量，其余旧向量继续复用。
def update_store_and_rebuild(
    embedder: Any | None = None,
    model_path: str | None = None,  # embeddeding 模型目录
    appended_entry: dict[str, Any] | None = None,  # 告诉函数“这次是新增了一条，而且这一条是谁”，方便走增量 embedding
    previous_entry: dict[str, Any] | None = None,  # 告诉函数“update 前的旧主记忆是谁”，方便比较本次到底影响了哪些 field_record
    updated_entry: dict[str, Any] | None = None,  # 告诉函数“这次更新后的主记忆是谁”，方便先按行写回再重算该家族
    deleted_ids: list[str] | None = None,   # 告诉函数“这次删了哪些 id”，方便复用旧向量缓存
    updated_source_ids: set[str] | None = None,  # 告诉函数“这次更新了哪些主记忆家族”，方便只重算这一家的主记忆和 field_record
    updated_field_names: set[str] | list[str] | tuple[str, ...] | None = None,  # 告诉函数“这次 update 真正改了哪些业务字段”，方便把 field_record 收窄成字段级增量
) -> dict[str, Any]:
    ensure_sqlite_store_ready()
    normalized_appended_entry = normalize_store_entry(appended_entry) if appended_entry is not None else None
    normalized_previous_entry = previous_entry if previous_entry is not None else None
    normalized_updated_entry = normalize_store_entry(updated_entry) if updated_entry is not None else None
    deleted_id_set = {str(record_id or "").strip() for record_id in deleted_ids or [] if str(record_id or "").strip()}
    normalized_updated_source_ids = {
        str(source_id or "").strip()
        for source_id in set(updated_source_ids or set())
        if str(source_id or "").strip()
    }
    normalized_updated_field_names = {
        str(field_name or "").strip()
        for field_name in set(updated_field_names or set())
        if str(field_name or "").strip()
    }
    background_rebuild_notice: dict[str, Any] | None = None
    deleted_source_ids_for_field_cleanup = set(deleted_id_set)
    updated_field_record_delta = {
        "unchanged_fields": set(),
        "created_fields": set(),
        "updated_fields": set(),
        "deleted_fields": set(),
        "fields_to_delete": [],
        "fields_to_upsert": [],
        "removed_field_record_ids": [],
    }
    updated_field_record_entries_for_vector: list[dict[str, Any]] = []

    # 先把本次主数据变化精确落到 SQLite，再同步受影响主记忆家族的 field_record。
    with closing(get_sqlite_connection()) as connection:
        initialize_sqlite_schema(connection)
        if deleted_id_set:
            delete_store_entries_by_ids(list(deleted_id_set), connection=connection)
            delete_field_records_by_source_ids(list(deleted_source_ids_for_field_cleanup), connection=connection)
        if normalized_appended_entry is not None:
            upsert_store_entries([normalized_appended_entry], connection=connection)
        if normalized_updated_entry is not None:
            upsert_store_entries([normalized_updated_entry], connection=connection)
        # 这里直接基于当前请求里已经确定好的最新主记忆对象派生 field_record。
        # 不再在事务提交前另开 SQLite 连接回读主记忆，避免读到旧版本或读不到新插入记录。
        refreshed_field_entries: list[dict[str, Any]] = []
        if normalized_appended_entry is not None and not is_field_record_entry(normalized_appended_entry):
            refreshed_field_entries.extend(build_field_record_entries(normalized_appended_entry))
        if (
            normalized_updated_entry is not None
            and normalized_previous_entry is not None
            and not is_field_record_entry(normalized_updated_entry)
        ):
            updated_field_record_delta = plan_field_record_delta(
                previous_entry=normalized_previous_entry,
                updated_entry=normalized_updated_entry,
                touched_field_names=normalized_updated_field_names,
            )
            updated_source_memory_id = str(normalized_updated_entry.get("id") or "").strip()
            if updated_field_record_delta["fields_to_delete"]:
                delete_field_records_by_source_and_fields(
                    updated_source_memory_id,
                    updated_field_record_delta["fields_to_delete"],
                    connection=connection,
                )
            for field_name in updated_field_record_delta["fields_to_upsert"]:
                saved_field_entry = save_fieldRecord(
                    normalized_updated_entry,
                    field_name,
                    connection=connection,
                )
                if saved_field_entry is not None:
                    updated_field_record_entries_for_vector.append(saved_field_entry)
        elif normalized_updated_entry is not None and not is_field_record_entry(normalized_updated_entry):
            if normalized_updated_source_ids:
                delete_field_records_by_source_ids(list(normalized_updated_source_ids), connection=connection)
            refreshed_field_entries.extend(build_field_record_entries(normalized_updated_entry))
            updated_field_record_entries_for_vector = list(refreshed_field_entries)
        if refreshed_field_entries:
            upsert_store_entries(refreshed_field_entries, connection=connection)
        connection.commit()

    current_entry_ids = fetch_ordered_store_entry_ids()
    public_timeline_entries = fetch_public_timeline_summary_entries(limit=3)

    # 写盘：主数据已经先落 SQLite；这里同步重写人类可读时间线。
    write_text_atomic(TIMELINE_PATH, build_timeline_markdown_from_batches(iter_public_entries_for_timeline()))
    # 获取模型目录
    resolved_model_path = model_path or resolve_embed_model_path()
    current_embedding_model_fingerprint = build_embedding_model_fingerprint(resolved_model_path)

    # 开始更新后续的 .npy / .faiss / meta 文件。
    if not current_entry_ids:  # 如果主数据列表为空，则清空embedding 缓存、FAISS 和 meta
        # 下面传入空 entries 和空矩阵，只是为了复用 rebuild_index_from_embeddings 里的清理逻辑，
        rebuild_stats = rebuild_index_from_embeddings(
            [],
            np.empty((0, 0), dtype="float32"),
            resolved_model_path,
            current_embedding_model_fingerprint
        )
        write_rebuild_state(
            build_rebuild_state_snapshot(
                state="idle",
                finished_at=now_iso(),
                target_entry_count=0,
                worker_pid=os.getpid(),
            )
        )
    else:  
        # 非空，先 load_reusable_embedding_cache() 尝试读取现有可复用的 embedding 缓存
        # 该函数为双返回值，所以 cached_embedding_state 命中时，里面同时包含：
        #   1. 已经算好的向量矩阵（numpy）
        #   2. 向量矩阵每一行对应的记录 id 顺序（row-to-id）
        # 后续 新增 / 删除 分支都会优先复用这份缓存。
        cached_embedding_state = load_reusable_embedding_cache(current_embedding_model_fingerprint)

        # save 的增量路径（当新增记录时执行的代码）：
        #   1. 只对新增记录生成 retrieval_text 并做一次 embedding
        #   2. 把旧缓存里的 id -> 向量映射 和这条新向量合并起来
        #   3. 按当前 sorted_entries 的完整顺序重新拼成整库矩阵
        #   4. 再统一重写 .npy、.faiss 和 meta
        # 这样减少了“重复 embedding 旧记录”，同时仍然保证行顺序和 row_to_id 完全一致。
        if normalized_appended_entry is not None and not deleted_id_set and cached_embedding_state is not None:  # 这次是新增一条记录、不是删除、并且缓存命中
            # 赋值 numpy 和 id 数组
            cached_embeddings, cached_row_ids = cached_embedding_state
            # 把 “旧矩阵第几行对应哪个 id” 转换成 id -> 向量 的字典。
            # 这样后面就能直接按记录 id 取旧向量，用来补进新增记录并按最新整库顺序重拼矩阵，
            cached_vector_by_id = {
                row_id: cached_embeddings[index]  # cached_embeddings[index] 可以理解成“矩阵中的第 index 行”
                for index, row_id in enumerate(cached_row_ids)
            }
            # current_entry_ids：“当前最新的，排序后的整库” 下的 id 列表。
            # 后面重拼矩阵、重写 meta、重建 FAISS，都会以这份顺序为准。
            new_entry_ids = [
                entry_id
                for entry_id in current_entry_ids
                if entry_id and entry_id not in cached_vector_by_id
            ]
            new_entry_id_set = set(new_entry_ids)
            # 只有在下面这些条件同时成立时，才允许走“只新增若干条新向量”的快路径：
            #   1. 至少识别出一批当前缓存里还没有的新 id；
            #   2. 新库条数 = 旧缓存条数 + 这批新 id 的数量；
            #   3. 当前整库里的每个 id，要么能在旧缓存里找到旧向量，要么属于这批新 id。
            # 只要有任意一个条件不成立，就退回全量重建，避免留下错位索引。
            if (
                new_entry_ids
                and len(cached_row_ids) + len(new_entry_ids) == len(current_entry_ids)
                and all(entry_id and (entry_id in cached_vector_by_id or entry_id in new_entry_id_set) for entry_id in current_entry_ids)
            ):
                # 调用 embedding 模型对新增的这一条记录做编码
                resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
                new_entries = fetch_store_entries_by_ids(new_entry_ids)
                new_entry_payloads = build_indexable_entry_payloads(new_entries)
                if len(new_entry_payloads) != len(new_entry_ids):
                    background_rebuild_notice = launch_background_full_rebuild(
                        resolved_model_path,
                        len(current_entry_ids),
                    )
                    rebuild_stats = load_current_index_stats()
                else:
                    new_embeddings = encode_texts(
                        [retrieval_text for _, _, retrieval_text in new_entry_payloads],
                        resolved_embedder,
                    )
                    # 把新记录向量补进 "id -> 向量" 字典后，旧记录和新记录的向量就都齐了。
                    for index, (entry_id, _, _) in enumerate(new_entry_payloads):
                        cached_vector_by_id[entry_id] = new_embeddings[index]
                    # 按“当前最新整库顺序”从字典里把向量重新取出来，拼成一份新的完整矩阵。
                    # 这里重拼的是矩阵顺序，不是重新计算旧记录的 embedding。
                    rebuilt_embeddings = np.vstack(
                        [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                    ).astype("float32")
                    # 用这份新矩阵统一重写 .npy、.faiss 和 meta。
                    rebuild_stats = rebuild_index_from_embeddings(
                        current_entry_ids,
                        rebuilt_embeddings,
                        resolved_model_path,
                        current_embedding_model_fingerprint
                    )
            else:
                # 只要新增快路径的前提不再安全，就回退到完整重建，
                # 但这轮不再把全量 embedding 堵在前台请求里，而是改成后台慢任务继续补齐索引。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(current_entry_ids),
                )
                rebuild_stats = load_current_index_stats()
        # 删除链路的缓存复用路径：
        # 这里不会重新调用 embedding 模型。
        # 仅仅把旧缓存中的 id -> 向量映射留下来，再按“删除后的当前记录顺序”重拼矩阵，
        # 最后用这份剩余矩阵重写 .npy、.faiss 和 meta。
        # 本质上是“删结构，不重算内容”。
        elif deleted_id_set and cached_embedding_state is not None:
            cached_embeddings, cached_row_ids = cached_embedding_state
            # 删除链路也先整理成 id -> 向量 的字典。
            # 这样后面只要拿“删除后的剩余 id 顺序”去取向量，就能直接重拼新矩阵，
            # 不需要重算 embedding，也不需要依赖旧矩阵原来的行号布局。
            cached_vector_by_id = {
                row_id: cached_embeddings[index]
                for index, row_id in enumerate(cached_row_ids)
            }
            # current_entry_ids 代表“删除完成后，当前整库还剩哪些记录，以及它们现在的顺序”。
            # 删除快路径的安全条件是：
            # 1. 删除后还保留、且仍应在索引里的每个 id，都必须能在旧缓存里找到。
            # 已删除但本来就不在索引里的空文本记录，不应该逼迫整库退回全量重建。
            for deleted_id in deleted_id_set:
                cached_vector_by_id.pop(deleted_id, None)
            if all(
                entry_id and entry_id in cached_vector_by_id for entry_id in current_entry_ids
            ):
                # 直接按删除后的最新顺序把剩余向量重拼成新矩阵。
                rebuilt_embeddings = np.vstack(
                    [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                ).astype("float32")
                # 用这份剩余矩阵重写 .npy、.faiss 和 meta。
                rebuild_stats = rebuild_index_from_embeddings(
                    current_entry_ids,
                    rebuilt_embeddings,
                    resolved_model_path,
                    current_embedding_model_fingerprint
                )
            else:
                # 如果旧缓存和当前删除结果对不上，就退回全量重建，
                # 但这轮不再把全量 embedding 堵在前台请求里，而是改成后台慢任务继续补齐索引。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(current_entry_ids),
                )
                rebuild_stats = load_current_index_stats()
        # update 的精确增量路径：
        # 这里仍然直接替换主记忆本体，但 field_record 不再整家重算，
        # 而是只删除受影响的旧子记录，并只重算新增或改值的那几个字段子记录。
        elif normalized_updated_entry is not None and cached_embedding_state is not None:
            cached_embeddings, cached_row_ids = cached_embedding_state
            cached_vector_by_id = {
                row_id: cached_embeddings[index]
                for index, row_id in enumerate(cached_row_ids)
            }
            removed_field_record_ids = [
                str(record_id or "").strip()
                for record_id in updated_field_record_delta["removed_field_record_ids"]
                if str(record_id or "").strip()
            ]
            reencoded_entries = [normalized_updated_entry, *updated_field_record_entries_for_vector]
            reencoded_entry_payloads = build_indexable_entry_payloads(reencoded_entries)
            reencoded_entry_id_set = {entry_id for entry_id, _, _ in reencoded_entry_payloads}
            for removed_id in removed_field_record_ids:
                cached_vector_by_id.pop(removed_id, None)
            if all(
                entry_id and (entry_id in cached_vector_by_id or entry_id in reencoded_entry_id_set)
                for entry_id in current_entry_ids
            ):
                if reencoded_entry_payloads:
                    resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
                    updated_embeddings = encode_texts(
                        [retrieval_text for _, _, retrieval_text in reencoded_entry_payloads],
                        resolved_embedder,
                    )
                    for index, (entry_id, _, _) in enumerate(reencoded_entry_payloads):
                        cached_vector_by_id[entry_id] = updated_embeddings[index]
                rebuilt_embeddings = np.vstack(
                    [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                ).astype("float32")
                rebuild_stats = rebuild_index_from_embeddings(
                    current_entry_ids,
                    rebuilt_embeddings,
                    resolved_model_path,
                    current_embedding_model_fingerprint
                )
            else:
                # 只要当前缓存无法证明“除了这次更新家族外，其余记录都还能直接复用旧向量”，
                # 就改成后台全量重建，避免把长时间 embedding 堵在当前工具调用里。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(current_entry_ids),
                )
                rebuild_stats = load_current_index_stats()
        else:
            # 这里是完整后门：
            # 首次建库、缓存不存在、缓存和当前模型/字段规则不一致、或新增/删除条件无法安全命中时，
            # 都改成后台慢任务去做“重新为全部记录生成 retrieval_text 并全量 embedding”的保守路线。
            background_rebuild_notice = launch_background_full_rebuild(
                resolved_model_path,
                len(current_entry_ids),
            )
            rebuild_stats = load_current_index_stats()

        if background_rebuild_notice is None:
            write_rebuild_state(
                build_rebuild_state_snapshot(
                    state="idle",
                    finished_at=now_iso(),
                    target_entry_count=len(current_entry_ids),
                    worker_pid=os.getpid(),
                )
            )

    # 最后返回给 save_record / delete_records_by_ids 一个简短摘要，
    # 让上层函数知道：当前整库一共有多少条、时间线前几项是什么、索引维度和条数是多少。
    # 这里不返回完整主数据，而只返回上层真正关心的结果摘要。
    timeline_summary = [f"{entry['updated_at']} | {entry['title']}" for entry in public_timeline_entries]
    resolved_return_entries: list[dict[str, Any]] = []
    if normalized_appended_entry is not None or normalized_updated_entry is not None:
        resolved_return_entry_ids = [
            str(entry.get("id") or "").strip()
            for entry in [normalized_appended_entry, normalized_updated_entry]
            if entry is not None and str(entry.get("id") or "").strip()
        ]
        if resolved_return_entry_ids:
            # save/update 只把本次真正触达的主记忆按 id 回源回来，避免为了返回摘要再把整库主数据搬进内存。
            resolved_return_entries = fetch_store_entries_by_ids(resolved_return_entry_ids)
    result = {
        "resolved_entries": resolved_return_entries,
        "total_entries": count_public_store_entries(),
        "timeline_summary": timeline_summary,
        "index_stats": rebuild_stats,
    }
    if background_rebuild_notice is not None:
        result.update(background_rebuild_notice)
    return result
