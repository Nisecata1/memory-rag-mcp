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
import sys
import time
import threading
from datetime import datetime
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
    store_path = resolve_configured_path(str(paths_config.get("store_path") or ""))
    timeline_path = resolve_configured_path(str(paths_config.get("timeline_path") or ""))
    embeddings_path = resolve_configured_path(str(paths_config.get("embeddings_path") or ""))
    index_path = resolve_configured_path(str(paths_config.get("index_path") or ""))
    meta_path = resolve_configured_path(str(paths_config.get("meta_path") or ""))

    embed_model_path = resolve_configured_path(str(embedding_config.get("model_path") or ""))
    embed_device = str(embedding_config.get("device") or "").strip()
    if not embed_device:
        raise RuntimeError(f"embedding.device in {CONFIG_PATH.name} must not be empty.")

    # 第五阶段：解析主数据结构、摘要策略和搜索限制这类业务规则配置。
    store_version = int(memory_config.get("store_version") or 0)
    if store_version <= 0:
        raise RuntimeError(f"memory.store_version in {CONFIG_PATH.name} must be a positive integer.")

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
    debug_fields = results_config.get("debug_fields")
    if not isinstance(search_fields, list) or not search_fields:
        raise RuntimeError(f"results.search_fields in {CONFIG_PATH.name} must be a non-empty list.")
    if not isinstance(detail_fields, list) or not detail_fields:
        raise RuntimeError(f"results.detail_fields in {CONFIG_PATH.name} must be a non-empty list.")
    if not isinstance(debug_fields, list):
        raise RuntimeError(f"results.debug_fields in {CONFIG_PATH.name} must be a list.")

    normalized_search_fields = tuple(str(item or "").strip() for item in search_fields if str(item or "").strip())
    normalized_detail_fields = tuple(str(item or "").strip() for item in detail_fields if str(item or "").strip())
    normalized_debug_fields = tuple(str(item or "").strip() for item in debug_fields if str(item or "").strip())
    if not normalized_search_fields or not normalized_detail_fields:
        raise RuntimeError(f"results section in {CONFIG_PATH.name} must contain valid field names.")

    return {
        "server_name": server_name,
        "ai_memory_dir": ai_memory_dir,
        "data_dir": data_dir,
        "store_path": store_path,
        "timeline_path": timeline_path,
        "embeddings_path": embeddings_path,
        "index_path": index_path,
        "meta_path": meta_path,
        "embed_model_path": embed_model_path,
        "embed_device": embed_device,
        "store_version": store_version,
        "memory_kind_values": set(normalized_kind_values),
        "tag_fallback_limit": tag_fallback_limit,
        "short_summary_max_length": short_summary_max_length,
        "max_top_k": max_top_k,
        "retrieval_field_candidates": normalized_field_candidates,
        "field_labels": normalized_field_labels,
        "search_result_fields": normalized_search_fields,
        "detail_result_fields": normalized_detail_fields,
        "debug_result_fields": normalized_debug_fields,
    }


# 模块导入时先生成一次项目级配置快照。
# 后面这整段模块级常量，都是从这份快照里展开出来的，以供后续业务代码使用。
PROJECT_CONFIG = load_project_config()  # 项目级配置快照。

# 这一组是路径与运行时基础常量，后续主数据、时间线、模型和索引链路都会直接依赖它们。
SERVER_NAME = PROJECT_CONFIG["server_name"]  # MCP 服务名称。
AI_MEMORY_DIR = PROJECT_CONFIG["ai_memory_dir"]  # 共享的 AI-memory 根目录。
DATA_DIR = PROJECT_CONFIG["data_dir"]  # 项目记忆数据目录。
STORE_PATH = PROJECT_CONFIG["store_path"]  # 主数据 JSON 文件。
TIMELINE_PATH = PROJECT_CONFIG["timeline_path"]  # 人类可读的时间线文件。
EMBEDDINGS_PATH = PROJECT_CONFIG["embeddings_path"]  # 已计算向量的本地缓存矩阵文件。
INDEX_PATH = PROJECT_CONFIG["index_path"]  # FAISS 向量索引文件。
META_PATH = PROJECT_CONFIG["meta_path"]  # 向量行号到记录 id 的映射文件。
REBUILD_STATE_PATH = DATA_DIR / "rebuild_state.json"  # 后台全量重建状态文件。
EMBED_MODEL_PATH = PROJECT_CONFIG["embed_model_path"]  # YAML 配置中的本地嵌入模型目录。
EMBED_DEVICE = PROJECT_CONFIG["embed_device"]  # YAML 配置中的嵌入设备。

# 这一组是业务规则与返回裁剪常量，负责约束搜索上限、检索字段以及接口默认返回字段。
MAX_TOP_K = PROJECT_CONFIG["max_top_k"]  # 搜索结果最大返回条数。
STORE_VERSION = PROJECT_CONFIG["store_version"]  # 当前主数据结构版本号。
TAG_FALLBACK_LIMIT = PROJECT_CONFIG["tag_fallback_limit"]  # 服务端兜底标签的最大数量。
SHORT_SUMMARY_MAX_LENGTH = PROJECT_CONFIG["short_summary_max_length"]  # 自动生成简短摘要时的最大字符数。
RETRIEVAL_FIELD_CANDIDATES = PROJECT_CONFIG["retrieval_field_candidates"]  # 统一的检索文本候选字段顺序。
RETRIEVAL_FIELD_SIGNATURE = "|".join(RETRIEVAL_FIELD_CANDIDATES)  # 当前检索字段规则签名，用来判断旧 embedding 是否失效。
FIELD_LABELS = PROJECT_CONFIG["field_labels"]  # 检索文本拼接时使用的人类可读字段标签。
MEMORY_KIND_VALUES = PROJECT_CONFIG["memory_kind_values"]  # 允许的记忆类型枚举值。
SEARCH_RESULT_FIELDS = PROJECT_CONFIG["search_result_fields"]  # search 默认返回的轻量字段。
DETAIL_RESULT_FIELDS = PROJECT_CONFIG["detail_result_fields"]  # 详情接口默认返回的完整业务字段。
DEBUG_RESULT_FIELDS = PROJECT_CONFIG["debug_result_fields"]  # 仅在调试模式下返回的内部字段。

UPDATE_ALLOWED_FIELDS = (
    "memory_kind",
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
)  # update 补丁允许修改的业务字段。
UPDATE_SYSTEM_FIELDS = (
    "id",
    "fingerprint",
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
)  # 对外工具允许创建或更新到的公共记忆类型。
INTERNAL_MEMORY_KIND_VALUES = ("field_record",)  # 只允许服务端内部派生的字段级记忆类型。
LEGACY_PUBLIC_MEMORY_KIND_ALIASES = {
    "chat_fragment": "chat_event",
    "knowledge_record": "fact",
}  # 旧主数据里的公共类型别名；只在读兼容链路里翻译成当前口径，不再接受新的工具输入继续写旧值。
FIELD_RECORD_CANDIDATE_FIELDS = (
    "title",
    "short_summary",
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
MATCHED_FIELDS_LIMIT = 3  # search 返回的 matched_fields 最多保留 3 个，避免结果对象膨胀。
BACKGROUND_REBUILD_MODE = "background_full_rebuild"  # 后台全量重建的固定模式名。
_BACKGROUND_REBUILD_LOCK = threading.Lock()  # 同一进程内的后台重建串行锁。
_BACKGROUND_REBUILD_THREAD: threading.Thread | None = None  # 当前进程里正在运行的后台重建线程引用。

# 类型声明上这是 Literal[...]，也就是“只允许固定几个字符串字面量”的类型；运行时拿到的数据类型仍然是 str，例如 "fact"。
# 对外 save / update 这类工具里的 memory_kind 只能传 "project_record"、"chat_event" 或 "fact"；最终进代码时就是这三个字符串之一，不是别的对象类型。
MemoryKind = Literal["project_record", "chat_event", "fact"]

# 类型声明上这是 Literal[...]，表示主数据实际允许保存的记忆类型全集；运行时仍然是 str，例如 "field_record"。
# 它和上面的 MemoryKind 不同：MemoryKind 面向工具层输入，StoredMemoryKind 面向主数据与内部派生记录，所以额外包含内部类型 "field_record"。
StoredMemoryKind = Literal["project_record", "chat_event", "fact", "field_record"]

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


# 把自由文本压缩成单行紧凑字符串，避免空白符干扰去重和存储。
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
# 默认只接受当前正式类型；只有旧主数据读兼容链路才允许把 chat_fragment / knowledge_record 翻译成 chat_event / fact。
def normalize_memory_kind(value: str | None, allow_legacy_aliases: bool = False) -> StoredMemoryKind:
    normalized = collapse_text(value)
    if allow_legacy_aliases and normalized in LEGACY_PUBLIC_MEMORY_KIND_ALIASES:
        normalized = LEGACY_PUBLIC_MEMORY_KIND_ALIASES[normalized]
    if normalized not in MEMORY_KIND_VALUES:
        raise ValueError(
            "memory_kind must be one of 'project_record', 'chat_event', 'fact' or 'field_record'"
        )
    return normalized  # type: ignore[return-value]


# 判断一条主数据记录是不是内部字段级记忆；调用方包括 search、timeline、save/update/delete 的边界校验。
# 输入是主数据中的单条记录 dict；输出是 bool，True 表示它的 memory_kind 是内部类型 field_record。
def is_field_record_entry(entry: dict[str, Any]) -> bool:
    return str(entry.get("memory_kind") or "").strip() == "field_record"


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


# 从现有 field_record 中提取“真正属于 source_field_name 的值”，兼容旧的膨胀型 field_record 和新的 one-hot 形态。
# 新 one-hot 记录会直接取对应字段；旧膨胀记录则优先从 detailed_summary / short_summary 中恢复原字段值。
def resolve_field_record_value(entry: dict[str, Any], source_field_name: str) -> str | list[str]:
    if source_field_name == "tags":
        for candidate in (entry.get("detailed_summary"), entry.get("short_summary"), entry.get("tags")):
            normalized_value = normalize_field_record_field_value(source_field_name, candidate)
            if normalized_value:
                return normalized_value
        return []

    if source_field_name == "reference_doc_path":
        for candidate in (
            entry.get("reference_doc_path"),
            entry.get("detailed_summary"),
            entry.get("short_summary"),
            entry.get("title"),
        ):
            normalized_value = normalize_field_record_field_value(source_field_name, candidate)
            if normalized_value:
                return normalized_value
        return ""

    for candidate in (
        entry.get(source_field_name),
        entry.get("detailed_summary"),
        entry.get("short_summary"),
        entry.get("title"),
    ):
        normalized_value = normalize_field_record_field_value(source_field_name, candidate)
        if normalized_value:
            return normalized_value
    return ""


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
    }
    payload[source_field_name] = list(field_value) if isinstance(field_value, list) else field_value
    field_fingerprint = payload_fingerprint(payload)
    return {
        "id": build_field_record_id(source_memory_id, source_field_name, field_fingerprint),
        "fingerprint": field_fingerprint,
        "created_at": created_at,
        "updated_at": updated_at,
        "source_memory_id": source_memory_id,
        "source_field_name": source_field_name,
        **payload,
    }


# 为旧数据生成详细总结兜底值，避免升级后因为缺字段导致记录失效。
def build_legacy_detailed_summary(
    title: str,
    problem_background: str,
    analysis: str,
    action_steps: str,
    validation_result: str,
    legacy_memory_text: str | None,
) -> str:
    resolved_legacy_text = normalize_text_block(legacy_memory_text)
    if resolved_legacy_text:
        return resolved_legacy_text

    parts = []
    if title:
        parts.append(f"标题：{title}")
    if problem_background:
        parts.append(f"问题背景：{problem_background}")
    if analysis:
        parts.append(f"分析：{analysis}")
    if action_steps:
        parts.append(f"步骤：{action_steps}")
    if validation_result:
        parts.append(f"验证结果：{validation_result}")
    return "\n".join(parts).strip()


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
    if memory_kind == "fact":
        field_candidates = FACT_RETRIEVAL_FIELD_CANDIDATES
    elif memory_kind == "field_record":
        source_field_name = str(entry.get("source_field_name") or "").strip()
        if source_field_name not in FIELD_RECORD_CANDIDATE_FIELDS:
            return []
        field_value = resolve_field_record_value(entry, source_field_name)
        return [source_field_name] if field_value else []
    else:
        field_candidates = RETRIEVAL_FIELD_CANDIDATES

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


# 根据结果或详细总结生成适合时间线显示的单行预览，避免 Markdown 视图过长。
def preview_text(value: str | None, limit: int = 160) -> str:
    single_line = collapse_text(value)
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit].rstrip()}..."


# 生成标准化后的记录载荷，作为去重和持久化的统一输入。
def build_payload(
    memory_kind: str,
    title: str,
    detailed_summary: str,
    short_summary: str | None = None,
    problem_background: str | None = None,
    analysis: str | None = None,
    action_steps: str | None = None,
    validation_result: str | None = None,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    strict_project_requirements: bool = True,
    require_short_summary: bool = True,
    validate_reference_doc_path: bool = True,
    allow_legacy_kind_aliases: bool = False,
) -> dict[str, Any]:
    resolved_memory_kind = normalize_memory_kind(
        memory_kind,
        allow_legacy_aliases=allow_legacy_kind_aliases,
    )
    resolved_title = collapse_text(title)
    resolved_problem_background = normalize_text_block(problem_background)
    resolved_analysis = normalize_text_block(analysis)
    resolved_action_steps = normalize_text_block(action_steps)
    resolved_validation_result = normalize_text_block(validation_result)
    resolved_detailed_summary = normalize_text_block(detailed_summary)
    resolved_source_paths = normalize_list(coerce_source_path_list_input(source_paths))
    resolved_reference_doc_path = resolve_reference_doc_path(
        reference_doc_path,
        require_exists=validate_reference_doc_path,
    )
    resolved_tags = normalize_list(coerce_tag_list_input(tags))

    if not resolved_title:
        raise ValueError("title is required")
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

    resolved_short_summary = build_short_summary(
        title=resolved_title,
        detailed_summary=resolved_detailed_summary,
        short_summary=short_summary,
    )

    payload = {
        "memory_kind": resolved_memory_kind,
        "title": resolved_title,
        "short_summary": resolved_short_summary,
        "problem_background": resolved_problem_background,
        "analysis": resolved_analysis,
        "action_steps": resolved_action_steps,
        "validation_result": resolved_validation_result,
        "detailed_summary": resolved_detailed_summary,
        "tags": resolved_tags,
        "source_paths": resolved_source_paths,
        "reference_doc_path": resolved_reference_doc_path,
    }
    payload["retrieval_fields"] = build_retrieval_fields(payload)
    return payload


# 对标准载荷做哈希，便于安全地判断是否重复保存。
def payload_fingerprint(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


# 基于指纹生成稳定的记录 id。
def payload_id(fingerprint: str) -> str:
    return f"pmem-{fingerprint[:12]}"


# 为内部 field_record 生成稳定 id；它属于服务端派生规则，不直接暴露给 save/update 调用方。
# 输入包括源头主记忆 id、原字段名和字段级 payload 指纹；输出是稳定的 str 类型内部记录 id，例如 fmem-xxxx。
def build_field_record_id(source_memory_id: str, source_field_name: str, fingerprint: str) -> str:
    id_seed = f"{source_memory_id}|{source_field_name}|{fingerprint}"
    return f"fmem-{hashlib.sha256(id_seed.encode('utf-8')).hexdigest()[:12]}"


# 把主数据中的单条记录迁移成当前正式 schema。
# 它会兼容旧字段名和旧时间字段（如 problem/action/result、memory_text、saved_at/date），
# 重新走 build_payload 统一正文与检索字段，并补齐或重算 id、fingerprint、created_at、updated_at。
# 返回值只是“规范化后的单条记录对象”，不负责写回主数据文件。
def normalize_store_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(f"Every entry in {STORE_PATH.name} must be an object")

    raw_title = entry.get("title")
    raw_short_summary = entry.get("short_summary")
    raw_problem_background = entry.get("problem_background")
    if raw_problem_background is None:
        raw_problem_background = entry.get("problem")
    raw_analysis = entry.get("analysis")
    raw_action_steps = entry.get("action_steps")
    if raw_action_steps is None:
        raw_action_steps = entry.get("action")
    raw_validation_result = entry.get("validation_result")
    if raw_validation_result is None:
        raw_validation_result = entry.get("result")
    raw_memory_kind = entry.get("memory_kind") or "project_record"
    raw_detailed_summary = entry.get("detailed_summary")
    raw_legacy_memory_text = entry.get("memory_text")
    raw_source_paths = entry.get("source_paths")
    raw_reference_doc_path = entry.get("reference_doc_path")
    raw_created_at = entry.get("created_at")
    raw_updated_at = entry.get("updated_at")
    raw_saved_at = entry.get("saved_at")
    raw_legacy_date = entry.get("date")
    raw_source_memory_id = entry.get("source_memory_id")
    raw_source_field_name = entry.get("source_field_name")
    normalized_created_at = resolve_record_timestamp(
        str(raw_created_at or ""),
        fallback=str(raw_saved_at or raw_legacy_date or ""),
    )
    normalized_updated_at = resolve_record_timestamp(
        str(raw_updated_at or ""),
        fallback=str(raw_saved_at or raw_legacy_date or ""),
    )

    resolved_memory_kind = normalize_memory_kind(
        str(raw_memory_kind),
        allow_legacy_aliases=True,
    )
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
        detailed_summary=build_legacy_detailed_summary(
            title=collapse_text(raw_title),
            problem_background=normalize_text_block(raw_problem_background),
            analysis=normalize_text_block(raw_analysis),
            action_steps=normalize_text_block(raw_action_steps),
            validation_result=normalize_text_block(raw_validation_result),
            legacy_memory_text=str(raw_detailed_summary or raw_legacy_memory_text or ""),
        ),
        short_summary=str(raw_short_summary or ""),
        problem_background=str(raw_problem_background or ""),
        analysis=str(raw_analysis or ""),
        action_steps=str(raw_action_steps or ""),
        validation_result=str(raw_validation_result or ""),
        tags=entry.get("tags"),
        source_paths=raw_source_paths,
        reference_doc_path=str(raw_reference_doc_path or ""),
        strict_project_requirements=False,
        require_short_summary=False,
        validate_reference_doc_path=False,
        allow_legacy_kind_aliases=True,
    )

    normalized_fingerprint = payload_fingerprint(payload)
    normalized_id = str(entry.get("id") or "").strip() or payload_id(normalized_fingerprint)

    normalized_entry = {
        "id": normalized_id,
        "fingerprint": normalized_fingerprint,
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
        if field_name in UPDATE_SYSTEM_FIELDS:
            raise ValueError(f"changes must not contain system field '{field_name}'")
        if field_name not in UPDATE_ALLOWED_FIELDS:
            raise ValueError(f"changes contains unsupported field '{field_name}'")
        if field_name in UPDATE_LIST_FIELDS and raw_field_value is None:
            raise ValueError(f"changes.{field_name} must use [] to clear the list")
        if raw_field_value is None and field_name not in UPDATE_NULL_CLEARABLE_FIELDS:
            raise ValueError(f"changes.{field_name} does not support null")
        if field_name == "memory_kind" and collapse_text(raw_field_value) == "field_record":
            raise ValueError("changes.memory_kind must not use internal type 'field_record'")
        normalized_changes[field_name] = raw_field_value

    if not normalized_changes:
        raise ValueError("changes must not be empty")
    return normalized_changes


# 用当前记录和已校验的 update 补丁合成一条候选新记录；上层函数 update_record 保留它，是为了让入口层继续看得见“补丁 -> 重建”主链路。
def build_update_candidate_record(source_entry: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    payload_inputs: dict[str, Any] = {}
    for field_name in UPDATE_ALLOWED_FIELDS:
        if field_name in changes:
            payload_inputs[field_name] = changes[field_name]
        else:
            payload_inputs[field_name] = source_entry.get(field_name)

    payload = build_payload(
        memory_kind=str(payload_inputs["memory_kind"] or ""),
        title=str(payload_inputs["title"] or ""),
        detailed_summary=str(payload_inputs["detailed_summary"] or ""),
        short_summary=payload_inputs.get("short_summary"),
        problem_background=payload_inputs.get("problem_background"),
        analysis=payload_inputs.get("analysis"),
        action_steps=payload_inputs.get("action_steps"),
        validation_result=payload_inputs.get("validation_result"),
        tags=payload_inputs.get("tags"),
        source_paths=payload_inputs.get("source_paths"),
        reference_doc_path=payload_inputs.get("reference_doc_path"),
    )
    normalized_fingerprint = payload_fingerprint(payload)

    return {
        "id": str(source_entry.get("id") or "").strip(),
        "fingerprint": normalized_fingerprint,
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
    for field_name in FIELD_RECORD_CANDIDATE_FIELDS:
        field_value = normalize_field_record_field_value(field_name, source_entry.get(field_name))
        if not field_value:
            continue
        field_entries.append(
            build_one_hot_field_record_entry(
                source_memory_id=source_memory_id,
                source_field_name=field_name,
                field_value=field_value,
                created_at=str(source_entry.get("created_at") or "").strip(),
                updated_at=str(source_entry.get("updated_at") or "").strip(),
            )
        )
    return field_entries


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


# 读取主数据 JSON；如果是首次使用，则返回空结构，并兼容旧版本记录。
def load_store() -> dict[str, Any]:
    ensure_memory_dir()
    if not STORE_PATH.exists():
        return {"version": STORE_VERSION, "entries": []}

    with STORE_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"{STORE_PATH.name} must contain an object")

    entries = data.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(f"{STORE_PATH.name} must contain an entries list")

    normalized_entries = [normalize_store_entry(entry) for entry in entries]
    return {
        "version": STORE_VERSION,
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
        "store_version_seen": STORE_VERSION,
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
    row_to_id = current_meta.get("row_to_id") or []
    if isinstance(row_to_id, list):
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


# 在当前 MCP 进程里后台执行一次完整索引重建；它只读最新 memory.json，并把索引侧文件补齐到一致状态。
def run_background_full_rebuild(started_at: str, model_path: str, target_entry_count: int) -> None:
    try:
        loaded_store = load_store()
        entries = list(loaded_store.get("entries") or [])
        rebuild_stats = rebuild_vector_index(entries, model_path=model_path)
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
        f"> 主数据源：{STORE_PATH}",
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


# 清洗详情接口传入的 ids，去掉空白项、按首次出现顺序去重，并保证最终列表非空。
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


# 这个函数就是根据传入的 entries 写进 meta 文件的
# entries 参数是上游已经排好序、并且与 embedding 矩阵行顺序一一对应的完整记录列表。
# update_store_and_rebuild / rebuild_vector_index 会先确定当前整库记录顺
# 再由 rebuild_index_from_embeddings 把这份顺序传进来。
# 这个函数只负责把“当前索引第几行对应哪个记录 id，以及索引使用了哪个模型和维度”写进 meta 文件。
def write_vector_meta(entries: list[dict[str, Any]], model_path: str, dim: int) -> None:
    meta = {
        "model": model_path,
        "dim": int(dim),
        "normalized": True,
        "retrieval_field_signature": RETRIEVAL_FIELD_SIGNATURE,
        "row_to_id": [
            {"id": entry["id"]}
            for entry in entries
        ],
    }
    write_json_atomic(META_PATH, meta)


# 校验并读取当前仍可复用的本地向量缓存和 meta 索引
# 上层函数：update_store_and_rebuild 会在 save / delete 等快路径里优先走这里，判断旧缓存还能不能继续复用。
def load_reusable_embedding_cache(model_path: str) -> tuple[np.ndarray, list[str]] | None:
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
    if str(meta.get("model") or "") != model_path:
        return None
    if str(meta.get("retrieval_field_signature") or "") != RETRIEVAL_FIELD_SIGNATURE:
        return None

    row_to_id = meta.get("row_to_id")
    if not isinstance(row_to_id, list):
        return None

    normalized_embeddings = np.asarray(embeddings, dtype="float32")
    if normalized_embeddings.shape[0] != len(row_to_id):
        return None

    dim = int(meta.get("dim") or 0)
    if not dim or normalized_embeddings.shape[1] != dim:
        return None

    row_ids: list[str] = []
    for row_item in row_to_id:
        if not isinstance(row_item, dict):
            return None
        row_id = str(row_item.get("id") or "").strip()
        if not row_id:
            return None
        row_ids.append(row_id)

    return normalized_embeddings, row_ids


# update_store_and_rebuild、rebuild_vector_index 和增量缓存路径都通过这里把向量缓存与索引产物同步写盘。
def rebuild_index_from_embeddings(
    entries: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_path: str,
) -> dict[str, Any]:
    if not entries:
        for artifact_path in (EMBEDDINGS_PATH, INDEX_PATH, META_PATH):
            if artifact_path.exists():
                artifact_path.unlink()
        return {"count": 0, "dim": 0}

    normalized_embeddings = l2_normalize(np.asarray(embeddings, dtype="float32")).astype("float32")
    if normalized_embeddings.ndim != 2:
        raise RuntimeError("Embedding cache must be a 2D float32 matrix.")
    if normalized_embeddings.shape[0] != len(entries):
        raise RuntimeError("Embedding cache row count does not match current entry count.")

    dim = int(normalized_embeddings.shape[1])
    write_embeddings_atomic(EMBEDDINGS_PATH, normalized_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    write_vector_meta(entries, model_path, dim)
    return {"count": len(entries), "dim": dim}


# 首次建库和缓存自愈
# 上层函数 update_store_and_rebuild 通过这里执行“重新编码全部记录”的完整后门
def rebuild_vector_index(
    entries: list[dict[str, Any]],
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    resolved_model_path = model_path or resolve_embed_model_path()
    if not entries:
        return rebuild_index_from_embeddings([], np.empty((0, 0), dtype="float32"), resolved_model_path)

    resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
    sorted_entries = sort_entries(entries)
    retrieval_texts = [build_retrieval_text(entry) for entry in sorted_entries]
    embeddings = encode_texts(retrieval_texts, resolved_embedder)
    return rebuild_index_from_embeddings(sorted_entries, embeddings, resolved_model_path)


# 顶层函数是 save_record、update_record 和 delete_records_by_ids 
# entries 参数来自 save_record 生成的 updated_entries = existing_entries + [saved_entry]
# 或者 update_record / delete_records_by_ids() 生成的完整候选列表
# 然后该函数把新纪录 entries 规范化、排序后整体写回主数据 JSON，
# 再基于新的主数据同步 field_record、重建时间线.md，并复用已有矩阵缓存重建索引
# 没有矩阵缓存就走全量重建路线
# update 时如果显式给出 updated_source_ids，就只重算该主记忆本体及其当前 field_record 家族，其余旧向量继续复用。
def update_store_and_rebuild(
    entries: list[dict[str, Any]],  # 传入的拼接了新记录的完整主数据列表
    embedder: Any | None = None,
    model_path: str | None = None,  # embeddeding 模型目录
    appended_entry: dict[str, Any] | None = None,  # 告诉函数“这次是新增了一条，而且这一条是谁”，方便走增量 embedding
    deleted_ids: list[str] | None = None,   # 告诉函数“这次删了哪些 id”，方便复用旧向量缓存
    updated_source_ids: set[str] | None = None,  # 告诉函数“这次更新了哪些主记忆家族”，方便只重算这一家的主记忆和 field_record
) -> dict[str, Any]:
    # 规范化并排序
    # 先把 主数据列表entries 规范化（统一成当前正式 schema），再按 updated_at / created_at / id 排序。
    normalized_entries = [normalize_store_entry(entry) for entry in entries]
    normalized_appended_entry = normalize_store_entry(appended_entry) if appended_entry is not None else None
    deleted_id_set = {str(record_id or "").strip() for record_id in deleted_ids or [] if str(record_id or "").strip()}
    normalized_updated_source_ids = {
        str(source_id or "").strip()
        for source_id in set(updated_source_ids or set())
        if str(source_id or "").strip()
    }
    synchronized_entries = synchronize_field_records(
        normalized_entries,
        appended_source_ids=(
            {str(normalized_appended_entry.get("id") or "").strip()}
            if normalized_appended_entry is not None and not is_field_record_entry(normalized_appended_entry)
            else None
        ),
        deleted_source_ids=deleted_id_set,
        updated_source_ids=normalized_updated_source_ids,
        full_regenerate=normalized_appended_entry is None and not deleted_id_set and not normalized_updated_source_ids,
    )
    # sorted_entries 就是完整库在内存中的最新状态：后面的主数据、时间线、向量矩阵、FAISS 和 meta 都以它为准。
    sorted_entries = sort_entries(synchronized_entries)
    public_sorted_entries = list_public_entries(sorted_entries)  # 对外统计和时间线只看主记忆，不把 field_record 算进去。
    store = {"version": STORE_VERSION, "entries": sorted_entries}
    background_rebuild_notice: dict[str, Any] | None = None

    # 写盘：写入拼接并排序后的最新的 主数据.JSON 和 时间线.md，这样上的业务事实源都已经更新成最新状态
    write_json_atomic(STORE_PATH, store)
    write_text_atomic(TIMELINE_PATH, build_timeline_markdown(sorted_entries))
    # 获取模型目录
    resolved_model_path = model_path or resolve_embed_model_path()

    # 开始更新后续的 .npy / .faiss / meta 文件。
    if not sorted_entries:  # 如果主数据列表为空，则清空embedding 缓存、FAISS 和 meta
        # 下面传入空 entries 和空矩阵，只是为了复用 rebuild_index_from_embeddings 里的清理逻辑，
        rebuild_stats = rebuild_index_from_embeddings([], np.empty((0, 0), dtype="float32"), resolved_model_path)
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
        cached_embedding_state = load_reusable_embedding_cache(resolved_model_path)

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
            current_entry_ids = [str(entry.get("id") or "").strip() for entry in sorted_entries]
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
                new_entries = [
                    entry
                    for entry in sorted_entries
                    if str(entry.get("id") or "").strip() in new_entry_id_set
                ]
                new_embeddings = encode_texts(
                    [build_retrieval_text(entry) for entry in new_entries],
                    resolved_embedder,
                )
                # 把新记录向量补进 "id -> 向量" 字典后，旧记录和新记录的向量就都齐了。
                for index, entry in enumerate(new_entries):
                    entry_id = str(entry.get("id") or "").strip()
                    cached_vector_by_id[entry_id] = new_embeddings[index]
                # 按“当前最新整库顺序”从字典里把向量重新取出来，拼成一份新的完整矩阵。
                # 这里重拼的是矩阵顺序，不是重新计算旧记录的 embedding。
                rebuilt_embeddings = np.vstack(
                    [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                ).astype("float32")
                # 用这份新矩阵统一重写 .npy、.faiss 和 meta。
                rebuild_stats = rebuild_index_from_embeddings(
                    sorted_entries,
                    rebuilt_embeddings,
                    resolved_model_path,
                )
            else:
                # 只要新增快路径的前提不再安全，就回退到完整重建，
                # 但这轮不再把全量 embedding 堵在前台请求里，而是改成后台慢任务继续补齐索引。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(sorted_entries),
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
            current_entry_ids = [str(entry.get("id") or "").strip() for entry in sorted_entries]
            # 删除快路径的安全条件是：
            # 1. 所有待删 id 都必须能在旧缓存里找到；
            # 2. 删除后还保留的每个 id，也都必须能在旧缓存里找到。
            # 满足这两个条件，就可以直接复用旧向量，不需要重新调用 embedding 模型。
            if all(deleted_id in cached_vector_by_id for deleted_id in deleted_id_set) and all(
                entry_id and entry_id in cached_vector_by_id for entry_id in current_entry_ids
            ):
                # 直接按删除后的最新顺序把剩余向量重拼成新矩阵。
                rebuilt_embeddings = np.vstack(
                    [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                ).astype("float32")
                # 用这份剩余矩阵重写 .npy、.faiss 和 meta。
                rebuild_stats = rebuild_index_from_embeddings(
                    sorted_entries,
                    rebuilt_embeddings,
                    resolved_model_path,
                )
            else:
                # 如果旧缓存和当前删除结果对不上，就退回全量重建，
                # 但这轮不再把全量 embedding 堵在前台请求里，而是改成后台慢任务继续补齐索引。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(sorted_entries),
                )
                rebuild_stats = load_current_index_stats()
        # update 的精确增量路径：
        # 这里不走“公共 delete + 公共 save”两步，而是直接在内存里替换旧主记忆家族，
        # 然后只重算这个主记忆本体及其当前 field_record 家族的向量。
        elif normalized_updated_source_ids and cached_embedding_state is not None:
            cached_embeddings, cached_row_ids = cached_embedding_state
            cached_vector_by_id = {
                row_id: cached_embeddings[index]
                for index, row_id in enumerate(cached_row_ids)
            }
            current_entry_ids = [str(entry.get("id") or "").strip() for entry in sorted_entries]
            updated_family_entries = [
                entry
                for entry in sorted_entries
                if (
                    str(entry.get("id") or "").strip() in normalized_updated_source_ids
                    or str(entry.get("source_memory_id") or "").strip() in normalized_updated_source_ids
                )
            ]
            updated_family_id_set = {
                str(entry.get("id") or "").strip()
                for entry in updated_family_entries
                if str(entry.get("id") or "").strip()
            }
            if (
                updated_family_id_set
                and all(
                    entry_id and (entry_id in cached_vector_by_id or entry_id in updated_family_id_set)
                    for entry_id in current_entry_ids
                )
            ):
                resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
                updated_embeddings = encode_texts(
                    [build_retrieval_text(entry) for entry in updated_family_entries],
                    resolved_embedder,
                )
                for index, entry in enumerate(updated_family_entries):
                    entry_id = str(entry.get("id") or "").strip()
                    cached_vector_by_id[entry_id] = updated_embeddings[index]
                rebuilt_embeddings = np.vstack(
                    [cached_vector_by_id[entry_id] for entry_id in current_entry_ids]
                ).astype("float32")
                rebuild_stats = rebuild_index_from_embeddings(
                    sorted_entries,
                    rebuilt_embeddings,
                    resolved_model_path,
                )
            else:
                # 只要当前缓存无法证明“除了这次更新家族外，其余记录都还能直接复用旧向量”，
                # 就改成后台全量重建，避免把长时间 embedding 堵在当前工具调用里。
                background_rebuild_notice = launch_background_full_rebuild(
                    resolved_model_path,
                    len(sorted_entries),
                )
                rebuild_stats = load_current_index_stats()
        else:
            # 这里是完整后门：
            # 首次建库、缓存不存在、缓存和当前模型/字段规则不一致、或新增/删除条件无法安全命中时，
            # 都改成后台慢任务去做“重新为全部记录生成 retrieval_text 并全量 embedding”的保守路线。
            background_rebuild_notice = launch_background_full_rebuild(
                resolved_model_path,
                len(sorted_entries),
            )
            rebuild_stats = load_current_index_stats()

        if background_rebuild_notice is None:
            write_rebuild_state(
                build_rebuild_state_snapshot(
                    state="idle",
                    finished_at=now_iso(),
                    target_entry_count=len(sorted_entries),
                    worker_pid=os.getpid(),
                )
            )

    # 最后返回给 save_record / delete_records_by_ids 一个简短摘要，
    # 让上层函数知道：当前整库一共有多少条、时间线前几项是什么、索引维度和条数是多少。
    # 这里不返回完整主数据，而只返回上层真正关心的结果摘要。
    timeline_summary = [
        f"{entry['updated_at']} | {entry['title']}"
        for entry in public_sorted_entries[:3]
    ]
    result = {
        "normalized_entries": sorted_entries,
        "total_entries": len(public_sorted_entries),
        "timeline_summary": timeline_summary,
        "index_stats": rebuild_stats,
    }
    if background_rebuild_notice is not None:
        result.update(background_rebuild_notice)
    return result
