"""memory-rag-mcp 的服务入口模块。

这个模块只保留三层内容：
1. FastMCP 服务对象创建。
2. 每个工具直接调用的高层主流程函数。
3. 对外暴露的 MCP 工具接口和 main() 启动入口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import faiss
from mcp.server.fastmcp import FastMCP
from pydantic import Field

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# 导入底层函数
from server_utils import (
    DEBUG_RESULT_FIELDS,
    DETAIL_RESULT_FIELDS,
    INDEX_PATH,
    MATCHED_FIELDS_LIMIT,
    MAX_TOP_K,
    META_PATH,
    SEARCH_RESULT_FIELDS,
    SERVER_NAME,
    MemoryKind,
    SourcePathListInput,
    TagListInput,
    build_fact_extraction_reminder,
    build_model_fingerprint,
    build_payload,
    build_store_entry_map,
    build_update_candidate_record,
    encode_texts,
    ensure_write_operations_allowed,
    get_embedder,
    is_field_record_entry,
    list_public_entries,
    load_rebuild_state,
    load_store,
    normalize_record_ids,
    now_iso,
    payload_fingerprint,
    payload_id,
    resolve_embed_device,
    resolve_embed_model_path,
    RETRIEVAL_FIELD_SIGNATURE,
    sort_entries,
    update_store_and_rebuild,
    validate_update_changes,
)


# 初始化 MCP 服务对象；底层规则和 IO / 索引逻辑已下沉到 server_utils.py，这里只保留服务入口和高层主链路。
server = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "保存高信息密度的项目经验、会话事件与事实记忆，生成按最近更新时间整理的时间线，"
        "并通过 RAG 风格的向量检索返回最相关的历史记录。"
    ),
)

# 按多条记录 id 回源主数据并返回完整详情记忆；保留在本文件是为了让“接口 -> 主数据回源 -> 结果裁剪”的链路仍能在入口层直接看清。
# 直接读 memory.json，按 id 回源，不碰 meta
def get_detail_records_by_ids(record_ids: list[str], include_debug: bool = False) -> dict[str, Any]:
    normalized_ids = normalize_record_ids(record_ids)

    loaded_store = load_store()
    store_entries = list(loaded_store.get("entries") or [])
    store_entry_map = build_store_entry_map(store_entries)
    records: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for normalized_record_id in normalized_ids:
        source_entry = store_entry_map.get(normalized_record_id)
        if source_entry is None:
            missing_ids.append(normalized_record_id)
            continue
        if is_field_record_entry(source_entry):
            raise ValueError(f"field_record is an internal index asset and cannot be read directly: {normalized_record_id}")
        # 详情读取链路直接在这里裁剪主数据字段，避免再跳一层只服务本函数的包装函数。
        result = {
            field_name: source_entry.get(field_name)
            for field_name in DETAIL_RESULT_FIELDS
        }
        if include_debug:
            result.update(
                {
                    field_name: source_entry.get(field_name)
                    for field_name in DEBUG_RESULT_FIELDS
                }
            )
        records.append(result)

    return {
        "ids": normalized_ids,
        "records": records,
        "missing_ids": missing_ids,
    }


# 按多条记录 id 从主数据中硬删除条目，并在成功删除后更新主数据与向量库；保留在本文件是为了让“接口 -> 主数据变更 -> 触发重建”的链路可见。
def delete_records_by_ids(
    record_ids: list[str],
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    ensure_write_operations_allowed()
    normalized_ids = normalize_record_ids(record_ids)
    loaded_store = load_store()
    store_entries = list(loaded_store.get("entries") or [])
    store_entry_map = build_store_entry_map(store_entries)
    deleted_ids: list[str] = []
    missing_ids: list[str] = []

    for normalized_record_id in normalized_ids:
        source_entry = store_entry_map.get(normalized_record_id)
        if source_entry is None:
            missing_ids.append(normalized_record_id)
            continue
        if is_field_record_entry(source_entry):
            raise ValueError(f"field_record is an internal index asset and cannot be deleted directly: {normalized_record_id}")
        deleted_ids.append(normalized_record_id)

    if not deleted_ids:
        return {
            "ids": normalized_ids,
            "deleted_ids": [],
            "missing_ids": missing_ids,
            "deleted_count": 0,
        }

    deleted_id_set = set(deleted_ids)
    remaining_entries = [
        entry
        for entry in store_entries
        if str(entry.get("id") or "").strip() not in deleted_id_set
    ]
    rebuild_result = update_store_and_rebuild(
        remaining_entries,
        embedder=embedder,
        model_path=model_path,
        deleted_ids=deleted_ids,
    )
    result = {
        "ids": normalized_ids,
        "deleted_ids": deleted_ids,
        "missing_ids": missing_ids,
        "deleted_count": len(deleted_ids),
    }
    if "index_refresh_state" in rebuild_result:
        result["index_refresh_state"] = rebuild_result["index_refresh_state"]
        result["index_refresh_mode"] = rebuild_result["index_refresh_mode"]
        result["search_available"] = rebuild_result["search_available"]
        result["message"] = rebuild_result["message"]
    return result


# 查询向量索引，并回源主 JSON 返回轻量候选结果；保留在本文件是为了让“查询输入 -> 向量检索 -> 主数据回源”的主流程保持可读。
def search_records(
    query: str,
    top_k: int | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")

    # 搜索链路直接在这里读取和校验 meta，避免再绕一层只被本函数使用的包装函数。
    if not META_PATH.exists():
        raise RuntimeError("Vector meta file does not exist yet. Save at least one record first.")
    with META_PATH.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise RuntimeError(f"{META_PATH.name} must contain an object")
    if "row_to_id" not in meta:
        raise RuntimeError("Vector meta file uses an old schema. Rebuild the index before searching.")
    row_to_id = meta.get("row_to_id") or []
    if not isinstance(row_to_id, list):
        raise RuntimeError(f"{META_PATH.name} must contain a row_to_id list")
    if not row_to_id:
        return []

    loaded_store = load_store()
    store_entries = list(loaded_store.get("entries") or [])
    store_entry_map = build_store_entry_map(store_entries)
    public_entries = list_public_entries(store_entries)
    public_entry_map = build_store_entry_map(public_entries)
    if not INDEX_PATH.exists():
        raise RuntimeError("Vector index file does not exist yet. Save at least one record first.")
    index = faiss.read_index(str(INDEX_PATH))
    resolved_model_path = model_path or resolve_embed_model_path()
    current_model_fingerprint = build_model_fingerprint(resolved_model_path)
    if str(meta.get("model_fingerprint") or "").strip() != current_model_fingerprint:
        raise RuntimeError(
            "Vector index was built with a different embedding model. Rebuild the index before searching."
        )
    if str(meta.get("retrieval_field_signature") or "") != RETRIEVAL_FIELD_SIGNATURE:
        raise RuntimeError(
            "Vector index retrieval field signature does not match current code. Rebuild the index before searching."
        )
    resolved_embedder = embedder or get_embedder(resolved_model_path, resolve_embed_device())
    query_vector = encode_texts([normalized_query], resolved_embedder)

    dim = int(meta.get("dim") or 0)
    if not dim:
        raise RuntimeError("Vector meta dim is missing or invalid.")
    if int(query_vector.shape[1]) != dim:
        raise RuntimeError(
            f"Query embedding dim {query_vector.shape[1]} does not match index dim {dim}."
        )

    resolved_top_k = 5 if top_k is None else max(1, min(int(top_k), MAX_TOP_K))
    search_count = len(row_to_id)
    scores, indices = index.search(query_vector, search_count)

    aggregated_results: dict[str, dict[str, Any]] = {}
    matched_field_scores: dict[str, dict[str, float]] = {}
    for row_index, score in zip(indices[0].tolist(), scores[0].tolist()):
        if row_index < 0 or row_index >= len(row_to_id):
            continue
        row_item = row_to_id[row_index]
        if not isinstance(row_item, dict):
            continue
        entry_id = str(row_item.get("id") or "").strip()
        if not entry_id:
            continue
        matched_entry = store_entry_map.get(entry_id)
        if matched_entry is None:
            continue
        if is_field_record_entry(matched_entry):
            source_memory_id = str(matched_entry.get("source_memory_id") or "").strip()
            source_entry = public_entry_map.get(source_memory_id)
            if source_entry is None:
                continue
            result_id = source_memory_id
            matched_field_name = str(matched_entry.get("source_field_name") or "").strip()
        else:
            source_entry = public_entry_map.get(entry_id)
            if source_entry is None:
                continue
            result_id = entry_id
            matched_field_name = ""
        # 轻量候选结果在搜索链路内部直接裁剪字段，避免再跳转到只服务本函数的包装层。
        result = aggregated_results.get(result_id)
        if result is None:
            result = {
                field_name: source_entry.get(field_name)
                for field_name in SEARCH_RESULT_FIELDS
            }
            result["score"] = float(score)
            aggregated_results[result_id] = result
        elif float(score) > float(result["score"]):
            result["score"] = float(score)
        if matched_field_name:
            matched_field_scores.setdefault(result_id, {})
            matched_field_scores[result_id][matched_field_name] = max(
                float(score),
                matched_field_scores[result_id].get(matched_field_name, float("-inf")),
            )

    sorted_results = sorted(
        aggregated_results.values(),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    final_results: list[dict[str, Any]] = []
    for result in sorted_results[:resolved_top_k]:
        result_id = str(result.get("id") or "").strip()
        field_score_map = matched_field_scores.get(result_id) or {}
        if field_score_map:
            result["matched_fields"] = [
                field_name
                for field_name, _ in sorted(
                    field_score_map.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:MATCHED_FIELDS_LIMIT]
            ]
        final_results.append(result)

    return final_results


# 保存一条结构化记忆；
# 新增时增量编码新记录，
# 如果 fingerprint 重复命中则直接返回现有旧记录；
def save_record(
    memory_kind: MemoryKind,
    title: str,
    detailed_summary: str,
    short_summary: str,
    problem_background: str | None = None,
    analysis: str | None = None,
    action_steps: str | None = None,
    validation_result: str | None = None,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    ensure_write_operations_allowed()
    payload = build_payload(
        memory_kind=memory_kind,
        title=title,
        detailed_summary=detailed_summary,
        short_summary=short_summary,
        problem_background=problem_background,
        analysis=analysis,
        action_steps=action_steps,
        validation_result=validation_result,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
    )
    fingerprint = payload_fingerprint(payload)
    entry_id = payload_id(fingerprint)
    loaded_store = load_store()
    existing_entries = list(loaded_store["entries"])
    public_existing_entries = list_public_entries(existing_entries)
    existing = next((item for item in public_existing_entries if item.get("fingerprint") == fingerprint), None)

    resolved_model_path = model_path or resolve_embed_model_path()

    deduped = existing is not None
    if deduped:
        normalized_existing_entries = sort_entries(public_existing_entries)
        current_index_stats = {"count": len(existing_entries), "dim": 0}
        if META_PATH.exists():
            try:
                with META_PATH.open("r", encoding="utf-8") as handle:
                    current_meta = json.load(handle)
                if isinstance(current_meta, dict):
                    current_index_stats["dim"] = int(current_meta.get("dim") or 0)
            except (OSError, json.JSONDecodeError, ValueError):
                current_index_stats["dim"] = 0
        timeline_summary = [
            f"{entry['updated_at']} | {entry['title']}"
            for entry in normalized_existing_entries[:3]
        ]
        return {
            "id": existing["id"],
            "updated_at": existing["updated_at"],
            "created_at": existing["created_at"],
            "tags": existing["tags"],
            "deduped": True,
            "total_entries": len(normalized_existing_entries),
            "timeline_summary": timeline_summary,
            "index_stats": current_index_stats,
        }
    else:
        created_now = now_iso()
        # 先在内存里组装“准备入库的新记录”
        saved_entry = {  # 这是一条新记录的完整主数据对象
            "id": entry_id,
            "fingerprint": fingerprint,
            "created_at": created_now,
            "updated_at": created_now,
            **payload,
        }
        # 这一步只是 Python 列表变成：[A, B, C] + [D] = [A, B, C, D]，还没写盘
        updated_entries = existing_entries + [saved_entry]

    rebuild_result = update_store_and_rebuild(
        updated_entries,  # 这个是直接拼接的（未排序的）主数据的列表（内存版本）
        embedder=embedder,
        model_path=resolved_model_path,
        appended_entry=saved_entry,  # 这个就是这次新加的这条记忆。作用就是告诉重建函数这是走新增路径，可以复用旧缓存
    )
    # 这个就是一个 list[dict[str, Any]]，意思是当前整库规范化并排序后的完整记录列表
    normalized_entries = rebuild_result["normalized_entries"]

    # 把字段拿出来并 return
    normalized_saved_entry = next(
        (item for item in normalized_entries if item.get("id") == saved_entry["id"]),
        saved_entry,
    )
    result = {
        "id": normalized_saved_entry["id"],
        "updated_at": normalized_saved_entry["updated_at"],
        "created_at": normalized_saved_entry["created_at"],
        "tags": normalized_saved_entry["tags"],
        "deduped": deduped,
        "total_entries": rebuild_result["total_entries"],
        "timeline_summary": rebuild_result["timeline_summary"],
        "index_stats": rebuild_result["index_stats"],
    }
    if "index_refresh_state" in rebuild_result:
        result["index_refresh_state"] = rebuild_result["index_refresh_state"]
        result["index_refresh_mode"] = rebuild_result["index_refresh_mode"]
        result["search_available"] = rebuild_result["search_available"]
        result["message"] = rebuild_result["message"]
    # 第九块：非 fact 记忆真实写入成功后，附带一个结构化的 fact 提取提醒。
    # 这里不代表服务端已自动提取 fact，而是把后续推荐流程显式返回给调用方。
    if str(normalized_saved_entry.get("memory_kind") or "").strip() != "fact":
        result["fact_extraction_reminder"] = build_fact_extraction_reminder()
    return result


# 按 id 对一条已有记忆做补丁式更新；
# 保留在本文件是为了让“接口输入 -> 主数据补丁 -> 重建触发”的主流程保持可见。
def update_record(
    record_id: str,
    changes: dict[str, Any],
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    ensure_write_operations_allowed()
    # 第一块：先规范化目标 id，尽早挡住空 id，避免后面把“没传主键”和“记录不存在”混成一类错误。
    normalized_record_id = str(record_id or "").strip()
    if not normalized_record_id:
        raise ValueError("id is required")

    # 第二块：校验补丁对象，并从主数据里定位这次要更新的源主记忆。
    # 这里仍然只允许更新公共主记忆；field_record 是内部索引资产，不能被外部直接当业务对象修改。
    normalized_changes = validate_update_changes(changes)
    loaded_store = load_store()  # return dict
    # existing_entries：整库所有记录的列表
    existing_entries = list(loaded_store.get("entries") or [])
    # store_entry_map：把这个列表改造成 id -> 记录 的字典
    store_entry_map = build_store_entry_map(existing_entries)
    # source_entry：从这个字典里拿出“本次要更新的那条旧记录”
    source_entry = store_entry_map.get(normalized_record_id)
    if source_entry is None:
        raise ValueError(f"record not found: {normalized_record_id}")
    if is_field_record_entry(source_entry):
        raise ValueError(f"field_record is an internal index asset and cannot be updated directly: {normalized_record_id}")

    # 第三块：基于“旧记录 + 补丁”合成候选新记录。
    # 这里会重新走底层 payload 规范化和 fingerprint 计算，所以后面可以直接判断：
    # 这次到底是 no-op，还是一次真实内容变更。
    candidate_entry = build_update_candidate_record(source_entry, normalized_changes)

    # 第四块：如果规范化后的候选记录和当前记录 fingerprint 一样，说明这次 update 没有带来真实内容变化。
    # 这种场景直接返回 updated=false，不刷新 updated_at，也不触发任何索引重建。
    if candidate_entry["fingerprint"] == source_entry.get("fingerprint"):
        normalized_existing_entries = sort_entries(list_public_entries(existing_entries))
        current_index_stats = {"count": len(existing_entries), "dim": 0}
        if META_PATH.exists():
            try:
                with META_PATH.open("r", encoding="utf-8") as handle:
                    current_meta = json.load(handle)
                if isinstance(current_meta, dict):
                    current_index_stats["dim"] = int(current_meta.get("dim") or 0)
            except (OSError, json.JSONDecodeError, ValueError):
                current_index_stats["dim"] = 0
        timeline_summary = [
            f"{entry['updated_at']} | {entry['title']}"
            for entry in normalized_existing_entries[:3]
        ]
        return {
            "id": source_entry["id"],
            "updated_at": source_entry["updated_at"],
            "created_at": source_entry["created_at"],
            "tags": source_entry["tags"],
            "updated": False,
            "total_entries": len(normalized_existing_entries),
            "timeline_summary": timeline_summary,
            "index_stats": current_index_stats,
        }

    # 第五块：检查“更新后的内容是否和另一条主记忆完全撞车”。
    # 如果新 fingerprint 已经属于别的主记忆，就直接报冲突；旧记录保持不动，不做隐式合并。
    conflicting_entry = next(
        (
            entry
            for entry in list_public_entries(existing_entries)
            if str(entry.get("id") or "").strip() != normalized_record_id
            and entry.get("fingerprint") == candidate_entry["fingerprint"]
        ),
        None,
    )
    if conflicting_entry is not None:
        raise ValueError(
            f"update conflict: updated content matches existing record {conflicting_entry['id']}"
        )

    # 第六块：在内存里构造更新后的最终记录，并把旧主数据列表中的这一条替换掉。
    # 这里保留原 id 和 created_at，只刷新 updated_at；语义仍然是“更新旧记录”，不是“删掉再新建一条”。
    updated_at = now_iso()
    updated_entry = {
        **candidate_entry,
        "updated_at": updated_at,
    }
    # updated_entries 就是“把目标那一条替换掉之后的完整记录列表”
    # Python 的列表推导式，遍历旧列表里的每一条记录，如果这条记录的 id 等于当前要更新的 id，就换成新的 updated_entry；否则原样保留
    updated_entries = [
        updated_entry if str(entry.get("id") or "").strip() == normalized_record_id else entry
        for entry in existing_entries
    ]

    # 第七块：把替换后的完整主数据交给底层统一写回，并显式告诉底层“这次更新了哪个主记忆家族”。
    # 底层会据此只重算该主记忆本体及其 field_record 家族的向量，其余旧向量继续复用。
    rebuild_result = update_store_and_rebuild(
        updated_entries,
        embedder=embedder,
        model_path=model_path,
        updated_source_ids={normalized_record_id},
    )

    # 第八块：从重建后的整库结果里回捞这条最新记录，再组装成 update 接口的最终返回摘要。
    normalized_entries = rebuild_result["normalized_entries"]
    normalized_updated_entry = next(
        (item for item in normalized_entries if item.get("id") == updated_entry["id"]),
        updated_entry,
    )
    result = {
        "id": normalized_updated_entry["id"],
        "updated_at": normalized_updated_entry["updated_at"],
        "created_at": normalized_updated_entry["created_at"],
        "tags": normalized_updated_entry["tags"],
        "updated": True,
        "total_entries": rebuild_result["total_entries"],
        "timeline_summary": rebuild_result["timeline_summary"],
        "index_stats": rebuild_result["index_stats"],
    }
    if "index_refresh_state" in rebuild_result:
        result["index_refresh_state"] = rebuild_result["index_refresh_state"]
        result["index_refresh_mode"] = rebuild_result["index_refresh_mode"]
        result["search_available"] = rebuild_result["search_available"]
        result["message"] = rebuild_result["message"]
    # 第九块：非 fact 记忆真实更新成功后，附带一个结构化的 fact 提取提醒。
    # 需不需要继续提取 fact 由调用方判断，服务端本轮只返回建议，不做自动语义提取。
    if str(normalized_updated_entry.get("memory_kind") or "").strip() != "fact":
        result["fact_extraction_reminder"] = build_fact_extraction_reminder()
    return result


# 对外暴露 save 工具，用于保存高信息密度记忆并重建索引。
@server.tool(
    name="save",
    description=(
        "保存一条高信息密度的项目经验、会话事件或事实记忆，更新最近更新时间线，并重建 RAG 向量索引。"
        "AI 在调用前应先整理内容；如用户指定了参考文档，AI 应先阅读文档，再提取并补充结构化字段。"
        "推荐流程：1. 如果本次要存的是非事件记忆，先 search 看看是否已有与当前主题相近的记忆。"
        "2. 如果已存在相近记忆，优先判断是否应该对已有记忆执行 update。"
        "3. 如果不存在合适的已有记忆，再调用 save 新建一条记忆。"
        "4. 每次 save 或 update 之后，再判断这条记忆里是否真的包含长期可复用、跨会话稳定、对长期陪伴有帮助的事实；只有确实存在这类事实时，才考虑继续按 fact 流程处理：先 search 相近 fact，若已有语义相近 fact 则优先 update，若没有相近 fact 则再 save 新 fact。"
        "如果内容主要是项目问题的排查、调试和解决过程，而 project_record 已经完整记录，则通常不需要再额外提取 fact。"
        "这里说的是推荐使用策略，不是服务端自动执行逻辑。"
    ),
)
def save(
    memory_kind: Annotated[
        MemoryKind,
        Field(
            description=(
                "必填。记忆类型，只能是 project_record、chat_event 或 fact。"
                "chat_event 一般没有特殊说明时就作为一般事件类型存储。"
                "fact 表示稳定事实记忆，以个人长期事实为主，更适合保存个人信息、长期偏好、稳定约束、长期状态、最近目标、已确认关系共识这类跨会话可复用事实。"
                "不建议把 project_record 里已经完整记录的项目 debug 过程、排错流水和解决链路再重复存成 fact；只有当项目内容被提炼成真正稳定、可跨会话复用的规则时，才考虑转成 fact。"
            )
        ),
    ],
    title: Annotated[
        str,
        Field(description="必填。记录标题，用一句话概括本次记忆。"),
    ],
    short_summary: Annotated[
        str,
        Field(
            description=(
                "必填。用于嵌入检索的简短摘要，建议控制在一两句话内。"
                "这个字段需要 AI 主动总结，会放在 retrieval_fields 的第二位。"
            ),
        ),
    ],
    detailed_summary: Annotated[
        str,
        Field(
            description=(
                "必填。AI 整理后的最终详细总结，也是正式入库字段。"
                "如果用户指定了参考文档，这里的信息不能比参考文档更少。"
            )
        ),
    ],
    problem_background: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存问题及其背景、故障现象，以及这条已解决问题为什么值得记录；不要写成流水账。",
        ),
    ] = None,
    analysis: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存原因判断、方案分析、排查结论，以及为什么这样判断、为什么这么做；要留下后续可复用的思路。",
        ),
    ] = None,
    action_steps: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存真正解决问题时执行的关键步骤、命令、操作顺序或具体处理动作，不要堆无用流水过程。",
        ),
    ] = None,
    validation_result: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存测试、验证、复现是否消失、验收结果等验证结论；默认应记录已经解决并验证过的问题。",
        ),
    ] = None,
    tags: Annotated[
        TagListInput,
        Field(
            default=None,
            description=(
                "最终存储必有。推荐由 AI 传入标签数组；如果缺失，服务端会做基础兜底补全。"
            ),
            json_schema_extra={"examples": [["Quicker", "Windows", "故障修复"]]},
        ),
    ] = None,
    source_paths: Annotated[
        SourcePathListInput,
        Field(
            default=None,
            description="可选。普通来源材料路径或链接列表，例如日志、网页、截图来源。",
            json_schema_extra={"examples": [["C:\\Users\\ndir\\notes\\issue-log.md"]]},
        ),
    ] = None,
    reference_doc_path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "可选。参考文档路径。仅当用户指定了现成文档让 AI 归档时填写，"
                "不与 source_paths 混用。"
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    return save_record(
        memory_kind=memory_kind,
        title=title,
        detailed_summary=detailed_summary,
        short_summary=short_summary,
        problem_background=problem_background,
        analysis=analysis,
        action_steps=action_steps,
        validation_result=validation_result,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
    )


# 对外暴露 update 工具，用于按 id 更新一条已有记忆并重建索引。
@server.tool(
    name="update",
    description=(
        "按 id 更新一条已有记忆。"
        "changes 是补丁对象，传什么字段就改什么字段；如果规范化后内容未变化，则返回 updated=false 且不重建。"
    ),
)
def update(
    id: Annotated[
        str,
        Field(description="必填。要更新的目标记录 id，例如 pmem-xxxx。"),
    ],
    changes: Annotated[
        dict[str, Any],
        Field(
            description=(
                "必填。要修改的字段键值对对象，只传要改的业务字段。"
                "允许字段包括 memory_kind、title、short_summary、detailed_summary、"
                "problem_background、analysis、action_steps、validation_result、"
                "tags、source_paths、reference_doc_path。"
            ),
            json_schema_extra={"examples": [{"title": "更新后的标题", "analysis": "补充后的分析"}]},
        ),
    ],
) -> dict[str, Any]:
    return update_record(record_id=id, changes=changes)


# 对外暴露 search 工具，用于返回最相似的轻量候选结果。
@server.tool(
    name="search",
    description=(
        "按向量相似度检索已保存的项目经验、会话事件和事实记忆，返回最相关的 top-k 条轻量候选记录。"
        "如果需要完整详情，请使用 get_details_by_ids。"
    ),
)
def search(
    query: Annotated[
        str,
        Field(description="必填。用于向量检索的查询文本，可以是问题、关键词、故障现象或主题。"),
    ],
    top_k: Annotated[
        int,
        Field(
            default=5,
            description="可选。期望返回的结果条数；不传默认 5，超过系统上限会自动截断。",
        ),
    ] = 5,
) -> dict[str, Any]:
    rebuild_state = load_rebuild_state()
    resolved_top_k = 5 if top_k is None else max(1, min(int(top_k), MAX_TOP_K))
    rebuild_state_name = str(rebuild_state.get("state") or "").strip()
    if rebuild_state_name == "running":
        return {
            "query": query,
            "top_k": resolved_top_k,
            "results": [],
            "search_available": False,
            "rebuild_state": "running",
            "message": "索引正在后台重建，请稍后重试 search。",
        }
    if rebuild_state_name == "failed":
        return {
            "query": query,
            "top_k": resolved_top_k,
            "results": [],
            "search_available": False,
            "rebuild_state": "failed",
            "message": "索引重建上一次失败了；先触发一次新的写入，让系统重新重建索引，再继续 search。",
            "last_error": str(rebuild_state.get("last_error") or ""),
        }
    results = search_records(query=query, top_k=top_k)
    return {
        "query": query,
        "top_k": resolved_top_k,
        "results": results,
    }


# 对外暴露 get_details_by_ids 工具，用于按多条记录 id 读取完整详情。
@server.tool(
    name="get_details_by_ids",
    description=(
        "按记录 ids 读取一批项目经验、会话事件或事实记忆的完整详情。"
        "如果部分 id 不存在，会在 missing_ids 中显式返回。"
        "如需查看内部调试字段，可显式传入 include_debug=true。"
    ),
)
def get_details_by_ids(
    ids: Annotated[
        list[str],
        Field(
            description="必填。要读取的记录 id 数组。即使只查一条，也需要传 ids=[\"pmem-xxxx\"]。",
            json_schema_extra={"examples": [["pmem-123456789abc", "pmem-abcdef123456"]]},
        ),
    ],
    include_debug: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "可选。是否返回内部调试字段。"
                "开启后会额外返回 fingerprint 和 retrieval_fields。"
            ),
        ),
    ] = False,
) -> dict[str, Any]:
    return get_detail_records_by_ids(record_ids=ids, include_debug=include_debug)


# 对外暴露 delete_by_ids 工具，用于从主数据中硬删除多条记录并重建库。
@server.tool(
    name="delete_by_ids",
    description=(
        "按记录 ids 从主数据中硬删除一批记忆。"
        "删除成功后会同步更新时间线并重建向量索引。"
        "如果部分 id 不存在，会在 missing_ids 中显式返回。"
    ),
)
def delete_by_ids(
    ids: Annotated[
        list[str],
        Field(
            description="必填。要删除的记录 id 数组。即使只删一条，也需要传 ids=[\"pmem-xxxx\"]。",
            json_schema_extra={"examples": [["pmem-123456789abc", "pmem-abcdef123456"]]},
        ),
    ],
) -> dict[str, Any]:
    return delete_records_by_ids(record_ids=ids)


# 以 stdio 方式启动 MCP 服务，供 Codex 本地连接。
def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
