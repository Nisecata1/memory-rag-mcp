"""memory-rag-mcp 的服务入口模块。

这个模块只保留三层内容：
1. FastMCP 服务对象创建的相关语句。
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

# 导入 server_utils 中的底层函数
from server_utils import (
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
    build_embedding_model_fingerprint,
    build_payload,
    build_project_id,
    build_update_candidate_record,
    count_public_store_entries,
    encode_texts,
    ensure_write_operations_allowed,
    fetch_entry_family_by_source_ids,
    fetch_public_timeline_summary_entries,
    fetch_project_registry_entry_by_title,
    fetch_store_entries_by_ids,
    fetch_store_entry_by_id,
    get_embedder,
    is_field_record_entry,
    load_rebuild_state,
    normalize_record_ids,
    now_iso,
    project_registry_has_project_records,
    generate_memory_id,
    resolve_embed_device,
    resolve_embed_model_path,
    RETRIEVAL_FIELD_SIGNATURE,
    update_store_and_rebuild,
    validate_update_changes,
    validate_update_changes_for_entry,
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
# 这里直接按 id 查 SQLite 主数据，不碰 meta。
def get_detail_records_by_ids(record_ids: list[str]) -> dict[str, Any]:
    normalized_ids = normalize_record_ids(record_ids)
    fetched_entries = fetch_store_entries_by_ids(normalized_ids)
    store_entry_map = {
        str(entry.get("id") or "").strip(): entry
        for entry in fetched_entries
    }
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
    fetched_entries = fetch_store_entries_by_ids(normalized_ids)
    store_entry_map = {
        str(entry.get("id") or "").strip(): entry
        for entry in fetched_entries
    }
    deleted_ids: list[str] = []
    missing_ids: list[str] = []

    for normalized_record_id in normalized_ids:
        source_entry = store_entry_map.get(normalized_record_id)
        if source_entry is None:
            missing_ids.append(normalized_record_id)
            continue
        if is_field_record_entry(source_entry):
            raise ValueError(f"field_record is an internal index asset and cannot be deleted directly: {normalized_record_id}")
        if (
            str(source_entry.get("memory_kind") or "").strip() == "project_registry"
            and project_registry_has_project_records(normalized_record_id)
        ):
            raise ValueError(
                f"project_registry still has project_records attached and cannot be deleted: {normalized_record_id}"
            )
        deleted_ids.append(normalized_record_id)

    if not deleted_ids:
        return {
            "ids": normalized_ids,
            "deleted_ids": [],
            "missing_ids": missing_ids,
            "deleted_count": 0,
        }
    rebuild_result = update_store_and_rebuild(
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


# 查询向量索引，并回源主 SQLite 返回轻量候选结果；保留在本文件是为了让“查询输入 -> 向量检索 -> 主数据回源”的主流程保持可读。
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
    row_to_id = meta.get("row_to_id")
    if not isinstance(row_to_id, list):
        raise RuntimeError(f"{META_PATH.name} must contain row_to_id as a string list")
    if not all(isinstance(row_item, str) and str(row_item or "").strip() for row_item in row_to_id):
        raise RuntimeError(f"{META_PATH.name} row_to_id must contain only non-empty strings")
    row_ids = [str(row_item or "").strip() for row_item in row_to_id]
    if not row_ids:
        return []

    if not INDEX_PATH.exists():
        raise RuntimeError("Vector index file does not exist yet. Save at least one record first.")
    index = faiss.read_index(str(INDEX_PATH))
    resolved_model_path = model_path or resolve_embed_model_path()
    current_embedding_model_fingerprint = build_embedding_model_fingerprint(resolved_model_path)
    if str(meta.get("embedding_model_fingerprint") or "").strip() != current_embedding_model_fingerprint:
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
    search_count = len(row_ids)
    # indices 是 FAISS 返回的 [[row_index_0, row_index_1, ...]] 形状的数组
    # 每个值就是向量在索引中的行号（位置编号）。
    scores, indices = index.search(query_vector, search_count)

    direct_entry_ids: set[str] = set()
    for row_index in indices[0].tolist():
        if row_index < 0 or row_index >= len(row_ids):
            continue
        entry_id = str(row_ids[row_index] or "").strip()
        if not entry_id:
            continue
        direct_entry_ids.add(entry_id)
    fetched_direct_entries = fetch_store_entries_by_ids(list(direct_entry_ids)) if direct_entry_ids else []
    direct_entry_map = {
        str(entry.get("id") or "").strip(): entry
        for entry in fetched_direct_entries
    }
    source_memory_ids = {
        str(entry.get("source_memory_id") or "").strip()
        for entry in fetched_direct_entries
        if is_field_record_entry(entry) and str(entry.get("source_memory_id") or "").strip()
    }
    family_entries = fetch_entry_family_by_source_ids(source_memory_ids) if source_memory_ids else []
    family_entry_map = {
        str(entry.get("id") or "").strip(): entry
        for entry in family_entries
    }
    public_entry_map = {
        str(entry.get("id") or "").strip(): entry
        for entry in fetched_direct_entries
        if not is_field_record_entry(entry)
    }
    public_entry_map.update(
        {
        str(entry.get("id") or "").strip(): entry
        for entry in family_entries
        if not is_field_record_entry(entry)
        }
    )

    aggregated_results: dict[str, dict[str, Any]] = {}
    matched_field_scores: dict[str, dict[str, float]] = {}
    for row_index, score in zip(indices[0].tolist(), scores[0].tolist()):
        if row_index < 0 or row_index >= len(row_ids):
            continue
        entry_id = str(row_ids[row_index] or "").strip()
        if not entry_id:
            continue
        matched_entry = direct_entry_map.get(entry_id) or family_entry_map.get(entry_id)
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
            source_entry = matched_entry
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
# 新增时增量编码新记录，并同步更新时间线与向量索引。
def save_record(
    memory_kind: str,
    title: str,
    detailed_summary: str,
    short_summary: str,
    overview_summary: str | None = None,
    problem_background: str | None = None,
    analysis: str | None = None,
    action_steps: str | None = None,
    validation_result: str | None = None,
    project_id: str | None = None,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    record_id_override: str | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    ensure_write_operations_allowed()
    payload = build_payload(
        memory_kind=memory_kind,
        title=title,
        detailed_summary=detailed_summary,
        short_summary=short_summary,
        overview_summary=overview_summary,
        problem_background=problem_background,
        analysis=analysis,
        action_steps=action_steps,
        validation_result=validation_result,
        project_id=project_id,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
    )
    entry_id = str(record_id_override or "").strip() or generate_memory_id(memory_kind)

    resolved_model_path = model_path or resolve_embed_model_path()

    created_now = now_iso()
    # 先在内存里组装“准备入库的新记录”
    saved_entry = {  # 这是一条新记录的完整主数据对象
        "id": entry_id,
        "created_at": created_now,
        "updated_at": created_now,
        **payload,
    }

    rebuild_result = update_store_and_rebuild(
        embedder=embedder,
        model_path=resolved_model_path,
        appended_entry=saved_entry,  # 这个就是这次新加的这条记忆。作用就是告诉重建函数这是走新增路径，可以复用旧缓存
    )
    resolved_entries = rebuild_result["resolved_entries"]

    # 把字段拿出来并 return
    normalized_saved_entry = next(
        (item for item in resolved_entries if item.get("id") == saved_entry["id"]),
        saved_entry,
    )
    result = {
        "id": normalized_saved_entry["id"],
        "updated_at": normalized_saved_entry["updated_at"],
        "created_at": normalized_saved_entry["created_at"],
        "tags": normalized_saved_entry["tags"],
        "total_entries": rebuild_result["total_entries"],
        "timeline_summary": rebuild_result["timeline_summary"],
        "index_stats": rebuild_result["index_stats"],
    }
    if str(normalized_saved_entry.get("project_id") or "").strip():
        result["project_id"] = str(normalized_saved_entry.get("project_id") or "").strip()
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
# 保留在主 server.py 文件是为了让“接口输入 -> 主数据补丁 -> 重建触发”的主流程保持可见。
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
    source_entry = fetch_store_entry_by_id(normalized_record_id)
    if source_entry is None:
        raise ValueError(f"record not found: {normalized_record_id}")
    if is_field_record_entry(source_entry):
        raise ValueError(f"field_record is an internal index asset and cannot be updated directly: {normalized_record_id}")
    normalized_changes = validate_update_changes_for_entry(source_entry, normalized_changes)

    # 第三块：基于“旧记录 + 补丁”合成候选新记录。
    # 这里会重新走底层 payload 规范化，保证 update 和 save 共用同一套字段收口规则。
    candidate_entry = build_update_candidate_record(source_entry, normalized_changes)

    # 第六块：在内存里构造更新后的最终记录，并把旧主数据列表中的这一条替换掉。
    # 这里保留原 id 和 created_at，只刷新 updated_at；语义仍然是“更新旧记录”，不是“删掉再新建一条”。
    updated_at = now_iso()
    updated_entry = {
        **candidate_entry,
        "updated_at": updated_at,
    }

    # 第七块：把替换后的完整主数据交给底层统一写回，并显式告诉底层“这次更新了哪个主记忆家族”。
    # 底层会据此只重算该主记忆本体及其 field_record 家族的向量，其余旧向量继续复用。
    rebuild_result = update_store_and_rebuild(
        embedder=embedder,
        model_path=model_path,
        previous_entry=source_entry,
        updated_entry=updated_entry,
        updated_source_ids={normalized_record_id},
        updated_field_names=set(normalized_changes.keys()),
    )

    # 第八块：从重建后的整库结果里回捞这条最新记录，再组装成 update 接口的最终返回摘要。
    resolved_entries = rebuild_result["resolved_entries"]
    normalized_updated_entry = next(
        (item for item in resolved_entries if item.get("id") == updated_entry["id"]),
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
    if str(normalized_updated_entry.get("project_id") or "").strip():
        result["project_id"] = str(normalized_updated_entry.get("project_id") or "").strip()
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


# 显式创建一条 project_registry 项目摘要；这里先挡住同名项目，再把项目摘要作为新的可检索主记忆写入统一索引。
def create_project_record(
    title: str,
    overview_summary: str,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    existing_project = fetch_project_registry_entry_by_title(title)
    if existing_project is not None:
        raise ValueError(
            f"project title already exists: {existing_project['title']} ({existing_project['id']})"
        )
    generated_project_id = build_project_id()
    result = save_record(
        memory_kind="project_registry",
        title=title,
        detailed_summary="",
        short_summary="",
        overview_summary=overview_summary,
        project_id=generated_project_id,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
        record_id_override=generated_project_id,
        embedder=embedder,
        model_path=model_path,
    )
    result["project_id"] = generated_project_id
    return result


# 向一个已存在 project_id 下面追加一条共享 project_record；这里先校验项目是否存在，再复用现有 save 增量刷新路径。
def save_project_record(
    project_id: str,
    title: str,
    short_summary: str,
    detailed_summary: str,
    problem_background: str,
    analysis: str,
    action_steps: str,
    validation_result: str,
    tags: list[str] | str | None = None,
    source_paths: list[str] | str | None = None,
    reference_doc_path: str | None = None,
    embedder: Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    normalized_project_id = str(project_id or "").strip()
    if not normalized_project_id:
        raise ValueError("project_id is required")
    project_entry = fetch_store_entry_by_id(normalized_project_id)
    if project_entry is None or str(project_entry.get("memory_kind") or "").strip() != "project_registry":
        raise ValueError(f"project_id does not exist: {normalized_project_id}")
    result = save_record(
        memory_kind="project_record",
        title=title,
        detailed_summary=detailed_summary,
        short_summary=short_summary,
        problem_background=problem_background,
        analysis=analysis,
        action_steps=action_steps,
        validation_result=validation_result,
        project_id=normalized_project_id,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
        embedder=embedder,
        model_path=model_path,
    )
    result["project_id"] = normalized_project_id
    return result


# 对外暴露 save_fact 工具，专门用于保存长期 fact 事实记忆。
@server.tool(
    name="save_fact",
    description=(
        "保存一条长期 fact 记忆，用于记录跨会话稳定的个人长期事实、偏好、约束或目标，更新最近更新时间线，并重建 RAG 向量索引。"
        "AI 在调用前应先整理内容；如用户指定了参考文档，AI 应先阅读文档，再提取并补充结构化字段。"
        "正式入库内容只应包含已验证的事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；不要把未验证的推测、归因或建议写成正式事实记忆。"
        "如果与已有已验证事实相似，优先先 search 并融合到已有记录或本次更新里，不要重复堆新的近似表述。"
        "推荐流程：1. 先 search 看看是否已有与当前主题相近的 fact 记忆。"
        "2. 如果已存在相近 fact，优先判断是否应该对已有记忆执行 update。"
        "3. 如果不存在合适 fact，再调用 save_fact 新建一条事实记忆。"
        "请注意，只有真正稳定、可跨会话复用的规则和个人事实才值得存为 fact；不要把项目问题排查记录或一次性事件存成 fact，它们有专属的 project_record 和 chat_event 接口。"
    ),
)
def save_fact(
    title: Annotated[
        str,
        Field(description="必填。记录标题，用一句话概括本次记忆。只写已验证事实、用户明确实践过的操作，或有直接证据支撑的结论；不要补写未经验证的推测、归因或建议。"),
    ],
    short_summary: Annotated[
        str,
        Field(
            description=(
                "必填。用于嵌入检索的简短摘要，建议控制在一两句话内。"
                "这个字段需要 AI 主动总结，会放在 retrieval_fields 的第二位。"
                "内容只应来自已验证事实、用户明确实践过的操作，或有直接证据支撑的结论；如果与已有已验证内容相似，优先融合整理，不要重复新编近似说法。"
            ),
        ),
    ],
    detailed_summary: Annotated[
        str,
        Field(
            description=(
                "必填。AI 整理后的最终详细总结，也是正式入库字段。"
                "如果用户指定了参考文档，这里的信息不能比参考文档更少。"
                "只允许写入已验证事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论，不得补写未经验证的推测、归因或建议。"
            )
        ),
    ],
    problem_background: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存问题及其背景、故障现象，以及这条已解决问题为什么值得记录；不要写成流水账。只写已验证事实、用户明确实践过的操作，或有直接证据支撑的结论；如果与已有已验证内容相似，优先融合整理。",
        ),
    ] = None,
    analysis: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存已证实的原因判断、取舍依据和排查结论，以及为什么这样判断、为什么这么做；要留下后续可复用的思路。这里只允许写入已验证事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论，不要写怀疑方向、猜测性归因或未证实判断。",
        ),
    ] = None,
    action_steps: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存真正解决问题时执行的关键步骤、命令、操作顺序或具体处理动作，不要堆无用流水过程。只写用户明确实践过的操作，或有文件、截图、日志、命令输出可直接证明的动作和结果；不要补写未经实践的建议性步骤。",
        ),
    ] = None,
    validation_result: Annotated[
        str | None,
        Field(
            default=None,
            description="project_record 必填，其他类型可选。用于保存测试、验证、复现是否消失、验收结果等验证结论；默认应记录已经解决并验证过的问题。这里只写已验证结果或能被文件、截图、日志、命令输出直接证明的结论；不要补写未经验证的判断。",
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
        memory_kind="fact",
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


# 对外暴露 save_chatEvent 工具，固定把当前写入保存成新的 chat_event 事件记忆。
@server.tool(
    name="save_chatEvent",
    description=(
        "专门用于保存一条会话事件记忆。"
        "这个接口固定把 memory_kind 设为 chat_event。"
        "chat_event 默认按新事件存储，不因为主题相近、人物相近或问题相近，就默认更新旧 chat_event。"
        "只有在补充刚写入不久的同一事件，或纠正原记录事实错误时，才应优先考虑 update 旧 chat_event。"
        "正式入库内容只应包含已验证的事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；不要把未验证的推测、归因或建议写成正式记忆。"
    ),
)
def save_chatEvent(
    title: Annotated[
        str,
        Field(description="必填。事件标题，用一句话概括这次会话事件。只写已验证事实、用户明确实践过的操作，或有直接证据支撑的结论；不要补写未经验证的推测、归因或建议。"),
    ],
    short_summary: Annotated[
        str,
        Field(
            description=(
                "必填。用于嵌入检索的简短摘要，建议控制在一两句话内。"
                "这个字段需要 AI 主动总结，会放在 retrieval_fields 的第二位。"
                "内容只应来自已验证事实、用户明确实践过的操作，或有直接证据支撑的结论；如果与已有已验证内容相似，优先融合整理，不要重复新编近似说法。"
            ),
        ),
    ],
    detailed_summary: Annotated[
        str,
        Field(
            description=(
                "必填。AI 整理后的最终详细总结，也是正式入库字段。"
                "如果用户指定了参考文档，这里的信息不能比参考文档更少。"
                "只允许写入已验证事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论，不得补写未经验证的推测、归因或建议。"
            )
        ),
    ],
    tags: Annotated[
        TagListInput,
        Field(
            default=None,
            description=(
                "最终存储必有。推荐由 AI 传入标签数组；如果缺失，服务端会做基础兜底补全。"
            ),
            json_schema_extra={"examples": [["chat_event", "memory-rag-mcp", "会话记录"]]},
        ),
    ] = None,
    source_paths: Annotated[
        SourcePathListInput,
        Field(
            default=None,
            description="可选。普通来源材料路径或链接列表，例如日志、网页、截图来源。",
            json_schema_extra={"examples": [["C:\\Users\\ndir\\notes\\chat-log.md"]]},
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
        memory_kind="chat_event",
        title=title,
        detailed_summary=detailed_summary,
        short_summary=short_summary,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
    )


# 对外暴露 create_project 工具，固定创建一条 project_registry 项目摘要记忆。
@server.tool(
    name="create_project",
    description=(
        "显式创建一条新的项目摘要记忆。"
        "这个接口固定把记录类型写成 project_registry，并自动生成 project_id。"
        "project_id 只用于聚合同项目的具体问题记录和项目摘要，不参与 embedding 文本。"
        "项目摘要会进入向量库，也会生成自己的 field_record。"
        "正式入库内容只应包含已验证的事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；不要把未验证的推测、归因或建议写成正式记忆。"
    ),
)
def create_project(
    title: Annotated[
        str,
        Field(description="必填。项目名称，用一句话概括项目本身。标题会作为项目摘要检索文本的一部分。"),
    ],
    overview_summary: Annotated[
        str,
        Field(
            description=(
                "必填。项目整体概述。"
                "它是 project_registry 的核心正文，会参与项目摘要的 embedding 和 field_record 生成。"
                "这里只允许写入已验证事实、用户明确实践过的操作，或有直接证据支撑的结论。"
            ),
        ),
    ],
    tags: Annotated[
        TagListInput,
        Field(
            default=None,
            description="可选。项目标签数组，会和 title、overview_summary 一起参与检索文本拼接。",
            json_schema_extra={"examples": [["memory-rag-mcp", "SQLite", "项目升级"]]},
        ),
    ] = None,
    source_paths: Annotated[
        SourcePathListInput,
        Field(
            default=None,
            description="可选。普通来源材料路径或链接列表，例如设计说明、任务文档、日志来源。",
            json_schema_extra={"examples": [["C:\\Users\\ndir\\docs\\project-brief.md"]]},
        ),
    ] = None,
    reference_doc_path: Annotated[
        str | None,
        Field(
            default=None,
            description="可选。参考文档路径。仅当用户明确指定了参考文档让 AI 归档时填写，不与 source_paths 混用。",
        ),
    ] = None,
) -> dict[str, Any]:
    return create_project_record(
        title=title,
        overview_summary=overview_summary,
        tags=tags,
        source_paths=source_paths,
        reference_doc_path=reference_doc_path,
    )


# 对外暴露 save_project 工具，固定把问题记录挂到一个已存在的 project_id 下面。
@server.tool(
    name="save_project",
    description=(
        "向一个已存在 project_id 的项目下保存一条新的 project_record。"
        "这个接口不会隐式创建项目；调用方必须先 create_project，再拿返回的 project_id 继续写项目问题。"
        "project_id 只负责把同项目的具体问题记录聚到一起，不参与 embedding 文本。"
        "正式入库内容只应包含已验证的事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；不要把未验证的推测、归因或建议写成正式记忆。"
    ),
)
def save_project(
    project_id: Annotated[
        str,
        Field(description="必填。已存在的 project_id；只有 project_registry 里存在的 id 才允许挂接项目问题。"),
    ],
    title: Annotated[
        str,
        Field(description="必填。问题记录标题，用一句话概括这次项目问题或子主题。"),
    ],
    short_summary: Annotated[
        str,
        Field(
            description=(
                "必填。用于嵌入检索的简短摘要，建议控制在一两句话内。"
                "内容只应来自已验证事实、用户明确实践过的操作，或有直接证据支撑的结论。"
            ),
        ),
    ],
    detailed_summary: Annotated[
        str,
        Field(
            description=(
                "必填。AI 整理后的正式详细总结。"
                "只允许写入已验证事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论。"
            ),
        ),
    ],
    problem_background: Annotated[
        str,
        Field(description="必填。问题背景、故障现象，以及这条项目问题为什么值得记录。"),
    ],
    analysis: Annotated[
        str,
        Field(description="必填。已证实的原因判断、取舍依据和排查结论；不要写怀疑方向或未证实归因。"),
    ],
    action_steps: Annotated[
        str,
        Field(description="必填。真正执行过的关键步骤、命令和操作顺序；不要补写未经实践的建议。"),
    ],
    validation_result: Annotated[
        str,
        Field(description="必填。测试、验证、复现是否消失或验收结果等已证实结论。"),
    ],
    tags: Annotated[
        TagListInput,
        Field(
            default=None,
            description="可选。项目问题标签数组；缺失时服务端会做基础兜底补全。",
            json_schema_extra={"examples": [["SQLite", "schema", "memory-rag-mcp"]]},
        ),
    ] = None,
    source_paths: Annotated[
        SourcePathListInput,
        Field(
            default=None,
            description="可选。普通来源材料路径或链接列表，例如日志、网页、截图来源。",
            json_schema_extra={"examples": [["C:\\Users\\ndir\\logs\\issue.log"]]},
        ),
    ] = None,
    reference_doc_path: Annotated[
        str | None,
        Field(
            default=None,
            description="可选。参考文档路径。仅当用户明确指定了参考文档让 AI 归档时填写，不与 source_paths 混用。",
        ),
    ] = None,
) -> dict[str, Any]:
    return save_project_record(
        project_id=project_id,
        title=title,
        short_summary=short_summary,
        detailed_summary=detailed_summary,
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
        "changes 是补丁对象，传什么字段就改什么字段；通过校验后会按当前类型写回主数据并刷新受影响索引。"
        "changes 中新增或替换的内容，只应包含已验证的事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；不要把未验证的推测、归因或建议补进已有记录。"
        "如果只是补充已有相似且已验证的内容，优先融合到原记录，而不是制造重复表达。"
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
                "允许字段包括 title、short_summary、overview_summary、detailed_summary、"
                "problem_background、analysis、action_steps、validation_result、"
                "tags、source_paths、reference_doc_path。"
                "传入的业务字段值只应包含已验证事实、用户明确实践过的操作，或能被文件、截图、日志、命令输出直接证明的结论；"
                "不要把未验证的推测、归因或建议写成更新内容。"
                "project_id 不允许通过 update 修改。"
                "记忆类型切换不再由 update 完成；如果要把 chat_event 调整成 project_record，应由调用方重新提炼内容后调用 save 新存，必要时再 delete 原记录。"
                "如果与原记录或其他已有记录中的已验证内容相似，优先融合整理，不要重复补一条近似说法。"
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
        "按向量相似度检索已保存的项目经验、项目摘要、会话事件和事实记忆，返回最相关的 top-k 条轻量候选记录。"
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
) -> dict[str, Any]:
    return get_detail_records_by_ids(record_ids=ids)


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
