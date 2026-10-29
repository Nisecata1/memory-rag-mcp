"""Pure memory and dialogue rules, independent of storage and runtime configuration.

Callers supply policy values explicitly so the same rules can be used by the MCP
service, a storage adapter, or an in-memory test without opening a database.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


def collapse_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_text_block(value: str | None) -> str:
    raw_text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw_text.split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def normalize_json_metadata(value: dict[str, Any], field_path: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{field_path} must contain only JSON-serializable values") from exc


def normalize_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def resolve_message_top_k(
    message_top_k: int | None = None, *, default_message_top_k: int = 3
) -> int:
    resolved = default_message_top_k if message_top_k is None else message_top_k
    if type(resolved) is not int or resolved <= 0:
        raise ValueError("message_top_k must be a positive integer.")
    return resolved


def select_matched_messages(
    message_scores: dict[str, tuple[float, int]],
    message_top_k: int | None = None,
    *,
    default_message_top_k: int = 3,
    message_score_gap: float = 0.2,
    return_message_num: int = 10,
) -> list[dict[str, Any]]:
    """Rank message anchors independently of the parent record's other fields."""
    limit = resolve_message_top_k(message_top_k, default_message_top_k=default_message_top_k)
    ranked = sorted(
        ((message_id, score, index) for message_id, (score, index) in message_scores.items() if math.isfinite(score)),
        key=lambda item: (-item[1], item[2], item[0]),
    )
    if not ranked:
        return []
    best_score = ranked[0][1]
    selected: list[dict[str, Any]] = []
    selected_positions: list[int] = []
    for message_id, score, index in ranked:
        difference = best_score - score
        # FAISS scores are float32; preserve the existing inclusive tolerance.
        if not (
            difference <= message_score_gap
            or (message_score_gap > 0 and math.isclose(difference, message_score_gap, rel_tol=0.0, abs_tol=1e-7))
        ):
            continue
        if any(abs(index - position) <= return_message_num for position in selected_positions):
            continue
        selected.append({"message_id": message_id, "score": float(score)})
        selected_positions.append(index)
        if len(selected) == limit:
            break
    return selected


def build_dialogue_context(
    dialogue: Any,
    matched_message_ids: list[str] | None = None,
    *,
    return_message_num: int = 10,
) -> dict[str, Any]:
    """Build an output-only view; never change the stored dialogue or messages."""
    if not isinstance(dialogue, dict) or not isinstance(dialogue.get("messages"), list):
        raise ValueError("raw_dialogue is not structured; request full_dialogue=true to read it.")
    messages = dialogue["messages"]
    positions = {message["message_id"]: index for index, message in enumerate(messages)}
    requested = list(dict.fromkeys(matched_message_ids or []))
    unresolved = [message_id for message_id in requested if message_id not in positions]
    anchors = [message_id for message_id in requested if message_id in positions]
    selected = {positions[message_id] for message_id in anchors}
    if anchors:
        mode = "matched_context"
        reason = "matched_messages"
        for message_id in anchors:
            center = positions[message_id]
            selected.update(range(max(0, center - return_message_num), min(len(messages), center + return_message_num + 1)))
    else:
        mode = "tail_preview"
        reason = "anchors_not_found" if requested else "no_anchors"
        count = 2 * return_message_num + 1
        selected = set(range(max(0, len(messages) - count), len(messages)))

    output: list[dict[str, Any]] = []

    def append_omission(count: int) -> None:
        if count:
            output.append({
                "type": "omission",
                "omitted_count": count,
                "content": f"[... 省略 {count} 条消息 ...]",
            })

    previous = -1
    for index in sorted(selected):
        append_omission(index - previous - 1)
        output.append(dict(messages[index]))
        previous = index
    append_omission(len(messages) - previous - 1)
    return {
        **dialogue,
        "messages": output,
        "context": {
            "mode": mode,
            "reason": reason,
            "total_messages": len(messages),
            "returned_messages": len(selected),
            "truncated": len(selected) < len(messages),
            "anchor_message_ids": anchors,
            "unresolved_message_ids": unresolved,
            "omitted_anchor_count": 0,
        },
    }


def normalize_raw_dialogue(
    value: Any,
    *,
    allow_legacy: bool = False,
    schema_version: int = 1,
    roles: tuple[str, ...] = ("user", "assistant"),
) -> str | dict[str, Any]:
    """Normalize a complete writable dialogue; reject read-only context views."""
    if not isinstance(value, dict):
        legacy_text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if legacy_text and not allow_legacy:
            raise ValueError("raw_dialogue must be a structured object with a non-empty messages array")
        return legacy_text

    if "context" in value:
        raise ValueError(
            "raw_dialogue is a read-only context view and cannot be saved or updated; "
            "request full_dialogue=true before replacing the complete dialogue."
        )
    unsupported_top_level_fields = set(value) - {"schema_version", "messages", "metadata"}
    if unsupported_top_level_fields:
        unsupported_text = ", ".join(sorted(str(item) for item in unsupported_top_level_fields))
        raise ValueError(f"raw_dialogue contains unsupported fields: {unsupported_text}")

    raw_schema_version = value.get("schema_version", schema_version)
    if raw_schema_version != schema_version:
        raise ValueError(f"raw_dialogue.schema_version must be {schema_version}")
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("raw_dialogue.messages must be a non-empty array")
    raw_top_level_metadata = value.get("metadata")
    if raw_top_level_metadata is not None and not isinstance(raw_top_level_metadata, dict):
        raise ValueError("raw_dialogue.metadata must be an object when provided")

    normalized_messages: list[dict[str, Any]] = []
    seen_message_ids: set[str] = set()
    for message_index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"raw_dialogue.messages[{message_index}] must be an object")
        unsupported_message_fields = set(raw_message) - {
            "message_id", "role", "content", "timestamp", "metadata",
        }
        if unsupported_message_fields:
            unsupported_text = ", ".join(sorted(str(item) for item in unsupported_message_fields))
            raise ValueError(f"raw_dialogue.messages[{message_index}] contains unsupported fields: {unsupported_text}")

        role = str(raw_message.get("role") or "").strip().lower()
        if role not in roles:
            raise ValueError(f"raw_dialogue.messages[{message_index}].role must be one of: {', '.join(roles)}")
        content = str(raw_message.get("content") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not content:
            raise ValueError(f"raw_dialogue.messages[{message_index}].content must not be empty")
        timestamp = str(raw_message.get("timestamp") or "").strip()
        raw_message_metadata = raw_message.get("metadata")
        if raw_message_metadata is not None and not isinstance(raw_message_metadata, dict):
            raise ValueError(f"raw_dialogue.messages[{message_index}].metadata must be an object when provided")

        message_id = str(raw_message.get("message_id") or "").strip()
        if not message_id:
            id_seed = json.dumps([message_index, role, content, timestamp], ensure_ascii=False, separators=(",", ":"))
            message_id = f"msg-{hashlib.sha256(id_seed.encode('utf-8')).hexdigest()[:12]}"
        if message_id in seen_message_ids:
            raise ValueError(f"raw_dialogue contains duplicate message_id: {message_id}")
        seen_message_ids.add(message_id)

        normalized_message: dict[str, Any] = {"message_id": message_id, "role": role, "content": content}
        if timestamp:
            normalized_message["timestamp"] = timestamp
        if raw_message_metadata:
            normalized_message["metadata"] = normalize_json_metadata(raw_message_metadata, f"raw_dialogue.messages[{message_index}].metadata")
        normalized_messages.append(normalized_message)

    normalized_dialogue: dict[str, Any] = {"schema_version": schema_version, "messages": normalized_messages}
    if raw_top_level_metadata:
        normalized_dialogue["metadata"] = normalize_json_metadata(raw_top_level_metadata, "raw_dialogue.metadata")
    return normalized_dialogue


def build_short_summary(
    title: str,
    detailed_summary: str,
    short_summary: str | None = None,
    *,
    max_length: int = 120,
) -> str:
    resolved_short_summary = collapse_text(short_summary)
    if resolved_short_summary:
        if len(resolved_short_summary) <= max_length:
            return resolved_short_summary
        return f"{resolved_short_summary[:max_length].rstrip()}..."

    normalized_title = collapse_text(title)
    normalized_summary = normalize_text_block(detailed_summary)
    candidate_lines = [collapse_text(line) for line in normalized_summary.split("\n") if collapse_text(line)]
    skip_prefixes = ("标题：", "title:", "title：")
    preferred_lines = [line for line in candidate_lines if line and line != normalized_title and not line.lower().startswith(skip_prefixes)]
    selected_line = preferred_lines[0] if preferred_lines else (candidate_lines[0] if candidate_lines else normalized_title)
    selected_line = re.sub(r"^(问题背景|问题|分析|步骤|验证结果|摘要|总结)[：:]\s*", "", selected_line).strip()
    if len(selected_line) <= max_length:
        return selected_line
    return f"{selected_line[:max_length].rstrip()}..."
