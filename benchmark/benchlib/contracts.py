"""Corpus-only adapter input; never forwards query labels."""
from __future__ import annotations
from typing import Any

def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in records:
        record_id = str(original.get("record_id") or "").strip()
        if not record_id or record_id in seen:
            raise ValueError("Each corpus record must have a unique non-empty record_id.")
        seen.add(record_id)
        messages: list[dict[str, str]] = []
        message_ids: set[str] = set()
        for item in original.get("messages", []):
            message_id = str(item.get("message_id") or "").strip()
            if not message_id or message_id in message_ids:
                raise ValueError(f"Duplicate or missing message_id in {record_id}.")
            message_ids.add(message_id)
            role = str(item.get("role") or "user").strip().lower()
            if role not in {"user", "assistant"}:
                raise ValueError("Corpus roles must be user or assistant; retain speaker names in content.")
            content = str(item.get("content") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not content:
                raise ValueError(f"Empty content for {record_id}/{message_id}.")
            messages.append({"message_id": message_id, "role": role, "content": content})
        if not messages:
            raise ValueError(f"Corpus record {record_id} has no messages.")
        normalized.append({
            "scenario_id": str(original.get("scenario_id") or ""),
            "record_id": record_id,
            "title": str(original.get("title") or record_id).strip(),
            "messages": messages,
        })
    return normalized
