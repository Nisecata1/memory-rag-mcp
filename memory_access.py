"""Memory reads expressed in business terms, independent of database syntax.

The application supplies the read functions and dialogue policy for each request.
There are no connections, file paths, configuration reads, or shared caches here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


Record = dict[str, Any]
ReadRecords = Callable[[list[str]], list[Record]]
NormalizeIds = Callable[[list[str]], list[str]]
BuildContext = Callable[[Any, list[str] | None], dict[str, Any]]


def read_details(
    record_ids: list[str],
    matched_message_ids: dict[str, list[str]] | None = None,
    full_dialogue: bool = False,
    *,
    read_records: ReadRecords,
    normalize_ids: NormalizeIds,
    build_context: BuildContext,
) -> dict[str, Any]:
    """Load each requested record once and select its output-only chat view."""
    normalized_ids = normalize_ids(record_ids)
    anchors_by_id = matched_message_ids if matched_message_ids is not None else {}
    if not full_dialogue:
        if not isinstance(anchors_by_id, dict) or any(
            record_id not in normalized_ids
            or not isinstance(message_ids, list)
            or any(not isinstance(message_id, str) or not message_id.strip() for message_id in message_ids)
            for record_id, message_ids in anchors_by_id.items()
        ):
            raise ValueError("matched_message_ids must map requested record IDs to non-empty message ID strings.")

    entries_by_id = {
        str(entry.get("id") or "").strip(): entry
        for entry in read_records(normalized_ids)
    }
    records: list[Record] = []
    missing_ids: list[str] = []
    for record_id in normalized_ids:
        entry = entries_by_id.get(record_id)
        if entry is None:
            missing_ids.append(record_id)
            continue
        if entry.get("memory_kind") == "chat_event" and not full_dialogue:
            entry = {
                **entry,
                "raw_dialogue": build_context(entry.get("raw_dialogue"), anchors_by_id.get(record_id)),
            }
        records.append(entry)
    return {"ids": normalized_ids, "records": records, "missing_ids": missing_ids}


@dataclass(frozen=True)
class SearchRecords:
    """Search projections plus the public parent records needed to display them."""

    direct_by_id: dict[str, Record]
    public_by_id: dict[str, Record]


def read_search_records(
    entry_ids: list[str],
    *,
    read_entries: ReadRecords,
    read_parents: ReadRecords,
) -> SearchRecords:
    """Read lightweight hits, reusing parents that were already among the hits."""
    direct = read_entries(entry_ids) if entry_ids else []
    direct_by_id = {str(entry.get("id") or "").strip(): entry for entry in direct}
    public_by_id = {
        entry_id: entry
        for entry_id, entry in direct_by_id.items()
        if entry.get("memory_kind") != "field_record"
    }
    missing_parent_ids = {
        str(entry.get("source_memory_id") or "").strip()
        for entry in direct
        if entry.get("memory_kind") == "field_record"
        and str(entry.get("source_memory_id") or "").strip()
    } - public_by_id.keys()
    if missing_parent_ids:
        for parent in read_parents(sorted(missing_parent_ids)):
            parent_id = str(parent.get("id") or "").strip()
            if parent_id in missing_parent_ids and parent.get("memory_kind") != "field_record":
                public_by_id[parent_id] = parent
    return SearchRecords(direct_by_id=direct_by_id, public_by_id=public_by_id)
