"""Private JSON-lines worker; never starts the MCP server or opens live memory."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def _initialize(request: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    started = time.perf_counter()
    project_dir = Path(request["project_dir"]).resolve()
    runtime = Path(request["runtime_dir"]).resolve()
    allowed_runs = (Path(__file__).resolve().parents[2] / "runs").resolve()
    production_data = (project_dir.parent / "memory-rag-mcp-data").resolve()
    if runtime == allowed_runs or not runtime.is_relative_to(allowed_runs) or runtime.is_relative_to(production_data):
        raise ValueError("Unsafe isolated runtime directory.")
    if not (project_dir / "server_utils.py").is_file() or not (project_dir / "server.py").is_file():
        raise ValueError("project_dir does not contain the expected memory-rag-mcp source.")
    radius = int(request.get("window_radius", 10))
    message_top_k = int(request.get("message_top_k", 3))
    if radius < 0 or message_top_k < 1:
        raise ValueError("Invalid window or anchor configuration.")
    db_path = runtime / "memory.db"

    # A forgotten path must fail closed before touching any production store.
    # The model and source files remain readable; source bytecode is disabled.
    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "sqlite3.connect":
            # CPython's sqlite audit event may expose the filename as bytes.
            candidate = Path(os.fsdecode(args[0])).resolve()
            if candidate != db_path:
                raise PermissionError(f"Benchmark worker may only connect to its isolated SQLite file: got {candidate!s}, expected {db_path!s}.")
        elif event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(args[0])).resolve()
            if candidate == production_data or candidate.is_relative_to(production_data):
                raise PermissionError("Benchmark worker cannot open production memory data.")

    sys.addaudithook(audit)
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(project_dir))
    import server_utils as utils

    # All nine independently captured paths must be redirected before any IO.
    paths = {
        "AI_MEMORY_DIR": runtime,
        "DATA_DIR": runtime / "data",
        "TIMELINE_PATH": runtime / "data" / "timeline.md",
        "EMBEDDINGS_PATH": runtime / "data" / "memory_embeddings.npy",
        "INDEX_PATH": runtime / "data" / "memory.faiss",
        "META_PATH": runtime / "data" / "meta.json",
        "REBUILD_STATE_PATH": runtime / "data" / "rebuild_state.json",
        "WRITE_LOCK_PATH": runtime / "data" / ".memory-rag-write.lock",
        "SQLITE_PATH": db_path,
    }
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_relative_to(runtime) or resolved.is_relative_to(production_data):
            raise ValueError(f"Unsafe redirected path for {name}.")
        setattr(utils, name, resolved)
    utils.RETURN_MESSAGE_NUM = radius
    utils.MESSAGE_TOP_K = message_top_k
    import server

    assert server.INDEX_PATH == utils.INDEX_PATH and server.META_PATH == utils.META_PATH
    import_ms = (time.perf_counter() - started) * 1000
    model_path = utils.resolve_embed_model_path()
    model_started = time.perf_counter()
    embedder = utils.get_embedder(model_path, utils.resolve_embed_device())
    model_load_ms = (time.perf_counter() - model_started) * 1000
    ingest_started = time.perf_counter()
    entries: list[dict[str, Any]] = []
    original_records = request["records"]
    seen: set[str] = set()
    for record in original_records:
        record_id = str(record["record_id"])
        if not record_id or record_id in seen or record_id.startswith("fmem-"):
            raise ValueError("Corpus record IDs must be unique public record IDs.")
        seen.add(record_id)
        messages = [
            {"message_id": str(item["message_id"]), "role": str(item["role"]), "content": str(item["content"])}
            for item in record["messages"]
        ]
        dialogue = utils.normalize_raw_dialogue({"schema_version": 1, "messages": messages})
        # Deliberately omit generated summaries, auto-tags, QA and answer fields.
        # The title and the original conversation are the only indexed inputs.
        entry = {
            "id": record_id, "memory_kind": "chat_event",
            "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00",
            "title": str(record.get("title") or record_id), "short_summary": "",
            "overview_summary": "", "problem_background": "", "analysis": "",
            "action_steps": "", "validation_result": "", "detailed_summary": "",
            "raw_dialogue": dialogue, "project_id": "", "tags": [], "source_paths": [],
            "reference_doc_path": "", "retrieval_fields": ["title"],
        }
        entries.append(entry)
        entries.extend(utils.build_field_record_entries(entry, dialogue_tokenizer=embedder.tokenizer, model_path=model_path))
    utils.ensure_sqlite_store_ready()
    with closing(utils.get_sqlite_connection()) as connection:
        utils.upsert_store_entries(entries, connection=connection)
        connection.commit()
    ingest_ms = (time.perf_counter() - ingest_started) * 1000
    index_started = time.perf_counter()
    index_stats = utils.rebuild_vector_index(entries=entries, embedder=embedder, model_path=model_path)
    build_stats = {
        "backend": "actual_memory_rag_mcp_bge", "model_path": model_path,
        "device": utils.resolve_embed_device(), "model_fingerprint": utils.build_embedding_model_fingerprint(model_path),
        "record_count": len(original_records),
        "message_count": sum(len(record["messages"]) for record in original_records),
        "index_vector_count": index_stats.get("count"), "embedding_dimension": index_stats.get("dim"),
        "window_radius": radius, "message_top_k": message_top_k,
        "message_score_gap": utils.MESSAGE_SCORE_GAP,
        "anchor_policy": "native cosine-gap ranking with position spacing",
        "chunk_max_tokens": utils.CHAT_EVENT_CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": utils.CHAT_EVENT_CHUNK_OVERLAP_TOKENS,
        "corpus_fields": "title and original message role/content; no QA, answers, generated summaries or tags",
        "runtime_dir": str(runtime),
        "timing": {"import_ms": import_ms, "model_load_ms": model_load_ms, "ingest_ms": ingest_ms,
                   "index_build_ms": (time.perf_counter() - index_started) * 1000,
                   "total_prepare_ms": (time.perf_counter() - started) * 1000},
    }
    return utils, server, embedder, build_stats


def _search(server: Any, embedder: Any, model_path: str, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    results = server.search_records(query=str(request["query"]), top_k=int(request.get("top_k", 5)),
                                    embedder=embedder, model_path=model_path)
    selected = {str(result["id"]): list(result.get("matched_message_ids") or []) for result in results}
    details = server.get_detail_records_by_ids([str(result["id"]) for result in results], matched_message_ids=selected)
    returned: list[dict[str, str]] = []
    for record in details["records"]:
        for message in record.get("raw_dialogue", {}).get("messages", []):
            if "message_id" in message:
                returned.append({"record_id": str(record["id"]), "message_id": str(message["message_id"]),
                                 "content": str(message["content"])})
    return {
        "ranked_record_ids": [str(result["id"]) for result in results],
        "scores": [float(result["score"]) for result in results],
        "selected_message_ids": selected, "returned_messages": returned,
        "latency_ms": (time.perf_counter() - started) * 1000, "runtime_status": "ok",
    }


def main() -> None:
    protocol_output = sys.stdout
    utils = server = embedder = None
    model_path = ""
    last_request_id = 0
    for line in sys.stdin:
        close_requested = False
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("A protocol request must be a JSON object.")
            request_id = request.get("request_id")
            if type(request_id) is not int or request_id <= last_request_id:
                raise ValueError("request_id must be a strictly increasing positive integer.")
            last_request_id = request_id
            with redirect_stdout(sys.stderr):
                operation = request.get("op")
                if operation == "init":
                    if server is not None:
                        raise RuntimeError("A project worker can only initialize once.")
                    utils, server, embedder, build_stats = _initialize(request)
                    model_path = build_stats["model_path"]
                    response = {"ok": True, "build_stats": build_stats}
                elif operation == "search":
                    if server is None:
                        raise RuntimeError("Project worker has not initialized.")
                    response = {"ok": True, "result": _search(server, embedder, model_path, request)}
                elif operation == "close":
                    response = {"ok": True}
                    close_requested = True
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        response["request_id"] = request_id
        protocol_output.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_output.flush()
        if close_requested:
            return


if __name__ == "__main__":
    main()
