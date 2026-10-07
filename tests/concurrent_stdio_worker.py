"""Isolated stdio MCP worker used by cross-process concurrency tests."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
import server_utils as utils


class DeterministicEmbedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = []
        for text in texts:
            raw = str(text).encode("utf-8")
            values = [(raw[index % len(raw)] if raw else 1) + index + 1 for index in range(8)]
            vectors.append(values)
        return np.asarray(vectors, dtype="float32")


def configure_runtime() -> Path:
    root = Path(os.environ["MEMORY_RAG_TEST_ROOT"]).resolve()
    data_dir = root / "data"
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    paths = {
        "AI_MEMORY_DIR": root,
        "DATA_DIR": data_dir,
        "TIMELINE_PATH": data_dir / "timeline.md",
        "EMBEDDINGS_PATH": data_dir / "memory_embeddings.npy",
        "INDEX_PATH": data_dir / "memory.faiss",
        "META_PATH": data_dir / "meta.json",
        "REBUILD_STATE_PATH": data_dir / "rebuild_state.json",
        "WRITE_LOCK_PATH": data_dir / ".memory-rag-write.lock",
        "SQLITE_PATH": root / "memory.db",
        "EMBED_MODEL_PATH": model_dir,
    }
    for name, value in paths.items():
        setattr(utils, name, value)
    with (root / "worker-pids.log").open("a", encoding="ascii") as log_handle:
        log_handle.write(f"{os.getpid()}\n")
    server.INDEX_PATH = utils.INDEX_PATH
    server.META_PATH = utils.META_PATH
    embedder = DeterministicEmbedder()
    utils.get_embedder = lambda *_args, **_kwargs: embedder
    server.get_embedder = lambda *_args, **_kwargs: embedder
    rebuild_delay = float(os.environ.get("MEMORY_RAG_REBUILD_DELAY", "0"))
    if rebuild_delay > 0:
        original_rebuild = utils.rebuild_vector_index

        def delayed_rebuild(*args, **kwargs):
            time.sleep(rebuild_delay)
            return original_rebuild(*args, **kwargs)

        utils.rebuild_vector_index = delayed_rebuild
    return root


def seed_database() -> None:
    root = configure_runtime()
    utils.ensure_sqlite_store_ready()
    payload = utils.build_payload(
        memory_kind="fact",
        title="concurrency seed",
        short_summary="seed summary",
        detailed_summary="seed details",
        tags=["concurrency-seed"],
        validate_reference_doc_path=False,
    )
    entry = {
        "id": "fact-concurrency-seed",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        **payload,
    }
    entries = [entry, *utils.build_field_record_entries(entry)]
    with utils.get_sqlite_connection() as connection:
        utils.upsert_store_entries(entries, connection=connection)
        connection.commit()
    stats = utils.rebuild_vector_index(
        entries=entries,
        embedder=DeterministicEmbedder(),
        model_path=str(root / "model"),
    )
    utils.write_rebuild_state(
        utils.build_rebuild_state_snapshot(
            state="idle",
            finished_at=utils.now_iso(),
            target_entry_count=int(stats.get("count") or 0),
            worker_pid=os.getpid(),
        )
    )


def inspect_database() -> None:
    configure_runtime()
    with utils.META_PATH.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with utils.EMBEDDINGS_PATH.open("rb") as handle:
        embeddings = np.load(handle, allow_pickle=False)
    index = faiss.read_index(str(utils.INDEX_PATH))
    with utils.get_sqlite_connection() as connection:
        field_record_count = int(connection.execute("SELECT COUNT(*) FROM field_records").fetchone()[0])
        orphan_field_record_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM field_records AS fields
                LEFT JOIN memory_registry AS registry ON registry.id = fields.source_memory_id
                WHERE registry.id IS NULL
                """
            ).fetchone()[0]
        )
    result = {
        "ids": utils.fetch_ordered_store_entry_ids(),
        "meta_ids": meta.get("row_to_id", []),
        "npy_shape": list(embeddings.shape),
        "faiss_ntotal": int(index.ntotal),
        "meta_dim": int(meta.get("dim") or 0),
        "field_record_count": field_record_count,
        "orphan_field_record_count": orphan_field_record_count,
        "state": utils.load_rebuild_state(),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


def hold_database_lock() -> None:
    root = configure_runtime()
    utils.ensure_sqlite_store_ready()
    seconds = float(os.environ.get("MEMORY_RAG_LOCK_SECONDS", "1"))
    ready_path = root / "lock-ready"
    with utils.get_sqlite_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        ready_path.write_text("ready", encoding="ascii")
        time.sleep(seconds)
        connection.rollback()


def main() -> None:
    mode = os.environ.get("MEMORY_RAG_WORKER_MODE", "stdio")
    if mode == "seed":
        seed_database()
        return
    if mode == "inspect":
        inspect_database()
        return
    if mode == "hold_db_lock":
        hold_database_lock()
        return
    configure_runtime()
    server.main()


if __name__ == "__main__":
    main()
