"""Version-specific initialization for the BGE-M3 / SQLite project worker."""
from __future__ import annotations
from pathlib import Path

from benchlib.contracts import normalize_records
from benchlib.worker_client import JsonlWorkerClient


class ProjectRetriever(JsonlWorkerClient):
    def __init__(self, records, *, project_dir: Path, runtime_dir: Path,
                 window_radius: int = 10, message_top_k: int = 3) -> None:
        project_dir = Path(project_dir).resolve()
        super().__init__(
            worker_path=Path(__file__).with_name("worker.py"),
            runtime_dir=runtime_dir,
            protected_data=project_dir.parent / "memory-rag-mcp-data",
            init_payload={
                "project_dir": str(project_dir), "records": normalize_records(records),
                "window_radius": int(window_radius), "message_top_k": int(message_top_k),
            },
        )


def create_retriever(records, *, project_dir, runtime_dir, window_radius, message_top_k, options):
    if options:
        raise ValueError("Project v1 has no additional adapter options")
    return ProjectRetriever(records, project_dir=project_dir, runtime_dir=runtime_dir,
                            window_radius=window_radius, message_top_k=message_top_k)
