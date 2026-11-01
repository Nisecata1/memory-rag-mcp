"""Implement only the system under evaluation; use shared scoring."""


def create_retriever(records, *, project_dir, runtime_dir, window_radius, message_top_k, options):
    # records contain original corpus, never questions or gold labels.
    # Return an object with search(query, top_k), close(), and optional build_stats.
    # A separate database process can use benchlib.worker_client.JsonlWorkerClient
    # with this package's own worker.py and backend-specific init_payload.
    raise NotImplementedError("Implement this version's adapter before enabling it")
