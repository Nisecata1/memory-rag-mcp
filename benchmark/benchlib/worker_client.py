"""Shared fail-stop JSON-lines transport; storage logic belongs to version workers."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

class JsonlWorkerClient:
    """JSON-lines client for a fresh, isolated real-project worker."""

    def __init__(self, *, worker_path: Path, init_payload: dict[str, Any], runtime_dir: Path,
                 protected_data: Path | None = None) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._log = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._next_request_id = 1
        self._failure_reason: str | None = None
        self._request_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        runtime = Path(runtime_dir).resolve()
        production_data = Path(protected_data).resolve() if protected_data is not None else None
        allowed_runs = (Path(__file__).resolve().parents[1] / "runs").resolve()
        if runtime == allowed_runs or not runtime.is_relative_to(allowed_runs) or (production_data is not None and runtime.is_relative_to(production_data)):
            raise ValueError("Project runtime_dir must be a dedicated directory beneath benchmark/runs/.")
        if runtime.exists() and any(runtime.iterdir()):
            raise FileExistsError(f"Refusing to reuse a non-empty project runtime directory: {runtime}")
        runtime.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = runtime
        self._log = (runtime / "worker.log").open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
        worker = Path(worker_path).resolve()
        if not worker.is_file():
            self._log.close()
            raise FileNotFoundError(worker)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-u", "-B", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, encoding="utf-8", cwd=runtime, env=environment,
                creationflags=flags,
            )
        except BaseException:
            self.close()
            raise
        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()
        try:
            response = self._request({**init_payload, "op": "init", "runtime_dir": str(runtime)}, timeout=1800)
            self.build_stats = response["build_stats"]
        except BaseException:
            self.close()
            raise

    def _read_responses(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                self._responses.put(line)
        except (OSError, ValueError):
            # A concurrent fail-stop may close the pipe after terminating it.
            pass
        finally:
            self._responses.put(None)

    def _request(self, request: dict[str, Any], *, timeout: float = 180) -> dict[str, Any]:
        # A request and its response form one indivisible transaction.  A timed
        # out worker is permanently unusable: its eventual answer must never be
        # consumed and scored as the next query's answer.
        with self._request_lock:
            process = self._process
            if self._failure_reason is not None or process is None or process.stdin is None or process.poll() is not None:
                raise RuntimeError(f"Project worker is unavailable ({self._failure_reason or 'not running'}); inspect {self.runtime_dir / 'worker.log'}")
            request_id = self._next_request_id
            self._next_request_id += 1
            envelope = {**request, "request_id": request_id}
            try:
                process.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = self._responses.get(timeout=timeout)
                if line is None:
                    raise ValueError("Worker closed its response pipe.")
                response = json.loads(line)
                if not isinstance(response, dict):
                    raise ValueError("Worker response must be a JSON object.")
                if type(response.get("request_id")) is not int or response["request_id"] != request_id:
                    raise ValueError(f"Worker response request_id mismatch: expected {request_id}, got {response.get('request_id')!r}.")
                if type(response.get("ok")) is not bool:
                    raise ValueError("Worker response must contain a boolean ok field.")
                if response["ok"]:
                    payload_key = {"init": "build_stats", "search": "result"}.get(str(request.get("op")))
                    if payload_key is not None and not isinstance(response.get(payload_key), dict):
                        raise ValueError(f"Worker response is missing its {payload_key} object.")
            except queue.Empty as exc:
                self._stop_worker(f"request {request_id} timed out")
                raise TimeoutError(f"Project worker timed out and was stopped; inspect {self.runtime_dir / 'worker.log'}") from exc
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                self._stop_worker(f"request {request_id} communication failure: {exc}")
                raise RuntimeError(f"Project worker protocol failed and was stopped: {exc}; inspect {self.runtime_dir / 'worker.log'}") from exc
            if not response["ok"]:
                # A matching, well-formed application error is not a stale or
                # ambiguous transport response and can be reported normally.
                raise RuntimeError(f"Project worker failed: {response.get('error')}; inspect {self.runtime_dir / 'worker.log'}")
            return response

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._request({"op": "search", "query": str(query), "top_k": int(top_k)})
        result = response["result"]
        result["ipc_wall_ms"] = (time.perf_counter() - started) * 1000
        return result

    def _stop_worker(self, reason: str, *, graceful: bool = False) -> None:
        if self._failure_reason is None:
            self._failure_reason = reason
        process = self._process
        self._process = None
        if process is not None:
            if process.poll() is None:
                if graceful and process.stdin is not None:
                    # EOF closes the idle protocol loop.  Never recursively
                    # call _request here or consume another queued response.
                    try:
                        process.stdin.close()
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        reader = self._reader_thread
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        self._reader_thread = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self) -> None:
        with self._request_lock:
            self._stop_worker("closed", graceful=True)

    def __enter__(self) -> JsonlWorkerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
