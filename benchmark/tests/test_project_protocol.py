"""Protocol failure tests use no project imports, models, or subprocesses."""

from __future__ import annotations

import io
import json
from pathlib import Path
import queue
import threading
import unittest
from unittest.mock import patch

from versions.project_v1 import worker as project_worker
from benchlib.retrievers import ProjectRetriever


class RecordingInput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    def write(self, text: str) -> int:
        self.sent.append(text)
        return super().write(text)


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = RecordingInput()
        self.stdout = io.StringIO()
        self.returncode = None
        self.terminations = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminations += 1
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def client_with_fake_process() -> tuple[ProjectRetriever, FakeProcess]:
    client = object.__new__(ProjectRetriever)
    process = FakeProcess()
    client._process = process
    client._log = None
    client._responses = queue.Queue()
    client._next_request_id = 1
    client._failure_reason = None
    client._request_lock = threading.Lock()
    client._reader_thread = None
    client.runtime_dir = Path("fake-isolated-runtime")
    return client, process


class ProjectProtocolTests(unittest.TestCase):
    def test_late_answer_after_timeout_cannot_be_used_for_next_query(self) -> None:
        client, process = client_with_fake_process()
        with self.assertRaises(TimeoutError):
            client._request({"op": "search", "query": "first"}, timeout=0.001)
        self.assertEqual(process.terminations, 1)
        client._responses.put(json.dumps({"request_id": 1, "ok": True, "result": {"query": "first"}}))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            client.search("second")
        self.assertEqual(len(process.stdin.sent), 1)
        self.assertEqual(client._responses.qsize(), 1)
        client.close()
        self.assertEqual(len(process.stdin.sent), 1)

    def test_mismatched_response_fails_stop(self) -> None:
        client, process = client_with_fake_process()
        client._responses.put(json.dumps({"request_id": 99, "ok": True, "result": {}}))
        with self.assertRaisesRegex(RuntimeError, "request_id mismatch"):
            client.search("first")
        self.assertEqual(process.terminations, 1)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            client.search("second")

    def test_invalid_json_eof_and_invalid_envelope_fail_stop(self) -> None:
        for response in ("not json", None, "[]", '{"request_id": 1, "ok": true}',
                         '{"request_id": true, "ok": true, "result": {}}'):
            with self.subTest(response=response):
                client, process = client_with_fake_process()
                client._responses.put(response)
                with self.assertRaisesRegex(RuntimeError, "protocol failed"):
                    client.search("first")
                self.assertEqual(process.terminations, 1)
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    client.search("second")

    def test_successful_transactions_have_increasing_ids(self) -> None:
        client, process = client_with_fake_process()
        for request_id in (1, 2):
            client._responses.put(json.dumps({"request_id": request_id, "ok": True, "result": {"query_id": request_id}}))
            result = client.search(f"query {request_id}")
            self.assertEqual(result["query_id"], request_id)
        self.assertEqual([json.loads(line)["request_id"] for line in process.stdin.sent], [1, 2])
        client.close()
        self.assertEqual(len(process.stdin.sent), 2)

    def test_worker_echoes_request_id_without_loading_model(self) -> None:
        lines = [
            {"request_id": 1, "op": "init"},
            {"request_id": 2, "op": "search", "query": "test"},
            {"request_id": 3, "op": "close"},
        ]
        source = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
        output = io.StringIO()
        with patch.object(project_worker, "_initialize", return_value=(None, object(), object(), {"model_path": "fake"})), \
             patch.object(project_worker, "_search", return_value={"ranked_record_ids": []}), \
             patch.object(project_worker.sys, "stdin", source), patch.object(project_worker.sys, "stdout", output):
            project_worker.main()
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([response["request_id"] for response in responses], [1, 2, 3])
        self.assertTrue(all(response["ok"] for response in responses))


if __name__ == "__main__":
    unittest.main()
