from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("concurrent_stdio_worker.py")
PYTHON = Path(sys.executable)


def _decode_tool_result(result: Any) -> dict[str, Any]:
    text_blocks = [
        str(getattr(block, "text", ""))
        for block in getattr(result, "content", [])
        if getattr(block, "type", "") == "text"
    ]
    text = text_blocks[-1] if text_blocks else ""
    if getattr(result, "isError", False):
        return {"_error": text}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"_error": f"MCP returned non-JSON text: {text}"}
    if not isinstance(decoded, dict):
        return {"_error": f"MCP returned non-object JSON: {decoded!r}"}
    return decoded


class StdioSessionPool:
    def __init__(self, root: Path, size: int, extra_env: dict[str, str] | None = None) -> None:
        self.root = root
        self.size = size
        self.extra_env = extra_env or {}
        self.stack = AsyncExitStack()
        self.sessions: list[ClientSession] = []
        self.worker_pids: list[int] = []
        self.sequence = 0
        self.pid_log_path = self.root / "worker-pids.log"
        self.operation_log_path = self.root / "operation-log.jsonl"
        self.stderr_sink = None

    async def __aenter__(self) -> "StdioSessionPool":
        previous_pid_lines = 0
        if self.pid_log_path.exists():
            previous_pid_lines = len(self.pid_log_path.read_text(encoding="ascii").splitlines())
        env = {
            **os.environ,
            "MEMORY_RAG_TEST_ROOT": str(self.root),
            "PYTHONPATH": str(ROOT),
            **self.extra_env,
        }
        self.stderr_sink = open(os.devnull, "w", encoding="utf-8")
        try:
            for _ in range(self.size):
                params = StdioServerParameters(
                    command=str(PYTHON),
                    args=["-u", str(WORKER)],
                    env=env,
                    cwd=str(ROOT),
                )
                streams = await self.stack.enter_async_context(stdio_client(params, errlog=self.stderr_sink))
                session = await self.stack.enter_async_context(
                    ClientSession(*streams, read_timeout_seconds=timedelta(seconds=20))
                )
                await session.initialize()
                self.sessions.append(session)
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.pid_log_path.exists():
                pid_lines = self.pid_log_path.read_text(encoding="ascii").splitlines()
                if len(pid_lines) >= previous_pid_lines + self.size:
                    self.worker_pids = [int(line) for line in pid_lines[previous_pid_lines:previous_pid_lines + self.size]]
                    break
            await asyncio.sleep(0.01)
        if len(self.worker_pids) != self.size:
            self.worker_pids = [-1] * self.size
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self.stack.aclose()
        finally:
            if self.stderr_sink is not None:
                self.stderr_sink.close()
                self.stderr_sink = None

    async def call(self, index: int, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        sequence = self.sequence
        started = time.monotonic()
        operation = {
            "pid": self.worker_pids[index] if index < len(self.worker_pids) else -1,
            "operation": tool_name,
            "request_sequence": sequence,
            "started_at": time.time(),
        }
        try:
            result = await self.sessions[index].call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(seconds=20),
            )
            decoded = _decode_tool_result(result)
            operation.update(
                {
                    "finished_at": time.time(),
                    "duration_seconds": time.monotonic() - started,
                    "result": decoded,
                }
            )
            return decoded
        except Exception as exc:
            operation.update(
                {
                    "finished_at": time.time(),
                    "duration_seconds": time.monotonic() - started,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            with self.operation_log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(json.dumps(operation, ensure_ascii=False) + "\n")

    async def call_parallel(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return await asyncio.gather(
            *(self.call(index, tool_name, arguments) for index, (tool_name, arguments) in enumerate(calls))
        )


class StdioSessionPoolTests(unittest.TestCase):
    def test_failed_initialization_closes_session_streams_and_log(self) -> None:
        closed = {"streams": False, "session": False}

        @asynccontextmanager
        async def fake_stdio(params, errlog):
            self.assertEqual(params.args, ["-u", str(WORKER)])
            try:
                yield (object(), object())
            finally:
                closed["streams"] = True

        class FailedSession:
            def __init__(self, *streams, read_timeout_seconds):
                self.timeout = read_timeout_seconds

            async def __aenter__(self):
                return self

            async def initialize(self):
                if self.timeout != timedelta(seconds=20):
                    raise AssertionError("initialization must have a read timeout")
                raise TimeoutError("initialization timed out")

            async def __aexit__(self, *exc):
                closed["session"] = True

        with tempfile.TemporaryDirectory() as directory:
            async def check() -> None:
                pool = StdioSessionPool(Path(directory), 1)
                with self.assertRaisesRegex(TimeoutError, "initialization timed out"):
                    async with pool:
                        self.fail("failed initialization entered the pool")
                self.assertIsNone(pool.stderr_sink)
            with patch.object(sys.modules[__name__], "stdio_client", fake_stdio), patch.object(sys.modules[__name__], "ClientSession", FailedSession):
                asyncio.run(check())
        self.assertEqual(closed, {"streams": True, "session": True})


class CrossProcessWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._seed()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _worker_command(self, mode: str, extra_env: dict[str, str] | None = None) -> list[str]:
        return [str(PYTHON), str(WORKER)]

    def _seed(self) -> None:
        env = {**os.environ, "MEMORY_RAG_TEST_ROOT": str(self.root), "MEMORY_RAG_WORKER_MODE": "seed"}
        completed = subprocess.run(
            self._worker_command("seed"),
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            self.fail(f"seed worker failed: {completed.stdout}\n{completed.stderr}")

    def _inspect(self) -> dict[str, Any]:
        env = {**os.environ, "MEMORY_RAG_TEST_ROOT": str(self.root), "MEMORY_RAG_WORKER_MODE": "inspect"}
        completed = subprocess.run(
            self._worker_command("inspect"),
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            self.fail(f"inspect worker failed: {completed.stdout}\n{completed.stderr}")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def _assert_index_consistent(self) -> None:
        inspected = self._inspect()
        self.assertEqual(inspected["ids"], inspected["meta_ids"])
        self.assertEqual(inspected["npy_shape"][0], len(inspected["ids"]))
        self.assertEqual(inspected["faiss_ntotal"], len(inspected["ids"]))
        self.assertEqual(inspected["npy_shape"][1], inspected["meta_dim"])
        self.assertEqual(inspected["orphan_field_record_count"], 0)
        self.assertEqual(inspected["state"]["state"], "idle")

    def _wait_for_idle(self, timeout_seconds: float = 15.0) -> None:
        state_path = self.root / "data" / "rebuild_state.json"
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("state") == "idle":
                    return
            time.sleep(0.05)
        self.fail(f"rebuild did not return to idle: {state_path.read_text(encoding='utf-8')}")

    async def _wait_for_idle_async(self, timeout_seconds: float = 15.0) -> None:
        state_path = self.root / "data" / "rebuild_state.json"
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("state") == "idle":
                    return
            await asyncio.sleep(0.05)
        self.fail(f"rebuild did not return to idle: {state_path.read_text(encoding='utf-8')}")

    def _run_batches(
        self,
        size: int,
        batches: list[list[tuple[str, dict[str, Any]]]],
        extra_env: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        async def runner() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            async with StdioSessionPool(self.root, size, extra_env=extra_env) as pool:
                for batch in batches:
                    results.extend(await pool.call_parallel(batch))
            return results

        return asyncio.run(runner())

    @staticmethod
    def _save_arguments(round_index: int, worker_index: int) -> dict[str, Any]:
        marker = f"cross-process-{round_index}-{worker_index}"
        return {
            "title": f"Concurrent {marker}",
            "short_summary": f"Concurrent save {marker}",
            "detailed_summary": f"Independent stdio worker wrote {marker}.",
            "tags": ["concurrency-test", marker],
        }

    def test_independent_stdio_saves_preserve_records_and_index(self) -> None:
        rounds = int(os.environ.get("MEMORY_RAG_CONCURRENCY_ROUNDS", "20"))
        results: list[dict[str, Any]] = []
        expected_new_records = 0
        for size in (2, 4, 8):
            batches = [
                [
                    ("save_fact", self._save_arguments(round_index, worker_index))
                    for worker_index in range(size)
                ]
                for round_index in range(rounds)
            ]
            results.extend(self._run_batches(size, batches))
            expected_new_records += rounds * size
        errors = [result for result in results if "_error" in result]
        self.assertFalse(errors, errors[:3])
        self.assertEqual(len({result["id"] for result in results}), expected_new_records)
        self._assert_index_consistent()

    def test_different_field_updates_are_serialized_and_merged(self) -> None:
        rounds = int(os.environ.get("MEMORY_RAG_CONCURRENCY_ROUNDS", "20"))
        batches = []
        expected_titles = set()
        expected_summaries = set()
        for round_index in range(rounds):
            title = f"title-update-{round_index}"
            summary = f"summary-update-{round_index}"
            expected_titles.add(title)
            expected_summaries.add(summary)
            batches.append(
                [
                    ("update", {"id": "fact-concurrency-seed", "changes": {"title": title}}),
                    ("update", {"id": "fact-concurrency-seed", "changes": {"short_summary": summary}}),
                ]
            )
        results = self._run_batches(2, batches)
        errors = [result for result in results if "_error" in result]
        self.assertFalse(errors, errors[:3])

        inspected = self._inspect()
        self.assertIn("fact-concurrency-seed", inspected["ids"])
        self._assert_index_consistent()

        async def read_detail() -> dict[str, Any]:
            async with StdioSessionPool(self.root, 1) as pool:
                return await pool.call(0, "get_details_by_ids", {"ids": ["fact-concurrency-seed"]})

        detail_result = asyncio.run(read_detail())
        record = detail_result["records"][0]
        self.assertIn(record["title"], expected_titles)
        self.assertIn(record["short_summary"], expected_summaries)

    def test_same_field_competition_keeps_a_complete_committed_value(self) -> None:
        rounds = int(os.environ.get("MEMORY_RAG_CONCURRENCY_ROUNDS", "20"))
        batches = []
        expected_titles = set()
        for round_index in range(rounds):
            first_title = f"same-field-first-{round_index}"
            second_title = f"same-field-second-{round_index}"
            expected_titles.update((first_title, second_title))
            batches.append(
                [
                    ("update", {"id": "fact-concurrency-seed", "changes": {"title": first_title}}),
                    ("update", {"id": "fact-concurrency-seed", "changes": {"title": second_title}}),
                ]
            )
        results = self._run_batches(2, batches)
        errors = [result for result in results if "_error" in result]
        self.assertFalse(errors, errors[:3])

        async def read_detail() -> dict[str, Any]:
            async with StdioSessionPool(self.root, 1) as pool:
                return await pool.call(0, "get_details_by_ids", {"ids": ["fact-concurrency-seed"]})

        detail_result = asyncio.run(read_detail())
        self.assertEqual(detail_result["missing_ids"], [])
        self.assertIn(detail_result["records"][0]["title"], expected_titles)
        self._assert_index_consistent()

    def test_save_update_delete_cross_competition_leaves_no_orphans(self) -> None:
        rounds = int(os.environ.get("MEMORY_RAG_CONCURRENCY_ROUNDS", "20"))
        results: list[dict[str, Any]] = []
        deleted_ids: list[str] = []
        for round_index in range(rounds):
            victim_result = self._run_batches(
                1,
                [[("save_fact", self._save_arguments(round_index, 99))]],
            )[0]
            self.assertNotIn("_error", victim_result, victim_result)
            victim_id = victim_result["id"]
            deleted_ids.append(victim_id)
            results.extend(
                self._run_batches(
                    3,
                    [[
                        ("save_fact", self._save_arguments(round_index, 100)),
                        ("update", {"id": "fact-concurrency-seed", "changes": {"title": f"cross-update-{round_index}"}}),
                        ("delete_by_ids", {"ids": [victim_id]}),
                    ]],
                )
            )
        errors = [result for result in results if "_error" in result]
        self.assertFalse(errors, errors[:3])
        inspected = self._inspect()
        self.assertEqual(inspected["orphan_field_record_count"], 0)
        self.assertTrue(all(victim_id not in inspected["ids"] for victim_id in deleted_ids))
        self._assert_index_consistent()

    def _save_while_database_locked(self, seconds: int) -> tuple[dict[str, Any], float]:
        ready = self.root / "lock-ready"
        env = {
            **os.environ,
            "MEMORY_RAG_TEST_ROOT": str(self.root),
            "MEMORY_RAG_WORKER_MODE": "hold_db_lock",
            "MEMORY_RAG_LOCK_SECONDS": str(seconds),
        }

        async def runner() -> tuple[dict[str, Any], float]:
            # Import/initialize the client before the lock window starts.
            async with StdioSessionPool(self.root, 1) as pool:
                holder = subprocess.Popen(
                    self._worker_command("hold_db_lock"), cwd=ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    deadline = time.monotonic() + 30
                    while not ready.exists() and time.monotonic() < deadline:
                        await asyncio.sleep(0.05)
                    self.assertTrue(ready.exists(), "lock holder did not acquire BEGIN IMMEDIATE")
                    started = time.monotonic()
                    result = await pool.call(0, "save_fact", self._save_arguments(0, 0))
                    elapsed = time.monotonic() - started
                finally:
                    try:
                        stdout, stderr = await asyncio.to_thread(holder.communicate, timeout=15)
                    except subprocess.TimeoutExpired:
                        holder.kill()
                        await asyncio.to_thread(holder.communicate, timeout=5)
                        raise
                    self.assertEqual(holder.returncode, 0, f"lock holder failed: {stdout}\n{stderr}")
                return result, elapsed

        return asyncio.run(asyncio.wait_for(runner(), timeout=90))

    def test_sqlite_lock_waits_then_fails_without_hanging(self) -> None:
        result, elapsed = self._save_while_database_locked(8)
        self.assertGreater(elapsed, 4)
        self.assertLess(elapsed, 15)
        self.assertIn("_error", result, result)
        self.assertRegex(result["_error"].lower(), "locked|timeout")

    def test_sqlite_lock_short_wait_succeeds(self) -> None:
        result, elapsed = self._save_while_database_locked(1)
        self.assertGreater(elapsed, 0.5)
        self.assertLess(elapsed, 15)
        self.assertNotIn("_error", result, result)

    def test_search_rejects_during_background_rebuild(self) -> None:
        env = {
            "MEMORY_RAG_REBUILD_DELAY": "2",
        }
        meta_path = self.root / "data" / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["retrieval_field_signature"] = "stale-test-signature"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        async def runner() -> tuple[dict[str, Any], dict[str, Any]]:
            async with StdioSessionPool(self.root, 2, extra_env=env) as pool:
                update_task = asyncio.create_task(
                    pool.call(0, "update", {"id": "fact-concurrency-seed", "changes": {"title": "rebuild trigger"}})
                )
                for _ in range(40):
                    await asyncio.sleep(0.05)
                    state_path = self.root / "data" / "rebuild_state.json"
                    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")).get("state") == "running":
                        break
                search_result = await pool.call(1, "search", {"query": "rebuild trigger"})
                update_result = await update_task
                await self._wait_for_idle_async()
                resumed = await pool.call(1, "search", {"query": "rebuild trigger"})
                self.assertNotIn("_error", resumed, resumed)
                self.assertTrue(resumed.get("results"), resumed)
                return update_result, search_result

        update_result, search_result = asyncio.run(runner())
        self.assertEqual(update_result.get("index_refresh_state"), "rebuilding")
        self.assertFalse(search_result.get("search_available", True))
        self.assertEqual(
            search_result.get("rebuild_state"),
            "running",
            f"{search_result}; state={self._inspect()['state']}; pids={(self.root / 'worker-pids.log').read_text(encoding='ascii')}",
        )
        self._assert_index_consistent()


if __name__ == "__main__":
    unittest.main()
