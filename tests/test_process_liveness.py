from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import server_utils as utils


@unittest.skipUnless(os.name == "nt", "Windows process handle regression tests")
class ProcessLivenessTests(unittest.TestCase):
    def test_probe_never_signals_a_live_child(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", "import sys; sys.stdin.readline(); print('survived')"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            with patch.object(utils.os, "kill", side_effect=AssertionError("Windows probe must not signal")):
                for _ in range(3):
                    self.assertTrue(utils.is_process_alive(process.pid))
                    self.assertIsNone(process.poll())
            stdout, stderr = process.communicate("\n", timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stdout.strip(), "survived")
        finally:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)

    def test_exited_child_and_invalid_pid_are_not_alive(self) -> None:
        with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
            process.wait(timeout=10)
            self.assertFalse(utils.is_process_alive(process.pid))
        self.assertFalse(utils.is_process_alive(0))
        self.assertFalse(utils.is_process_alive(-1))

    def test_unknown_probe_failure_is_conservative_and_handles_are_closed(self) -> None:
        api = MagicMock()
        api.OpenProcess.return_value = None
        with patch.object(ctypes, "WinDLL", return_value=api), patch.object(ctypes, "get_last_error") as error:
            error.return_value = 5
            self.assertTrue(utils.is_process_alive(123))
            error.return_value = 87
            self.assertFalse(utils.is_process_alive(123))
            api.CloseHandle.assert_not_called()
            api.OpenProcess.return_value = 456
            for wait_result, expected in ((0, False), (258, True), (0xFFFFFFFF, True)):
                api.WaitForSingleObject.return_value = wait_result
                self.assertIs(utils.is_process_alive(123), expected)
            self.assertEqual(api.CloseHandle.call_count, 3)
            api.OpenProcess.assert_called_with(0x00100000, False, 123)


if __name__ == "__main__":
    unittest.main()
