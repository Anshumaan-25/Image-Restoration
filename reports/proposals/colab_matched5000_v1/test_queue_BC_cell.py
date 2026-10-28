"""Offline dispatch checks for the operator's queued continuation cell."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

CELL = (Path(__file__).parent / "queue_BC_cell.py").read_text()


class ContinuationTests(unittest.TestCase):
    """Use a fake launcher; no CUDA, training, Drive access or model changes."""

    def scope(self, launcher: Mock, results: dict | None = None) -> dict:
        """Build notebook globals without executing a real workload."""
        return {"APPROVED_LONG_RUN": True,
                "results": {"A": {"completed": True, "step": 5000}} if results is None else results,
                "run_arm": launcher, "project": Path("project"),
                "reviewed_plan": Path("reviewed_plan.json"), "backup_root": Path("Drive"), "json": json}

    def execute(self, scope: dict) -> str:
        """Execute the exact operator cell, retaining its visible messages."""
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(CELL, "queue_BC_cell.py", "exec"), scope)
        return output.getvalue()

    def test_success_runs_only_B_then_C(self) -> None:
        """The already-running A is never restarted."""
        launcher = Mock(side_effect=[{"completed": True, "step": 5000}] * 2)
        scope = self.scope(launcher)
        output = self.execute(scope)
        self.assertEqual([call.args[2] for call in launcher.call_args_list], ["B_seed42", "C_seed42"])
        self.assertTrue(all(call.kwargs == {"approved": True} for call in launcher.call_args_list))
        self.assertEqual(set(scope["results"]), {"A", "B", "C"})
        self.assertIn("ALL THREE ARMS COMPLETED", output)

    def test_failed_A_blocks_everything(self) -> None:
        """A failed/absent/incomplete result cannot launch another arm."""
        for results in ({}, {"A": {"completed": False}}):
            launcher = Mock()
            with self.assertRaises(RuntimeError):
                self.execute(self.scope(launcher, results))
            launcher.assert_not_called()

    def test_unapproved_or_duplicate_execution_is_blocked(self) -> None:
        """Repeated queue clicks cannot retrain completed arms."""
        launcher = Mock()
        scope = self.scope(launcher)
        scope["APPROVED_LONG_RUN"] = False
        with self.assertRaises(RuntimeError):
            self.execute(scope)
        scope = self.scope(launcher)
        scope["results"]["B"] = {"completed": True}
        with self.assertRaises(RuntimeError):
            self.execute(scope)
        launcher.assert_not_called()

    def test_B_exception_or_incomplete_blocks_C(self) -> None:
        """A wall-time stop, backup failure or GPU error never advances to C."""
        for effect in (RuntimeError("wall-time/backup failure"), {"completed": False}):
            launcher = Mock(side_effect=effect) if isinstance(effect, Exception) else Mock(return_value=effect)
            scope = self.scope(launcher)
            with self.assertRaises(RuntimeError):
                self.execute(scope)
            self.assertEqual(launcher.call_count, 1)
            self.assertEqual(launcher.call_args.args[2], "B_seed42")
            self.assertEqual(set(scope["results"]), {"A"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
