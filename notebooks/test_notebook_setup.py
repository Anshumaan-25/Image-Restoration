"""Regression checks for visible subprocess failures in the Colab notebook."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any


class NotebookSetupTests(unittest.TestCase):
    """Execute the notebook's actual checked helper without running notebook cells."""

    def load_helper(self, project: Path) -> Any:
        """Compile only the checked function with a temporary working directory."""
        notebook = json.loads((Path(__file__).parent / "free_gpu.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("def checked(*args):"))
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "checked")
        scope = {"subprocess": subprocess, "sys": sys, "project": project}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_checked", "exec"), scope)
        return scope["checked"]

    def test_success_prints_and_retains_child_output(self) -> None:
        """Successful stdout and stderr both appear in the cell output."""
        with tempfile.TemporaryDirectory() as tmp:
            checked = self.load_helper(Path(tmp))
            output = io.StringIO()
            with redirect_stdout(output):
                result = checked("-c", "import sys; print('success'); print('warning', file=sys.stderr)")
            self.assertEqual(result.returncode, 0)
            self.assertIn("success", output.getvalue())
            self.assertIn("warning", output.getvalue())
            self.assertEqual(result.stdout, output.getvalue())

    def test_failure_prints_cause_and_keeps_nonzero_exit(self) -> None:
        """A failure remains fatal and includes its original error text."""
        with tempfile.TemporaryDirectory() as tmp:
            checked = self.load_helper(Path(tmp))
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as raised:
                checked("-c", "import sys; print('underlying error', file=sys.stderr); sys.exit(7)")
            self.assertEqual(raised.exception.returncode, 7)
            self.assertIn("underlying error", output.getvalue())
            self.assertEqual(raised.exception.output, output.getvalue())

    def test_bootstrap_and_qa_initialize_reports(self) -> None:
        """Either entry point creates reports safely, including an existing folder."""
        notebook = json.loads((Path(__file__).parent / "free_gpu.ipynb").read_text())
        for marker in ("os.chdir(project)", "def checked(*args):"):
            source = next("".join(cell["source"]) for cell in notebook["cells"]
                          if cell["cell_type"] == "code" and marker in "".join(cell["source"]))
            statements = [node for node in ast.parse(source).body
                          if isinstance(node, ast.Expr) and "reports" in ast.unparse(node)]
            self.assertEqual(len(statements), 1)
            with tempfile.TemporaryDirectory() as tmp:
                scope = {"project": Path(tmp)}
                code = compile(ast.Module(body=statements, type_ignores=[]), "reports_init", "exec")
                exec(code, scope)
                exec(code, scope)
                self.assertTrue((Path(tmp) / "reports").is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
