import os
import tempfile
import unittest
from pathlib import Path

from tokencut.repository_map import build_repository_map


class RepositoryMapTests(unittest.TestCase):
    def test_map_includes_top_level_symbols_and_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text(
                "import json\nfrom pkg.tools import run\n\n"
                "def execute(): pass\n\nclass Agent:\n    async def plan(self): pass\n"
            )
            result = build_repository_map(root)
            self.assertEqual(result.modules[0].imports, ("json", "pkg.tools"))
            self.assertEqual(result.modules[0].symbols, ("execute", "Agent", "Agent.plan"))
            self.assertIn("Agent.plan", result.render())

    def test_map_skips_pipe_and_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.py").write_text("raise RuntimeError('do not run')\n")
            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "stream.py")
            result = build_repository_map(root)
            self.assertEqual(result.modules[0].path, "safe.py")
            if hasattr(os, "mkfifo"):
                self.assertIn("not a regular file", result.skipped[0])

    def test_invalid_path_and_file_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_repository_map(Path(directory) / "missing")
            with self.assertRaises(ValueError):
                build_repository_map(directory, max_files=0)
