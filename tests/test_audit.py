import os
import tempfile
import unittest
from pathlib import Path

from tokencut.audit import audit, scan_python


class AuditTests(unittest.TestCase):
    def test_rules_and_lines(self):
        source = (
            "import json\nimport subprocess\n"
            "history = []\nfor step in range(10):\n"
            "    client.chat.completions.create(messages=history)\n"
            "data = path.read_text()\n"
            "payload = json.dumps(data, indent=4)\n"
            "subprocess.run(['test'], capture_output=True)\n"
        )
        findings = scan_python(source)
        self.assertEqual({f.rule for f in findings}, {"TC001", "TC002", "TC003", "TC004", "TC005"})
        self.assertEqual([f.line for f in findings if f.rule == "TC001"], [5])

    def test_strings_and_comments_not_treated_as_code(self):
        source = '# data = path.read_text()\nx = "json.dumps(data, indent=4)"\n'
        self.assertEqual(scan_python(source), [])

    def test_bounded_and_compact_patterns_are_not_flagged(self):
        source = "data = stream.read(200)\nvalue = json.dumps(data, separators=(',', ':'))\n"
        self.assertEqual(scan_python(source), [])

    def test_aliases_and_inline_suppression(self):
        source = (
            "from json import dumps as dump\nx = dump(data, indent=4)  # tokencut: ignore[TC004]\n"
        )
        self.assertEqual(scan_python(source), [])
        self.assertEqual(scan_python(source.split("  #")[0])[0].rule, "TC004")

    def test_string_cannot_suppress_rule(self):
        source = 'x = json.dumps({"note": "tokencut: ignore[TC004]"}, indent=4)\n'
        self.assertEqual(scan_python(source)[0].rule, "TC004")

    def test_syntax_error_visible(self):
        self.assertEqual(scan_python("def broken(")[0].rule, "TC000")

    def test_docstrings_not_reported_as_prompts(self):
        self.assertEqual(scan_python('"""' + "documentation " * 300 + '"""'), [])

    def test_large_string_is_reported(self):
        findings = scan_python('PROMPT = "' + "instruction " * 300 + '"')
        self.assertEqual(findings[0].rule, "TC006")

    def test_scan_does_not_execute_and_skips_dependencies_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text(
                "raise RuntimeError('DO NOT RUN')\nx = path.read_text()\n"
            )
            (root / "node_modules").mkdir()
            (root / "node_modules" / "ignored.py").write_text("x = path.read_text()")
            (root / "linked.py").symlink_to(root / "agent.py")
            result = audit(root)
            self.assertEqual(result.files_scanned, 1)
            self.assertEqual(len(result.findings), 1)
            self.assertEqual(len(result.skipped), 1)
            self.assertEqual(result.findings[0].path, "agent.py")
            self.assertEqual(audit(root, exclude=("agent.py",)).files_scanned, 0)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes require POSIX")
    def test_named_pipe_is_skipped_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root / "stream.py")
            result = audit(root)
            self.assertEqual(result.files_scanned, 0)
            self.assertEqual(len(result.skipped), 1)
            self.assertIn("not a regular file", result.skipped[0])

    def test_missing_path_is_error(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            audit(Path(directory) / "missing")

    def test_negative_and_none_read_sizes_are_unbounded(self):
        findings = scan_python("x = stream.read(-1)\ny = stream.read(None)\n")
        self.assertEqual([f.rule for f in findings], ["TC003", "TC003"])

    def test_comprehensions_are_loops(self):
        findings = scan_python("results = [client.create(messages=history) for item in tasks]")
        self.assertIn("TC002", {finding.rule for finding in findings})
