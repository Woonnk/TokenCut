import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tokencut.bench import benchmark
from tokencut.cli import main
from tokencut.tokens import TokenCounter


class CliTests(unittest.TestCase):
    def invoke(self, args, stdin=""):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(stdin)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_profile_from_stdin(self):
        code, output, _ = self.invoke(
            ["profile", "-", "--counter", "estimate", "--format", "json"], "abcd"
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["tokens"], 1)

    def test_optimizer_output_and_report_separate(self):
        data = {"query": "x", "chunks": [{"id": "a", "kind": "json", "text": '{\n  "a": 1\n}'}]}
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            diff = Path(directory) / "change.diff"
            code, output, errors = self.invoke(
                [
                    "optimize",
                    "-",
                    "--counter",
                    "estimate",
                    "--report",
                    str(report),
                    "--diff",
                    str(diff),
                ],
                json.dumps(data),
            )
            self.assertEqual(code, 0, errors)
            self.assertEqual(json.loads(output)["chunks"][0]["text"], '{"a":1}')
            self.assertIn("before_tokens", json.loads(report.read_text()))
            self.assertEqual(
                json.loads(report.read_text())["after_tokens"],
                TokenCounter("estimate").count(output),
            )
            self.assertIn("before.json", diff.read_text())
            self.assertIn("TokenCut:", errors)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("existing")
            args = ["optimize", "-", "--counter", "estimate", "-o", str(path)]
            code, _, errors = self.invoke(args, '{"query":"x"}')
            self.assertEqual(code, 2)
            self.assertIn("Output exists", errors)
            self.assertEqual(path.read_text(), "existing")
            code, _, _ = self.invoke([*args, "--force"], '{"query":"x"}')
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(path.read_text())["query"], "x")

    def test_budget_failure_emits_no_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            report = Path(directory) / "report.json"
            code, stdout, _ = self.invoke(
                [
                    "optimize",
                    "-",
                    "--counter",
                    "estimate",
                    "--budget",
                    "1",
                    "-o",
                    str(output),
                    "--report",
                    str(report),
                ],
                '{"query":"must keep this"}',
            )
            self.assertEqual(code, 3)
            self.assertEqual(stdout, "")
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_paths_cannot_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "same.json")
            code, _, _ = self.invoke(
                ["optimize", "-", "--counter", "estimate", "-o", path, "--report", path],
                '{"query":"x"}',
            )
            self.assertEqual(code, 2)

    def test_input_cannot_be_overwritten_even_with_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"query":"x"}')
            code, _, _ = self.invoke(
                ["optimize", str(path), "--counter", "estimate", "-o", str(path), "--force"]
            )
            self.assertEqual(code, 2)

    def test_malformed_input_has_clean_error(self):
        for text in ['{"query":"x","query":"lost"}', '{"query":', '{"query":NaN}']:
            code, output, error = self.invoke(["optimize", "-", "--counter", "estimate"], text)
            self.assertEqual(code, 2)
            self.assertEqual(output, "")
            self.assertNotIn("Traceback", error)

    def test_request_cli(self):
        request = {"messages": [{"role": "tool", "name": "data", "content": '{\n    "a": 1\n}'}]}
        code, output, _ = self.invoke(
            ["request", "-", "--tool", "data=json", "--counter", "estimate"], json.dumps(request)
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["messages"][0]["content"], '{"a":1}')

    def test_audit_ci_failure_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.py"
            path.write_text("client.create(messages=history)\n")
            code, output, _ = self.invoke(
                ["audit", str(path), "--format", "json", "--fail-on", "warning"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output)["findings"][0]["rule"], "TC001")

    def test_repository_map_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.py"
            path.write_text("import json\ndef run(): pass\n")
            code, output, _ = self.invoke(["map", str(path), "--format", "json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["modules"][0]["symbols"], ["run"])

    def test_typescript_audit_and_fix_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "agent.ts"
            original = "const output = JSON.stringify(data, null, 2);\n"
            file.write_text(original)
            code, output, _ = self.invoke(["audit", str(file), "--format", "json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["findings"][0]["rule"], "TC004")
            code, diff, _ = self.invoke(["fix", str(file)])
            self.assertEqual(code, 0)
            self.assertIn("JSON.stringify(data)", diff)
            self.assertEqual(file.read_text(), original)

    def test_dry_run_uses_agreed_savings_target(self):
        code, output, _ = self.invoke(["eval", "--dry-run", "--format", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["target_savings_percent"], 30)

    def test_benchmark_checks(self):
        result = benchmark(TokenCounter("estimate"), runs=1)
        self.assertTrue(result["all_checks_passed"])
        self.assertFalse(result["quality_verified"])
        self.assertEqual(result["cases"][-1]["savings_percent"], 0)

    def test_failed_benchmark_returns_nonzero(self):
        suite = [{"name": "missing", "packet": {"query": "x"}, "must_keep": ["absent"]}]
        code, output, _ = self.invoke(
            ["bench", "--suite", "-", "--runs", "1", "--format", "json", "--counter", "estimate"],
            json.dumps(suite),
        )
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output)["all_checks_passed"])

    def test_module_entrypoint(self):
        process = subprocess.run(
            [sys.executable, "-m", "tokencut", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("TokenCut 0.5.0", process.stdout)


if __name__ == "__main__":
    unittest.main()
