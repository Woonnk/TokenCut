import os
import tempfile
import unittest
from pathlib import Path

from tokencut.audit import audit
from tokencut.fixes import compact_js_json_preview
from tokencut.javascript import inspect_module, inspect_waste
from tokencut.repository_map import build_repository_map


class JavaScriptTests(unittest.TestCase):
    def test_module_index_and_audit(self):
        source = (
            'import {readFileSync} from "node:fs";\n'
            'const api = require("openai");\n'
            'export function run() {}\n'
            'export class Agent {\n  async plan() {}\n}\n'
            'for (let n=0; n<3; n++) {\n'
            '  client.chat.completions.create({messages: history});\n}\n'
            'const raw = readFileSync("logs.txt");\n'
            'const payload = JSON.stringify(raw, null, 2);\n'
        )
        imports, symbols = inspect_module(source)
        self.assertEqual(imports, ("node:fs", "openai"))
        self.assertEqual(symbols, ("run", "Agent", "Agent.plan"))
        self.assertEqual(
            inspect_waste(source),
            [(8, "TC001"), (8, "TC002"), (10, "TC003"), (11, "TC004")],
        )

    def test_strings_comments_and_templates_not_scanned(self):
        source = (
            '// JSON.stringify(data, null, 2)\n'
            'const note = "client.create({messages:history})";\n'
            'const template = `readFileSync("secret")`;\n'
            'const rx = /JSON.stringify(data,null,2)/;\n'
        )
        self.assertEqual(inspect_waste(source), [])

    def test_unrelated_history_object_does_not_trigger_model_history_finding(self):
        source = "client.create({input: task}); const other = {messages: history};\n"
        self.assertNotIn("TC001", [rule for _, rule in inspect_waste(source)])

    def test_js_ts_tree_and_safety_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.ts").write_text("export const run = async () => 1;\n")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "unused.js").write_text("JSON.stringify(x,null,2)")
            (root / "linked.js").symlink_to(root / "main.ts")
            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "pipe.js")
            mapping = build_repository_map(root)
            self.assertEqual([module.path for module in mapping.modules], ["main.ts"])
            self.assertEqual(mapping.modules[0].symbols, ("run",))
            result = audit(root)
            self.assertEqual(result.files_scanned, 1)
            self.assertEqual(result.findings, ())
            self.assertTrue(any("linked.js" in item for item in result.skipped))

    def test_javascript_preview_only_changes_simple_call(self):
        source = (
            'const note = "JSON.stringify(x,null,2)";\n'
            'const payload = JSON.stringify(data, null, 2);\n'
        )
        result = compact_js_json_preview(source, "agent.ts")
        self.assertEqual(result.source, source.replace("data, null, 2", "data"))
        self.assertTrue(result.changed)
        self.assertIn("agent.ts (proposed)", result.diff)

    def test_preview_skips_nested_and_dynamic_arguments(self):
        source = (
            "const a = JSON.stringify(load(), null, 2);\n"
            "const b = JSON.stringify(data, replacer, 2);\n"
        )
        self.assertFalse(compact_js_json_preview(source).changed)

    def test_extensions_reject_unknown_single_file(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "data.txt"
            file.write_text("hello")
            with self.assertRaises(ValueError):
                audit(file)
            with self.assertRaises(ValueError):
                build_repository_map(file)
