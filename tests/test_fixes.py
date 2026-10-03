import unittest

from tokencut.fixes import compact_json_preview


class FixTests(unittest.TestCase):
    def test_compact_json_preview_is_review_only(self):
        source = "import json\npayload = json.dumps(data, indent=4)\n"
        result = compact_json_preview(source, "agent.py")
        self.assertTrue(result.changed)
        self.assertEqual(source, "import json\npayload = json.dumps(data, indent=4)\n")
        self.assertIn("separators=(',', ':')", result.source)
        self.assertIn("agent.py (proposed)", result.diff)

    def test_multiline_and_existing_separators_are_skipped(self):
        source = (
            "import json\n"
            "one = json.dumps(data, separators=(',', ':'), indent=2)\n"
            "two = json.dumps(\n    data, indent=2\n)\n"
        )
        result = compact_json_preview(source)
        self.assertFalse(result.changed)
        self.assertEqual(result.source, source)
        self.assertEqual(result.skipped, 2)

    def test_invalid_source_does_not_produce_a_patch(self):
        with self.assertRaises(ValueError):
            compact_json_preview("def broken(")
