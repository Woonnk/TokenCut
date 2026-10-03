import json
import random
import unittest

from tokencut import BudgetExceeded, Chunk, ContextPacket, TokenCounter, optimize
from tokencut.core import rank_chunks
from tokencut.filters import compact_json, trim_log


class TokenTests(unittest.TestCase):
    def test_estimate_is_labeled_and_handles_unicode(self):
        counter = TokenCounter("estimate")
        self.assertTrue(counter.metadata()["approximate"])
        self.assertEqual(counter.count(""), 0)
        self.assertEqual(counter.count("abcd"), 1)
        self.assertEqual(counter.count("\U0001f680"), 1)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            TokenCounter("unknown")

    def test_optional_real_counter(self):
        try:
            import tiktoken
        except ImportError:
            self.skipTest("Optional tiktoken is not installed")
        counter = TokenCounter("tiktoken")
        text = "hello world <|endoftext|> \u65e5\u672c\u8a9e"
        expected = len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
        self.assertEqual(counter.count(text), expected)
        self.assertFalse(counter.approximate)

    def test_exact_counter_budget_invariants(self):
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("Optional tiktoken is not installed")
        counter = TokenCounter("tiktoken")
        rng = random.Random(123)
        for _ in range(25):
            chunks = tuple(
                Chunk(
                    f"c{i}",
                    rng.choice(["alpha retry ", "fonts ", "\u65e5\u672c\u8a9e"])
                    * rng.randint(1, 20),
                    pinned=i == 0,
                )
                for i in range(8)
            )
            packet = ContextPacket("retry", chunks, ("Keep me.",))
            minimum = counter.count(
                ContextPacket(packet.query, chunks[:1], packet.instructions).render()
            )
            budget = minimum + rng.randint(0, 200)
            result = optimize(packet, budget=budget, counter=counter)
            self.assertLessEqual(counter.count(result.render()), budget)
            self.assertIn(chunks[0], result.packet.chunks)


class ModelTests(unittest.TestCase):
    def test_round_trip(self):
        packet = ContextPacket("question", [Chunk("a", "value", source="docs/a")], ["be careful"])
        self.assertEqual(ContextPacket.from_dict(json.loads(packet.render())), packet)

    def test_unknown_fields_are_not_silently_dropped(self):
        for value in [
            {"query": "x", "messages": []},
            {"query": "x", "chunks": [{"id": "a", "text": "x", "secret_constraint": "keep"}]},
        ]:
            with self.assertRaises(ValueError):
                ContextPacket.from_dict(value)

    def test_bad_types_and_duplicate_ids(self):
        for value in [
            None,
            [],
            {"query": 4},
            {"query": "x", "instructions": "bad"},
            {"query": "x", "instructions": [False]},
            {"query": "x", "chunks": [{"id": "a", "text": "x", "pinned": "false"}]},
            {"query": "x", "chunks": [{"id": "a", "text": "x", "kind": []}]},
            {"query": "x", "chunks": [{"id": "a", "text": "x"}] * 2},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContextPacket.from_dict(value)


class FilterTests(unittest.TestCase):
    def test_json_preserves_lexemes_order_duplicates_and_strings(self):
        text = ' { "x" : 1.000000000000000000001, "x":1e999, "s":"a \\" b \\\\ c" } '
        self.assertEqual(
            compact_json(text), '{"x":1.000000000000000000001,"x":1e999,"s":"a \\" b \\\\ c"}'
        )

    def test_json_unicode_round_trip(self):
        data = {"value": "\u65e5\u672c\u8a9e", "nested": [{"a": 3}, None, True, False]}
        self.assertEqual(json.loads(compact_json(json.dumps(data, indent=4))), data)

    def test_bad_json_rejected(self):
        for text in ['{"broken":', '{"a": NaN}', '{"a": Infinity}', '{"a": 1} trailing']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                compact_json(text)

    def test_log_keeps_errors_neighbors_and_tail(self):
        lines = [f"INFO heartbeat {i}" for i in range(100)]
        lines[40] = "ERROR request 123 failed"
        result = trim_log("\n".join(lines))
        for line in [lines[0], lines[38], lines[40], lines[42], lines[-1]]:
            self.assertIn(line, result.text)
        self.assertNotIn("heartbeat 70\n", result.text)
        self.assertIn("TokenCut omitted", result.text)
        self.assertGreater(result.removed_lines, 0)

    def test_full_long_traceback_is_preserved(self):
        stack = "Traceback (most recent call last):\n" + "  frame in module.py\n" * 30
        stack += "ValueError: sentinel\n"
        text = "INFO normal\n" * 40 + stack + "INFO normal\n" * 40
        self.assertIn(stack, trim_log(text).text)

    def test_log_line_endings_and_noop_are_preserved(self):
        text = "ERROR one\r\n  details\r\nERROR two\r\n"
        self.assertEqual(trim_log(text).text, text)
        self.assertEqual(trim_log("").text, "")

    def test_duplicate_log_events_are_not_collapsed(self):
        text = "ERROR repeated request\n" * 40
        self.assertEqual(trim_log(text).text, text)

    def test_invalid_log_options(self):
        for kwargs in [{"context": -1}, {"tail": -1}, {"tail": True}]:
            with self.assertRaises(ValueError):
                trim_log("abc", **kwargs)


class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter("estimate")

    def test_protected_context_is_byte_for_byte_unchanged(self):
        packet = ContextPacket(
            "DO NOT CHANGE",
            (
                Chunk("system-copy", ' { "secret": "never reveal" } ', "json", True),
                Chunk("code", "def f():\n    return  42\n", "code", True),
            ),
            ("Keep every instruction exactly.",),
        )
        result = optimize(packet, counter=self.counter)
        self.assertEqual(result.packet, packet)
        self.assertEqual(result.report["risk"], "none")

    def test_dedupe_respects_provenance_and_pinned_duplicates(self):
        packet = ContextPacket(
            "x",
            (
                Chunk("a", "same " * 100, source="a"),
                Chunk("b", "same " * 100, source="a"),
                Chunk("c", "same " * 100, source="b"),
                Chunk("d", "same " * 100, pinned=True, source="a"),
                Chunk("e", "same " * 100, pinned=True, source="a"),
            ),
        )
        result = optimize(packet, counter=self.counter)
        self.assertEqual([c.id for c in result.packet.chunks], ["c", "d", "e"])
        self.assertTrue(all(not item["lossy"] for item in result.report["changes"]))
        self.assertEqual(optimize(packet, counter=self.counter, deduplicate=False).packet, packet)

    def test_invalid_json_is_kept(self):
        packet = ContextPacket("x", (Chunk("data", "not json", "json"),))
        self.assertEqual(optimize(packet, counter=self.counter).packet, packet)

    def test_empty_packet(self):
        packet = ContextPacket("")
        self.assertEqual(optimize(packet, counter=self.counter).packet, packet)

    def test_budget_refuses_to_remove_protected_context(self):
        packet = ContextPacket("question", (Chunk("required", "must keep " * 100, pinned=True),))
        with self.assertRaises(BudgetExceeded) as raised:
            optimize(packet, 20, self.counter)
        self.assertGreater(raised.exception.required, 20)

    def test_error_log_cannot_be_dropped_for_budget(self):
        packet = ContextPacket("unrelated", (Chunk("log", "ERROR diagnostic\n" * 100, "log"),))
        with self.assertRaises(BudgetExceeded):
            optimize(packet, 20, self.counter)

    def test_budget_uses_full_serialized_payload_and_restores_order(self):
        chunks = [
            Chunk("irrelevant", "web fonts " * 100),
            Chunk("payment", "payment retry idempotency"),
            Chunk("rule", "keep API", pinned=True),
        ]
        packet = ContextPacket("payment retry", chunks, ("Don't leak secrets.",))
        expected = ContextPacket(packet.query, chunks[1:], packet.instructions)
        budget = self.counter.count(expected.render())
        result = optimize(packet, budget, self.counter)
        self.assertEqual(result.packet, expected)
        self.assertLessEqual(self.counter.count(result.render()), budget)
        self.assertTrue(result.report["budget_met"])
        self.assertEqual(result.report["risk"], "lossy")
        self.assertFalse(result.report["quality_verified"])

    def test_no_log_filter(self):
        packet = ContextPacket("x", (Chunk("log", "INFO running\n" * 100, "log"),))
        self.assertEqual(optimize(packet, counter=self.counter, filter_logs=False).packet, packet)

    def test_diff_and_report(self):
        packet = ContextPacket("x", (Chunk("data", '{\n    "a": 1\n}', "json"),))
        result = optimize(packet, counter=self.counter)
        self.assertIn("--- before.json", result.diff())
        self.assertEqual(result.report["after_tokens"], self.counter.count(result.render()))
        self.assertTrue(result.report["counter"]["approximate"])

    def test_keyword_retrieval_is_stable(self):
        chunks = [Chunk("a", "fonts"), Chunk("b", "retry idempotency"), Chunk("c", "fonts")]
        self.assertEqual([c.id for c in rank_chunks("retry", chunks)], ["b", "a", "c"])
        self.assertEqual(rank_chunks("", chunks), chunks)

    def test_invalid_budget(self):
        for budget in [0, -1, True, 1.5, "100"]:
            with self.assertRaises(ValueError):
                optimize(ContextPacket("x"), budget, self.counter)

    def test_randomized_invariants(self):
        rng = random.Random(42)
        for trial in range(100):
            chunks = tuple(
                Chunk(
                    f"c{i}",
                    rng.choice(["retry", "fonts", "alpha"]) * rng.randint(1, 60),
                    pinned=i == 0,
                )
                for i in range(rng.randint(1, 10))
            )
            packet = ContextPacket("retry", chunks, ("Never drop this.",))
            minimum = self.counter.count(
                ContextPacket(packet.query, chunks[:1], packet.instructions).render()
            )
            budget = rng.randint(minimum, minimum + 300)
            result = optimize(packet, budget, self.counter)
            with self.subTest(trial=trial):
                self.assertEqual(result.packet.query, packet.query)
                self.assertEqual(result.packet.instructions, packet.instructions)
                self.assertIn(chunks[0], result.packet.chunks)
                self.assertLessEqual(result.report["after_tokens"], budget)
                self.assertLessEqual(result.report["after_tokens"], result.report["before_tokens"])


if __name__ == "__main__":
    unittest.main()
