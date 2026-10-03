import copy
import unittest

from tokencut import TokenCounter, optimize_request


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter("estimate")
        self.request = {
            "model": "your-model",
            "metadata": {"tag": "keep"},
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "read_log"}}],
            "messages": [
                {"role": "system", "content": "Never change this."},
                {"role": "developer", "content": "Never change this either."},
                {"role": "user", "content": "Find the error."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "read_log", "arguments": '{"path":"run.log"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call1",
                    "content": "INFO routine\n" * 60 + "ERROR sentinel\n",
                },
                {"role": "assistant", "content": "My answer."},
            ],
        }

    def test_protocol_and_input_immutable(self):
        original = copy.deepcopy(self.request)
        result = optimize_request(
            self.request, tool_kinds={"read_log": "log"}, counter=self.counter
        )
        self.assertEqual(self.request, original)
        self.assertLess(result.report["after_tokens"], result.report["before_tokens"])
        self.assertIn("ERROR sentinel", result.request["messages"][4]["content"])
        normalized = copy.deepcopy(result.request)
        normalized["messages"][4]["content"] = original["messages"][4]["content"]
        self.assertEqual(normalized, original)

    def test_no_allowlist_is_noop(self):
        self.assertEqual(
            optimize_request(self.request, tool_kinds={}, counter=self.counter).request,
            self.request,
        )

    def test_multimodal_tool_output_is_untouched(self):
        self.request["messages"][4]["content"] = [{"type": "text", "text": "INFO routine\n" * 100}]
        result = optimize_request(
            self.request, tool_kinds={"read_log": "log"}, counter=self.counter
        )
        self.assertEqual(result.request, self.request)

    def test_json_allowlist(self):
        self.request["messages"][4]["content"] = '{\n  "answer": [1, 2, 3]\n}'
        result = optimize_request(
            self.request, tool_kinds={"read_log": "json"}, counter=self.counter
        )
        self.assertEqual(result.request["messages"][4]["content"], '{"answer":[1,2,3]}')

    def test_invalid_json_remains_unchanged(self):
        result = optimize_request(
            self.request, tool_kinds={"read_log": "json"}, counter=self.counter
        )
        self.assertEqual(result.request, self.request)

    def test_non_tool_messages_are_never_filtered(self):
        for message in self.request["messages"]:
            if message["role"] != "tool":
                message["content"] = "INFO routine\n" * 100
        result = optimize_request(
            self.request, tool_kinds={"read_log": "log"}, counter=self.counter
        )
        for i, message in enumerate(self.request["messages"]):
            if message["role"] != "tool":
                self.assertEqual(result.request["messages"][i], message)

    def test_bad_request_rejected(self):
        for request in [None, {}, {"messages": "bad"}, {"messages": [None]}]:
            with self.assertRaises(ValueError):
                optimize_request(request, tool_kinds={}, counter=self.counter)

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            optimize_request(self.request, tool_kinds={"read_log": "code"}, counter=self.counter)

    def test_unhashable_kind_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            optimize_request(self.request, tool_kinds={"read_log": []}, counter=self.counter)

    def test_malformed_optional_fields_are_not_used_for_classification(self):
        self.request["messages"][3]["tool_calls"] = 123
        self.request["messages"][4]["tool_call_id"] = []
        result = optimize_request(
            self.request, tool_kinds={"read_log": "log"}, counter=self.counter
        )
        self.assertEqual(result.request, self.request)

    def test_ambiguous_ids_do_not_trigger_filtering(self):
        calls = self.request["messages"][3]["tool_calls"]
        calls.append(copy.deepcopy(calls[0]))
        result = optimize_request(
            self.request, tool_kinds={"read_log": "log"}, counter=self.counter
        )
        self.assertEqual(result.request, self.request)

    def test_call_name_takes_precedence_over_conflicting_message_name(self):
        self.request["messages"][4]["name"] = "different_tool"
        result = optimize_request(
            self.request, tool_kinds={"different_tool": "log"}, counter=self.counter
        )
        self.assertEqual(result.request, self.request)
