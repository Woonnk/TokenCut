import copy
import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tokencut.cli import main
from tokencut.demo_agent import demo_agent_cases
from tokencut.evaluation import (
    EvaluationCase,
    evaluate,
    grade,
    load_cases,
    plan_evaluation,
    read_usage,
    run_agent,
)
from tokencut.providers import OpenAIChatClient
from tokencut.tokens import TokenCounter


def sample_case(name="task"):
    return EvaluationCase.from_dict(
        {
            "name": name,
            "query": "Read the log and return JSON with request_id and allowed.",
            "tools": {
                "read_log": {
                    "kind": "log",
                    "description": "Read the job log.",
                    "content": "INFO heartbeat\n" * 100 + "ERROR request_id=secret-42\n",
                }
            },
            "expected": {"request_id": "secret-42", "allowed": False},
            "required_tools": ["read_log"],
        }
    )


def response(message, prompt=100, completion=10, finish="stop", model="fixed-model"):
    return {
        "model": model,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


def tool_response(call_id="call1", name="read_log", arguments="{}"):
    return response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        finish="tool_calls",
    )


def answer_response(prompt=1000, correct=True):
    return response(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "request_id": "secret-42" if correct else "wrong",
                    "allowed": False,
                }
            ),
        },
        prompt=prompt,
    )


class ScriptedClient:
    source = "test-double"
    max_retries = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    def close(self):
        self.closed = True


class ValidationTests(unittest.TestCase):
    def test_builtin_cases_have_separate_expectations(self):
        cases = demo_agent_cases()
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(case.expected and case.required_tools for case in cases))

    def test_invalid_suites_and_unknown_fields_rejected(self):
        for value in [None, [], {}, [None], [{"name": "x", "extra": "hidden"}]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_cases(value)

    def test_duplicate_case_names_rejected(self):
        data = {"name": "same", "query": "q", "tools": sample_case().tools, "expected": {"a": 1}}
        with self.assertRaises(ValueError):
            load_cases([data, data])

    def test_invalid_tool_definitions(self):
        data = {"name": "x", "query": "q", "tools": sample_case().tools, "expected": {"a": 1}}
        for name, tool in [
            ("../bad", data["tools"]["read_log"]),
            ("x", {}),
            ("x", {"kind": [], "description": "", "content": ""}),
        ]:
            with self.assertRaises(ValueError):
                EvaluationCase.from_dict({**data, "tools": {name: tool}})

    def test_missing_or_invalid_required_tools(self):
        data = {"name": "x", "query": "q", "tools": sample_case().tools, "expected": {"a": 1}}
        for required in ["read_log", ["absent"], [[]]]:
            with self.assertRaises(ValueError):
                EvaluationCase.from_dict({**data, "required_tools": required})

    def test_plan_limits_and_no_usage_claims(self):
        plan = plan_evaluation([sample_case()], model=None, repeats=3, max_steps=4)
        self.assertEqual(plan["maximum_model_calls"], 24)
        self.assertEqual(plan["live_model_calls"], 0)
        self.assertNotIn("total_token_savings_percent", plan)
        for options in [
            {"repeats": 0},
            {"max_steps": True},
            {"max_completion_tokens": -1},
            {"target_savings": float("nan")},
            {"target_savings": 101},
        ]:
            with self.assertRaises(ValueError):
                plan_evaluation([sample_case()], model=None, **options)

    def test_grader_requires_valid_json_types_and_tool_use(self):
        case = sample_case()
        valid = '{"request_id":"secret-42","allowed":false}'
        self.assertTrue(all(check["passed"] for check in grade(valid, case, {"read_log"})))
        for answer in [
            None,
            "```json\n" + valid + "\n```",
            '{"request_id":"secret-42","allowed":0}',
            '{"request_id":"secret-42","allowed":true,"allowed":false}',
            '{"request_id":"secret-42","allowed":NaN}',
        ]:
            self.assertFalse(all(check["passed"] for check in grade(answer, case, {"read_log"})))
        self.assertFalse(all(check["passed"] for check in grade(valid, case, set())))

    def test_usage_does_not_double_count_reasoning_or_cached_tokens(self):
        usage = read_usage(response({"role": "assistant"}, prompt=100, completion=20))
        self.assertEqual(usage["total_tokens"], 120)
        self.assertEqual(usage["reasoning_output_tokens"], 4)
        self.assertEqual(usage["cached_input_tokens"], 20)

    def test_missing_and_malformed_usage_stays_unknown(self):
        for usage in [
            None,
            {},
            {"prompt_tokens": True, "completion_tokens": 5},
            {"prompt_tokens": -2, "completion_tokens": 5},
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99},
        ]:
            self.assertIsNone(read_usage({"usage": usage}))
        usage = read_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        self.assertEqual(usage["total_tokens"], 15)
        self.assertIsNone(usage["cached_input_tokens"])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter("estimate")

    def run_case(self, client, optimized=True, **options):
        return run_agent(
            sample_case(),
            client,
            model="fixed-model",
            optimized=optimized,
            counter=self.counter,
            **options,
        )

    def test_agent_uses_tools_preserves_protocol_and_hides_answer_key(self):
        client = ScriptedClient([tool_response(), answer_response()])
        result = self.run_case(client)
        self.assertTrue(result["passed"])
        self.assertEqual(result["usage"]["total_tokens"], 1120)
        self.assertEqual(result["model_calls"], 2)
        self.assertNotIn("secret-42", json.dumps(client.requests[0]))
        self.assertNotIn("expected", json.dumps(client.requests))
        request = client.requests[1]
        self.assertEqual(request["messages"][0], client.requests[0]["messages"][0])
        self.assertEqual(request["messages"][1], client.requests[0]["messages"][1])
        self.assertEqual(request["messages"][-1]["tool_call_id"], "call1")
        self.assertIn("ERROR request_id=secret-42", request["messages"][-1]["content"])
        self.assertIn("TokenCut omitted", request["messages"][-1]["content"])
        self.assertFalse(request["store"])
        self.assertNotIn("answer", result)
        self.assertNotIn("messages", result)

    def test_baseline_keeps_full_tool_output(self):
        client = ScriptedClient([tool_response(), answer_response()])
        result = self.run_case(client, optimized=False)
        self.assertEqual(
            client.requests[1]["messages"][-1]["content"],
            sample_case().tools["read_log"]["content"],
        )
        self.assertEqual(result["steps"][1]["removed_request_text_tokens"], 0)

    def test_multitool_responses_are_paired(self):
        first = tool_response()
        first["choices"][0]["message"]["tool_calls"].append(
            tool_response("call2")["choices"][0]["message"]["tool_calls"][0]
        )
        client = ScriptedClient([first, answer_response()])
        result = self.run_case(client)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [m["tool_call_id"] for m in client.requests[1]["messages"] if m["role"] == "tool"],
            ["call1", "call2"],
        )

    def test_duplicate_call_ids_fail_without_executing_tools(self):
        first = tool_response()
        calls = first["choices"][0]["message"]["tool_calls"]
        calls.append(copy.deepcopy(calls[0]))
        result = self.run_case(ScriptedClient([first]))
        self.assertEqual(result["status"], "invalid_tool_calls")
        self.assertEqual(result["tool_calls"], 0)
        self.assertFalse(result["passed"])

    def test_unknown_tools_and_arguments_never_execute_code(self):
        for first in [
            tool_response(name="run_shell"),
            tool_response(arguments='{"command":"bad"}'),
        ]:
            client = ScriptedClient([first, answer_response()])
            result = self.run_case(client)
            self.assertEqual(result["tool_errors"], 1)
            self.assertFalse(result["passed"])
            self.assertIn("Unknown tool", client.requests[1]["messages"][-1]["content"])

    def test_step_limit_stops_loops_and_counts_every_response(self):
        result = self.run_case(
            ScriptedClient([tool_response(), tool_response("call2")]), max_steps=2
        )
        self.assertEqual(result["status"], "step_limit")
        self.assertFalse(result["passed"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 220)

    def test_provider_error_retains_prior_usage_without_exposing_exception(self):
        result = self.run_case(
            ScriptedClient([tool_response(), RuntimeError("sk-secret-token private prompt")])
        )
        self.assertEqual(result["status"], "provider_error")
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["usage"])
        self.assertEqual(result["known_usage_lower_bound"]["total_tokens"], 110)
        self.assertNotIn("sk-secret-token", json.dumps(result))

    def test_truncated_answer_is_not_a_success(self):
        last = answer_response()
        last["choices"][0]["finish_reason"] = "length"
        result = self.run_case(ScriptedClient([tool_response(), last]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "incomplete")

    def test_content_is_opt_in(self):
        result = self.run_case(
            ScriptedClient([tool_response(), answer_response()]), include_content=True
        )
        self.assertIn("secret-42", result["answer"])
        self.assertIn("messages", result)


class ComparisonTests(unittest.TestCase):
    def evaluate_case(self, responses, source="test-double", **options):
        client = ScriptedClient(responses)
        client.source = source
        return evaluate(
            [sample_case()],
            client,
            model="fixed-model",
            counter=TokenCounter("estimate"),
            **options,
        )

    def test_test_doubles_never_become_live_savings_claims(self):
        result = self.evaluate_case(
            [tool_response(), answer_response(2000), tool_response(), answer_response(200)]
        )
        self.assertTrue(result["task_checks_passed"])
        self.assertEqual(result["source"], "test-double")
        self.assertFalse(result["gate_passed"])
        self.assertIsNone(result["total_token_savings_percent"])

    def test_live_gate_requires_quality_and_total_savings(self):
        result = self.evaluate_case(
            [tool_response(), answer_response(2000), tool_response(), answer_response(200)],
            source="live",
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["summary"]["baseline"]["usage"]["total_tokens"], 2120)
        self.assertEqual(result["summary"]["optimized"]["usage"]["total_tokens"], 320)
        self.assertEqual(result["model_calls"], 4)
        self.assertGreater(result["total_token_savings_percent"], 50)

    def test_wrong_answer_blocks_gate_even_when_tokens_drop(self):
        result = self.evaluate_case(
            [
                tool_response(),
                answer_response(2000),
                tool_response(),
                answer_response(100, correct=False),
            ],
            source="live",
        )
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["quality_regressions"], [{"case": "task", "repeat": 1}])
        self.assertIsNone(result["summary"]["optimized"]["tokens_per_successful_task"])

    def test_extra_output_can_erase_input_savings(self):
        last = answer_response(100)
        last["usage"] = {"prompt_tokens": 100, "completion_tokens": 3000, "total_tokens": 3100}
        result = self.evaluate_case(
            [tool_response(), answer_response(1000), tool_response(), last], source="live"
        )
        self.assertLess(result["total_token_savings_percent"], 0)
        self.assertFalse(result["gate_passed"])

    def test_missing_usage_disables_savings_instead_of_becoming_zero(self):
        last = answer_response(100)
        last.pop("usage")
        result = self.evaluate_case(
            [tool_response(), answer_response(2000), tool_response(), last], source="live"
        )
        self.assertIsNone(result["total_token_savings_percent"])
        self.assertIsNone(result["summary"]["optimized"]["usage"])
        self.assertFalse(result["gate_passed"])

    def test_model_drift_blocks_gate(self):
        last = answer_response(100)
        last["model"] = "different-model"
        result = self.evaluate_case(
            [tool_response(), answer_response(2000), tool_response(), last], source="live"
        )
        self.assertFalse(result["model_identity_consistent"])
        self.assertFalse(result["gate_passed"])

    def test_repeats_alternate_order_and_reset_messages(self):
        client = ScriptedClient([tool_response(), answer_response()] * 4)
        result = evaluate(
            [sample_case()],
            client,
            model="fixed-model",
            counter=TokenCounter("estimate"),
            repeats=2,
        )
        self.assertEqual(
            [run["variant"] for run in result["runs"]],
            ["baseline", "optimized", "optimized", "baseline"],
        )
        self.assertEqual([len(client.requests[i]["messages"]) for i in [0, 2, 4, 6]], [2] * 4)

    def test_unaccounted_transport_retries_disable_usage_claims(self):
        client = ScriptedClient(
            [tool_response(), answer_response(2000), tool_response(), answer_response(100)]
        )
        client.source = "live"
        client.max_retries = 2
        result = evaluate(
            [sample_case()], client, model="fixed-model", counter=TokenCounter("estimate")
        )
        self.assertIsNone(result["total_token_savings_percent"])
        self.assertIsNone(result["summary"]["baseline"]["usage"])
        self.assertFalse(result["gate_passed"])


class TransportAndCliTests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_never_creates_a_client(self):
        with patch("tokencut.providers.OpenAIChatClient") as client:
            code, output, _ = self.invoke(
                ["eval", "--dry-run", "--counter", "estimate", "--format", "json"]
            )
        client.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["live_model_calls"], 0)

    def test_missing_model_and_key_are_actionable_errors(self):
        code, _, error = self.invoke(["eval", "--counter", "estimate"])
        self.assertEqual(code, 2)
        self.assertIn("--model", error)
        with patch.dict(os.environ, {}, clear=True):
            code, output, error = self.invoke(
                ["eval", "--model", "chosen-model", "--counter", "estimate"]
            )
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("OPENAI_API_KEY", error)
        self.assertNotIn("Traceback", error)

    def test_provider_disables_retries_and_does_not_accept_ambient_endpoint(self):
        module = types.ModuleType("openai")
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "https://wrong.example"},
            ),
            patch.dict("sys.modules", {"openai": module}),
        ):
            with patch.object(module, "OpenAI", create=True) as sdk:
                client = OpenAIChatClient(timeout=15)
                self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)
                self.assertEqual(sdk.call_args.kwargs["timeout"], 15)
                self.assertEqual(sdk.call_args.kwargs["base_url"], "https://api.openai.com/v1")
                client.close()
                sdk.return_value.close.assert_called_once()

    def test_custom_suite_and_report_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "name": "custom",
                            "query": "Question",
                            "tools": sample_case().tools,
                            "expected": {"answer": 1},
                        }
                    ]
                )
            )
            output = Path(directory) / "plan.json"
            code, _, error = self.invoke(
                [
                    "eval",
                    "--suite",
                    str(path),
                    "--dry-run",
                    "--counter",
                    "estimate",
                    "--format",
                    "json",
                    "-o",
                    str(output),
                ]
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output.read_text())["cases"], ["custom"])
            code, _, _ = self.invoke(
                [
                    "eval",
                    "--suite",
                    str(path),
                    "--dry-run",
                    "--counter",
                    "estimate",
                    "-o",
                    str(path),
                    "--force",
                ]
            )
            self.assertEqual(code, 2)

    def test_actual_sdk_serializes_the_tool_loop_without_network(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("Optional live SDK is not installed")
        requests = []
        responses = [tool_response(), answer_response()]

        def handler(request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 0,
                    **responses.pop(0),
                },
            )

        def factory(**kwargs):
            return OpenAI(
                **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
            )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}),
            patch("openai.OpenAI", side_effect=factory),
        ):
            client = OpenAIChatClient()
            client.source = "test-double"
            try:
                result = run_agent(
                    sample_case(),
                    client,
                    model="fixed-model",
                    optimized=True,
                    counter=TokenCounter("estimate"),
                )
            finally:
                client.close()
        self.assertTrue(result["passed"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["max_completion_tokens"], 1024)
        self.assertIn("TokenCut omitted", requests[1]["messages"][-1]["content"])

    def test_actual_sdk_does_not_retry_server_errors(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("Optional live SDK is not installed")
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                500, json={"error": {"message": "do not echo credentials", "type": "server_error"}}
            )

        def factory(**kwargs):
            return OpenAI(
                **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
            )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}),
            patch("openai.OpenAI", side_effect=factory),
        ):
            client = OpenAIChatClient()
            client.source = "test-double"
            try:
                result = run_agent(
                    sample_case(),
                    client,
                    model="fixed-model",
                    optimized=True,
                    counter=TokenCounter("estimate"),
                )
            finally:
                client.close()
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["status"], "provider_error")
        self.assertEqual(result["steps"][0]["error_code"], "model_call_failed")
        self.assertNotIn("echo credentials", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
