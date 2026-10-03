"""Read-only demo tasks. Expected answers stay outside model context."""

import json

from .evaluation import EvaluationCase, load_cases


def demo_agent_cases() -> list[EvaluationCase]:
    noise = "\n".join(f"INFO worker heartbeat sequence={i} status=healthy" for i in range(120))
    shutdown = "\n".join(f"INFO shutdown step={i} complete" for i in range(25))
    return load_cases(
        [
            {
                "name": "payment-timeout",
                "query": "Inspect the job log and retry configuration. Return JSON fields "
                "request_id (string), failure_code (string), timeout_seconds (integer), "
                "and max_attempts (integer).",
                "tools": {
                    "read_job_log": {
                        "kind": "log",
                        "description": "Read this payment job's log.",
                        "content": noise + "\nERROR request_id=pay-482 "
                        "failure_code=UPSTREAM_TIMEOUT timeout_seconds=30\n" + shutdown,
                    },
                    "read_retry_config": {
                        "kind": "json",
                        "description": "Read retry configuration.",
                        "content": json.dumps(
                            {
                                "max_attempts": 3,
                                "backoff": "exponential",
                                "idempotency": "reuse_original_key",
                            },
                            indent=4,
                        ),
                    },
                },
                "expected": {
                    "request_id": "pay-482",
                    "failure_code": "UPSTREAM_TIMEOUT",
                    "timeout_seconds": 30,
                    "max_attempts": 3,
                },
                "required_tools": ["read_job_log", "read_retry_config"],
            },
            {
                "name": "deployment-failure",
                "query": "Inspect the build log. Return JSON fields failed_test (string), "
                "exception (string), and deployment_allowed (boolean). "
                "A deployment is allowed only if all tests passed.",
                "tools": {
                    "read_build_log": {
                        "kind": "log",
                        "description": "Read the deployment's build and test log.",
                        "content": noise + "\nFAILED test_retry_keeps_original_key\n"
                        "Traceback (most recent call last):\n"
                        '  File "tests/test_retry.py", line 42, in test_retry_keeps_original_key\n'
                        "    assert retry.key == original.key\n"
                        "AssertionError: idempotency key changed\n" + shutdown,
                    }
                },
                "expected": {
                    "failed_test": "test_retry_keeps_original_key",
                    "exception": "AssertionError",
                    "deployment_allowed": False,
                },
                "required_tools": ["read_build_log"],
            },
            {
                "name": "healthy-control",
                "query": "Read job status. Return JSON fields job_id (string), "
                "status (string), and retry_needed (boolean). "
                "A successful job must not be retried.",
                "tools": {
                    "read_status": {
                        "kind": "json",
                        "description": "Read the job status.",
                        "content": '{"job_id":"job-19","status":"succeeded"}',
                    }
                },
                "expected": {"job_id": "job-19", "status": "succeeded", "retry_needed": False},
                "required_tools": ["read_status"],
            },
        ]
    )
