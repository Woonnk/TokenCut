"""Optional model transport. Imported SDK and credentials are needed only for live runs."""

from __future__ import annotations

import os

from .evaluation import ModelCallError


class OpenAIChatClient:
    source = "live"
    max_retries = 0

    def __init__(self, timeout: float = 30) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Live evaluation requires OPENAI_API_KEY configured locally. "
                "Do not paste it into chat. Use --dry-run to inspect the evaluation without a key."
            )
        if not 0 < timeout <= 120:
            raise ValueError("timeout must be greater than 0 and at most 120 seconds")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError(
                "Install the live extra from the project: pip install -e '.[live]'"
            ) from exc
        # Disable hidden retries so failed/possibly-billed calls cannot disappear from measurements.
        self._client = OpenAI(
            api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout
        )

    def complete(self, request: dict) -> dict:
        try:
            response = self._client.chat.completions.create(**request)
            return response.model_dump(mode="json", exclude_none=True)
        except Exception as exc:
            code = {
                "AuthenticationError": "authentication_error",
                "PermissionDeniedError": "permission_error",
                "RateLimitError": "rate_limit_error",
                "APITimeoutError": "timeout",
                "APIConnectionError": "connection_error",
                "BadRequestError": "invalid_request",
                "NotFoundError": "invalid_request",
            }.get(type(exc).__name__, "model_call_failed")
            raise ModelCallError(code) from None

    def close(self) -> None:
        self._client.close()
