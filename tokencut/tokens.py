"""Counts text, not a provider's hidden chat framing or billed usage."""

from __future__ import annotations

import math
from typing import Any


class TokenCounter:
    def __init__(self, mode: str = "auto", encoding: str = "cl100k_base") -> None:
        if mode not in {"auto", "estimate", "tiktoken"}:
            raise ValueError("counter must be auto, estimate, or tiktoken")
        self._encoding: Any = None
        self.warning: str | None = None
        self.name = "estimate:utf8-bytes/4"
        if mode != "estimate":
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(encoding)
                self.name = f"tiktoken:{encoding}"
            except Exception as exc:
                if mode == "tiktoken":
                    raise ValueError(
                        f"Cannot load tokenizer {encoding!r}. Install tokencut-agent[tokens] "
                        "and ensure the encoding is cached or downloadable."
                    ) from exc
                self.warning = "Tokenizer unavailable; using approximate UTF-8 bytes / 4 counts."

    @property
    def approximate(self) -> bool:
        return self._encoding is None

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("TokenCounter.count expects text")
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return math.ceil(len(text.encode("utf-8")) / 4)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "approximate": self.approximate,
            "scope": "serialized text only; not provider-billed usage",
        }
