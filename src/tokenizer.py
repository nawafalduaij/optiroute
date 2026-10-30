"""Real token accounting via tiktoken (same encoder for Arabic and English)."""
from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _encoder(encoding: str):
    return tiktoken.get_encoding(encoding)


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(_encoder(encoding).encode(text, disallowed_special=()))


def count_chat_tokens(messages: list[dict[str, str]], encoding: str = "o200k_base") -> int:
    """Chat-format token count: content tokens + a fixed 4-token per-message envelope + 3 priming."""
    return sum(count_tokens(m["content"], encoding) + 4 for m in messages) + 3
