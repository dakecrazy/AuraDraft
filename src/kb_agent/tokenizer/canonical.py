"""Canonical tokenizer — thin wrapper around tiktoken.

All indexing uses this single tokenizer as the canonical token space.
Tokenizer adapters (for other models) are layered on top in M3+.
"""

import os
import tiktoken
from pathlib import Path


# Ensure tiktoken caches BPE files under the project directory
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "tiktoken"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE_DIR))


class CanonicalTokenizer:
    """Single canonical tokenizer for the entire knowledge base index.

    Defaults to cl100k_base (GPT-4 / GPT-3.5), which is the most widely
    compatible BPE encoding.  All documents, regardless of source model,
    are re-encoded into this space at ingestion time.
    """

    def __init__(self, encoding_name: str = "o200k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name

    # ── public API ────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Encode text into canonical token IDs.

        disallowed_special=() — treat tiktoken special tokens (e.g. the literal
        ``<|endoftext|>`` that appears in ML/AI papers) as plain text. Without
        this, any document containing such a literal crashes with a
        "disallowed special token" error.
        """
        return self._enc.encode(text, disallowed_special=())

    def encode_truncated(self, text: str, max_tokens: int = 8000) -> list[int]:
        """Encode and truncate to *max_tokens* tokens."""
        ids = self._enc.encode(text)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._enc.decode(token_ids)

    def decode_topk(self, token_ids: list[int], k: int = 10) -> str:
        """Decode the first *k* token IDs (useful for cluster label hints)."""
        return self._enc.decode(token_ids[:k])

    def count(self, text: str) -> int:
        """Return the number of tokens *text* would produce."""
        return len(self._enc.encode(text))

    # ── properties ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def __repr__(self) -> str:
        return f"CanonicalTokenizer({self._name}, vocab={self.vocab_size})"