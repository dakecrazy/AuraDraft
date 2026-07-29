"""TokenIndexEngine — the core indexing and retrieval engine.

Three-layer token index:
  Layer 1 — Token inverted index (row-level postings, BM25 scoring)
  Layer 2 — Bigram index (deterministic composite keys, no hash())
  Layer 3 — Chunk token binary storage (struct-packed for compactness)

All token IDs are in the *canonical* tokenizer space.
"""

import json
import math
import struct
import uuid
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone

import numpy as np

from kb_agent.storage.db import Database
from kb_agent.tokenizer.canonical import CanonicalTokenizer

# BM25 parameters (standard values from Robertson & Zaragoza 2009)
K1 = 1.2
B = 0.75


class TokenIndexEngine:
    """Index and search documents using token-level inverted and bigram indices."""

    def __init__(
        self,
        db: Database,
        tokenizer: CanonicalTokenizer,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):
        self.db = db
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── public API ────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its index entries.

        Returns True if the document existed, False otherwise.
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT total_tokens FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM token_postings WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM bigram_postings WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunk_tokens WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return True

    def index_document(
        self,
        doc_id: str | None = None,
        text: str | None = None,
        file_path: str = "",
        category: str = "",
        tags: list[str] | None = None,
        summary: str = "",
    ) -> dict:
        """Index a document into the three-layer token index.

        If *doc_id* is None, one is auto-generated.
        If *text* is None, it is loaded from *file_path*.
        **Idempotent:** if *doc_id* already exists, the old index entries
        are deleted first (within the same transaction).

        Returns a summary dict with token counts.
        """
        if text is None:
            from kb_agent.document.loader import load_text

            text = load_text(file_path)

        doc_id = doc_id or uuid.uuid4().hex[:12]
        token_ids = self.tokenizer.encode(text)
        total_tokens = len(token_ids)
        chunks = self._chunk_tokens(token_ids)

        # ── collect all rows before inserting ──────────────────────
        posting_rows: list[tuple[int, str, int, int, int]] = []
        bigram_rows: list[tuple[int, int, str, int, int]] = []
        chunk_rows: list[tuple[str, str, int, bytes]] = []

        for chunk_idx, chunk_tokens in enumerate(chunks):
            chunk_id = f"{doc_id}_{chunk_idx}"

            # Layer 1: token postings
            for pos, tid in enumerate(chunk_tokens):
                posting_rows.append((tid, doc_id, chunk_idx, pos, 1))

            # Layer 2: bigram postings (deterministic composite key)
            for i in range(len(chunk_tokens) - 1):
                bigram_rows.append(
                    (chunk_tokens[i], chunk_tokens[i + 1], doc_id, chunk_idx, i)
                )

            # Layer 3: packed chunk token sequence
            packed = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
            chunk_rows.append((chunk_id, doc_id, chunk_idx, packed))

        # ── atomic insert (idempotent: delete-then-insert) ──────────
        with self.db.transaction() as conn:
            # Remove old entries for this doc_id if re-indexing
            conn.execute("DELETE FROM token_postings WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM bigram_postings WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunk_tokens WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            if posting_rows:
                conn.executemany(
                    "INSERT INTO token_postings "
                    "(token_id, doc_id, chunk_id, position, frequency) "
                    "VALUES (?, ?, ?, ?, ?)",
                    posting_rows,
                )
            if bigram_rows:
                conn.executemany(
                    "INSERT INTO bigram_postings "
                    "(t1, t2, doc_id, chunk_id, position) VALUES (?, ?, ?, ?, ?)",
                    bigram_rows,
                )
            if chunk_rows:
                conn.executemany(
                    "INSERT INTO chunk_tokens "
                    "(chunk_id, doc_id, chunk_idx, token_ids) VALUES (?, ?, ?, ?)",
                    chunk_rows,
                )
            conn.execute(
                "INSERT INTO documents "
                "(doc_id, file_path, category, total_tokens, chunk_count, tags, summary, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    file_path,
                    category,
                    total_tokens,
                    len(chunks),
                    json.dumps(tags or [], ensure_ascii=False),
                    summary,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        # Update avg_dl AFTER the document is committed
        self.db.update_avg_dl(total_tokens)

        return {
            "doc_id": doc_id,
            "total_tokens": total_tokens,
            "chunk_count": len(chunks),
            "unique_tokens": len(set(token_ids)),
        }

    # ── search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Search the index.

        Modes:
          "exact"   — Layer 1 only: BM25 on token inverted index
          "phrase"  — Layer 2 only: bigram phrase matching
          "hybrid"  — weighted combination (0.6 exact + 0.4 phrase)
        """
        query_tokens = self.tokenizer.encode(query)
        if not query_tokens:
            return []

        if mode == "exact":
            return self._search_exact(query_tokens, top_k)
        elif mode == "phrase":
            return self._search_phrase(query_tokens, top_k)
        else:
            return self._search_hybrid(query_tokens, top_k)

    # ── utility ───────────────────────────────────────────────────

    def get_chunk_text(self, chunk_id: str) -> str:
        """Reconstruct text from a stored chunk's packed token IDs."""
        conn = self.db.connect()
        row = conn.execute(
            "SELECT token_ids FROM chunk_tokens WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return ""
        blob = row["token_ids"]
        if len(blob) % 4 != 0:
            return ""  # corrupted blob
        token_ids = list(struct.unpack(f"{len(blob) // 4}I", blob))
        return self.tokenizer.decode(token_ids)

    def get_stats(self) -> dict:
        """Return index-level statistics."""
        conn = self.db.connect()
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM documents"
        ).fetchone()[0]
        unique_tokens = conn.execute(
            "SELECT COUNT(DISTINCT token_id) FROM token_postings"
        ).fetchone()[0]
        avg_dl = self.db.get_stat("avg_dl", 0.0)
        return {
            "total_documents": total_docs,
            "total_tokens_indexed": total_tokens,
            "unique_token_types": unique_tokens,
            "avg_document_length": round(avg_dl, 1),
        }

    # ── internal: chunking ────────────────────────────────────────

    def _chunk_tokens(self, token_ids: list[int]) -> list[list[int]]:
        """Split token sequence into overlapping chunks."""
        chunks: list[list[int]] = []
        start = 0
        while start < len(token_ids):
            end = start + self.chunk_size
            chunks.append(token_ids[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── internal: BM25 search ─────────────────────────────────────

    def _search_exact(
        self, query_tokens: list[int], top_k: int
    ) -> list[dict]:
        """Layer 1: BM25 scoring on token inverted index."""
        conn = self.db.connect()
        N = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if N == 0:
            return []
        avg_dl = self.db.get_stat("avg_dl", 1000.0)

        doc_scores: dict[str, float] = defaultdict(float)

        for tid in set(query_tokens):
            # Document frequency (number of docs containing this token)
            df_row = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) AS df FROM token_postings "
                "WHERE token_id = ?",
                (tid,),
            ).fetchone()
            df = df_row["df"] if df_row else 0
            if df == 0:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            # Fetch all postings for this token
            rows = conn.execute(
                "SELECT doc_id, COUNT(*) AS tf, "
                "MAX(documents.total_tokens) AS dl "
                "FROM token_postings "
                "JOIN documents USING (doc_id) "
                "WHERE token_id = ? "
                "GROUP BY doc_id",
                (tid,),
            ).fetchall()

            for row in rows:
                doc_id = row["doc_id"]
                tf = row["tf"]
                dl = row["dl"]
                score = idf * (tf * (K1 + 1)) / (
                    tf + K1 * (1 - B + B * dl / avg_dl)
                )
                doc_scores[doc_id] += score

        return self._format_results(doc_scores, top_k)

    def _search_phrase(
        self, query_tokens: list[int], top_k: int
    ) -> list[dict]:
        """Layer 2: bigram phrase matching."""
        conn = self.db.connect()
        doc_scores: dict[str, float] = defaultdict(float)

        for i in range(len(query_tokens) - 1):
            t1, t2 = query_tokens[i], query_tokens[i + 1]
            rows = conn.execute(
                "SELECT doc_id, COUNT(*) AS cnt "
                "FROM bigram_postings "
                "WHERE t1 = ? AND t2 = ? "
                "GROUP BY doc_id",
                (t1, t2),
            ).fetchall()
            for row in rows:
                doc_scores[row["doc_id"]] += row["cnt"]

        return self._format_results(doc_scores, top_k)

    def _search_hybrid(
        self, query_tokens: list[int], top_k: int
    ) -> list[dict]:
        """Hybrid: weighted combination of exact and phrase scores."""
        exact = self._search_exact(query_tokens, top_k * 2)
        phrase = self._search_phrase(query_tokens, top_k * 2)

        combined: dict[str, float] = defaultdict(float)
        for r in exact:
            combined[r["doc_id"]] += r["score"] * 0.6
        for r in phrase:
            combined[r["doc_id"]] += r["score"] * 0.4

        return self._format_results(combined, top_k)

    def _format_results(
        self, scores: dict[str, float], top_k: int
    ) -> list[dict]:
        """Sort by descending score and attach document metadata."""
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        conn = self.db.connect()
        results: list[dict] = []
        for doc_id, score in ranked:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if row:
                results.append(
                    {
                        "doc_id": row["doc_id"],
                        "file_path": row["file_path"],
                        "category": row["category"],
                        "total_tokens": row["total_tokens"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "summary": row["summary"],
                        "score": round(score, 4),
                    }
                )
        return results