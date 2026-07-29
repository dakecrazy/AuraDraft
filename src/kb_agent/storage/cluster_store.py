"""ClusterStore — persistence layer for KnowledgeCluster and token doc-frequency.

Shares the same Database instance as TokenIndexEngine (single SQLite file).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from kb_agent.cluster.model import KnowledgeCluster

if TYPE_CHECKING:
    from kb_agent.storage.db import Database


class ClusterStore:
    """Thin persistence for clusters and token-level document frequency."""

    def __init__(self, db: Database):
        self.db = db

    # ── cluster CRUD ──────────────────────────────────────────────

    def save_cluster(self, cluster: KnowledgeCluster) -> None:
        """INSERT OR REPLACE a cluster into the clusters table."""
        conn = self.db.connect()
        conn.execute(
            "INSERT OR REPLACE INTO clusters "
            "(cluster_id, label, centroid, member_docs, knowledge_card, doc_count, created_at, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cluster.cluster_id,
                cluster.label,
                json.dumps(cluster.centroid, ensure_ascii=False),
                json.dumps(cluster.member_doc_ids, ensure_ascii=False),
                cluster.knowledge_card,
                cluster.doc_count,
                cluster.created_at,
                cluster.last_updated,
            ),
        )
        conn.commit()

    def load_all_clusters(self) -> list[KnowledgeCluster]:
        """Load every cluster from the database."""
        conn = self.db.connect()
        rows = conn.execute("SELECT * FROM clusters").fetchall()
        clusters: list[KnowledgeCluster] = []
        for row in rows:
            clusters.append(
                KnowledgeCluster(
                    cluster_id=row["cluster_id"],
                    centroid={
                        int(k): v
                        for k, v in json.loads(row["centroid"]).items()
                    },
                    member_doc_ids=json.loads(row["member_docs"]),
                    label=row["label"] or "",
                    knowledge_card=row["knowledge_card"] or "",
                    doc_count=row["doc_count"],
                    created_at=row["created_at"] or "",
                    last_updated=row["last_updated"] or "",
                )
            )
        return clusters

    def delete_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster. Returns True if it existed."""
        conn = self.db.connect()
        cursor = conn.execute(
            "DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── token document frequency ──────────────────────────────────

    def update_doc_freq(self, token_ids: list[int]) -> None:
        """Increment doc_count for each unique token_id."""
        conn = self.db.connect()
        unique = set(token_ids)
        for tid in unique:
            conn.execute(
                "INSERT INTO token_doc_freq (token_id, doc_count) "
                "VALUES (?, 1) "
                "ON CONFLICT(token_id) DO UPDATE SET doc_count = doc_count + 1",
                (tid,),
            )
        conn.commit()

    def get_doc_freq(self, token_id: int) -> int:
        """Return the number of documents containing *token_id*."""
        conn = self.db.connect()
        row = conn.execute(
            "SELECT doc_count FROM token_doc_freq WHERE token_id = ?",
            (token_id,),
        ).fetchone()
        return row["doc_count"] if row else 1  # 1 = Laplace smoothing

    def get_all_doc_freqs(self, token_ids: list[int]) -> dict[int, int]:
        """Batch lookup of doc frequencies for multiple token IDs."""
        if not token_ids:
            return {}
        conn = self.db.connect()
        placeholders = ",".join("?" for _ in token_ids)
        rows = conn.execute(
            f"SELECT token_id, doc_count FROM token_doc_freq "
            f"WHERE token_id IN ({placeholders})",
            token_ids,
        ).fetchall()
        freqs = {r["token_id"]: r["doc_count"] for r in rows}
        # Laplace smoothing: unseen tokens get df=1
        for tid in token_ids:
            freqs.setdefault(tid, 1)
        return freqs

    def get_total_docs(self) -> int:
        """Return total number of indexed documents (IDF denominator)."""
        conn = self.db.connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return row["n"] if row else 0