"""KnowledgeCluster — a statistical cluster of documents in token space.

M2: pure statistics, no LLM.  The centroid is a sparse token-frequency
vector (L2-normalised).  M3+ will add knowledge_card for LLM-driven
reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class KnowledgeCluster:
    """A cluster of documents grouped by token-frequency similarity.

    The *centroid* is a sparse dict ``{token_id: weight}`` where weights
    are L2-normalised TF values.  *member_doc_ids* tracks which documents
    belong to this cluster (for splitting / merging in M3+).

    *knowledge_card* is a free-text summary maintained by the LLM
    (M3+).  It describes what this cluster "knows" — core concepts,
    evolution, open questions, and cross-links.
    """

    cluster_id: str
    centroid: dict[int, float]  # token_id → L2-normalised weight
    member_doc_ids: list[str] = field(default_factory=list)
    label: str = ""
    knowledge_card: str = ""
    doc_count: int = 0
    created_at: str = ""
    last_updated: str = ""

    # ── serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe dict (centroid keys → str)."""
        return {
            "cluster_id": self.cluster_id,
            "centroid": {str(k): v for k, v in self.centroid.items()},
            "member_doc_ids": self.member_doc_ids,
            "label": self.label,
            "knowledge_card": self.knowledge_card,
            "doc_count": self.doc_count,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeCluster:
        """Restore from JSON-safe dict (centroid keys ← int)."""
        return cls(
            cluster_id=d["cluster_id"],
            centroid={int(k): v for k, v in d["centroid"].items()},
            member_doc_ids=d.get("member_doc_ids", []),
            label=d.get("label", ""),
            knowledge_card=d.get("knowledge_card", ""),
            doc_count=d.get("doc_count", 0),
            created_at=d.get("created_at", ""),
            last_updated=d.get("last_updated", ""),
        )

    # ── prompt context (M3) ───────────────────────────────────────

    def to_prompt_context(self) -> str:
        """Format this cluster for LLM prompt context.

        This is the canonical format that both MoERouter and
        MockLLMClient agree on.  Uses ``=== {cluster_id} ===``
        as the section marker so the mock's regex can find it.
        """
        card = self.knowledge_card or "(暂无知识档案)"
        return (
            f"=== {self.cluster_id} ===\n"
            f"标签: {self.label}\n"
            f"{card}\n"
            f"文档数量: {self.doc_count}"
        )

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        return (
            f"KnowledgeCluster(id={self.cluster_id}, label={self.label!r}, "
            f"docs={self.doc_count}, centroid_tokens={len(self.centroid)})"
        )