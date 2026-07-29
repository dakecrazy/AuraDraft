"""MoE-style router: L0 prefilter + L1 deep classification.

L0 uses token-frequency similarity (from TokenClusterEngine) to select
Top-K candidate clusters.  L1 formats their knowledge cards into a
prompt and calls the LLM for deep reasoning.

Prompt format is carefully aligned with MockLLMClient's regex patterns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb_agent.cluster.manager import TokenClusterEngine
    from kb_agent.llm.client import LLMClient


class MoERouter:
    """Two-level router: statistical prefilter → LLM deep classification."""

    def __init__(
        self,
        cluster_engine: TokenClusterEngine,
        llm: LLMClient,
        top_k_candidates: int = 5,
        min_similarity: float = 0.05,
    ):
        self.cluster_engine = cluster_engine
        self.llm = llm
        self.top_k = top_k_candidates
        self.min_sim = min_similarity

    # ── public API ────────────────────────────────────────────────

    def classify(self, doc_id: str, text: str) -> dict[str, Any]:
        """Classify a document using L0 prefilter + L1 deep classification.

        Returns a dict with:
          action: "assigned" | "new_cluster"
          cluster_id: str
          cluster_label: str
          reasoning: str
          knowledge_delta: str
          cross_links: list
          new_cluster_suggestion: str | None
          card_update: str
        """
        # L0: statistical prefilter
        candidates = self._prefilter(text)
        if not candidates:
            # No clusters exist yet → first document
            return {
                "action": "new_cluster",
                "cluster_id": "__first__",
                "cluster_label": "",
                "reasoning": "First document in the knowledge base",
                "knowledge_delta": "",
                "cross_links": [],
                "new_cluster_suggestion": None,
                "card_update": "",
            }

        # L1: deep classification via LLM
        result = self._deep_classify(text, candidates)
        return result

    # ── L0: statistical prefilter ─────────────────────────────────

    def _prefilter(self, text: str) -> list[dict[str, Any]]:
        """Use token-frequency similarity to select Top-K candidate clusters.

        Returns a list of dicts with cluster_id, label, knowledge_card, doc_count.
        """
        if not self.cluster_engine.clusters:
            return []

        signature = self.cluster_engine._extract_signature(text)
        scored: list[tuple[str, float]] = []
        for cid, cluster in self.cluster_engine.clusters.items():
            sim = self.cluster_engine._sparse_cosine(
                signature, cluster.centroid
            )
            if sim >= self.min_sim:
                scored.append((cid, sim))

        scored.sort(key=lambda x: -x[1])
        top = scored[: self.top_k]

        candidates = []
        for cid, sim in top:
            cluster = self.cluster_engine.clusters[cid]
            candidates.append(
                {
                    "cluster_id": cid,
                    "label": cluster.label,
                    "knowledge_card": cluster.knowledge_card,
                    "doc_count": cluster.doc_count,
                    "similarity": round(sim, 4),
                }
            )
        return candidates

    # ── L1: deep classification ───────────────────────────────────

    def _deep_classify(
        self, text: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Format candidates into a prompt and call the LLM for deep reasoning.

        Prompt format aligns with MockLLMClient regex patterns:
          - "【新文档】" marks the document text
          - "=== {cluster_id} ===" marks each candidate (using cluster_id, not label)
        """
        # Build candidate context
        cards_parts = []
        for c in candidates:
            card = c.get("knowledge_card", "") or "(暂无知识档案)"
            cards_parts.append(
                f"=== {c['cluster_id']} ===\n"
                f"标签: {c['label']}\n"
                f"{card}\n"
                f"文档数量: {c['doc_count']}"
            )
        cards_context = "\n\n".join(cards_parts)

        # Truncate text to avoid blowing the context window
        text_truncated = text[:6000]

        prompt = f"""你是一个知识管理员，负责维护一个知识库的分类体系。

【新文档】
{text_truncated}

【候选知识领域】（共 {len(candidates)} 个）
{cards_context}

请做出判断：
【归属】这篇文档最属于哪个领域？给出理由。
   如果不属于任何现有领域，说明应该创建什么新领域。
【知识增量】它对目标领域的知识有什么具体补充？
【跨域桥接】它是否同时和另一个领域相关？
【知识延伸】基于这篇文档，你能推断出什么新知识？
【档案更新建议】目标领域的 knowledge_card 应该怎么修改？

输出 JSON 格式。"""

        try:
            result = self.llm.generate_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: treat as new cluster
            result = {
                "primary_cluster": "__new__",
                "reasoning": f"LLM response parsing failed: {e}",
                "knowledge_delta": "",
                "cross_links": [],
                "new_cluster_suggestion": None,
                "card_update": "",
            }

        # Normalise the result
        primary = result.get("primary_cluster", "__new__")

        if primary == "__new__":
            return {
                "action": "new_cluster",
                "cluster_id": "__new__",
                "cluster_label": result.get("new_cluster_suggestion", ""),
                "reasoning": result.get("reasoning", ""),
                "knowledge_delta": result.get("knowledge_delta", ""),
                "cross_links": result.get("cross_links", []),
                "new_cluster_suggestion": result.get("new_cluster_suggestion"),
                "card_update": result.get("card_update", ""),
            }
        else:
            # Look up the cluster label from the engine
            cluster = self.cluster_engine.clusters.get(primary)
            label = cluster.label if cluster else ""
            return {
                "action": "assigned",
                "cluster_id": primary,
                "cluster_label": label,
                "reasoning": result.get("reasoning", ""),
                "knowledge_delta": result.get("knowledge_delta", ""),
                "cross_links": result.get("cross_links", []),
                "new_cluster_suggestion": None,
                "card_update": result.get("card_update", ""),
            }