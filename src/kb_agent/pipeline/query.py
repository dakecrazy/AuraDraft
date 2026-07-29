# DEPRECATED: 被 tools/ops.py 替代，保留仅为 M3/M4 测试
"""Query pipeline — BM25 search + LLM answer synthesis + cluster navigation.

Flow:
  1. Tokenize query → BM25 hybrid search (Layer 1 + Layer 2)
  2. Extract top-K chunk texts
  3. Find relevant clusters via token signature
  4. (Optional) LLM synthesises answer from chunk texts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb_agent.index.engine import TokenIndexEngine
    from kb_agent.cluster.manager import TokenClusterEngine
    from kb_agent.llm.client import LLMClient


class QueryPipeline:
    """End-to-end query pipeline: search → cluster nav → LLM answer."""

    def __init__(
        self,
        index_engine: TokenIndexEngine,
        cluster_engine: TokenClusterEngine,
        llm: LLMClient,
        top_k_docs: int = 5,
        top_k_chunks: int = 3,
    ):
        self.index = index_engine
        self.cluster = cluster_engine
        self.llm = llm
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks

    def query(self, question: str, mode: str = "hybrid") -> dict[str, Any]:
        """Query the knowledge base.

        Returns:
          answer: str (LLM-synthesised, or fallback)
          sources: list of matched documents
          relevant_clusters: dict of cluster_label → similarity
        """
        # ── 1. BM25 search ────────────────────────────────────────
        doc_results = self.index.search(question, top_k=self.top_k_docs, mode=mode)

        # ── 2. Extract chunk texts ────────────────────────────────
        chunks: list[dict[str, Any]] = []
        for r in doc_results[: self.top_k_chunks]:
            doc_id = r["doc_id"]
            chunk_text = self.index.get_chunk_text(f"{doc_id}_0")
            if chunk_text:
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "file_path": r.get("file_path", ""),
                        "category": r.get("category", ""),
                        "text": chunk_text[:1000],  # truncate for context
                        "score": r.get("score", 0),
                    }
                )

        # ── 3. Find relevant clusters via doc-to-cluster mapping ────
        relevant_clusters: dict[str, float] = {}
        for r in doc_results:
            doc_id = r["doc_id"]
            score = r.get("score", 0)
            for cid, cluster in self.cluster.clusters.items():
                if doc_id in cluster.member_doc_ids:
                    label = cluster.label or cid
                    relevant_clusters[label] = max(
                        relevant_clusters.get(label, 0), score
                    )

        # ── 4. LLM answer synthesis ───────────────────────────────
        answer = self._synthesize(question, chunks)

        return {
            "answer": answer,
            "sources": doc_results,
            "relevant_clusters": dict(
                sorted(relevant_clusters.items(), key=lambda x: -x[1])
            ),
        }

    # ── internal ──────────────────────────────────────────────────

    def _synthesize(
        self, question: str, chunks: list[dict[str, Any]]
    ) -> str:
        """Ask the LLM to synthesise an answer from chunk texts.

        Falls back to a simple message if no chunks or LLM fails.
        """
        if not chunks:
            return "未找到相关文档。"

        context_parts = []
        for c in chunks:
            context_parts.append(
                f"[来源: {c.get('file_path', c['doc_id'])}]\n{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""基于以下参考资料回答问题。如果资料不足，请说明。

参考资料：
{context}

问题：{question}

回答要求：
- 只基于参考资料回答
- 如果资料不足，明确说明
- 引用来源（文件名）"""
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            # Fallback: return first chunk text
            return chunks[0]["text"][:500] if chunks else "无法生成回答。"