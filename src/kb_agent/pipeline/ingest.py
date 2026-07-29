# DEPRECATED: 被 tools/ops.py 替代，保留仅为 M3/M4 测试
"""Ingestion pipeline — orchestrates index → prefilter → LLM → cluster update.

This is the central orchestration layer that ties M1 (index), M2 (statistical
prefilter), and M3 (LLM deep classification) together.

IMPORTANT: The pipeline does NOT call TokenClusterEngine.classify_and_assign().
It uses the engine's internal methods directly, with MoERouter making the
routing decision.  This avoids double-classification conflicts.
"""

from __future__ import annotations

import uuid
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kb_agent.document.loader import load_text
from kb_agent.cluster.model import KnowledgeCluster

if TYPE_CHECKING:
    from kb_agent.cluster.manager import TokenClusterEngine
    from kb_agent.index.engine import TokenIndexEngine
    from kb_agent.llm.client import LLMClient
    from kb_agent.router.moe_router import MoERouter
    from kb_agent.storage.cluster_store import ClusterStore


class IngestionPipeline:
    """End-to-end ingestion pipeline.

    Flow:
      1. Index the document (always)
      2. Prefilter via MoERouter (statistical)
      3. If no clusters exist → create first cluster with LLM-generated card
      4. Deep classify via MoERouter (LLM)
      5. Execute the decision: assign/create + update knowledge_card
      6. Update token doc-frequency
    """

    def __init__(
        self,
        index_engine: TokenIndexEngine,
        cluster_engine: TokenClusterEngine,
        router: MoERouter,
        llm: LLMClient,
        store: ClusterStore,
        archive_root: str = "./knowledge_base",
    ):
        self.index = index_engine
        self.cluster = cluster_engine
        self.router = router
        self.llm = llm
        self.store = store
        self.archive_root = Path(archive_root)

    def ingest(
        self,
        file_path: str,
        doc_id: str | None = None,
        category: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest a document: index + classify + update knowledge base.

        Returns a detailed result dict with all stages.
        """
        # ── 0. Load text ──────────────────────────────────────────
        text = load_text(file_path)
        doc_id = doc_id or uuid.uuid4().hex[:12]

        # ── 1. Index (always) ─────────────────────────────────────
        index_result = self.index.index_document(
            doc_id=doc_id,
            text=text,
            file_path=file_path,
            category=category,
            tags=tags,
        )
        candidates = self.router._prefilter(text)

        # ── 3. Classify ───────────────────────────────────────────
        if not candidates:
            # First document → create first cluster
            classification = self._create_first_cluster(doc_id, text)
        else:
            # Deep classify via LLM
            classification = self.router._deep_classify(text, candidates)

        # ── 4. Execute decision ───────────────────────────────────
        sig = self.cluster._extract_signature(text)

        if classification["action"] == "assigned":
            cid = classification["cluster_id"]
            self.cluster._add_to_cluster(cid, doc_id, sig)

            # Update knowledge_card via LLM
            cluster = self.cluster.clusters[cid]
            if classification.get("card_update"):
                cluster.knowledge_card = self._update_card(
                    cluster, classification["card_update"]
                )
                cluster.last_updated = KnowledgeCluster.now()
                self.store.save_cluster(cluster)

        elif classification["action"] == "new_cluster":
            # Create cluster with LLM-generated label and card
            label = classification.get("cluster_label", "")
            if not label:
                label = self._generate_label(text)

            # Create the cluster (persists immediately without card)
            create_result = self.cluster._create_new_cluster(doc_id, sig, text)
            cid = create_result["cluster_id"]

            # Generate and set the knowledge_card
            cluster = self.cluster.clusters[cid]
            cluster.label = label
            cluster.knowledge_card = self._generate_card(text, label)
            cluster.last_updated = KnowledgeCluster.now()
            self.store.save_cluster(cluster)

            classification["cluster_id"] = cid
            classification["cluster_label"] = label

        # ── 5. Archive physical file ──────────────────────────────
        label = classification.get("cluster_label", "")
        if label:
            archived_path = self._archive_file(file_path, label, doc_id)
            # Update the index to point to the archived location
            if archived_path != file_path:
                conn = self.index.db.connect()
                conn.execute(
                    "UPDATE documents SET file_path = ? WHERE doc_id = ?",
                    (archived_path, doc_id),
                )
                conn.commit()
        else:
            archived_path = file_path

        # ── 6. Update doc-freq ────────────────────────────────────
        self.cluster._update_doc_freq(doc_id)

        return {
            "doc_id": doc_id,
            "index": index_result,
            "classification": classification,
            "archived_path": archived_path,
            "cluster_count": self.cluster.get_cluster_count(),
        }

    # ── internal helpers ──────────────────────────────────────────

    def _archive_file(self, src_path: str, label: str, doc_id: str) -> str:
        """Copy the source file to ``archive_root/{sanitized_label}/``.

        Handles filename conflicts by appending the doc_id.
        Returns the destination path.
        """
        safe_label = (
            re.sub(r'[^\w\u4e00-\u9fff]', "_", label).strip("_") or "unclassified"
        )
        target_dir = self.archive_root / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
        src = Path(src_path)
        target = target_dir / src.name
        if target.exists():
            target = target_dir / f"{src.stem}_{doc_id}{src.suffix}"
        shutil.copy2(str(src), str(target))
        return str(target)

    def _create_first_cluster(
        self, doc_id: str, text: str
    ) -> dict[str, Any]:
        """Handle the first document — create initial cluster."""
        label = self._generate_label(text)
        return {
            "action": "new_cluster",
            "cluster_id": "__pending__",
            "cluster_label": label,
            "reasoning": "First document — created initial cluster",
            "knowledge_delta": "",
            "cross_links": [],
            "new_cluster_suggestion": label,
            "card_update": "",
        }

    def _generate_label(self, text: str) -> str:
        """Ask the LLM to generate a short label for a new cluster."""
        prompt = f"""根据以下文档内容，生成一个简短的知识领域标签（2-6个字）。

文档：
{text[:2000]}

只返回标签本身，不要其他内容。"""
        try:
            label = self.llm.generate(prompt).strip()
            # Clean up
            label = label.replace('"', "").replace("'", "").strip()
            return label[:20] if label else "未分类"
        except Exception:
            return "未分类"

    def _generate_card(self, text: str, label: str) -> str:
        """Ask the LLM to generate an initial knowledge_card."""
        prompt = f"""你是一个知识管理员。一个新的知识领域「{label}」刚刚被创建。

根据以下文档内容，生成该领域的初始知识档案。要求：
- 核心知识：列出关键概念和它们之间的关系
- 知识演进：当前的技术/知识状态
- 未解决的问题：基于文档内容，列出开放性问题
- 代表性文档：这篇文档作为起点

文档：
{text[:4000]}

输出纯文本格式，不要 JSON。"""
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return f"领域: {label}\n基于初始文档生成的知识档案。"

    def _update_card(self, cluster, delta_hint: str) -> str:
        """Ask the LLM to update an existing knowledge_card."""
        prompt = f"""你是一个知识管理员。

当前知识档案（已积累 {cluster.doc_count} 篇文档）：
{cluster.knowledge_card}

新文档带来的知识增量：
{delta_hint}

请重写知识档案。要求：
- 不是简单追加，而是重新组织知识
- 如果新知识修正了旧知识，要更新而不是并列
- 保持结构：核心知识 / 知识演进 / 未解决的问题 / 关联领域
- 如果某个"未解决问题"有了进展，更新它的状态

输出纯文本格式，不要 JSON。"""
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return cluster.knowledge_card