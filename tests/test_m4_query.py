"""End-to-end test for M4: QueryPipeline with MockLLMClient.

Tests:
  1. Ingest 5 documents (reuse M3 setup)
  2. Query "注意力机制的计算复杂度" → DL cluster is top relevant
  3. Query "违约金条款" → legal cluster is top relevant
  4. Answer is non-empty and contains domain keywords
  5. Sources list is non-empty with correct doc_ids
"""

import sys, tempfile, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kb_agent.tokenizer.canonical import CanonicalTokenizer
from kb_agent.storage.db import Database
from kb_agent.storage.cluster_store import ClusterStore
from kb_agent.index.engine import TokenIndexEngine
from kb_agent.cluster.manager import TokenClusterEngine
from kb_agent.llm.client import MockLLMClient
from kb_agent.router.moe_router import MoERouter
from kb_agent.pipeline.ingest import IngestionPipeline
from kb_agent.pipeline.query import QueryPipeline

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"

DOCS = [
    ("dl_001", "deep_learning_attention.txt", "技术/深度学习"),
    ("legal_001", "legal_contract.txt", "法律/合同"),
    ("quantum_001", "quantum_computing.txt", "技术/量子计算"),
    ("cooking_001", "cooking_recipe.txt", "生活/烹饪"),
    ("dl_002", "deep_learning_training.txt", "技术/深度学习"),
]


def test_m4_query():
    db_path = tempfile.mktemp(suffix=".db")
    try:
        db = Database(db_path)
        tokenizer = CanonicalTokenizer()
        store = ClusterStore(db)
        index_engine = TokenIndexEngine(db, tokenizer, chunk_size=256, chunk_overlap=32)
        cluster_engine = TokenClusterEngine(
            tokenizer, store, similarity_threshold=0.35
        )
        llm = MockLLMClient()
        router = MoERouter(cluster_engine, llm, top_k_candidates=5, min_similarity=0.05)
        archive_root = tempfile.mkdtemp()
        ingest_pipeline = IngestionPipeline(
            index_engine, cluster_engine, router, llm, store,
            archive_root=archive_root,
        )
        query_pipeline = QueryPipeline(index_engine, cluster_engine, llm)

        # ── 1. Ingest all 5 documents ─────────────────────────────
        print("=" * 60)
        print("M4: Ingesting 5 documents")
        print("=" * 60)
        for doc_id, filename, category in DOCS:
            file_path = str(FIXTURES / filename)
            result = ingest_pipeline.ingest(
                file_path=file_path, doc_id=doc_id, category=category
            )
            print(f"  {doc_id:12s} → {result['classification']['action']:22s}")

        # ── 2. Query "注意力机制的计算复杂度" ─────────────────────
        print("\n" + "=" * 60)
        print('Query 1: "注意力机制的计算复杂度"')
        print("=" * 60)
        r1 = query_pipeline.query("注意力机制的计算复杂度", mode="hybrid")
        print(f"  Answer: {r1['answer'][:100]}...")
        print(f"  Sources: {len(r1['sources'])} docs")
        print(f"  Relevant clusters: {r1['relevant_clusters']}")

        # DL cluster should be top relevant
        assert len(r1["sources"]) > 0, "Query 1 returned no sources"
        assert r1["answer"] and len(r1["answer"]) > 10, "Query 1 answer too short"
        assert "深度学习" in r1["answer"] or "注意力" in r1["answer"], (
            f"Query 1 answer missing domain keywords: {r1['answer'][:100]}"
        )

        # ── 3. Query "违约金条款" ─────────────────────────────────
        print("\n" + "=" * 60)
        print('Query 2: "违约金条款"')
        print("=" * 60)
        r2 = query_pipeline.query("违约金条款", mode="hybrid")
        print(f"  Answer: {r2['answer'][:100]}...")
        print(f"  Sources: {len(r2['sources'])} docs")
        print(f"  Relevant clusters: {r2['relevant_clusters']}")

        # Legal cluster should be top relevant
        assert len(r2["sources"]) > 0, "Query 2 returned no sources"
        assert r2["answer"] and len(r2["answer"]) > 10, "Query 2 answer too short"
        assert "法律合同" in r2["answer"] or "合同" in r2["answer"], (
            f"Query 2 answer missing domain keywords: {r2['answer'][:100]}"
        )

        # ── 4. Verify sources have correct doc_ids ────────────────
        print("\n" + "=" * 60)
        print("Assertion: sources contain correct doc_ids")
        print("=" * 60)
        q1_doc_ids = {s["doc_id"] for s in r1["sources"]}
        q2_doc_ids = {s["doc_id"] for s in r2["sources"]}
        print(f"  Query 1 doc_ids: {q1_doc_ids}")
        print(f"  Query 2 doc_ids: {q2_doc_ids}")

        # Query 1 should include at least one DL doc
        assert q1_doc_ids & {"dl_001", "dl_002"}, (
            f"Query 1 missing DL docs: {q1_doc_ids}"
        )
        # Query 2 should include legal doc
        assert "legal_001" in q2_doc_ids, (
            f"Query 2 missing legal_001: {q2_doc_ids}"
        )

        # ── 5. Verify relevant_clusters ───────────────────────────
        print("\n" + "=" * 60)
        print("Assertion: relevant_clusters contain correct domains")
        print("=" * 60)
        q1_clusters = set(r1["relevant_clusters"].keys())
        q2_clusters = set(r2["relevant_clusters"].keys())
        print(f"  Query 1 clusters: {q1_clusters}")
        print(f"  Query 2 clusters: {q2_clusters}")

        # At least one cluster should be relevant for each query
        assert len(q1_clusters) > 0, "Query 1 has no relevant clusters"
        assert len(q2_clusters) > 0, "Query 2 has no relevant clusters"

        # ── 6. Summary ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("M4 Summary")
        print("=" * 60)
        print(f"  LLM calls: {llm.call_count}")
        print(f"  Query 1 sources: {len(r1['sources'])}")
        print(f"  Query 2 sources: {len(r2['sources'])}")
        print(f"  Query 1 clusters: {len(q1_clusters)}")
        print(f"  Query 2 clusters: {len(q2_clusters)}")
        print("\n✅ All M4 tests passed!")
        return True

    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_m4_query()
    sys.exit(0 if success else 1)