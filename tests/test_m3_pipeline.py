"""End-to-end test for M3: IngestionPipeline with MockLLMClient.

Tests:
  1. Pipeline ingests 5 documents (DL×2, legal, quantum, cooking)
  2. DL documents are classified into the SAME cluster (L1 deep classification)
  3. knowledge_card is non-empty for every cluster
  4. Persistence: close DB, reopen, verify clusters + cards survive
  5. LLM call count is reasonable
"""

import json
import sys
import tempfile
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

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"

DOCS = [
    ("dl_001", "deep_learning_attention.txt", "技术/深度学习"),
    ("legal_001", "legal_contract.txt", "法律/合同"),
    ("quantum_001", "quantum_computing.txt", "技术/量子计算"),
    ("cooking_001", "cooking_recipe.txt", "生活/烹饪"),
    ("dl_002", "deep_learning_training.txt", "技术/深度学习"),
]


def test_m3_pipeline():
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
        pipeline = IngestionPipeline(
            index_engine, cluster_engine, router, llm, store,
            archive_root=archive_root,
        )

        # ── 1. Ingest all 5 documents ─────────────────────────────
        print("=" * 60)
        print("M3: Ingesting 5 documents via IngestionPipeline")
        print("=" * 60)

        results = []
        for doc_id, filename, category in DOCS:
            file_path = str(FIXTURES / filename)
            result = pipeline.ingest(
                file_path=file_path,
                doc_id=doc_id,
                category=category,
            )
            results.append(result)

            cls = result["classification"]
            print(
                f"  {doc_id:12s} → {cls['action']:22s} "
                f"cluster={cls.get('cluster_id', '?'):12s} "
                f"label={cls.get('cluster_label', '')[:30]:30s}"
            )

        # ── 2. Verify DL documents are in the SAME cluster ────────
        print("\n" + "=" * 60)
        print("Assertion: DL documents share a cluster")
        print("=" * 60)

        dl_clusters = set()
        for r in results:
            doc_id = r["doc_id"]
            cid = r["classification"]["cluster_id"]
            if doc_id.startswith("dl_"):
                dl_clusters.add(cid)
                print(f"  {doc_id} → cluster {cid}")

        assert len(dl_clusters) == 1, (
            f"Expected DL docs in 1 cluster, got {len(dl_clusters)}: {dl_clusters}"
        )
        print(f"  ✅ DL documents merged into cluster: {dl_clusters.pop()}")

        # ── 3. Verify knowledge_card is non-empty ─────────────────
        print("\n" + "=" * 60)
        print("Assertion: all clusters have non-empty knowledge_card")
        print("=" * 60)

        all_clusters = cluster_engine.get_all_clusters()
        for c in all_clusters:
            has_card = bool(c.knowledge_card and len(c.knowledge_card.strip()) > 10)
            print(
                f"  {c.cluster_id:12s} label={c.label[:25]:25s} "
                f"card_len={len(c.knowledge_card):5d} docs={c.doc_count}"
            )
            assert has_card, (
                f"Cluster {c.cluster_id} ({c.label}) has empty or trivial knowledge_card"
            )
        print(f"  ✅ All {len(all_clusters)} clusters have valid knowledge_cards")

        # ── 4. Verify persistence ─────────────────────────────────
        print("\n" + "=" * 60)
        print("Persistence test: close DB, reopen, verify clusters + cards")
        print("=" * 60)

        before = {
            c.cluster_id: {
                "label": c.label,
                "doc_count": c.doc_count,
                "card_len": len(c.knowledge_card),
            }
            for c in all_clusters
        }

        db.close()
        db2 = Database(db_path)
        store2 = ClusterStore(db2)
        cluster_engine2 = TokenClusterEngine(
            tokenizer, store2, similarity_threshold=0.35
        )

        after = {
            c.cluster_id: {
                "label": c.label,
                "doc_count": c.doc_count,
                "card_len": len(c.knowledge_card),
            }
            for c in cluster_engine2.get_all_clusters()
        }

        assert len(after) == len(before), (
            f"Cluster count mismatch: before={len(before)}, after={len(after)}"
        )
        for cid, info in before.items():
            assert cid in after, f"Cluster {cid} missing after reload"
            assert after[cid]["doc_count"] == info["doc_count"], (
                f"doc_count mismatch for {cid}: {after[cid]['doc_count']} vs {info['doc_count']}"
            )
            assert after[cid]["card_len"] == info["card_len"], (
                f"card length mismatch for {cid}: {after[cid]['card_len']} vs {info['card_len']}"
            )
        print(f"  ✅ All {len(before)} clusters survived reload with correct cards")

        # ── 5. Verify LLM call count ──────────────────────────────
        print("\n" + "=" * 60)
        print(f"LLM call count: {llm.call_count}")
        print("=" * 60)
        # Each document: 1 classify + 1 card gen/update = 2 calls
        # First doc also gets label gen = 3 calls
        # Total: 3 + 4*2 = 11 calls
        expected_min = 5  # at least 1 per doc
        expected_max = 20  # generous upper bound
        assert expected_min <= llm.call_count <= expected_max, (
            f"LLM call count {llm.call_count} outside expected range [{expected_min}, {expected_max}]"
        )
        print(f"  ✅ LLM call count {llm.call_count} is reasonable")

        # ── 6. Print summary ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("M3 Summary")
        print("=" * 60)
        for r in results:
            cls = r["classification"]
            print(
                f"  {r['doc_id']:12s} → {cls['action']:22s} "
                f"cluster={cls.get('cluster_id', '?'):12s}"
            )
        print(f"\n  Total clusters: {cluster_engine.get_cluster_count()}")
        print(f"  Total documents: {len(results)}")
        print(f"  LLM calls: {llm.call_count}")
        print("\n✅ All M3 tests passed!")
        return True

    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_m3_pipeline()
    sys.exit(0 if success else 1)