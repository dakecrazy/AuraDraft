"""End-to-end test for M2: TokenClusterEngine clustering + persistence.

Tests:
  1. Index 4 documents (DL, legal, quantum, cooking)
  2. Classify each → verify at least 2 distinct clusters
  3. Print similarity matrix for human inspection
  4. Verify persistence: close DB, reopen, clusters survive
  5. Verify cluster labels are non-empty
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
from kb_agent.document.loader import load_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"

DOCS = [
    ("dl_001", "deep_learning_attention.txt", "技术/深度学习"),
    ("legal_001", "legal_contract.txt", "法律/合同"),
    ("quantum_001", "quantum_computing.txt", "技术/量子计算"),
    ("cooking_001", "cooking_recipe.txt", "生活/烹饪"),
    ("dl_002", "deep_learning_training.txt", "技术/深度学习"),
]


def test_clustering_and_persistence():
    db_path = tempfile.mktemp(suffix=".db")
    try:
        db = Database(db_path)
        tokenizer = CanonicalTokenizer()
        store = ClusterStore(db)
        engine = TokenIndexEngine(db, tokenizer, chunk_size=256, chunk_overlap=32)
        cluster_engine = TokenClusterEngine(
            tokenizer, store, similarity_threshold=0.35
        )

        # ── 1. Index + classify each document ─────────────────────
        print("=" * 60)
        print("M2: Indexing and classifying 4 documents")
        print("=" * 60)

        results = []
        for doc_id, filename, category in DOCS:
            file_path = str(FIXTURES / filename)
            text = load_text(file_path)

            # IMPORTANT: index FIRST so IDF has document counts
            r_index = engine.index_document(
                doc_id=doc_id,
                file_path=file_path,
                category=category,
            )

            # THEN classify (uses IDF from documents table)
            r_classify = cluster_engine.classify_and_assign(
                doc_id=doc_id,
                text=text,
                method="tfidf_topk",
            )

            results.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "tokens": r_index["total_tokens"],
                    "action": r_classify["action"],
                    "cluster_id": r_classify["cluster_id"],
                    "cluster_label": r_classify["cluster_label"],
                    "similarity": r_classify.get("similarity", 0.0),
                }
            )
            print(
                f"  {doc_id:12s} → {r_classify['action']:22s} "
                f"cluster={r_classify['cluster_id']:8s} "
                f"label={r_classify['cluster_label'][:30]:30s} "
                f"sim={r_classify.get('similarity', 0.0):.4f}"
            )

        # ── 2. Verify at least 2 distinct clusters ────────────────
        cluster_ids = set(r["cluster_id"] for r in results)
        cluster_count = cluster_engine.get_cluster_count()
        print(f"\n  Distinct clusters: {len(cluster_ids)} (engine reports {cluster_count})")
        assert len(cluster_ids) >= 2, (
            f"Expected at least 2 clusters, got {len(cluster_ids)}"
        )
        assert cluster_count == len(cluster_ids), (
            f"Engine count ({cluster_count}) != distinct IDs ({len(cluster_ids)})"
        )

        # ── 3. Print similarity matrix ────────────────────────────
        print("\n" + "=" * 60)
        print("Similarity matrix (cosine between cluster centroids)")
        print("=" * 60)

        all_clusters = cluster_engine.get_all_clusters()
        labels = [c.label[:20] for c in all_clusters]
        ids = [c.cluster_id for c in all_clusters]

        # Header
        header = f"{'':20s}" + "".join(f"{l:>12s}" for l in labels)
        print(header)
        print("-" * len(header))

        for i, ci in enumerate(all_clusters):
            row = f"{labels[i]:20s}"
            for j, cj in enumerate(all_clusters):
                sim = TokenClusterEngine._sparse_cosine(ci.centroid, cj.centroid)
                row += f"{sim:>12.4f}"
            print(row)

        # ── 4. Persistence test ───────────────────────────────────
        print("\n" + "=" * 60)
        print("Persistence test: close DB, reopen, verify clusters survive")
        print("=" * 60)

        # Save cluster IDs and labels before close
        before = {
            c.cluster_id: {"label": c.label, "doc_count": c.doc_count}
            for c in all_clusters
        }

        # Close and reopen
        db.close()
        db2 = Database(db_path)
        store2 = ClusterStore(db2)
        cluster_engine2 = TokenClusterEngine(
            tokenizer, store2, similarity_threshold=0.35
        )

        after = {
            c.cluster_id: {"label": c.label, "doc_count": c.doc_count}
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
            assert after[cid]["label"] == info["label"], (
                f"label mismatch for {cid}: {after[cid]['label']!r} vs {info['label']!r}"
            )

        print(f"  ✅ All {len(before)} clusters survived reload with correct doc_count and label")

        # ── 5. Verify labels are non-empty ────────────────────────
        for c in cluster_engine2.get_all_clusters():
            assert c.label.strip(), f"Cluster {c.cluster_id} has empty label"

        print("  ✅ All cluster labels are non-empty")
        print("\n✅ All M2 tests passed!")

        # ── 6. Print summary ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for r in results:
            print(
                f"  {r['doc_id']:12s} → {r['cluster_label'][:35]:35s} "
                f"(sim={r['similarity']:.4f})"
            )
        print(f"\n  Total clusters: {cluster_count}")
        print(f"  Total documents: {len(results)}")

        return True

    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_clustering_and_persistence()
    sys.exit(0 if success else 1)