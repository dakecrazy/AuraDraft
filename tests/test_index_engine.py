"""End-to-end test for M1: TokenIndexEngine ingestion + search.

Tests:
  1. Index two Chinese documents (deep learning + legal contract)
  2. Verify stats are correct
  3. Search "注意力机制" → deep_learning_attention.txt wins
  4. Search "违约金" → legal_contract.txt wins
  5. Verify chunk text round-trip
  6. Compare exact / phrase / hybrid modes
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kb_agent.tokenizer.canonical import CanonicalTokenizer
from kb_agent.storage.db import Database
from kb_agent.index.engine import TokenIndexEngine
from kb_agent.document.loader import load_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"


def test_index_and_search():
    db_path = tempfile.mktemp(suffix=".db")
    try:
        db = Database(db_path)
        tokenizer = CanonicalTokenizer()
        engine = TokenIndexEngine(db, tokenizer, chunk_size=256, chunk_overlap=32)

        # ── 1. Index two documents ────────────────────────────────
        doc1_path = str(FIXTURES / "deep_learning_attention.txt")
        doc2_path = str(FIXTURES / "legal_contract.txt")

        r1 = engine.index_document(
            doc_id="dl_001",
            file_path=doc1_path,
            category="技术/深度学习",
            tags=["attention", "transformer"],
            summary="深度学习注意力机制综述",
        )
        r2 = engine.index_document(
            doc_id="legal_001",
            file_path=doc2_path,
            category="法律/合同",
            tags=["买卖合同", "违约责任"],
            summary="服务器采购合同范本",
        )

        print(f"Doc 1: {json.dumps(r1, ensure_ascii=False)}")
        print(f"Doc 2: {json.dumps(r2, ensure_ascii=False)}")

        # ── 2. Verify stats ───────────────────────────────────────
        stats = engine.get_stats()
        print(f"Stats: {json.dumps(stats, ensure_ascii=False)}")
        assert stats["total_documents"] == 2, f"Expected 2 docs, got {stats['total_documents']}"
        assert stats["total_tokens_indexed"] > 0, "Zero tokens indexed"
        assert stats["unique_token_types"] > 0, "Zero unique token types"
        assert stats["avg_document_length"] > 0, f"avg_dl is {stats['avg_document_length']}"

        # ── 3. Search "注意力机制" → deep learning wins ────────────
        results = engine.search("注意力机制", top_k=5, mode="hybrid")
        print(f"\nSearch '注意力机制' (hybrid): {json.dumps(results, ensure_ascii=False)}")
        assert len(results) > 0, "No results for '注意力机制'"
        assert results[0]["doc_id"] == "dl_001", (
            f"Expected dl_001 first, got {results[0]['doc_id']} "
            f"(score={results[0]['score']})"
        )

        # ── 4. Search "违约金" → legal contract wins ───────────────
        results = engine.search("违约金", top_k=5, mode="hybrid")
        print(f"\nSearch '违约金' (hybrid): {json.dumps(results, ensure_ascii=False)}")
        assert len(results) > 0, "No results for '违约金'"
        assert results[0]["doc_id"] == "legal_001", (
            f"Expected legal_001 first, got {results[0]['doc_id']} "
            f"(score={results[0]['score']})"
        )

        # ── 5. Chunk text round-trip ──────────────────────────────
        chunk_text = engine.get_chunk_text("dl_001_0")
        print(f"\nChunk text (first 100 chars): {chunk_text[:100]}")
        assert len(chunk_text) > 50, "Chunk text too short"
        assert "注意力" in chunk_text, "Chunk text missing '注意力'"

        # ── 6. Compare search modes ───────────────────────────────
        exact = engine.search("注意力机制", top_k=5, mode="exact")
        phrase = engine.search("注意力机制", top_k=5, mode="phrase")
        hybrid = engine.search("注意力机制", top_k=5, mode="hybrid")
        print(f"\nMode comparison for '注意力机制':")
        print(f"  exact:  {len(exact)} results, top={exact[0]['doc_id'] if exact else 'none'}")
        print(f"  phrase: {len(phrase)} results, top={phrase[0]['doc_id'] if phrase else 'none'}")
        print(f"  hybrid: {len(hybrid)} results, top={hybrid[0]['doc_id'] if hybrid else 'none'}")

        # At least one mode should return results
        assert len(exact) + len(phrase) + len(hybrid) > 0, "All modes returned zero results"

        print("\n✅ All M1 tests passed!")
        return True

    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_index_and_search()
    sys.exit(0 if success else 1)