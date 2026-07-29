"""Test for tools layer — simulates a complete Hermes agent orchestration cycle.

Flow (no pipeline, no LLM — pure tool orchestration):
  1. init_kb → session.connect()
  2. kb_ingest(dl_001) → signature
  3. kb_prefilter(signature) → empty (first doc)
  4. kb_create("深度学习", card, dl_001) → cid_dl
  5. kb_archive(dl_001, "深度学习")
  6. kb_ingest(legal_001) → signature
  7. kb_prefilter(signature) → [dl cluster]
  8. kb_create("法律合同", card, legal_001) → cid_legal
  9. kb_ingest(dl_002) → signature
  10. kb_prefilter(signature) → [dl, legal]
  11. kb_assign(dl_002, cid_dl, updated_card)
  12. kb_search("注意力机制") → dl_001 + dl_002 top
  13. kb_get_cards([cid_dl, cid_legal]) → verify cards
  14. kb_update_card(cid_legal, new_card) → verify update
  15. Persistence: close + reopen → verify everything survives
"""

import sys, tempfile, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kb_agent.tools import (
    init_kb, kb_ingest, kb_prefilter, kb_get_cards,
    kb_assign, kb_create, kb_update_card, kb_search, kb_archive,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"


def test_tools_orchestration():
    db_path = tempfile.mktemp(suffix=".db")
    archive_root = tempfile.mkdtemp()
    try:
        # ── 1. init + connect ─────────────────────────────────────
        session = init_kb(
            db_path=db_path,
            archive_root=archive_root,
            similarity_threshold=0.35,
        )
        session.connect()
        assert session._connected, "Session not connected"

        # ── 2. kb_ingest(dl_001) ──────────────────────────────────
        dl1_path = str(FIXTURES / "deep_learning_attention.txt")
        r1 = kb_ingest(session, dl1_path, doc_id="dl_001", category="技术/深度学习")
        print(f"kb_ingest(dl_001): doc_id={r1['doc_id']}, tokens={r1['total_tokens']}")
        assert r1["doc_id"] == "dl_001"
        assert r1["total_tokens"] > 0

        # ── 3. kb_prefilter → empty (first doc) ───────────────────
        candidates = kb_prefilter(session, "dl_001")
        print(f"kb_prefilter(after dl_001): {candidates}")
        assert candidates == [], f"Expected empty, got {candidates}"

        # ── 4. kb_create("深度学习", card, dl_001) ────────────────
        card_dl = "领域：深度学习\n核心知识：注意力机制、Transformer、多头注意力"
        c1 = kb_create(session, "深度学习", card_dl, "dl_001")
        print(f"kb_create(深度学习): cluster_id={c1['cluster_id']}")
        assert c1["cluster_id"]
        assert c1["cluster_label"] == "深度学习"
        assert c1["doc_count"] == 1
        cid_dl = c1["cluster_id"]

        # ── 5. kb_archive(dl_001, "深度学习") ─────────────────────
        a1 = kb_archive(session, dl1_path, "深度学习", doc_id="dl_001")
        print(f"kb_archive: {a1['archived_path']}")
        assert Path(a1["archived_path"]).exists()
        assert "深度学习" in a1["archived_path"]

        # ── 6. kb_ingest(legal_001) ───────────────────────────────
        legal_path = str(FIXTURES / "legal_contract.txt")
        r2 = kb_ingest(session, legal_path, doc_id="legal_001", category="法律/合同")
        print(f"kb_ingest(legal_001): tokens={r2['total_tokens']}")

        # ── 7. kb_prefilter → should return dl cluster ────────────
        candidates = kb_prefilter(session, "legal_001")
        print(f"kb_prefilter(after legal_001): {candidates}")
        assert len(candidates) >= 1, f"Expected >=1 candidate, got {candidates}"
        assert candidates[0]["cluster_id"] == cid_dl, (
            f"Expected {cid_dl}, got {candidates[0]['cluster_id']}"
        )

        # ── 8. kb_create("法律合同", card, legal_001) ─────────────
        card_legal = "领域：法律合同\n核心知识：合同条款、违约责任、付款方式"
        c2 = kb_create(session, "法律合同", card_legal, "legal_001")
        print(f"kb_create(法律合同): cluster_id={c2['cluster_id']}")
        cid_legal = c2["cluster_id"]

        # ── 9. kb_ingest(dl_002) ──────────────────────────────────
        dl2_path = str(FIXTURES / "deep_learning_training.txt")
        r3 = kb_ingest(session, dl2_path, doc_id="dl_002", category="技术/深度学习")
        print(f"kb_ingest(dl_002): tokens={r3['total_tokens']}")

        # ── 10. kb_prefilter → should return both clusters ────────
        candidates = kb_prefilter(session, "dl_002")
        print(f"kb_prefilter(after dl_002): {candidates}")
        assert len(candidates) >= 1, f"Expected >=1 candidate, got {candidates}"
        found_dl = any(c["cluster_id"] == cid_dl for c in candidates)
        assert found_dl, f"Expected {cid_dl} in candidates: {candidates}"

        # ── 11. kb_assign(dl_002, cid_dl, updated_card) ──────────
        updated_card = "领域：深度学习\n核心知识：注意力机制、Transformer、训练优化"
        a2 = kb_assign(session, "dl_002", cid_dl, card_text=updated_card)
        print(f"kb_assign(dl_002 → {cid_dl}): {a2}")
        assert a2["cluster_id"] == cid_dl
        assert a2["doc_count"] >= 2, f"Expected >=2 docs, got {a2['doc_count']}"

        # ── 12. kb_search("注意力机制") ───────────────────────────
        results = kb_search(session, "注意力机制", top_k=5, mode="hybrid")
        print(f"kb_search('注意力机制'): {len(results)} results")
        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        top_ids = [r["doc_id"] for r in results]
        assert "dl_001" in top_ids, f"Expected dl_001 in top results: {top_ids}"
        assert "dl_002" in top_ids, f"Expected dl_002 in top results: {top_ids}"

        # ── 13. kb_get_cards ──────────────────────────────────────
        cards = kb_get_cards(session, [cid_dl, cid_legal])
        print(f"kb_get_cards: {list(cards.keys())}")
        assert cid_dl in cards
        assert cid_legal in cards
        assert "深度学习" in cards[cid_dl]
        assert "法律合同" in cards[cid_legal]

        # ── 14. kb_update_card ────────────────────────────────────
        new_card = "领域：法律合同\n核心知识：买卖合同、违约责任、争议解决"
        u1 = kb_update_card(session, cid_legal, new_card)
        print(f"kb_update_card({cid_legal}): {u1}")
        cards2 = kb_get_cards(session, [cid_legal])
        assert cards2[cid_legal] == new_card, (
            f"Card not updated: {cards2[cid_legal][:50]} != {new_card[:50]}"
        )

        # ── 15. Persistence ───────────────────────────────────────
        session.close()
        assert not session._connected, "Session not closed"

        session2 = init_kb(db_path=db_path, archive_root=archive_root)
        session2.connect()
        cards3 = kb_get_cards(session2, [cid_dl, cid_legal])
        assert cid_dl in cards3, f"Cluster {cid_dl} missing after reload"
        assert cid_legal in cards3, f"Cluster {cid_legal} missing after reload"
        assert "深度学习" in cards3[cid_dl], "DL card lost after reload"
        assert "争议解决" in cards3[cid_legal], "Legal card lost after reload"
        session2.close()

        print("\n✅ All tools tests passed!")
        return True

    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_tools_orchestration()
    sys.exit(0 if success else 1)