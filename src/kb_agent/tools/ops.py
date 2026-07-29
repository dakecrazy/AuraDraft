"""8 atomic tool functions for Hermes agent orchestration.

Every function takes a ``KnowledgeBaseSession`` as the first argument.
The session must be connected (``session.connect()``) before calling any tool.

Tools:
  kb_ingest         — Index a document (tokenize + chunk + build inverted index)
  kb_prefilter      — Find Top-K candidate clusters by token-frequency similarity
  kb_get_cards      — Fetch knowledge_card text for one or more clusters
  kb_assign         — Assign a document to an existing cluster
  kb_create         — Create a new cluster with a label and knowledge_card
  kb_update_card    — Update a cluster's knowledge_card
  kb_search         — BM25 search (exact / phrase / hybrid)
  kb_archive        — Copy a file to ``archive_root/{label}/``
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from kb_agent.tools.session import KnowledgeBaseSession
from kb_agent.cluster.model import KnowledgeCluster


# ── 1. kb_ingest ──────────────────────────────────────────────────

def kb_ingest(
    session: KnowledgeBaseSession,
    file_path: str,
    doc_id: str | None = None,
    category: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Index a document into the three-layer token index.

    **Does NOT classify or assign.**  That is the caller's (Hermes')
    responsibility.  Returns the doc_id and index stats.

    The token signature is stored in the ``doc_signatures`` table and
    is NOT returned to the caller — use ``kb_prefilter(doc_id)`` to
    retrieve candidate clusters.

    IMPORTANT: does NOT update token doc-frequency — that happens
    when the document is actually assigned to a cluster (kb_assign /
    kb_create).
    """
    from kb_agent.document.loader import load_text

    text = load_text(file_path)
    doc_id = doc_id or uuid.uuid4().hex[:12]

    index_result = session.index_engine.index_document(
        doc_id=doc_id,
        text=text,
        file_path=file_path,
        category=category,
        tags=tags,
    )

    # Extract signature and store in DB
    signature = session.cluster_engine._extract_signature(text)
    conn = session.db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO doc_signatures (doc_id, signature, created_at) "
        "VALUES (?, ?, ?)",
        (doc_id, json.dumps({str(k): v for k, v in signature.items()}), KnowledgeCluster.now()),
    )
    conn.commit()

    return {
        "doc_id": doc_id,
        "total_tokens": index_result["total_tokens"],
        "chunk_count": index_result["chunk_count"],
        "unique_tokens": index_result["unique_tokens"],
    }


# ── 2. kb_prefilter ───────────────────────────────────────────────

def kb_prefilter(
    session: KnowledgeBaseSession,
    doc_id: str,
    top_k: int = 5,
    min_similarity: float = 0.05,
) -> list[dict[str, Any]]:
    """Find Top-K candidate clusters by token-frequency similarity.

    Reads the document's signature from the ``doc_signatures`` table
    (stored by ``kb_ingest``).

    When the total number of clusters is ≤ *top_k*, returns ALL clusters
    without similarity filtering (L0 router is for scaling to 10000+
    clusters, not for filtering 1-2).

    Returns compact results — no knowledge_card text (use kb_get_cards
    to fetch those on demand).

    Each result::
        {"cluster_id": str, "label": str, "similarity": float, "doc_count": int}
    """
    if not session.cluster_engine.clusters:
        return []

    # Read signature from DB
    conn = session.db.connect()
    row = conn.execute(
        "SELECT signature FROM doc_signatures WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if not row:
        return []  # doc not ingested yet
    sig_int = {int(k): v for k, v in json.loads(row["signature"]).items()}

    # When few clusters exist, return all without filtering
    if len(session.cluster_engine.clusters) <= top_k:
        return [
            {
                "cluster_id": cid,
                "label": cluster.label,
                "similarity": round(
                    session.cluster_engine._sparse_cosine(sig_int, cluster.centroid), 4
                ),
                "doc_count": cluster.doc_count,
            }
            for cid, cluster in session.cluster_engine.clusters.items()
        ]

    scored: list[tuple[str, float]] = []
    for cid, cluster in session.cluster_engine.clusters.items():
        sim = session.cluster_engine._sparse_cosine(sig_int, cluster.centroid)
        if sim >= min_similarity:
            scored.append((cid, sim))

    scored.sort(key=lambda x: -x[1])
    top = scored[:top_k]

    return [
        {
            "cluster_id": cid,
            "label": session.cluster_engine.clusters[cid].label,
            "similarity": round(sim, 4),
            "doc_count": session.cluster_engine.clusters[cid].doc_count,
        }
        for cid, sim in top
    ]


# ── 3. kb_get_cards ───────────────────────────────────────────────

def kb_get_cards(
    session: KnowledgeBaseSession,
    cluster_ids: list[str],
) -> dict[str, str]:
    """Fetch knowledge_card text for one or more clusters.

    Returns ``{cluster_id: card_text}``.  Missing cluster_ids are
    omitted from the result.
    """
    result: dict[str, str] = {}
    for cid in cluster_ids:
        cluster = session.cluster_engine.clusters.get(cid)
        if cluster:
            result[cid] = cluster.knowledge_card or "(暂无知识档案)"
    return result


# ── 4. kb_assign ──────────────────────────────────────────────────

def kb_assign(
    session: KnowledgeBaseSession,
    doc_id: str,
    cluster_id: str,
    card_text: str | None = None,
) -> dict[str, Any]:
    """Assign a document to an existing cluster.

    Reads the token signature from the ``doc_signatures`` table
    (stored by ``kb_ingest``).  Updates the cluster centroid
    (running average) and, optionally, the knowledge_card.

    Also updates token doc-frequency for this document.
    """
    # Read signature from DB
    conn = session.db.connect()
    row = conn.execute(
        "SELECT signature FROM doc_signatures WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if not row:
        return {"error": f"Document {doc_id} not found — ingest it first"}
    sig_int = {int(k): v for k, v in json.loads(row["signature"]).items()}

    # Add to cluster
    session.cluster_engine._add_to_cluster(cluster_id, doc_id, sig_int)

    # Optionally update card
    if card_text is not None:
        cluster = session.cluster_engine.clusters[cluster_id]
        cluster.knowledge_card = card_text
        cluster.last_updated = KnowledgeCluster.now()
        session.store.save_cluster(cluster)

    # Update doc-freq (now that the doc is truly assigned)
    session.cluster_engine._update_doc_freq(doc_id)

    cluster = session.cluster_engine.clusters[cluster_id]
    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster.label,
        "doc_count": cluster.doc_count,
        "card_updated": card_text is not None,
    }


# ── 5. kb_create ──────────────────────────────────────────────────

def kb_create(
    session: KnowledgeBaseSession,
    label: str,
    card_text: str,
    doc_id: str,
) -> dict[str, Any]:
    """Create a new cluster with a label and knowledge_card.

    The document is assigned as the first member.  Returns the new
    cluster_id.

    Also updates token doc-frequency for this document.
    """
    # Read signature from DB
    conn = session.db.connect()
    row = conn.execute(
        "SELECT signature FROM doc_signatures WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if not row:
        return {"error": f"Document {doc_id} not found — ingest it first"}
    sig_int = {int(k): v for k, v in json.loads(row["signature"]).items()}

    # Load text for _create_new_cluster (needs text for fallback label)
    from kb_agent.document.loader import load_text
    file_row = conn.execute(
        "SELECT file_path FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    text = load_text(file_row["file_path"]) if file_row else ""

    # Create cluster (persists immediately)
    create_result = session.cluster_engine._create_new_cluster(
        doc_id, sig_int, text
    )
    cid = create_result["cluster_id"]

    # Set label and card
    cluster = session.cluster_engine.clusters[cid]
    cluster.label = label
    cluster.knowledge_card = card_text
    cluster.last_updated = KnowledgeCluster.now()
    session.store.save_cluster(cluster)

    # Update doc-freq
    session.cluster_engine._update_doc_freq(doc_id)

    return {
        "cluster_id": cid,
        "cluster_label": label,
        "doc_count": 1,
    }


# ── 6. kb_update_card ─────────────────────────────────────────────

def kb_update_card(
    session: KnowledgeBaseSession,
    cluster_id: str,
    card_text: str,
) -> dict[str, Any]:
    """Update a cluster's knowledge_card.

    Returns the cluster_id and label.
    """
    cluster = session.cluster_engine.clusters.get(cluster_id)
    if not cluster:
        return {"error": f"Cluster {cluster_id} not found"}

    cluster.knowledge_card = card_text
    cluster.last_updated = KnowledgeCluster.now()
    session.store.save_cluster(cluster)

    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster.label,
    }


# ── 7. kb_search ──────────────────────────────────────────────────

def kb_search(
    session: KnowledgeBaseSession,
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """BM25 search across the token index.

    Modes: "exact" (Layer 1), "phrase" (Layer 2), "hybrid" (0.6+0.4).
    Returns document metadata with BM25 scores.
    """
    return session.index_engine.search(query, top_k=top_k, mode=mode)


# ── 8. kb_archive ─────────────────────────────────────────────────

def kb_archive(
    session: KnowledgeBaseSession,
    file_path: str,
    label: str,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Copy a file to ``archive_root/{sanitized_label}/``.

    Handles filename conflicts by appending doc_id.
    Updates the index's ``file_path`` to point to the archived location.

    Returns the destination path.
    """
    safe_label = (
        re.sub(r'[^\w\u4e00-\u9fff]', "_", label).strip("_") or "unclassified"
    )
    target_dir = session.archive_root / safe_label
    target_dir.mkdir(parents=True, exist_ok=True)
    src = Path(file_path)
    target = target_dir / src.name
    if target.exists() and doc_id:
        target = target_dir / f"{src.stem}_{doc_id}{src.suffix}"
    shutil.copy2(str(src), str(target))
    archived_path = str(target)

    # Update the index to point to the archived location
    if doc_id and archived_path != file_path:
        conn = session.db.connect()
        conn.execute(
            "UPDATE documents SET file_path = ? WHERE doc_id = ?",
            (archived_path, doc_id),
        )
        conn.commit()

    return {"archived_path": archived_path}