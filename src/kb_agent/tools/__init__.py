"""Atomic tools for Hermes agent orchestration.

Usage (from a Hermes session)::

    from kb_agent.tools import init_kb, kb_ingest, kb_prefilter, ...

    session = init_kb("kb_index.db")
    session.connect()
    try:
        result = kb_ingest(session, "paper.txt")
        candidates = kb_prefilter(session, result["signature"])
        # ... Hermes reads candidates, decides, calls kb_create/kb_assign
    finally:
        session.close()
"""

from kb_agent.tools.session import KnowledgeBaseSession
from kb_agent.tools.ops import (
    kb_ingest,
    kb_prefilter,
    kb_get_cards,
    kb_assign,
    kb_create,
    kb_update_card,
    kb_search,
    kb_archive,
)


def init_kb(
    db_path: str = "kb_index.db",
    archive_root: str = "./knowledge_base",
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    similarity_threshold: float = 0.35,
) -> KnowledgeBaseSession:
    """Create and return a connected KnowledgeBaseSession.

    This is the single entry point for Hermes to initialise the
    knowledge base.  All engines are lazily created on first use.
    """
    return KnowledgeBaseSession(
        db_path=db_path,
        archive_root=archive_root,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
    )


__all__ = [
    "init_kb",
    "kb_ingest",
    "kb_prefilter",
    "kb_get_cards",
    "kb_assign",
    "kb_create",
    "kb_update_card",
    "kb_search",
    "kb_archive",
    "KnowledgeBaseSession",
]