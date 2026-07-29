"""KnowledgeBaseSession — shared session holding all engine instances.

Every tool function accepts a session as its first argument so that
components (db, tokenizer, engines) are initialised once and reused
across calls within the same Hermes turn.
"""

from __future__ import annotations

from pathlib import Path

from kb_agent.tokenizer.canonical import CanonicalTokenizer
from kb_agent.storage.db import Database
from kb_agent.storage.cluster_store import ClusterStore
from kb_agent.index.engine import TokenIndexEngine
from kb_agent.cluster.manager import TokenClusterEngine


class KnowledgeBaseSession:
    """Shared session holding all knowledge-base engine instances.

    Usage::

        session = KnowledgeBaseSession("kb_index.db")
        session.connect()
        try:
            result = kb_search(session, query="注意力机制")
        finally:
            session.close()
    """

    def __init__(
        self,
        db_path: str = "kb_index.db",
        archive_root: str = "./knowledge_base",
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        similarity_threshold: float = 0.35,
    ):
        self.db_path = db_path
        self.archive_root = Path(archive_root)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

        # Lazy-init: engines are created on first connect()
        self.db: Database | None = None
        self.tokenizer: CanonicalTokenizer | None = None
        self.store: ClusterStore | None = None
        self.index_engine: TokenIndexEngine | None = None
        self.cluster_engine: TokenClusterEngine | None = None
        self._connected = False

    # ── lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise all engines (idempotent)."""
        if self._connected:
            return
        self.db = Database(self.db_path)
        self.tokenizer = CanonicalTokenizer()
        self.store = ClusterStore(self.db)
        self.index_engine = TokenIndexEngine(
            self.db, self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.cluster_engine = TokenClusterEngine(
            self.tokenizer, self.store,
            similarity_threshold=self.similarity_threshold,
        )
        self._connected = True

    def close(self) -> None:
        """Close the database connection."""
        if self.db is not None:
            self.db.close()
            self.db = None
        self._connected = False

    def __enter__(self) -> KnowledgeBaseSession:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()